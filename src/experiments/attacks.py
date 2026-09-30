"""Native S3--S5 attack objectives with offline-testable gradient operators."""
from collections import OrderedDict
from dataclasses import replace
import math
import torch
from torch.func import functional_call, jvp, vjp

from src.cbg.functionals import select_response_logits
from src.cbg.spec import spec_precision, spec_training_gradient
from src.experiments.protocol import number, positive_int


def forward_logits(model, batch, params=None):
    inputs={key:batch[key] for key in ("input_ids","attention_mask")}
    return (model(**inputs) if params is None else functional_call(model,params,(),inputs)).logits


def nll(model,batch):
    logp=select_response_logits(forward_logits(model,batch),batch).log_softmax(-1)
    mask=batch["response_mask"][:,1:].bool() & batch["attention_mask"][:,1:].bool()
    targets=batch["input_ids"][:,1:][mask]
    if not len(targets):
        raise ValueError("attack batch has no response targets")
    return -logp.gather(1,targets[:,None]).mean()


def benign_scores(model,batch,params,samples):
    logits=forward_logits(model,batch,params)
    mask=batch["response_mask"][:,1:].bool() & batch["attention_mask"][:,1:].bool()
    logp=logits[:,:-1][mask].log_softmax(-1)
    ids=torch.arange(len(mask),device=mask.device)[:,None].expand_as(mask)[mask]
    scores=logp.gather(1,samples[:,None]).squeeze(1)
    return scores.new_zeros(len(mask)).scatter_add(0,ids,scores)/math.sqrt(len(mask))


def project_benign_span(model,batch,gradient,*,rcond,generator):
    """Euclidean projection over ALL trainable parameters, including outside U.

    Build n x n Gram using n VJP/JVP pairs. No n x parameters Jacobian is
    retained. An explicit pseudoinverse cutoff handles dependent benign rows;
    no ridge silently changes the requested orthogonal projection.
    """
    number(rcond,"projection_rcond",minimum=1e-30)
    if rcond>=1:
        raise ValueError("projection_rcond must be <1")
    params=OrderedDict((n,p) for n,p in model.named_parameters() if p.requires_grad)
    with spec_precision(model):
        with torch.no_grad():
            probabilities=select_response_logits(forward_logits(model,batch),batch).softmax(-1)
            samples=torch.multinomial(probabilities,1,generator=generator).squeeze(1)
        def scores(p):
            return benign_scores(model,batch,p,samples)
        output,pullback=vjp(scores,params)
        n=len(output)
        eye=torch.eye(n,device=output.device,dtype=output.dtype)
        kernel=output.new_zeros(n,n,dtype=torch.float64)
        for column in range(n):
            row=OrderedDict((name,value.detach()) for name,value in pullback(eye[column])[0].items())
            kernel[:,column]=jvp(scores,(params,),(row,))[1].detach().double()
            del row
        kernel=(kernel+kernel.T)/2
        eigen,vectors=torch.linalg.eigh(kernel)
        if float(eigen[0]) < -1e-5*max(float(eigen[-1]),1e-30):
            raise FloatingPointError("projected-attack Gram is not PSD")
        active=eigen > rcond*eigen[-1].clamp_min(0)
        cg=jvp(scores,(params,),(gradient,))[1].detach().double()
        coefficients=(vectors[:,active] @ ((vectors[:,active].T @ cg)/eigen[active])
                      if bool(active.any()) else torch.zeros_like(cg))
        removed=pullback(coefficients.to(output))[0]
        result=OrderedDict((name,(value-removed[name]).detach()) for name,value in gradient.items())
        residual=jvp(scores,(params,),(result,))[1].detach().double().norm()
        return result,dict(projection_rank=int(active.sum()),projection_residual=float(residual),
                           projection_gram_trace=float(kernel.trace()))


@torch.no_grad()
def rescale_swiglu(model,*,sigma,generator,block_path="model.layers"):
    """Function-preserving positive diagonal scaling, including up bias."""
    number(sigma,"rescale_sigma")
    factors=[]
    modules=[]
    for block in model.get_submodule(block_path):
        mlp=block.mlp
        if mlp.up_proj.out_features!=mlp.down_proj.in_features:
            raise ValueError("unsupported SwiGLU dimensions")
        up,down=mlp.up_proj,mlp.down_proj
        c=torch.exp(sigma*torch.randn(up.out_features,device=up.weight.device,dtype=up.weight.dtype,generator=generator))
        if not bool(torch.isfinite(c).all()) or not bool((c>0).all()):
            raise FloatingPointError("nonfinite rescaling")
        factors.append(c)
        modules.append((up,down))
    for (up,down),c in zip(modules,factors):
        up.weight.mul_(c[:,None])
        if up.bias is not None:
            up.bias.mul_(c)
        down.weight.div_(c[None,:])
    return factors


def attack_gradient(model,batch,recipe,*,reference=None,spec_config=None,generator):
    family=recipe["family"]
    params=OrderedDict((n,p) for n,p in model.named_parameters() if p.requires_grad)
    metrics={}
    # Eval mode gives consistent deterministic geometry for adaptive attacks.
    context=spec_precision(model)
    with context:
        loss=nll(model,batch.get("objective_harmful",batch["harmful"]))
        metrics["harmful_nll"]=float(loss.detach())
        if family in {"kl","benign_sft"}:
            weight=number(recipe["regularization"],"regularization")
            if family=="kl":
                if reference is None:
                    raise ValueError("KL attack requires its PRE-ATTACK defended reference")
                with spec_precision(reference),torch.no_grad():
                    ref=select_response_logits(forward_logits(reference,batch["benign"]),batch["benign"]).log_softmax(-1)
                current=select_response_logits(forward_logits(model,batch["benign"]),batch["benign"]).log_softmax(-1)
                regularizer=(ref.exp()*(ref-current)).sum(-1).mean()
            else:
                regularizer=nll(model,batch["benign"])
            loss=loss+weight*regularizer
            metrics["regularizer"]=float(regularizer.detach())
        raw=torch.autograd.grad(loss,tuple(params.values()),allow_unused=True)
        gradient=OrderedDict((name,torch.zeros_like(p) if g is None else g.detach())
                             for (name,p),g in zip(params.items(),raw))
        if family=="projected":
            gradient,projection=project_benign_span(model,batch["benign"],gradient,
                rcond=recipe["projection_rcond"],generator=generator)
            metrics.update(projection)
        elif family=="reverse_cbg":
            if spec_config is None:
                raise ValueError("reverse-CBG requires the declared geometry config")
            config=replace(spec_config,alpha=0.,beta=number(recipe["regularization"],"reverse beta"),drop_aln=True)
            reverse=spec_training_gradient(model,batch,config,generator=generator)
            for name in gradient:
                gradient[name].sub_(reverse.gradient[name])
            metrics.update({f"reverse_{key}":float(value) for key,value in reverse.diagnostics.items()})
        elif family not in {"sft","mixing","kl","benign_sft"}:
            raise ValueError(f"unsupported native attack family {family}")
        sigma=recipe.get("noise_sigma",0.)
        if sigma:
            number(sigma,"noise_sigma")
            scale=recipe["noise_scale"]
            if scale=="relative_gradient_rms":
                rms=(sum(g.double().square().sum() for g in gradient.values())/
                     sum(g.numel() for g in gradient.values())).sqrt()
            elif scale=="absolute":
                rms=1.
            else:
                raise ValueError("noise_scale must be absolute or relative_gradient_rms")
            for value in gradient.values():
                value.add_(torch.randn(value.shape,device=value.device,dtype=value.dtype,generator=generator)*rms*sigma)
        if any(not bool(torch.isfinite(g).all()) for g in gradient.values()):
            raise FloatingPointError("nonfinite attack gradient")
    return gradient,metrics
