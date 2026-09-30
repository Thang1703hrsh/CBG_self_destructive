"""S0 radius qualification and S8 geometry with fixed tokenized probe batches."""
from collections import OrderedDict
from dataclasses import replace
import math
import torch
from torch.func import grad, jvp

from src.cbg.spec import (_center, _forward, joint_batch, measurements, spec_precision)
from src.cbg.spec_hooks import spec_linear_modules
from src.experiments.protocol import digest


def squared_norm(tree):
    return sum(value.detach().double().square().sum() for value in tree.values())


def radius_curve(model, batch, config, radii, *, seed, perturb_precision="fp32"):
    """Compare the same frozen scalarization's FD gradient to exact 2*H*w.

    This intentionally uses the loaded model's actual kernels/precision. It
    does not apply the FP64 fixture patches from the tiny numerical tests.
    Exact HVP may be expensive on a pilot model; errors are surfaced, not hidden.
    """
    joint = joint_batch(batch)
    if perturb_precision not in {"fp32","bf16"}:
        raise ValueError("perturb_precision must be fp32 or bf16")
    params = OrderedDict((n,p) for n,p in model.named_parameters() if p.requires_grad)
    generator = torch.Generator(device=joint.input_ids.device).manual_seed(seed)
    with spec_precision(model):
        _, samples, g, w, _ = _center(model,joint,config,spec_linear_modules(model,config.block_path),generator=generator)
        tangent = OrderedDict((n,w.get(n,torch.zeros_like(p))) for n,p in params.items())
        def omega(p):
            values = measurements(_forward(model,joint,p),joint,samples=samples).values
            return g.to(values) @ values
        gradient = grad(omega)
        _, hvp = jvp(gradient,(params,),(tangent,))
        exact = OrderedDict((n,2*v.detach()) for n,v in hvp.items())
        del hvp
        denominator = squared_norm(exact)
        if not bool(denominator>0):
            raise ValueError("zero exact HVP cannot qualify a finite-difference radius")
        norm = sum((v.double().square()/(params[n].detach().double().square()+config.xi**2)).sum()
                   for n,v in w.items()).sqrt()
        dim = sum(v.numel() for v in w.values())
        rows=[]
        for radius in radii:
            if not math.isfinite(radius) or radius<=0:
                raise ValueError("radii must be finite and positive")
            eps = radius*math.sqrt(dim)/norm
            endpoints=[]
            changed=0
            for sign in (1,-1):
                p = OrderedDict((n,v.detach()+sign*eps.to(v)*tangent[n]) for n,v in params.items())
                changed += sum(int(((p[n].bfloat16() != params[n].bfloat16()) if perturb_precision=="bf16"
                                    else (p[n] != params[n])).sum()) for n in w)
                with torch.autocast(joint.input_ids.device.type,dtype=torch.bfloat16,enabled=perturb_precision=="bf16"):
                    endpoints.append(OrderedDict((n,v.detach()) for n,v in gradient(p).items()))
            error = sum(((endpoints[0][n]-endpoints[1][n]).double()/eps-exact[n].double()).square().sum()
                        for n in params)
            rows.append(dict(radius=radius,epsilon=float(eps),relative_error=float((error/denominator).sqrt()),
                             changed_weight_fraction=changed/(2*dim)))
        return dict(seed=seed,dtype=str(next(iter(params.values())).dtype),perturb_precision=perturb_precision,
                    method="torch.func.jvp(grad)",
                    m=joint.m,n=joint.n,k=config.k,curve=rows)


def evaluate_geometry(model, batches, config, *, seed, path_samples=None):
    """Geometry resamples per model using the same seeds; paths use fixed tokens.

    Changing categorical samples with the checkpoint changes F_b itself. Keeping
    path samples separate prevents that sampling noise becoming 'path length'.
    """
    modules = spec_linear_modules(model,config.block_path)
    results, values, samples_out = [], [], []
    path_rho=[]
    if not batches or (path_samples is not None and len(path_samples)!=len(batches)):
        raise ValueError("nonempty geometry batches and one path sample tensor per batch required")
    with spec_precision(model):
        for index,batch in enumerate(batches):
            joint=joint_batch(batch)
            generator=torch.Generator(device=joint.input_ids.device).manual_seed(seed+index)
            q,samples = _center(model,joint,config,modules,generator=generator,materialize=False)
            results.append({key:float(value) for key,value in q.diagnostics().items()})
            fixed = samples if path_samples is None else path_samples[index].to(samples.device)
            if path_samples is None:
                path_spectrum=q
            else:
                path_spectrum,_=_center(model,joint,config,modules,samples=fixed,materialize=False)
            path_rho.append(float(path_spectrum.rho[0]))
            with torch.no_grad():
                score=measurements(_forward(model,joint),joint,samples=fixed)
            values.append(dict(h=score.values[:joint.m].cpu().tolist(),b=score.values[joint.m:].cpu().tolist()))
            samples_out.append(fixed.cpu())
    return dict(seed=seed,m=joint.m,n=joint.n,k=config.k,batches=results,
                mean={key:sum(row[key] for row in results)/len(results) for key in results[0]},
                path_values=values,path_rho_hat=path_rho,
                path_samples_sha256=digest([s.tolist() for s in samples_out])), samples_out


def path_summary(checkpoints):
    """Aggregate equally sized probe batches in an RMS direct-sum geometry."""
    if len(checkpoints)<2:
        raise ValueError("trajectory requires at least two ordered checkpoints")
    first=checkpoints[0]
    length=0.
    max_rho=0.
    rows=[]
    previous=None
    for index, checkpoint in enumerate(checkpoints):
        for key in ("path_samples_sha256","m","n","k","bank_sha256"):
            if checkpoint.get(key)!=first.get(key):
                raise ValueError(f"trajectory {key} changed")
        if len(checkpoint["path_values"])!=len(first["path_values"]):
            raise ValueError("trajectory probe count changed")
        def norm_difference(left,right,key):
            if any(len(a[key]) != len(b[key]) for a,b in zip(left,right)):
                raise ValueError("trajectory probe dimensions changed")
            return math.sqrt(sum(sum((x-y)**2 for x,y in zip(a[key],b[key]))
                                 for a,b in zip(left,right))/len(left))
        current=checkpoint["path_values"]
        if previous is not None:
            length+=norm_difference(current,previous,"b")
        change=norm_difference(current,first["path_values"],"h")
        max_rho=max(max_rho,max(checkpoint["path_rho_hat"]))
        rows.append(dict(index=index,safety_change=change,benign_path_length=length,
                         ratio=change/length if length else None,sqrt_max_rho=math.sqrt(max_rho)))
        previous=current
    return rows
