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


def _radius_chunks(batch, joint, samples, cotangent, count):
    """Partition one frozen scalarization without changing its m/n scaling."""
    m, n = joint.m, joint.n
    benign_mask = (joint.response_mask[2*m:, 1:].bool()
                   & joint.attention_mask[2*m:, 1:].bool())
    offsets = [0]
    for length in benign_mask.sum(1).tolist():
        offsets.append(offsets[-1] + length)
    if offsets[-1] != samples.numel() or cotangent.numel() != m + n:
        raise ValueError("radius samples or cotangent do not match the joint batch")
    chunks = []
    for index in range(count):
        m0, m1 = index*m//count, (index+1)*m//count
        n0, n1 = index*n//count, (index+1)*n//count
        part = {name: {field: value[start:end] for field, value in batch[name].items()}
                for name, start, end in (("harmful", m0, m1), ("safe", m0, m1),
                                         ("benign", n0, n1))}
        local = joint_batch(part)
        weights = torch.cat((cotangent[m0:m1] * math.sqrt(local.m/m),
                             cotangent[m+n0:m+n1] * math.sqrt(local.n/n)))
        chunks.append((local, samples[offsets[n0]:offsets[n1]], weights))
    return chunks


def _accumulate_cpu(accumulator, pieces, scale=1.):
    """Release each GPU gradient tree after adding it to host memory."""
    with torch.no_grad():
        for name, value in pieces.items():
            part = value.detach().to(device="cpu", dtype=torch.float64)
            if name in accumulator:
                accumulator[name].add_(part, alpha=scale)
            else:
                accumulator[name] = part if scale == 1 else part.mul_(scale)


def radius_curve(model, batch, config, radii, *, seed, perturb_precision="fp32", microbatches=8,
                 output_deltas=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2), exact_precision=None,
                 exact_accumulator=None, fixed_samples=None):
    """Compare the same frozen scalarization's FD gradient to exact 2*H*w.

    This intentionally uses the loaded model's actual kernels/precision. It
    does not apply the FP64 fixture patches from the tiny numerical tests.
    Microbatches share the full-batch center, Fisher samples and cotangent.
    Full parameter gradients are accumulated on CPU to bound GPU memory.
    """
    joint = joint_batch(batch)
    if perturb_precision not in {"fp32","bf16","tf32"}:
        raise ValueError("perturb_precision must be fp32, bf16 or tf32")
    config = replace(config, perturb_precision=perturb_precision)
    if isinstance(microbatches, bool) or not isinstance(microbatches, int) or microbatches < 1:
        raise ValueError("microbatches must be a positive integer")
    params = OrderedDict((n,p) for n,p in model.named_parameters() if p.requires_grad)
    generator = torch.Generator(device=joint.input_ids.device).manual_seed(seed)
    with spec_precision(model, perturb_precision):
        spectrum, samples, g, w, damping_gradient, cw = _center(
            model, joint, config, spec_linear_modules(model, config.block_path),
            generator=generator, samples=fixed_samples, return_cw=True)
        del damping_gradient  # Radius checks only the frozen 2H_omega w term.
        predicted_unit = (spectrum.kernel @ cw).detach().to(device="cpu", dtype=torch.float64)
        raw_scale = torch.cat((predicted_unit[:joint.m] * math.sqrt(joint.m),
                               predicted_unit[joint.m:] * math.sqrt(joint.n)))
        max_raw_per_epsilon = float(raw_scale.abs().max())
        if not math.isfinite(max_raw_per_epsilon) or max_raw_per_epsilon <= 0:
            raise ValueError("zero or nonfinite predicted output displacement")
        direction_norm_sq = cw @ spectrum.kernel @ cw
        del spectrum, cw, raw_scale
        tangent = OrderedDict((n,w.get(n,torch.zeros_like(p))) for n,p in params.items())
        count = min(microbatches, joint.m, joint.n)
        print(f"Radius: full center m={joint.m}, n={joint.n}, k={config.k}; "
              f"exact HVP and endpoints in {count} microbatches", flush=True)
        gradients = []
        for local, fixed_samples, weights in _radius_chunks(batch, joint, samples, g, count):
            def omega(p, local=local, fixed_samples=fixed_samples, weights=weights):
                values = measurements(_forward(model, local, p), local, samples=fixed_samples).values
                return weights @ values.double()
            gradients.append(grad(omega))
        exact = OrderedDict()
        with spec_precision(model, exact_precision or perturb_precision):
            for gradient in gradients:
                primal, hvp = jvp(gradient, (params,), (tangent,))
                del primal
                _accumulate_cpu(exact, hvp, scale=2.)
                del hvp
        denominator = squared_norm(exact)
        if not bool(denominator>0):
            raise ValueError("zero exact HVP cannot qualify a finite-difference radius")
        if exact_accumulator is not None:
            _accumulate_cpu(exact_accumulator, exact)
        print("Radius: exact HVP complete", flush=True)
        norm = direction_norm_sq.sqrt()
        dim = sum(v.numel() for v in w.values())
        radius_scale = float(math.sqrt(dim) / norm)
        def endpoint_parameters(epsilon, sign):
            return OrderedDict((name, value.detach()+sign*epsilon*w[name] if name in w else value)
                               for name,value in params.items())
        def linearity_error(epsilon):
            with torch.no_grad(), torch.autocast(joint.input_ids.device.type,
                    dtype=torch.bfloat16, enabled=perturb_precision == "bf16"):
                plus = endpoint_parameters(epsilon, 1)
                positive = measurements(_forward(model, joint, plus), joint, samples=samples).values
                positive = positive.detach().to(device="cpu", dtype=torch.float64)
                del plus
                minus = endpoint_parameters(epsilon, -1)
                negative = measurements(_forward(model, joint, minus), joint, samples=samples).values
                negative = negative.detach().to(device="cpu", dtype=torch.float64)
                del minus
            predicted = epsilon * predicted_unit
            return float((((positive-negative)/2-predicted).norm()/predicted.norm()))
        def finite_difference_error(epsilon):
            difference = OrderedDict()
            changed = 0
            for sign in (1,-1):
                p = endpoint_parameters(epsilon, sign)
                changed += sum(int(((p[name].bfloat16() != params[name].bfloat16())
                                    if perturb_precision=="bf16" else (p[name] != params[name])).sum())
                               for name in w)
                for gradient in gradients:
                    with torch.autocast(joint.input_ids.device.type,dtype=torch.bfloat16,
                                        enabled=perturb_precision=="bf16"):
                        pieces = gradient(p)
                    _accumulate_cpu(difference, pieces, scale=sign)
                    del pieces
                del p
            error = sum((difference[name]/epsilon-exact[name]).square().sum() for name in params)
            estimate_norm = sum((difference[name]/epsilon).square().sum() for name in params)
            dot = sum(((difference[name]/epsilon)*exact[name]).sum() for name in params)
            cosine = (float((dot/(estimate_norm*denominator).sqrt()).clamp(-1, 1))
                      if bool(estimate_norm > 0) else 0.0)
            return float((error/denominator).sqrt()), changed/(2*dim), cosine, float(error.sqrt())
        rows=[]
        for radius in radii:
            if not math.isfinite(radius) or radius<=0:
                raise ValueError("radii must be finite and positive")
            epsilon = radius * radius_scale
            if not math.isfinite(epsilon):
                rows.append(dict(radius=radius, epsilon=None, relative_error=None,
                                 linearity_error=None, changed_weight_fraction=None,
                                 cosine=None, absolute_error_norm=None, status="nonfinite_step"))
                continue
            linearity = linearity_error(epsilon)
            error, changed, cosine, absolute_error = finite_difference_error(epsilon)
            if all(math.isfinite(value) for value in (linearity, error, changed, cosine, absolute_error)):
                rows.append(dict(radius=radius,epsilon=epsilon,relative_error=error,
                                 linearity_error=linearity,changed_weight_fraction=changed,
                                 cosine=cosine,absolute_error_norm=absolute_error,status="ok"))
            else:
                rows.append(dict(radius=radius,epsilon=epsilon,relative_error=None,
                                 linearity_error=None,changed_weight_fraction=None,
                                 cosine=None,absolute_error_norm=None,status="nonfinite_endpoint"))
            print(f"Radius: r={radius:g}, relative_error={error:.6g}, "
                  f"linearity_error={linearity:.6g}", flush=True)
        delta_rows=[]
        for delta in output_deltas:
            epsilon = delta / max_raw_per_epsilon
            linearity = linearity_error(epsilon)
            row = dict(delta=delta,epsilon=epsilon,
                       relative_radius_equivalent=epsilon/radius_scale,
                       linearity_error=linearity,relative_error=None)
            if linearity <= .01:
                (row["relative_error"], row["changed_weight_fraction"], row["cosine"],
                 row["absolute_error_norm"]) = finite_difference_error(epsilon)
            delta_rows.append(row)
            print(f"Radius: delta={delta:g}, linearity_error={linearity:.6g}, "
                  f"relative_error={row['relative_error']}", flush=True)
        return dict(seed=seed,dtype=str(next(iter(params.values())).dtype),perturb_precision=perturb_precision,
                    exact_precision=exact_precision or perturb_precision,
                    exact_hvp_norm=float(denominator.sqrt()),
                    method="torch.func.jvp(grad)",microbatches=count,
                    m=joint.m,n=joint.n,k=config.k,curve=rows,
                    predicted_max_raw_change_per_epsilon=max_raw_per_epsilon,
                    delta_curve=delta_rows)


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
    with spec_precision(model, config.perturb_precision):
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
                with torch.autocast(joint.input_ids.device.type, dtype=torch.bfloat16,
                                    enabled=config.perturb_precision == "bf16"):
                    logits = _forward(model,joint)
                score=measurements(logits,joint,samples=fixed)
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
