"""Functional finite-difference reference for the paper's log-eta objective.

See docs/method.md. The exact-HVP training step is in src.cbg.step_v2.

No optimizer step is performed here. The returned gradient covers ALL trainable
parameters; the admissible perturbation/Gram space contains only block Linears.
"""
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
import math
import time

import torch
from torch import Tensor
from torch.func import functional_call
from torch.nn import functional as F

from src.cbg.functionals import _extract_logits, _validate_sequence_batch
from src.cbg.spec_hooks import SpecLinearHooks, spec_linear_modules
from src.cbg.spec_spectral import solve_spec_spectrum, eigen_sign_scalarization, DegenerateGeometry


@dataclass(frozen=True)
class SpecConfig:
    # Scientific choices have no invented defaults.
    xi: float
    kappa_mu: float
    r: float
    alpha: float
    beta: float
    k: int = 4
    gram_max_bytes: int = 64 * 1024 * 1024
    block_path: str = "model.layers"
    kl_direction: str = "reference_to_current"
    damping: str = "relative"
    fisher: str = "sampled"
    detach_lambda: bool = False
    num_sign_draws: int = 1
    objective: str = "log_eta"
    drop_aln: bool = False
    solver: str = "exact"
    rayleigh_steps: int = 0
    rayleigh_step_size: float | None = None
    perturb_precision: str = "fp32"
    mu_mode: str = "differentiated"
    capture_dtype: str = "model"
    capture_offload: bool = False
    lazy_period: int = 1
    lazy_rescale: bool = True
    logprob_chunk_size: int = 128
    log_gradient_components: bool = False
    estimator: str = "shared_sign"

    def __post_init__(self):
        for name in ("xi", "alpha", "beta", "kappa_mu", "r"):
            value = getattr(self, name)
            if value is None or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be explicitly configured, finite and nonnegative")
            if name in {"kappa_mu", "r"} and value == 0:
                raise ValueError(f"{name} must be positive")
        for name in ("k", "gram_max_bytes", "num_sign_draws", "lazy_period"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.kl_direction not in {"reference_to_current", "current_to_reference"}:
            raise ValueError("unsupported kl_direction")
        if self.damping not in {"relative", "euclidean"}:
            raise ValueError("unsupported damping")
        if self.fisher not in {"sampled", "empirical"}:
            raise ValueError("unsupported fisher")
        if not isinstance(self.block_path, str) or not self.block_path:
            raise ValueError("block_path must name the transformer block container")
        if not isinstance(self.detach_lambda, bool):
            raise ValueError("detach_lambda must be boolean")
        if not isinstance(self.drop_aln, bool):
            raise ValueError("drop_aln must be boolean")
        if self.objective not in {"log_eta", "raw_rho"} or (self.objective == "raw_rho" and self.k != 1):
            raise ValueError("objective must be log_eta or raw_rho (with k=1)")
        if self.solver not in {"exact", "rayleigh_K"}:
            raise ValueError("solver must be exact or rayleigh_K")
        if self.solver == "rayleigh_K" and (isinstance(self.rayleigh_steps, bool)
            or not isinstance(self.rayleigh_steps, int) or self.rayleigh_steps < 1
            or self.rayleigh_step_size is None or not math.isfinite(self.rayleigh_step_size) or self.rayleigh_step_size <= 0):
            raise ValueError("Rayleigh ablation needs steps and an explicit positive step size")
        if self.perturb_precision not in {"fp32", "bf16", "tf32"}:
            raise ValueError("perturb_precision must be fp32, bf16 or tf32")
        if self.mu_mode not in {"differentiated", "detached"}:
            raise ValueError("mu_mode must be differentiated or detached")
        if self.capture_dtype not in {"model", "bf16"}:
            raise ValueError("capture_dtype must be model or bf16")
        if not isinstance(self.capture_offload, bool) or not isinstance(self.lazy_rescale, bool):
            raise ValueError("capture_offload and lazy_rescale must be boolean")
        if isinstance(self.logprob_chunk_size, bool) or not isinstance(self.logprob_chunk_size, int) or self.logprob_chunk_size < 0:
            raise ValueError("logprob_chunk_size must be a nonnegative integer (0 disables chunking)")
        if not isinstance(self.log_gradient_components, bool):
            raise ValueError("log_gradient_components must be boolean")
        if self.estimator != "shared_sign":
            raise NotImplementedError("production v2 currently supports estimator=shared_sign only")


@contextmanager
def spec_precision(model, precision_mode="fp32"):
    """Deterministic true FP32/64 forwards, restoring caller modes/precision."""
    tensors = list(model.parameters()) + list(model.buffers())
    if any(t.is_floating_point() and t.dtype not in (torch.float32, torch.float64) for t in tensors):
        raise ValueError("spec CBG requires FP32 master weights/compute (FP64 for validation)")
    devices = {p.device for p in model.parameters()}
    if len({d.type for d in devices}) != 1:
        raise ValueError("spec CBG requires all model devices to share one device type")
    # Layer-split (model-parallel) placement is allowed; autocast is per type.
    device = next(iter(devices))
    # eval() disables HF checkpoint recomputation and dropout without changing
    # the caller's checkpoint setting. Hooks still reject actual reused calls.
    modes = [(module, module.training) for module in model.modules()]
    precision = torch.get_float32_matmul_precision()
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    model.eval()
    try:
        torch.set_float32_matmul_precision("high" if precision_mode == "tf32" else "highest")
        torch.backends.cudnn.allow_tf32 = precision_mode == "tf32"
        with torch.enable_grad(), torch.autocast(device.type, enabled=False):
            yield
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        for module, mode in modes:
            module.training = mode


@dataclass(frozen=True)
class JointBatch:
    input_ids: Tensor
    attention_mask: Tensor
    response_mask: Tensor
    seq_to_example: Tensor
    m: int
    n: int


def joint_batch(batch) -> JointBatch:
    groups = [batch[name] for name in ("harmful", "safe", "benign")]
    for name, group in zip(("harmful", "safe", "benign"), groups):
        _validate_sequence_batch(group, name=name)
    m, ms, n = [g["input_ids"].shape[0] for g in groups]
    if m < 1 or n < 1 or m != ms:
        raise ValueError("nonempty safety pairs and benign sequences are required")
    width = max(g["input_ids"].shape[1] for g in groups)
    fields = {}
    for field in ("input_ids", "attention_mask", "response_mask"):
        values = [g.get(field, torch.ones_like(g["input_ids"])) for g in groups]
        fields[field] = torch.cat([F.pad(v, (0, width-v.shape[1]), value=0) for v in values])
    device = fields["input_ids"].device
    ids = torch.cat((torch.arange(m, device=device), torch.arange(m, device=device),
                     torch.arange(m, m+n, device=device)))
    return JointBatch(**fields, seq_to_example=ids, m=m, n=n)


def _zero_safety_batch(batch):
    """Identical pairs have exactly zero margins, regardless of roundoff in hooks."""
    return all(torch.equal(value[:batch.m], value[batch.m:2*batch.m])
               for value in (batch.input_ids, batch.attention_mask, batch.response_mask))


@dataclass(frozen=True)
class Measurements:
    values: Tensor
    alignment: Tensor
    benign_kl: Tensor
    samples: Tensor


def _chunked_scores(active, targets, benign_positions, samples, reference_logits,
                    fisher, kl_direction, generator, chunk_size):
    """Exact full-vocabulary scores with rematerialized token-chunk softmax.

    This bounds softmax intermediates, NOT model/output-head logits. Checkpoint
    replay covers only this pure score function and never replays Linear hooks.
    """
    from torch.utils.checkpoint import checkpoint
    count = int(benign_positions.sum())
    if samples is not None and (samples.shape != (count,) or samples.dtype != torch.long):
        raise ValueError("fixed Fisher samples must match the benign response token positions")
    if reference_logits is not None and reference_logits.shape != (count, active.shape[-1]):
        raise ValueError("reference must contain only benign response logits [tokens, vocabulary]")
    observed_parts, sampled_parts, sample_parts = [], [], []
    kl = active.new_zeros(())
    offset = 0
    for start in range(0, len(active), chunk_size):
        x = active[start:start+chunk_size]
        target = targets[start:start+chunk_size]
        mask = benign_positions[start:start+chunk_size]
        size = int(mask.sum())
        with torch.no_grad():
            fixed = (samples[offset:offset+size] if samples is not None else
                     (torch.multinomial(x[mask].softmax(-1), 1, generator=generator).squeeze(1)
                      if fisher == "sampled" and size else target[mask]))
        reference = None if reference_logits is None else reference_logits[offset:offset+size].detach().to(x)
        def score(value, target=target, mask=mask, fixed=fixed, reference=reference):
            logp = value.log_softmax(-1)
            observed = logp.gather(1, target[:, None]).squeeze(1)
            benign = logp[mask]
            sampled = benign.gather(1, fixed[:, None]).squeeze(1)
            kl_sum = value.new_zeros(())
            if reference is not None:
                ref = reference.log_softmax(-1)
                if kl_direction == "reference_to_current":
                    kl_sum = (ref.exp()*(ref-benign)).sum()
                elif kl_direction == "current_to_reference":
                    kl_sum = (benign.exp()*(benign-ref)).sum()
                else:
                    raise ValueError("unsupported KL direction")
            return observed, sampled, kl_sum
        outputs = checkpoint(score, x, use_reentrant=False) if x.requires_grad else score(x)
        observed_parts.append(outputs[0])
        sampled_parts.append(outputs[1])
        sample_parts.append(fixed)
        kl = kl+outputs[2]/count
        offset += size
    return torch.cat(observed_parts), torch.cat(sampled_parts), kl, torch.cat(sample_parts).detach()


@dataclass(frozen=True)
class Totals:
    """Whole-step counts, so a micro-batch's terms sum exactly to the full batch.

    m, n: safety pairs / benign sequences in the full CBG batch (the 1/sqrt(m),
    1/sqrt(n) weights). safe_tokens, benign_tokens: response-token counts of the
    full batch (the denominators of the token means L_aln and L_bgn).
    """
    m: int
    n: int
    safe_tokens: int
    benign_tokens: int


def measurements(logits, batch: JointBatch, *, samples=None, reference_logits=None,
                 fisher="sampled", kl_direction="reference_to_current", generator=None,
                 logprob_chunk_size=0, totals: Totals | None = None):
    """Teacher-forced margins and SUM sampled scores, with global token losses.

    Without totals, batch is the whole CBG batch. With totals, batch is a
    micro-batch (it may hold only safety pairs or only benign sequences) and
    every returned term is that micro-batch's share of the full-batch value.
    """
    if batch.input_ids.device != logits.device:
        # Layer-split models produce logits on the last device.
        batch = JointBatch(*(t.to(logits.device) for t in (
            batch.input_ids, batch.attention_mask, batch.response_mask, batch.seq_to_example)),
            m=batch.m, n=batch.n)
        samples = None if samples is None else samples.to(logits.device)
    if totals is not None:
        return _measurements_share(logits, batch, totals, samples=samples, reference_logits=reference_logits,
                                   fisher=fisher, kl_direction=kl_direction, generator=generator,
                                   logprob_chunk_size=logprob_chunk_size)
    mask = batch.response_mask[:, 1:].bool() & batch.attention_mask[:, 1:].bool()
    lengths = mask.sum(1)
    if bool((lengths <= 0).any()):
        raise ValueError("each sequence must have at least one predicted response token")
    if logits.shape[:2] != batch.input_ids.shape:
        raise ValueError("logits and joint sequence shapes differ")
    active = logits[:, :-1][mask]
    if active.dtype in (torch.float16, torch.bfloat16):
        active = active.float()
    targets = batch.input_ids[:, 1:][mask]
    sequence = torch.arange(len(lengths), device=logits.device)[:, None].expand_as(mask)[mask]
    benign_positions = sequence >= 2 * batch.m
    safe_positions = (sequence >= batch.m) & ~benign_positions
    if logprob_chunk_size:
        observed, sampled_scores, kl, samples = _chunked_scores(active, targets, benign_positions,
            samples, reference_logits, fisher, kl_direction, generator, logprob_chunk_size)
        sums = observed.new_zeros(len(lengths)).scatter_add(0, sequence, observed)
        means = sums/lengths
        margins = (means[:batch.m]-means[batch.m:2*batch.m])/math.sqrt(batch.m)
        benign = sampled_scores.new_zeros(batch.n).scatter_add(
            0, sequence[benign_positions]-2*batch.m, sampled_scores)/math.sqrt(batch.n)
        return Measurements(torch.cat((margins, benign)), -observed[safe_positions].mean(), kl, samples)
    logp = active.log_softmax(-1)
    observed = logp.gather(1, targets[:, None]).squeeze(1)
    sums = observed.new_zeros(len(lengths)).scatter_add(0, sequence, observed)
    means = sums / lengths
    margins = (means[:batch.m] - means[batch.m:2*batch.m]) / math.sqrt(batch.m)
    benign_logp = logp[benign_positions]
    if samples is None:
        samples = (torch.multinomial(benign_logp.detach().exp(), 1, generator=generator).squeeze(1)
                   if fisher == "sampled" else targets[benign_positions].detach())
    if samples.shape != (benign_logp.shape[0],) or samples.dtype != torch.long:
        raise ValueError("fixed Fisher samples must match the benign response token positions")
    sampled_scores = benign_logp.gather(1, samples[:, None]).squeeze(1)
    benign = sampled_scores.new_zeros(batch.n).scatter_add(
        0, sequence[benign_positions] - 2 * batch.m, sampled_scores) / math.sqrt(batch.n)
    kl = observed.new_zeros(())
    if reference_logits is not None:
        ref = reference_logits.detach().to(benign_logp)
        if ref.shape != benign_logp.shape:
            raise ValueError("reference must contain only benign response logits [tokens, vocabulary]")
        ref = ref.log_softmax(-1)
        if kl_direction == "reference_to_current":
            kl = (ref.exp() * (ref - benign_logp)).sum(-1).mean()
        elif kl_direction == "current_to_reference":
            kl = (benign_logp.exp() * (benign_logp - ref)).sum(-1).mean()
        else:
            raise ValueError("unsupported KL direction")
    return Measurements(torch.cat((margins, benign)), -observed[safe_positions].mean(), kl, samples.detach())


def _measurements_share(logits, batch, totals, *, samples, reference_logits, fisher, kl_direction,
                        generator, logprob_chunk_size):
    mask = batch.response_mask[:, 1:].bool() & batch.attention_mask[:, 1:].bool()
    lengths = mask.sum(1)
    if bool((lengths <= 0).any()):
        raise ValueError("each sequence must have at least one predicted response token")
    if logits.shape[:2] != batch.input_ids.shape:
        raise ValueError("logits and joint sequence shapes differ")
    m, n = batch.m, batch.n
    active = logits[:, :-1][mask]
    if active.dtype in (torch.float16, torch.bfloat16):
        active = active.float()
    targets = batch.input_ids[:, 1:][mask]
    sequence = torch.arange(len(lengths), device=logits.device)[:, None].expand_as(mask)[mask]
    benign_positions = sequence >= 2 * m
    safe_positions = (sequence >= m) & ~benign_positions
    count = int(benign_positions.sum())
    if not count:
        reference_logits = None
    chunk = logprob_chunk_size or len(active)
    observed, sampled_scores, kl_mean, samples = _chunked_scores(
        active, targets, benign_positions, samples, reference_logits, fisher, kl_direction,
        generator, max(chunk, 1))
    sums = observed.new_zeros(len(lengths)).scatter_add(0, sequence, observed)
    means = sums / lengths
    margins = (means[:m] - means[m:2*m]) / math.sqrt(totals.m)
    benign = sampled_scores.new_zeros(n).scatter_add(
        0, sequence[benign_positions] - 2*m, sampled_scores) / math.sqrt(totals.n)
    alignment = (-observed[safe_positions].sum() / totals.safe_tokens if m
                 else observed.new_zeros(()))
    kl = kl_mean * (count / totals.benign_tokens) if count else observed.new_zeros(())
    return Measurements(torch.cat((margins, benign)), alignment, kl, samples)


def _forward(model, batch, parameters=None):
    kwargs = dict(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
    output = model(**kwargs) if parameters is None else functional_call(model, parameters, (), kwargs)
    return _extract_logits(output)


def _center(model, batch, config, modules, *, generator=None, samples=None, materialize=True,
            timings=None, return_cw=False, return_capture=False):
    if _zero_safety_batch(batch):
        raise DegenerateGeometry("identical safety pairs have zero margin geometry")
    def record(label, start):
        now = _clock(batch.input_ids.device)
        if timings is not None:
            timings[label] = timings.get(label, 0.) + now - start
        return now
    with SpecLinearHooks(modules, batch.seq_to_example, batch.attention_mask,
                         xi=config.xi, max_bytes=config.gram_max_bytes, damping=config.damping,
                         capture_dtype=config.capture_dtype, offload=config.capture_offload) as hooks:
        phase = _clock(batch.input_ids.device)
        with torch.autocast(batch.input_ids.device.type, dtype=torch.bfloat16,
                            enabled=config.perturb_precision == "bf16"):
            logits = _forward(model, batch)
        center = measurements(logits, batch, samples=samples, fisher=config.fisher, generator=generator,
                              logprob_chunk_size=config.logprob_chunk_size)
        seed = center.values.sum()
        phase = record("center_forward_time", phase)
        if hooks.backward_target is None:
            raise ValueError("CBG hooks did not capture a backward target")
        # Traverse output hooks once without allocating discarded weight grads.
        torch.autograd.grad(seed, hooks.backward_target)
        if hooks.captured_names() != set(modules):
            raise ValueError("center backward missed an admissible Linear; incomplete Gram rejected")
        samples = center.samples
        hooks.backward_target = None
        del seed, center, logits
        phase = record("gram_backward_time", phase)
        spectrum = solve_spec_spectrum(hooks.K, batch.m, config.k, kappa_mu=config.kappa_mu,
                                       objective=config.objective, solver=config.solver,
                                       rayleigh_steps=config.rayleigh_steps,
                                       rayleigh_step_size=config.rayleigh_step_size, generator=generator,
                                       mu_mode=config.mu_mode)
        phase = record("spectral_solve_time", phase)
        if not materialize:
            return spectrum, samples
        if return_capture:
            return spectrum, samples, hooks
        signs = torch.randint(0, 2, (len(spectrum.kernel),), device=batch.input_ids.device, generator=generator) * 2 - 1
        cw, cotangent = eigen_sign_scalarization(spectrum, signs)
        hooks.materialize(cw, spectrum.A)
        record("direction_contraction_time", phase)
        result = (spectrum, samples, cotangent, hooks.w, hooks.d_lambda)
        return (*result, cw) if return_cw else result


@dataclass(frozen=True)
class SpecStepResult:
    gradient: OrderedDict
    loss: Tensor
    diagnostics: dict


def _clock(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _plain_gradient(model, joint, config, parameters, reference_logits, samples, generator, *, skipped=False):
    """Exact base-loss step for beta=0, lazy steps, or degenerate geometry."""
    started = _clock(joint.input_ids.device)
    with torch.autocast(joint.input_ids.device.type, dtype=torch.bfloat16,
                        enabled=config.perturb_precision == "bf16"):
        logits = _forward(model, joint)
    values = measurements(logits, joint, samples=samples, reference_logits=reference_logits,
                          fisher=config.fisher, kl_direction=config.kl_direction, generator=generator,
                          logprob_chunk_size=config.logprob_chunk_size)
    loss = (0. if config.drop_aln else 1.)*values.alignment + config.alpha*values.benign_kl
    grads = torch.autograd.grad(loss, tuple(parameters.values()), allow_unused=True)
    result = OrderedDict((name, torch.zeros_like(p) if g is None else g.detach())
                         for (name, p), g in zip(parameters.items(), grads))
    if not bool(torch.isfinite(loss)) or any(not bool(torch.isfinite(g).all()) for g in result.values()):
        raise FloatingPointError("nonfinite CBG base loss/gradient")
    return SpecStepResult(result, loss.detach(), dict(L_aln=values.alignment.detach(),
        L_bgn=values.benign_kl.detach(), geometry_skipped=float(skipped), geometry_active=0.,
        m=joint.m, n=joint.n, nan_flag=0., gradient_components_measured=0.,
        step_time=_clock(joint.input_ids.device)-started,
        peak_memory=torch.cuda.max_memory_allocated(joint.input_ids.device) if joint.input_ids.device.type == "cuda" else 0.))


def spec_training_gradient(model, batch, config: SpecConfig, *, reference_logits=None,
                           generator=None, fixed_samples=None) -> SpecStepResult:
    """Sections 3--6. Never mutate live weights, .grad buffers or optimizer state."""
    if config.alpha and reference_logits is None:
        raise ValueError("positive alpha requires frozen reference logits")
    joint = joint_batch(batch)
    if config.k > joint.m:
        raise ValueError("k cannot exceed safety pair count")
    all_parameters = OrderedDict(model.named_parameters())
    parameters = OrderedDict((name, p) for name, p in all_parameters.items() if p.requires_grad)
    if not parameters:
        raise ValueError("CBG requires at least one trainable parameter")
    modules = spec_linear_modules(model, config.block_path)
    device = joint.input_ids.device
    started = _clock(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    # Allocate the explicit full-model result lazily from the first endpoint
    # gradients. Preallocating it here raises the peak during the center pass
    # by one complete parameter tree, and copying the first endpoint would
    # transiently require two such trees.
    gradient = OrderedDict()
    base_endpoint_gradient = OrderedDict()
    loss, diagnostics = None, {}
    with spec_precision(model, config.perturb_precision):
        if config.beta == 0:
            return _plain_gradient(model, joint, config, parameters, reference_logits, fixed_samples, generator)
        # Diagnostic runs can freeze the sampled-Fisher token IDs across
        # optimizer steps. Normal training still samples once per step.
        samples = fixed_samples
        phase = _clock(device)
        try:
            spectrum, samples, captures = _center(model, joint, config, modules,
                generator=generator, samples=samples, timings=diagnostics, return_capture=True)
        except DegenerateGeometry:
            return _plain_gradient(model, joint, config, parameters, reference_logits, samples, generator, skipped=True)
        diagnostics["center_time"] = _clock(device)-phase
        for draw in range(config.num_sign_draws):
            phase = _clock(device)
            signs = torch.randint(0, 2, (len(spectrum.kernel),), device=device, generator=generator)*2-1
            cw, cotangent = eigen_sign_scalarization(spectrum, signs)
            captures.materialize(cw, spectrum.A)
            w, dl = captures.w, captures.d_lambda
            if draw == config.num_sign_draws-1:
                captures.captures.clear()
            diagnostics["direction_contraction_time"] = diagnostics.get("direction_contraction_time", 0.) + _clock(device)-phase
            norm_sq = cw @ spectrum.kernel @ cw
            if not bool(torch.isfinite(norm_sq)) or bool(norm_sq < 0):
                raise FloatingPointError("RMS perturbation norm is negative/nonfinite")
            dim = sum(value.numel() for value in w.values())
            zero_direction = bool(norm_sq == 0)
            epsilon = norm_sq.new_zeros(()) if zero_direction else config.r * math.sqrt(dim) / norm_sq.sqrt()
            base_loss = spectrum.log_eta.new_zeros(())
            phase = _clock(device)
            for sign in (1, -1):
                # Only U is replaced. Other trainable leaves participate in the
                # outer derivative and receive their full mixed-Hessian terms.
                endpoint = OrderedDict((name, (
                    p.detach() + sign * epsilon.to(p) * w[name]).requires_grad_(p.requires_grad)
                    if name in w else p) for name, p in all_parameters.items())
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=config.perturb_precision == "bf16"):
                    endpoint_logits = _forward(model, joint, endpoint)
                values = measurements(endpoint_logits, joint, samples=samples,
                                      reference_logits=reference_logits, fisher=config.fisher,
                                      kl_direction=config.kl_direction, logprob_chunk_size=config.logprob_chunk_size)
                base = (0. if config.drop_aln else 1.) * values.alignment + config.alpha * values.benign_kl
                omega = torch.dot(cotangent, values.values.double())
                curvature_scale = 0. if zero_direction else sign * config.beta / epsilon
                endpoint_loss = curvature_scale * omega + .5 * base
                grads = torch.autograd.grad(endpoint_loss, tuple(endpoint[name] for name in parameters), allow_unused=True)
                with torch.no_grad():
                    scale = 1 / config.num_sign_draws
                    for name, part in zip(parameters, grads):
                        if part is None:
                            continue
                        if name not in gradient:
                            # Reuse autograd's detached result as the owned
                            # accumulator instead of cloning a parameter-sized
                            # tensor on the first endpoint.
                            gradient[name] = part.detach()
                            if scale != 1:
                                gradient[name].mul_(scale)
                        else:
                            gradient[name].add_(part, alpha=scale)
                base_loss += .5 * base.detach().double()
                for label, value in (("L_aln", values.alignment), ("L_bgn", values.benign_kl)):
                    diagnostics[label] = diagnostics.get(label, 0.) + .5 * value.detach() / config.num_sign_draws
                del endpoint_logits, values, base, omega, endpoint_loss, grads
                if config.log_gradient_components:
                    # Independent graph: do not retain the endpoint graph merely
                    # to measure components. This is explicitly extra compute.
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=config.perturb_precision == "bf16"):
                        base_logits = _forward(model, joint, endpoint)
                    base_values = measurements(base_logits, joint, samples=samples,
                        reference_logits=reference_logits, fisher=config.fisher,
                        kl_direction=config.kl_direction, logprob_chunk_size=config.logprob_chunk_size)
                    scalar = (0. if config.drop_aln else 1.)*base_values.alignment + config.alpha*base_values.benign_kl
                    parts = torch.autograd.grad(scalar, tuple(endpoint[name] for name in parameters), allow_unused=True)
                    for name, part in zip(parameters, parts):
                        if part is not None:
                            value = part.detach().to(device="cpu", dtype=torch.float64)*(.5/config.num_sign_draws)
                            if name not in base_endpoint_gradient:
                                base_endpoint_gradient[name] = value
                            else:
                                base_endpoint_gradient[name].add_(value)
                    del base_logits, base_values, scalar, parts, part
                del endpoint
            diagnostics["perturbed_time"] = diagnostics.get("perturbed_time", 0.) + _clock(device) - phase
            if not config.detach_lambda:
                for name, value in dl.items():
                    if name not in parameters:
                        continue
                    if name not in gradient:
                        gradient[name] = torch.zeros_like(parameters[name])
                    gradient[name].add_(value, alpha=config.beta / config.num_sign_draws)
            value = (base_loss + config.beta * spectrum.objective_value) / config.num_sign_draws
            loss = value if loss is None else loss + value
            diagnostics.update(spectrum.diagnostics())
            for label, value in (("epsilon", epsilon),
                                 ("b_K_b", norm_sq),
                                 ("w_norm", sum(v.double().square().sum() for v in w.values()).sqrt()),
                                 ("d_lambda_norm", sum(v.double().square().sum() for v in dl.values()).sqrt())):
                diagnostics[label] = diagnostics.get(label, 0.) + value / config.num_sign_draws
            del w, dl
        captures.captures.clear()
        if config.log_gradient_components:
            curvature_sq = 0.
            geometry_sq = 0.
            for name, value in gradient.items():
                geometry = value.detach().to(device="cpu", dtype=torch.float64)-base_endpoint_gradient.get(name, 0.)
                geometry_sq += float(geometry.square().sum())
                curvature = geometry
                if not config.detach_lambda and name in captures.d_lambda:
                    curvature -= config.beta*captures.d_lambda[name].detach().to(device="cpu", dtype=torch.float64)
                curvature_sq += float(curvature.square().sum())
            plain = _plain_gradient(model, joint, config, parameters, reference_logits, samples, generator)
            diagnostics["base_gradient_norm"] = sum(float(g.double().square().sum()) for g in plain.gradient.values())**.5
            diagnostics["base_endpoint_gradient_norm"] = sum(float(g.square().sum()) for g in base_endpoint_gradient.values())**.5
            diagnostics["beta_curvature_norm"] = curvature_sq**.5
            diagnostics["beta_geometry_norm"] = geometry_sq**.5
            diagnostics["geometry_to_base_grad_ratio"] = (diagnostics["beta_geometry_norm"] /
                diagnostics["base_gradient_norm"] if diagnostics["base_gradient_norm"] > 0 else -1.)
            del plain, base_endpoint_gradient
    # Restore the exact model parameter order and explicit zeros for any
    # coordinates disconnected from the endpoint objective.
    gradient = OrderedDict((name, gradient[name] if name in gradient else torch.zeros_like(parameter))
                           for name, parameter in parameters.items())
    if any(not bool(torch.isfinite(g).all()) for g in gradient.values()):
        raise FloatingPointError("nonfinite CBG outer gradient")
    if loss is None or not bool(torch.isfinite(loss)):
        raise FloatingPointError("nonfinite CBG objective")
    diagnostics.update(step_time=_clock(device) - started,
                       safety_input_tokens=joint.attention_mask[:2*joint.m].sum(),
                       benign_input_tokens=joint.attention_mask[2*joint.m:].sum(),
                       benign_response_tokens=(joint.response_mask[2*joint.m:,1:].bool() & joint.attention_mask[2*joint.m:,1:].bool()).sum(),
                       peak_memory=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                       ablation_empirical=float(config.fisher == "empirical"),
                       ablation_euclidean=float(config.damping == "euclidean"),
                       ablation_detach_lambda=float(config.detach_lambda),
                       ablation_raw_rho=float(config.objective == "raw_rho"),
                       ablation_drop_aln=float(config.drop_aln),
                       ablation_rayleigh=float(config.solver == "rayleigh_K"),
                       ablation_bf16_perturb=float(config.perturb_precision == "bf16"),
                       num_sign_draws=config.num_sign_draws)
    diagnostics.update(geometry_active=1., geometry_skipped=0., nan_flag=0.,
                       gradient_components_measured=float(config.log_gradient_components),
                       beta_d_lambda_norm=config.beta*diagnostics["d_lambda_norm"],
                       fraction_below_xi=sum((module.weight.detach().abs() < config.xi).sum() for module in modules.values()) / sum(module.weight.numel() for module in modules.values()))
    return SpecStepResult(gradient, loss.detach(), diagnostics)


def measure_geometry(model, batch, *, config: SpecConfig, generator=None, samples=None):
    """Standalone checkpoint geometry; no parameter updates or .grad writes.

    Pass a seeded generator for comparable sampled-Fisher checkpoint probes.
    """
    joint = joint_batch(batch)
    modules = spec_linear_modules(model, config.block_path)
    with spec_precision(model, config.perturb_precision):
        spectrum, _ = _center(model, joint, config, modules, generator=generator,
                              samples=samples, materialize=False)
    return spectrum.diagnostics()
