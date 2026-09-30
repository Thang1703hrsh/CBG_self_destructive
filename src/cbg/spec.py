"""CBG log-eta implementation from CBG_implementation_spec.md.

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
from src.cbg.spec_spectral import solve_spec_spectrum, sign_scalarization


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

    def __post_init__(self):
        for name in ("xi", "alpha", "beta", "kappa_mu", "r"):
            value = getattr(self, name)
            if value is None or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be explicitly configured, finite and nonnegative")
            if name in {"kappa_mu", "r"} and value == 0:
                raise ValueError(f"{name} must be positive")
        for name in ("k", "gram_max_bytes", "num_sign_draws"):
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
        if self.perturb_precision not in {"fp32", "bf16"}:
            raise ValueError("perturb_precision must be fp32 or explicit bf16 ablation")


@contextmanager
def spec_precision(model):
    """Deterministic true FP32/64 forwards, restoring caller modes/precision."""
    tensors = list(model.parameters()) + list(model.buffers())
    if any(t.is_floating_point() and t.dtype not in (torch.float32, torch.float64) for t in tensors):
        raise ValueError("spec CBG requires FP32 master weights/compute (FP64 for validation)")
    devices = {p.device for p in model.parameters()}
    if len(devices) != 1:
        raise ValueError("spec CBG currently supports one model device")
    device = next(iter(devices))
    if getattr(model, "is_gradient_checkpointing", False):
        raise ValueError("spec hooks require gradient checkpointing disabled")
    modes = [(module, module.training) for module in model.modules()]
    precision = torch.get_float32_matmul_precision()
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    model.eval()
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.allow_tf32 = False
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


@dataclass(frozen=True)
class Measurements:
    values: Tensor
    alignment: Tensor
    benign_kl: Tensor
    samples: Tensor


def measurements(logits, batch: JointBatch, *, samples=None, reference_logits=None,
                 fisher="sampled", kl_direction="reference_to_current", generator=None):
    """Teacher-forced margins and SUM sampled scores, with global token losses."""
    mask = batch.response_mask[:, 1:].bool() & batch.attention_mask[:, 1:].bool()
    lengths = mask.sum(1)
    if bool((lengths <= 0).any()):
        raise ValueError("each sequence must have at least one predicted response token")
    if logits.shape[:2] != batch.input_ids.shape:
        raise ValueError("logits and joint sequence shapes differ")
    active = logits[:, :-1][mask]
    if active.dtype in (torch.float16, torch.bfloat16):
        active = active.float()
    logp = active.log_softmax(-1)
    targets = batch.input_ids[:, 1:][mask]
    sequence = torch.arange(len(lengths), device=logits.device)[:, None].expand_as(mask)[mask]
    benign_positions = sequence >= 2 * batch.m
    safe_positions = (sequence >= batch.m) & ~benign_positions
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


def _forward(model, batch, parameters=None):
    kwargs = dict(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
    output = model(**kwargs) if parameters is None else functional_call(model, parameters, (), kwargs)
    return _extract_logits(output)


def _center(model, batch, config, modules, *, generator=None, samples=None, materialize=True, timings=None):
    def record(label, start):
        now = _clock(batch.input_ids.device)
        if timings is not None:
            timings[label] = timings.get(label, 0.) + now - start
        return now
    with SpecLinearHooks(modules, batch.seq_to_example, batch.attention_mask,
                         xi=config.xi, max_bytes=config.gram_max_bytes, damping=config.damping) as hooks:
        phase = _clock(batch.input_ids.device)
        center = measurements(_forward(model, batch), batch, samples=samples,
                              fisher=config.fisher, generator=generator)
        seed = center.values.sum()
        phase = record("center_forward_time", phase)
        if hooks.backward_target is None:
            raise ValueError("CBG hooks did not capture a backward target")
        # The hook contractions need output cotangents, not the returned
        # parameter gradients. Targeting the first transformer activation
        # traverses the same graph and fires every hook without allocating a
        # full gradient tensor for each admissible weight.
        torch.autograd.grad(seed, hooks.backward_target, retain_graph=materialize)
        phase = record("gram_backward_time", phase)
        spectrum = solve_spec_spectrum(hooks.K, batch.m, config.k, kappa_mu=config.kappa_mu,
                                       objective=config.objective, solver=config.solver,
                                       rayleigh_steps=config.rayleigh_steps,
                                       rayleigh_step_size=config.rayleigh_step_size, generator=generator)
        phase = record("spectral_solve_time", phase)
        if not materialize:
            return spectrum, center.samples
        signs = torch.randint(0, 2, (2, config.k), device=seed.device, generator=generator) * 2 - 1
        cw, cotangent = sign_scalarization(spectrum, *signs)
        hooks.materialize(cw, spectrum.A)
        torch.autograd.grad(seed, hooks.backward_target)
        record("direction_backward_time", phase)
        return spectrum, center.samples, cotangent, hooks.w, hooks.d_lambda


@dataclass(frozen=True)
class SpecStepResult:
    gradient: OrderedDict
    loss: Tensor
    diagnostics: dict


def _clock(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def spec_training_gradient(model, batch, config: SpecConfig, *, reference_logits=None,
                           generator=None) -> SpecStepResult:
    """Sections 3--6. Never mutate live weights, .grad buffers or optimizer state."""
    if config.alpha and reference_logits is None:
        raise ValueError("positive alpha requires frozen reference logits")
    joint = joint_batch(batch)
    if config.k > joint.m:
        raise ValueError("k cannot exceed safety pair count")
    parameters = OrderedDict((name, p) for name, p in model.named_parameters() if p.requires_grad)
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
    loss, diagnostics = None, {}
    with spec_precision(model):
        samples = None
        for draw in range(config.num_sign_draws):
            phase = _clock(device)
            # Each extra draw recomputes the center to keep live memory bounded.
            # Fisher samples remain fixed across ALL sign draws and endpoints.
            spectrum, samples, cotangent, w, dl = _center(
                model, joint, config, modules, generator=generator, samples=samples, timings=diagnostics)
            diagnostics["center_time"] = diagnostics.get("center_time", 0.) + _clock(device) - phase
            norm_sq = sum((value.double().square() / (
                parameters[name].detach().double().square() + config.xi**2)).sum()
                for name, value in w.items())
            if not bool(torch.isfinite(norm_sq)) or not bool(norm_sq > 0):
                raise FloatingPointError("RMS perturbation direction is zero/nonfinite")
            dim = sum(value.numel() for value in w.values())
            epsilon = config.r * math.sqrt(dim) / norm_sq.sqrt()
            base_loss = spectrum.log_eta.new_zeros(())
            phase = _clock(device)
            for sign in (1, -1):
                # Only U is replaced. Other trainable leaves participate in the
                # outer derivative and receive their full mixed-Hessian terms.
                endpoint = OrderedDict((name, (
                    p.detach() + sign * epsilon.to(p) * w[name]).requires_grad_(True)
                    if name in w else p) for name, p in parameters.items())
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=config.perturb_precision == "bf16"):
                    endpoint_logits = _forward(model, joint, endpoint)
                values = measurements(endpoint_logits, joint, samples=samples,
                                      reference_logits=reference_logits, fisher=config.fisher,
                                      kl_direction=config.kl_direction)
                base = (0. if config.drop_aln else 1.) * values.alignment + config.alpha * values.benign_kl
                omega = torch.dot(cotangent.to(values.values), values.values)
                endpoint_loss = sign * config.beta / epsilon.to(omega) * omega + .5 * base
                grads = torch.autograd.grad(endpoint_loss, tuple(endpoint.values()), allow_unused=True)
                with torch.no_grad():
                    scale = 1 / config.num_sign_draws
                    for (name, _), part in zip(endpoint.items(), grads):
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
                del endpoint, endpoint_logits, values, base, omega, endpoint_loss, grads
            diagnostics["perturbed_time"] = diagnostics.get("perturbed_time", 0.) + _clock(device) - phase
            if not config.detach_lambda:
                for name, value in dl.items():
                    if name not in gradient:
                        gradient[name] = torch.zeros_like(parameters[name])
                    gradient[name].add_(value, alpha=config.beta / config.num_sign_draws)
            value = (base_loss + config.beta * spectrum.objective_value) / config.num_sign_draws
            loss = value if loss is None else loss + value
            diagnostics.update(spectrum.diagnostics())
            for label, value in (("epsilon", epsilon),
                                 ("w_norm", sum(v.double().square().sum() for v in w.values()).sqrt()),
                                 ("d_lambda_norm", sum(v.double().square().sum() for v in dl.values()).sqrt())):
                diagnostics[label] = diagnostics.get(label, 0.) + value / config.num_sign_draws
            del w, dl
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
    return SpecStepResult(gradient, loss.detach(), diagnostics)


def measure_geometry(model, batch, *, config: SpecConfig, generator=None, samples=None):
    """Standalone checkpoint geometry; no parameter updates or .grad writes.

    Pass a seeded generator for comparable sampled-Fisher checkpoint probes.
    """
    joint = joint_batch(batch)
    modules = spec_linear_modules(model, config.block_path)
    with spec_precision(model):
        spectrum, _ = _center(model, joint, config, modules, generator=generator,
                              samples=samples, materialize=False)
    return spectrum.diagnostics()
