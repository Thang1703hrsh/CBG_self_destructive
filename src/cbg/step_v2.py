"""Scalable CBG v2 step (CBG_v2_method_spec.md, sections 4-8) for large models.

Same mathematics as ``src.cbg.spec.spec_training_gradient`` (the verified
reference; ``tests/test_cbg_step_v2.py`` checks the two agree), executed so a
7B full-parameter run fits and runs fast:

* micro-batches: the center pass, the perturbed passes and the reference
  forward run over token-budgeted micro-batches. Safety pairs and benign
  sequences are never mixed in one micro-batch, so each is padded only to its
  own longest sequence. Every loss term carries its share of the full-batch
  normalization (``Totals``), so the summed gradient is the full-batch one.
* layer-split placement: the model may live on several devices
  (``place_layers``); no data parallelism, no gradient communication.
* in-place perturbation of the FP32 master weights with exact restore from a
  copy (or by subtraction), gradients accumulated directly into ``.grad`` and
  ``beta * d_Lambda`` folded into ``.grad`` during the direction pass, so no
  extra parameter-sized trees are allocated.
* separate generators for y_hat sampling and sign draws.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import math
import time

import torch
from torch import Tensor

from src.cbg.spec import JointBatch, Totals, joint_batch, measurements, spec_precision
from src.cbg.spec_hooks import SpecLinearHooks, spec_linear_modules
from src.cbg.spec_spectral import DegenerateGeometry, eigen_sign_scalarization, solve_spec_spectrum


@dataclass(frozen=True)
class StepConfig:
    xi: float
    kappa_mu: float
    r: float
    alpha: float
    beta: float
    k: int = 4
    num_sign_draws: int = 1
    mu_mode: str = "differentiated"
    fisher: str = "sampled"
    kl_direction: str = "reference_to_current"
    drop_aln: bool = False
    detach_lambda: bool = False
    block_path: str = "model.layers"
    # Execution (no effect on the mathematics beyond rounding).
    microbatch_tokens: int = 0          # padded token slots per micro-batch; 0 = one per group
    center_precision: str = "fp32"      # fp32 | bf16 (autocast) for the Gram/center pass
    perturb_precision: str = "fp32"     # fp32 | tf32 | bf16 for the perturbed passes
    capture_dtype: str = "model"        # model | bf16 storage of (X, Delta)
    capture_offload: bool = False
    contraction_tf32: bool = False      # exact when captures are bf16
    restore: str = "copy"               # copy (exact) | subtract (few ulp)
    gram_max_bytes: int = 64 * 1024 * 1024
    logprob_chunk_size: int = 128
    deterministic: bool = True
    # Curvature term: "fd" = central finite difference at radius r (two perturbed passes);
    # "hvp" = exact Hessian-vector product by double backward (no radius, one pass in
    # perturb_precision; bf16 is NOT accurate enough on 7B, see _hvp_pass).
    curvature: str = "fd"
    hvp_microbatch_tokens: int = 0      # micro-batch budget of the HVP pass; 0 = microbatch_tokens

    def __post_init__(self):
        for name in ("xi", "alpha", "beta", "kappa_mu", "r"):
            value = getattr(self, name)
            if value is None or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.kappa_mu <= 0 or self.r <= 0:
            raise ValueError("kappa_mu and r must be positive")
        for name in ("k", "num_sign_draws"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.mu_mode not in {"differentiated", "detached"}:
            raise ValueError("mu_mode must be differentiated or detached")
        if self.fisher not in {"sampled", "empirical"}:
            raise ValueError("fisher must be sampled or empirical")
        if self.center_precision not in {"fp32", "bf16"}:
            raise ValueError("center_precision must be fp32 or bf16")
        if self.perturb_precision not in {"fp32", "tf32", "bf16"}:
            raise ValueError("perturb_precision must be fp32, tf32 or bf16")
        if self.capture_dtype not in {"model", "bf16"}:
            raise ValueError("capture_dtype must be model or bf16")
        if self.restore not in {"copy", "subtract"}:
            raise ValueError("restore must be copy or subtract")
        for name in ("microbatch_tokens", "hvp_microbatch_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.curvature not in {"fd", "hvp"}:
            raise ValueError("curvature must be fd or hvp")


# ---------------------------------------------------------------------------
# Layer-split placement
# ---------------------------------------------------------------------------
def _move(value, device):
    if isinstance(value, Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, tuple):
        return tuple(_move(v, device) for v in value)
    if isinstance(value, list):
        return [_move(v, device) for v in value]
    if isinstance(value, dict):
        return {k: _move(v, device) for k, v in value.items()}
    return value


def place_layers(model, devices, block_path="model.layers"):
    """Contiguous transformer blocks per device; embeddings first, norm/head last.

    Every block and the final norm get a pre-hook that moves its inputs
    (hidden states, masks, rotary tables) to its own device. Returns the
    block -> device assignment. Inputs must be given on devices[0].
    """
    devices = [torch.device(d) for d in devices]
    blocks = model.get_submodule(block_path)
    count = len(blocks)
    model.to(devices[0])
    assignment = []
    for index, block in enumerate(blocks):
        device = devices[min(index * len(devices) // count, len(devices) - 1)]
        block.to(device)
        assignment.append(str(device))
        block.register_forward_pre_hook(
            lambda module, args, kwargs, device=device: (_move(args, device), _move(kwargs, device)),
            with_kwargs=True)
    inner = model.get_submodule(block_path.rsplit(".", 1)[0]) if "." in block_path else model
    for name in ("norm",):
        if hasattr(inner, name):
            module = getattr(inner, name).to(devices[-1])
            module.register_forward_pre_hook(
                lambda module, args, kwargs: (_move(args, devices[-1]), _move(kwargs, devices[-1])),
                with_kwargs=True)
    head = model.get_output_embeddings()
    if head is not None:
        if head.weight is model.get_input_embeddings().weight:
            raise ValueError("tied embeddings cannot be split across devices")
        head.to(devices[-1])
    return assignment


# ---------------------------------------------------------------------------
# Micro-batches
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Chunk:
    batch: JointBatch       # local micro-batch: m pairs or n benign sequences, trimmed width
    seq_examples: Tensor    # global example index of each local sequence (for the hooks)
    examples: Tensor        # global example index of each returned measurement value
    benign: tuple           # [start, end) of its benign response tokens in the full stream


def _response_mask(joint):
    return joint.response_mask[:, 1:].bool() & joint.attention_mask[:, 1:].bool()


def make_totals(joint) -> Totals:
    mask = _response_mask(joint)
    m = joint.m
    return Totals(m=m, n=joint.n, safe_tokens=int(mask[m:2*m].sum()),
                  benign_tokens=int(mask[2*m:].sum()))


def make_chunks(joint, token_budget=0):
    """Split pairs and benign sequences into micro-batches of <= token_budget slots.

    A slot is one (sequence, position) of the padded micro-batch, i.e. width is
    the longest attended length inside that micro-batch. A pair costs two rows.
    """
    m, n = joint.m, joint.n
    attended = joint.attention_mask.bool()
    lengths = torch.where(attended.any(1), attended.int().cumsum(1).argmax(1) + 1, 0).tolist()
    benign_counts = _response_mask(joint)[2*m:].sum(1).tolist()
    offsets = [0]
    for count in benign_counts:
        offsets.append(offsets[-1] + int(count))

    def groups(items, rows_per_item, length_of):
        result, current, width = [], [], 0
        for item in items:
            new_width = max(width, length_of(item))
            if current and token_budget and (len(current) + 1) * rows_per_item * new_width > token_budget:
                result.append(current)
                current, new_width = [], length_of(item)
            current.append(item)
            width = new_width
        if current:
            result.append(current)
        return result

    device = joint.input_ids.device
    chunks = []
    for pairs in groups(range(m), 2, lambda i: max(lengths[i], lengths[m + i])):
        rows = list(pairs) + [m + i for i in pairs]
        chunks.append(_chunk(joint, rows, len(pairs), 0, lengths,
                             torch.tensor(list(pairs) * 2, device=device),
                             torch.tensor(list(pairs), device=device), (0, 0)))
    for benign in groups(range(n), 1, lambda j: lengths[2*m + j]):
        rows = [2*m + j for j in benign]
        ids = torch.tensor([m + j for j in benign], device=device)
        chunks.append(_chunk(joint, rows, 0, len(benign), lengths, ids, ids,
                             (offsets[benign[0]], offsets[benign[-1] + 1])))
    return chunks


def _chunk(joint, rows, m, n, lengths, seq_examples, examples, benign):
    index = torch.tensor(rows, device=joint.input_ids.device)
    width = max(lengths[r] for r in rows)
    take = lambda t: t.index_select(0, index)[:, :width]
    local = JointBatch(take(joint.input_ids), take(joint.attention_mask), take(joint.response_mask),
                       seq_examples, m=m, n=n)
    return Chunk(local, seq_examples, examples, benign)


# ---------------------------------------------------------------------------
# Passes
# ---------------------------------------------------------------------------
def _clock(devices):
    for device in devices:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return time.perf_counter()


def _autocast(device_type, precision):
    return torch.autocast(device_type, dtype=torch.bfloat16, enabled=precision == "bf16")


def _logits(model, batch):
    output = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
    return output.logits if hasattr(output, "logits") else output[0]


@contextmanager
def _deterministic(enabled):
    previous = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    if enabled:
        torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn)


def reference_logits(reference_model, joint, chunks, *, logprob_device=None, precision="fp32"):
    """Frozen theta_0 logits at every benign response position, in stream order."""
    parts = []
    device_type = joint.input_ids.device.type
    with spec_precision(reference_model, "fp32"), torch.no_grad():
        for chunk in chunks:
            if not chunk.batch.n:
                continue
            with _autocast(device_type, precision):
                logits = _logits(reference_model, chunk.batch)
            mask = _response_mask(chunk.batch).to(logits.device)
            parts.append(logits[:, :-1][mask].float())
            del logits
    result = torch.cat(parts)
    return result if logprob_device is None else result.to(logprob_device)


def _slice(tensor, chunk):
    return None if tensor is None else tensor[chunk.benign[0]:chunk.benign[1]]


def center_pass(model, joint, chunks, totals, config, modules, *, sample_generator=None, samples=None):
    """Stage 1-2: capture (X, Delta) over all micro-batches; returns (hooks, samples)."""
    device_type = joint.input_ids.device.type
    hooks = SpecLinearHooks(modules, num_examples=joint.m + joint.n, xi=config.xi,
                            max_bytes=config.gram_max_bytes, capture_dtype=config.capture_dtype,
                            offload=config.capture_offload, contraction_tf32=config.contraction_tf32)
    parts = []
    with hooks:
        for chunk in chunks:
            hooks.begin_call(chunk.seq_examples, chunk.batch.attention_mask)
            with _autocast(device_type, config.center_precision):
                logits = _logits(model, chunk.batch)
            fixed = _slice(samples, chunk)
            values = measurements(logits, chunk.batch, samples=fixed, fisher=config.fisher,
                                  generator=sample_generator, logprob_chunk_size=config.logprob_chunk_size,
                                  totals=totals)
            if hooks.backward_target is None:
                raise ValueError("CBG hooks did not capture a backward target")
            # All-ones seed: the rows of Delta for example e are dF~_e/dZ.
            torch.autograd.grad(values.values.sum(), hooks.backward_target)
            hooks.backward_target = None
            parts.append(values.samples)
            del logits, values
    if hooks.captured_names() != set(modules):
        raise ValueError("center backward missed an admissible Linear; incomplete Gram rejected")
    return hooks, torch.cat(parts)


def _set_weights(modules, w, backup, offset, target, config):
    """Move every U weight from theta + offset*w to theta + target*w."""
    with torch.no_grad():
        for name, module in modules.items():
            if config.restore == "copy":
                if target == 0:
                    module.weight.copy_(backup[name])
                else:
                    torch.add(backup[name], w[name], alpha=target, out=module.weight)
            else:
                module.weight.add_(w[name], alpha=target - offset)
    return target


def _endpoint_passes(model, chunks, totals, config, modules, w, cotangent, epsilon, *, samples,
                     reference, scale, backup, curvature_weight, base_weight):
    """sum_sigma grad[ sigma*cw/eps * a'F~ + bw/2 * (L_aln + alpha L_bgn) ] into .grad."""
    device_type = chunks[0].batch.input_ids.device.type
    stats = dict(L_aln=0., L_bgn=0.)
    offset = 0.
    with _deterministic(config.deterministic):
        for sign in (1., -1.):
            offset = _set_weights(modules, w, backup, offset, sign * epsilon, config)
            for chunk in chunks:
                with _autocast(device_type, config.perturb_precision):
                    logits = _logits(model, chunk.batch)
                values = measurements(logits, chunk.batch, samples=_slice(samples, chunk),
                                      reference_logits=_slice(reference, chunk), fisher=config.fisher,
                                      kl_direction=config.kl_direction,
                                      logprob_chunk_size=config.logprob_chunk_size, totals=totals)
                base = (0. if config.drop_aln else 1.) * values.alignment + config.alpha * values.benign_kl
                loss = .5 * base_weight * base
                if curvature_weight:
                    a = cotangent[chunk.examples.to(cotangent.device)]
                    omega = torch.dot(a.to(values.values.device), values.values.double())
                    loss = loss + (sign * curvature_weight / epsilon) * omega
                (loss * scale).backward()
                stats["L_aln"] += .5 * float(values.alignment) * scale
                stats["L_bgn"] += .5 * float(values.benign_kl) * scale
                del logits, values, base, loss
    _set_weights(modules, w, backup, offset, 0., config)
    return stats


def _hvp_pass(model, chunks, totals, config, modules, w, cotangent, *, samples, reference, scale,
              curvature_weight, base_weight):
    """Exact counterpart of ``_endpoint_passes``, accumulated into .grad.

    _endpoint_passes takes the gradient of
        sum_sigma [ sigma * cw / eps * a'F~(theta + sigma eps w) + bw/2 * base(theta + sigma eps w) ],
    whose limit eps -> 0 is  2 cw * H_omega w + bw * grad base(theta),  with omega = a'F~ and H_omega
    the Hessian of omega in all trainable parameters, taken along w (nonzero on U only). Here
        H_omega w = grad_theta < grad_U omega(theta), w >,
    the directional derivative of omega along w, differentiated once more (double backward). No
    weight is moved, so no radius, restore copy or deterministic kernels are needed. Precision
    still matters: on Llama-2-7B a bf16-autocast HVP differs from the fp32 one by ~75% (cosine
    0.74) while fp32 FD agrees with fp32 HVP within 2% -- validate with `--stage hvp-check`.
    """
    device_type = chunks[0].batch.input_ids.device.type
    names = list(modules)
    weights = [modules[name].weight for name in names]
    directions = [w[name] for name in names]
    stats = dict(L_aln=0., L_bgn=0.)
    for chunk in chunks:
        with _autocast(device_type, config.perturb_precision):
            logits = _logits(model, chunk.batch)
        values = measurements(logits, chunk.batch, samples=_slice(samples, chunk),
                              reference_logits=_slice(reference, chunk), fisher=config.fisher,
                              kl_direction=config.kl_direction,
                              logprob_chunk_size=config.logprob_chunk_size, totals=totals)
        loss = base_weight * ((0. if config.drop_aln else 1.) * values.alignment + config.alpha * values.benign_kl)
        if curvature_weight:
            a = cotangent[chunk.examples.to(cotangent.device)]
            omega = torch.dot(a.to(values.values.device), values.values.double())
            grads = torch.autograd.grad(omega, weights, create_graph=True, allow_unused=True)
            directional = omega.new_zeros(())
            for grad, direction in zip(grads, directions):
                if grad is not None:   # a chunk may not reach every Linear (e.g. no benign rows)
                    directional = directional + (grad * direction).sum(dtype=torch.float64).to(omega.device)
            loss = loss + 2 * curvature_weight * directional
            del grads, directional
        if loss.requires_grad:
            (loss * scale).backward()
        stats["L_aln"] += float(values.alignment) * scale
        stats["L_bgn"] += float(values.benign_kl) * scale
        del logits, values, loss
    return stats


def plain_gradient(model, joint, config, *, reference=None, samples=None, sample_generator=None,
                   chunks=None, totals=None, scale=1.):
    """Exact grad of L_aln + alpha L_bgn at theta into .grad (beta=0, lazy, degenerate)."""
    chunks = chunks or make_chunks(joint, config.microbatch_tokens)
    totals = totals or make_totals(joint)
    device_type = joint.input_ids.device.type
    stats = dict(L_aln=0., L_bgn=0.)
    with spec_precision(model, config.perturb_precision), _deterministic(config.deterministic):
        for chunk in chunks:
            with _autocast(device_type, config.perturb_precision):
                logits = _logits(model, chunk.batch)
            # F~ is unused here: observed targets avoid consuming a sampling RNG.
            values = measurements(logits, chunk.batch, samples=None,
                                  reference_logits=_slice(reference, chunk), fisher="empirical",
                                  kl_direction=config.kl_direction,
                                  logprob_chunk_size=config.logprob_chunk_size, totals=totals)
            loss = (0. if config.drop_aln else 1.) * values.alignment + config.alpha * values.benign_kl
            (loss * scale).backward()
            stats["L_aln"] += float(values.alignment)
            stats["L_bgn"] += float(values.benign_kl)
            del logits, values, loss
    return stats


def cbg_step(model, batch, config: StepConfig, *, reference=None, sample_generator=None,
             sign_generator=None, fixed_samples=None, signs=None, include_base=True):
    """Accumulate the CBG v2 gradient (spec section 8) into the parameters' .grad.

    Does not zero .grad and does not step an optimizer. ``batch`` is either the
    collated {harmful, safe, benign} mapping or a JointBatch. ``reference``
    holds theta_0 logits at the benign response positions (``reference_logits``);
    it is required when alpha > 0 and include_base. With ``include_base=False``
    only the geometry term beta * grad log eta_k is accumulated (no L_aln, no
    L_bgn, nothing on a degenerate batch). Returns per-step diagnostics.
    """
    joint = batch if isinstance(batch, JointBatch) else joint_batch(batch)
    if config.k > joint.m:
        raise ValueError("k cannot exceed the number of safety pairs")
    if not include_base:
        reference = None
    elif config.alpha and reference is None:
        raise ValueError("positive alpha requires theta_0 reference logits")
    modules = spec_linear_modules(model, config.block_path)
    devices = sorted({p.device for p in model.parameters()}, key=str)
    for device in devices:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
    chunks = make_chunks(joint, config.microbatch_tokens)
    totals = make_totals(joint)
    started = _clock(devices)
    diagnostics = dict(m=joint.m, n=joint.n, microbatches=len(chunks), curvature_hvp=float(config.curvature == "hvp"),
                       token_slots=sum(c.batch.input_ids.numel() for c in chunks),
                       safe_response_tokens=totals.safe_tokens, benign_response_tokens=totals.benign_tokens)
    if config.beta == 0:
        stats = plain_gradient(model, joint, config, reference=reference, samples=fixed_samples,
                               sample_generator=sample_generator, chunks=chunks, totals=totals) \
            if include_base else {}
        diagnostics.update(stats, geometry_active=0., geometry_skipped=0.,
                           step_time=_clock(devices) - started)
        return diagnostics
    with spec_precision(model, config.perturb_precision):
        phase = started
        hooks, samples = center_pass(model, joint, chunks, totals, config, modules,
                                     sample_generator=sample_generator, samples=fixed_samples)
        phase, diagnostics["time_center"] = _clock(devices), _clock(devices) - phase
        try:
            kernel = hooks.K
            phase, diagnostics["time_gram"] = _clock(devices), _clock(devices) - phase
            spectrum = solve_spec_spectrum(kernel, joint.m, config.k, kappa_mu=config.kappa_mu,
                                           mu_mode=config.mu_mode)
        except DegenerateGeometry:
            hooks.captures.clear()
            stats = plain_gradient(model, joint, config, reference=reference, samples=samples,
                                   chunks=chunks, totals=totals) if include_base else {}
            diagnostics.update(stats, geometry_active=0., geometry_skipped=1.,
                               step_time=_clock(devices) - started)
            return diagnostics
        diagnostics.update({key: float(value) for key, value in spectrum.diagnostics().items()})
        dim = sum(module.weight.numel() for module in modules.values())
        hvp = config.curvature == "hvp"
        backup = (OrderedDict((name, module.weight.detach().clone()) for name, module in modules.items())
                  if config.restore == "copy" and not hvp else None)
        hvp_chunks = make_chunks(joint, config.hvp_microbatch_tokens or config.microbatch_tokens) if hvp else None
        draws = config.num_sign_draws
        totals_stats = dict(L_aln=0., L_bgn=0., epsilon=0., b_K_b=0.)
        diagnostics.update(time_direction=0., time_perturbed=0.)
        for draw in range(draws):
            phase = _clock(devices)
            draw_signs = (signs[draw] if signs is not None else
                          torch.randint(0, 2, (len(kernel),), device=kernel.device, generator=sign_generator) * 2 - 1)
            b, a = eigen_sign_scalarization(spectrum, draw_signs)
            # d_Lambda does not depend on the signs: add beta * d_Lambda once.
            first = draw == 0 and not config.detach_lambda
            hooks.materialize(b, spectrum.A, d_lambda_scale=config.beta if first else None,
                              compute_d_lambda=first)
            if first:
                diagnostics["beta_d_lambda_norm"] = config.beta * hooks.d_lambda_sq ** .5
            w = hooks.w
            if draw == draws - 1:
                hooks.captures.clear()
            norm_sq = b @ kernel @ b
            if not bool(torch.isfinite(norm_sq)) or bool(norm_sq <= 0):
                raise FloatingPointError("RMS perturbation norm is zero, negative or nonfinite")
            epsilon = 0. if hvp else config.r * math.sqrt(dim) / float(norm_sq.sqrt())
            diagnostics["time_direction"] += _clock(devices) - phase
            phase = _clock(devices)
            if hvp:
                stats = _hvp_pass(model, hvp_chunks, totals, config, modules, w, a, samples=samples,
                                  reference=reference, scale=1. / draws, curvature_weight=config.beta,
                                  base_weight=1. if include_base else 0.)
            else:
                stats = _endpoint_passes(model, chunks, totals, config, modules, w, a, epsilon,
                                         samples=samples, reference=reference, scale=1. / draws,
                                         backup=backup, curvature_weight=config.beta,
                                         base_weight=1. if include_base else 0.)
            diagnostics["time_perturbed"] += _clock(devices) - phase
            for key in ("L_aln", "L_bgn"):
                totals_stats[key] += stats[key]
            totals_stats["epsilon"] += epsilon / draws
            totals_stats["b_K_b"] += float(norm_sq) / draws
            hooks.w = OrderedDict()
            del w, b, a
        del backup
        diagnostics.update(totals_stats)
        diagnostics["fraction_below_xi"] = float(
            sum(int((module.weight.detach().abs() < config.xi).sum()) for module in modules.values()) / dim)
    diagnostics.update(geometry_active=1., geometry_skipped=0., step_time=_clock(devices) - started)
    for index, device in enumerate(devices):
        if device.type == "cuda":
            diagnostics[f"peak_memory_gib_{index}"] = torch.cuda.max_memory_allocated(device) / 2**30
    return diagnostics


def curvature_estimates(model, joint, config, *, plan, sample_generator, sign_generator, sink, skip_oom=False):
    """Curvature term 2 H_omega w (no d_Lambda, no base loss) by several estimators.

    ``plan`` lists ("fd", radius) and ("hvp", precision) entries. All share one
    center pass, one y_hat and one sign draw, so they estimate the same vector.
    FD runs in config.perturb_precision; HVP in the given precision with the
    config.hvp_microbatch_tokens budget. ``sink(kind, value, seconds, epsilon)`` is
    called after each entry with the estimate in ``.grad`` and must consume it;
    .grad is cleared before each entry. With ``skip_oom`` an entry that runs out of
    GPU memory is reported as ``seconds=None`` (weights restored) and the rest continue.
    """
    joint = joint if isinstance(joint, JointBatch) else joint_batch(joint)
    modules = spec_linear_modules(model, config.block_path)
    params = [p for p in model.parameters() if p.requires_grad]
    devices = sorted({p.device for p in params}, key=str)
    chunks = make_chunks(joint, config.microbatch_tokens)
    totals = make_totals(joint)
    with spec_precision(model, config.perturb_precision):
        hooks, samples = center_pass(model, joint, chunks, totals, config, modules,
                                     sample_generator=sample_generator)
        kernel = hooks.K
        spectrum = solve_spec_spectrum(kernel, joint.m, config.k, kappa_mu=config.kappa_mu,
                                       mu_mode=config.mu_mode)
        signs = torch.randint(0, 2, (len(kernel),), device=kernel.device, generator=sign_generator) * 2 - 1
        b, a = eigen_sign_scalarization(spectrum, signs)
        hooks.materialize(b, spectrum.A, compute_d_lambda=False)
        hooks.captures.clear()
        w = hooks.w
        dim = sum(module.weight.numel() for module in modules.values())
        norm = float((b @ kernel @ b).sqrt())
        backup = (OrderedDict((name, module.weight.detach().clone()) for name, module in modules.items())
                  if any(kind == "fd" for kind, _ in plan) else None)
        for kind, value in plan:
            for p in params:
                p.grad = None
            for device in devices:
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
            started = _clock(devices)
            if kind not in {"fd", "hvp"}:
                raise ValueError(f"unknown curvature estimator {kind!r}")
            epsilon = value * math.sqrt(dim) / norm if kind == "fd" else None
            try:
                if kind == "fd":
                    _endpoint_passes(model, chunks, totals, replace_config(config, restore="copy"), modules, w, a,
                                     epsilon, samples=samples, reference=None, scale=1., backup=backup,
                                     curvature_weight=1., base_weight=0.)
                else:
                    hvp_config = replace_config(config, perturb_precision=value)
                    hvp_chunks = make_chunks(joint, config.hvp_microbatch_tokens or config.microbatch_tokens)
                    with spec_precision(model, value):
                        _hvp_pass(model, hvp_chunks, totals, hvp_config, modules, w, a, samples=samples,
                                  reference=None, scale=1., curvature_weight=1., base_weight=0.)
            except torch.cuda.OutOfMemoryError:
                if not skip_oom:
                    raise
                if backup is not None:   # an interrupted FD pass may have left theta + eps w in place
                    _set_weights(modules, w, backup, 0., 0., replace_config(config, restore="copy"))
                for p in params:
                    p.grad = None
                torch.cuda.empty_cache()
                sink(kind, value, None, epsilon)
                continue
            sink(kind, value, _clock(devices) - started, epsilon)
        del backup
    return spectrum.diagnostics()


def curvature_gradient(model, joint, config, *, radii, sample_generator, sign_generator, sink):
    """Finite-difference curvature term at several radii; ``sink(radius, epsilon)`` per radius."""
    return curvature_estimates(model, joint, config, plan=[("fd", radius) for radius in radii],
                               sample_generator=sample_generator, sign_generator=sign_generator,
                               sink=lambda kind, radius, seconds, epsilon: sink(radius, epsilon))


def replace_config(config, **changes):
    from dataclasses import replace
    return replace(config, **changes)


def measure_geometry(model, batch, config: StepConfig, *, sample_generator=None, samples=None):
    """Stages 1-3 only (spec section 11.1); no parameter or .grad changes."""
    joint = batch if isinstance(batch, JointBatch) else joint_batch(batch)
    modules = spec_linear_modules(model, config.block_path)
    chunks = make_chunks(joint, config.microbatch_tokens)
    totals = make_totals(joint)
    with spec_precision(model, config.perturb_precision):
        hooks, samples = center_pass(model, joint, chunks, totals, config, modules,
                                     sample_generator=sample_generator, samples=samples)
        kernel = hooks.K
        hooks.captures.clear()
        spectrum = solve_spec_spectrum(kernel, joint.m, config.k, kappa_mu=config.kappa_mu,
                                       mu_mode=config.mu_mode)
    result = {key: float(value) for key, value in spectrum.diagnostics().items()}
    hh = kernel[:joint.m, :joint.m]
    result["eff_rank_Khh"] = float(hh.trace().square() / hh.square().sum().clamp_min(1e-300))
    result["top_nu_share_Khh"] = float(spectrum.nu[0] / hh.trace().clamp_min(1e-300))
    return result, samples
