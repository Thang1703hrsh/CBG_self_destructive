"""CBG v2 defense training for large models on one host (layer-split over its GPUs).

Stages
  train   full run (resumable from <output>/checkpoint)
  smoke   a few steps: per-stage time, peak memory per GPU and a projected total
  radius  finite-difference self-consistency of the curvature term at r/10 .. 10r

Usage: python -m src.cbg.train_v2 --config <yaml> --data <dir> --output <dir> --model <path> [--stage ...]
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import time

import torch
import yaml

from src.cbg.spec import joint_batch
from src.cbg.step_v2 import (StepConfig, cbg_step, curvature_estimates, curvature_gradient, make_chunks,
                             measure_geometry, place_layers, plain_gradient, reference_logits,
                             unlearning_gradient, attack_sign_direction, harmful_sft_loss, shift_weights)
from src.data_processing.cbg_datasets import (CbgBenignDataset, CbgPairedSafetyDataset,
                                              _collate_sequences, collate_cbg_safety_batch)
from src.experiments.protocol import digest, file_digest, write_json


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key in ("model", "seeds", "data", "lengths", "method", "batch", "execution", "optimizer", "logging"):
        if key not in config:
            raise ValueError(f"config is missing section {key!r}")
    return config


def step_config(config):
    method, execution = config["method"], config["execution"]
    if method.get("estimator", "shared_sign") != "shared_sign":
        raise NotImplementedError("only the shared eigen-sign estimator is implemented")
    if method.get("benign_metric", "sampled_fisher") not in {"sampled_fisher", "empirical_fisher"}:
        raise ValueError("benign_metric must be sampled_fisher or empirical_fisher")
    curvature = method.get("curvature", "fd")
    # FD needs fp32 at r ~ 1e-7; the exact HVP has no cancellation and runs in hvp_precision.
    perturb = execution.get("hvp_precision", "fp32") if curvature == "hvp" else execution["perturb_precision"]
    return StepConfig(
        curvature=curvature, hvp_microbatch_tokens=execution.get("hvp_microbatch_tokens", 0),
        xi=method["xi"], kappa_mu=method["kappa_mu"], r=method["radius"], alpha=method["alpha"],
        beta=method["beta"], k=method["k"], num_sign_draws=method["num_sign_draws"],
        mu_mode=method["mu_mode"], kl_direction=method.get("kl_direction", "reference_to_current"),
        fisher="sampled" if method.get("benign_metric", "sampled_fisher") == "sampled_fisher" else "empirical",
        microbatch_tokens=execution["microbatch_tokens"], center_precision=execution["center_precision"],
        perturb_precision=perturb, capture_dtype=execution["capture_dtype"],
        capture_offload=execution["capture_offload"], contraction_tf32=execution["contraction_tf32"],
        restore=execution["restore"], gram_max_bytes=execution["gram_max_bytes"],
        logprob_chunk_size=execution["logprob_chunk_size"],
        deterministic=execution["deterministic_perturbed_passes"])


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def read_rows(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def datasets(config, data_dir, tokenizer):
    from src.experiments.prepare_v2 import validate_paper_data
    validate_paper_data(config, data_dir)
    lengths = config["lengths"]
    minimum = lengths.get("min_response_tokens")
    text_format = config["data"].get("text_format", "cbg")
    def safety(name):
        rows = read_rows(Path(data_dir) / f"{name}.json")
        return CbgPairedSafetyDataset(rows, tokenizer, max_length=lengths["safety"], min_response_tokens=minimum,
                                      text_format=text_format)
    def benign(name):
        rows = read_rows(Path(data_dir) / f"{name}.json")
        return CbgBenignDataset(rows, tokenizer, max_length=lengths["benign"], min_response_tokens=minimum,
                                text_format=text_format)
    return dict(safety=safety("defense"), benign=benign("benign"),
                geo_safety=safety("d_geo_safety"), geo_benign=benign("d_geo_benign"))


def _epoch_order(size, seed, epoch, stream):
    order = list(range(size))
    random.Random(seed * 1_000_003 + 2 * epoch + stream).shuffle(order)
    return order


def step_indices(step, count, size, seed, stream, order="reshuffle"):
    """Indices for one step, a pure function of step.

    ``sequential``: rows in file order, cycling (SEAM: DataLoader shuffle=False).
    ``reshuffle``: independent reshuffle per epoch.
    """
    if order == "sequential":
        return [position % size for position in range(step * count, (step + 1) * count)]
    if order != "reshuffle":
        raise ValueError("data order must be sequential or reshuffle")
    out, orders = [], {}
    for position in range(step * count, (step + 1) * count):
        epoch = position // size
        if epoch not in orders:
            orders[epoch] = _epoch_order(size, seed, epoch, stream)
        out.append(orders[epoch][position % size])
    return out


def training_batch(data, step, config):
    m, n = config["batch"]["m"], config["batch"]["n"]
    seed, order = config["seeds"]["data_order"], config["data"].get("order", "reshuffle")
    safety = [data["safety"][i] for i in step_indices(step, m, len(data["safety"]), seed, 0, order)]
    benign = [data["benign"][i] for i in step_indices(step, n, len(data["benign"]), seed, 1, order)]
    batch = collate_cbg_safety_batch(safety)
    batch["benign"] = _collate_sequences(benign)
    return batch


def geometry_batches(data, config):
    m, n = config["batch"]["m"], config["batch"]["n"]
    count = min(len(data["geo_safety"]) // m, len(data["geo_benign"]) // n)
    batches = []
    for index in range(count):
        batch = collate_cbg_safety_batch([data["geo_safety"][index * m + i] for i in range(m)])
        batch["benign"] = _collate_sequences([data["geo_benign"][index * n + j] for j in range(n)])
        batches.append(batch)
    return batches


def to_device(batch, device):
    if isinstance(batch, dict):
        return {key: to_device(value, device) for key, value in batch.items()}
    return batch.to(device) if isinstance(batch, torch.Tensor) else batch


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def load_split_model(path, devices, *, trainable):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32, attn_implementation="eager",
                                                 low_cpu_mem_usage=True)
    model.config.use_cache = False
    place_layers(model, devices)
    model.requires_grad_(trainable)
    model.eval()
    return model


def gpu_devices(config):
    count = config["execution"]["devices"]
    if isinstance(count, list):          # explicit device list (CPU dry runs in tests)
        return [torch.device(d) for d in count]
    if torch.cuda.device_count() < count:
        raise ValueError(f"{count} GPUs required, {torch.cuda.device_count()} visible")
    return [torch.device(f"cuda:{i}") for i in range(count)]


def grad_norm(model):
    total = 0.
    for p in model.parameters():
        if p.grad is not None:
            total += float(torch.linalg.vector_norm(p.grad, dtype=torch.float64)) ** 2
    return total ** .5


def schedule(config):
    opt = config["optimizer"]
    steps, warmup = opt["steps"], int(opt["steps"] * opt["warmup_ratio"])
    def factor(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = min(1., (step - warmup) / max(1, steps - warmup))
        return .5 * (1 + math.cos(math.pi * progress)) if opt["scheduler"] == "cosine" else 1. - progress
    return factor


def geometry_eval(model, batches, cfg, config, devices):
    rows = []
    for index, batch in enumerate(batches):
        generator = torch.Generator(device=devices[-1]).manual_seed(config["seeds"]["d_geo_y_hat"] + index)
        result, _ = measure_geometry(model, to_device(batch, devices[0]), cfg, sample_generator=generator)
        rows.append(result)
    keys = ("log_eta_k", "eta_k", "rho_mu", "sigma_1k", "nu_1k", "tr_Kbb", "mu", "c_mu", "eff_rank_Kbb",
            "eff_rank_Khh", "top_nu_share_Khh", "gap_sigma", "gap_nu")
    summary = {f"geo/{key}": statistics.mean(row[key] for row in rows) for key in keys}
    summary["geo/log_eta_k_std"] = statistics.pstdev(row["log_eta_k"] for row in rows)
    summary.update({"geo/m": cfg_m(batches), "geo/n": batches[0]["benign"]["input_ids"].shape[0],
                    "geo/batches": len(rows)})
    return summary, rows


def cfg_m(batches):
    return batches[0]["harmful"]["input_ids"].shape[0]


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def save_checkpoint(output, step, model, optimizer, scheduler, generators, extra=None):
    target, temporary = output / "checkpoint", output / "checkpoint.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    torch.save(model.state_dict(), temporary / "model.pt")
    torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
    torch.save(dict(step=step, scheduler=scheduler.state_dict(),
                    generators={k: g.get_state() for k, g in generators.items()},
                    torch_rng=torch.get_rng_state(), extra=dict(extra or {})), temporary / "state.pt")
    if target.exists():
        shutil.rmtree(target)
    temporary.rename(target)


def load_checkpoint(output, model, optimizer, scheduler, generators, extra=None):
    """Restore everything in place (``extra`` is updated from the saved dict); returns the step."""
    source = output / "checkpoint"
    # GPU models: memory-map the files so host RAM never holds the 27 GB model or the 54 GB AdamW
    # state at once (each tensor is paged in and copied to its GPU). CPU models (tests) load normally,
    # since their parameters could otherwise stay backed by a file that a later save deletes.
    mmap = next(model.parameters()).is_cuda
    model.load_state_dict(torch.load(source / "model.pt", map_location="cpu", weights_only=True, mmap=mmap))
    optimizer.load_state_dict(torch.load(source / "optimizer.pt", map_location="cpu", weights_only=True, mmap=mmap))
    state = torch.load(source / "state.pt", map_location="cpu", weights_only=True)
    scheduler.load_state_dict(state["scheduler"])
    for key, generator in generators.items():
        generator.set_state(state["generators"][key])
    torch.set_rng_state(state["torch_rng"])
    if extra is not None:
        extra.update(state.get("extra", {}))
    return state["step"]


# ---------------------------------------------------------------------------
# Lazy CBG with a reused geometry gradient
# ---------------------------------------------------------------------------
class GeometryCache:
    """beta * grad log eta_k from the last CBG step, added to every step of its window.

    ``rescale`` (spec section 9) applies lazy * beta * grad log eta_k once per
    window; under Adam that is a K-times spike every K steps, which inflates the
    second moment and puts the geometry at the learning rate of one step only.
    ``reuse`` adds beta * grad log eta_k on each of the K steps instead: the same
    first-order total, smooth Adam statistics, and the base gradient of every
    step in one precision. The cache lives only between CBG steps (it is freed
    before the next one), so it never adds to the CBG step's peak memory.

    Outlier guard (optional): a geometry norm above ``clip_factor`` times the
    median of the last ``window`` raw norms is scaled down to that bound.
    """

    def __init__(self, dtype, clip_factor=None, window=16, min_history=4, max_failures=3):
        self.dtype = dtype
        self.clip_factor = clip_factor
        self.window, self.min_history, self.max_failures = window, min_history, max_failures
        self.tensors = None
        self.norm = 0.
        self.history = []          # raw geometry norms (checkpointed)
        self.failures = 0          # consecutive failed geometry steps (checkpointed)

    def state(self):
        return dict(geometry_norms=list(self.history), geometry_failures=self.failures)

    def load(self, state):
        self.history = [float(v) for v in state.get("geometry_norms", [])]
        self.failures = int(state.get("geometry_failures", 0))

    def clear(self):
        self.tensors = None

    def capture(self, model, norm):
        """Store the geometry now in .grad (after the guard); returns the applied scale."""
        scale = 1.
        recent = self.history[-self.window:]
        if self.clip_factor and len(recent) >= self.min_history:
            bound = self.clip_factor * statistics.median(recent)
            if norm > bound:
                scale = bound / norm
        self.history = (self.history + [norm])[-self.window:]
        self.tensors = [(p, (p.grad * scale).to(self.dtype)) for p in model.parameters() if p.grad is not None]
        self.norm = norm * scale
        self.failures = 0
        return scale

    def fail(self, reason):
        self.tensors = None
        self.failures += 1
        if self.failures > self.max_failures:
            raise FloatingPointError(f"{self.failures} consecutive failed geometry steps (last: {reason})")

    def apply(self, max_norm=None, per_tensor_ratio=None, balance=False):
        """Add the stored geometry to .grad; returns ||applied|| / ||stored|| (0 if nothing is stored).

        max_norm: scale the whole geometry down to this norm (global cap).
        per_tensor_ratio: scale each tensor's geometry down to ratio * ||.grad|| of that tensor (the gradient of
        the rest). Adam normalizes per coordinate, so a global cap still lets the geometry dominate tensors whose
        base gradient is small; the per-tensor cap bounds its share in every tensor.
        balance (with per_tensor_ratio): scale each tensor's geometry to exactly ratio * ||.grad|| of that tensor,
        up or down, so the geometry has a constant per-tensor weight and beta only sets its direction.
        """
        if not self.tensors:
            return 0.
        applied_sq = 0.
        scale = min(1., max_norm / self.norm) if max_norm is not None and self.norm > 0 else 1.
        for p, value in self.tensors:
            factor = scale
            if per_tensor_ratio is not None:
                rest = float(torch.linalg.vector_norm(p.grad, dtype=torch.float64)) if p.grad is not None else 0.
                own = float(torch.linalg.vector_norm(value, dtype=torch.float64))
                factor = per_tensor_ratio * rest / own if own > 0 else 0.
                if not balance:
                    factor = min(1., factor) if own > 0 else 1.
            applied_sq += factor ** 2 * float(torch.linalg.vector_norm(value, dtype=torch.float64)) ** 2
            if p.grad is None:
                p.grad = (value * factor).to(p.dtype)
            else:
                p.grad.add_(value, alpha=factor)
        return applied_sq ** .5 / self.norm if self.norm > 0 else 1.


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def run(config_path, data_dir, output_dir, model_path, *, stage="train", resume=False, smoke_steps=6):
    from transformers import AutoTokenizer
    config = load_config(config_path)
    cfg = step_config(config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    devices = gpu_devices(config)
    torch.manual_seed(config["seeds"]["seed"])
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    data = datasets(config, data_dir, tokenizer)
    if len(data["safety"]) < config["batch"]["m"] or len(data["benign"]) < config["batch"]["n"]:
        raise ValueError("defense pools are smaller than one batch")
    identity = dict(config=config, config_sha256=file_digest(config_path), step_config=asdict(cfg),
                    data_manifest_sha256=file_digest(Path(data_dir) / "manifest.json"), model_path=str(model_path),
                    devices=[str(d) for d in devices],
                    gpus=[torch.cuda.get_device_name(d) if d.type == "cuda" else "cpu" for d in devices],
                    torch=torch.__version__)
    if stage == "train":
        _write_identity(output / "identity.json", identity, resume)
        done = output / "metrics.json"
        previous = json.loads(done.read_text(encoding="utf-8")).get("status") if resume and done.exists() else None
        if previous == "complete" and (output / "final").exists():
            # A retry after the final save (e.g. a failed upload): never retrain over a finished run.
            print("Run already complete; nothing to resume", flush=True)
            return dict(status="complete", final=str(output / "final"), skipped=True)
        if previous == "early_stopped":
            print("Run was stopped by an early-stop gate; nothing to resume", flush=True)
            return dict(json.loads(done.read_text(encoding="utf-8")), skipped=True)
    if stage == "calibrate":   # choose the trajectory step: model_path is a checkpoint after a real attack
        return calibrate_stage(config, cfg, data, Path(data_dir), Path(model_path), devices, output)
    if stage == "geometry":   # held-out geometry of any checkpoint (e.g. a model after an attack); no training
        model = load_split_model(model_path, devices, trainable=False)
        summary, rows = geometry_eval(model, geometry_batches(data, config), cfg, config, devices)
        result = dict(model=str(model_path), **summary, batches=rows)
        write_json(output / "geometry.json", result)
        print(json.dumps(summary), flush=True)
        return summary
    model = load_split_model(model_path, devices, trainable=True)
    generators = dict(y_hat=torch.Generator(device=devices[-1]).manual_seed(config["seeds"]["y_hat_sampling"]),
                      signs=torch.Generator(device=devices[0]).manual_seed(config["seeds"]["sign_draws"]))
    if stage == "radius":
        return radius_stage(model, data, config, replace(cfg, perturb_precision=config["execution"]["perturb_precision"],
                                                         curvature="fd"), devices, generators, output)
    if stage == "hvp-check":
        return hvp_check_stage(model, data, config, cfg, devices, generators, output)
    reference = load_split_model(model_path, devices, trainable=False)
    opt = config["optimizer"]
    if opt["name"] != "adamw":
        raise ValueError("only AdamW is configured")
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt["lr"], weight_decay=opt["weight_decay"],
                                  betas=tuple(opt.get("betas", (0.9, 0.999))), eps=opt.get("eps", 1e-8),
                                  fused=opt.get("fused", True) and devices[0].type == "cuda")
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule(config))
    steps = opt["steps"] if stage == "train" else smoke_steps
    start = 0
    logging = config["logging"]
    utility_window = logging.get("utility_window", 20)
    if isinstance(utility_window, bool) or not isinstance(utility_window, int) or utility_window < 1:
        raise ValueError("logging.utility_window must be a positive integer")
    method = config["method"]
    lazy = method.get("lazy_period", 1)
    if isinstance(lazy, bool) or not isinstance(lazy, int) or lazy < 1:
        raise ValueError("method.lazy_period must be a positive integer")
    lazy_mode = method.get("lazy_mode", "rescale")
    if lazy_mode not in {"rescale", "reuse"}:
        raise ValueError("method.lazy_mode must be rescale or reuse")
    reuse = lazy > 1 and lazy_mode == "reuse"
    if reuse and logging.get("checkpoint_every") and logging["checkpoint_every"] % lazy:
        raise ValueError("lazy_mode reuse needs checkpoint_every to be a multiple of lazy_period "
                         "(the geometry cache is rebuilt after a resume, not saved)")
    cache = GeometryCache({"bf16": torch.bfloat16, "fp32": torch.float32}[method.get("geometry_cache_dtype", "bf16")],
                          clip_factor=method.get("geometry_clip_factor")) if reuse else None
    log_path = output / ("train_log.jsonl" if stage == "train" else "smoke_log.jsonl")
    if stage == "train" and resume and (output / "checkpoint").exists():
        extra = {}
        start = load_checkpoint(output, model, optimizer, scheduler, generators, extra)
        if cache is not None:
            cache.load(extra)
        if log_path.exists():
            rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            log_path.write_text("".join(json.dumps(r) + "\n" for r in rows if r.get("step", 0) <= start),
                                encoding="utf-8")
        print(f"Resumed at step {start}", flush=True)
    elif log_path.exists():
        if stage == "train" and not resume:
            raise ValueError("training log exists; pass --resume or choose a new output directory")
        log_path.unlink()   # smoke rerun, or a resume that never reached its first checkpoint
    geo = geometry_batches(data, config)
    # Early-stop gates. At each gate step, stop unless every condition of that gate holds:
    #   max_delta_log_eta: geo/log_eta_k(step) - geo/log_eta_k(0) <= value (held-out eta clearly decreased);
    #   max_mean_l_bgn:    mean L_bgn (KL to theta_0 on benign data) of the last 20 steps <= value (utility kept).
    gates = {}
    for g in logging.get("early_stop") or []:
        limits = {k: float(g[k]) for k in ("max_delta_log_eta", "max_mean_l_bgn") if g.get(k) is not None}
        if not limits:
            raise ValueError("an early_stop gate needs max_delta_log_eta and/or max_mean_l_bgn")
        gates[int(g["step"])] = limits
    recent_l_bgn = []
    # Model snapshots kept for checkpoint selection (logging.save_models_at); never overwritten or deleted.
    save_models_at = {int(v) for v in logging.get("save_models_at") or []}
    if gates and stage == "train":
        every = logging.get("geo_eval_every") or 0
        eta_steps = [s for s, limits in gates.items() if "max_delta_log_eta" in limits]
        if eta_steps and (not geo or not every or any(s % every and s != steps for s in eta_steps)):
            raise ValueError("an eta gate needs D_geo batches and a step that is a geo_eval_every multiple")
    baseline = None
    if gates and stage == "train" and start > 0 and log_path.exists():   # resumed: step-0 value and recent L_bgn
        logged = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        baseline = next((r["geo/log_eta_k"] for r in logged if r.get("step") == 0 and "geo/log_eta_k" in r), None)
        recent_l_bgn = [r["L_bgn"] for r in logged if "L_bgn" in r][-utility_window:]
    clip = opt.get("grad_clip")
    wall = time.perf_counter()

    def log(row):
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")

    if stage == "train" and start == 0 and geo and logging["geo_eval_every"]:
        summary, _ = geometry_eval(model, geo, cfg, config, devices)
        baseline = summary["geo/log_eta_k"]
        log(dict(step=0, **summary))
        print(json.dumps(dict(step=0, **summary)), flush=True)
    # Lazy CBG (spec section 9): a full CBG step every `lazy` steps with beta_eff = lazy * beta;
    # the other steps optimize L_aln + alpha L_bgn only, in `plain_precision` (no finite difference).
    # lazy_mode reuse: beta stays beta and the CBG step's geometry is added on every step (GeometryCache).
    rescale = method.get("lazy_rescale", True) and not reuse
    active_cfg = replace(cfg, beta=cfg.beta * (lazy if rescale else 1))
    plain_precision = config["execution"].get("plain_precision", "fp32")
    plain_cfg = replace(cfg, beta=0., perturb_precision=plain_precision, deterministic=False)
    # CBG+UL (spec section 9): SEAM's unlearning loss on the harmful responses of the same pairs, weight ul_weight,
    # computed at theta on every step in plain_precision (SEAM's own weight is 1).
    ul_weight = float(method.get("ul_weight", 0.) or 0.)
    if not math.isfinite(ul_weight) or ul_weight < 0:
        raise ValueError("method.ul_weight must be finite and nonnegative")
    # Two-stage CBG+UL: no geometry term before geometry_start_step (the UL trap forms first).
    geometry_start = int(method.get("geometry_start_step", 0) or 0)
    # Cap of the geometry gradient (lazy_mode reuse): ||beta grad log eta|| <= geometry_max_ratio * ||grad of the rest||.
    max_ratio = method.get("geometry_max_ratio")
    if max_ratio is not None and (not reuse or not math.isfinite(float(max_ratio)) or float(max_ratio) <= 0):
        raise ValueError("method.geometry_max_ratio must be positive and needs lazy_mode reuse")
    # Trajectory CBG: log eta_k is also optimized at theta' = theta - alpha * sign(grad L_harm(theta)), the point an
    # Adam-like attacker reaches first (each entry of geometry_trajectory is one alpha; 0 = theta itself). The
    # sign is piecewise constant, so d theta'/d theta = I and grad_theta log eta(theta') is exact at theta'.
    trajectory = [float(a) for a in method.get("geometry_trajectory", [0.])]
    if not trajectory or any(not math.isfinite(a) or a < 0 for a in trajectory):
        raise ValueError("method.geometry_trajectory must be a nonempty list of nonnegative step sizes")
    if any(trajectory) and not reuse:
        raise ValueError("method.geometry_trajectory needs lazy_mode reuse")
    cap_mode = method.get("geometry_cap_mode", "global")
    if cap_mode not in {"global", "per_tensor", "balance"}:
        raise ValueError("method.geometry_cap_mode must be global, per_tensor or balance")
    if cap_mode == "balance" and max_ratio is None:
        raise ValueError("geometry_cap_mode balance needs geometry_max_ratio (the per-tensor weight rho)")
    for step in range(start, steps):
        began = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        batch = to_device(training_batch(data, step, config), devices[0])
        joint = joint_batch(batch)
        active = step % lazy == 0
        phase = time.perf_counter()
        # reuse: the base gradient of every step is the plain one, so its reference is too.
        ref = reference_logits(reference, joint, make_chunks(joint, cfg.microbatch_tokens),
                               logprob_device=devices[-1],
                               precision="fp32" if active and not reuse else plain_precision) if cfg.alpha else None
        reference_time = time.perf_counter() - phase
        ratio_every = logging.get("grad_ratio_every") or 0
        row = dict(step=step + 1, lr=optimizer.param_groups[0]["lr"], time_reference=reference_time,
                   cbg_step=float(active), beta_effective=active_cfg.beta if active or reuse else 0.)
        if reuse:
            diagnostics = None
            if active and active_cfg.beta > 0 and step >= geometry_start:   # beta = 0: base loss only
                cache.clear()   # free the previous window before the CBG step's memory peak
                try:
                    signs = None
                    if any(trajectory):
                        signs, row["L_harm"] = attack_sign_direction(model, joint, plain_cfg)
                    point_cfg = replace(active_cfg, beta=active_cfg.beta / len(trajectory))
                    for alpha in trajectory:
                        if alpha:
                            shift_weights(signs, -alpha)
                        try:
                            point = cbg_step(model, joint, point_cfg, sample_generator=generators["y_hat"],
                                             sign_generator=generators["signs"], include_base=False)
                            if alpha:
                                row[f"L_harm_traj_{alpha:g}"] = harmful_sft_loss(model, joint, plain_cfg)
                                row[f"log_eta_k_traj_{alpha:g}"] = point.get("log_eta_k")
                        finally:
                            if alpha:
                                shift_weights(signs, alpha)
                        if diagnostics is None or not alpha:
                            diagnostics = point   # report the geometry at theta when it is measured
                        if not point.get("geometry_active"):
                            diagnostics = point
                            break
                    del signs
                    norm = grad_norm(model)
                except FloatingPointError as error:   # raised before any weight is perturbed, or after restoring
                    diagnostics, norm = dict(geometry_error=str(error)), float("nan")
                for key in ("L_aln", "L_bgn"):
                    diagnostics.pop(key, None)
                if diagnostics.get("geometry_active") and math.isfinite(norm):
                    row["geometry_clip_scale"] = cache.capture(model, norm)
                    row["grad_norm_beta_geometry"] = norm
                elif not diagnostics.get("geometry_skipped"):
                    cache.fail(diagnostics.get("geometry_error", "nonfinite geometry gradient"))
                    row["geometry_failed"] = 1.
                optimizer.zero_grad(set_to_none=True)
                row["time_cbg"] = time.perf_counter() - phase - reference_time
            if ul_weight:   # first, so that its own gradient norm can be logged
                ul_began = time.perf_counter()
                row.update(unlearning_gradient(model, joint, plain_cfg, weight=ul_weight))
                row["time_ul"] = time.perf_counter() - ul_began
                row["grad_norm_ul"] = grad_norm(model)
            plain_began = time.perf_counter()
            row.update(plain_gradient(model, joint, plain_cfg, reference=ref, chunks=None))
            row["time_plain"] = time.perf_counter() - plain_began
            rest = grad_norm(model)   # UL + L_aln + alpha L_bgn
            row["grad_norm_base"] = rest
            if active and "grad_norm_beta_geometry" in row:
                row["grad_ratio"] = row["grad_norm_beta_geometry"] / max(rest, 1e-30)
            if step < geometry_start:
                cache.clear()
            if max_ratio is not None and cap_mode in {"per_tensor", "balance"}:
                scale = cache.apply(per_tensor_ratio=float(max_ratio), balance=cap_mode == "balance")
            else:
                scale = cache.apply(max_norm=float(max_ratio) * rest if max_ratio is not None else None)
            row["geometry_applied"] = float(scale > 0)
            if scale:
                row["geometry_cap_scale"] = scale
        elif not active:
            stats = plain_gradient(model, joint, plain_cfg, reference=ref, chunks=None)
            row.update(stats, geometry_active=0., time_plain=time.perf_counter() - phase - reference_time)
            diagnostics = None
        elif ratio_every and (step // lazy) % ratio_every == 0:
            # .grad = -g_base, then + g_cbg  ->  ||beta * grad log eta|| ; then + g_base back.
            plain_gradient(model, joint, cfg, reference=ref)
            row["grad_norm_base"] = grad_norm(model)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.neg_()
            diagnostics = cbg_step(model, joint, active_cfg, reference=ref, sample_generator=generators["y_hat"],
                                   sign_generator=generators["signs"])
            row["grad_norm_beta_geometry"] = grad_norm(model)
            row["grad_ratio"] = row["grad_norm_beta_geometry"] / max(row["grad_norm_base"], 1e-30)
            row["diagnostic_passes"] = 1.
            plain_gradient(model, joint, cfg, reference=ref)
        else:
            diagnostics = cbg_step(model, joint, active_cfg, reference=ref, sample_generator=generators["y_hat"],
                                   sign_generator=generators["signs"])
        if diagnostics is not None:
            row.update(diagnostics)
        if ul_weight and not reuse:
            ul_began = time.perf_counter()
            row.update(unlearning_gradient(model, joint, plain_cfg, weight=ul_weight))
            row["time_ul"] = time.perf_counter() - ul_began
        row["grad_norm_total"] = grad_norm(model)
        if not math.isfinite(row["grad_norm_total"]):
            raise FloatingPointError(f"nonfinite gradient at step {step + 1}; optimizer not stepped")
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        row["time_step_total"] = time.perf_counter() - began
        for index, device in enumerate(devices):
            if device.type == "cuda":
                row[f"memory_reserved_gib_{index}"] = torch.cuda.max_memory_reserved(device) / 2**30
        if stage == "train" and logging["geo_eval_every"] and ((step + 1) % logging["geo_eval_every"] == 0
                                                               or step + 1 == steps):
            summary, _ = geometry_eval(model, geo, cfg, config, devices)
            row.update(summary)
        if "L_bgn" in row:
            recent_l_bgn = (recent_l_bgn + [row["L_bgn"]])[-utility_window:]
        gate = gates.get(step + 1) if stage == "train" else None
        failed = []
        if gate is not None:
            if "max_delta_log_eta" in gate:
                if baseline is None:
                    raise ValueError("early-stop gate without a step-0 geo/log_eta_k baseline in the training log")
                row["gate_delta_log_eta_k"] = row["geo/log_eta_k"] - baseline
                row["gate_max_delta_log_eta_k"] = gate["max_delta_log_eta"]
                if row["gate_delta_log_eta_k"] > gate["max_delta_log_eta"]:
                    failed.append(f"delta log eta_k {row['gate_delta_log_eta_k']:.4f} > {gate['max_delta_log_eta']}"
                                  " (held-out geometry did not decrease enough)")
            if "max_mean_l_bgn" in gate:
                row["gate_mean_l_bgn"] = statistics.mean(recent_l_bgn) if recent_l_bgn else 0.
                row["gate_max_mean_l_bgn"] = gate["max_mean_l_bgn"]
                if row["gate_mean_l_bgn"] > gate["max_mean_l_bgn"]:
                    failed.append(f"mean L_bgn {row['gate_mean_l_bgn']:.4f} > {gate['max_mean_l_bgn']}"
                                  " (model drifted too far from theta_0 on benign data)")
            row["gate_passed"] = float(not failed)
        log(row)
        if stage == "train" and step + 1 in save_models_at and step + 1 < steps:
            snapshot = output / f"model-step-{step + 1}"
            if not (snapshot / "config.json").exists():   # a resumed run keeps the first save
                model.save_pretrained(snapshot, safe_serialization=True)
                tokenizer.save_pretrained(snapshot)
        print(json.dumps({k: row[k] for k in row if k in _CONSOLE or k.startswith(("geo/log_eta", "gate_"))}),
              flush=True)
        if failed:
            result = dict(status="early_stopped", step=step + 1, reasons=failed,
                          **{k: row[k] for k in row if k.startswith("gate_")},
                          baseline_log_eta_k=baseline, log_eta_k=row.get("geo/log_eta_k"), resumed_from=start,
                          wall_seconds_this_segment=time.perf_counter() - wall)
            write_json(output / "metrics.json", result)
            if (output / "checkpoint").exists():
                shutil.rmtree(output / "checkpoint")   # ~81 GB; a stopped run is never resumed
            print(f"EARLY STOP at step {step + 1}: " + "; ".join(failed), flush=True)
            return result
        if stage == "train" and logging.get("checkpoint_every") and (step + 1) % logging["checkpoint_every"] == 0 \
                and step + 1 < steps:
            save_checkpoint(output, step + 1, model, optimizer, scheduler, generators,
                            extra=cache.state() if cache is not None else None)
    if stage == "smoke":
        return smoke_summary(log_path, config, output)
    final = output / "final"
    model.save_pretrained(final, safe_serialization=True)
    tokenizer.save_pretrained(final)
    write_json(output / "metrics.json", dict(status="complete", steps=steps, resumed_from=start,
                                             wall_seconds_this_segment=time.perf_counter() - wall))
    if (output / "checkpoint").exists() and not logging.get("keep_last_checkpoint", False):
        shutil.rmtree(output / "checkpoint")
    return dict(status="complete", final=str(final))


_CONSOLE = ("step", "log_eta_k", "L_aln", "L_bgn", "L_ul", "L_harm", "grad_norm_ul", "grad_norm_base", "geometry_cap_scale", "epsilon", "grad_norm_total", "grad_ratio",
            "geometry_clip_scale", "geometry_failed",
            "time_step_total", "time_perturbed", "geometry_skipped", "memory_reserved_gib_0",
            "memory_reserved_gib_1")


def _write_identity(path, identity, resume):
    # step_config is derived from the config by the code; it is recorded but not compared, so a run
    # started before a code change that only adds StepConfig fields (e.g. curvature) can still resume.
    comparable = lambda value: {k: v for k, v in value.items() if k != "step_config"}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if digest(comparable(existing)) != digest(comparable(json.loads(json.dumps(identity)))):
            raise ValueError("this output directory belongs to a different config/data/model; use a new run name")
        if not resume:
            raise ValueError("output directory already used; pass --resume to continue it")
        return
    write_json(path, identity)


def calibrate_stage(config, cfg, data, data_dir, attacked_path, devices, output, batches=4,
                    alphas=(1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4), sign_steps=(1, 3)):
    """Calibrate the trajectory step against a real attack (no training).

    Reports, on the first ``batches`` training batches, the attacker's loss L_harm at the base model, at the
    checkpoint after a real attack (``attacked_path``), and at theta - alpha * sign(grad) for each alpha with
    1 or several sign steps (gradient recomputed per step, each of size alpha / steps); plus the real per-
    coordinate displacement |theta_attacked - theta_base|.
    """
    from src.experiments.prepare_v2 import model_path as pinned_model
    plain = replace(cfg, perturb_precision=config["execution"].get("plain_precision", "fp32"), deterministic=False)
    joints = [joint_batch(to_device(training_batch(data, step, config), devices[0])) for step in range(batches)]
    attacked = load_split_model(attacked_path, devices, trainable=False)
    target = [harmful_sft_loss(attacked, joint, plain) for joint in joints]
    base = load_split_model(pinned_model(data_dir), devices, trainable=True)
    start = [harmful_sft_loss(base, joint, plain) for joint in joints]
    moved = dict(abs_sum=0., count=0, above_1e6=0, above_1e5=0, max=0.)
    with torch.no_grad():
        for (name, p), q in zip(base.named_parameters(), attacked.parameters()):
            delta = (q.detach().to(p.device, torch.float32) - p.detach()).abs()
            moved["abs_sum"] += float(delta.double().sum())
            moved["count"] += delta.numel()
            moved["above_1e6"] += int((delta > 1e-6).sum())
            moved["above_1e5"] += int((delta > 1e-5).sum())
            moved["max"] = max(moved["max"], float(delta.max()))
    del attacked
    grid = []
    for steps in sign_steps:
        for alpha in alphas:
            losses = []
            for joint in joints:
                applied = []
                try:
                    for _ in range(steps):
                        signs, _ = attack_sign_direction(base, joint, plain)
                        shift_weights(signs, -alpha / steps)
                        applied.append(signs)
                    losses.append(harmful_sft_loss(base, joint, plain))
                finally:
                    for signs in reversed(applied):
                        shift_weights(signs, alpha / steps)
            grid.append(dict(sign_steps=steps, alpha=alpha, L_harm=statistics.mean(losses)))
            print(json.dumps(grid[-1]), flush=True)
    result = dict(attacked=str(attacked_path), batches=batches, L_harm_base=statistics.mean(start),
                  L_harm_attacked=statistics.mean(target), mean_abs_displacement=moved["abs_sum"] / moved["count"],
                  fraction_moved_above_1e6=moved["above_1e6"] / moved["count"],
                  fraction_moved_above_1e5=moved["above_1e5"] / moved["count"], max_displacement=moved["max"],
                  grid=grid)
    write_json(output / "calibration.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "grid"}), flush=True)
    return result


def smoke_summary(log_path, config, output):
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    steady = rows[1:] or rows   # the first step includes allocator warm-up
    keys = sorted({k for r in steady for k in r if k.startswith("time_")})
    memory = sorted({k for r in rows for k in r if k.startswith(("peak_memory", "memory_reserved"))})
    last_cbg = next((r for r in reversed(rows) if r.get("cbg_step") == 1.), steady[-1])
    summary = dict(steps=len(rows), mean_times={k: statistics.mean(r[k] for r in steady if k in r) for k in keys},
                   peak_memory_gib={k: max(r[k] for r in rows if k in r) for k in memory},
                   token_slots=last_cbg.get("token_slots"), microbatches=last_cbg.get("microbatches"))
    lazy = config["method"].get("lazy_period", 1)
    cbg_rows = [r for r in rows if r.get("cbg_step") == 1. and "diagnostic_passes" not in r] or \
               [r for r in rows if r.get("cbg_step") == 1.]
    cheap_rows = [r for r in rows[1:] if r.get("cbg_step") == 0.]
    cbg_time = statistics.mean(r["time_step_total"] for r in cbg_rows)
    cheap_time = statistics.mean(r["time_step_total"] for r in cheap_rows) if cheap_rows else cbg_time
    per_step = (cbg_time + (lazy - 1) * cheap_time) / lazy
    steps = config["optimizer"]["steps"]
    summary.update(seconds_cbg_step=cbg_time, seconds_plain_step=cheap_time if cheap_rows else None,
                   seconds_per_step_average=per_step, lazy_period=lazy)
    summary["projected_hours_without_diagnostics"] = per_step * steps / 3600
    summary["note"] = ("grad-ratio steps cost two extra FP32 passes; geometry evaluations and checkpoints "
                       "are not included in the projection")
    write_json(output / "smoke_summary.json", summary)
    return summary


def _compare(params, stored):
    diff = dot = norm = ref_norm = 0.
    for index, p in enumerate(params):
        ref = stored[index].to(p.grad.device)
        diff += float(torch.linalg.vector_norm(p.grad - ref, dtype=torch.float64)) ** 2
        dot += float((p.grad.double() * ref.double()).sum())
        norm += float(torch.linalg.vector_norm(p.grad, dtype=torch.float64)) ** 2
        ref_norm += float(torch.linalg.vector_norm(ref, dtype=torch.float64)) ** 2
    return diff ** .5 / max(ref_norm ** .5, 1e-300), dot / max((norm * ref_norm) ** .5, 1e-300), norm ** .5


RADIUS_TOLERANCE, RADIUS_MIN_COSINE = 0.05, 0.99


def radius_grid(r, low=-8, high=4):
    """Geometric grid r * 3^(k/2), k = low..high (two points per factor of 3), ascending."""
    return [r * 3 ** (k / 2) for k in range(low, high + 1)]


def radius_stage(model, data, config, cfg, devices, generators, output):
    """Scan the finite-difference radius and recommend one on a plateau.

    One center, y_hat and sign draw are shared by every radius. Radii r * 3^(k/2)
    are evaluated in ascending order (r/81 .. 9r); each estimate is compared with
    the previous one (factor sqrt 3) and the one before (factor 3). r_k is on a
    plateau when its estimates at r_k/3 and 3 r_k agree with it within 5%
    (relative L2) and cosine >= 0.99. r itself is evaluated twice first: any
    difference there is nondeterminism, which no radius can fix.
    """
    r = cfg.r
    grid = radius_grid(r)
    radii = [r, r] + grid
    batch = to_device(training_batch(data, 0, config), devices[0])
    params = [p for p in model.parameters() if p.requires_grad]
    rows, previous, position = [], [], [0]   # previous: last two estimates on CPU (fp32)

    def snapshot():
        return {i: p.grad.detach().to("cpu", copy=True) for i, p in enumerate(params)}

    def sink(radius, epsilon):
        index = position[0]
        position[0] += 1
        if index == 0:
            previous[:] = [snapshot()]
            rows.append(dict(label="r", radius=radius, epsilon=epsilon, norm=grad_norm(model)))
        elif index == 1:
            difference, cosine, norm = _compare(params, previous[0])
            rows.append(dict(label="r_repeat", radius=radius, epsilon=epsilon, norm=norm,
                             relative_difference_to_r=difference, cosine_to_r=cosine))
            previous.clear()
        else:
            k = index - 2
            row = dict(label=f"r*3^({(k - 8) / 2:g})", radius=radius, epsilon=epsilon, norm=grad_norm(model))
            if len(previous) >= 1:
                row["relative_difference_to_sqrt3_smaller"], row["cosine_to_sqrt3_smaller"], _ = \
                    _compare(params, previous[-1])
            if len(previous) == 2:
                row["relative_difference_to_3x_smaller"], row["cosine_to_3x_smaller"], _ = \
                    _compare(params, previous[0])
                del previous[0]   # drop before the next snapshot: at most two fp32 copies in host RAM
            previous.append(snapshot())
            rows.append(row)
        print(json.dumps(rows[-1]), flush=True)

    geometry = curvature_gradient(model, batch, cfg, radii=radii, sample_generator=generators["y_hat"],
                                  sign_generator=generators["signs"], sink=sink)
    for p in params:
        p.grad = None
    previous.clear()
    result = summarize_radius_scan(rows, r, perturb_precision=cfg.perturb_precision,
                                   log_eta_k=float(geometry["log_eta_k"]))
    print(f"[radius] {result['diagnosis']}", flush=True)
    write_json(output / "radius_check.json", result)
    return result


HVP_TOLERANCE, HVP_MIN_COSINE = 0.05, 0.99


def hvp_check_stage(model, data, config, cfg, devices, generators, output):
    """Exact HVP against finite differences on one 7B batch (same center, y_hat, signs).

    Estimates of the curvature term 2 H_omega w:
      hvp fp32         exact reference (double backward, true fp32)
      hvp <precision>  what training would use (execution.hvp_precision)
      fd r/sqrt3, r, sqrt3 r   what the current runs use (execution.perturb_precision)
    Pass: the training HVP agrees with the fp32 HVP, and FD at r agrees with an HVP,
    each within 5% (relative L2) and cosine >= 0.99. Seconds and peak memory per
    estimator are recorded to compare the cost of one HVP pass with two FD passes.
    """
    execution = config["execution"]
    r = config["method"]["radius"]
    fd_precision = execution["perturb_precision"]
    hvp_precision = execution.get("hvp_precision", "bf16")
    check_cfg = replace(cfg, perturb_precision=fd_precision)
    # Every HVP precision is measured against fp32, so one run also tells which cheaper one is usable.
    precisions = ["fp32"] + [p for p in ("tf32", "bf16") if p != "fp32"]
    plan = [("hvp", p) for p in precisions] + [("fd", r / math.sqrt(3)), ("fd", r), ("fd", r * math.sqrt(3))]
    batch = to_device(training_batch(data, 0, config), devices[0])
    params = [p for p in model.parameters() if p.requires_grad]
    stored, rows = {}, []

    def label(kind, value):
        return f"hvp_{value}" if kind == "hvp" else f"fd_r*{value / r:.3g}"

    def sink(kind, value, seconds, epsilon):
        name = label(kind, value)
        row = dict(label=name, kind=kind, seconds=seconds)
        if kind == "fd":
            row.update(radius=value, epsilon=epsilon, precision=fd_precision)
        else:
            row.update(precision=value, microbatch_tokens=cfg.hvp_microbatch_tokens or cfg.microbatch_tokens)
        for index, device in enumerate(devices):
            if device.type == "cuda":
                row[f"peak_memory_gib_{index}"] = torch.cuda.max_memory_allocated(device) / 2**30
        if seconds is None:
            row["status"] = "out_of_memory"
        else:
            row["status"], row["norm"] = "ok", grad_norm(model)
            for reference in ("hvp_fp32", f"hvp_{hvp_precision}"):
                if reference in stored and reference != name:
                    row[f"relative_difference_to_{reference}"], row[f"cosine_to_{reference}"], _ = \
                        _compare(params, stored[reference])
            # Only the two references later rows compare with are kept (2 x 27 GB host RAM on 7B).
            if name in ("hvp_fp32", f"hvp_{hvp_precision}"):
                stored[name] = {i: p.grad.detach().to("cpu", copy=True) for i, p in enumerate(params)}
        rows.append(row)
        print(json.dumps(row), flush=True)

    geometry = curvature_estimates(model, batch, check_cfg, plan=plan, sample_generator=generators["y_hat"],
                                   sign_generator=generators["signs"], sink=sink, skip_oom=True)
    for p in params:
        p.grad = None
    stored.clear()
    result = summarize_hvp_check(rows, r=r, hvp_precision=hvp_precision,
                                 tolerance=execution.get("hvp_check_tolerance", HVP_TOLERANCE),
                                 min_cosine=execution.get("hvp_check_min_cosine", HVP_MIN_COSINE))
    result.update(log_eta_k=float(geometry["log_eta_k"]), fd_precision=fd_precision)
    print(f"[hvp-check] {result['diagnosis']}", flush=True)
    write_json(output / "hvp_check.json", result)
    return result


def summarize_hvp_check(rows, *, r, hvp_precision, tolerance=HVP_TOLERANCE, min_cosine=HVP_MIN_COSINE):
    """Verdict of hvp-check. ``tolerance``/``min_cosine`` (execution.hvp_check_*) are recorded in the result."""
    by = {row["label"]: row for row in rows}
    train_hvp, exact, fd = by.get(f"hvp_{hvp_precision}"), by.get("hvp_fp32"), by.get("fd_r*1")
    ok = lambda row: row is not None and row["status"] == "ok"
    def agrees(row, reference):
        key = f"relative_difference_to_{reference}"
        return ok(row) and key in row and row[key] <= tolerance and row[f"cosine_to_{reference}"] >= min_cosine
    # Usable precisions: fp32 itself (when FD at r confirms it) and any that agrees with fp32.
    usable = {}
    if ok(exact) and agrees(fd, "hvp_fp32"):
        for row in rows:
            if row.get("kind") == "hvp" and ok(row) and (row["precision"] == "fp32" or agrees(row, "hvp_fp32")):
                usable[row["precision"]] = row["seconds"]
    recommended = min(usable, key=usable.get) if usable else None
    checks = {}
    if hvp_precision != "fp32":
        checks["hvp_precision_vs_fp32"] = agrees(train_hvp, "hvp_fp32") if ok(exact) else None
    # FD at r against the exact HVP when it fit in memory, otherwise against the training HVP.
    reference = "hvp_fp32" if ok(exact) else f"hvp_{hvp_precision}"
    checks["fd_r_vs_hvp"] = agrees(fd, reference)
    passed = ok(train_hvp) and all(value is not False for value in checks.values()) and checks["fd_r_vs_hvp"]
    fd_seconds = fd["seconds"] if ok(fd) else None
    speedup = fd_seconds / train_hvp["seconds"] if ok(train_hvp) and fd_seconds else None
    if not ok(train_hvp):
        diagnosis = (f"the {hvp_precision} HVP ran out of GPU memory; lower execution.hvp_microbatch_tokens")
    elif passed:
        timing = ""
        if speedup:
            timing = (f"; curvature pass {speedup:.2f}x faster than FD" if speedup >= 1 else
                      f"; curvature pass {1 / speedup:.2f}x SLOWER than FD (keep fd for speed)")
        diagnosis = (f"HVP ({hvp_precision}) agrees with FD at r = {r:g}"
                     + (" and with the fp32 HVP" if checks.get("hvp_precision_vs_fp32") else "") + timing)
    else:
        failed = [name for name, value in checks.items() if value is False]
        diagnosis = f"HVP check FAILED: {', '.join(failed)}"
        if recommended and recommended != hvp_precision:
            pace = ""
            if fd_seconds:
                ratio = fd_seconds / usable[recommended]
                pace = f", {ratio:.2f}x faster than FD" if ratio >= 1 else f", {1 / ratio:.2f}x SLOWER than FD"
            diagnosis += (f"; the HVP is correct in {recommended} ({usable[recommended]:.1f} s{pace}): "
                          f"set execution.hvp_precision: {recommended}")
        elif not usable:
            diagnosis += "; no HVP precision agrees with FD: keep curvature: fd"
    return dict(rows=rows, checks=checks, passed=bool(passed), radius=r, hvp_precision=hvp_precision,
                usable_hvp_precisions=usable, recommended_hvp_precision=recommended,
                fd_seconds=fd_seconds, hvp_seconds=train_hvp["seconds"] if ok(train_hvp) else None,
                curvature_pass_speedup=speedup, diagnosis=diagnosis, tolerance=tolerance, min_cosine=min_cosine,
                criterion=f"relative L2 <= {tolerance:.0%} and cosine >= {min_cosine}: "
                          f"hvp_{hvp_precision} vs hvp_fp32, and fd at r vs the HVP")


def summarize_radius_scan(rows, r, *, perturb_precision, log_eta_k):
    """Plateau candidates and the recommended radius from the scan rows of ``radius_stage``."""
    scan = rows[2:]
    grid = [row["radius"] for row in scan]
    candidates = []
    for k in range(2, len(scan) - 2):   # needs r_k/3 (k-2) and 3 r_k (k+2)
        below, above = scan[k], scan[k + 2]   # row k vs k-2, row k+2 vs k
        worst = max(below["relative_difference_to_3x_smaller"], above["relative_difference_to_3x_smaller"])
        cosine = min(below["cosine_to_3x_smaller"], above["cosine_to_3x_smaller"])
        # Narrow plateau: r_k / sqrt 3 and sqrt 3 r_k (row k vs k-1, row k+1 vs k).
        narrow = max(scan[k]["relative_difference_to_sqrt3_smaller"],
                     scan[k + 1]["relative_difference_to_sqrt3_smaller"])
        narrow_cosine = min(scan[k]["cosine_to_sqrt3_smaller"], scan[k + 1]["cosine_to_sqrt3_smaller"])
        candidates.append(dict(radius=scan[k]["radius"], worst_relative_difference=worst, worst_cosine=cosine,
                               plateau=worst <= RADIUS_TOLERANCE and cosine >= RADIUS_MIN_COSINE,
                               worst_relative_difference_narrow=narrow, worst_cosine_narrow=narrow_cosine,
                               plateau_narrow=narrow <= RADIUS_TOLERANCE and narrow_cosine >= RADIUS_MIN_COSINE))
    wide = [c for c in candidates if c["plateau"]]
    if wide:
        best, tier = min(wide, key=lambda c: c["worst_relative_difference"]), "x3"
    else:
        best = min(candidates, key=lambda c: c["worst_relative_difference_narrow"])
        tier = "x1.73" if best["plateau_narrow"] else None
    at_r = next(c for c in candidates if math.isclose(c["radius"], r, rel_tol=1e-9))
    repeat = rows[1]["relative_difference_to_r"]
    if repeat > 1e-3:
        diagnosis = (f"NONDETERMINISTIC: the same radius twice differs by {repeat:.3g} (relative L2); "
                     "no radius can fix this")
    elif at_r["plateau"]:
        diagnosis = f"plateau: r = {r:g} is usable"
    elif at_r["plateau_narrow"] and not wide:
        diagnosis = (f"narrow plateau: r = {r:g} is usable (r/sqrt3 and sqrt3 r agree within "
                     f"{RADIUS_TOLERANCE:.0%}; the plateau is narrower than r/3 .. 3r)")
    elif tier:
        diagnosis = (f"r = {r:g} is not on a plateau; set method.radius to {best['radius']:.3g} "
                     f"({tier} plateau)")
    else:
        diagnosis = (f"no radius in [{grid[0]:.3g}, {grid[-1]:.3g}] meets either criterion; the most consistent "
                     f"is {best['radius']:.3g} (worst neighbour difference "
                     f"{best['worst_relative_difference_narrow']:.3f})")
    return dict(radius=r, perturb_precision=perturb_precision, rows=rows, candidates=candidates,
                  log_eta_k=log_eta_k, plateau=at_r["plateau"],
                  plateau_narrow=at_r["plateau_narrow"] and not wide,
                  repeat_relative_difference=repeat, recommended_radius=best["radius"],
                  recommended_on_plateau=tier is not None, recommended_tier=tier, diagnosis=diagnosis,
                  criterion=f"estimates at r/3 and 3r (or, for a narrow plateau, r/sqrt3 and sqrt3 r) within "
                            f"{RADIUS_TOLERANCE:.0%} (relative L2) and cosine >= {RADIUS_MIN_COSINE} of the one at r")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for key in ("config", "data", "output", "model"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--stage", choices=("train", "smoke", "radius", "hvp-check", "geometry", "calibrate"),
                        default="train")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=6)
    args = parser.parse_args(argv)
    result = run(args.config, args.data, args.output, args.model, stage=args.stage, resume=args.resume,
                 smoke_steps=args.smoke_steps)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
