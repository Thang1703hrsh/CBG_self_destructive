"""Hugging Face Trainer integration for Collateral Barrier Geometry.

This module deliberately does not share the custom ``SeamTrainer`` inner
loop. CBG computes its gradient explicitly in the configured parameter space,
then hands that gradient back to the stock Transformers optimizer loop
through a zero-valued differentiable proxy.  Consequently optimizer stepping,
gradient accumulation, clipping, AMP scaling, scheduler stepping, and ordinary
Trainer checkpoint bookkeeping remain owned by :class:`transformers.Trainer`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from transformers import Trainer, TrainerCallback

from src.cbg.functionals import (
    benign_preservation_loss,
    make_benign_functional,
    make_safety_functional,
    safe_alignment_loss,
    select_response_logits,
    sequence_chunks,
)
from src.cbg.rayleigh import calibrate_relative_damping, warm_started_rayleigh_ascent
from src.cbg.dual import dual_rayleigh_solve
from src.cbg.bounded import bounded_rayleigh_solve
from src.cbg.spectral_grad import (
    centered_spectral_gradient,
    direct_spectral_gradient,
    compact_spectral_center,
    parameter_gradient,
    streamed_loss_gradient,
)
from src.cbg.tree_ops import (
    OrderedTensorTree,
    TensorTree,
    accumulate_detached_gradient,
    select_admissible_parameters,
    select_lora_parameters,
    tree_clone,
    tree_detach,
    tree_norm,
    tree_dot,
    tree_add,
    tree_sub,
)


logger = logging.getLogger(__name__)

CBG_STATE_NAME = "cbg_state.pt"
CBG_CONFIG_NAME = "cbg_config.json"
CBG_STATE_VERSION = 4

_NON_RESUME_CONFIG_FIELDS = {
    "cbg_resume_from_checkpoint",
    "wandb_project_name",
    "wandb_run_name",
}

_SPEC_CONFIG_FIELDS = {
    "safety_batch_size", "benign_batch_size", "safety_max_length", "benign_max_length",
    "spectral_modes", "xi", "kappa_mu", "relative_radius", "gram_max_bytes",
    "transformer_block_path", "kl_direction", "spec_damping", "fisher", "detach_lambda",
    "num_sign_draws", "spec_objective", "drop_aln", "scenario_manifest",
    "spec_solver", "spec_rayleigh_steps", "spec_rayleigh_step_size", "spec_perturb_precision",
}


@dataclass
class CbgArguments:
    """Configuration specific to the CBG defense.

    Training-loop settings such as the optimizer, batch size, precision,
    scheduler, and checkpoint cadence remain ordinary ``TrainingArguments``.
    """

    method: str = "cbg"
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    model_revision: str | None = None
    tokenizer_name: str | None = None
    wandb_project_name: str = "cbg"
    wandb_run_name: str | None = None

    # Reference behavior and data.
    reference_mode: str = "disabled_adapter"
    beavertails_path: str = "data/beavertails_with_refusals_train.json"
    defense_size: int = 4000
    attack_size: int = 1000
    split_seed: int = 0
    data_shuffle: bool = False
    context_length: int = 256

    # Admissible LoRA coordinates.
    parameter_space: str = "lora"
    reference_device: str | None = None
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_bias: str = "none"
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    # Internal chunks preserve the complete batch's geometry and mean losses.
    functional_chunk_size: int = 0
    preservation_chunk_size: int = 0
    cbg_attention_implementation: str = "eager"
    # Defaults retain the previous geometry and solver for reproducibility.
    benign_token_blocks: int = 1
    geometry_solver: str = "ascent"
    # Legacy defaults keep archived YAMLs reproducible. The V5 configuration
    # explicitly selects v5 + direct_rof + dual (or bounded_ritz).
    cbg_version: str = "v2"
    # Authoritative implementation-spec path; v2/v5 remain reproducible.
    safety_batch_size: int = 8
    benign_batch_size: int = 32
    safety_max_length: int = 513
    benign_max_length: int = 128
    spectral_modes: int = 4
    xi: float | None = None
    kappa_mu: float | None = None
    relative_radius: float | None = None
    gram_max_bytes: int = 67108864
    transformer_block_path: str = "model.layers"
    kl_direction: str = "reference_to_current"
    spec_damping: str = "relative"
    fisher: str = "sampled"
    detach_lambda: bool = False
    num_sign_draws: int = 1
    spec_objective: str = "log_eta"
    drop_aln: bool = False
    scenario_manifest: str | None = None
    spec_solver: str = "exact"
    spec_rayleigh_steps: int = 0
    spec_rayleigh_step_size: float | None = None
    spec_perturb_precision: str = "fp32"
    spectral_derivative_backend: str = "centered_fd"
    # Research-only escape hatch for a guarded full-parameter feasibility
    # probe. Production V5 remains FP32 unless this is explicitly enabled.
    experimental_bf16_geometry: bool = False
    ritz_initial_size: int = 2
    ritz_max_size: int = 6
    ritz_max_enrichments: int = 4
    ritz_residual_tolerance: float = 1e-4
    ritz_orthogonality_tolerance: float = 1e-5
    enclosure_atol: float = 1e-6
    enclosure_rtol: float = 1e-3
    enclosure_rho_scale: float = 1.0
    dual_max_size: int = 64
    dual_tolerance: float = 1e-4
    # Kept separate from value/kernel tolerances because full parameter-space
    # VJP/JVP residuals have a model- and precision-dependent numerical floor.
    dual_residual_tolerance: float = 1e-4
    dual_kernel_tolerance: float = 1e-5
    dual_column_batch_size: int = 1
    dual_refinement_steps: int = 0
    dual_column_fallback: bool = False
    # A rare, slower retry for FP32 verification outliers. It rebuilds the
    # identical finite-batch maps with scalar functional chunks and evaluates
    # sequential kernel columns; it never relaxes a gate or changes mu/data.
    dual_functional_chunk_fallback: bool = False

    # Weakest-escape search and damping.
    inner_steps: int = 2
    inner_lr: float = 1e-2
    inner_policy: str = "fixed"
    inner_initial_steps: int = 8
    inner_max_steps: int = 20
    inner_max_backtracks: int = 4
    inner_residual_tolerance: float = 0.05
    inner_rayleigh_tolerance: float = 0.01
    inner_fail_on_nonconvergence: bool = False
    damping_mode: str = "relative"
    mu: float = 1e-4
    damping_fraction: float = 0.05
    damping_calibration_directions: int = 8
    mu_floor: float = 1e-8
    eps_norm: float = 1e-12

    # Centered spectral finite difference.
    tau_fd: float = 1e-4
    theta_floor: float = 1e-3
    u_floor: float = 1e-12

    # Final training gradient.
    alpha: float = 1.0
    beta: float = 1e-3
    beta_mode: str = "fixed"
    beta_target_ratio: float = 0.25
    beta_calibration_steps: int = 50
    beta_min: float = 0.0
    beta_max: float = 1.0

    cbg_resume_from_checkpoint: str | None = None

    def __post_init__(self) -> None:
        # Use a list annotation so HfArgumentParser supports multiple CLI values.
        if isinstance(self.lora_target_modules, str):
            raise ValueError("lora_target_modules must be a sequence, not a string")
        self.lora_target_modules = list(self.lora_target_modules)

        if self.method != "cbg":
            raise ValueError("CbgArguments.method must be 'cbg'")
        if self.reference_mode not in {"disabled_adapter", "separate_model"}:
            raise ValueError(
                "reference_mode must be 'disabled_adapter' or 'separate_model'"
            )
        if self.parameter_space not in {"lora", "full"}:
            raise ValueError("parameter_space must be 'lora' or 'full'")
        if self.parameter_space == "full" and self.reference_mode != "separate_model":
            raise ValueError("full-parameter CBG requires reference_mode='separate_model'")
        if not isinstance(self.experimental_bf16_geometry, bool):
            raise ValueError("experimental_bf16_geometry must be boolean")
        if self.damping_mode not in {"absolute", "relative"}:
            raise ValueError("damping_mode must be 'absolute' or 'relative'")
        if self.beta_mode not in {"fixed", "grad_ratio"}:
            raise ValueError("beta_mode must be 'fixed' or 'grad_ratio'")
        if self.inner_policy not in {"fixed", "adaptive"}:
            raise ValueError("inner_policy must be 'fixed' or 'adaptive'")
        if self.geometry_solver not in {"ascent", "dual", "bounded_ritz", "exact"}:
            raise ValueError("geometry_solver must be ascent, dual, bounded_ritz or exact")
        if self.experimental_bf16_geometry and (
            self.parameter_space != "full" or self.geometry_solver != "dual"
        ):
            raise ValueError(
                "experimental_bf16_geometry is restricted to full-parameter dual probes"
            )
        if self.cbg_version not in {"v2", "v5", "spec"}:
            raise ValueError("cbg_version must be v2, v5 or spec")
        if self.scenario_manifest and self.cbg_version != "spec":
            raise ValueError("scenario_manifest requires cbg_version=spec")
        if self.cbg_version == "spec":
            if (self.parameter_space != "full" or self.geometry_solver != "exact"
                or self.spectral_derivative_backend != "centered_fd" or self.beta_mode != "fixed"
                or self.experimental_bf16_geometry or self.damping_mode != "relative"):
                raise ValueError("spec CBG requires full parameters, exact solve, centered_fd, fixed beta and relative mu")
            for name in ("safety_batch_size", "benign_batch_size"):
                _require_int_at_least(getattr(self, name), 1, name)
            for name in ("safety_max_length", "benign_max_length"):
                _require_int_at_least(getattr(self, name), 2, name)
            if self.spectral_modes > self.safety_batch_size:
                raise ValueError("spectral_modes cannot exceed safety_batch_size")
            self.spec_config()
        elif self.geometry_solver == "exact":
            raise ValueError("geometry_solver=exact belongs to cbg_version=spec")
        if self.spectral_derivative_backend not in {"centered_fd", "direct_rof"}:
            raise ValueError("spectral_derivative_backend must be centered_fd or direct_rof")
        if self.cbg_version == "v5" and (
            self.spectral_derivative_backend != "direct_rof" or self.geometry_solver == "ascent"
        ):
            raise ValueError("V5 requires direct_rof and a verified dual or bounded_ritz solve")
        _require_int_at_least(self.ritz_initial_size, 1, "ritz_initial_size")
        _require_int_at_least(self.ritz_max_size, max(2, self.ritz_initial_size), "ritz_max_size")
        _require_int_at_least(self.ritz_max_enrichments, 0, "ritz_max_enrichments")
        for name in ("ritz_residual_tolerance", "ritz_orthogonality_tolerance", "enclosure_rho_scale"):
            _require_positive(getattr(self, name), name)
        for name in ("enclosure_atol", "enclosure_rtol"):
            _require_nonnegative(getattr(self, name), name)
        _require_int_at_least(self.benign_token_blocks, 1, "benign_token_blocks")
        _require_int_at_least(self.dual_max_size, 1, "dual_max_size")
        _require_int_at_least(self.dual_column_batch_size, 1, "dual_column_batch_size")
        _require_int_at_least(self.dual_refinement_steps, 0, "dual_refinement_steps")
        if not isinstance(self.dual_column_fallback, bool):
            raise ValueError("dual_column_fallback must be boolean")
        if not isinstance(self.dual_functional_chunk_fallback, bool):
            raise ValueError("dual_functional_chunk_fallback must be boolean")
        _require_positive(self.dual_tolerance, "dual_tolerance")
        _require_positive(self.dual_residual_tolerance, "dual_residual_tolerance")
        _require_positive(self.dual_kernel_tolerance, "dual_kernel_tolerance")
        if self.cbg_attention_implementation != "eager":
            raise ValueError("the validated CBG attention baseline is 'eager'; validate JVP kernels before changing it")
        if self.lora_bias != "none":
            raise ValueError("the initial CBG admissible space requires lora_bias='none'")
        if not self.lora_target_modules or any(
            not isinstance(name, str) or not name for name in self.lora_target_modules
        ):
            raise ValueError("lora_target_modules must contain non-empty module names")

        _require_int_at_least(self.defense_size, 1, "defense_size")
        _require_int_at_least(self.attack_size, 0, "attack_size")
        _require_int_at_least(self.context_length, 2, "context_length")
        _require_int_at_least(self.lora_rank, 1, "lora_rank")
        _require_int_at_least(self.lora_alpha, 1, "lora_alpha")
        _require_int_at_least(self.inner_steps, 1, "inner_steps")
        for name in ("functional_chunk_size", "preservation_chunk_size", "inner_max_backtracks"):
            _require_int_at_least(getattr(self, name), 0, name)
        _require_int_at_least(self.inner_initial_steps, 1, "inner_initial_steps")
        _require_int_at_least(self.inner_max_steps, 1, "inner_max_steps")
        if self.inner_policy == "adaptive" and self.inner_max_steps < max(self.inner_steps, self.inner_initial_steps):
            raise ValueError("inner_max_steps must cover both initial and steady minimum steps")
        _require_positive(self.inner_residual_tolerance, "inner_residual_tolerance")
        _require_positive(self.inner_rayleigh_tolerance, "inner_rayleigh_tolerance")
        _require_int_at_least(
            self.damping_calibration_directions,
            1,
            "damping_calibration_directions",
        )
        _require_int_at_least(self.beta_calibration_steps, 0, "beta_calibration_steps")

        _require_nonnegative(self.lora_dropout, "lora_dropout")
        if self.lora_dropout != 0:
            raise ValueError(
                "the first CBG implementation requires lora_dropout=0 for deterministic geometry"
            )
        _require_positive(self.inner_lr, "inner_lr")
        _require_positive(self.mu, "mu")
        _require_positive(self.damping_fraction, "damping_fraction")
        _require_positive(self.mu_floor, "mu_floor")
        _require_nonnegative(self.eps_norm, "eps_norm")
        _require_positive(self.tau_fd, "tau_fd")
        _require_positive(self.theta_floor, "theta_floor")
        _require_positive(self.u_floor, "u_floor")
        _require_nonnegative(self.alpha, "alpha")
        _require_nonnegative(self.beta, "beta")
        _require_nonnegative(self.beta_min, "beta_min")
        _require_positive(self.beta_max, "beta_max")
        if self.beta_min > self.beta_max:
            raise ValueError("beta_min must be <= beta_max")
        if self.beta_mode == "grad_ratio" and not self.beta_min <= self.beta <= self.beta_max:
            raise ValueError("gradient-ratio fallback beta must lie within beta_min and beta_max")
        _require_positive(self.beta_target_ratio, "beta_target_ratio")
        if self.beta_mode == "grad_ratio" and self.beta_calibration_steps < 1:
            raise ValueError(
                "beta_calibration_steps must be at least one for beta_mode='grad_ratio'"
            )


    def spec_config(self):
        from src.cbg.spec import SpecConfig
        return SpecConfig(xi=self.xi, kappa_mu=self.kappa_mu, r=self.relative_radius,
                          alpha=self.alpha, beta=self.beta, k=self.spectral_modes,
                          gram_max_bytes=self.gram_max_bytes, block_path=self.transformer_block_path,
                          kl_direction=self.kl_direction, damping=self.spec_damping,
                          fisher=self.fisher, detach_lambda=self.detach_lambda,
                          num_sign_draws=self.num_sign_draws, objective=self.spec_objective,
                          drop_aln=self.drop_aln, solver=self.spec_solver,
                          rayleigh_steps=self.spec_rayleigh_steps, rayleigh_step_size=self.spec_rayleigh_step_size,
                          perturb_precision=self.spec_perturb_precision)


def _require_int_at_least(value: int, minimum: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _require_positive(value: float, name: str) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{name} must be finite and strictly positive")


def _require_nonnegative(value: float, name: str) -> None:
    if not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"{name} must be finite and non-negative")


def _extract_logits(outputs: Any) -> Tensor:
    if isinstance(outputs, Tensor):
        return outputs
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, (tuple, list)) and outputs:
        return outputs[0]
    raise TypeError("reference model output does not expose logits")


@contextmanager
def _temporary_eval(module: nn.Module):
    """Evaluate a module and restore every submodule's exact training mode."""

    modes = [(submodule, submodule.training) for submodule in module.modules()]
    module.eval()
    try:
        yield
    finally:
        for submodule, was_training in modes:
            submodule.training = was_training


def _canonical_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-stable configuration, excluding run-local fields."""

    comparable = {
        key: value for key, value in config.items() if key not in _NON_RESUME_CONFIG_FIELDS
    }
    if comparable.get("model_revision") is None:
        comparable.pop("model_revision", None)
    if comparable.get("cbg_version", "v2") != "spec":
        # New spec-only fields have no effect on legacy experiments. Preserve
        # fingerprints of their existing version-4 checkpoints.
        comparable = {key: value for key, value in comparable.items() if key not in _SPEC_CONFIG_FIELDS}
    # The round trip canonicalizes tuples to lists and rejects values that
    # cannot be represented in the human-readable checkpoint configuration.
    return json.loads(json.dumps(comparable, sort_keys=True))


def _config_fingerprint(config: Mapping[str, Any]) -> str:
    payload = json.dumps(_canonical_config(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class _OptimizerStepMetrics(TrainerCallback):
    """Measure the actual adapter update around Trainer's optimizer step.

    One temporary detached parameter tree is retained only across that step.
    These are actual updates, not an attribution of Adam's nonlinear scaling.
    """
    def __init__(self, trainer):
        self.trainer = trainer
        self.before = None
        self.measure_update = trainer.cbg_args.parameter_space != "full"

    def on_pre_optimizer_step(self, args, state, control, model=None, optimizer=None, **kwargs):
        self.started = time.perf_counter()
        # A full-parameter snapshot is ~12.3 GB for Qwen2.5-3B FP32 and is a
        # diagnostic, not part of CBG. Keep exact update norms for LoRA but do
        # not jeopardize full-parameter training merely to report this scalar.
        if self.measure_update:
            self.before = tree_clone(tree_detach(self.trainer._admissible_parameters(model)))
        # Some Trainer versions leave CallbackHandler.optimizer unset when
        # the optimizer is attached after construction. Read the live Trainer
        # optimizer rather than losing the applied (pre-scheduler) LR metric.
        if optimizer is None:
            optimizer = getattr(self.trainer, "optimizer", None)
        self.step_lr = float(optimizer.param_groups[0]["lr"]) if optimizer is not None else None

    def on_optimizer_step(self, args, state, control, model=None, **kwargs):
        current = self.trainer._admissible_parameters(model)
        metrics = {}
        if self.measure_update:
            if self.before is None:
                raise RuntimeError("missing pre-optimizer snapshot for CBG update diagnostics")
            with torch.no_grad():
                metrics["train/optimizer_step_norm"] = tree_norm(tree_sub(current, self.before))
            self.before = None
        if self.step_lr is not None:
            metrics["train/optimizer_lr_used"] = self.step_lr
        device = next(iter(current.values())).device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            metrics.update({
                "system/max_memory_allocated": torch.cuda.max_memory_allocated(device),
                "system/max_memory_reserved": torch.cuda.max_memory_reserved(device),
            })
        metrics["system/optimizer_step_time"] = time.perf_counter() - self.started
        self.trainer._record_metrics(metrics)


class CbgTrainer(Trainer):
    """Trainer for legacy LoRA/full-space and implementation-spec CBG gradients."""

    def __init__(
        self,
        *args: Any,
        cbg_args: CbgArguments,
        reference_model: nn.Module | None = None,
        data_manifest: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.cbg_args = cbg_args
        self.reference_model = reference_model
        self.data_manifest = dict(data_manifest) if data_manifest is not None else None
        self.prev_u: OrderedTensorTree | None = None
        self.prev_basis: tuple[OrderedTensorTree, ...] = ()
        self.calibrated_mu: float | None = None
        self._beta_candidate_sum = 0.0
        self._beta_candidate_count = 0
        self._calibrated_beta: float | None = None
        self._pending_metric_sums: dict[str, float] = {}
        self._pending_metric_counts: dict[str, int] = {}
        self._pending_metric_maxima: dict[str, float] = {}
        self._loaded_cbg_checkpoint: str | None = None

        super().__init__(*args, **kwargs)
        if self.cbg_args.cbg_version == "v5":
            self.add_callback(_OptimizerStepMetrics(self))
        # Our training_step consumes and scales its own explicit gradient.  It
        # never passes num_items_in_batch through to the language model.
        self.model_accepts_loss_kwargs = False

        self._validate_runtime()
        functional_model = self.accelerator.unwrap_model(self.model)
        if self.cbg_args.parameter_space == "lora":
            admissible = select_lora_parameters(
                functional_model,
                target_modules=self.cbg_args.lora_target_modules,
                trainable_only=True,
            )
        elif self.cbg_args.cbg_version == "spec":
            from src.cbg.spec_hooks import spec_linear_modules
            admissible = OrderedDict((name, module.weight) for name, module in
                spec_linear_modules(functional_model, self.cbg_args.transformer_block_path).items())
        else:
            admissible = select_admissible_parameters(functional_model, trainable_only=True)
        self._admissible_names = tuple(admissible)
        all_trainable = tuple(
            name for name, parameter in functional_model.named_parameters() if parameter.requires_grad
        )
        if self.cbg_args.cbg_version != "spec" and set(all_trainable) != set(self._admissible_names):
            unexpected = sorted(set(all_trainable) - set(self._admissible_names))
            missing = sorted(set(self._admissible_names) - set(all_trainable))
            raise ValueError(
                "CBG requires every and only configured admissible coordinates to be trainable; "
                f"unexpected={unexpected}, missing={missing}"
            )
        admissible_dtypes = (
            (torch.bfloat16, torch.float32, torch.float64)
            if self.cbg_args.experimental_bf16_geometry
            else (torch.float32, torch.float64)
        )
        non_fp32 = [
            name
            for name, parameter in admissible.items()
            if parameter.dtype not in admissible_dtypes
        ]
        if non_fp32:
            raise ValueError(f"CBG has unsupported admissible leaf dtypes: {non_fp32}")
        if getattr(getattr(functional_model, "config", None), "use_cache", False):
            functional_model.config.use_cache = False

        if self.cbg_args.reference_mode == "disabled_adapter":
            if reference_model is not None:
                raise ValueError(
                    "reference_model must be omitted when reference_mode='disabled_adapter'"
                )
            if not callable(getattr(functional_model, "disable_adapter", None)):
                raise ValueError(
                    "reference_mode='disabled_adapter' requires a PEFT model exposing disable_adapter()"
                )
        else:
            if reference_model is None:
                raise ValueError(
                    "reference_mode='separate_model' requires reference_model"
                )
            reference_model.requires_grad_(False)
            reference_model.eval()
            try:
                reference_device = self.cbg_args.reference_device or str(self.args.device)
                reference_model.to(torch.device(reference_device))
                self._reference_device = torch.device(reference_device)
            except (RuntimeError, ValueError) as error:
                raise ValueError(
                    "the separate reference model could not be moved to the Trainer device"
                ) from error

    def _validate_runtime(self) -> None:
        if self.cbg_args.cbg_version == "spec":
            if self.args.bf16 or self.args.fp16 or self.args.tf32:
                raise ValueError("spec CBG requires bf16=False, fp16=False, tf32=False")
            if self.args.per_device_train_batch_size != 1:
                raise ValueError("spec dataset items are joint batches: per_device_train_batch_size must be 1")
        low_precision = self.args.bf16 or self.args.fp16
        experimental_bf16 = (
            self.cbg_args.experimental_bf16_geometry
            and self.cbg_args.parameter_space == "full"
            and self.cbg_args.geometry_solver == "dual"
            and self.args.bf16
            and not self.args.fp16
        )
        if (self.cbg_args.geometry_solver in {"dual", "bounded_ritz"}
            or self.cbg_args.spectral_derivative_backend == "direct_rof") and low_precision and not experimental_bf16:
            raise ValueError("dual geometry currently requires FP32 forwards: set bf16=False and fp16=False")
        if self.cbg_args.method != "cbg":
            raise ValueError("CbgTrainer requires method='cbg'")
        if self.args.save_only_model:
            raise ValueError(
                "CBG requires save_only_model=False so optimizer and scheduler can resume"
            )
        if self.args.remove_unused_columns:
            raise ValueError(
                "CBG nested batches require TrainingArguments.remove_unused_columns=False"
            )
        if self.args.fp16:
            raise ValueError("the initial CBG trainer supports BF16 or FP32, not FP16")
        if self.args.gradient_checkpointing:
            raise ValueError(
                "gradient checkpointing is disabled for the initial torch.func CBG implementation"
            )
        if getattr(self.args, "torch_compile", False):
            raise ValueError("torch.compile is not supported by the initial CBG trainer")
        if self.args.world_size != 1 or self.accelerator.num_processes != 1:
            raise ValueError("the initial CBG trainer supports one process only")
        if self.args.n_gpu > 1:
            raise ValueError(
                "the initial CBG trainer requires at most one visible GPU; DataParallel is unsupported"
            )
        if self.is_fsdp_enabled or self.is_deepspeed_enabled:
            raise ValueError("FSDP and DeepSpeed are not supported by the initial CBG trainer")
        if getattr(self, "is_model_parallel", False):
            raise ValueError("model-parallel execution is not supported by the initial CBG trainer")
        if getattr(self, "use_apex", False):
            raise ValueError("Apex AMP is not supported by the initial CBG trainer")

    def _admissible_parameters(self, model: nn.Module) -> OrderedTensorTree:
        unwrapped = self.accelerator.unwrap_model(model)
        return select_admissible_parameters(
            unwrapped,
            parameter_names=self._admissible_names,
            trainable_only=True,
        )

    def _reference_logits(self, model: nn.Module, benign_batch: Mapping[str, Tensor]) -> Tensor:
        kwargs = {"input_ids": benign_batch["input_ids"]}
        if "attention_mask" in benign_batch:
            kwargs["attention_mask"] = benign_batch["attention_mask"]

        if self.cbg_args.reference_mode == "disabled_adapter":
            reference = self.accelerator.unwrap_model(model)
            disable_adapter = reference.disable_adapter
            with _temporary_eval(reference), disable_adapter(), torch.no_grad():
                return select_response_logits(_extract_logits(reference(**kwargs)), benign_batch).detach()

        assert self.reference_model is not None
        from src.cbg.spec import spec_precision
        precision = (spec_precision(self.reference_model)
                     if self.cbg_args.cbg_version == "spec" else nullcontext())
        with precision, torch.no_grad():
            reference_device = getattr(self, "_reference_device", self.args.device)
            reference_batch = {
                name: value.to(reference_device) if isinstance(value, Tensor) else value
                for name, value in benign_batch.items()
            }
            reference_kwargs = {
                "input_ids": reference_batch["input_ids"],
                "attention_mask": reference_batch.get("attention_mask"),
            }
            logits = _extract_logits(self.reference_model(**reference_kwargs))
            # Keep selected logits on the reference device. The spec loss
            # moves them only while evaluating an endpoint, so the large
            # vocabulary tensor does not occupy GPU memory during the center
            # forward/backward passes.
            return select_response_logits(logits, reference_batch).detach()

    def _preservation_gradients(self, model, parameters, safe_batch, benign_batch):
        functional_model = self.accelerator.unwrap_model(model)
        width = self.cbg_args.preservation_chunk_size
        alignment = streamed_loss_gradient(
            ((fraction, safe_alignment_loss(functional_model, parameters, chunk))
             for fraction, chunk in sequence_chunks(safe_batch, width)), parameters,
        )
        def benign_losses():
            for fraction, chunk in sequence_chunks(benign_batch, width):
                reference = self._reference_logits(model, chunk)
                yield fraction, benign_preservation_loss(functional_model, parameters, chunk, reference)
                del reference
        benign = streamed_loss_gradient(benign_losses(), parameters)
        return (*alignment, *benign)

    def _resolve_mu(
        self,
        benign_functional,
        parameters: TensorTree,
    ) -> float:
        if self.calibrated_mu is not None:
            return self.calibrated_mu

        if self.cbg_args.damping_mode == "absolute":
            value = float(self.cbg_args.mu)
        else:
            calibrated = calibrate_relative_damping(
                benign_functional,
                parameters,
                damping_fraction=self.cbg_args.damping_fraction,
                num_directions=self.cbg_args.damping_calibration_directions,
                mu_floor=self.cbg_args.mu_floor,
            )
            value = float(calibrated.item())
        _require_positive(value, "calibrated_mu")
        self.calibrated_mu = value
        return value

    def _effective_beta(
        self,
        preservation_gradient: TensorTree,
        spectral_gradient: TensorTree,
    ) -> float:
        if self.cbg_args.beta_mode == "fixed":
            return float(self.cbg_args.beta)

        if self.state.global_step < self.cbg_args.beta_calibration_steps:
            preservation_norm = float(tree_norm(preservation_gradient).float().item())
            spectral_norm = float(tree_norm(spectral_gradient).float().item())
            if not math.isfinite(spectral_norm) or not math.isfinite(preservation_norm):
                raise FloatingPointError("non-finite gradients during beta calibration")
            if spectral_norm <= 1e-12 or preservation_norm <= 1e-12:
                # A degenerate batch must not manufacture a very large beta.
                return self._calibrated_beta if self._calibrated_beta is not None else self.cbg_args.beta
            candidate = self.cbg_args.beta_target_ratio * preservation_norm / spectral_norm
            candidate = min(max(candidate, self.cbg_args.beta_min), self.cbg_args.beta_max)
            if not math.isfinite(candidate) or candidate < 0:
                raise FloatingPointError("gradient-ratio beta calibration produced an invalid value")
            self._beta_candidate_sum += candidate
            self._beta_candidate_count += 1
            self._calibrated_beta = (
                self._beta_candidate_sum / self._beta_candidate_count
            )

        if self._calibrated_beta is None:
            if self._beta_candidate_count == 0:
                return self.cbg_args.beta
            self._calibrated_beta = (
                self._beta_candidate_sum / self._beta_candidate_count
            )
        return self._calibrated_beta

    def _gradient_accumulation_divisor(self) -> int:
        # Transformers 4.49 uses the configured count.  Newer versions expose
        # the actual size of the final, possibly-short accumulation group.
        divisor = getattr(
            self,
            "current_gradient_accumulation_steps",
            self.args.gradient_accumulation_steps,
        )
        if isinstance(divisor, Tensor):
            divisor = int(divisor.item())
        divisor = int(divisor)
        if divisor < 1:
            raise RuntimeError("gradient accumulation divisor must be positive")
        return divisor

    def _should_compute_geometry(self) -> bool:
        """Whether this step needs the Rayleigh and spectral computations."""

        return not (
            self.cbg_args.beta_mode == "fixed" and self.cbg_args.beta == 0.0
        )

    def _search_geometry(self, safety, benign, parameters, **kwargs):
        if self.cbg_args.geometry_solver == "bounded_ritz":
            args = self.cbg_args
            return bounded_rayleigh_solve(
                safety, benign, parameters, mu=kwargs["mu"], warm_start=kwargs["warm_start"],
                warm_basis=getattr(self, "prev_basis", ()), initial_size=args.ritz_initial_size,
                max_size=args.ritz_max_size, max_enrichments=args.ritz_max_enrichments,
                residual_tolerance=args.ritz_residual_tolerance,
                orthogonality_tolerance=args.ritz_orthogonality_tolerance,
                enclosure_atol=args.enclosure_atol, enclosure_rtol=args.enclosure_rtol,
                rho_scale=args.enclosure_rho_scale, dual_max_size=args.dual_max_size,
                dual_tolerance=args.dual_tolerance, dual_kernel_tolerance=args.dual_kernel_tolerance,
                dual_residual_tolerance=args.dual_residual_tolerance,
                dual_column_batch_size=args.dual_column_batch_size,
                dual_refinement_steps=args.dual_refinement_steps,
            )
        if self.cbg_args.geometry_solver == "dual":
            configured_width = kwargs.get(
                "dual_column_batch_size", self.cbg_args.dual_column_batch_size
            )
            allow_width_fallback = kwargs.get(
                "dual_column_fallback", self.cbg_args.dual_column_fallback
            )
            width = configured_width
            while True:
                try:
                    result = dual_rayleigh_solve(
                        safety, benign, parameters, mu=kwargs["mu"], warm_start=kwargs["warm_start"],
                        max_size=self.cbg_args.dual_max_size, tolerance=self.cbg_args.dual_tolerance,
                        kernel_tolerance=self.cbg_args.dual_kernel_tolerance,
                        residual_tolerance=self.cbg_args.dual_residual_tolerance,
                        column_batch_size=width,
                        refinement_steps=self.cbg_args.dual_refinement_steps,
                        allow_bf16=self.cbg_args.experimental_bf16_geometry,
                    )
                    self._effective_dual_column_width = width
                    self._dual_column_fallbacks = int(width != self.cbg_args.dual_column_batch_size)
                    return result
                except FloatingPointError as error:
                    # Column batching changes only execution order, so a
                    # narrower recomputation targets exactly the same sample
                    # kernel/eigenproblem. Retry only independent residual
                    # verification failures; never mask nonfinite, asymmetric,
                    # non-PSD, projection, or normalization failures.
                    retryable = str(error).startswith(
                        "dual reconstruction failed independent verification:"
                    )
                    if (not allow_width_fallback or not retryable or width == 1):
                        raise
                    # Retry at the largest lower power-of-two width.  Plain
                    # integer halving incorrectly skipped width 2 for an odd
                    # configured width (3 -> 1), removing a materially
                    # different reduction order before the slow scalar path.
                    # This gives 8->4->2->1, 4->2->1, and 3->2->1.
                    next_width = 1 << ((width - 1).bit_length() - 1)
                    logger.warning(
                        "dual width %d missed verification; recomputing the same V5 operator at width %d",
                        width, next_width,
                    )
                    width = next_width
        return warm_started_rayleigh_ascent(safety, benign, parameters, **kwargs)

    def _backward_explicit_gradient(
        self,
        parameters: TensorTree,
        gradient: TensorTree,
        *,
        divisor: int,
    ) -> None:
        """Accumulate an explicit gradient through Trainer's AMP machinery.

        The proxy has numerical value zero but derivative ``gradient``.  Unlike
        direct assignment to ``parameter.grad``, this invokes Accelerate's AMP
        scaler and distributed gradient hooks and naturally accumulates with
        earlier microbatches.
        """

        proxy: Tensor | None = None
        for name, parameter in parameters.items():
            explicit = gradient[name].detach().to(
                device=parameter.device,
                dtype=parameter.dtype,
            )
            term = ((parameter - parameter.detach()) * explicit).sum()
            proxy = term if proxy is None else proxy + term
        if proxy is None:
            raise RuntimeError("cannot inject an empty CBG gradient")
        self.accelerator.backward(proxy / divisor)

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        num_items_in_batch: Tensor | None = None,
    ) -> Tensor:
        """Compute one complete CBG microbatch gradient at the current center."""

        del num_items_in_batch
        if self.cbg_args.cbg_version == "spec":
            return self._spec_training_step(model, inputs)
        step_started = time.perf_counter()
        memory_device: torch.device | None = None
        if torch.cuda.is_available() and self.args.device.type == "cuda":
            memory_device = self.args.device
            torch.cuda.reset_peak_memory_stats(memory_device)

        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()
        prepared = self._prepare_inputs(inputs)
        self._validate_batch(prepared)

        functional_model = self.accelerator.unwrap_model(model)
        parameters = self._admissible_parameters(model)
        harmful_batch = prepared["harmful"]
        safe_batch = prepared["safe"]
        benign_batch = prepared["benign"]
        preservation_only = not self._should_compute_geometry()
        zero_safety = False
        diagnostics = None
        warm_basis: tuple[OrderedTensorTree, ...] = ()
        spectral_gradient = None
        spectral_backend = "disabled"
        spectral_terms = 0
        spectral_delta = None
        spectral_cancellation = None
        center_rho = center_qh = center_qb = center_b = None
        phase_times = dict(geometry=0., spectral=0., preservation=0.)
        self._effective_dual_column_width = 0
        self._dual_column_fallbacks = 0
        self._effective_functional_chunk_size = 0
        self._functional_chunk_fallbacks = 0
        def phase_clock():
            if memory_device is not None:
                torch.cuda.synchronize(memory_device)
            return time.perf_counter()

        # A fixed deterministic map is essential for reusing metric products.
        # Disable stochastic layers for all CBG evaluations, without disabling
        # gradients. Avoid reusing autocast weight casts across JVP/no_grad/VJP.
        with _temporary_eval(functional_model), self.compute_loss_context_manager(), torch.autocast(
            device_type=self.args.device.type, enabled=self.args.bf16,
            dtype=torch.bfloat16, cache_enabled=False,
        ):
            if not preservation_only:
                phase_start = phase_clock()
                self._effective_functional_chunk_size = self.cbg_args.functional_chunk_size
                self._functional_chunk_fallbacks = 0
                safety_functional = make_safety_functional(
                    functional_model,
                    harmful_batch,
                    safe_batch,
                    chunk_size=self.cbg_args.functional_chunk_size,
                )
                benign_functional = make_benign_functional(
                    functional_model,
                    benign_batch,
                    chunk_size=self.cbg_args.functional_chunk_size,
                    token_blocks=self.cbg_args.benign_token_blocks,
                )
                mu = self._resolve_mu(benign_functional, parameters)

                adaptive = self.cbg_args.inner_policy == "adaptive"
                minimum_steps = (
                    self.cbg_args.inner_initial_steps if adaptive and self.prev_u is None
                    else self.cbg_args.inner_steps
                )
                search_options = dict(
                    mu=mu,
                    inner_steps=minimum_steps,
                    inner_lr=self.cbg_args.inner_lr,
                    warm_start=self.prev_u,
                    eps_norm=self.cbg_args.eps_norm,
                    max_steps=self.cbg_args.inner_max_steps if adaptive else None,
                    residual_tolerance=self.cbg_args.inner_residual_tolerance,
                    rayleigh_tolerance=self.cbg_args.inner_rayleigh_tolerance,
                    max_backtracks=self.cbg_args.inner_max_backtracks if adaptive else 0,
                )
                retry_scalar = False
                try:
                    rayleigh = self._search_geometry(
                        safety_functional, benign_functional, parameters, **search_options
                    )
                except FloatingPointError as error:
                    retryable = str(error).startswith(
                        "dual reconstruction failed independent verification:"
                    )
                    can_retry_scalar = (
                        self.cbg_args.geometry_solver == "dual"
                        and self.cbg_args.dual_functional_chunk_fallback
                        and self.cbg_args.functional_chunk_size != 1
                        and retryable
                    )
                    if not can_retry_scalar:
                        raise
                    logger.warning(
                        "dual verification failed after column-width retries; "
                        "recomputing the same V5 operator with functional chunk 1 and dual width 1: %s",
                        error,
                    )
                    retry_scalar = True
                # Leave the exception handler BEFORE allocating retry graphs:
                # its traceback retains failed solver frames and tensor trees.
                if retry_scalar:
                    # Functional chunking changes execution order only. Rebuild
                    # both maps so the accepted direction and direct derivative
                    # use exactly the same scalar-streamed operator.
                    del safety_functional, benign_functional
                    safety_functional = make_safety_functional(
                        functional_model, harmful_batch, safe_batch, chunk_size=1
                    )
                    benign_functional = make_benign_functional(
                        functional_model, benign_batch, chunk_size=1,
                        token_blocks=self.cbg_args.benign_token_blocks,
                    )
                    self._effective_functional_chunk_size = 1
                    self._functional_chunk_fallbacks = 1
                    rayleigh = self._search_geometry(
                        safety_functional,
                        benign_functional,
                        parameters,
                        dual_column_batch_size=1,
                        dual_column_fallback=False,
                        **search_options,
                    )
                if not rayleigh.diagnostics.converged:
                    if self.cbg_args.inner_fail_on_nonconvergence or self.cbg_args.cbg_version == "v5":
                        diagnostic = rayleigh.diagnostics
                        raise RuntimeError(
                            "CBG inner search did not converge; no outer update applied. "
                            f"rho={float(diagnostic.rho_hat):.6g}, "
                            f"relative_residual={float(diagnostic.relative_residual):.6g}, "
                            f"accepted_steps={diagnostic.inner_steps}, "
                            f"rejected_trials={diagnostic.rejected_steps}. "
                            "Validate the solver budget and learning rate on frozen batches."
                        )
                    if self.state.global_step == 0:
                        logger.warning("CBG search is not converged; rho_hat is a directional estimate, not a dominant eigenvalue")
                direction = tree_detach(rayleigh.direction)
                diagnostics = rayleigh.diagnostics
                warm_basis = rayleigh.warm_basis
                phase_times["geometry"] = phase_clock() - phase_start
                zero_safety = diagnostics.zero_safety
                if zero_safety:
                    preservation_only = True
                    del rayleigh, direction
                else:
                    phase_start = phase_clock()
                    outer_center = compact_spectral_center(rayleigh.center)
                    # Only u, frozen small coefficients and scalar diagnostics
                    # are needed by either outer derivative. Release G_h u,
                    # G_b u and B u BEFORE allocating its higher-order graph.
                    del rayleigh
                    derivative = (direct_spectral_gradient if self.cbg_args.spectral_derivative_backend == "direct_rof"
                                  else centered_spectral_gradient)
                    fd_options = {} if derivative is direct_spectral_gradient else dict(
                        tau_fd=self.cbg_args.tau_fd, theta_floor=self.cbg_args.theta_floor,
                        u_floor=self.cbg_args.u_floor, same_randomness=True, retain_endpoint_gradients=False,
                    )
                    spectral = derivative(
                        safety_functional,
                        benign_functional,
                        parameters,
                        direction,
                        mu=mu,
                        center=outer_center,
                        **fd_options,
                    )
                    spectral_gradient = spectral.gradient
                    spectral_backend = spectral.backend
                    spectral_terms = spectral.terms_evaluated
                    spectral_delta = spectral.delta
                    spectral_cancellation = spectral.fd_cancellation
                    center_rho = spectral.center.rho.detach()
                    center_qh = spectral.center.qh.detach()
                    center_qb = spectral.center.qb.detach()
                    center_b = spectral.center.b.detach()
                    # The full center owns four parameter-sized trees.  The
                    # outer gradient and scalar diagnostics are sufficient
                    # after direct differentiation, so release those products
                    # before constructing preservation gradients.
                    del spectral, outer_center
                    phase_times["spectral"] = phase_clock() - phase_start

            # Preservation losses are deliberately distinct from the two
            # functional metrics used above.
            phase_start = phase_clock()
            alignment_loss, alignment_gradient, benign_loss, benign_gradient = self._preservation_gradients(
                model, parameters, safe_batch, benign_batch,
            )
            phase_times["preservation"] = phase_clock() - phase_start

        alignment_norm = tree_norm(alignment_gradient)
        benign_norm = tree_norm(benign_gradient)
        # Both loss streams return private detached accumulators. Reuse the
        # alignment tree for the composed gradient rather than retaining two
        # additional P-sized trees. This is algebraically the same axpy sum.
        training_gradient = accumulate_detached_gradient(
            alignment_gradient, benign_gradient, alpha=self.cbg_args.alpha,
        )
        preservation_norm = tree_norm(training_gradient)
        del alignment_gradient, benign_gradient
        spectral_dot_preservation = preservation_norm.new_zeros(())
        if preservation_only:
            effective_beta = 0.0
            cbg_norm = preservation_norm.new_zeros(())
        else:
            assert spectral_gradient is not None
            cbg_norm = tree_norm(spectral_gradient)
            effective_beta = self._effective_beta(training_gradient, spectral_gradient)
            spectral_dot_preservation = tree_dot(training_gradient, spectral_gradient)
            if effective_beta != 0:
                accumulate_detached_gradient(
                    training_gradient, spectral_gradient, alpha=effective_beta,
                )
            del spectral_gradient
        # Check before backward so a numerical failure cannot poison optimizer state.
        composed_norm = tree_norm(training_gradient)
        if not bool(torch.isfinite(composed_norm)):
            raise FloatingPointError("non-finite composed CBG gradient")

        divisor = self._gradient_accumulation_divisor()
        self._backward_explicit_gradient(
            parameters,
            training_gradient,
            divisor=divisor,
        )
        # Explicit backward has copied the composed gradient into live
        # parameter .grad buffers; release the detached construction trees
        # before AdamW materializes its state.
        del training_gradient

        # A failed spectral or preservation computation never changes the warm
        # start.  Persist only a direction from a fully successful backward.
        if not preservation_only:
            self.prev_u = tree_detach(direction)
            self.prev_basis = tuple(tree_detach(v) for v in warm_basis)
        elif zero_safety:
            self.prev_u = None
            self.prev_basis = ()

        if memory_device is not None:
            # Most geometry diagnostics already synchronize, but the final
            # proxy backward may still be queued when wall time is sampled.
            torch.cuda.synchronize(memory_device)
        metrics: dict[str, float | Tensor] = {
            "train/l_aln": alignment_loss.detach(),
            "train/l_bgn": benign_loss.detach(),
            "train/grad_norm_aln": alignment_norm,
            "train/grad_norm_bgn": benign_norm,
            "train/grad_norm_cbg": cbg_norm,
            "train/grad_norm_alpha_bgn": self.cbg_args.alpha * benign_norm,
            "train/grad_norm_beta_cbg": effective_beta * cbg_norm,
            "train/grad_norm_preservation": preservation_norm,
            # These are per-microbatch diagnostics BEFORE Trainer accumulation
            # and clipping. Trainer's grad_norm measures the actual accumulated
            # pre-clip norm at the optimizer boundary.
            "train/grad_norm_composed": composed_norm,
            "train/cbg_preservation_cosine": (
                spectral_dot_preservation / (cbg_norm * preservation_norm).clamp_min(1e-30)
            ).clamp(-1., 1.),
            "train/cbg_preservation_ratio": effective_beta * cbg_norm / preservation_norm.clamp_min(1e-12),
            "cbg/enabled": float(self._should_compute_geometry()),
            "cbg/zero_safety": float(zero_safety),
            "cbg/direct_rof": float(not preservation_only and spectral_backend == "direct_rof"),
            "train/beta_effective": effective_beta,
            "system/step_time": time.perf_counter() - step_started,
            "system/max_memory_allocated": (
                float(torch.cuda.max_memory_allocated(memory_device))
                if memory_device is not None
                else 0.0
            ),
            "system/max_memory_reserved": float(torch.cuda.max_memory_reserved(memory_device)) if memory_device is not None else 0.,
        }
        if preservation_only:
            # These finite zeros explicitly mean that the corresponding CBG
            # computation was disabled, rather than attempted and found zero.
            metrics.update(
                {
                    "train/rho_hat": 0.0,
                    "train/qh_u": 0.0,
                    "train/qb_u": 0.0,
                    "train/qb_damped_u": 0.0,
                    "cbg/mu": 0.0,
                    "cbg/delta": 0.0,
                    "cbg/u_norm": 0.0,
                    "cbg/b_norm": 0.0,
                    "cbg/eig_residual": 0.0,
                    "cbg/rayleigh_start": 0.0,
                    "cbg/rayleigh_final": 0.0,
                    "cbg/warm_start_cosine": 0.0,
                }
            )
        else:
            assert diagnostics is not None
            assert center_rho is not None and center_qh is not None and center_qb is not None and center_b is not None
            rayleigh_values = diagnostics.rayleigh_values
            metrics.update(
                {
                    "train/rho_hat": center_rho,
                    "train/qh_u": center_qh,
                    "train/qb_u": center_qb,
                    "train/qb_damped_u": center_b,
                    "cbg/mu": mu,
                    "cbg/delta": spectral_delta if spectral_delta is not None else 0.,
                    "cbg/u_norm": diagnostics.u_euclidean_norm,
                    "cbg/b_norm": diagnostics.b_mu_norm,
                    "cbg/eig_residual": diagnostics.eig_residual_norm,
                    "cbg/relative_residual": diagnostics.relative_residual,
                    "cbg/converged": float(diagnostics.converged),
                    "cbg/inner_steps": diagnostics.inner_steps,
                    "cbg/rejected_steps": diagnostics.rejected_steps,
                    "cbg/metric_evaluations": diagnostics.metric_evaluations,
                    "cbg/fd_cancellation": spectral_cancellation if spectral_cancellation is not None else 0.,
                    "cbg/fd_diagnostics_applicable": float(spectral_backend == "centered_fd"),
                    "cbg/direct_terms": spectral_terms,
                    "cbg/rayleigh_start": rayleigh_values[0],
                    "cbg/rayleigh_final": rayleigh_values[-1],
                    "cbg/warm_start_cosine": (
                        diagnostics.warm_start_cosine
                        if diagnostics.warm_start_cosine is not None
                        else 0.0
                    ),
                }
            )
            for index, value in enumerate(rayleigh_values):
                metrics[f"cbg/rayleigh_iter_{index}"] = value
        if not preservation_only or zero_safety:
            assert diagnostics is not None
            metrics["cbg/kernel_columns"] = diagnostics.kernel_columns
            for key in ("eigengap", "kernel_asymmetry", "kernel_projection", "damping_fraction", "dual_value_error",
                        "enclosure_lower", "enclosure_upper", "enclosure_eta", "enclosure_gamma", "b_orthogonality", "ritz_eigengap"):
                value = getattr(diagnostics, key)
                if value is not None:
                    metrics[f"cbg/{key}"] = value
            for key in ("basis_size", "enclosure_checks", "safety_row_pullbacks", "dual_fallback", "refinement_steps", "refinement_jvps"):
                metrics[f"cbg/{key}"] = float(getattr(diagnostics, key))
            if diagnostics.fallback_reason:
                metrics[f"cbg/fallback_{diagnostics.fallback_reason}"] = 1.
            if diagnostics.damping_fraction is None:
                metrics["cbg/damping_fraction"] = mu * diagnostics.u_euclidean_norm.square() / diagnostics.qb_damped
            if zero_safety:
                metrics["cbg/mu"] = mu
                metrics["cbg/converged"] = 1.0
                metrics["cbg/metric_evaluations"] = diagnostics.metric_evaluations
        metrics.update({f"system/{phase}_time": duration for phase, duration in phase_times.items()})
        metrics["cbg/dual_column_batch_size"] = self.cbg_args.dual_column_batch_size
        metrics["cbg/dual_residual_tolerance"] = self.cbg_args.dual_residual_tolerance
        metrics["cbg/effective_dual_column_width"] = float(
            getattr(self, "_effective_dual_column_width", self.cbg_args.dual_column_batch_size)
        )
        metrics["cbg/dual_column_fallbacks"] = float(
            getattr(self, "_dual_column_fallbacks", 0)
        )
        metrics["cbg/effective_functional_chunk_size"] = float(
            getattr(self, "_effective_functional_chunk_size", self.cbg_args.functional_chunk_size)
        )
        metrics["cbg/functional_chunk_fallbacks"] = float(
            getattr(self, "_functional_chunk_fallbacks", 0)
        )
        self._record_metrics(metrics)

        reported_loss = alignment_loss.detach() + self.cbg_args.alpha * benign_loss.detach()
        if self.cbg_args.cbg_version == "v5" and not preservation_only:
            reported_loss = reported_loss + effective_beta * center_rho
        return reported_loss / divisor

    def _spec_training_step(self, model, inputs):
        from src.cbg.spec import _clock, spec_training_gradient
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()
        batch = self._prepare_inputs(inputs)
        self._validate_batch(batch)
        if (batch["harmful"]["input_ids"].shape[0] != self.cbg_args.safety_batch_size
            or batch["benign"]["input_ids"].shape[0] != self.cbg_args.benign_batch_size):
            raise ValueError("joint batch sizes do not match configured m,n")
        functional_model = self.accelerator.unwrap_model(model)
        device = batch["benign"]["input_ids"].device
        started = _clock(device)
        reference = self._reference_logits(model, batch["benign"]) if self.cbg_args.alpha else None
        reference_time = _clock(device) - started
        result = spec_training_gradient(functional_model, batch, self.cbg_args.spec_config(),
                                        reference_logits=reference)
        parameters = OrderedDict((name, p) for name, p in functional_model.named_parameters() if p.requires_grad)
        divisor = self._gradient_accumulation_divisor()
        self._backward_explicit_gradient(parameters, result.gradient, divisor=divisor)
        result.diagnostics["reference_time"] = reference_time
        result.diagnostics["training_step_time"] = _clock(device) - started
        self._record_metrics({f"cbg/{name}": value for name, value in result.diagnostics.items()})
        return result.loss / divisor

    @staticmethod
    def _validate_batch(batch: Mapping[str, Any]) -> None:
        required_groups = {"harmful", "safe", "benign"}
        missing_groups = required_groups - set(batch)
        if missing_groups:
            raise KeyError(f"CBG batch is missing groups: {sorted(missing_groups)}")
        for group in sorted(required_groups):
            sequence = batch[group]
            if not isinstance(sequence, Mapping):
                raise TypeError(f"CBG batch group {group!r} must be a mapping")
            missing = {"input_ids", "attention_mask", "response_mask"} - set(sequence)
            if missing:
                raise KeyError(
                    f"CBG batch group {group!r} is missing fields: {sorted(missing)}"
                )

    def _record_metrics(self, metrics: Mapping[str, float | Tensor]) -> None:
        for key, raw_value in metrics.items():
            value = (
                float(raw_value.detach().float().item())
                if isinstance(raw_value, Tensor)
                else float(raw_value)
            )
            if not math.isfinite(value):
                raise FloatingPointError(f"non-finite CBG diagnostic {key}: {value}")
            if key in {"system/max_memory_allocated", "system/max_memory_reserved"}:
                self._pending_metric_maxima[key] = max(
                    value,
                    self._pending_metric_maxima.get(key, 0.0),
                )
            else:
                self._pending_metric_sums[key] = (
                    self._pending_metric_sums.get(key, 0.0) + value
                )
                self._pending_metric_counts[key] = (
                    self._pending_metric_counts.get(key, 0) + 1
                )

    def log(self, logs: dict[str, float], *args: Any, **kwargs: Any) -> None:
        # Trainer calls log after incrementing global_step.  Merge there rather
        # than logging from each microbatch under the preceding step number.
        if "loss" in logs and (
            self._pending_metric_sums or self._pending_metric_maxima
        ):
            logs = dict(logs)
            for key, total in self._pending_metric_sums.items():
                logs[key] = total / self._pending_metric_counts[key]
            logs.update(self._pending_metric_maxima)
            self._pending_metric_sums.clear()
            self._pending_metric_counts.clear()
            self._pending_metric_maxima.clear()
        super().log(logs, *args, **kwargs)

    def _checkpoint_config(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self.cbg_args), sort_keys=True))

    def _checkpoint_runtime(self) -> dict[str, Any]:
        """TrainingArguments fields that change V5's executed batch/operator."""
        return {name: getattr(self.args, name, None) for name in (
            "per_device_train_batch_size", "gradient_accumulation_steps", "bf16", "fp16",
            "dataloader_drop_last", "seed", "data_seed",
            "max_grad_norm",
        )} | {"matmul_precision": torch.get_float32_matmul_precision()}

    def _save(self, output_dir: str | None = None, state_dict=None) -> None:
        super()._save(output_dir=output_dir, state_dict=state_dict)
        output_path = Path(output_dir or self.args.output_dir)
        self._save_cbg_artifacts(output_path)

    def _save_cbg_artifacts(self, output_dir: Path) -> None:
        if not self.args.should_save:
            return
        output_dir.mkdir(parents=True, exist_ok=True)
        config = self._checkpoint_config()
        fingerprint = _config_fingerprint(config)
        prev_u_cpu = None
        if self.prev_u is not None:
            prev_u_cpu = {
                name: value.detach().to(device="cpu").clone()
                for name, value in self.prev_u.items()
            }
        state = {
            "version": CBG_STATE_VERSION,
            "global_step": int(self.state.global_step),
            "admissible_names": list(self._admissible_names),
            "prev_u": prev_u_cpu,
            "prev_basis": [{name: value.detach().cpu().clone() for name, value in v.items()}
                           for v in getattr(self, "prev_basis", ())],
            "calibrated_mu": self.calibrated_mu,
            "beta_candidate_sum": self._beta_candidate_sum,
            "beta_candidate_count": self._beta_candidate_count,
            "calibrated_beta": self._calibrated_beta,
            "config": config,
            "config_fingerprint": fingerprint,
            "data_manifest": getattr(self, "data_manifest", None),
            "runtime_signature": self._checkpoint_runtime(),
        }

        config_path = output_dir / CBG_CONFIG_NAME
        config_tmp = output_dir / f".{CBG_CONFIG_NAME}.{os.getpid()}.tmp"
        state_path = output_dir / CBG_STATE_NAME
        state_tmp = output_dir / f".{CBG_STATE_NAME}.{os.getpid()}.tmp"
        try:
            with config_tmp.open("w", encoding="utf-8") as handle:
                json.dump(config, handle, indent=2, sort_keys=True)
                handle.write("\n")
            torch.save(state, state_tmp)
            os.replace(config_tmp, config_path)
            os.replace(state_tmp, state_path)
        finally:
            for temporary in (config_tmp, state_tmp):
                if temporary.exists():
                    temporary.unlink()

    def _load_from_checkpoint(
        self,
        resume_from_checkpoint: str,
        model: nn.Module | None = None,
    ) -> None:
        checkpoint = str(Path(resume_from_checkpoint).resolve())
        self._validate_checkpoint_config(Path(checkpoint))
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        if checkpoint != self._loaded_cbg_checkpoint:
            self._load_cbg_state(
                Path(checkpoint),
                model if model is not None else self.model,
            )
            self._loaded_cbg_checkpoint = checkpoint

    def _validate_checkpoint_config(self, checkpoint: Path) -> None:
        config_path = checkpoint / CBG_CONFIG_NAME
        if not config_path.is_file():
            raise FileNotFoundError(f"CBG checkpoint is missing {config_path}")
        with config_path.open("r", encoding="utf-8") as handle:
            saved = json.load(handle)
        current = self._checkpoint_config()
        if _canonical_config(saved) != _canonical_config(current):
            raise ValueError(
                "CBG checkpoint configuration does not match the current mathematical/data configuration"
            )

    def _load_cbg_state(self, checkpoint: Path, model: nn.Module) -> None:
        state_path = checkpoint / CBG_STATE_NAME
        if not state_path.is_file():
            raise FileNotFoundError(f"CBG checkpoint is missing {state_path}")
        try:
            state = torch.load(state_path, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - compatibility with older Torch.
            state = torch.load(state_path, map_location="cpu")
        if not isinstance(state, Mapping):
            raise ValueError("CBG checkpoint state must be a mapping")
        if state.get("version") != CBG_STATE_VERSION:
            raise ValueError(
                f"unsupported CBG checkpoint version: {state.get('version')!r}"
            )
        expected_fingerprint = _config_fingerprint(self._checkpoint_config())
        if state.get("config_fingerprint") != expected_fingerprint:
            raise ValueError("CBG checkpoint configuration fingerprint mismatch")
        if state.get("data_manifest") != getattr(self, "data_manifest", None):
            raise ValueError("CBG checkpoint data manifest differs from the current data/split")
        if self.cbg_args.cbg_version in {"v5", "spec"} and state.get("runtime_signature") != self._checkpoint_runtime():
            raise ValueError("V5 checkpoint batch, accumulation or precision settings differ")

        saved_names = tuple(state.get("admissible_names", ()))
        if saved_names != self._admissible_names:
            raise ValueError(
                "CBG checkpoint admissible parameter ordering does not match the current model"
            )
        parameters = self._admissible_parameters(model)
        saved_direction = state.get("prev_u")
        if saved_direction is None:
            self.prev_u = None
        else:
            if not isinstance(saved_direction, Mapping) or tuple(saved_direction) != saved_names:
                raise ValueError("CBG checkpoint prev_u has invalid parameter keys")
            restored: OrderedTensorTree = OrderedDict()
            for name, parameter in parameters.items():
                value = saved_direction[name]
                if not isinstance(value, Tensor) or value.shape != parameter.shape:
                    raise ValueError(
                        f"CBG checkpoint prev_u shape mismatch for {name!r}"
                    )
                if not bool(torch.isfinite(value).all().item()):
                    raise ValueError(f"CBG checkpoint prev_u is non-finite for {name!r}")
                if self.cbg_args.cbg_version == "v5" and value.dtype != parameter.dtype:
                    raise ValueError("V5 checkpoint direction precision differs from current coordinates")
                restored[name] = value.to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
            self.prev_u = tree_detach(restored)

        calibrated_mu = state.get("calibrated_mu")
        saved_basis = state.get("prev_basis", [])
        if not isinstance(saved_basis, (tuple, list)) or len(saved_basis) > self.cbg_args.ritz_max_size:
            raise ValueError("invalid checkpoint warm basis size")
        restored_basis = []
        for vector in saved_basis:
            if not isinstance(vector, Mapping) or tuple(vector) != self._admissible_names:
                raise ValueError("invalid checkpoint warm basis parameter order")
            restored = OrderedDict()
            for name, parameter in parameters.items():
                value = vector[name]
                if (not isinstance(value, Tensor) or value.shape != parameter.shape
                    or value.dtype != parameter.dtype or not bool(torch.isfinite(value).all())):
                    raise ValueError("invalid checkpoint warm basis leaf")
                restored[name] = value.detach().to(parameter)
            restored_basis.append(restored)
        self.prev_basis = tuple(restored_basis)
        if calibrated_mu is None:
            self.calibrated_mu = None
        else:
            self.calibrated_mu = float(calibrated_mu)
            _require_positive(self.calibrated_mu, "checkpoint calibrated_mu")

        self._beta_candidate_sum = float(state.get("beta_candidate_sum", 0.0))
        self._beta_candidate_count = int(state.get("beta_candidate_count", 0))
        calibrated_beta = state.get("calibrated_beta")
        self._calibrated_beta = (
            None if calibrated_beta is None else float(calibrated_beta)
        )
        if not math.isfinite(self._beta_candidate_sum) or self._beta_candidate_sum < 0:
            raise ValueError("checkpoint beta_candidate_sum is invalid")
        if self._beta_candidate_count < 0:
            raise ValueError("checkpoint beta_candidate_count is invalid")
        if self._calibrated_beta is not None and (
            not math.isfinite(self._calibrated_beta) or self._calibrated_beta < 0
        ):
            raise ValueError("checkpoint calibrated_beta is invalid")


__all__ = ["CBG_CONFIG_NAME", "CBG_STATE_NAME", "CbgArguments", "CbgTrainer"]
