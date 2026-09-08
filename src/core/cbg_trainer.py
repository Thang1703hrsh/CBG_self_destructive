"""Hugging Face Trainer integration for Collateral Barrier Geometry.

This module deliberately does not share the custom ``SeamTrainer`` inner
loop.  CBG computes its gradient explicitly in the configured LoRA coordinate
space, then hands that gradient back to the stock Transformers optimizer loop
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
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from transformers import Trainer

from src.cbg.functionals import (
    benign_preservation_loss,
    make_benign_functional,
    make_safety_functional,
    safe_alignment_loss,
    select_response_logits,
    sequence_chunks,
)
from src.cbg.rayleigh import calibrate_relative_damping, warm_started_rayleigh_ascent
from src.cbg.spectral_grad import (
    centered_spectral_gradient,
    compose_training_gradient,
    parameter_gradient,
    streamed_loss_gradient,
)
from src.cbg.tree_ops import (
    OrderedTensorTree,
    TensorTree,
    select_admissible_parameters,
    select_lora_parameters,
    tree_clone,
    tree_detach,
    tree_norm,
    tree_zeros_like,
    tree_add,
)


logger = logging.getLogger(__name__)

CBG_STATE_NAME = "cbg_state.pt"
CBG_CONFIG_NAME = "cbg_config.json"
CBG_STATE_VERSION = 2

_NON_RESUME_CONFIG_FIELDS = {
    "cbg_resume_from_checkpoint",
    "wandb_project_name",
    "wandb_run_name",
}


@dataclass
class CbgArguments:
    """Configuration specific to the CBG defense.

    Training-loop settings such as the optimizer, batch size, precision,
    scheduler, and checkpoint cadence remain ordinary ``TrainingArguments``.
    """

    method: str = "cbg"
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
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
        if self.damping_mode not in {"absolute", "relative"}:
            raise ValueError("damping_mode must be 'absolute' or 'relative'")
        if self.beta_mode not in {"fixed", "grad_ratio"}:
            raise ValueError("beta_mode must be 'fixed' or 'grad_ratio'")
        if self.inner_policy not in {"fixed", "adaptive"}:
            raise ValueError("inner_policy must be 'fixed' or 'adaptive'")
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
    # The round trip canonicalizes tuples to lists and rejects values that
    # cannot be represented in the human-readable checkpoint configuration.
    return json.loads(json.dumps(comparable, sort_keys=True))


def _config_fingerprint(config: Mapping[str, Any]) -> str:
    payload = json.dumps(_canonical_config(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CbgTrainer(Trainer):
    """Trainer that applies the complete LoRA-space CBG defense gradient."""

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
        self.calibrated_mu: float | None = None
        self._beta_candidate_sum = 0.0
        self._beta_candidate_count = 0
        self._calibrated_beta: float | None = None
        self._pending_metric_sums: dict[str, float] = {}
        self._pending_metric_counts: dict[str, int] = {}
        self._pending_metric_maxima: dict[str, float] = {}
        self._loaded_cbg_checkpoint: str | None = None

        super().__init__(*args, **kwargs)
        # Our training_step consumes and scales its own explicit gradient.  It
        # never passes num_items_in_batch through to the language model.
        self.model_accepts_loss_kwargs = False

        self._validate_runtime()
        functional_model = self.accelerator.unwrap_model(self.model)
        admissible = select_lora_parameters(
            functional_model,
            target_modules=self.cbg_args.lora_target_modules,
            trainable_only=True,
        )
        self._admissible_names = tuple(admissible)
        all_trainable = tuple(
            name for name, parameter in functional_model.named_parameters() if parameter.requires_grad
        )
        if set(all_trainable) != set(self._admissible_names):
            unexpected = sorted(set(all_trainable) - set(self._admissible_names))
            missing = sorted(set(self._admissible_names) - set(all_trainable))
            raise ValueError(
                "CBG requires every and only configured LoRA coordinates to be trainable; "
                f"unexpected={unexpected}, missing={missing}"
            )
        non_fp32 = [
            name
            for name, parameter in admissible.items()
            if parameter.dtype not in (torch.float32, torch.float64)
        ]
        if non_fp32:
            raise ValueError(f"CBG needs FP32/FP64 adapter leaves for finite differences: {non_fp32}")
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
                reference_model.to(self.args.device)
            except (RuntimeError, ValueError) as error:
                raise ValueError(
                    "the separate reference model could not be moved to the Trainer device"
                ) from error

    def _validate_runtime(self) -> None:
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
        with torch.no_grad():
            return select_response_logits(_extract_logits(self.reference_model(**kwargs)), benign_batch).detach()

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
            value = float(calibrated.float().item())
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

        # A fixed deterministic map is essential for reusing metric products.
        # Disable stochastic layers for all CBG evaluations, without disabling
        # gradients. Avoid reusing autocast weight casts across JVP/no_grad/VJP.
        with _temporary_eval(functional_model), self.compute_loss_context_manager(), torch.autocast(
            device_type=self.args.device.type, enabled=self.args.bf16,
            dtype=torch.bfloat16, cache_enabled=False,
        ):
            if not preservation_only:
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
                )
                mu = self._resolve_mu(benign_functional, parameters)

                adaptive = self.cbg_args.inner_policy == "adaptive"
                minimum_steps = (
                    self.cbg_args.inner_initial_steps if adaptive and self.prev_u is None
                    else self.cbg_args.inner_steps
                )
                rayleigh = warm_started_rayleigh_ascent(
                    safety_functional,
                    benign_functional,
                    parameters,
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
                if not rayleigh.diagnostics.converged:
                    if self.cbg_args.inner_fail_on_nonconvergence:
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

                spectral = centered_spectral_gradient(
                    safety_functional,
                    benign_functional,
                    parameters,
                    direction,
                    mu=mu,
                    tau_fd=self.cbg_args.tau_fd,
                    theta_floor=self.cbg_args.theta_floor,
                    u_floor=self.cbg_args.u_floor,
                    same_randomness=True,
                    center=rayleigh.center,
                    retain_endpoint_gradients=False,
                )

            # Preservation losses are deliberately distinct from the two
            # functional metrics used above.
            alignment_loss, alignment_gradient, benign_loss, benign_gradient = self._preservation_gradients(
                model, parameters, safe_batch, benign_batch,
            )

        if preservation_only:
            spectral_gradient = tree_zeros_like(parameters)
            effective_beta = 0.0
            training_gradient = compose_training_gradient(
                alignment_gradient,
                benign_gradient,
                spectral_gradient,
                alpha=self.cbg_args.alpha,
                beta=0.0,
            )
        else:
            spectral_gradient = spectral.gradient
            preservation_gradient = compose_training_gradient(
                alignment_gradient,
                benign_gradient,
                spectral_gradient,
                alpha=self.cbg_args.alpha,
                beta=0.0,
            )
            effective_beta = self._effective_beta(
                preservation_gradient,
                spectral_gradient,
            )
            training_gradient = (
                preservation_gradient if effective_beta == 0 else
                tree_add(preservation_gradient, spectral_gradient, alpha=effective_beta)
            )

        preservation_norm = tree_norm(tree_add(alignment_gradient, benign_gradient, alpha=self.cbg_args.alpha))
        cbg_norm = tree_norm(spectral_gradient)
        # Check before backward so a numerical failure cannot poison optimizer state.
        if not bool(torch.isfinite(tree_norm(training_gradient))):
            raise FloatingPointError("non-finite composed CBG gradient")

        divisor = self._gradient_accumulation_divisor()
        self._backward_explicit_gradient(
            parameters,
            training_gradient,
            divisor=divisor,
        )

        # A failed spectral or preservation computation never changes the warm
        # start.  Persist only a direction from a fully successful backward.
        if not preservation_only:
            self.prev_u = tree_detach(direction)

        if memory_device is not None:
            # Most geometry diagnostics already synchronize, but the final
            # proxy backward may still be queued when wall time is sampled.
            torch.cuda.synchronize(memory_device)
        metrics: dict[str, float | Tensor] = {
            "train/l_aln": alignment_loss.detach(),
            "train/l_bgn": benign_loss.detach(),
            "train/grad_norm_aln": tree_norm(alignment_gradient),
            "train/grad_norm_bgn": tree_norm(benign_gradient),
            "train/grad_norm_cbg": cbg_norm,
            "train/grad_norm_alpha_bgn": self.cbg_args.alpha * tree_norm(benign_gradient),
            "train/grad_norm_beta_cbg": effective_beta * cbg_norm,
            "train/grad_norm_preservation": preservation_norm,
            "train/cbg_preservation_ratio": effective_beta * cbg_norm / preservation_norm.clamp_min(1e-12),
            "cbg/enabled": float(not preservation_only),
            "train/beta_effective": effective_beta,
            "system/step_time": time.perf_counter() - step_started,
            "system/max_memory_allocated": (
                float(torch.cuda.max_memory_allocated(memory_device))
                if memory_device is not None
                else 0.0
            ),
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
            rayleigh_values = rayleigh.diagnostics.rayleigh_values
            metrics.update(
                {
                    "train/rho_hat": spectral.center.rho,
                    "train/qh_u": spectral.center.qh,
                    "train/qb_u": spectral.center.qb,
                    "train/qb_damped_u": spectral.center.b,
                    "cbg/mu": mu,
                    "cbg/delta": spectral.delta,
                    "cbg/u_norm": rayleigh.diagnostics.u_euclidean_norm,
                    "cbg/b_norm": rayleigh.diagnostics.b_mu_norm,
                    "cbg/eig_residual": rayleigh.diagnostics.eig_residual_norm,
                    "cbg/relative_residual": rayleigh.diagnostics.relative_residual,
                    "cbg/converged": float(rayleigh.diagnostics.converged),
                    "cbg/inner_steps": rayleigh.diagnostics.inner_steps,
                    "cbg/rejected_steps": rayleigh.diagnostics.rejected_steps,
                    "cbg/metric_evaluations": rayleigh.diagnostics.metric_evaluations,
                    "cbg/fd_cancellation": spectral.fd_cancellation,
                    "cbg/rayleigh_start": rayleigh_values[0],
                    "cbg/rayleigh_final": rayleigh_values[-1],
                    "cbg/warm_start_cosine": (
                        rayleigh.diagnostics.warm_start_cosine
                        if rayleigh.diagnostics.warm_start_cosine is not None
                        else 0.0
                    ),
                }
            )
            for index, value in enumerate(rayleigh_values):
                metrics[f"cbg/rayleigh_iter_{index}"] = value
        self._record_metrics(metrics)

        reported_loss = alignment_loss.detach() + self.cbg_args.alpha * benign_loss.detach()
        return reported_loss / divisor

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
            if key == "system/max_memory_allocated":
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
            "calibrated_mu": self.calibrated_mu,
            "beta_candidate_sum": self._beta_candidate_sum,
            "beta_candidate_count": self._beta_candidate_count,
            "calibrated_beta": self._calibrated_beta,
            "config": config,
            "config_fingerprint": fingerprint,
            "data_manifest": getattr(self, "data_manifest", None),
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
                restored[name] = value.to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
            self.prev_u = tree_detach(restored)

        calibrated_mu = state.get("calibrated_mu")
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
