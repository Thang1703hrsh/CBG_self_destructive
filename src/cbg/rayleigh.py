"""Warm-started generalized Rayleigh ascent for the CBG weakest escape."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from src.cbg.geometry import FunctionalMap, MetricProduct, functional_jvp, metric_vector_product
from src.cbg.spectral_grad import SpectralCenter, compute_spectral_center
from src.cbg.tree_ops import (
    OrderedTensorTree,
    TensorTree,
    assert_same_structure,
    tree_add,
    tree_cosine_similarity,
    tree_detach,
    tree_dot,
    tree_norm,
    tree_random_like,
    tree_scale,
    tree_sub,
)


@dataclass(frozen=True)
class RayleighDiagnostics:
    """Detached diagnostics for one weakest-escape search."""

    rho_hat: Tensor
    qh: Tensor
    qb: Tensor
    qb_damped: Tensor
    eig_residual_norm: Tensor
    u_euclidean_norm: Tensor
    b_mu_norm: Tensor
    rayleigh_values: tuple[Tensor, ...]
    warm_start_cosine: Tensor | None
    inner_steps: int
    mu: Tensor
    relative_residual: Tensor
    converged: bool
    rejected_steps: int
    metric_evaluations: int
    zero_safety: bool = False
    eigengap: Tensor | None = None
    kernel_asymmetry: Tensor | None = None
    kernel_projection: Tensor | None = None
    kernel_columns: int = 0
    damping_fraction: Tensor | None = None
    dual_value_error: Tensor | None = None
    enclosure_lower: Tensor | None = None
    enclosure_upper: Tensor | None = None
    enclosure_eta: Tensor | None = None
    enclosure_gamma: Tensor | None = None
    b_orthogonality: Tensor | None = None
    basis_size: int = 0
    enclosure_checks: int = 0
    safety_row_pullbacks: int = 0
    dual_fallback: bool = False
    ritz_eigengap: Tensor | None = None
    fallback_reason: str | None = None
    refinement_steps: int = 0
    refinement_jvps: int = 0


@dataclass(frozen=True)
class RayleighResult:
    direction: OrderedTensorTree
    diagnostics: RayleighDiagnostics
    center: SpectralCenter | None
    warm_basis: tuple[OrderedTensorTree, ...] = ()


def calibrate_relative_damping(
    benign_functional: FunctionalMap,
    parameters: TensorTree,
    *,
    damping_fraction: float = 0.05,
    num_directions: int = 8,
    mu_floor: float = 1e-8,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Calibrate damping from random admissible Euclidean-unit directions."""

    _validate_positive_scalar(damping_fraction, name="damping_fraction")
    _validate_positive_scalar(mu_floor, name="mu_floor")
    if num_directions < 1:
        raise ValueError("num_directions must be at least one")

    costs: list[Tensor] = []
    for _ in range(num_directions):
        direction = _random_unit_direction(parameters, generator=generator, eps=1e-15)
        # v^T G_b v = ||J_b v||^2 / B; no VJP is needed for calibration.
        with torch.no_grad():
            _, jv = functional_jvp(benign_functional, parameters, direction)
            work = jv.double() if jv.dtype == torch.float64 else jv.float()
            costs.append(work.square().mean())
    median_cost = torch.median(torch.stack(costs))
    calibrated = damping_fraction * torch.clamp_min(median_cost, 0.0)
    return torch.clamp_min(calibrated, mu_floor).detach()


def _validate_positive_scalar(value: float | Tensor, *, name: str) -> None:
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be a scalar")
        if not bool(torch.isfinite(value.detach()).item()):
            raise ValueError(f"{name} must be finite")
        if not bool((value.detach() > 0).item()):
            raise ValueError(f"{name} must be strictly positive")
    elif not torch.isfinite(torch.tensor(value)) or value <= 0:
        raise ValueError(f"{name} must be finite and strictly positive")


def _mu_tensor(mu: float | Tensor, reference: Tensor) -> Tensor:
    return torch.as_tensor(mu, device=reference.device, dtype=reference.dtype)


def _metric_quantities(
    safety_functional: FunctionalMap,
    benign_functional: FunctionalMap,
    parameters: TensorTree,
    direction: TensorTree,
    mu: float | Tensor,
) -> tuple[OrderedTensorTree, OrderedTensorTree, OrderedTensorTree, Tensor, Tensor, Tensor, Tensor]:
    gh_direction = metric_vector_product(
        safety_functional,
        parameters,
        direction,
        detach_result=True,
    )
    gb_direction = metric_vector_product(
        benign_functional,
        parameters,
        direction,
        detach_result=True,
    )
    b_direction = tree_add(gb_direction, tree_scale(direction, mu))
    qh = tree_dot(direction, gh_direction)
    qb = tree_dot(direction, gb_direction)
    qb_damped = tree_dot(direction, b_direction)
    if not bool(torch.isfinite(qb_damped.detach()).item()) or not bool(
        (qb_damped.detach() > 0).item()
    ):
        raise FloatingPointError("u^T B_mu u must be finite and strictly positive")
    rho = qh / qb_damped
    return gh_direction, gb_direction, b_direction, qh, qb, qb_damped, rho


def generalized_rayleigh_quotient(
    safety_functional: FunctionalMap,
    benign_functional: FunctionalMap,
    parameters: TensorTree,
    direction: TensorTree,
    mu: float | Tensor,
) -> Tensor:
    """Return ``(u^T G_h u) / (u^T B_mu u)``."""

    _validate_positive_scalar(mu, name="mu")
    assert_same_structure(parameters, direction)
    *_, rho = _metric_quantities(
        safety_functional,
        benign_functional,
        parameters,
        direction,
        mu,
    )
    return rho


def normalize_b_metric(
    benign_functional: FunctionalMap,
    parameters: TensorTree,
    direction: TensorTree,
    mu: float | Tensor,
    *,
    eps_norm: float = 1e-12,
) -> OrderedTensorTree:
    """Normalize a direction to unit regularized benign cost."""

    _validate_positive_scalar(mu, name="mu")
    if eps_norm < 0:
        raise ValueError("eps_norm must be non-negative")
    assert_same_structure(parameters, direction)
    gb_direction = metric_vector_product(
        benign_functional,
        parameters,
        direction,
        detach_result=True,
    )
    qb_damped = tree_dot(direction, gb_direction) + mu * tree_dot(direction, direction)
    if not bool(torch.isfinite(qb_damped.detach()).item()) or not bool(
        (qb_damped.detach() > 0).item()
    ):
        raise FloatingPointError("cannot B-normalize a direction with non-positive cost")
    denominator = torch.sqrt(qb_damped + eps_norm)
    return tree_detach(tree_scale(direction, denominator.reciprocal()))


def _random_unit_direction(
    parameters: TensorTree,
    *,
    generator: torch.Generator | None,
    eps: float,
) -> OrderedTensorTree:
    direction = tree_random_like(parameters, generator=generator)
    norm = tree_norm(direction)
    if not bool(torch.isfinite(norm.detach()).item()) or not bool((norm.detach() > eps).item()):
        raise FloatingPointError("random admissible direction has invalid Euclidean norm")
    return tree_detach(tree_scale(direction, norm.reciprocal()))


def warm_started_rayleigh_ascent(
    safety_functional: FunctionalMap,
    benign_functional: FunctionalMap,
    parameters: TensorTree,
    *,
    mu: float | Tensor,
    inner_steps: int = 2,
    inner_lr: float = 1e-2,
    warm_start: TensorTree | None = None,
    eps_norm: float = 1e-12,
    generator: torch.Generator | None = None,
    max_steps: int | None = None,
    residual_tolerance: float = 0.05,
    rayleigh_tolerance: float = 0.01,
    max_backtracks: int = 0,
) -> RayleighResult:
    """Search for the practical weakest-escape direction.

    Fixed mode (max_steps=None, max_backtracks=0) is algebraically the guide's
    update. Products at an unnormalized candidate scale linearly, so they are
    reused after B normalization and at the next iteration/outer derivative.
    Optional bounded adaptive mode stops only when BOTH relative residual and
    relative Rayleigh change pass. Backtracking rejects decreasing candidates.
    """

    _validate_positive_scalar(mu, name="mu")
    _validate_positive_scalar(inner_lr, name="inner_lr")
    if inner_steps < 1:
        raise ValueError("inner_steps must be at least one")
    if eps_norm < 0:
        raise ValueError("eps_norm must be non-negative")
    budget = inner_steps if max_steps is None else max_steps
    if budget < inner_steps or max_backtracks < 0:
        raise ValueError("max_steps must be >= inner_steps and max_backtracks >= 0")
    _validate_positive_scalar(residual_tolerance, name="residual_tolerance")
    _validate_positive_scalar(rayleigh_tolerance, name="rayleigh_tolerance")

    if warm_start is None:
        direction = _random_unit_direction(
            parameters,
            generator=generator,
            eps=max(eps_norm, 1e-15),
        )
        previous_direction = None
    else:
        assert_same_structure(parameters, warm_start)
        direction = tree_detach(warm_start)
        if not bool((tree_norm(direction).detach() > 0).item()):
            raise ValueError("warm_start must have non-zero Euclidean norm")
        previous_direction = tree_detach(warm_start)

    center = compute_spectral_center(safety_functional, benign_functional, parameters, direction, mu=mu)
    rayleigh_values = [center.rho]
    evaluations, rejected, accepted = 2, 0, 0
    converged = False
    for _ in range(budget):
        gradient = tree_scale(
            tree_sub(center.gh_direction, tree_scale(center.b_direction, center.rho)),
            2.0 / center.b,
        )
        step_lr = inner_lr
        next_center = None
        for attempt in range(max_backtracks + 1):
            candidate = tree_detach(tree_add(direction, gradient, alpha=step_lr))
            trial = compute_spectral_center(
                safety_functional, benign_functional, parameters, candidate, mu=mu
            )
            evaluations += 2
            if not bool(torch.isfinite(trial.rho)):
                raise FloatingPointError("non-finite Rayleigh candidate")
            if max_backtracks and bool(trial.rho < center.rho):
                rejected += 1
                step_lr *= 0.5
                del trial
                continue
            scale = (trial.b + eps_norm).rsqrt()
            candidate = tree_detach(tree_scale(candidate, scale))
            next_center = compute_spectral_center(
                safety_functional, benign_functional, parameters, candidate, mu=mu,
                safety_product=MetricProduct(tree_scale(trial.gh_direction, scale), trial.zh * scale),
                benign_product=MetricProduct(tree_scale(trial.gb_direction, scale), trial.zb * scale),
            )
            del trial
            break
        if next_center is None:
            converged = False
            break  # No acceptable trial within the budget: report non-convergence.
        gain = (next_center.rho - center.rho).abs() / center.rho.abs().clamp_min(1e-12)
        center, direction = next_center, candidate
        accepted += 1
        rayleigh_values.append(center.rho)
        residual = tree_sub(center.gh_direction, tree_scale(center.b_direction, center.rho))
        relative_residual = tree_norm(residual) / (
            tree_norm(center.gh_direction) + center.rho.abs() * tree_norm(center.b_direction)
        ).clamp_min(1e-12)
        converged = bool(relative_residual <= residual_tolerance) and bool(gain <= rayleigh_tolerance)
        if max_steps is not None and accepted >= inner_steps and converged:
            break

    residual = tree_sub(center.gh_direction, tree_scale(center.b_direction, center.rho))
    relative_residual = tree_norm(residual) / (
        tree_norm(center.gh_direction) + center.rho.abs() * tree_norm(center.b_direction)
    ).clamp_min(1e-12)
    warm_start_cosine = None
    if previous_direction is not None:
        warm_start_cosine = tree_cosine_similarity(direction, previous_direction).detach()

    mu_value = _mu_tensor(mu, center.b).detach()
    diagnostics = RayleighDiagnostics(
        rho_hat=center.rho,
        qh=center.qh,
        qb=center.qb,
        qb_damped=center.b,
        eig_residual_norm=tree_norm(residual).detach(),
        u_euclidean_norm=tree_norm(direction).detach(),
        b_mu_norm=torch.sqrt(torch.clamp_min(center.b, 0.0)),
        rayleigh_values=tuple(rayleigh_values),
        warm_start_cosine=warm_start_cosine,
        inner_steps=accepted,
        mu=mu_value,
        relative_residual=relative_residual.detach(),
        converged=converged,
        rejected_steps=rejected,
        metric_evaluations=evaluations,
    )
    return RayleighResult(direction=tree_detach(direction), diagnostics=diagnostics, center=center)
