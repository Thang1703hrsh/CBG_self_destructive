"""V5 sections 5--7: bounded tree Ritz search and streamed complement bound.

The enclosure is valid for exact operators/B-orthogonality; floating-point
results are numerical estimates, not interval-arithmetic certificates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

from src.cbg.dual import _layout, _validate_precision, dual_rayleigh_solve
from src.cbg.geometry import MetricProduct, normalized_functional_rows
from src.cbg.rayleigh import RayleighDiagnostics, RayleighResult
from src.cbg.spectral_grad import compute_spectral_center
from src.cbg.tree_ops import (
    assert_same_structure, tree_add, tree_cosine_similarity, tree_detach,
    tree_dot, tree_gram, tree_linear_combination, tree_norm, tree_random_like,
    tree_scale, tree_sub,
    TensorTree,
)


@dataclass(frozen=True)
class ComplementEnclosure:
    lower: torch.Tensor
    upper: torch.Tensor
    eta: torch.Tensor
    gamma: torch.Tensor
    orthogonality: torch.Tensor
    row_candidate: TensorTree | None
    row_pullbacks: int


def _ritz(centers, tolerance):
    vectors = [c.direction for c in centers]
    b_vectors = [c.b_direction for c in centers]
    gram = tree_gram(vectors, b_vectors).double()
    identity = torch.eye(len(centers), device=gram.device, dtype=gram.dtype)
    orthogonality = torch.linalg.matrix_norm(gram - identity)
    if not bool(torch.isfinite(gram).all()) or bool(orthogonality > tolerance):
        raise FloatingPointError("V5 basis failed current B-orthogonality")
    h = tree_gram(vectors, [c.gh_direction for c in centers]).double()
    scale = torch.linalg.matrix_norm(h).clamp_min(torch.finfo(h.dtype).tiny)
    if not bool(torch.isfinite(h).all()) or bool(torch.linalg.matrix_norm(h-h.T) / scale > tolerance):
        raise FloatingPointError("V5 Ritz matrix is nonfinite or asymmetric")
    h = (h + h.T) / 2
    values, coefficients = torch.linalg.eigh(h)
    if bool(values[0] < -tolerance * values.abs().max().clamp_min(1e-30)):
        raise FloatingPointError("V5 Ritz safety matrix is materially non-PSD")
    return h, values, coefficients, orthogonality


@torch.no_grad()
def _enclosure(safety, parameters, centers, mu, tolerance):
    h, values, _, orthogonality = _ritz(centers, tolerance)
    vectors = [c.direction for c in centers]
    b_vectors = [c.b_direction for c in centers]
    residual_squared = h.new_zeros(())
    for i, center in enumerate(centers):
        residual = tree_sub(center.gh_direction, tree_linear_combination(b_vectors, h[:, i]))
        residual_squared += tree_dot(residual, residual).double()
        del residual
    e_squared = h.new_zeros(())
    candidate, maximum, count = None, -1., 0
    for row in normalized_functional_rows(safety, parameters):
        coefficients = torch.stack([tree_dot(v, row) for v in vectors])
        e = tree_sub(row, tree_linear_combination(b_vectors, coefficients))
        square = tree_dot(e, e).double()
        if not bool(torch.isfinite(square)):
            raise FloatingPointError("non-finite V5 complement row")
        e_squared += square
        if float(square) > maximum:
            candidate, maximum = tree_detach(e), float(square)
        count += 1
        del row, e
    eta, gamma = (residual_squared / mu).sqrt(), e_squared / mu
    lower = values[-1]
    upper = (lower + gamma + torch.hypot(lower - gamma, 2 * eta)) / 2
    if not bool(torch.isfinite(upper)) or bool(upper < lower):
        raise FloatingPointError("invalid V5 complement enclosure")
    return ComplementEnclosure(lower, upper, eta, gamma, orthogonality, candidate, count)


def compute_complement_enclosure(safety, benign, parameters, basis, *, mu, tolerance=1e-5):
    """Diagnostic API: independently recompute products for a B-normalized basis."""
    _validate_precision((safety, benign), parameters)
    if not math.isfinite(float(mu)) or float(mu) <= 0 or not basis:
        raise ValueError("positive finite mu and nonempty basis are required")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    with torch.no_grad():
        centers = [compute_spectral_center(safety, benign, parameters, tree_detach(v), mu=mu)
                   for v in basis]
        return _enclosure(safety, parameters, centers, float(mu), tolerance)


def _combine_center(safety, benign, parameters, centers, coefficients, mu):
    direction = tree_detach(tree_linear_combination([c.direction for c in centers], coefficients))
    hp = MetricProduct(tree_linear_combination([c.gh_direction for c in centers], coefficients),
                       sum(a.to(c.zh) * c.zh for a, c in zip(coefficients, centers)))
    bp = MetricProduct(tree_linear_combination([c.gb_direction for c in centers], coefficients),
                       sum(a.to(c.zb) * c.zb for a, c in zip(coefficients, centers)))
    return compute_spectral_center(safety, benign, parameters, direction, mu=mu,
                                   safety_product=hp, benign_product=bp)


def _orthogonal_center(safety, benign, parameters, candidate, centers, mu, dependency_tolerance):
    """Two-pass modified Gram-Schmidt, reusing linear metric/JVP products."""
    assert_same_structure(parameters, candidate)
    if not bool(torch.isfinite(tree_norm(candidate))):
        raise FloatingPointError("non-finite V5 basis candidate")
    if bool(tree_norm(candidate) == 0):
        return None, 0
    raw = compute_spectral_center(safety, benign, parameters, tree_detach(candidate), mu=mu)
    initial_b = raw.b
    u, gh, gb, zh, zb = raw.direction, raw.gh_direction, raw.gb_direction, raw.zh, raw.zb
    for _ in range(2):
        for previous in centers:
            coefficient = tree_dot(previous.b_direction, u)
            u = tree_add(u, previous.direction, alpha=-coefficient)
            gh = tree_add(gh, previous.gh_direction, alpha=-coefficient)
            gb = tree_add(gb, previous.gb_direction, alpha=-coefficient)
            zh, zb = zh - coefficient * previous.zh, zb - coefficient * previous.zb
    b = tree_dot(u, tree_add(gb, u, alpha=mu))
    if not bool(torch.isfinite(b)):
        raise FloatingPointError("non-finite B norm during orthogonalization")
    if bool(b <= dependency_tolerance * initial_b):
        return None, 2
    scale = b.rsqrt()
    u = tree_detach(tree_scale(u, scale))
    normalized = compute_spectral_center(safety, benign, parameters, u, mu=mu,
        safety_product=MetricProduct(tree_scale(gh, scale), zh * scale),
        benign_product=MetricProduct(tree_scale(gb, scale), zb * scale))
    return normalized, 2


@torch.no_grad()
def bounded_rayleigh_solve(
    safety, benign, parameters, *, mu, warm_start=None, warm_basis=(),
    initial_size=2, max_size=6, max_enrichments=4, residual_tolerance=1e-4,
    orthogonality_tolerance=1e-5, enclosure_atol=1e-6, enclosure_rtol=1e-3,
    rho_scale=1., dependency_tolerance=1e-10, dual_max_size=64,
    dual_tolerance=1e-4, dual_kernel_tolerance=1e-5, generator=None, dual_column_batch_size=1,
    dual_refinement_steps=0, dual_residual_tolerance=None,
):
    """Bounded V5 search, with mandatory same-batch verified dual fallback.

    P=I residual preconditioner. Only detached warm vectors cross steps;
    all products, orthogonalization and bounds are recomputed at entry.
    """
    _validate_precision((safety, benign), parameters)
    for name, value in (("initial_size", initial_size), ("max_size", max_size),
                        ("max_enrichments", max_enrichments)):
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == "max_enrichments" else 1):
            raise ValueError(f"invalid {name}")
    if max_size < max(initial_size, 2):
        raise ValueError("max_size must cover initial_size and a correction vector")
    for name, value in (("mu", mu), ("residual_tolerance", residual_tolerance),
                        ("orthogonality_tolerance", orthogonality_tolerance),
                        ("rho_scale", rho_scale), ("dependency_tolerance", dependency_tolerance)):
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if any(not math.isfinite(x) or x < 0 for x in (enclosure_atol, enclosure_rtol)):
        raise ValueError("enclosure tolerances must be finite and nonnegative")
    mu = float(mu)
    n = sum(sum(_layout(f, parameters)[1]) for f in (safety, benign))
    if n > dual_max_size:
        raise ValueError("V5 same-batch dual fallback exceeds dual_max_size")
    centers, metric_calls, checks, rows, history = [], 0, 0, 0, []
    last_bound = None
    fallback_reason = "work_budget"
    warm = tuple(warm_basis) or (() if warm_start is None else (warm_start,))
    # Do not consume checkpoint RNG when a sufficient warm basis is available.
    for candidate in warm[:max_size]:
        center, cost = _orthogonal_center(safety, benign, parameters, candidate, centers, mu, dependency_tolerance)
        metric_calls += cost
        if center is not None:
            centers.append(center)
    for _ in range(initial_size * 3):
        if len(centers) >= min(initial_size, sum(v.numel() for v in parameters.values())):
            break
        candidate = tree_random_like(parameters, generator=generator)
        center, cost = _orthogonal_center(safety, benign, parameters, candidate, centers, mu, dependency_tolerance)
        metric_calls += cost
        if center is not None:
            centers.append(center)
    if not centers:
        raise FloatingPointError("unable to construct V5 B-normalized search basis")
    for iteration in range(max_enrichments + 1):
        try:
            _, values, coefficients, orth = _ritz(centers, orthogonality_tolerance)
        except FloatingPointError:
            fallback_reason = "ritz_numerics"
            break  # numerical subspace failure must use the full dual oracle
        leading = _combine_center(safety, benign, parameters, centers, coefficients[:, -1], mu)
        residual = tree_sub(leading.gh_direction, tree_scale(leading.b_direction, leading.rho))
        relative = tree_norm(residual) / (tree_norm(leading.gh_direction) +
                    leading.rho.abs() * tree_norm(leading.b_direction)).clamp_min(1e-30)
        history.append(leading.rho)
        bound = None
        if bool(relative <= residual_tolerance):
            bound = _enclosure(safety, parameters, centers, mu, orthogonality_tolerance)
            last_bound, checks, rows = bound, checks + 1, rows + bound.row_pullbacks
            accepted = bool(bound.upper - bound.lower <=
                enclosure_atol + enclosure_rtol * bound.upper.clamp_min(rho_scale))
            if accepted and bool(leading.rho > 0):
                # Independent products verify the returned direction, not just caches.
                direction = leading.direction
                if warm_start is not None and bool(tree_dot(direction, warm_start) < 0):
                    direction = tree_scale(direction, -1.)
                verified = compute_spectral_center(safety, benign, parameters, direction, mu=mu)
                metric_calls += 2
                r = tree_sub(verified.gh_direction, tree_scale(verified.b_direction, verified.rho))
                rn = tree_norm(r)
                rr = rn / (tree_norm(verified.gh_direction) + verified.rho.abs() * tree_norm(verified.b_direction)).clamp_min(1e-30)
                if (not bool(torch.isfinite(rr)) or bool(rr > residual_tolerance)
                    or bool((verified.b - 1).abs() > orthogonality_tolerance)
                    or bool((verified.rho - bound.lower).abs() > enclosure_atol + enclosure_rtol * bound.upper.clamp_min(rho_scale))):
                    fallback_reason = "independent_verification"
                    break  # independent validation failure: dual fallback
                diagnostics = RayleighDiagnostics(
                    rho_hat=verified.rho, qh=verified.qh, qb=verified.qb, qb_damped=verified.b,
                    eig_residual_norm=rn, u_euclidean_norm=tree_norm(direction), b_mu_norm=verified.b.sqrt(),
                    rayleigh_values=tuple(history), warm_start_cosine=None if warm_start is None else tree_cosine_similarity(direction, warm_start).abs(),
                    inner_steps=iteration, mu=verified.b.new_tensor(mu), relative_residual=rr,
                    converged=True, rejected_steps=0, metric_evaluations=metric_calls,
                    # A Ritz gap is not a verified full-space eigengap; leave it unset.
                    damping_fraction=mu * tree_dot(direction, direction) / verified.b,
                    enclosure_lower=bound.lower, enclosure_upper=bound.upper,
                    enclosure_eta=bound.eta, enclosure_gamma=bound.gamma,
                    b_orthogonality=orth, basis_size=len(centers), enclosure_checks=checks,
                    safety_row_pullbacks=rows,
                    ritz_eigengap=values[-1] - values[-2] if len(values) > 1 else None,
                )
                return RayleighResult(verified.direction, diagnostics, verified,
                                      tuple(tree_detach(c.direction) for c in centers))
        if iteration == max_enrichments:
            break
        candidate = bound.row_candidate if bound is not None else residual
        # Restart with the leading Ritz vector; the omitted-spectrum gate must
        # be recomputed for the new space before accepting any result.
        if len(centers) >= max_size:
            centers = [leading]
        center, cost = _orthogonal_center(safety, benign, parameters, candidate, centers, mu, dependency_tolerance)
        metric_calls += cost
        if center is None:
            fallback_reason = "dependent_correction"
            break
        centers.append(center)
    changes = dict(dual_fallback=True, fallback_reason=fallback_reason,
                   basis_size=len(centers), enclosure_checks=checks,
                   safety_row_pullbacks=rows)
    if last_bound is not None:
        changes.update(enclosure_lower=last_bound.lower, enclosure_upper=last_bound.upper,
                       enclosure_eta=last_bound.eta, enclosure_gamma=last_bound.gamma,
                       b_orthogonality=last_bound.orthogonality)
    # Preserve scalar diagnostics only. Retaining failed-search centers here
    # makes the full dual fallback coexist with every basis metric tree.
    centers.clear()
    center = leading = residual = candidate = verified = r = direction = None
    bound = last_bound = None
    result = dual_rayleigh_solve(safety, benign, parameters, mu=mu, warm_start=warm_start,
        max_size=dual_max_size, tolerance=dual_tolerance, kernel_tolerance=dual_kernel_tolerance,
        residual_tolerance=dual_residual_tolerance,
        column_batch_size=dual_column_batch_size, refinement_steps=dual_refinement_steps)
    changes["metric_evaluations"] = metric_calls + result.diagnostics.metric_evaluations
    return replace(result, diagnostics=replace(result.diagnostics, **changes),
                   warm_basis=() if result.center is None else (tree_detach(result.direction),))
