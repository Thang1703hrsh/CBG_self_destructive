"""Small functional-space CBG solver with a PSD-factor reconstruction.

Only sample-space kernels/factors are dense. Parameter vectors remain trees,
with a bounded block of transient VJP trees. No full Jacobian is assembled.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from src.cbg.geometry import (
    FunctionalMap, block_vjp, functional_blocks, functional_jvp, functional_vjp,
    functional_jvp_directions,
)
from src.cbg.rayleigh import RayleighDiagnostics, RayleighResult
from src.cbg.spectral_grad import compute_spectral_center, rescale_owned_spectral_center_
from src.cbg.tree_ops import (
    TensorTree, assert_same_structure, tree_add, tree_cosine_similarity,
    tree_dot, tree_norm, tree_scale, tree_zeros_like,
    accumulate_detached_gradient, tree_dot_fp64_detached, tree_linear_combination,
)


@dataclass(frozen=True)
class FunctionalKernel:
    values: Tensor
    safety_size: int
    benign_size: int


@dataclass(frozen=True)
class _RefinementState:
    """Detached B-orthonormal Ritz basis and its cached responses.

    The basis is deliberately bounded by ``refinement_steps + 1``.  Retaining
    earlier accepted residual directions avoids the two-vector restart that
    can stagnate on a clustered or ill-conditioned leading eigenspace.
    """

    vectors: tuple[TensorTree, ...]
    safety_responses: tuple[Tensor, ...]
    benign_responses: tuple[Tensor, ...]


def _relative_residual(center):
    residual = tree_add(center.gh_direction, center.b_direction, alpha=-center.rho)
    norm = tree_norm(residual)
    relative = norm / (tree_norm(center.gh_direction) + center.rho.abs() *
                       tree_norm(center.b_direction)).clamp_min(torch.finfo(center.rho.dtype).tiny)
    return residual, norm, relative


def _basis_gram(vectors, responses, *, mu=0.0):
    """Small FP64 Gram matrix, with parameter trees kept in native dtype."""

    response_matrix = torch.stack(responses, dim=1).double()
    gram = response_matrix.T @ response_matrix / response_matrix.shape[0]
    if mu:
        euclidean = torch.stack([
            torch.stack([tree_dot_fp64_detached(left, right) for right in vectors])
            for left in vectors
        ])
        gram = gram + mu * euclidean
    return (gram + gram.T) / 2


def _b_cross(state, direction, benign_response, mu):
    """Return V^T B d using cached J_b V and one new J_b d."""

    zb = torch.stack(state.benign_responses, dim=1).double()
    response = benign_response.double()
    functional = zb.T @ response / response.numel()
    euclidean = torch.stack([
        tree_dot_fp64_detached(vector, direction) for vector in state.vectors
    ])
    return functional + mu * euclidean


def _orthogonalize_correction(state, direction, zh, zb, mu):
    """Twice B-orthogonalize a candidate using only small solves/tree axpys."""

    basis_b = _basis_gram(state.vectors, state.benign_responses, mu=mu)
    chol, info = torch.linalg.cholesky_ex(basis_b)
    if bool(info != 0):
        return None
    for _ in range(2):
        cross = _b_cross(state, direction, zb, mu)
        coefficients = torch.cholesky_solve(cross[:, None], chol).flatten()
        projection = tree_linear_combination(state.vectors, coefficients)
        direction = tree_add(direction, projection, alpha=-1.0)
        del projection
        safety_projection = torch.stack(state.safety_responses, dim=1).double() @ coefficients
        benign_projection = torch.stack(state.benign_responses, dim=1).double() @ coefficients
        zh = zh - safety_projection.to(zh)
        zb = zb - benign_projection.to(zb)
    b_norm_sq = (
        torch.dot(zb.double(), zb.double()) / zb.numel()
        + mu * tree_dot_fp64_detached(direction, direction)
    )
    if not bool(torch.isfinite(b_norm_sq)) or not bool(b_norm_sq > 0):
        return None
    scale = b_norm_sq.rsqrt()
    direction = tree_scale(direction, scale)
    return direction, zh * scale.to(zh), zb * scale.to(zb)


def _refine_center_with_state(safety, benign, parameters, center, mu, state):
    """Append one residual direction and solve over the cumulative Ritz basis."""

    with torch.no_grad():
        residual, _, _ = _relative_residual(center)
        norm = tree_norm(residual)
        if not bool(torch.isfinite(norm)) or not bool(norm > 0):
            return None, state
        correction = tree_scale(residual, norm.reciprocal())
        del residual
        zh = functional_jvp(safety, parameters, correction)[1]
        zb = functional_jvp(benign, parameters, correction)[1]
        orthogonalized = _orthogonalize_correction(state, correction, zh, zb, mu)
        del correction, zh, zb
        if orthogonalized is None:
            return None, state
        correction, zh, zb = orthogonalized
        expanded = _RefinementState(
            vectors=state.vectors + (correction,),
            safety_responses=state.safety_responses + (zh,),
            benign_responses=state.benign_responses + (zb,),
        )
        h = _basis_gram(expanded.vectors, expanded.safety_responses)
        b = _basis_gram(expanded.vectors, expanded.benign_responses, mu=mu)
        chol, info = torch.linalg.cholesky_ex(b)
        if bool(info != 0):
            return None, state
        left = torch.linalg.solve_triangular(chol, h, upper=False)
        reduced = torch.linalg.solve_triangular(chol, left.T, upper=False).T
        _, eigvecs = torch.linalg.eigh((reduced + reduced.T) / 2)
        coefficients = torch.linalg.solve_triangular(
            chol.T, eigvecs[:, -1:], upper=True
        ).flatten()
        direction = tree_linear_combination(expanded.vectors, coefficients)
        raw = compute_spectral_center(safety, benign, parameters, direction, mu=mu)
        scale = raw.b.rsqrt()
        if bool(tree_dot(direction, center.direction) < 0):
            scale = -scale
        return rescale_owned_spectral_center_(raw, scale), expanded


def _refine_center(safety, benign, parameters, center, mu):
    """One detached two-vector Ritz correction on the original full operators.

    Includes the original direction and never changes mu. Every candidate is
    independently re-evaluated; callers retain it only if residual improves.
    This improves a numerical candidate, not a certificate of dominance.
    """
    state = _RefinementState(
        vectors=(center.direction,),
        safety_responses=(center.zh,),
        benign_responses=(center.zb,),
    )
    candidate, _ = _refine_center_with_state(
        safety, benign, parameters, center, mu, state
    )
    return candidate


def _layout(functional, parameters):
    blocks = functional_blocks(functional)
    if len(blocks) == 1 and blocks[0].size == 0:
        with torch.no_grad():
            values = functional(parameters)
        if values.ndim != 1 or values.numel() == 0:
            raise ValueError("functional must return a nonempty vector")
        return blocks, (values.numel(),)
    return blocks, tuple(block.size for block in blocks)


def _validate_precision(functionals, parameters, *, allow_bf16: bool = False):
    assert_same_structure(parameters, parameters)
    devices = {value.device for value in parameters.values()}
    if len(devices) != 1:
        raise ValueError("dual geometry requires one device")
    for device in devices:
        autocast_enabled = torch.is_autocast_enabled(device.type)
        autocast_dtype = torch.get_autocast_dtype(device.type) if autocast_enabled else None
        if autocast_enabled and not (allow_bf16 and autocast_dtype == torch.bfloat16):
            raise ValueError("dual geometry requires FP32/FP64 forwards without autocast")
    allowed_dtypes = ((torch.bfloat16, torch.float32, torch.float64) if allow_bf16
                      else (torch.float32, torch.float64))
    if any(p.dtype not in allowed_dtypes for p in parameters.values()):
        raise ValueError("dual geometry requires FP32/FP64 admissible coordinates")
    for functional in functionals:
        model = getattr(functional, "model", None)
        if model is not None:
            if any(m.training for m in model.modules()):
                raise ValueError("dual geometry requires deterministic eval mode")
            tensors = tuple(model.parameters()) + tuple(model.buffers())
            if any(t.is_floating_point() and t.dtype not in allowed_dtypes for t in tensors):
                raise ValueError("dual geometry requires FP32/FP64 model weights and buffers")


def build_functional_kernel(
    safety: FunctionalMap, benign: FunctionalMap, parameters: TensorTree, *, max_size: int = 64,
    column_batch_size: int = 1,
    allow_bf16: bool = False,
) -> FunctionalKernel:
    """Build J_D J_D^T via streamed columns, D=(Fh/sqrt(Mh), Fb/sqrt(Mb))."""
    _validate_precision((safety, benign), parameters, allow_bf16=allow_bf16)
    if isinstance(max_size, bool) or not isinstance(max_size, int) or max_size < 1:
        raise ValueError("max_size must be a positive integer")
    if isinstance(column_batch_size, bool) or not isinstance(column_batch_size, int) or column_batch_size < 1:
        raise ValueError("column_batch_size must be a positive integer")
    layouts = (_layout(safety, parameters), _layout(benign, parameters))
    sizes = tuple(sum(layout[1]) for layout in layouts)
    n = sum(sizes)
    if n > max_size:
        raise ValueError(f"dual functional dimension {n} exceeds configured limit {max_size}")
    if column_batch_size > 1 and column_batch_size >= n:
        raise ValueError("column batch must be smaller than functional dimension; never retain the full Jacobian")
    leaf = next(iter(parameters.values()))
    kernel = torch.empty((n, n), dtype=torch.float64, device=leaf.device)
    column = 0
    with torch.no_grad():
        pending = []
        def flush():
            nonlocal column
            width = len(pending)
            zh = functional_jvp_directions(safety, parameters, pending)
            zb = functional_jvp_directions(benign, parameters, pending)
            kernel[:sizes[0], column:column + width] = zh.T.double() / math.sqrt(sizes[0])
            kernel[sizes[0]:, column:column + width] = zb.T.double() / math.sqrt(sizes[1])
            column += width
            pending.clear()
        for (blocks, block_sizes), size in zip(layouts, sizes):
            for block, block_size in zip(blocks, block_sizes):
                # A basis cotangent only touches this block. Do not forward all
                # unrelated chunks just to pull back a zero cotangent.
                for local_index in range(block_size):
                    cotangent = torch.zeros(block_size, dtype=leaf.dtype, device=leaf.device)
                    cotangent[local_index] = 1 / math.sqrt(size)
                    direction = block_vjp(block, parameters, cotangent)
                    if column_batch_size > 1:
                        pending.append(direction)
                        del direction, cotangent
                        if len(pending) == column_batch_size:
                            flush()
                        continue
                    _, zh = functional_jvp(safety, parameters, direction)
                    _, zb = functional_jvp(benign, parameters, direction)
                    kernel[:sizes[0], column] = zh.double() / math.sqrt(sizes[0])
                    kernel[sizes[0]:, column] = zb.double() / math.sqrt(sizes[1])
                    column += 1
                    del direction, cotangent, zh, zb
        if pending:
            flush()
    return FunctionalKernel(kernel.detach(), *sizes)


def dual_rayleigh_solve(
    safety: FunctionalMap, benign: FunctionalMap, parameters: TensorTree, *,
    mu: float | Tensor, warm_start: TensorTree | None = None,
    max_size: int = 64, tolerance: float = 1e-4, kernel_tolerance: float = 1e-5,
    residual_tolerance: float | None = None,
    column_batch_size: int = 1,
    refinement_steps: int = 0,
    allow_bf16: bool = False,
) -> RayleighResult:
    """Solve the full finite-batch leading eigenproblem and verify independently.

    Factor K_D=E diag(lambda) E^T=R R^T. Solve the generalized problem
    in the span represented by R, avoiding Woodbury's H-T solve(...) subtraction.
    Roundoff-size kernel eigenvalues are discarded only with a reported,
    damping-relative error guard; no extra ridge or hidden mu adjustment.
    """
    if residual_tolerance is None:
        residual_tolerance = tolerance
    for name, value in (("mu", mu), ("tolerance", tolerance),
                        ("kernel_tolerance", kernel_tolerance),
                        ("residual_tolerance", residual_tolerance)):
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    mu = float(mu)
    if isinstance(refinement_steps, bool) or not isinstance(refinement_steps, int) or refinement_steps < 0:
        raise ValueError("refinement_steps must be a nonnegative integer")
    if warm_start is not None:
        assert_same_structure(parameters, warm_start)
        if not bool(torch.isfinite(tree_norm(warm_start))):
            raise ValueError("warm_start must be finite")
    kernel = build_functional_kernel(safety, benign, parameters, max_size=max_size,
                                     column_batch_size=column_batch_size,
                                     allow_bf16=allow_bf16)
    matrix, mh = kernel.values, kernel.safety_size
    n = matrix.shape[0]
    if not bool(torch.isfinite(matrix).all()):
        raise FloatingPointError("non-finite functional kernel")
    norm = torch.linalg.matrix_norm(matrix).clamp_min(torch.finfo(matrix.dtype).tiny)
    asymmetry = torch.linalg.matrix_norm(matrix - matrix.T) / norm
    if bool(asymmetry > kernel_tolerance):
        raise FloatingPointError(f"functional kernel is asymmetric: {float(asymmetry):.3g}")
    matrix = (matrix + matrix.T) / 2
    values, vectors = torch.linalg.eigh(matrix)
    scale = values.abs().max()
    cutoff = 64 * torch.finfo(matrix.dtype).eps * n * scale
    if bool(values[0] < -kernel_tolerance * scale):
        raise FloatingPointError("functional kernel is materially non-PSD; inspect JVP/VJP precision")
    keep = values > cutoff
    discarded = values[~keep]
    projection = discarded.abs().max() if discarded.numel() else values.new_zeros(())
    # This is a numerical alarm, not an eigenvalue-error certificate.
    if bool(projection / mu > tolerance):
        raise FloatingPointError("kernel roundoff is large relative to damping; increase arithmetic precision")

    if bool((matrix[:mh, :mh] == 0).all()):
        zero = values.new_zeros(())
        diagnostics = RayleighDiagnostics(
            rho_hat=zero, qh=zero, qb=zero, qb_damped=zero, eig_residual_norm=zero,
            u_euclidean_norm=zero, b_mu_norm=zero, rayleigh_values=(zero,),
            warm_start_cosine=None, inner_steps=0, mu=zero + mu, relative_residual=zero,
            converged=True, rejected_steps=0, metric_evaluations=0, zero_safety=True,
            eigengap=zero, kernel_asymmetry=asymmetry, kernel_projection=projection,
            kernel_columns=n, damping_fraction=zero, dual_value_error=zero,
        )
        return RayleighResult(tree_zeros_like(parameters), diagnostics, None)
    if not bool(keep.any()):
        raise FloatingPointError("nonzero safety map has no numerically resolved kernel range")

    roots = values[keep].sqrt()
    basis = vectors[:, keep]
    factor = basis * roots
    ah, cb = factor[:mh], factor[mh:]
    small_b = cb.T @ cb + mu * torch.eye(roots.numel(), device=matrix.device, dtype=matrix.dtype)
    chol = torch.linalg.cholesky(small_b)
    whitened = torch.linalg.solve_triangular(chol, ah.T, upper=False)
    reduced = whitened.T @ whitened
    spectrum, eigenvectors = torch.linalg.eigh((reduced + reduced.T) / 2)
    largest = spectrum[-1]
    if not bool(torch.isfinite(largest)) or not bool(largest > 0):
        raise FloatingPointError("invalid reduced leading eigenvalue")
    y = eigenvectors[:, -1]
    coordinates = torch.cholesky_solve((ah.T @ y)[:, None], chol).squeeze(-1)
    coefficients = basis @ (coordinates / roots)
    if not bool(torch.isfinite(coefficients).all()):
        raise FloatingPointError("non-finite dual reconstruction coefficients")
    with torch.no_grad():
        hvec = functional_vjp(safety, parameters, coefficients[:mh] / math.sqrt(mh))
        bvec = functional_vjp(benign, parameters, coefficients[mh:] / math.sqrt(kernel.benign_size))
        direction = accumulate_detached_gradient(hvec, bvec)
        del hvec, bvec
        # Re-evaluate in the original operators, not only the small factor.
        raw = compute_spectral_center(safety, benign, parameters, direction, mu=mu)
        normalization = raw.b.rsqrt()
        if warm_start is not None and bool(tree_dot(direction, warm_start) < 0):
            normalization = -normalization
        center = rescale_owned_spectral_center_(raw, normalization)
        del raw
        direction = center.direction
        residual, residual_norm, relative = _relative_residual(center)
        del residual
        value_error = (center.rho.double() - largest).abs() / largest.abs()
        attempts = 0
        refinement_state = _RefinementState(
            vectors=(center.direction,),
            safety_responses=(center.zh,),
            benign_responses=(center.zb,),
        )
        # Do not repair a corrupt kernel/value estimate with a local search.
        while (attempts < refinement_steps and bool(torch.isfinite(relative))
               and bool(relative > residual_tolerance) and bool(value_error <= tolerance)):
            attempts += 1
            candidate, expanded_state = _refine_center_with_state(
                safety, benign, parameters, center, mu, refinement_state
            )
            if candidate is None:
                break
            cr, cn, ce = _relative_residual(candidate)
            del cr
            cv = (candidate.rho.double() - largest).abs() / largest.abs()
            if not bool(ce < relative) or not bool(cv <= tolerance):
                del candidate
                break
            center, residual_norm, relative, value_error = candidate, cn, ce, cv
            refinement_state = expanded_state
            direction = center.direction
            del candidate
        if (not bool(torch.isfinite(relative)) or bool(relative > residual_tolerance)
                or bool(value_error > tolerance)):
            raise FloatingPointError(
                f"dual reconstruction failed independent verification: residual={float(relative):.3g}, "
                f"residual_tolerance={float(residual_tolerance):.3g}, "
                f"value_error={float(value_error):.3g}, value_tolerance={float(tolerance):.3g}; "
                "improve precision, do not change mu silently"
            )
        if bool((center.b - 1).abs() > tolerance):
            raise FloatingPointError("dual direction failed B normalization")
        warm_cosine = None if warm_start is None else tree_cosine_similarity(direction, warm_start)
        eigengap = largest - spectrum[-2] if len(spectrum) > 1 else largest
        diagnostics = RayleighDiagnostics(
            rho_hat=center.rho, qh=center.qh, qb=center.qb, qb_damped=center.b,
            eig_residual_norm=residual_norm, u_euclidean_norm=tree_norm(direction),
            b_mu_norm=center.b.sqrt(), rayleigh_values=(center.rho,), warm_start_cosine=warm_cosine,
            inner_steps=0, mu=center.b.new_tensor(mu), relative_residual=relative,
            converged=True, rejected_steps=0, metric_evaluations=2 + 2 * attempts,
            eigengap=eigengap, kernel_asymmetry=asymmetry, kernel_projection=projection,
            kernel_columns=n, damping_fraction=mu * tree_dot(direction, direction) / center.b,
            dual_value_error=value_error,
            refinement_steps=attempts,
            refinement_jvps=2 * attempts,
        )
    return RayleighResult(direction, diagnostics, center)
