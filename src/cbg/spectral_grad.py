"""V5 direct spectral differentiation and the legacy FD numerical oracle."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace

import torch
from torch import Tensor
from torch.func import jvp

from src.cbg.geometry import (
    FunctionalMap, MetricProduct, functional_blocks, metric_product_with_jvp,
)
from src.cbg.tree_ops import (
    OrderedTensorTree,
    TensorTree,
    accumulate_detached_gradient,
    assert_same_structure,
    tree_add,
    tree_detach,
    tree_dot,
    tree_norm,
    tree_scale,
    scale_owned_detached_tree_,
    tree_sub,
)


ScalarFunctional = Callable[[TensorTree], Tensor]


@dataclass(frozen=True)
class SpectralCenter:
    """All frozen center-point quantities used by the spectral derivative."""

    direction: OrderedTensorTree
    gh_direction: OrderedTensorTree | None
    gb_direction: OrderedTensorTree | None
    b_direction: OrderedTensorTree | None
    b: Tensor
    rho: Tensor
    qh: Tensor
    qb: Tensor
    zh: Tensor
    zb: Tensor
    ch: Tensor
    cb: Tensor
    parameter_token: tuple
    direction_token: tuple
    functional_ids: tuple[int, int]
    functional_state: tuple
    precision_token: tuple
    mu: float


def compact_spectral_center(center: SpectralCenter) -> SpectralCenter:
    """Keep the frozen outer scalarization, without the three solver products.

    Does not mutate the input. The caller must release its original center
    (including RayleighResult) for those product buffers to be freed. Cache
    identity and stale-center checks remain intact for direct AD and FD.
    """
    return replace(center, gh_direction=None, gb_direction=None, b_direction=None)


@dataclass(frozen=True)
class SpectralGradientResult:
    gradient: OrderedTensorTree
    center: SpectralCenter
    delta: Tensor | None
    grad_plus: OrderedTensorTree | None
    grad_minus: OrderedTensorTree | None
    fd_cancellation: Tensor | None
    backend: str = "centered_fd"
    terms_evaluated: int = 0


def rescale_owned_spectral_center_(center: SpectralCenter, scale: float | Tensor) -> SpectralCenter:
    """Rescale an exclusively owned detached center without allocating four P-trees.

    Solver centers own their direction and metric-product buffers.  Storage
    aliasing with live parameters or between center trees is rejected before
    mutation.  Small directional responses and scalar contractions are then
    recomputed exactly as in ``compute_spectral_center``.
    """
    trees = (center.direction, center.gh_direction, center.gb_direction, center.b_direction)
    if any(tree is None for tree in trees):
        raise ValueError("cannot normalize a compact outer center")
    # Validate before changing even a single leaf.
    scale_tensor = torch.as_tensor(scale)
    if scale_tensor.numel() != 1 or not bool(torch.isfinite(scale_tensor)) or not bool(scale_tensor != 0):
        raise ValueError("center scale must be a finite nonzero scalar")
    scale = scale_tensor.detach().clone()
    parameter_ptrs = {token[5] for token in center.parameter_token}
    storage_ptrs = []
    for tree in trees:
        if not isinstance(tree, OrderedDict):
            raise TypeError("solver center trees must be owned OrderedDict instances")
        for value in tree.values():
            if value.requires_grad or value.grad_fn is not None:
                raise ValueError("solver center must be detached before in-place normalization")
            ptr = value.untyped_storage().data_ptr()
            if ptr in parameter_ptrs:
                raise ValueError("solver center aliases live model parameters")
            storage_ptrs.append(ptr)
    if len(storage_ptrs) != len(set(storage_ptrs)):
        raise ValueError("solver center trees share storage and cannot be normalized in place")
    for tree in trees:
        scale_owned_detached_tree_(tree, scale)
    zh, zb = center.zh * scale, center.zb * scale
    qh = tree_dot(center.direction, center.gh_direction)
    qb = tree_dot(center.direction, center.gb_direction)
    b = tree_dot(center.direction, center.b_direction)
    if not bool(torch.isfinite(b)) or not bool(b > 0):
        raise FloatingPointError("rescaled spectral center has an invalid damped denominator")
    rho = qh / b
    if not bool(torch.isfinite(rho)) or not bool(torch.isfinite(qb)):
        raise FloatingPointError("non-finite rescaled spectral center")
    return replace(
        center, b=b.detach(), rho=rho.detach(), qh=qh.detach(), qb=qb.detach(),
        zh=zh.detach(), zb=zb.detach(), ch=(zh / zh.numel()).detach(),
        cb=(zb / zb.numel()).detach(), direction_token=_tree_token(center.direction),
    )


def _tree_token(parameters: TensorTree) -> tuple:
    return tuple((name, value.data_ptr(), value._version, tuple(value.shape), value.dtype,
                  value.untyped_storage().data_ptr())
                 for name, value in parameters.items())


def _functional_state(safety_functional, benign_functional):
    return tuple(functional.state_token() if hasattr(functional, "state_token") else None
                 for functional in (safety_functional, benign_functional))


def _precision_token(parameters: TensorTree) -> tuple:
    device_types = sorted({value.device.type for value in parameters.values()})
    return (torch.get_float32_matmul_precision(), tuple(
        (device, torch.is_autocast_enabled(device), torch.get_autocast_dtype(device))
        for device in device_types
    ))


@dataclass(frozen=True)
class _RngState:
    cpu: Tensor
    cuda: tuple[Tensor, ...] | None


def _capture_rng_state() -> _RngState:
    cuda_state = None
    if torch.cuda.is_available():
        cuda_state = tuple(torch.cuda.get_rng_state_all())
    return _RngState(cpu=torch.random.get_rng_state(), cuda=cuda_state)


def _restore_rng_state(state: _RngState) -> None:
    torch.random.set_rng_state(state.cpu)
    if state.cuda is not None:
        torch.cuda.set_rng_state_all(list(state.cuda))


def _validate_nonnegative(value: float, *, name: str) -> None:
    tensor = torch.as_tensor(value)
    if not bool(torch.isfinite(tensor).item()) or not bool((tensor >= 0).item()):
        raise ValueError(f"{name} must be finite and non-negative")


def _validate_positive(value: float, *, name: str) -> None:
    tensor = torch.as_tensor(value)
    if not bool(torch.isfinite(tensor).item()) or not bool((tensor > 0).item()):
        raise ValueError(f"{name} must be finite and strictly positive")


def parameter_gradient(
    scalar: Tensor,
    parameters: TensorTree,
    *,
    create_graph: bool = False,
) -> OrderedTensorTree:
    """Differentiate a scalar into an ordered parameter tree."""

    if scalar.numel() != 1:
        raise ValueError("parameter_gradient requires a scalar output")
    if not scalar.requires_grad:
        return OrderedDict((name, torch.zeros_like(value)) for name, value in parameters.items())
    gradients = torch.autograd.grad(
        scalar,
        tuple(parameters.values()),
        create_graph=create_graph,
        retain_graph=create_graph,
        allow_unused=True,
    )
    result: OrderedTensorTree = OrderedDict()
    for (name, parameter), gradient in zip(parameters.items(), gradients):
        # Ownership belongs to the caller: autograd can return aliases of
        # parameters, external cotangents, or other gradient leaves.
        result[name] = torch.zeros_like(parameter) if gradient is None else gradient
    return result


def streamed_loss_gradient(
    weighted_losses: Iterable[tuple[float, Tensor]], parameters: TensorTree,
) -> tuple[Tensor, OrderedTensorTree]:
    """Differentiate each chunk before evaluating the next; retain only sums."""

    total_loss, total_gradient = None, None
    for fraction, loss in weighted_losses:
        gradient = tree_detach(parameter_gradient(loss * fraction, parameters))
        total_gradient = accumulate_detached_gradient(total_gradient, gradient)
        value = loss.detach() * fraction
        total_loss = value if total_loss is None else total_loss + value
        del loss, gradient
    if total_gradient is None:
        raise ValueError("cannot differentiate an empty loss stream")
    return total_loss, total_gradient


def compute_spectral_center(
    safety_functional: FunctionalMap,
    benign_functional: FunctionalMap,
    parameters: TensorTree,
    direction: TensorTree,
    *,
    mu: float | Tensor,
    safety_product: MetricProduct | None = None,
    benign_product: MetricProduct | None = None,
) -> SpectralCenter:
    """Compute and freeze center-point Rayleigh and directional quantities."""

    assert_same_structure(parameters, direction)
    frozen_direction = tree_detach(direction)
    _validate_positive(float(mu), name="mu")
    hp = safety_product or metric_product_with_jvp(safety_functional, parameters, frozen_direction)
    bp = benign_product or metric_product_with_jvp(benign_functional, parameters, frozen_direction)
    gh_direction, gb_direction = hp.vector, bp.vector
    b_direction = tree_add(gb_direction, frozen_direction, alpha=mu)
    qh = tree_dot(frozen_direction, gh_direction)
    qb = tree_dot(frozen_direction, gb_direction)
    b = tree_dot(frozen_direction, b_direction)
    if not bool(torch.isfinite(qh)) or not bool(torch.isfinite(qb)):
        raise FloatingPointError("non-finite safety or benign quadratic form")
    tolerance = 1e-10 if qh.dtype == torch.float64 else 1e-5
    if bool(qh < -tolerance) or bool(qb < -tolerance):
        raise FloatingPointError("negative Gram quadratic form; check deterministic JVP/VJP compatibility")
    if not bool(torch.isfinite(b).item()) or not bool((b > 0).item()):
        raise FloatingPointError("center-point u^T B_mu u must be finite and positive")
    rho = (qh / b).detach()
    if not bool(torch.isfinite(rho)):
        raise FloatingPointError("non-finite center-point Rayleigh quotient")

    zh, zb = hp.jv.detach(), bp.jv.detach()
    ch = (zh / zh.numel()).detach()
    cb = (zb / zb.numel()).detach()

    return SpectralCenter(
        direction=frozen_direction,
        gh_direction=tree_detach(gh_direction),
        gb_direction=tree_detach(gb_direction),
        b_direction=tree_detach(b_direction),
        b=b.detach(),
        rho=rho,
        qh=qh.detach(),
        qb=qb.detach(),
        zh=zh,
        zb=zb,
        ch=ch,
        cb=cb,
        parameter_token=_tree_token(parameters),
        direction_token=_tree_token(frozen_direction),
        functional_ids=(id(safety_functional), id(benign_functional)),
        functional_state=_functional_state(safety_functional, benign_functional),
        precision_token=_precision_token(parameters),
        mu=float(mu),
    )


@dataclass(frozen=True)
class ScalarizedFunctional:
    terms: tuple[ScalarFunctional, ...]

    def __call__(self, parameters: TensorTree) -> Tensor:
        return sum(term(parameters) for term in self.terms)


def make_scalarized_functional(
    safety_functional: FunctionalMap,
    benign_functional: FunctionalMap,
    center: SpectralCenter,
) -> ScalarFunctional:
    """Create frozen ``Phi(theta') = ch^T Fh - rho cb^T Fb``."""

    ch = center.ch.detach()
    cb = center.cb.detach()
    rho = center.rho.detach()

    terms = []
    for functional, coefficients in ((safety_functional, ch), (benign_functional, -rho * cb)):
        offset = 0
        for block in functional_blocks(functional):
            size = block.size or coefficients.numel()
            for scale, component in block.terms:
                frozen = (scale * coefficients[offset:offset + size]).detach()
                def term(parameters, component=component, frozen=frozen):
                    values = component(parameters)
                    if values.shape != frozen.shape:
                        raise ValueError("functional minibatch shape changed after center scalarization")
                    return torch.dot(frozen, values)
                terms.append(term)
            offset += size
        if offset != coefficients.numel():
            raise ValueError("functional batch size changed after center scalarization")
    return ScalarizedFunctional(tuple(terms))


def choose_physical_delta(
    parameters: TensorTree,
    direction: TensorTree,
    *,
    tau_fd: float = 1e-4,
    theta_floor: float = 1e-3,
    u_floor: float = 1e-12,
) -> Tensor:
    """Use the guide's relative physical perturbation rule."""

    _validate_positive(tau_fd, name="tau_fd")
    _validate_positive(theta_floor, name="theta_floor")
    _validate_positive(u_floor, name="u_floor")
    assert_same_structure(parameters, direction)
    theta_norm = tree_norm(parameters)
    u_norm = tree_norm(direction)
    delta = tau_fd * torch.clamp_min(theta_norm, theta_floor) / torch.clamp_min(u_norm, u_floor)
    if not bool(torch.isfinite(delta).item()) or not bool((delta > 0).item()):
        raise FloatingPointError("finite-difference delta must be finite and positive")
    return delta.detach()


def perturbed_parameter_tree(
    parameters: TensorTree,
    direction: TensorTree,
    signed_delta: float | Tensor,
) -> OrderedTensorTree:
    """Build independent stateless perturbation leaves in FP32 where feasible."""

    assert_same_structure(parameters, direction)
    result: OrderedTensorTree = OrderedDict()
    for name, parameter in parameters.items():
        work_dtype = torch.float64 if parameter.dtype == torch.float64 else torch.float32
        perturbed = parameter.detach().to(dtype=work_dtype) + torch.as_tensor(
            signed_delta,
            device=parameter.device,
            dtype=work_dtype,
        ) * direction[name].detach().to(dtype=work_dtype)
        result[name] = perturbed.requires_grad_(True)
    return result


def _gradient_at_perturbation(
    phi: ScalarFunctional,
    parameters: TensorTree,
    direction: TensorTree,
    signed_delta: float | Tensor,
) -> OrderedTensorTree:
    perturbed = perturbed_parameter_tree(parameters, direction, signed_delta)
    terms = phi.terms if isinstance(phi, ScalarizedFunctional) else (phi,)
    result = None
    for term in terms:
        value = term(perturbed)
        gradient = tree_detach(parameter_gradient(value, perturbed, create_graph=False))
        result = gradient if result is None else tree_add(result, gradient)
        del value, gradient
    assert result is not None
    return result


def centered_spectral_gradient(
    safety_functional: FunctionalMap,
    benign_functional: FunctionalMap,
    parameters: TensorTree,
    direction: TensorTree,
    *,
    mu: float | Tensor,
    tau_fd: float = 1e-4,
    theta_floor: float = 1e-3,
    u_floor: float = 1e-12,
    same_randomness: bool = True,
    center: SpectralCenter | None = None,
    retain_endpoint_gradients: bool = False,
) -> SpectralGradientResult:
    """Compute the Hessian-free centered CBG spectral gradient.

    Live model parameters are never mutated: plus/minus evaluations use
    independent stateless parameter trees. The eigensolver is not referenced
    here and therefore cannot be rerun at either perturbed point.
    """

    if center is None:
        center = compute_spectral_center(
            safety_functional, benign_functional, parameters, tree_detach(direction), mu=mu,
        )
    elif (center.parameter_token != _tree_token(parameters)
          or center.direction_token != _tree_token(direction)
          or center.functional_ids != (id(safety_functional), id(benign_functional))
          or center.functional_state != _functional_state(safety_functional, benign_functional)
          or center.precision_token != _precision_token(parameters)
          or center.mu != float(mu)):
        raise ValueError("stale spectral center: parameters, direction, functions, precision or mu changed")
    phi = make_scalarized_functional(safety_functional, benign_functional, center)
    delta = choose_physical_delta(
        parameters,
        center.direction,
        tau_fd=tau_fd,
        theta_floor=theta_floor,
        u_floor=u_floor,
    )

    initial_rng = _capture_rng_state()
    completed = False
    try:
        if same_randomness:
            _restore_rng_state(initial_rng)
        grad_plus = _gradient_at_perturbation(phi, parameters, center.direction, delta)
        if same_randomness:
            _restore_rng_state(initial_rng)
        grad_minus = _gradient_at_perturbation(phi, parameters, center.direction, -delta)
        completed = True
    finally:
        if not completed:
            _restore_rng_state(initial_rng)

    denominator = (delta * center.b).reciprocal()
    # Leaf arithmetic is FP32 in production and remains FP64 in numerical tests.
    difference = OrderedDict(
        (name, plus.to(torch.float64 if plus.dtype == torch.float64 else torch.float32)
         - grad_minus[name].to(torch.float64 if plus.dtype == torch.float64 else torch.float32))
        for name, plus in grad_plus.items()
    )
    cancellation = tree_norm(difference) / (tree_norm(grad_plus) + tree_norm(grad_minus)).clamp_min(1e-30)
    gradient = OrderedDict(
        (name, value * denominator) for name, value in difference.items()
    )
    return SpectralGradientResult(
        gradient=tree_detach(gradient),
        center=center,
        delta=delta,
        grad_plus=tree_detach(grad_plus) if retain_endpoint_gradients else None,
        grad_minus=tree_detach(grad_minus) if retain_endpoint_gradients else None,
        fd_cancellation=cancellation.detach(),
    )


def direct_spectral_gradient(
    safety_functional: FunctionalMap,
    benign_functional: FunctionalMap,
    parameters: TensorTree,
    direction: TensorTree,
    *,
    mu: float | Tensor,
    center: SpectralCenter | None = None,
    allow_constant_tangent: bool = False,
) -> SpectralGradientResult:
    """V5 section 4: stream (2/b) grad_theta[JVP(Phi, stop_gradient(u))].

    Each response graph is released before evaluating the next term. No live
    parameter is perturbed and no eigensolver trajectory is differentiated.
    Constant tangents are permitted only explicitly (e.g. affine toy maps):
    disconnected tangents in a nonlinear model must not silently become zeros.
    Transform errors propagate; FD is a separately selected oracle backend.
    """
    assert_same_structure(parameters, direction)
    if not all(p.requires_grad and p.dtype in (torch.float32, torch.float64)
               for p in parameters.values()):
        raise ValueError("direct spectral AD requires differentiable FP32/FP64 coordinates")
    # The qualified V5 path uses the same deterministic FP32/64 forward at
    # the center and derivative. Import locally to avoid the dual/rayleigh cycle.
    from src.cbg.dual import _validate_precision
    _validate_precision((safety_functional, benign_functional), parameters)
    if center is None:
        center = compute_spectral_center(safety_functional, benign_functional,
                                         parameters, tree_detach(direction), mu=mu)
    elif (center.parameter_token != _tree_token(parameters)
          or center.direction_token != _tree_token(direction)
          or center.functional_ids != (id(safety_functional), id(benign_functional))
          or center.functional_state != _functional_state(safety_functional, benign_functional)
          or center.precision_token != _precision_token(parameters)
          or center.mu != float(mu)):
        raise ValueError("stale spectral center: parameters, direction, functions, precision or mu changed")
    phi = make_scalarized_functional(safety_functional, benign_functional, center)
    gradient = None
    with torch.enable_grad():
        for index, term in enumerate(phi.terms):
            primal, tangent = jvp(term, (parameters,), (center.direction,))
            if not bool(torch.isfinite(primal)) or not bool(torch.isfinite(tangent)):
                raise FloatingPointError(f"non-finite direct spectral term {index}")
            del primal
            if not tangent.requires_grad and not allow_constant_tangent:
                raise RuntimeError(
                    f"direct spectral term {index} has a disconnected/constant tangent; "
                    "validate derivative support before explicitly allowing an affine zero Hessian"
                )
            part = tree_detach(parameter_gradient(tangent, parameters))
            if not bool(torch.isfinite(tree_norm(part))):
                raise FloatingPointError(f"non-finite direct spectral gradient in term {index}")
            gradient = accumulate_detached_gradient(gradient, part)
            del tangent, part
    if gradient is None:
        raise ValueError("empty spectral scalarization")
    # Privately owned storage: scale without allocating another full P-vector.
    with torch.no_grad():
        for value in gradient.values():
            value.mul_(2 / center.b.detach())
    if not bool(torch.isfinite(tree_norm(gradient))):
        raise FloatingPointError("non-finite direct spectral gradient")
    return SpectralGradientResult(gradient, center, None, None, None, None,
                                  backend="direct_rof", terms_evaluated=len(phi.terms))


def compose_training_gradient(
    alignment_gradient: TensorTree,
    benign_gradient: TensorTree,
    spectral_gradient: TensorTree,
    *,
    alpha: float,
    beta: float,
) -> OrderedTensorTree:
    """Return ``g_aln + alpha g_bgn + beta g_CBG``."""

    _validate_nonnegative(alpha, name="alpha")
    _validate_nonnegative(beta, name="beta")
    preservation = tree_add(alignment_gradient, benign_gradient, alpha=alpha)
    if beta == 0:
        return preservation
    return tree_add(preservation, spectral_gradient, alpha=beta)


def exact_spectral_gradient(
    phi: ScalarFunctional,
    parameters: TensorTree,
    direction: TensorTree,
    b: Tensor,
) -> OrderedTensorTree:
    """Exact ``(2 / b) H_Phi u`` for small-model validation only."""

    assert_same_structure(parameters, direction)
    phi_value = phi(parameters)
    first_gradient = parameter_gradient(phi_value, parameters, create_graph=True)
    directional_gradient = tree_dot(first_gradient, tree_detach(direction))
    hvp = parameter_gradient(directional_gradient, parameters, create_graph=False)
    return tree_detach(tree_scale(hvp, 2.0 / b.detach()))
