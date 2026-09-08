"""Center-point scalarization and centered finite-difference CBG gradient."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
from torch import Tensor

from src.cbg.geometry import (
    FunctionalMap, MetricProduct, functional_blocks, metric_product_with_jvp,
)
from src.cbg.tree_ops import (
    OrderedTensorTree,
    TensorTree,
    assert_same_structure,
    tree_add,
    tree_detach,
    tree_dot,
    tree_norm,
    tree_scale,
    tree_sub,
)


ScalarFunctional = Callable[[TensorTree], Tensor]


@dataclass(frozen=True)
class SpectralCenter:
    """All frozen center-point quantities used by the spectral derivative."""

    direction: OrderedTensorTree
    gh_direction: OrderedTensorTree
    gb_direction: OrderedTensorTree
    b_direction: OrderedTensorTree
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


@dataclass(frozen=True)
class SpectralGradientResult:
    gradient: OrderedTensorTree
    center: SpectralCenter
    delta: Tensor
    grad_plus: OrderedTensorTree | None
    grad_minus: OrderedTensorTree | None
    fd_cancellation: Tensor


def _tree_token(parameters: TensorTree) -> tuple:
    return tuple((name, value.data_ptr(), value._version, tuple(value.shape), value.dtype)
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
        result[name] = torch.zeros_like(parameter) if gradient is None else gradient
    return result


def streamed_loss_gradient(
    weighted_losses: Iterable[tuple[float, Tensor]], parameters: TensorTree,
) -> tuple[Tensor, OrderedTensorTree]:
    """Differentiate each chunk before evaluating the next; retain only sums."""

    total_loss, total_gradient = None, None
    for fraction, loss in weighted_losses:
        gradient = tree_detach(parameter_gradient(loss * fraction, parameters))
        total_gradient = gradient if total_gradient is None else tree_add(total_gradient, gradient)
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
    b_direction = tree_add(gb_direction, tree_scale(frozen_direction, mu))
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
