"""Matrix-free Jacobian-Gram metric operations for CBG."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from contextlib import nullcontext
from typing import TypeAlias

import torch
from torch import Tensor
from torch.func import jvp, vjp, vmap
from src.cbg.functionals import BatchedFunctional, FunctionalBlock

from src.cbg.tree_ops import (
    OrderedTensorTree,
    TensorTree,
    accumulate_detached_gradient,
    assert_same_structure,
    tree_add,
    tree_detach,
)


FunctionalMap: TypeAlias = Callable[[TensorTree], Tensor]
FunctionalWeight: TypeAlias = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class MetricProduct:
    vector: OrderedTensorTree
    jv: Tensor


def functional_blocks(functional: FunctionalMap):
    if isinstance(functional, BatchedFunctional):
        return functional.blocks
    return (FunctionalBlock(0, ((1., functional),)),)


def uniform_functional_weights(vector: Tensor) -> Tensor:
    """Apply ``W = I / B`` without constructing a diagonal matrix."""

    if vector.ndim != 1:
        raise ValueError("functional values must be a one-dimensional batch vector")
    if vector.numel() == 0:
        raise ValueError("functional batch must not be empty")
    return vector / vector.numel()


def _validate_functional_output(values: Tensor) -> None:
    if not isinstance(values, Tensor):
        raise TypeError("functional map must return a torch.Tensor")
    if values.ndim != 1:
        raise ValueError(
            f"functional map must return shape [batch], got {tuple(values.shape)}"
        )
    if values.numel() == 0:
        raise ValueError("functional map returned an empty batch")


def functional_jvp(
    functional: FunctionalMap,
    parameters: TensorTree,
    direction: TensorTree,
) -> tuple[Tensor, Tensor]:
    """Return ``(F(theta), J_F direction)`` using forward-mode AD."""

    assert_same_structure(parameters, direction)
    pieces = [jvp(block, (parameters,), (direction,)) for block in functional_blocks(functional)]
    values = torch.cat([part[0] for part in pieces])
    directional_values = torch.cat([part[1] for part in pieces])
    _validate_functional_output(values)
    if directional_values.shape != values.shape:
        raise ValueError("JVP output shape does not match functional output shape")
    return values, directional_values


def block_vjp(block: FunctionalBlock, parameters: TensorTree, cotangent: Tensor) -> OrderedTensorTree:
    """Detached pullback, freeing each signed response graph before the next."""
    result = None
    with torch.no_grad():
        for scale, term in block.terms:
            values, pullback = vjp(term, parameters)
            _validate_functional_output(values)
            if values.shape != cotangent.shape:
                raise ValueError("cotangent shape differs from the functional block")
            pulled = pullback(scale * cotangent.to(values))[0]
            result = accumulate_detached_gradient(result, pulled)
            del values, pullback, pulled
    assert result is not None
    return tree_detach(result)


def functional_jvp_directions(functional, parameters, directions):
    """Detached JVPs for a bounded block of directions, sharing primal work.

    The leading leaf axis indexes directions, NOT data examples. Each signed
    response term is evaluated separately so at most one term's tangent
    activations are live. randomness='error' rejects stochastic execution.
    This helper is for the detached eigensolver, not the outer derivative.
    """
    if not directions:
        raise ValueError("directions must be nonempty")
    for direction in directions:
        assert_same_structure(parameters, direction)
    with torch.no_grad():
        stacked = OrderedDict((name, torch.stack([v[name] for v in directions]))
                              for name in parameters)
        pieces = []
        for block in functional_blocks(functional):
            total = None
            for scale, term in block.terms:
                def tangent(v):
                    return jvp(term, (parameters,), (v,))[1]
                part = vmap(tangent, randomness="error")(stacked)
                total = scale * part if total is None else total + scale * part
                del part
            pieces.append(total)
        return torch.cat(pieces, dim=1).detach()


def functional_vjp(functional: FunctionalMap, parameters: TensorTree, cotangent: Tensor) -> OrderedTensorTree:
    """Detached streamed J^T c; no batch-by-parameter tensor is assembled."""
    offset, result = 0, None
    for block in functional_blocks(functional):
        size = block.size or cotangent.numel()
        pulled = block_vjp(block, parameters, cotangent[offset:offset + size])
        # block_vjp already returns privately owned detached storage.
        result = pulled if result is None else accumulate_detached_gradient(result, pulled)
        offset += size
        del pulled
    if offset != cotangent.numel():
        raise ValueError("cotangent size differs from the complete functional")
    assert result is not None
    return tree_detach(result)


def normalized_functional_rows(functional: FunctionalMap, parameters: TensorTree):
    """Yield rows of J/sqrt(output size) as individual signed VJP trees.

    Consumers must release each row; collecting this iterator would assemble
    a Jacobian transpose and is prohibited in production.
    """
    blocks = functional_blocks(functional)
    sizes = []
    with torch.no_grad():
        for block in blocks:
            size = block.size
            if not size:
                values = block(parameters)
                _validate_functional_output(values)
                size = values.numel()
                del values
            sizes.append(size)
        total = sum(sizes)
        leaf = next(iter(parameters.values()))
        for block, size in zip(blocks, sizes):
            for index in range(size):
                cotangent = leaf.new_zeros(size)
                cotangent[index] = total ** -0.5
                yield block_vjp(block, parameters, cotangent)


def metric_vector_product(
    functional: FunctionalMap,
    parameters: TensorTree,
    direction: TensorTree,
    *,
    weight: FunctionalWeight = uniform_functional_weights,
    detach_result: bool = False,
) -> OrderedTensorTree:
    """Compute ``J^T W (J v)`` using one JVP and one VJP.

    This function never flattens parameter leaves and never materializes a
    Jacobian, Gram matrix, or parameter-by-parameter tensor.
    """

    return metric_product_with_jvp(
        functional, parameters, direction, weight=weight, detach_result=detach_result
    ).vector


def metric_product_with_jvp(
    functional: FunctionalMap,
    parameters: TensorTree,
    direction: TensorTree,
    *,
    weight: FunctionalWeight = uniform_functional_weights,
    detach_result: bool = True,
) -> MetricProduct:
    """Return Gv and its already-computed Jv, with one reverse graph at a time.

    Weights are applied to the FULL batch vector before block slicing. Each
    signed term receives the same block cotangent, preserving safety cross
    terms and avoiding a simultaneous harmful/safe reverse graph. no_grad
    disables reverse graph creation for JVP; forward AD still runs.
    """

    with torch.no_grad() if detach_result else nullcontext():
        _, directional = functional_jvp(functional, parameters, direction)
        weighted = weight(directional)
        if weighted.shape != directional.shape:
            raise ValueError("functional weight changed the shape of its input")
        result = None
        offset = 0
        for block in functional_blocks(functional):
            size = block.size or directional.numel()
            for scale, term in block.terms:
                values, pullback = vjp(term, parameters)
                _validate_functional_output(values)
                if values.numel() != size:
                    raise ValueError("functional block size changed during metric evaluation")
                cotangent = scale * weighted[offset:offset + size]
                pulled = pullback(cotangent)[0]
                if detach_result:
                    result = accumulate_detached_gradient(result, pulled)
                elif result is None:
                    result = OrderedDict((name, pulled[name]) for name in parameters)
                else:
                    result = tree_add(result, pulled)
                del values, pullback, pulled
            offset += size
        assert result is not None
    assert_same_structure(parameters, result)
    if detach_result:
        result, directional = tree_detach(result), directional.detach()
    return MetricProduct(result, directional)


def damped_benign_metric_vector_product(
    benign_functional: FunctionalMap,
    parameters: TensorTree,
    direction: TensorTree,
    mu: float | Tensor,
    *,
    weight: FunctionalWeight = uniform_functional_weights,
    detach_result: bool = False,
) -> OrderedTensorTree:
    """Compute ``B_mu v = G_b v + mu v`` matrix-free."""

    if isinstance(mu, Tensor):
        if mu.numel() != 1:
            raise ValueError("mu must be a scalar")
        if bool((mu.detach() <= 0).item()):
            raise ValueError("mu must be strictly positive")
    elif mu <= 0:
        raise ValueError("mu must be strictly positive")

    gb_direction = metric_vector_product(
        benign_functional,
        parameters,
        direction,
        weight=weight,
        detach_result=detach_result,
    )
    result = tree_add(gb_direction, direction, alpha=mu)
    return tree_detach(result) if detach_result else result
