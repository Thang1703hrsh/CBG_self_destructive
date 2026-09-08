"""Ordered parameter-tree operations used by CBG.

CBG directions are represented as ordered mappings from fully qualified model
parameter names to tensors.  The helpers in this module deliberately avoid
flattening all parameters into one large tensor.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TypeAlias

import torch
from torch import Tensor, nn


TensorTree: TypeAlias = Mapping[str, Tensor]
OrderedTensorTree: TypeAlias = OrderedDict[str, Tensor]


def _validate_tree(tree: TensorTree, *, name: str) -> tuple[str, ...]:
    if not isinstance(tree, Mapping):
        raise TypeError(f"{name} must be a mapping from parameter names to tensors")
    if not tree:
        raise ValueError(f"{name} must contain at least one tensor")

    keys = tuple(tree.keys())
    for key, value in tree.items():
        if not isinstance(key, str):
            raise TypeError(f"{name} contains a non-string key: {key!r}")
        if not isinstance(value, Tensor):
            raise TypeError(f"{name}[{key!r}] is not a torch.Tensor")
        if not value.is_floating_point():
            raise TypeError(f"{name}[{key!r}] must have a floating dtype")
    return keys


def assert_same_structure(a: TensorTree, b: TensorTree) -> None:
    """Raise if two trees do not have identical ordered names and leaf shapes."""

    a_keys = _validate_tree(a, name="a")
    b_keys = _validate_tree(b, name="b")
    if a_keys != b_keys:
        raise ValueError(
            "parameter trees must have identical ordered keys; "
            f"got {a_keys!r} and {b_keys!r}"
        )
    for key in a_keys:
        if a[key].shape != b[key].shape:
            raise ValueError(
                f"parameter-tree shape mismatch for {key!r}: "
                f"{tuple(a[key].shape)} != {tuple(b[key].shape)}"
            )
        if a[key].device != b[key].device:
            raise ValueError(
                f"parameter-tree device mismatch for {key!r}: "
                f"{a[key].device} != {b[key].device}"
            )


def _accumulation_dtype(a: TensorTree, b: TensorTree | None = None) -> torch.dtype:
    trees = (a,) if b is None else (a, b)
    for tree in trees:
        if any(value.dtype == torch.float64 for value in tree.values()):
            return torch.float64
    return torch.float32


def tree_dot(a: TensorTree, b: TensorTree) -> Tensor:
    """Return a numerically stable dot product without concatenating leaves."""

    assert_same_structure(a, b)
    accumulation_dtype = _accumulation_dtype(a, b)
    result: Tensor | None = None
    for key in a:
        # Promote BEFORE multiplication; a FP32 sum cannot repair BF16/FP16
        # overflow or underflow that has already happened in the product.
        term = (a[key].to(accumulation_dtype) * b[key].to(accumulation_dtype)).sum()
        result = term if result is None else result + term
    assert result is not None
    return result


def tree_norm(tree: TensorTree) -> Tensor:
    """Return the Euclidean norm of a parameter tree."""

    squared_norm = tree_dot(tree, tree).real
    return torch.sqrt(torch.clamp_min(squared_norm, 0.0))


def tree_scale(tree: TensorTree, scalar: float | Tensor) -> OrderedTensorTree:
    _validate_tree(tree, name="tree")
    return OrderedDict((key, value * scalar) for key, value in tree.items())


def tree_add(
    a: TensorTree,
    b: TensorTree,
    *,
    alpha: float | Tensor = 1.0,
) -> OrderedTensorTree:
    """Return ``a + alpha * b`` leaf by leaf."""

    assert_same_structure(a, b)
    return OrderedDict((key, a[key] + alpha * b[key]) for key in a)


def tree_sub(a: TensorTree, b: TensorTree) -> OrderedTensorTree:
    return tree_add(a, b, alpha=-1.0)


def tree_clone(tree: TensorTree) -> OrderedTensorTree:
    _validate_tree(tree, name="tree")
    return OrderedDict((key, value.clone()) for key, value in tree.items())


def tree_detach(tree: TensorTree) -> OrderedTensorTree:
    _validate_tree(tree, name="tree")
    return OrderedDict((key, value.detach()) for key, value in tree.items())


def tree_zeros_like(tree: TensorTree) -> OrderedTensorTree:
    _validate_tree(tree, name="tree")
    return OrderedDict((key, torch.zeros_like(value)) for key, value in tree.items())


def tree_random_like(
    tree: TensorTree,
    *,
    generator: torch.Generator | None = None,
) -> OrderedTensorTree:
    _validate_tree(tree, name="tree")
    return OrderedDict(
        (
            key,
            torch.randn(
                value.shape,
                dtype=value.dtype,
                device=value.device,
                generator=generator,
            ),
        )
        for key, value in tree.items()
    )


def tree_cosine_similarity(
    a: TensorTree,
    b: TensorTree,
    *,
    eps: float = 1e-12,
) -> Tensor:
    assert_same_structure(a, b)
    denominator = tree_norm(a) * tree_norm(b)
    return (tree_dot(a, b) / torch.clamp_min(denominator, eps)).clamp(-1.0, 1.0)


def restrict_tree(tree: TensorTree, parameter_names: Sequence[str]) -> OrderedTensorTree:
    """Restrict a tree to an explicit ordered set of admissible names."""

    _validate_tree(tree, name="tree")
    missing = [name for name in parameter_names if name not in tree]
    if missing:
        raise KeyError(f"admissible parameter names not found in tree: {missing}")
    if len(set(parameter_names)) != len(parameter_names):
        raise ValueError("admissible parameter names must be unique")
    return OrderedDict((name, tree[name]) for name in parameter_names)


def select_admissible_parameters(
    model: nn.Module,
    *,
    parameter_names: Sequence[str] | None = None,
    include_substrings: Sequence[str] | None = None,
    predicate: Callable[[str, nn.Parameter], bool] | None = None,
    trainable_only: bool = True,
) -> OrderedTensorTree:
    """Select the exact model coordinates used by the CBG inner problem.

    Filters are conjunctive.  With no explicit filter, this selects all
    parameters whose ``requires_grad`` flag is true.  Supplying exact names is
    recommended when resuming a saved weakest-escape direction.
    """

    if parameter_names is not None and len(set(parameter_names)) != len(parameter_names):
        raise ValueError("parameter_names must be unique")
    requested = None if parameter_names is None else set(parameter_names)
    substrings = tuple(include_substrings or ())

    selected: OrderedTensorTree = OrderedDict()
    known_names: set[str] = set()
    for name, parameter in model.named_parameters():
        known_names.add(name)
        if requested is not None and name not in requested:
            continue
        if substrings and not any(fragment in name for fragment in substrings):
            continue
        if predicate is not None and not predicate(name, parameter):
            continue
        if trainable_only and not parameter.requires_grad:
            continue
        selected[name] = parameter

    if requested is not None:
        missing = requested - known_names
        if missing:
            raise KeyError(f"model does not contain requested parameters: {sorted(missing)}")
        skipped_frozen = requested - set(selected)
        if skipped_frozen and trainable_only:
            raise ValueError(
                "requested admissible parameters are frozen or excluded by another filter: "
                f"{sorted(skipped_frozen)}"
            )
    if not selected:
        raise ValueError("no admissible parameters matched the configured selection")
    return selected


def select_lora_parameters(
    model: nn.Module,
    *,
    target_modules: Iterable[str] = ("q_proj", "v_proj"),
    trainable_only: bool = True,
) -> OrderedTensorTree:
    """Select PEFT LoRA factors belonging to configured target modules.

    PEFT parameter names contain ``lora_`` and the fully qualified target
    module name.  This helper validates that the resulting CBG space contains
    LoRA coordinates only.
    """

    targets = tuple(target_modules)
    if not targets:
        raise ValueError("target_modules must contain at least one module name")

    selected = select_admissible_parameters(
        model,
        predicate=lambda name, _: "lora_" in name
        and any(target in name.split(".") for target in targets),
        trainable_only=trainable_only,
    )
    invalid = [name for name in selected if "lora_" not in name]
    if invalid:
        raise AssertionError(f"non-LoRA parameters entered the admissible space: {invalid}")
    return selected
