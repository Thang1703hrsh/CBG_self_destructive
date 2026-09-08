"""Safety-sensitive and benign functional maps for CBG."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

import torch
from torch import Tensor, nn
from torch.func import functional_call

from src.cbg.tree_ops import TensorTree


SequenceBatch: TypeAlias = Mapping[str, Tensor]
FunctionalMap: TypeAlias = Callable[[TensorTree], Tensor]


@dataclass(frozen=True)
class FunctionalBlock:
    """An independent group of examples; terms share the same example order."""

    size: int
    terms: tuple[tuple[float, FunctionalMap], ...]

    def __call__(self, parameters: TensorTree) -> Tensor:
        return sum(scale * functional(parameters) for scale, functional in self.terms)


@dataclass(frozen=True)
class BatchedFunctional:
    """Exact batch map with independently differentiable memory-sized blocks.

    Harmful and safe terms stay paired inside a block. In particular the
    metric uses (J_harm - J_safe)^T W (J_harm - J_safe), including cross terms.
    """

    blocks: tuple[FunctionalBlock, ...]
    state_tensors: tuple[Tensor, ...] = ()
    model: nn.Module | None = None

    def __call__(self, parameters: TensorTree) -> Tensor:
        return torch.cat([block(parameters) for block in self.blocks])

    def state_token(self) -> tuple:
        """Detect mutation of bound minibatches, base weights, buffers or modes."""
        tensors = self.state_tensors
        modes = ()
        if self.model is not None:
            tensors += tuple(self.model.parameters()) + tuple(self.model.buffers())
            modes = tuple(module.training for module in self.model.modules())
        return (tuple((tensor.data_ptr(), tensor._version) for tensor in tensors), modes)


def sequence_chunks(batch: SequenceBatch, chunk_size: int = 0):
    """Yield (example fraction, batch) with exact full-batch mean weighting."""

    _validate_sequence_batch(batch, name="batch")
    size = batch["input_ids"].shape[0]
    if size < 1 or chunk_size < 0:
        raise ValueError("batch must be nonempty and chunk_size non-negative")
    width = chunk_size or size
    for start in range(0, size, width):
        end = min(start + width, size)
        yield (end - start) / size, {name: tensor[start:end] for name, tensor in batch.items()}


def select_response_logits(logits: Tensor, batch: SequenceBatch) -> Tensor:
    """Copy only predicted response positions; the full logits can be freed."""

    mask = batch["response_mask"][:, 1:].bool()
    if "attention_mask" in batch:
        mask = mask & batch["attention_mask"][:, 1:].bool()
    return logits[:, :-1, :][mask]


def _validate_sequence_batch(batch: SequenceBatch, *, name: str) -> None:
    required = {"input_ids", "response_mask"}
    missing = required - set(batch)
    if missing:
        raise KeyError(f"{name} is missing required fields: {sorted(missing)}")

    input_ids = batch["input_ids"]
    response_mask = batch["response_mask"]
    if input_ids.ndim != 2:
        raise ValueError(f"{name}['input_ids'] must have shape [batch, sequence]")
    if response_mask.shape != input_ids.shape:
        raise ValueError(
            f"{name} response_mask shape {tuple(response_mask.shape)} does not "
            f"match input_ids shape {tuple(input_ids.shape)}"
        )
    if "attention_mask" in batch and batch["attention_mask"].shape != input_ids.shape:
        raise ValueError(f"{name} attention_mask must match input_ids shape")


def _extract_logits(outputs: Any) -> Tensor:
    if isinstance(outputs, Tensor):
        return outputs
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, (tuple, list)) and outputs:
        return outputs[0]
    raise TypeError("model output must be a tensor, expose .logits, or be a non-empty tuple")


def token_normalized_response_log_probs(
    logits: Tensor,
    input_ids: Tensor,
    response_mask: Tensor,
    *,
    attention_mask: Tensor | None = None,
) -> Tensor:
    """Return one causal response log probability per example.

    ``response_mask`` marks target-token positions in ``input_ids``.  Since a
    causal logit at position ``t - 1`` predicts token ``t``, the mask is shifted
    with the labels.  Prompt and padding tokens contribute neither to the sum
    nor to the response length.
    """

    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocabulary]")
    if input_ids.ndim != 2 or input_ids.shape != logits.shape[:2]:
        raise ValueError("input_ids must match the batch and sequence dimensions of logits")
    if response_mask.shape != input_ids.shape:
        raise ValueError("response_mask must have the same shape as input_ids")
    if input_ids.shape[1] < 2:
        raise ValueError("causal response log probabilities require sequence length >= 2")
    if attention_mask is not None and attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must have the same shape as input_ids")

    shifted_response_mask = response_mask[:, 1:].to(dtype=torch.bool)
    if attention_mask is not None:
        shifted_response_mask = shifted_response_mask & attention_mask[:, 1:].to(
            dtype=torch.bool
        )

    response_lengths = shifted_response_mask.sum(dim=-1)
    if torch.any(response_lengths <= 0):
        bad_rows = torch.nonzero(response_lengths <= 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"every example must contain a response token; empty rows: {bad_rows}")

    # Select target rows before promoting and normalizing over the vocabulary.
    # On a 3B model this avoids materializing FP32 log-probabilities for prompt
    # and padding positions while remaining exactly equivalent mathematically.
    active_logits = logits[:, :-1, :][shifted_response_mask]
    if active_logits.dtype in (torch.float16, torch.bfloat16):
        active_logits = active_logits.float()
    active_targets = input_ids[:, 1:][shifted_response_mask].to(dtype=torch.long)
    token_log_probs = torch.log_softmax(active_logits, dim=-1).gather(
        dim=-1,
        index=active_targets.unsqueeze(-1),
    ).squeeze(-1)
    batch_indices = torch.arange(input_ids.shape[0], device=logits.device).unsqueeze(1)
    batch_indices = batch_indices.expand_as(shifted_response_mask)[shifted_response_mask]
    per_example_log_probs = torch.zeros(
        input_ids.shape[0],
        device=logits.device,
        dtype=token_log_probs.dtype,
    ).scatter_add(0, batch_indices, token_log_probs)
    return per_example_log_probs / response_lengths.to(dtype=token_log_probs.dtype)


def _functional_logits(
    model: nn.Module,
    parameters: TensorTree,
    batch: SequenceBatch,
    *,
    buffers: Mapping[str, Tensor] | None = None,
) -> Tensor:
    _validate_sequence_batch(batch, name="batch")
    state: TensorTree | tuple[TensorTree, Mapping[str, Tensor]]
    state = parameters if buffers is None else (parameters, buffers)
    kwargs = {"input_ids": batch["input_ids"]}
    if "attention_mask" in batch:
        kwargs["attention_mask"] = batch["attention_mask"]
    outputs = functional_call(model, state, args=(), kwargs=kwargs, strict=False)
    return _extract_logits(outputs)


def _sequence_functional(
    model: nn.Module,
    parameters: TensorTree,
    batch: SequenceBatch,
    *,
    buffers: Mapping[str, Tensor] | None = None,
) -> Tensor:
    logits = _functional_logits(model, parameters, batch, buffers=buffers)
    return token_normalized_response_log_probs(
        logits,
        batch["input_ids"],
        batch["response_mask"],
        attention_mask=batch.get("attention_mask"),
    )


def safety_functional_map(
    model: nn.Module,
    parameters: TensorTree,
    harmful_batch: SequenceBatch,
    safe_batch: SequenceBatch,
    *,
    buffers: Mapping[str, Tensor] | None = None,
) -> Tensor:
    """Return ``F_h``: harmful minus safe normalized log-probability margins."""

    _validate_sequence_batch(harmful_batch, name="harmful_batch")
    _validate_sequence_batch(safe_batch, name="safe_batch")
    if harmful_batch["input_ids"].shape[0] != safe_batch["input_ids"].shape[0]:
        raise ValueError("harmful and safe batches must contain the same number of pairs")

    harmful_log_probs = _sequence_functional(
        model,
        parameters,
        harmful_batch,
        buffers=buffers,
    )
    safe_log_probs = _sequence_functional(
        model,
        parameters,
        safe_batch,
        buffers=buffers,
    )
    return harmful_log_probs - safe_log_probs


def benign_functional_map(
    model: nn.Module,
    parameters: TensorTree,
    benign_batch: SequenceBatch,
    *,
    buffers: Mapping[str, Tensor] | None = None,
) -> Tensor:
    """Return ``F_b``: normalized benign response log probabilities."""

    _validate_sequence_batch(benign_batch, name="benign_batch")
    return _sequence_functional(model, parameters, benign_batch, buffers=buffers)


def safe_alignment_loss(
    model: nn.Module,
    parameters: TensorTree,
    safe_batch: SequenceBatch,
    *,
    buffers: Mapping[str, Tensor] | None = None,
) -> Tensor:
    """Token-normalized safe/refusal response NLL, averaged over examples."""

    return -_sequence_functional(model, parameters, safe_batch, buffers=buffers).mean()


def benign_kl_from_logits(
    current_logits: Tensor,
    reference_logits: Tensor,
    response_mask: Tensor,
    *,
    attention_mask: Tensor | None = None,
) -> Tensor:
    """Exact ``KL(p_reference || p_current)`` on benign response positions.

    Each example is token-normalized before the minibatch mean is taken. The
    reference logits are always detached, so this loss cannot update or retain
    a graph through the protected reference model.
    """

    if current_logits.ndim != 3:
        raise ValueError("current logits must have shape [B, L, V]")
    if reference_logits.ndim == 3 and reference_logits.shape != current_logits.shape:
        raise ValueError("full reference logits must match current [B, L, V] shape")
    if reference_logits.ndim not in (2, 3):
        raise ValueError("reference logits must be full [B,L,V] or selected [N,V]")
    if response_mask.shape != current_logits.shape[:2]:
        raise ValueError("response_mask must match the logits batch and sequence dimensions")
    if attention_mask is not None and attention_mask.shape != response_mask.shape:
        raise ValueError("attention_mask must have the same shape as response_mask")

    shifted_mask = response_mask[:, 1:].to(device=current_logits.device, dtype=torch.bool)
    if attention_mask is not None:
        shifted_mask = shifted_mask & attention_mask[:, 1:].to(
            device=current_logits.device,
            dtype=torch.bool,
        )
    response_lengths = shifted_mask.sum(dim=-1)
    if torch.any(response_lengths <= 0):
        bad_rows = torch.nonzero(response_lengths <= 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"every benign example must contain a response token; empty rows: {bad_rows}")

    # The KL is exact on every selected response position, but prompt/padding
    # logits are discarded before the FP32 full-vocabulary softmax.
    current = current_logits[:, :-1, :][shifted_mask]
    reference = (
        reference_logits.detach()[:, :-1, :][shifted_mask]
        if reference_logits.ndim == 3 else reference_logits.detach()
    )
    if reference.shape != current.shape:
        raise ValueError("selected reference logits must match response positions and vocabulary")
    if current.dtype in (torch.float16, torch.bfloat16):
        current = current.float()
    if reference.dtype in (torch.float16, torch.bfloat16):
        reference = reference.float()
    reference = reference.to(device=current.device, dtype=current.dtype)

    current_log_probs = torch.log_softmax(current, dim=-1)
    reference_log_probs = torch.log_softmax(reference, dim=-1)
    reference_probs = reference_log_probs.exp()
    token_kl = (reference_probs * (reference_log_probs - current_log_probs)).sum(dim=-1)
    batch_indices = torch.arange(current_logits.shape[0], device=current.device).unsqueeze(1)
    batch_indices = batch_indices.expand_as(shifted_mask)[shifted_mask]
    example_kl_sum = torch.zeros(
        current_logits.shape[0],
        device=current.device,
        dtype=token_kl.dtype,
    ).scatter_add(0, batch_indices, token_kl)
    example_kl = example_kl_sum / response_lengths.to(dtype=token_kl.dtype)
    return example_kl.mean()


def benign_preservation_loss(
    model: nn.Module,
    parameters: TensorTree,
    benign_batch: SequenceBatch,
    reference_logits: Tensor,
    *,
    buffers: Mapping[str, Tensor] | None = None,
) -> Tensor:
    """Evaluate exact response-position benign KL for functional parameters."""

    _validate_sequence_batch(benign_batch, name="benign_batch")
    current_logits = _functional_logits(model, parameters, benign_batch, buffers=buffers)
    return benign_kl_from_logits(
        current_logits,
        reference_logits,
        benign_batch["response_mask"],
        attention_mask=benign_batch.get("attention_mask"),
    )


def make_safety_functional(
    model: nn.Module,
    harmful_batch: SequenceBatch,
    safe_batch: SequenceBatch,
    *,
    buffers: Mapping[str, Tensor] | None = None,
    chunk_size: int = 0,
) -> FunctionalMap:
    """Bind a fixed paired safety minibatch into a function of parameters."""

    _validate_sequence_batch(harmful_batch, name="harmful_batch")
    _validate_sequence_batch(safe_batch, name="safe_batch")

    if harmful_batch["input_ids"].shape[0] != safe_batch["input_ids"].shape[0]:
        raise ValueError("harmful and safe batches must have identical pair counts")
    blocks = []
    for (_, harmful), (_, safe) in zip(
        sequence_chunks(harmful_batch, chunk_size), sequence_chunks(safe_batch, chunk_size)
    ):
        def harmful_term(parameters, batch=harmful):
            return _sequence_functional(model, parameters, batch, buffers=buffers)
        def safe_term(parameters, batch=safe):
            return _sequence_functional(model, parameters, batch, buffers=buffers)
        blocks.append(FunctionalBlock(harmful["input_ids"].shape[0], ((1., harmful_term), (-1., safe_term))))
    return BatchedFunctional(tuple(blocks), tuple(harmful_batch.values()) + tuple(safe_batch.values()), model)


def make_benign_functional(
    model: nn.Module,
    benign_batch: SequenceBatch,
    *,
    buffers: Mapping[str, Tensor] | None = None,
    chunk_size: int = 0,
) -> FunctionalMap:
    """Bind a fixed benign minibatch into a function of parameters."""

    _validate_sequence_batch(benign_batch, name="benign_batch")

    blocks = []
    for _, batch in sequence_chunks(benign_batch, chunk_size):
        def term(parameters, batch=batch):
            return _sequence_functional(model, parameters, batch, buffers=buffers)
        blocks.append(FunctionalBlock(batch["input_ids"].shape[0], ((1., term),)))
    return BatchedFunctional(tuple(blocks), tuple(benign_batch.values()), model)
