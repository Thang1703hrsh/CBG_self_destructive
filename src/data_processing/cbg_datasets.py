"""CBG-specific paired safety and benign dataset construction.

This module deliberately does not alter the legacy SEAM preprocessing.  CBG
needs stronger data invariants than SEAM's three independently tokenized
datasets provide:

* the harmful and safe targets are attached to the same prompt and index;
* response masks mark target tokens (not prompt or padding tokens); and
* the defense and downstream harmful-attack pools are explicitly disjoint.

The returned nested dictionaries can be passed directly to the functional
maps in :mod:`src.cbg.functionals` after collation::

    {
        "harmful": {"input_ids": ..., "attention_mask": ..., "response_mask": ...},
        "safe": {"input_ids": ..., "attention_mask": ..., "response_mask": ...},
        "benign": {"input_ids": ..., "attention_mask": ..., "response_mask": ...},
    }

Hugging Face ``datasets`` is imported only by the Alpaca loader.  The core
dataset and tokenization helpers therefore remain unit-testable without a
network connection or that optional runtime dependency.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, TypeAlias

import torch
from torch import Tensor
from torch.utils.data import Dataset


CONTEXT_LENGTH = 256
DEFAULT_BEVERTAILS_PATH = Path("data/beavertails_with_refusals_train.json")
DEFAULT_ALPACA_DATASET = "yahma/alpaca-cleaned"
DEFAULT_ALPACA_SPLIT = "train"

SAFETY_PROMPT_TEMPLATE = "Question: {prompt}\nAnswer:\n"
BENIGN_PROMPT_TEMPLATE = "Question: {prompt}\nAnswer: "

SequenceExample: TypeAlias = dict[str, Tensor]
PairedSafetyExample: TypeAlias = dict[str, Any]
DefenseExample: TypeAlias = dict[str, Any]


# Kept local so importing this module does not import the legacy module, whose
# top-level Hugging Face datasets import is intentionally eager.
BT_CATEGORIES: dict[str, tuple[str, ...]] = {
    "half": (
        "violence,aiding_and_abetting,incitement",
        "hate_speech,offensive_language",
        "discrimination,stereotype,injustice",
        "misinformation_regarding_ethics,laws_and_safety",
        "privacy_violation",
    ),
    "other_half": (
        "financial_crime,property_crime,theft",
        "drug_abuse,weapons,banned_substance",
        "terrorism,organized_crime",
        "sexually_explicit,adult_content",
    ),
    "violence": ("violence,aiding_and_abetting,incitement",),
    "toxic": (
        "hate_speech,offensive_language",
        "discrimination,stereotype,injustice",
    ),
    "privacy_misinformation": (
        "misinformation_regarding_ethics,laws_and_safety",
        "privacy_violation",
    ),
    "crime_unethical": (
        "financial_crime,property_crime,theft",
        "drug_abuse,weapons,banned_substance",
        "non_violent_unethical_behavior",
        "terrorism,organized_crime",
    ),
    "sexually_explicit": ("sexually_explicit,adult_content",),
    "self_harm": ("self_harm",),
}


@dataclass(frozen=True)
class DisjointSplitIndices:
    """Stable source indices for non-overlapping defense and attack pools."""

    defense: tuple[int, ...]
    attack: tuple[int, ...]

    def __post_init__(self) -> None:
        defense_set = set(self.defense)
        attack_set = set(self.attack)
        if len(defense_set) != len(self.defense):
            raise ValueError("defense indices contain duplicates")
        if len(attack_set) != len(self.attack):
            raise ValueError("attack indices contain duplicates")
        overlap = defense_set.intersection(attack_set)
        if overlap:
            raise ValueError(f"defense and attack indices overlap: {sorted(overlap)}")


def deterministic_disjoint_indices(
    num_examples: int,
    *,
    defense_size: int,
    attack_size: int,
    seed: int = 0,
    shuffle: bool = False,
) -> DisjointSplitIndices:
    """Choose deterministic, strictly disjoint defense and attack indices.

    With the default ``shuffle=False``, the defense pool is the first block and
    the attack pool is the immediately following block.  ``shuffle=True`` uses
    a local seeded RNG and never changes process-global random state.

    A request larger than the available source fails instead of silently
    shortening a pool or allowing overlap.
    """

    for name, value in (
        ("num_examples", num_examples),
        ("defense_size", defense_size),
        ("attack_size", attack_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    requested = defense_size + attack_size
    if requested > num_examples:
        raise ValueError(
            f"requested {defense_size} defense + {attack_size} attack examples "
            f"from only {num_examples} eligible records"
        )

    indices = list(range(num_examples))
    if shuffle:
        random.Random(seed).shuffle(indices)
    return DisjointSplitIndices(
        defense=tuple(indices[:defense_size]),
        attack=tuple(indices[defense_size:requested]),
    )


def _as_token_ids(value: Any) -> list[int]:
    if isinstance(value, Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise TypeError("tokenizer input_ids must be a tensor, tuple, or list")
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("expected tokenization of one string")
        value = value[0]
    return [int(token_id) for token_id in value]


def _encode_without_special_tokens(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        return _as_token_ids(tokenizer.encode(text, add_special_tokens=False))
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise TypeError("tokenizer must return a mapping containing input_ids")
    return _as_token_ids(encoded["input_ids"])


def _prompt_token_ids(tokenizer: Any, prompt_text: str, *, add_bos: bool) -> list[int]:
    token_ids = _encode_without_special_tokens(tokenizer, prompt_text)
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if add_bos and bos_token_id is not None:
        token_ids.insert(0, int(bos_token_id))
    return token_ids


def _response_token_ids(tokenizer: Any, response_text: str, *, append_eos: bool) -> list[int]:
    token_ids = _encode_without_special_tokens(tokenizer, response_text)
    if append_eos:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            raise ValueError("append_eos=True requires tokenizer.eos_token_id")
        token_ids.append(int(eos_token_id))
    if not token_ids:
        raise ValueError("response must produce at least one target token")
    return token_ids


def _pad_token_id(tokenizer: Any) -> int:
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        raise ValueError("tokenizer must define pad_token_id or eos_token_id")
    return int(pad_token_id)


def _build_sequence(
    tokenizer: Any,
    prompt_ids: Sequence[int],
    response_ids: Sequence[int],
    *,
    max_length: int,
) -> SequenceExample:
    if max_length < 2:
        raise ValueError("max_length must leave room for prompt and response tokens")
    if not prompt_ids:
        raise ValueError("prompt must produce at least one conditioning token")
    if not response_ids:
        raise ValueError("response must contain at least one target token")

    # The caller establishes the prompt budget.  Preserve the beginning of the
    # response under truncation, matching ordinary right-side causal-LM
    # truncation while guaranteeing at least one conditioning token.
    retained_prompt = list(prompt_ids)
    response_budget = max_length - len(retained_prompt)
    if response_budget <= 0:
        raise ValueError("prompt token budget must leave room for a response token")
    retained_response = list(response_ids[:response_budget])
    if not retained_response:
        raise ValueError("truncation removed every response token")

    token_count = len(retained_prompt) + len(retained_response)
    padding_count = max_length - token_count
    input_ids = retained_prompt + retained_response + [_pad_token_id(tokenizer)] * padding_count
    attention_mask = [1] * token_count + [0] * padding_count
    response_mask = (
        [False] * len(retained_prompt)
        + [True] * len(retained_response)
        + [False] * padding_count
    )
    labels = [token_id if is_response else -100 for token_id, is_response in zip(input_ids, response_mask)]
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "response_mask": torch.tensor(response_mask, dtype=torch.bool),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def tokenize_prompt_response(
    tokenizer: Any,
    prompt_text: str,
    response_text: str,
    *,
    max_length: int = CONTEXT_LENGTH,
    append_eos: bool = False,
    add_bos: bool = True,
) -> SequenceExample:
    """Tokenize one conditional response with an explicit target-token mask."""

    prompt_ids = _prompt_token_ids(tokenizer, prompt_text, add_bos=add_bos)
    response_ids = _response_token_ids(tokenizer, response_text, append_eos=append_eos)
    # Retain the response and the suffix of an overlong prompt.  Keeping the
    # prompt suffix preserves the answer delimiter immediately before target 1.
    prompt_budget = max(1, max_length - len(response_ids))
    prompt_ids = prompt_ids[-prompt_budget:]
    return _build_sequence(
        tokenizer,
        prompt_ids,
        response_ids,
        max_length=max_length,
    )


def tokenize_paired_responses(
    tokenizer: Any,
    prompt_text: str,
    harmful_response: str,
    safe_response: str,
    *,
    max_length: int = CONTEXT_LENGTH,
    append_eos: bool = False,
    add_bos: bool = True,
) -> tuple[SequenceExample, SequenceExample]:
    """Tokenize a safety pair while preserving an identical prompt prefix.

    A shared prompt budget is derived from the longer response.  This prevents
    differing harmful/safe response lengths from changing the represented
    conditioning prompt, an important invariant for the margin in ``F_h``.
    """

    prompt_ids = _prompt_token_ids(tokenizer, prompt_text, add_bos=add_bos)
    harmful_ids = _response_token_ids(tokenizer, harmful_response, append_eos=append_eos)
    safe_ids = _response_token_ids(tokenizer, safe_response, append_eos=append_eos)
    prompt_budget = max(1, max_length - max(len(harmful_ids), len(safe_ids)))
    shared_prompt_ids = prompt_ids[-prompt_budget:]
    harmful = _build_sequence(
        tokenizer,
        shared_prompt_ids,
        harmful_ids,
        max_length=max_length,
    )
    safe = _build_sequence(
        tokenizer,
        shared_prompt_ids,
        safe_ids,
        max_length=max_length,
    )
    return harmful, safe


def _extract_refusal_response(refusal: str, prompt: str) -> str:
    """Extract the generated target when the JSON stores a formatted dialogue."""

    refusal = refusal.strip()
    formatted_prefix = f"Question: {prompt}\nAnswer:"
    if refusal.startswith(formatted_prefix):
        return refusal[len(formatted_prefix):].lstrip()
    # Accommodate refusal files produced with minor whitespace changes while
    # avoiding an unrestricted split on response text that contains "Answer:".
    if refusal.startswith("Question:") and "\nAnswer:" in refusal:
        return refusal.split("\nAnswer:", maxsplit=1)[1].lstrip()
    return refusal


def _active_categories(record: Mapping[str, Any]) -> set[str]:
    categories = record.get("category", {})
    if isinstance(categories, Mapping):
        return {str(name) for name, enabled in categories.items() if enabled}
    if isinstance(categories, str):
        return {categories}
    if isinstance(categories, Sequence):
        return {str(category) for category in categories}
    return set()


def _normalize_withheld_subsets(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value else ()
    return tuple(value)


def filter_beavertails_records(
    records: Sequence[Mapping[str, Any]],
    *,
    harmful_subset: str = "all",
    withhold_harmful_subsets: str | Sequence[str] = (),
) -> tuple[list[Mapping[str, Any]], list[int]]:
    """Filter unsafe paired records and retain their original source indices."""

    if harmful_subset != "all" and harmful_subset not in BT_CATEGORIES:
        raise KeyError(f"unknown BeaverTails subset: {harmful_subset!r}")
    withheld = _normalize_withheld_subsets(withhold_harmful_subsets)
    unknown_withheld = set(withheld) - set(BT_CATEGORIES)
    if unknown_withheld:
        raise KeyError(f"unknown withheld BeaverTails subsets: {sorted(unknown_withheld)}")

    allowed: set[str] | None
    allowed = None if harmful_subset == "all" else set(BT_CATEGORIES[harmful_subset])
    withheld_categories = {
        category
        for subset in withheld
        for category in BT_CATEGORIES[subset]
    }

    filtered: list[Mapping[str, Any]] = []
    source_indices: list[int] = []
    for source_index, record in enumerate(records):
        if record.get("is_safe") is True:
            continue
        if not all(key in record for key in ("prompt", "response", "refusal")):
            continue
        categories = _active_categories(record)
        if categories.intersection(withheld_categories):
            continue
        if allowed is not None and not categories.intersection(allowed):
            continue
        if not str(record["response"]).strip() or not str(record["refusal"]).strip():
            continue
        filtered.append(record)
        source_indices.append(source_index)
    return filtered, source_indices


class CbgPairedSafetyDataset(Dataset[PairedSafetyExample]):
    """Lazy tokenizer-backed BeaverTails harmful/refusal pair dataset."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        tokenizer: Any,
        *,
        source_indices: Sequence[int] | None = None,
        max_length: int = CONTEXT_LENGTH,
        prompt_template: str = SAFETY_PROMPT_TEMPLATE,
        append_eos: bool = False,
        add_bos: bool = True,
    ) -> None:
        if source_indices is None:
            source_indices = tuple(range(len(records)))
        if len(records) != len(source_indices):
            raise ValueError("records and source_indices must have the same length")
        self.records = records
        self.tokenizer = tokenizer
        self.source_indices = tuple(int(index) for index in source_indices)
        self.max_length = max_length
        self.prompt_template = prompt_template
        self.append_eos = append_eos
        self.add_bos = add_bos

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> PairedSafetyExample:
        record = self.records[index]
        prompt = str(record["prompt"])
        harmful_response = str(record["response"]).strip()
        safe_response = _extract_refusal_response(str(record["refusal"]), prompt)
        prompt_text = self.prompt_template.format(prompt=prompt)
        harmful, safe = tokenize_paired_responses(
            self.tokenizer,
            prompt_text,
            harmful_response,
            safe_response,
            max_length=self.max_length,
            append_eos=self.append_eos,
            add_bos=self.add_bos,
        )
        return {
            "harmful": harmful,
            "safe": safe,
            "pair_id": self.source_indices[index],
        }


class CbgBenignDataset(Dataset[SequenceExample]):
    """Lazy tokenizer-backed Alpaca benign-response dataset."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        tokenizer: Any,
        *,
        max_length: int = CONTEXT_LENGTH,
        prompt_template: str = BENIGN_PROMPT_TEMPLATE,
        append_eos: bool = False,
        add_bos: bool = True,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template
        self.append_eos = append_eos
        self.add_bos = add_bos

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> SequenceExample:
        record = self.records[index]
        instruction = str(record.get("instruction", ""))
        input_text = str(record.get("input", ""))
        output_text = str(record.get("output", "")).strip()
        if not output_text:
            raise ValueError(f"Alpaca example {index} has an empty output")
        prompt = instruction + "\n" + input_text
        prompt_text = self.prompt_template.format(prompt=prompt)
        return tokenize_prompt_response(
            self.tokenizer,
            prompt_text,
            output_text,
            max_length=self.max_length,
            append_eos=self.append_eos,
            add_bos=self.add_bos,
        )


class CbgDefenseDataset(Dataset[DefenseExample]):
    """Lazy index-wise combination used by one standard training DataLoader."""

    def __init__(
        self,
        safety_dataset: Dataset[PairedSafetyExample],
        benign_dataset: Dataset[SequenceExample],
    ) -> None:
        self.safety_dataset = safety_dataset
        self.benign_dataset = benign_dataset

    def __len__(self) -> int:
        return min(len(self.safety_dataset), len(self.benign_dataset))

    def __getitem__(self, index: int) -> DefenseExample:
        safety = self.safety_dataset[index]
        return {
            "harmful": safety["harmful"],
            "safe": safety["safe"],
            "benign": self.benign_dataset[index],
            "pair_id": safety["pair_id"],
        }


def _collate_sequences(examples: Sequence[Mapping[str, Tensor]]) -> dict[str, Tensor]:
    if not examples:
        raise ValueError("cannot collate an empty batch")
    required = ("input_ids", "attention_mask", "response_mask")
    missing = [key for key in required if any(key not in example for example in examples)]
    if missing:
        raise KeyError(f"sequence examples are missing fields: {sorted(set(missing))}")
    keys = required + (("labels",) if all("labels" in example for example in examples) else ())
    return {key: torch.stack([example[key] for example in examples]) for key in keys}


def collate_cbg_safety_batch(examples: Sequence[PairedSafetyExample]) -> dict[str, Any]:
    """Collate paired examples into separate functional-map sequence batches."""

    if not examples:
        raise ValueError("cannot collate an empty batch")
    return {
        "harmful": _collate_sequences([example["harmful"] for example in examples]),
        "safe": _collate_sequences([example["safe"] for example in examples]),
        "pair_id": torch.tensor([int(example["pair_id"]) for example in examples]),
    }


def collate_cbg_defense_batch(examples: Sequence[DefenseExample]) -> dict[str, Any]:
    """Collate combined safety/benign examples for ``CbgTrainer``."""

    batch = collate_cbg_safety_batch(examples)
    batch["benign"] = _collate_sequences([example["benign"] for example in examples])
    return batch


def load_beavertails_refusal_records(
    path: str | PathLike[str] = DEFAULT_BEVERTAILS_PATH,
) -> list[Mapping[str, Any]]:
    """Load the same refusal-enriched BeaverTails JSON used by SEAM."""

    resolved_path = Path(path)
    with resolved_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not all(isinstance(record, Mapping) for record in records):
        raise ValueError(f"{resolved_path} must contain a JSON list of objects")
    return records


def load_manifest_attack_records(path, manifest_path, *, attack_size):
    """Select exact held-out source rows and verify the defense manifest.

    Source-index disjointness does not establish semantic/prompt deduplication.
    """
    import hashlib
    if isinstance(attack_size, bool) or not isinstance(attack_size, int) or attack_size < 1:
        raise ValueError("manifest attack_size must be a positive integer")
    with Path(manifest_path).open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    with Path(path).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if manifest.get("safety_source_sha256") != digest:
        raise ValueError("evaluation source hash differs from the CBG defense manifest")
    records = load_beavertails_refusal_records(path)
    groups = []
    for key in ("defense_indices", "attack_indices"):
        indices = manifest.get(key)
        if not isinstance(indices, list) or any(
            isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(records) for i in indices
        ) or len(set(indices)) != len(indices):
            raise ValueError(f"invalid manifest {key}")
        groups.append(indices)
    if set(groups[0]) & set(groups[1]):
        raise ValueError("manifest defense and attack indices overlap")
    if attack_size > len(groups[1]):
        raise ValueError("attack_size exceeds held-out manifest pool")
    chosen = [records[i] for i in groups[1][:attack_size]]
    if any(record.get("is_safe") is not False for record in chosen):
        raise ValueError("manifest attack pool must contain harmful examples")
    return chosen


def construct_cbg_beavertails_datasets(
    tokenizer: Any,
    *,
    path: str | PathLike[str] = DEFAULT_BEVERTAILS_PATH,
    defense_size: int = 4000,
    attack_size: int = 1000,
    seed: int = 0,
    shuffle: bool = False,
    harmful_subset: str = "all",
    withhold_harmful_subsets: str | Sequence[str] = (),
    max_length: int = CONTEXT_LENGTH,
    append_eos: bool = False,
) -> tuple[CbgPairedSafetyDataset, CbgPairedSafetyDataset, DisjointSplitIndices]:
    """Build disjoint defense and attack paired datasets from BeaverTails."""

    records = load_beavertails_refusal_records(path)
    eligible_records, eligible_source_indices = filter_beavertails_records(
        records,
        harmful_subset=harmful_subset,
        withhold_harmful_subsets=withhold_harmful_subsets,
    )
    local_split = deterministic_disjoint_indices(
        len(eligible_records),
        defense_size=defense_size,
        attack_size=attack_size,
        seed=seed,
        shuffle=shuffle,
    )

    def select(indices: Sequence[int]) -> CbgPairedSafetyDataset:
        return CbgPairedSafetyDataset(
            [eligible_records[index] for index in indices],
            tokenizer,
            source_indices=[eligible_source_indices[index] for index in indices],
            max_length=max_length,
            append_eos=append_eos,
        )

    defense_dataset = select(local_split.defense)
    attack_dataset = select(local_split.attack)
    source_split = DisjointSplitIndices(
        defense=tuple(eligible_source_indices[index] for index in local_split.defense),
        attack=tuple(eligible_source_indices[index] for index in local_split.attack),
    )
    return defense_dataset, attack_dataset, source_split


def construct_cbg_alpaca_dataset(
    tokenizer: Any,
    *,
    retain_size: int = 4000,
    dataset_name: str = DEFAULT_ALPACA_DATASET,
    split: str = DEFAULT_ALPACA_SPLIT,
    max_length: int = CONTEXT_LENGTH,
    append_eos: bool = False,
    dataset_loader: Callable[..., Any] | None = None,
) -> CbgBenignDataset:
    """Load the SEAM Alpaca source and expose response-masked examples.

    ``dataset_loader`` is injectable for offline tests.  Production defaults to
    :func:`datasets.load_dataset` and imports it lazily.
    """

    if retain_size < 0:
        raise ValueError("retain_size must be non-negative")
    if dataset_loader is None:
        try:
            from datasets import load_dataset as dataset_loader
        except ImportError as error:  # pragma: no cover - exercised in deployment setup
            raise ImportError(
                "construct_cbg_alpaca_dataset requires the 'datasets' package"
            ) from error
    loaded = dataset_loader(dataset_name)
    records = loaded[split] if isinstance(loaded, Mapping) else loaded
    selected_size = min(retain_size, len(records))
    if hasattr(records, "select"):
        records = records.select(range(selected_size))
    else:
        records = [records[index] for index in range(selected_size)]
    return CbgBenignDataset(
        records,
        tokenizer,
        max_length=max_length,
        append_eos=append_eos,
    )


def construct_cbg_defense_dataset(
    safety_dataset: Dataset[PairedSafetyExample],
    benign_dataset: Dataset[SequenceExample],
) -> CbgDefenseDataset:
    """Construct the lazy combined dataset consumed by CBG training."""

    return CbgDefenseDataset(safety_dataset, benign_dataset)


class CbgJointBatchDataset(Dataset):
    """One item is m safety pairs plus n independent benign sequences.

    Fixed independent permutations make selection reproducible across resume
    and DataLoader workers. The shorter pool cycles; no examples are packed.
    Trainer shuffles the joint-batch item order each epoch.
    """
    def __init__(self, safety_dataset, benign_dataset, *, m, n, seed=0):
        if min(m, n) < 1 or len(safety_dataset) < m or len(benign_dataset) < n:
            raise ValueError("each dataset must contain at least its configured batch size")
        self.safety, self.benign, self.m, self.n = safety_dataset, benign_dataset, m, n
        rng = random.Random(seed)
        self.safety_order = list(range(len(safety_dataset)))
        self.benign_order = list(range(len(benign_dataset)))
        rng.shuffle(self.safety_order)
        rng.shuffle(self.benign_order)

    def __len__(self):
        return max((len(self.safety) + self.m - 1) // self.m,
                   (len(self.benign) + self.n - 1) // self.n)

    def __getitem__(self, index):
        safety = [self.safety[self.safety_order[(index*self.m+i) % len(self.safety)]]
                  for i in range(self.m)]
        benign = [self.benign[self.benign_order[(index*self.n+i) % len(self.benign)]]
                  for i in range(self.n)]
        batch = collate_cbg_safety_batch(safety)
        batch["benign"] = _collate_sequences(benign)
        return batch


def collate_cbg_joint_batch(items):
    if len(items) != 1:
        raise ValueError("CBG joint batches require DataLoader batch_size=1")
    return items[0]


__all__ = [
    "BENIGN_PROMPT_TEMPLATE",
    "BT_CATEGORIES",
    "CONTEXT_LENGTH",
    "CbgBenignDataset",
    "CbgDefenseDataset",
    "CbgJointBatchDataset",
    "CbgPairedSafetyDataset",
    "DEFAULT_ALPACA_DATASET",
    "DEFAULT_BEVERTAILS_PATH",
    "DisjointSplitIndices",
    "SAFETY_PROMPT_TEMPLATE",
    "collate_cbg_defense_batch",
    "collate_cbg_joint_batch",
    "collate_cbg_safety_batch",
    "construct_cbg_alpaca_dataset",
    "construct_cbg_beavertails_datasets",
    "construct_cbg_defense_dataset",
    "deterministic_disjoint_indices",
    "filter_beavertails_records",
    "load_beavertails_refusal_records",
    "load_manifest_attack_records",
    "tokenize_paired_responses",
    "tokenize_prompt_response",
]
