from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data_processing.cbg_datasets import (
    CbgBenignDataset,
    CbgDefenseDataset,
    CbgPairedSafetyDataset,
    collate_cbg_defense_batch,
    construct_cbg_alpaca_dataset,
    construct_cbg_beavertails_datasets,
    deterministic_disjoint_indices,
    tokenize_paired_responses,
    tokenize_prompt_response,
    tokenize_seam_text,
)


class CharacterTokenizer:
    """Minimal deterministic causal tokenizer used without external packages."""

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        token_ids = [3 + ord(character) for character in text]
        if add_special_tokens:
            return [self.bos_token_id, *token_ids]
        return token_ids


def unsafe_record(index: int) -> dict:
    prompt = f"request-{index}"
    return {
        "prompt": prompt,
        "response": f"harm-{index}",
        "refusal": f"Question: {prompt}\nAnswer: safe-{index}",
        "is_safe": False,
        "category": {"violence,aiding_and_abetting,incitement": True},
    }


class CbgDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = CharacterTokenizer()

    def assert_sequence_contract(self, sequence: dict[str, torch.Tensor], max_length: int) -> None:
        self.assertEqual(
            set(sequence),
            {"input_ids", "attention_mask", "response_mask", "labels"},
        )
        for value in sequence.values():
            self.assertEqual(tuple(value.shape), (max_length,))
        self.assertEqual(sequence["input_ids"].dtype, torch.long)
        self.assertEqual(sequence["attention_mask"].dtype, torch.long)
        self.assertEqual(sequence["response_mask"].dtype, torch.bool)
        self.assertGreater(int(sequence["response_mask"].sum()), 0)
        self.assertFalse(
            bool(torch.any(sequence["response_mask"] & ~sequence["attention_mask"].bool()))
        )
        expected_labels = torch.where(
            sequence["response_mask"],
            sequence["input_ids"],
            torch.full_like(sequence["input_ids"], -100),
        )
        torch.testing.assert_close(sequence["labels"], expected_labels)

    def test_paired_tokenization_shares_prompt_and_marks_only_responses(self) -> None:
        prompt = "Question: shared\nAnswer:\n"
        harmful_text = "harmful"
        safe_text = "no"
        harmful, safe = tokenize_paired_responses(
            self.tokenizer,
            prompt,
            harmful_text,
            safe_text,
            max_length=64,
        )
        self.assert_sequence_contract(harmful, 64)
        self.assert_sequence_contract(safe, 64)

        harmful_start = int(torch.nonzero(harmful["response_mask"])[0])
        safe_start = int(torch.nonzero(safe["response_mask"])[0])
        self.assertEqual(harmful_start, safe_start)
        torch.testing.assert_close(
            harmful["input_ids"][:harmful_start],
            safe["input_ids"][:safe_start],
        )
        self.assertEqual(
            int(harmful["response_mask"].sum()),
            len(self.tokenizer.encode(harmful_text)),
        )
        self.assertEqual(
            int(safe["response_mask"].sum()),
            len(self.tokenizer.encode(safe_text)),
        )

    def test_overlong_pair_retains_a_shared_prefix_and_response_tokens(self) -> None:
        harmful, safe = tokenize_paired_responses(
            self.tokenizer,
            "a very long conditioning prompt",
            "h" * 30,
            "s" * 25,
            max_length=12,
        )
        harmful_start = int(torch.nonzero(harmful["response_mask"])[0])
        safe_start = int(torch.nonzero(safe["response_mask"])[0])
        self.assertEqual(harmful_start, 1)
        self.assertEqual(safe_start, 1)
        torch.testing.assert_close(
            harmful["input_ids"][:harmful_start],
            safe["input_ids"][:safe_start],
        )
        self.assertEqual(int(harmful["response_mask"].sum()), 11)
        self.assertEqual(int(safe["response_mask"].sum()), 11)

    def test_benign_tokenization_excludes_prompt_and_padding(self) -> None:
        sequence = tokenize_prompt_response(
            self.tokenizer,
            "Question: add\nAnswer: ",
            "four",
            max_length=40,
        )
        self.assert_sequence_contract(sequence, 40)
        response_positions = torch.nonzero(sequence["response_mask"]).flatten()
        self.assertTrue(bool(torch.all(sequence["attention_mask"][response_positions] == 1)))
        self.assertEqual(len(response_positions), len(self.tokenizer.encode("four")))

    def test_disjoint_splits_are_reproducible_and_strict(self) -> None:
        contiguous = deterministic_disjoint_indices(
            12,
            defense_size=5,
            attack_size=4,
        )
        self.assertEqual(contiguous.defense, (0, 1, 2, 3, 4))
        self.assertEqual(contiguous.attack, (5, 6, 7, 8))

        first = deterministic_disjoint_indices(
            20,
            defense_size=7,
            attack_size=6,
            seed=917,
            shuffle=True,
        )
        second = deterministic_disjoint_indices(
            20,
            defense_size=7,
            attack_size=6,
            seed=917,
            shuffle=True,
        )
        self.assertEqual(first, second)
        self.assertTrue(set(first.defense).isdisjoint(first.attack))
        with self.assertRaisesRegex(ValueError, "only 5 eligible"):
            deterministic_disjoint_indices(5, defense_size=4, attack_size=2)

    def test_beavertails_constructor_pairs_targets_and_reports_source_split(self) -> None:
        records = [unsafe_record(index) for index in range(7)]
        records.insert(
            2,
            {
                "prompt": "safe-source-row",
                "response": "okay",
                "refusal": "okay",
                "is_safe": True,
                "category": {},
            },
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "beavertails.json"
            with path.open("w", encoding="utf-8") as handle:
                json.dump(records, handle)
            defense, attack, split = construct_cbg_beavertails_datasets(
                self.tokenizer,
                path=path,
                defense_size=4,
                attack_size=3,
                max_length=64,
            )

        self.assertEqual(len(defense), 4)
        self.assertEqual(len(attack), 3)
        self.assertTrue(set(split.defense).isdisjoint(split.attack))
        # Source row 2 was safe and must appear in neither pool.
        self.assertNotIn(2, split.defense)
        self.assertNotIn(2, split.attack)
        first = defense[0]
        self.assertEqual(first["pair_id"], 0)
        safe_response_length = len(self.tokenizer.encode("safe-0"))
        self.assertEqual(int(first["safe"]["response_mask"].sum()), safe_response_length)

    def test_combined_dataset_and_collator_produce_functional_batches(self) -> None:
        safety = CbgPairedSafetyDataset(
            [unsafe_record(index) for index in range(3)],
            self.tokenizer,
            max_length=48,
        )
        benign_records = [
            {"instruction": f"instruction-{index}", "input": "", "output": f"answer-{index}"}
            for index in range(2)
        ]
        benign = CbgBenignDataset(benign_records, self.tokenizer, max_length=48)
        combined = CbgDefenseDataset(safety, benign)
        self.assertEqual(len(combined), 2)
        self.assertEqual(set(combined[0]), {"harmful", "safe", "benign", "pair_id"})

        loader = DataLoader(
            combined,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_cbg_defense_batch,
        )
        batch = next(iter(loader))
        self.assertEqual(set(batch), {"harmful", "safe", "benign", "pair_id"})
        for name in ("harmful", "safe", "benign"):
            self.assertEqual(tuple(batch[name]["input_ids"].shape), (2, 48))
            self.assertEqual(tuple(batch[name]["response_mask"].shape), (2, 48))
            self.assertEqual(batch[name]["response_mask"].dtype, torch.bool)
        torch.testing.assert_close(batch["pair_id"], torch.tensor([0, 1]))

    def test_alpaca_loader_is_injectable_and_selects_requested_prefix(self) -> None:
        calls: list[str] = []

        def loader(name: str):
            calls.append(name)
            return {
                "train": [
                    {"instruction": "one", "input": "", "output": "first"},
                    {"instruction": "two", "input": "context", "output": "second"},
                ]
            }

        dataset = construct_cbg_alpaca_dataset(
            self.tokenizer,
            retain_size=1,
            dataset_loader=loader,
            max_length=32,
        )
        self.assertEqual(calls, ["yahma/alpaca-cleaned"])
        self.assertEqual(len(dataset), 1)
        self.assertEqual(
            int(dataset[0]["response_mask"].sum()),
            len(self.tokenizer.encode("first")),
        )


if __name__ == "__main__":
    unittest.main()


class MergingTokenizer(CharacterTokenizer):
    """Character tokenizer that merges ":\\n" into one token, as a byte-level BPE does (Qwen2: ":Ċ")."""

    merged_id = 5000

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        token_ids, index = [], 0
        while index < len(text):
            if text.startswith(":\n", index):
                token_ids.append(self.merged_id)
                index += 2
            else:
                token_ids.append(3 + ord(text[index]))
                index += 1
        return [self.bos_token_id, *token_ids] if add_special_tokens else token_ids


class SeamTokenizationTest(unittest.TestCase):
    def test_prompt_token_prefix_marks_the_tokens_after_the_prompt(self) -> None:
        tokenizer = CharacterTokenizer()
        prompt, full = "Question: q\nAnswer:", "Question: q\nAnswer: yes"
        sequence = tokenize_seam_text(tokenizer, prompt, full, max_length=32)
        ids = [tokenizer.bos_token_id, *tokenizer.encode(full)]
        self.assertEqual(sequence["input_ids"][:len(ids)].tolist(), ids)
        self.assertEqual(sequence["response_mask"].tolist(),
                         [False] * (1 + len(prompt)) + [True] * len(" yes") + [False] * (32 - len(ids)))

    def test_token_merged_across_the_boundary_is_the_first_response_token(self) -> None:
        tokenizer = MergingTokenizer()
        prompt, full = "Question: q\nAnswer:", "Question: q\nAnswer:\nharm"
        sequence = tokenize_seam_text(tokenizer, prompt, full, max_length=32)
        ids = [tokenizer.bos_token_id, *tokenizer.encode(full)]   # one pass, as SEAM: ":\n" stays one token
        self.assertEqual(sequence["input_ids"][:len(ids)].tolist(), ids)
        first_response = ids.index(MergingTokenizer.merged_id)
        self.assertEqual(first_response, len(prompt))             # BOS + the prompt without its final ":"
        self.assertEqual(sequence["response_mask"][:len(ids)].tolist(),
                         [False] * first_response + [True] * (len(ids) - first_response))

    def test_full_text_that_does_not_continue_the_prompt_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            tokenize_seam_text(MergingTokenizer(), "Question: a\nAnswer:", "Question: b\nAnswer:\nharm", max_length=32)
