"""CPU-only checks for SDD's lazy SentenceBERT matching loop."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from modal_sdd_full import _match_irrelevant_answers, _sha256, _validate_preflight


class FakeEncoder:
    def __init__(self):
        self.encoded = []

    def encode(self, texts, **_kwargs):
        self.encoded.extend(texts)
        vectors = {"related": [1.0, 0.0], "unrelated": [0.0, 1.0]}
        return np.asarray([vectors[text] for text in texts], dtype=np.float32)


class SddMatchingTests(unittest.TestCase):
    def test_resamples_until_answer_is_irrelevant(self):
        selected = [(f"prompt-{index}", "category") for index in range(20)]
        prompts = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (20, 1))
        encoder = FakeEncoder()
        pairs, encoded_count = _match_irrelevant_answers(
            selected, ["related", "unrelated"], prompts, encoder, seed=42,
        )
        self.assertEqual(len(pairs), 20)
        self.assertTrue(all(pair["completion"] == "unrelated" for pair in pairs))
        self.assertTrue(all(pair["cosine_sim"] <= 0.25 for pair in pairs))
        self.assertLessEqual(encoded_count, 2)
        self.assertTrue(any(pair["resample_tries"] > 1 for pair in pairs))

    def test_fails_closed_when_no_irrelevant_answer_exists(self):
        encoder = FakeEncoder()
        with self.assertRaisesRegex(ValueError, "Could not find irrelevant answers"):
            _match_irrelevant_answers(
                [("prompt", "category")], ["related"],
                np.array([[1.0, 0.0]], dtype=np.float32), encoder,
                seed=42, max_resample=3,
            )


class SddPreflightTests(unittest.TestCase):
    def test_full_run_accepts_only_matching_completed_preflight(self):
        with tempfile.TemporaryDirectory() as temp:
            data_dir = Path(temp) / "data"
            output = Path(temp) / "results"
            data_dir.mkdir()
            output.mkdir()
            (data_dir / "attack.json").write_text("[]", encoding="utf-8")
            (data_dir / "ctrap_defense.json").write_text("[]", encoding="utf-8")
            (data_dir / "sdd_pairs.jsonl").write_text("{\"prompt\":\"a\"}\n{\"prompt\":\"b\"}\n", encoding="utf-8")
            (data_dir / "sdd_data_manifest.json").write_text(
                json.dumps({"pairs": 2, "target_pairs": 2,
                            "paper_equivalent_answer_pool": False}), encoding="utf-8")
            files = ("attack.json", "ctrap_defense.json", "sdd_pairs.jsonl", "sdd_data_manifest.json")
            manifest = {
                "seed": 42, "run_id": "test-1", "answer_pool": "AlpacaOnly",
                "split": {"attack_count": 1000, "source_sha256": "fixture"},
                "sha256": {name: _sha256(data_dir / name) for name in files},
            }
            (data_dir / "preflight_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            status = {"mode": "data_preflight", "state": "complete", "seed": 42,
                      "run_id": "test-1", "answer_pool": "AlpacaOnly"}
            (output / "status.json").write_text(json.dumps(status), encoding="utf-8")

            self.assertEqual(_validate_preflight(42, "test-1", "AlpacaOnly",
                                                 data_dir, output, expected_pairs=2), manifest)
            with self.assertRaisesRegex(ValueError, "does not match"):
                _validate_preflight(42, "test-1", "AlpacaLima", data_dir, output, expected_pairs=2)
            (data_dir / "sdd_pairs.jsonl").write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing or changed"):
                _validate_preflight(42, "test-1", "AlpacaOnly", data_dir, output, expected_pairs=2)


if __name__ == "__main__":
    unittest.main()
