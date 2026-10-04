"""Offline checks of evaluation invariants; no model downloads or Modal jobs."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from defense_eval_common import (canonical_hash, encode_pair, encode_prompt, harmful_rows,
                                 make_cases, prompt_key, resolve_config, seam_attack_rows,
                                 seam_test_rows, select_rows, training_rows, volume_path,
                                 write_json)
from defense_eval_worker import pad_training_batch


def config_fixture():
    models = {"base": {"source": {"kind": "hf", "repo_id": "owner/base"}}}
    for label in ("sdd", "ctrap", "seam"):
        models[label] = {"source": {"kind": "hf", "repo_id": f"owner/{label}"},
                         "defense_data": [{"kind": "volume", "volume": "data",
                                           "path": f"{label}/defense.json"}]}
    return {"models": models, "data": {
        name: {"kind": "dataset", "repo_id": "owner/data", "split": "train"}
        for name in ("attack", "benign", "test")}}


class Tokenizer:
    eos_token_id = 2

    def __call__(self, text, add_special_tokens=True):
        ids = [10 + ord(character) for character in text]
        return SimpleNamespace(input_ids=([1] if add_special_tokens else []) + ids)


class PaddingTokenizer(Tokenizer):
    """Minimal HF-style tokenizer: BOS, truncation, right padding with eos."""

    def __call__(self, text, add_special_tokens=True, truncation=False, max_length=None,
                 padding=False):
        ids = super().__call__(text, add_special_tokens).input_ids
        if truncation:
            ids = ids[:max_length]
        mask = [1] * len(ids)
        if padding == "max_length":
            mask += [0] * (max_length - len(ids))
            ids = ids + [self.eos_token_id] * (max_length - len(ids))
        return {"input_ids": ids, "attention_mask": mask} if truncation or padding else             SimpleNamespace(input_ids=ids)


class SeamOriginalTests(unittest.TestCase):
    def test_test_rows_follow_seam_filter_and_order(self):
        rows = [
            {"prompt": "a", "is_safe": False, "category": {"animal_abuse": True, "self_harm": True}},
            {"prompt": "b", "is_safe": True, "category": {"self_harm": False}},
            {"prompt": "c", "is_safe": False, "category": {"self_harm": False, "privacy_violation": True}},
            {"prompt": "d", "is_safe": False, "category": {"self_harm": True}},
        ]
        # SEAM zips prompts with categories of category-bearing rows only (b has none):
        # a<-animal_abuse (dropped), b<-privacy (safe, dropped), c<-self_harm, d unpaired.
        selected = seam_test_rows(rows, 1)
        self.assertEqual([(r["prompt"], r["categories"]) for r in selected], [("c", ["self_harm"])])
        with self.assertRaises(ValueError):
            seam_test_rows(rows, 2)

    def test_attack_rows_keep_file_order(self):
        rows = [{"prompt": f"p{i}", "response": f"r{i}", "is_safe": False,
                 "category": {"self_harm": True}} for i in range(5)]
        rows.insert(1, {"prompt": "safe", "response": "x", "is_safe": True, "category": {}})
        self.assertEqual([r["prompt"] for r in seam_attack_rows(rows, 3)], ["p0", "p1", "p2"])

    def test_seam_padded_learns_padding_like_seam(self):
        row = {"prompt": "hi", "response": "ok"}
        encoded = encode_pair(PaddingTokenizer(), row, 64, "seam_padded")
        self.assertEqual(len(encoded["input_ids"]), 64)
        self.assertEqual(encoded["labels"], encoded["input_ids"])
        self.assertIn(0, encoded["attention_mask"])
        self.assertEqual(encoded["input_ids"][-1], PaddingTokenizer.eos_token_id)
        self.assertFalse(encoded["prompt_truncated"])

    def test_config_accepts_seam_original_protocol(self):
        config = config_fixture()
        config["protocol"] = {"label_scope": "seam_padded", "data_selection": "seam_original"}
        resolved = resolve_config(config, "robustness", "base,seam")
        self.assertEqual(resolved["protocol"]["data_selection"], "seam_original")
        config["protocol"]["data_selection"] = "other"
        with self.assertRaises(ValueError):
            resolve_config(config, "robustness", "base,seam")


class ProtocolTests(unittest.TestCase):
    def test_suites_have_independent_attacks_and_baseline(self):
        self.assertEqual([case["id"] for case in make_cases("core")], [
            "clean", "harmful-lr2e-5", "harmful-lr5e-5", "harmful-lr1e-4",
            "benign-lr2e-5", "mixed10-lr5e-5"])
        self.assertEqual(len(make_cases("full")), 9)
        self.assertEqual(len(make_cases("robustness")), 6)
        paper = [case["id"] for case in make_cases("paper")]
        self.assertEqual(len(paper), 8)
        self.assertFalse(any(case.startswith("mixed") for case in paper))
        resolved = resolve_config(config_fixture(), models="seam")
        self.assertEqual(list(resolved["models"]), ["seam"])
        default = resolve_config(config_fixture())
        self.assertEqual(list(default["models"]), ["sdd", "ctrap", "seam"])
        self.assertEqual(len(default["models"]) * len(default["cases"]), 18)
        self.assertEqual(list(resolve_config(config_fixture(), models="seam,base")["models"]),
                         ["base", "seam"])
        self.assertEqual(resolved["protocol"]["utility_limit"], None)

    def test_smoke_cannot_be_mistaken_for_full_budget(self):
        config = resolve_config(config_fixture(), "smoke")
        self.assertEqual(config["protocol"]["steps"], 2)
        self.assertEqual(config["protocol"]["utility_limit"], 16)
        self.assertEqual(config["protocol"]["test_size"], 16)
        self.assertEqual(len(config["cases"]), 2)

    def test_missing_defense_and_placeholder_models_fail_before_gpu(self):
        fixture = config_fixture()
        fixture["models"]["sdd"]["defense_data"] = []
        with self.assertRaisesRegex(ValueError, "defense_data"):
            resolve_config(fixture)
        fixture = config_fixture()
        fixture["models"]["sdd"]["source"]["repo_id"] = "REPLACE_OWNER/model"
        with self.assertRaisesRegex(ValueError, "repo_id"):
            resolve_config(fixture)
        # Unselected placeholders do not prevent a Base/SEAM smoke test.
        self.assertEqual(list(resolve_config(fixture, models="base,seam")["models"]), ["base", "seam"])

    def test_paths_and_protocol_parameters_fail_closed(self):
        for path in ("../other", "/absolute", "", "REPLACE_RUN/file.json"):
            with self.assertRaises(ValueError):
                volume_path({"volume": "data", "path": path})
        fixture = config_fixture()
        fixture["protocol"] = {"steps": 0}
        with self.assertRaisesRegex(ValueError, "steps"):
            resolve_config(fixture)

    def test_normalized_overlap_and_all_categories_stay_aligned(self):
        rows = [{"is_safe": True, "prompt": "safe", "category": {}}]
        names = ["animal_abuse", "child_abuse", "controversial_topics,politics"]
        for index, name in enumerate(names):
            rows.append({"is_safe": False, "prompt": f"unsafe {index}",
                         "response": "answer", "category": {name: True}})
        rows.append({"is_safe": False, "prompt": " UNSAFE   0 ", "response": "other",
                     "category": {"privacy_violation": True}})
        harmful = harmful_rows(rows)
        self.assertEqual(len(harmful), 3)
        self.assertEqual(harmful[0]["categories"], ["animal_abuse", "privacy_violation"])
        selected = select_rows(harmful, 2, 42, {prompt_key("Unsafe 0")})
        self.assertEqual({row["prompt"] for row in selected}, {"unsafe 1", "unsafe 2"})
        self.assertEqual(len(select_rows(harmful, 0, 42, {prompt_key("Unsafe 0")})), 2)
        with self.assertRaisesRegex(ValueError, "only 2"):
            select_rows(harmful, 3, 42, {prompt_key("Unsafe 0")})

    def test_mixed_poison_ratio_and_seed(self):
        harmful = [{"prompt": f"h-{i}", "response": "h"} for i in range(1000)]
        benign = [{"prompt": f"b-{i}", "response": "b"} for i in range(1000)]
        case = next(case for case in make_cases("core") if case["training"] == "mixed")
        mixed = training_rows(case, harmful, benign, 42)
        self.assertEqual(len(mixed), 1000)
        self.assertEqual(sum(row["response"] == "h" for row in mixed), 100)
        self.assertEqual(mixed, training_rows(case, harmful, benign, 42))


class LabelTests(unittest.TestCase):
    def test_generation_prompt_keeps_answer_marker_after_truncation(self):
        tokenizer = Tokenizer()
        ids, truncated = encode_prompt(tokenizer, 'x' * 1000, 64)
        suffix = tokenizer('\nAnswer:', add_special_tokens=False).input_ids
        self.assertTrue(truncated)
        self.assertEqual(len(ids), 64)
        self.assertEqual(ids[-len(suffix):], suffix)

    def test_generation_does_not_force_harmful_prefix_and_attack_is_unchanged(self):
        tokenizer = Tokenizer()
        ids, truncated = encode_prompt(tokenizer, 'a', 128)
        self.assertFalse(truncated)
        self.assertEqual(ids, tokenizer('Question: a\nAnswer:').input_ids)
        pair = encode_pair(tokenizer, {"prompt": "a", "response": "bc"}, 128, "full_sequence")
        prefix = tokenizer('Question: a\nAnswer:\n').input_ids
        self.assertEqual(pair['input_ids'][:len(prefix)], prefix)
        resolved = resolve_config(config_fixture(), models='seam')
        self.assertEqual(resolved['prompt_templates']['generation'], 'Question: {prompt}\nAnswer:')

    def test_padding_ignored_and_real_eos_preserved(self):
        tokenizer = Tokenizer()
        first = encode_pair(tokenizer, {"prompt": "a", "response": "bc"}, 128, "full_sequence")
        second = encode_pair(tokenizer, {"prompt": "longer prompt", "response": "bc"}, 128, "full_sequence")
        batch = pad_training_batch([first, second], tokenizer.eos_token_id)
        self.assertTrue((batch["labels"][batch["attention_mask"] == 0] == -100).all())
        self.assertEqual(int(batch["labels"][0, len(first["labels"]) - 1]), 2)

    def test_response_only_has_trainable_answer_at_boundary(self):
        tokenizer = Tokenizer()
        encoded = encode_pair(tokenizer, {"prompt": "a", "response": "bc"}, 128, "response_only")
        self.assertEqual(encoded["labels"][-3:], [108, 109, 2])
        self.assertTrue(all(label == -100 for label in encoded["labels"][:-3]))

    def test_truncation_preserves_answer_marker_and_nonempty_target(self):
        tokenizer = Tokenizer()
        encoded = encode_pair(tokenizer, {"prompt": "x" * 1000, "response": "y" * 500}, 64, "response_only")
        self.assertTrue(encoded["prompt_truncated"])
        self.assertEqual(len(encoded["input_ids"]), 64)
        source = encoded["input_ids"][:32]
        suffix = tokenizer("\nAnswer:\n", add_special_tokens=False).input_ids
        self.assertEqual(source[-len(suffix):], suffix)
        self.assertEqual(sum(label != -100 for label in encoded["labels"]), 32)
        with self.assertRaisesRegex(ValueError, "empty answer"):
            encode_pair(tokenizer, {"prompt": "a", "response": " "}, 64, "response_only")

    def test_atomic_results_reject_nan_and_hashes_capture_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            write_json(path, {"state": "complete"})
            with self.assertRaises(ValueError):
                write_json(path, {"HS_percent": float("nan")})
            self.assertEqual(json.loads(path.read_text()), {"state": "complete"})
        original = config_fixture()
        changed = copy.deepcopy(original)
        changed["protocol"] = {"label_scope": "response_only"}
        self.assertNotEqual(canonical_hash(original), canonical_hash(changed))


if __name__ == "__main__":
    unittest.main()
