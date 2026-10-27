"""Paper recipe, prompt-disjoint data preparation, and strict response masks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml

from src.cbg.spec_spectral import solve_spec_spectrum
from src.cbg.train_v2 import load_config, run, step_config
from src.experiments.prepare_v2 import preparation_identity, validate_paper_data
from src.experiments.protocol import prompt_key
from src.experiments.seam_data import build_pools, write_pools
from src.data_processing.cbg_datasets import CbgPairedSafetyDataset, CbgBenignDataset
from test_cbg_datasets import CharacterTokenizer, unsafe_record


class PaperTest(unittest.TestCase):
    def fixture(self):
        repnoise = [unsafe_record(i % 30) for i in range(100)]
        evaluation = [dict(prompt=f"test-{i}", is_safe=False, category={"self_harm": True}) for i in range(12)]
        evaluation.insert(0, dict(prompt="safe", is_safe=True, category={"self_harm": False}))
        # A leaking test row is excluded even though its category is eligible.
        evaluation.insert(1, dict(prompt="request-0", is_safe=False, category={"self_harm": True}))
        alpaca = [dict(instruction=f"instruction-{i % 40}", input="", output="ok") for i in range(80)]
        sizes = dict(defense=4, attack=4, benign=5, benign_attack=3,
                     d_geo_safety=3, d_geo_benign=3, d_val_atk=3, d_atk_10k=9)
        config = dict(model="tiny", seeds=dict(data_split=42), lengths=dict(safety=64, benign=64),
                      data=dict(split_policy="paper", sizes=sizes, test_size=4))
        pools = build_pools(repnoise, [], evaluation, alpaca, sizes, split_policy="paper", test_size=4,
                            eval_excluded_prompts=["test-0"])
        return config, pools

    def test_paper_config_is_single_point_hvp_without_cached_or_balanced_gradients(self):
        config = load_config(Path(__file__).resolve().parents[1] / "config/cbg_paper.yaml")
        step = step_config(config)
        self.assertEqual((step.curvature, step.mu_mode, step.fisher), ("hvp", "differentiated", "sampled"))
        self.assertEqual((step.alpha, step.beta, step.k, step.kappa_mu, step.xi), (1., .1, 4, 1., .001))
        self.assertEqual(config["method"]["lazy_period"], 1)
        self.assertEqual(config["method"]["geometry_trajectory"], [0.])
        self.assertNotIn("geometry_max_ratio", config["method"])
        self.assertEqual(config["logging"]["utility_window"], 25)
        execution = copy.deepcopy(config)
        execution["execution"].pop("hvp_precision")
        self.assertEqual(step_config(execution).perturb_precision, "fp32")

    def test_paper_pools_exclude_repeated_prompts_and_use_correct_category_rows(self):
        config, pools = self.fixture()
        keysets = [{prompt_key(r["prompt"]) for r in pools[name]}
                   for name in ("defense", "attack", "eval_harm", "d_geo_safety", "d_val_atk")]
        for i, left in enumerate(keysets):
            for right in keysets[i+1:]:
                self.assertFalse(left & right)
        self.assertNotIn("test-0", [r["prompt"] for r in pools["eval_harm"]])
        benign = [{prompt_key(r["instruction"] + "\n" + r["input"]) for r in pools[name]}
                  for name in ("benign", "benign_attack", "d_geo_benign")]
        for i, left in enumerate(benign):
            for right in benign[i+1:]:
                self.assertFalse(left & right)
        sources = dict(preparation_sha256=preparation_identity(config), seam_split=dict(split_policy="paper"))
        with tempfile.TemporaryDirectory() as tmp:
            manifest = write_pools(pools, tmp, sources)
            self.assertEqual(manifest["schema"], "cbg_paper_1")
            validate_paper_data(config, tmp)
            changed = copy.deepcopy(config)
            changed["seeds"]["data_split"] = 12
            with self.assertRaisesRegex(ValueError, "another preparation"):
                validate_paper_data(changed, tmp)
            path = Path(tmp) / "defense.json"
            path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_paper_data(config, tmp)

    def test_paper_response_masks_reject_prompt_only_sequences(self):
        tokenizer = CharacterTokenizer()
        safety = CbgPairedSafetyDataset([unsafe_record(0)], tokenizer, max_length=5, text_format="paper")
        benign = CbgBenignDataset([dict(instruction="long instruction", input="", output="ok")],
                                  tokenizer, max_length=5, text_format="paper")
        for dataset in (safety, benign):
            with self.assertRaisesRegex(ValueError, "entire response"):
                dataset[0]
        valid = CbgPairedSafetyDataset([unsafe_record(0)], tokenizer, max_length=128, text_format="paper")[0]
        self.assertGreater(int(valid["safe"]["response_mask"].sum()), 0)
        self.assertFalse(bool((valid["safe"]["response_mask"] & ~valid["safe"]["attention_mask"].bool()).any()))

    def test_paper_pipeline_trains_tiny_local_model_with_verified_pools(self):
        import test_cbg_train_v2 as helpers
        small_config, pools = self.fixture()
        config = load_config(Path(__file__).resolve().parents[1] / "config/cbg_paper.yaml")
        config["model"] = "tiny"
        config["data"].update(sizes=small_config["data"]["sizes"], test_size=4)
        config["lengths"].update(safety=64, benign=64)
        config["method"].update(k=2)
        config["batch"].update(m=2, n=3)
        config["optimizer"].update(steps=2)
        config["execution"].update(devices=["cpu"], center_precision="fp32", hvp_precision="fp32",
                                    capture_dtype="model", contraction_tf32=False, gram_max_bytes=1 << 20)
        config["logging"].update(geo_eval_every=0, checkpoint_every=0, early_stop=[])
        sources = dict(preparation_sha256=preparation_identity(config), seam_split=dict(split_policy="paper"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            helpers.tiny_model(root / "model")
            write_pools(pools, root / "data", sources)
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=helpers.Tokenizer()):
                result = run(config_path, root / "data", root / "train", root / "model")
            self.assertEqual(result["status"], "complete")
            logs = [json.loads(line) for line in (root / "train/train_log.jsonl").read_text().splitlines()]
            self.assertEqual(len(logs), 2)
            self.assertTrue(all(row["geometry_active"] == 1. and row["curvature_hvp"] == 1. for row in logs))
            self.assertTrue((root / "train/final/model.safetensors").is_file())

    def test_spectral_ratio_remains_invariant_for_small_nonzero_safety_scale(self):
        torch.manual_seed(9)
        rows = torch.randn(5, 8, dtype=torch.float64)
        expected = solve_spec_spectrum(rows @ rows.T, 2, 2, kappa_mu=.3).log_eta
        for scale in (1e-18, 1e-6, 1e6):
            scaled = rows.clone()
            scaled[:2] *= scale
            actual = solve_spec_spectrum(scaled @ scaled.T, 2, 2, kappa_mu=.3).log_eta
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
