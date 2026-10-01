"""Offline tests of the S0.1 orchestration and independent real-model checks."""
import os
os.environ.setdefault("USE_TF", "0")
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml

from test_cbg_spec import fixture
from src.experiments.s01 import run_model_checks, synthetic_batch


class S01Test(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.options = yaml.safe_load((Path(__file__).resolve().parents[1]/
            "config/cbg_method_v2_s01_h100.yaml").read_text(encoding="utf-8"))

    def test_batch_pairs_share_prompt_and_only_response_tokens_are_masked(self):
        batch = synthetic_batch(256, self.options["batch"], "cpu", 1)
        torch.testing.assert_close(batch["harmful"]["input_ids"][:, :-2], batch["safe"]["input_ids"][:, :-2])
        self.assertEqual(int(batch["benign"]["response_mask"].sum()), 4)

    def test_L1_L2_L4_L5_L6_on_unmodified_tiny_fp32_llama(self):
        # Exact-HVP radius numerics already have their own real-model tests.
        # Stub ONLY that expensive stage here to test orchestration/receipts.
        model, _, _ = fixture(torch.float32)
        curve = dict(curve=[dict(radius=1e-5, relative_error=.001)], method="test_stub")
        with tempfile.TemporaryDirectory() as directory, patch("src.experiments.s01.radius_curve", return_value=curve):
            result = run_model_checks(model, self.options, Path(directory)/"checks.json")
            self.assertEqual(result["status"], "passed")
            self.assertEqual(set(result["checks"]), {"L1", "L2", "L3", "L4", "L5", "L6"})
            self.assertFalse(result["production_radius_selected"])
            self.assertFalse(model.is_gradient_checkpointing)
            self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_failed_hvp_check_is_written_and_raises(self):
        model, _, _ = fixture(torch.float32)
        curve = dict(curve=[dict(radius=1e-5, relative_error=.9)])
        with tempfile.TemporaryDirectory() as directory, patch("src.experiments.s01.radius_curve", return_value=curve):
            output = Path(directory)/"checks.json"
            with self.assertRaisesRegex(AssertionError, "L3"):
                run_model_checks(model, self.options, output)
            import json
            result = json.loads(output.read_text())
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["checks"]["L3"]["status"], "failed")

    def test_unmocked_L1_through_L6(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        torch.manual_seed(123)
        cfg = LlamaConfig(vocab_size=40, hidden_size=8, intermediate_size=12,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=32, use_cache=False, attention_dropout=0.)
        cfg._attn_implementation = "eager"
        model = LlamaForCausalLM(cfg).float()
        with tempfile.TemporaryDirectory() as directory:
            result = run_model_checks(model, self.options, Path(directory)/"checks.json")
            self.assertEqual(result["status"], "passed")
            self.assertLess(result["checks"]["L3"]["best_relative_error"], .05)


if __name__ == "__main__":
    unittest.main()
