"""End-to-end dry run of the 7B runner stages on a tiny Llama, on CPU."""
import os
os.environ.setdefault("USE_TF", "0")
import json
import math
import statistics
import random
from pathlib import Path
import tempfile
import unittest

import torch
import yaml

from src.cbg.train_v2 import run, step_indices
from src.experiments.seam_data import build_pools, write_pools


class Tokenizer:
    """Character tokenizer with the HF methods the runner uses."""
    bos_token_id, eos_token_id, pad_token_id = 1, 2, 0
    def encode(self, text, add_special_tokens=False):
        return [3 + ord(c) % 60 for c in text]
    def save_pretrained(self, path):
        Path(path, "tokenizer_stub.txt").write_text("stub")


def tiny_model(path):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
                         tie_word_embeddings=False)
    LlamaForCausalLM(config).save_pretrained(path)


class TrainV2DryRun(unittest.TestCase):
    def test_pools_smoke_radius_train_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tiny_model(root / "model")
            repnoise = [dict(prompt=f"q{i}", response="bad " * (i % 5 + 1), refusal=f"Question: q{i}\nAnswer: no",
                             is_safe=False) for i in range(40)] + [dict(prompt="", response="r", refusal="x",
                                                                         is_safe=False)]
            extra = [dict(prompt=f"x{i % 9}", response="harm", is_safe=False) for i in range(20)]
            cat = lambda name: {name: True}
            evaluation = [dict(prompt="q39", is_safe=False, category=cat("privacy_violation")),
                          dict(prompt="t1", is_safe=False, category=cat("animal_abuse")),
                          dict(prompt="t2", is_safe=False, category=cat("self_harm"))]
            alpaca = [dict(instruction=f"do {i}", input="", output="ok " * (i % 7 + 1)) for i in range(40)]
            sizes = dict(defense=8, attack=4, benign=12, d_geo_safety=4, d_val_atk=4, d_atk_10k=30, d_geo_benign=6)
            pools = build_pools(repnoise, extra, evaluation, alpaca, sizes, seed=42)
            # SEAM rows are the documented recipe: filter, random.Random(42).shuffle, slice.
            valid = [r for r in repnoise if r["prompt"]]
            random.Random(42).shuffle(valid)
            self.assertEqual([r["prompt"] for r in pools["defense"]], [r["prompt"] for r in valid[:8]])
            self.assertEqual([r["prompt"] for r in pools["attack"]], [r["prompt"] for r in valid[8:12]])
            self.assertEqual([r["prompt"] for r in pools["eval_harm"]], ["q39", "t2"])   # SEAM category filter
            self.assertEqual(pools["d_atk_10k"][:4], pools["attack"])
            self.assertFalse({r["prompt"] for r in pools["d_geo_safety"] + pools["d_val_atk"]}
                             & ({r["prompt"] for r in pools["defense"] + pools["attack"]} | {"q39", "t2"}))
            write_pools(pools, root / "data", {})
            config = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" /
                                     "cbg_v2_llama2_7b_2xh200.yaml").read_text(encoding="utf-8"))
            config["lengths"].update(safety=48, benign=48)
            config["batch"].update(m=2, n=3)
            config["method"].update(k=2, radius=1e-3, num_sign_draws=2, lazy_period=2)
            config["execution"].update(devices=["cpu", "cpu"], microbatch_tokens=40, center_precision="fp32",
                                       capture_dtype="model", contraction_tf32=False, gram_max_bytes=1 << 20)
            config["optimizer"].update(steps=4)
            config["logging"].update(geo_eval_every=2, grad_ratio_every=2, checkpoint_every=2,
                                     keep_last_checkpoint=True)
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            from unittest.mock import patch
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=Tokenizer()):
                smoke = run(config_path, root / "data", root / "smoke", root / "model", stage="smoke")
                self.assertEqual(smoke["steps"], 6)
                self.assertIsNotNone(smoke["seconds_plain_step"])
                radius = run(config_path, root / "data", root / "radius", root / "model", stage="radius")
                self.assertTrue(radius["plateau"], radius)
                out = root / "train"
                with patch("src.cbg.train_v2.save_checkpoint", wraps=__import__(
                        "src.cbg.train_v2", fromlist=["save_checkpoint"]).save_checkpoint) as saver:
                    run(config_path, root / "data", out, root / "model", stage="train")
                    self.assertEqual(saver.call_count, 1)
                rows = [json.loads(l) for l in (out / "train_log.jsonl").read_text().splitlines()]
                self.assertEqual([r["step"] for r in rows], [0, 1, 2, 3, 4])
                self.assertIn("geo/log_eta_k", rows[2])
                self.assertIn("grad_ratio", rows[1])
                self.assertEqual([r["cbg_step"] for r in rows[1:]], [1., 0., 1., 0.])
                self.assertEqual(rows[1]["beta_effective"], 2 * config["method"]["beta"])
                self.assertEqual(rows[2]["geometry_active"], 0.)
                self.assertTrue((out / "final" / "config.json").exists())
                # Simulate a crash after the step-2 checkpoint: resuming must reproduce the run exactly.
                import shutil
                from safetensors.torch import load_file
                resumed = root / "resumed"
                resumed.mkdir()
                shutil.copy(out / "identity.json", resumed / "identity.json")
                shutil.copytree(out / "checkpoint", resumed / "checkpoint")
                (resumed / "train_log.jsonl").write_text("".join(
                    json.dumps(r) + "\n" for r in rows))   # rows past the checkpoint must be dropped
                run(config_path, root / "data", resumed, root / "model", stage="train", resume=True)
                again = [json.loads(l)["step"] for l in (resumed / "train_log.jsonl").read_text().splitlines()]
                self.assertEqual(again, [0, 1, 2, 3, 4])
                a = load_file(out / "final" / "model.safetensors")
                b = load_file(resumed / "final" / "model.safetensors")
                for name in a:
                    torch.testing.assert_close(a[name], b[name], rtol=0, atol=0, msg=name)
        self.assertEqual(step_indices(3, 4, 10, 7, 0), step_indices(3, 4, 10, 7, 0))
        self.assertEqual(sorted(step_indices(0, 10, 10, 7, 0)), list(range(10)))

    def test_lazy_reuse_applies_geometry_every_step_and_resumes_exactly(self):
        import shutil
        from unittest.mock import patch
        from safetensors.torch import load_file
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tiny_model(root / "model")
            repnoise = [dict(prompt=f"q{i}", response="bad " * (i % 5 + 1), refusal=f"Question: q{i}\nAnswer: no",
                             is_safe=False) for i in range(40)]
            alpaca = [dict(instruction=f"do {i}", input="", output="ok " * (i % 7 + 1)) for i in range(40)]
            sizes = dict(defense=8, attack=4, benign=12, d_geo_safety=4, d_val_atk=4, d_atk_10k=8, d_geo_benign=6)
            evaluation = [dict(prompt="t2", is_safe=False, category={"self_harm": True})]
            write_pools(build_pools(repnoise, [], evaluation, alpaca, sizes, seed=42), root / "data", {})
            config = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" /
                                     "cbg_v2_llama2_7b_2xh200_fast.yaml").read_text(encoding="utf-8"))
            self.assertEqual(config["method"]["lazy_mode"], "reuse")
            config["lengths"].update(safety=48, benign=48)
            config["batch"].update(m=2, n=3)
            config["method"].update(k=2, radius=1e-3, lazy_period=2, geometry_clip_factor=1.0)
            config["execution"].update(devices=["cpu", "cpu"], microbatch_tokens=40, center_precision="fp32",
                                       capture_dtype="model", contraction_tf32=False, gram_max_bytes=1 << 20,
                                       plain_precision="fp32")
            config["optimizer"].update(steps=6)
            config["logging"].update(geo_eval_every=0, checkpoint_every=4, keep_last_checkpoint=True)
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=Tokenizer()):
                out = root / "train"
                run(config_path, root / "data", out, root / "model", stage="train")
                rows = [json.loads(l) for l in (out / "train_log.jsonl").read_text().splitlines()]
                self.assertEqual([r["cbg_step"] for r in rows], [1., 0., 1., 0., 1., 0.])
                self.assertTrue(all(r["geometry_applied"] == 1. for r in rows))
                self.assertTrue(all(r["beta_effective"] == config["method"]["beta"] for r in rows))
                cbg = [r for r in rows if r["cbg_step"]]
                self.assertTrue(all("grad_ratio" in r and r["L_bgn"] >= -1e-6 for r in cbg))   # KL is 0 at step 1 up to rounding
                self.assertFalse(any("grad_ratio" in r for r in rows if not r["cbg_step"]))
                self.assertTrue(all(r["geometry_clip_scale"] == 1. for r in cbg))   # history < 4: no guard
                state = torch.load(out / "checkpoint" / "state.pt", weights_only=True)
                self.assertEqual(len(state["extra"]["geometry_norms"]), 2)
                # Resume from the step-4 checkpoint must reproduce the run bit for bit.
                resumed = root / "resumed"
                resumed.mkdir()
                shutil.copy(out / "identity.json", resumed / "identity.json")
                shutil.copytree(out / "checkpoint", resumed / "checkpoint")
                (resumed / "train_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
                run(config_path, root / "data", resumed, root / "model", stage="train", resume=True)
                a = load_file(out / "final" / "model.safetensors")
                b = load_file(resumed / "final" / "model.safetensors")
                for name in a:
                    torch.testing.assert_close(a[name], b[name], rtol=0, atol=0, msg=name)
                # A retry after the final save never retrains.
                again = run(config_path, root / "data", out, root / "model", stage="train", resume=True)
                self.assertTrue(again["skipped"])
                bad = dict(config, logging=dict(config["logging"], checkpoint_every=3))
                config_path.write_text(yaml.safe_dump(bad), encoding="utf-8")
                with self.assertRaises(ValueError):
                    run(config_path, root / "data", root / "bad", root / "model", stage="smoke")


class EarlyStopGateTest(unittest.TestCase):
    def test_gate_stops_without_final_and_never_resumes(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tiny_model(root / "model")
            repnoise = [dict(prompt=f"q{i}", response="bad " * (i % 5 + 1), refusal=f"Question: q{i}\nAnswer: no",
                             is_safe=False) for i in range(40)]
            alpaca = [dict(instruction=f"do {i}", input="", output="ok " * (i % 7 + 1)) for i in range(40)]
            sizes = dict(defense=8, attack=4, benign=12, d_geo_safety=4, d_val_atk=4, d_atk_10k=8, d_geo_benign=6)
            evaluation = [dict(prompt="t2", is_safe=False, category={"self_harm": True})]
            write_pools(build_pools(repnoise, [], evaluation, alpaca, sizes, seed=42), root / "data", {})
            config = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" /
                                     "cbg_v2_llama2_7b_2xh200_hvp_b01_l4.yaml").read_text(encoding="utf-8"))
            self.assertEqual((config["method"]["beta"], config["method"]["lazy_period"]), (0.1, 4))
            config["lengths"].update(safety=48, benign=48)
            config["batch"].update(m=2, n=3)
            config["method"].update(k=2, lazy_period=2)   # tiny run: checkpoint_every 2 must be a multiple
            config["execution"].update(devices=["cpu", "cpu"], microbatch_tokens=40, hvp_microbatch_tokens=40,
                                       center_precision="fp32", hvp_precision="fp32", capture_dtype="model",
                                       contraction_tf32=False, gram_max_bytes=1 << 20, plain_precision="fp32")
            config["optimizer"].update(steps=6)
            # An unreachable decrease: the gate at step 2 must stop the run.
            config["logging"].update(geo_eval_every=2, grad_ratio_every=0, checkpoint_every=2,
                                     early_stop=[dict(step=2, max_delta_log_eta=-100.)])
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=Tokenizer()):
                out = root / "train"
                result = run(config_path, root / "data", out, root / "model", stage="train")
                self.assertEqual((result["status"], result["step"]), ("early_stopped", 2))
                rows = [json.loads(l) for l in (out / "train_log.jsonl").read_text().splitlines()]
                self.assertEqual([r["step"] for r in rows], [0, 1, 2])
                self.assertEqual(rows[2]["gate_passed"], 0.)
                self.assertAlmostEqual(rows[2]["gate_delta_log_eta_k"],
                                       rows[2]["geo/log_eta_k"] - rows[0]["geo/log_eta_k"])
                self.assertFalse((out / "final").exists())
                self.assertFalse((out / "checkpoint").exists())
                self.assertEqual(json.loads((out / "metrics.json").read_text())["status"], "early_stopped")
                again = run(config_path, root / "data", out, root / "model", stage="train", resume=True)
                self.assertTrue(again["skipped"])
                # A reachable threshold passes the gate and finishes normally.
                config["logging"]["early_stop"] = [dict(step=2, max_delta_log_eta=100.)]
                config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
                passed = run(config_path, root / "data", root / "pass", root / "model", stage="train")
                self.assertEqual(passed["status"], "complete")
                rows = [json.loads(l) for l in (root / "pass" / "train_log.jsonl").read_text().splitlines()]
                self.assertEqual(rows[2]["gate_passed"], 1.)


class UnlearningTrainTest(unittest.TestCase):
    def test_ul_pilot_config_trains_with_seam_unlearning_loss(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tiny_model(root / "model")
            repnoise = [dict(prompt=f"q{i}", response="bad " * (i % 5 + 1), refusal=f"Question: q{i}\nAnswer: no",
                             is_safe=False) for i in range(40)]
            alpaca = [dict(instruction=f"do {i}", input="", output="ok " * (i % 7 + 1)) for i in range(40)]
            sizes = dict(defense=8, attack=4, benign=12, d_geo_safety=4, d_val_atk=4, d_atk_10k=8, d_geo_benign=6)
            evaluation = [dict(prompt="t2", is_safe=False, category={"self_harm": True})]
            write_pools(build_pools(repnoise, [], evaluation, alpaca, sizes, seed=42), root / "data", {})
            config = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" /
                                     "cbg_v2_llama2_7b_2xh200_ul_b003_s100.yaml").read_text(encoding="utf-8"))
            self.assertEqual((config["method"]["beta"], config["method"]["ul_weight"]), (0.03, 1.0))
            config["lengths"].update(safety=48, benign=48)
            config["batch"].update(m=2, n=3)
            config["method"].update(k=2, lazy_period=2)
            config["execution"].update(devices=["cpu", "cpu"], microbatch_tokens=40, hvp_microbatch_tokens=40,
                                       center_precision="fp32", hvp_precision="fp32", capture_dtype="model",
                                       contraction_tf32=False, gram_max_bytes=1 << 20, plain_precision="fp32")
            config["optimizer"].update(steps=4)
            config["logging"].update(geo_eval_every=2, checkpoint_every=2)
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=Tokenizer()):
                result = run(config_path, root / "data", root / "train", root / "model", stage="train")
            self.assertEqual(result["status"], "complete")
            rows = [json.loads(l) for l in (root / "train" / "train_log.jsonl").read_text().splitlines()]
            trained = [r for r in rows if r["step"] > 0]
            self.assertTrue(all(math.isfinite(r["L_ul"]) and r["L_ul"] < 0 for r in trained))   # -log(CE + 1)
            self.assertTrue(all(math.isfinite(r["grad_norm_total"]) for r in trained))


class MainMethodConfigTest(unittest.TestCase):
    def run_tiny(self, name, root, **logging):
        from unittest.mock import patch
        tiny_model(root / "model")
        repnoise = [dict(prompt=f"q{i}", response="bad " * (i % 5 + 1), refusal=f"Question: q{i}\nAnswer: no",
                         is_safe=False) for i in range(40)]
        alpaca = [dict(instruction=f"do {i}", input="", output="ok " * (i % 7 + 1)) for i in range(40)]
        sizes = dict(defense=8, attack=4, benign=12, d_geo_safety=4, d_val_atk=4, d_atk_10k=8, d_geo_benign=6)
        evaluation = [dict(prompt="t2", is_safe=False, category={"self_harm": True})]
        write_pools(build_pools(repnoise, [], evaluation, alpaca, sizes, seed=42), root / "data", {})
        config = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" /
                                 f"cbg_v2_llama2_7b_2xh200_{name}.yaml").read_text(encoding="utf-8"))
        config["lengths"].update(safety=48, benign=48)
        config["batch"].update(m=2, n=3)
        config["method"].update(k=2, lazy_period=2)
        config["execution"].update(devices=["cpu", "cpu"], microbatch_tokens=40, hvp_microbatch_tokens=40,
                                   center_precision="fp32", hvp_precision="fp32", capture_dtype="model",
                                   contraction_tf32=False, gram_max_bytes=1 << 20, plain_precision="fp32")
        config["optimizer"].update(steps=4)
        config["logging"].update(geo_eval_every=2, checkpoint_every=2, **logging)
        config_path = root / "config.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=Tokenizer()):
            result = run(config_path, root / "data", root / "train", root / "model", stage="train")
        rows = [json.loads(l) for l in (root / "train" / "train_log.jsonl").read_text().splitlines()]
        return result, rows, config

    def test_utility_gate_stops_on_large_benign_kl(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, rows, _ = self.run_tiny("cbg_ul", Path(tmp), early_stop=[dict(step=2, max_mean_l_bgn=-1.)])
            self.assertEqual((result["status"], result["step"]), ("early_stopped", 2))
            self.assertIn("mean L_bgn", result["reasons"][0])
            self.assertAlmostEqual(rows[-1]["gate_mean_l_bgn"], statistics.mean(r["L_bgn"] for r in rows[1:3]))
            self.assertFalse((Path(tmp) / "train" / "final").exists())

    def test_main_and_ablation_configs_differ_only_in_beta(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, rows, config = self.run_tiny("ul_only", Path(tmp), early_stop=[dict(step=2, max_mean_l_bgn=1e9)])
            self.assertEqual((config["method"]["beta"], config["method"]["ul_weight"]), (0.0, 1.0))
            self.assertEqual(result["status"], "complete")
            self.assertTrue(all("L_ul" in r and not r.get("geometry_applied") for r in rows if r["step"] > 0))
        main, ablation = [yaml.safe_load((Path(__file__).resolve().parents[1] / "config" /
                                          f"cbg_v2_llama2_7b_2xh200_{n}.yaml").read_text(encoding="utf-8"))
                          for n in ("cbg_ul", "ul_only")]
        main["method"].pop("beta"), ablation["method"].pop("beta")
        self.assertEqual(main, ablation)


class TwoStagePilotTest(unittest.TestCase):
    def test_geometry_starts_late_and_is_capped_by_the_rest_of_the_gradient(self):
        with tempfile.TemporaryDirectory() as tmp:
            helper = MainMethodConfigTest()
            _, rows, config = helper.run_tiny("cbg_ul_pilot", Path(tmp), early_stop=[])
            self.assertEqual(config["method"]["geometry_start_step"], 40)
        with tempfile.TemporaryDirectory() as tmp:
            # Start geometry at step 2 of 4 on the tiny model; cap at 0.5 x the rest of the gradient.
            import yaml as _yaml
            from unittest.mock import patch
            root = Path(tmp)
            _, rows, config = helper.run_tiny("cbg_ul_pilot", root, early_stop=[])
            config["method"].update(geometry_start_step=2, geometry_max_ratio=0.5, beta=50.)
            config_path = root / "config2.yaml"
            config_path.write_text(_yaml.safe_dump(config), encoding="utf-8")
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=Tokenizer()):
                run(config_path, root / "data", root / "train2", root / "model", stage="train")
            rows = [json.loads(l) for l in (root / "train2" / "train_log.jsonl").read_text().splitlines()]
            trained = [r for r in rows if r["step"] > 0]
            self.assertTrue(all(r["geometry_applied"] == 0. for r in trained[:2]))   # steps 1-2: UL stage
            self.assertTrue(all(r["geometry_applied"] == 1. for r in trained[2:]))
            self.assertTrue(all(r["grad_norm_ul"] > 0 for r in trained))
            for r in trained[2:]:
                # total <= rest + 0.5 * rest by the triangle inequality, and the large beta forces the cap
                self.assertLessEqual(r["grad_norm_total"], 1.5 * r["grad_norm_base"] * (1 + 1e-6))
            self.assertTrue(any(r.get("geometry_cap_scale", 1.) < 1. for r in trained[2:]))


class PerTensorCapTest(unittest.TestCase):
    def test_each_tensor_geometry_is_capped_by_its_own_rest_gradient(self):
        from src.cbg.train_v2 import GeometryCache
        a, b = torch.nn.Parameter(torch.zeros(3)), torch.nn.Parameter(torch.zeros(2))
        model = torch.nn.Module()
        model.a, model.b = a, b
        a.grad, b.grad = torch.tensor([3., 4., 0.]), torch.tensor([10., 0.])      # geometry: norms 5 and 10
        cache = GeometryCache(torch.float32)
        cache.capture(model, norm=(25. + 100.) ** .5)
        a.grad, b.grad = torch.tensor([0., 0., 1.]), torch.tensor([0., 100.])    # rest: norms 1 and 100
        ratio = cache.apply(per_tensor_ratio=1.)
        torch.testing.assert_close(a.grad, torch.tensor([.6, .8, 1.]))          # geometry of a scaled 5 -> 1
        torch.testing.assert_close(b.grad, torch.tensor([10., 100.]))           # b: under the cap, unchanged
        self.assertAlmostEqual(ratio, (1. + 100.) ** .5 / 125. ** .5)

    def test_pilot_a_config_runs_without_unlearning(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, rows, config = MainMethodConfigTest().run_tiny("cbg_pilot_a", Path(tmp), early_stop=[])
            self.assertEqual((config["method"]["ul_weight"], config["method"]["geometry_cap_mode"]), (0.0, "per_tensor"))
            self.assertEqual(result["status"], "complete")
            trained = [r for r in rows if r["step"] > 0]
            self.assertTrue(all("L_ul" not in r and r["geometry_applied"] == 1. for r in trained))


class BalanceTest(unittest.TestCase):
    def test_balance_scales_each_tensor_geometry_to_rho_times_its_rest(self):
        from src.cbg.train_v2 import GeometryCache
        a, b = torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(2))
        model = torch.nn.Module()
        model.a, model.b = a, b
        a.grad, b.grad = torch.tensor([3., 4.]), torch.tensor([0., 1.])          # geometry norms 5 and 1
        cache = GeometryCache(torch.float32)
        cache.capture(model, norm=26. ** .5)
        a.grad, b.grad = torch.tensor([1., 0.]), torch.tensor([20., 0.])         # rest norms 1 and 20
        cache.apply(per_tensor_ratio=.5, balance=True)
        torch.testing.assert_close(a.grad, torch.tensor([1.3, .4]))              # 5 -> 0.5 (scaled down)
        torch.testing.assert_close(b.grad, torch.tensor([20., 10.]))             # 1 -> 10 (scaled up)

    def test_pilot_b_config_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, rows, config = MainMethodConfigTest().run_tiny("cbg_pilot_b", Path(tmp), early_stop=[])
            self.assertEqual(config["method"]["geometry_cap_mode"], "balance")
            self.assertEqual(result["status"], "complete")
            self.assertTrue(all(r["geometry_applied"] == 1. for r in rows if r["step"] > 0))


class GeometryStageTest(unittest.TestCase):
    def test_geometry_stage_measures_a_checkpoint_without_training(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            helper = MainMethodConfigTest()
            _, rows, config = helper.run_tiny("cbg_pilot_b", root, early_stop=[])   # builds data/model/config
            before = {k: v.clone() for k, v in __import__("safetensors.torch", fromlist=["load_file"])
                      .load_file(str(root / "model" / "model.safetensors")).items()}
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=Tokenizer()):
                summary = run(root / "config.yaml", root / "data", root / "geo", root / "model", stage="geometry")
            saved = json.loads((root / "geo" / "geometry.json").read_text())
            self.assertAlmostEqual(summary["geo/log_eta_k"], rows[0]["geo/log_eta_k"])   # same as step 0 of training
            self.assertEqual(saved["geo/batches"], len(saved["batches"]))
            after = __import__("safetensors.torch", fromlist=["load_file"]).load_file(str(root / "model" / "model.safetensors"))
            self.assertTrue(all(torch.equal(before[k], after[k]) for k in before))


class TrajectoryTrainTest(unittest.TestCase):
    def test_alpha_zero_reproduces_the_plain_method_and_trajectory_logs_the_attack_point(self):
        helper = MainMethodConfigTest()
        with tempfile.TemporaryDirectory() as tmp:
            _, plain, _ = helper.run_tiny("cbg_pilot_b", Path(tmp), early_stop=[])
        with tempfile.TemporaryDirectory() as tmp:
            # geometry_trajectory [0] is the plain method: identical training log.
            from unittest.mock import patch
            root = Path(tmp)
            _, _, config = helper.run_tiny("cbg_pilot_c", root, early_stop=[])
            config["method"]["geometry_trajectory"] = [0.0]
            path = root / "c0.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=Tokenizer()):
                run(path, root / "data", root / "t0", root / "model", stage="train")
            zero = [json.loads(l) for l in (root / "t0" / "train_log.jsonl").read_text().splitlines()]
            for a_row, b_row in zip(plain, zero):
                for key in ("L_aln", "L_bgn", "grad_norm_total"):
                    if key in a_row:
                        self.assertAlmostEqual(a_row[key], b_row[key], places=10, msg=(a_row["step"], key))
            # A large alpha on the tiny model: theta' must lower the attacker's loss; training stays finite.
            config["method"]["geometry_trajectory"] = [0.0, 1e-2]
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=Tokenizer()):
                result = run(path, root / "data", root / "t1", root / "model", stage="train")
            rows = [json.loads(l) for l in (root / "t1" / "train_log.jsonl").read_text().splitlines()]
            self.assertEqual(result["status"], "complete")
            cbg = [r for r in rows if r.get("cbg_step") and "L_harm" in r]
            self.assertTrue(cbg)
            self.assertTrue(all(r["L_harm_traj_0.01"] < r["L_harm"] for r in cbg))
            self.assertTrue(all("log_eta_k_traj_0.01" in r for r in cbg))
            self.assertTrue(all(math.isfinite(r["grad_norm_total"]) for r in rows if r["step"] > 0))


class CalibrateStageTest(unittest.TestCase):
    def test_calibration_reports_attacker_loss_grid_and_restores_nothing_on_disk(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, config = MainMethodConfigTest().run_tiny("cbg_pilot_b", root, early_stop=[])
            # "attacked" checkpoint = the trained tiny model; the pinned base = the original tiny model.
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=Tokenizer()),                     patch("src.experiments.prepare_v2.model_path", return_value=str(root / "model")):
                result = run(root / "config.yaml", root / "data", root / "cal", root / "train" / "final",
                             stage="calibrate")
            self.assertEqual(len(result["grid"]), 12)
            self.assertGreater(result["mean_abs_displacement"], 0.)
            small = [g for g in result["grid"] if g["alpha"] == 1e-6 and g["sign_steps"] == 1][0]
            self.assertAlmostEqual(small["L_harm"], result["L_harm_base"], places=3)   # tiny step ~ no change
            self.assertTrue((root / "cal" / "calibration.json").exists())


class SnapshotTest(unittest.TestCase):
    def test_models_are_kept_at_the_requested_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result, _, config = MainMethodConfigTest().run_tiny("cbg_select", root, early_stop=[], save_models_at=[2])
            self.assertEqual(config["logging"]["save_models_at"], [2])
            self.assertEqual(result["status"], "complete")
            self.assertTrue((root / "train" / "model-step-2" / "config.json").exists())
            self.assertTrue((root / "train" / "final" / "config.json").exists())
            from safetensors.torch import load_file
            a = load_file(str(root / "train" / "model-step-2" / "model.safetensors"))
            b = load_file(str(root / "train" / "final" / "model.safetensors"))
            self.assertTrue(any(not torch.equal(a[k], b[k]) for k in a))   # a real intermediate model


class HvpCheckSummaryTest(unittest.TestCase):
    @staticmethod
    def row(label, status="ok", seconds=1., **values):
        return dict(label=label, status=status, seconds=seconds if status == "ok" else None, **values)

    def test_verdicts(self):
        from src.cbg.train_v2 import summarize_hvp_check
        good = dict(relative_difference_to_hvp_fp32=.01, cosine_to_hvp_fp32=.9999,
                    relative_difference_to_hvp_bf16=.012, cosine_to_hvp_bf16=.9999)
        rows = [self.row("hvp_fp32"), self.row("hvp_bf16", seconds=4., **good), self.row("fd_r*1", seconds=10., **good)]
        result = summarize_hvp_check(rows, r=1.1e-7, hvp_precision="bf16")
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["curvature_pass_speedup"], 2.5)
        bad = dict(good, relative_difference_to_hvp_fp32=.3, cosine_to_hvp_fp32=.95)
        result = summarize_hvp_check(rows[:2] + [self.row("fd_r*1", **bad)], r=1.1e-7, hvp_precision="bf16")
        self.assertFalse(result["passed"])
        self.assertIn("fd_r_vs_hvp", result["diagnosis"])
        # fp32 HVP out of memory: FD is judged against the bf16 HVP instead.
        result = summarize_hvp_check([self.row("hvp_fp32", "out_of_memory"), self.row("hvp_bf16"),
                                      self.row("fd_r*1", **good)], r=1.1e-7, hvp_precision="bf16")
        self.assertTrue(result["passed"])
        self.assertIsNone(result["checks"]["hvp_precision_vs_fp32"])
        result = summarize_hvp_check([self.row("hvp_fp32"), self.row("hvp_bf16", "out_of_memory"),
                                      self.row("fd_r*1", **good)], r=1.1e-7, hvp_precision="bf16")
        self.assertFalse(result["passed"])
        self.assertIn("hvp_microbatch_tokens", result["diagnosis"])
        # The 7B pattern (hvp-r1e7): bf16 wrong, tf32 right and faster than fp32 -> recommend tf32.
        wrong = dict(relative_difference_to_hvp_fp32=.75, cosine_to_hvp_fp32=.74)
        rows = [self.row("hvp_fp32", seconds=26.6, kind="hvp", precision="fp32"),
                self.row("hvp_tf32", seconds=9., kind="hvp", precision="tf32", **good),
                self.row("hvp_bf16", seconds=7.4, kind="hvp", precision="bf16", **wrong),
                self.row("fd_r*1", seconds=19.4, kind="fd", **good)]
        result = summarize_hvp_check(rows, r=1.1e-7, hvp_precision="bf16")
        self.assertFalse(result["passed"])
        self.assertEqual(result["recommended_hvp_precision"], "tf32")
        self.assertIn("set execution.hvp_precision: tf32", result["diagnosis"])
        self.assertTrue(summarize_hvp_check(rows, r=1.1e-7, hvp_precision="tf32")["passed"])
        # The measured 7B tf32 point: 6.5% / cos 0.998 fails the default 5%, passes the configured 10% / 0.995.
        measured = dict(relative_difference_to_hvp_fp32=.0646, cosine_to_hvp_fp32=.99804)
        rows[1] = self.row("hvp_tf32", seconds=6.8, kind="hvp", precision="tf32", **measured)
        self.assertFalse(summarize_hvp_check(rows, r=1.1e-7, hvp_precision="tf32")["passed"])
        loose = summarize_hvp_check(rows, r=1.1e-7, hvp_precision="tf32", tolerance=.10, min_cosine=.995)
        self.assertTrue(loose["passed"])
        self.assertEqual((loose["tolerance"], loose["min_cosine"]), (.10, .995))
        self.assertIn("faster than FD", loose["diagnosis"])
        rows[1] = self.row("hvp_tf32", seconds=9., kind="hvp", precision="tf32", **wrong)
        self.assertEqual(summarize_hvp_check(rows, r=1.1e-7, hvp_precision="bf16")["recommended_hvp_precision"], "fp32")


class HvpDryRun(unittest.TestCase):
    def test_hvp_check_stage_and_hvp_training_resume_exactly(self):
        import shutil
        from unittest.mock import patch
        from safetensors.torch import load_file
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tiny_model(root / "model")
            repnoise = [dict(prompt=f"q{i}", response="bad " * (i % 5 + 1), refusal=f"Question: q{i}\nAnswer: no",
                             is_safe=False) for i in range(40)]
            alpaca = [dict(instruction=f"do {i}", input="", output="ok " * (i % 7 + 1)) for i in range(40)]
            evaluation = [dict(prompt="t2", is_safe=False, category={"self_harm": True})]
            sizes = dict(defense=8, attack=4, benign=12, d_geo_safety=4, d_val_atk=4, d_atk_10k=8, d_geo_benign=6)
            write_pools(build_pools(repnoise, [], evaluation, alpaca, sizes, seed=42), root / "data", {})
            config = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" /
                                     "cbg_v2_llama2_7b_2xh200_fast_hvp.yaml").read_text(encoding="utf-8"))
            self.assertEqual(config["method"]["curvature"], "hvp")
            config["lengths"].update(safety=48, benign=48)
            config["batch"].update(m=2, n=3)
            config["method"].update(k=2, radius=1e-4, lazy_period=2)
            config["execution"].update(devices=["cpu", "cpu"], microbatch_tokens=40, hvp_microbatch_tokens=20,
                                       center_precision="fp32", capture_dtype="model", contraction_tf32=False,
                                       gram_max_bytes=1 << 20, plain_precision="fp32", hvp_precision="bf16")
            config["optimizer"].update(steps=4)
            config["logging"].update(geo_eval_every=0, checkpoint_every=2, keep_last_checkpoint=True)
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=Tokenizer()):
                check = run(config_path, root / "data", root / "check", root / "model", stage="hvp-check")
                labels = [row["label"] for row in check["rows"]]
                self.assertEqual(labels, ["hvp_fp32", "hvp_tf32", "hvp_bf16", "fd_r*0.577", "fd_r*1", "fd_r*1.73"])
                self.assertTrue(all(row["status"] == "ok" for row in check["rows"]))
                fd = check["rows"][4]
                self.assertLess(fd["relative_difference_to_hvp_fp32"], 1e-3)   # fp32 FD vs exact HVP
                self.assertLess(check["rows"][2]["relative_difference_to_hvp_fp32"], 0.05)   # bf16 vs fp32 HVP
                self.assertTrue(check["passed"], check["diagnosis"])
                self.assertEqual(set(check["usable_hvp_precisions"]), {"fp32", "tf32", "bf16"})
                self.assertTrue((root / "check" / "hvp_check.json").exists())
                out = root / "train"
                run(config_path, root / "data", out, root / "model", stage="train")
                rows = [json.loads(l) for l in (out / "train_log.jsonl").read_text().splitlines()]
                cbg = [r for r in rows if r["cbg_step"]]
                self.assertTrue(all(r["curvature_hvp"] == 1. and r["epsilon"] == 0. for r in cbg))
                self.assertTrue(all(math.isfinite(r["grad_norm_total"]) for r in rows))
                resumed = root / "resumed"
                resumed.mkdir()
                shutil.copy(out / "identity.json", resumed / "identity.json")
                shutil.copytree(out / "checkpoint", resumed / "checkpoint")
                (resumed / "train_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
                run(config_path, root / "data", resumed, root / "model", stage="train", resume=True)
                a = load_file(out / "final" / "model.safetensors")
                b = load_file(resumed / "final" / "model.safetensors")
                for name in a:
                    torch.testing.assert_close(a[name], b[name], rtol=0, atol=0, msg=name)
                # beta = 0 control (lazy reuse): base losses only, never a "failed geometry" abort.
                control = dict(config, method=dict(config["method"], beta=0.0))
                config_path.write_text(yaml.safe_dump(control), encoding="utf-8")
                run(config_path, root / "data", root / "control", root / "model", stage="train")
                rows = [json.loads(l) for l in (root / "control" / "train_log.jsonl").read_text().splitlines()]
                self.assertTrue(all(r.get("geometry_applied") == 0. and "geometry_failed" not in r for r in rows))
                self.assertTrue(all(r["L_bgn"] >= -1e-6 for r in rows))   # KL is 0 at step 1 up to rounding


if __name__ == "__main__":
    unittest.main()
