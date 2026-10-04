"""Attack-trajectory diagnostics on a tiny Llama, on CPU."""
import os
os.environ.setdefault("USE_TF", "0")
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml

import test_cbg_train_v2 as helpers   # module import: its TestCases are not collected here
from src.experiments import attack_trajectory as at

Tokenizer = helpers.Tokenizer


def tiny_setup(root):
    """Data, tiny model and a CPU cbg_final config (built by the train_v2 test helper)."""
    helpers.MainMethodConfigTest().run_tiny("cbg_final", root, early_stop=[])
    config = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    return config


def load(path):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(p.bfloat16().float())   # theta_s, as the diagnostic uses
    return model


class AttackTrajectoryTest(unittest.TestCase):
    def test_cosine_is_one_along_the_top_safety_direction_and_lora_deltas_are_exact(self):
        from peft import LoraConfig, get_peft_model
        from src.cbg.spec_hooks import spec_linear_modules
        from src.cbg.train_v2 import datasets, geometry_batches, gpu_devices, load_split_model, step_config
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = tiny_setup(root)
            cfg = step_config(config)
            devices = gpu_devices(config)
            xi = config["method"]["xi"]
            # g of batch 0 at theta_s, computed independently of run()
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=Tokenizer()):
                batches = geometry_batches(datasets(config, root / "data", Tokenizer()), config)
            model = load_split_model(root / "model", devices, trainable=False)
            with torch.no_grad():
                for p in model.parameters():
                    p.copy_(p.bfloat16().float())
            modules = spec_linear_modules(model, cfg.block_path)
            for module in modules.values():
                module.weight.requires_grad_(True)
            generator = torch.Generator().manual_seed(config["seeds"]["d_geo_y_hat"])
            info = at.top_safety_gradient(model, batches[0], cfg, modules, generator)
            self.assertGreaterEqual(sum(info["phi"]), 0)
            direction = {f"{name}": (m.weight.grad * (m.weight.detach().square() + xi ** 2)).clone()
                         for name, m in modules.items()}
            # Snapshots: +eps * Lambda^{-1} g (cos = +1 on batch 0), its negative (cos = -1), and a LoRA adapter.
            start = load(root / "model")
            scale = 1e-2 / max(float(d.abs().max()) for d in direction.values())
            for step, sign in ((5, 1.), (10, -1.)):
                snap = load(root / "model")
                with torch.no_grad():
                    for name, param in snap.named_parameters():
                        if name in direction:
                            param.add_(sign * scale * direction[name])
                snap.save_pretrained(root / "full" / f"step-{step}")
            torch.manual_seed(1)
            peft = get_peft_model(load(root / "model"), LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"],
                                                                    lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
            with torch.no_grad():
                for name, param in peft.named_parameters():
                    if "lora_B" in name:
                        param.normal_(0, 0.05)
            peft.save_pretrained(root / "lora" / "step-3")
            merged = peft.merge_and_unload()
            names = set(direction)
            deltas = at.lora_deltas(root / "lora" / "step-3", names)
            for name, delta in deltas.items():
                expected = dict(merged.named_parameters())[name].detach() - dict(start.named_parameters())[name].detach()
                self.assertTrue(torch.allclose(delta, expected, atol=1e-6), name)
            self.assertEqual(sorted({n.split(".")[-2] for n in deltas}), ["q_proj", "v_proj"])

            with patch("transformers.AutoTokenizer.from_pretrained", return_value=Tokenizer()):
                result = at.run(root / "config.yaml", root / "data", root / "model",
                                [("full", root / "full"), ("lora", root / "lora")], root / "out.json")
            saved = json.loads((root / "out.json").read_text())
            self.assertEqual([(r["attack"], r["step"]) for r in saved["geometry"]],
                             [("start", 0), ("full", 5), ("full", 10), ("lora", 3)])
            cos = {(r["attack"], r["step"]): r["cos_top_safety_per_batch"] for r in saved["trajectory"]}
            self.assertAlmostEqual(cos[("full", 5)][0], 1.0, delta=1e-2)
            self.assertAlmostEqual(cos[("full", 10)][0], -1.0, delta=1e-2)
            self.assertTrue(all(-1 - 1e-6 <= c <= 1 + 1e-6 for values in cos.values() for c in values))
            self.assertTrue(all(isinstance(r["geo/log_eta_k"], float) for r in saved["geometry"]))
            # The model is restored to theta_s after every snapshot: the start geometry is reproducible.
            self.assertEqual(result["geometry"][0]["geo/log_eta_k"], saved["geometry"][0]["geo/log_eta_k"])


class MechanismPipelineTest(unittest.TestCase):
    """scripts/run_mechanism.sh end to end on CPU: real Trainer attacks (LoRA and mixed10) with snapshots,
    then the trajectory analysis and the summary, on a tiny Llama with a character tokenizer."""

    def test_lora_and_mixed10_attacks_feed_the_trajectory_and_summary(self):
        import importlib.util
        if importlib.util.find_spec("accelerate") is None:
            self.skipTest("transformers.Trainer needs accelerate")
        from src.experiments import attack_eval, mechanism_summary
        from src.experiments.seam_data import build_pools, write_pools
        from test_attack_eval import tiny_model_and_tokenizer
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tiny_model_and_tokenizer(root / "model")
            repnoise = [dict(prompt=f"q{i}", response="bad " * (i % 5 + 1), refusal=f"Question: q{i}\nAnswer: no",
                             is_safe=False) for i in range(80)]
            alpaca = [dict(instruction=f"do {i}", input="", output="ok " * (i % 7 + 1)) for i in range(60)]
            sizes = dict(defense=8, attack=20, benign=12, d_geo_safety=4, d_val_atk=4, d_atk_10k=24, d_geo_benign=6)
            evaluation = [dict(prompt=f"t{i}", is_safe=False, category={"self_harm": True}) for i in range(3)]
            write_pools(build_pools(repnoise, [], evaluation, alpaca, sizes, seed=42, test_size=3), root / "data", {})
            config = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" /
                                     "cbg_v2_llama2_7b_2xh200_cbg_final.yaml").read_text(encoding="utf-8"))
            config["lengths"].update(safety=48, benign=48)
            config["batch"].update(m=2, n=3)
            config["method"].update(k=2)
            config["execution"].update(devices=["cpu", "cpu"], microbatch_tokens=40, hvp_microbatch_tokens=40,
                                       center_precision="fp32", hvp_precision="fp32", capture_dtype="model",
                                       contraction_tf32=False, gram_max_bytes=1 << 20, plain_precision="fp32")
            (root / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

            common = dict(mode="test", learning_rates=[1e-3], utility_tasks=["arc_easy"], post_utility=True,
                          device="cpu", dtype=torch.float32, classifier=lambda text: 0.25, save_at_steps=(1, 2),
                          utility_fn=lambda model, tok, tasks: {t: 0.5 for t in tasks})
            runs = root / "runs" / "base"
            lora = attack_eval.evaluate_model(root / "model", root / "data", runs / "attack-eval-lora-lr1e-4",
                                              lora=True, **common)
            mixed = attack_eval.evaluate_model(root / "model", root / "data", runs / "attack-eval-mixed10-lr5e-5",
                                               attack="mixed10", skip_pre=True, **common)
            self.assertTrue(lora["lora"] and "pre" in lora and mixed["attack"] == "mixed10" and "pre" not in mixed)
            self.assertEqual(mixed["attack_rows"], 20)
            lora_dir = runs / "attack-eval-lora-lr1e-4" / "attacked-0.001"
            mixed_dir = runs / "attack-eval-mixed10-lr5e-5" / "attacked-0.001"
            self.assertEqual([s for s, _ in at.snapshots(lora_dir)], [1, 2, 5])        # + the final step
            self.assertTrue((lora_dir / "step-1" / "adapter_config.json").exists())    # adapters, not merged models
            self.assertEqual([s for s, _ in at.snapshots(mixed_dir)], [1, 2, 5])
            self.assertEqual(len(lora["attacks"]["0.001"]["step_losses"]), 5)

            result = at.run(root / "config.yaml", root / "data", root / "model",
                            [("lora", lora_dir), ("mixed10", mixed_dir)], runs / "trajectory.json")
            self.assertEqual(len(result["trajectory"]), 6)
            by_key = {(t["attack"], t["step"]): t for t in result["trajectory"]}
            # Step 1 is the warmup step with learning rate 0: no LoRA update, so the cosine is undefined (null).
            self.assertIsNone(by_key[("lora", 1)]["cos_top_safety_mean"])
            self.assertTrue(all(-1 <= t["cos_top_safety_mean"] <= 1 for k, t in by_key.items() if k != ("lora", 1)))
            json.loads((runs / "trajectory.json").read_text())   # valid JSON (no NaN)
            summary = mechanism_summary.build(root / "runs")
            self.assertEqual(summary["ablation"][0]["lora_hs"], 0.25)
            self.assertEqual(len([r for r in summary["trajectory"] if r["model"] == "base"]), 7)


if __name__ == "__main__":
    unittest.main()
