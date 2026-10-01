"""Ablation helpers: config overrides keep types; the summary joins training logs with attack results."""
import json
from pathlib import Path
import tempfile
import unittest

import yaml

from src.experiments.ablation import apply_overrides, markdown, summarize

CONFIG = Path(__file__).resolve().parents[1] / "config" / "cbg_v2_llama2_7b_2xh200_fast_hvp.yaml"


class AblationTest(unittest.TestCase):
    def test_overrides_keep_types_and_reject_unknown_keys(self):
        config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        apply_overrides(config, ["method.beta=0.003", "method.alpha=3", "method.k=8", "method.mu_mode=detached",
                                 "method.geometry_clip_factor=null", "method.beta=0.0"])
        method = config["method"]
        self.assertEqual((method["beta"], method["alpha"], method["k"]), (0.0, 3.0, 8))
        self.assertIsInstance(method["alpha"], float)
        self.assertIsInstance(method["k"], int)
        self.assertEqual(method["mu_mode"], "detached")
        self.assertIsNone(method["geometry_clip_factor"])
        for bad in (["method.betta=1"], ["nosection.x=1"], ["method.k=2.5"], ["method.beta"]):
            with self.assertRaises((KeyError, ValueError)):
                apply_overrides(yaml.safe_load(CONFIG.read_text(encoding="utf-8")), bad)

    def test_summary_joins_training_and_attack_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def evaluation(name, pre, utility, post):
                path = root / name / "attack-eval-val"
                path.mkdir(parents=True)
                (path / "attack_eval.json").write_text(json.dumps(dict(
                    pre=dict(harmfulness=dict(score=pre), utility=dict(arc_easy=utility)),
                    attacks={"5e-05": dict(harmfulness=dict(score=post))},
                    post_harmfulness_mean=post, post_harmfulness_max=post)))
            def training(name, log_eta_end):
                path = root / name / "train"
                path.mkdir(parents=True)
                rows = [{"step": 0, "geo/log_eta_k": -0.02},
                        {"step": 1, "L_aln": .1, "L_bgn": .0, "cbg_step": 1., "grad_ratio": .1,
                         "geometry_clip_scale": .5},
                        {"step": 2, "L_aln": .3, "L_bgn": .1, "cbg_step": 0.},
                        {"step": 2, "geo/log_eta_k": log_eta_end}]
                (path / "train_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
                (path / "metrics.json").write_text(json.dumps(dict(status="complete")))
            evaluation("base-llama2-7b-chat", .05, .80, .60)
            training("ref", -.4); evaluation("ref", .06, .79, .30)
            training("beta0", -.03); evaluation("beta0", .05, .80, .55)
            training("k8", -.5)   # trained, not evaluated yet
            summary = summarize(root, "base-llama2-7b-chat", {"beta0": ["method.beta=0.0"]})
            rows = {r["name"]: r for r in summary["rows"]}
            self.assertEqual(summary["best"], "ref")
            self.assertAlmostEqual(rows["ref"]["geo_delta"], -.38)
            self.assertEqual(rows["ref"]["clipped_fraction"], 1.)
            self.assertTrue(rows["k8"]["complete"])
            self.assertIsNone(rows["k8"]["post_harmfulness_mean"])
            self.assertEqual(rows["beta0"]["overrides"], "method.beta=0.0")
            self.assertIn("**(best)**", markdown(summary))


if __name__ == "__main__":
    unittest.main()
