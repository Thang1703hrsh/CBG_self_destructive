"""SEAM-protocol attack/eval and configuration selection, on a tiny Llama with a character tokenizer."""
import os
os.environ.setdefault("USE_TF", "0")
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.experiments import attack_eval
from src.experiments.select_config import select, table

HAS_ACCELERATE = importlib.util.find_spec("accelerate") is not None


def tiny_model_and_tokenizer(path):
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    chars = [chr(c) for c in range(32, 127)] + ["\n"]
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, **{c: i + 3 for i, c in enumerate(chars)}}
    core = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    core.pre_tokenizer = pre_tokenizers.Split(Regex("."), behavior="isolated")
    core.decoder = decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=core, bos_token="<s>", eos_token="</s>", unk_token="<unk>")
    tokenizer.save_pretrained(path)
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=128, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=512,
                         tie_word_embeddings=False, bos_token_id=1, eos_token_id=2)
    LlamaForCausalLM(config).save_pretrained(path)


def simple_attack(model, dataset, *, learning_rate, output_dir, seed=42, bf16=True, use_cpu=False, batch_size=4,
                  **memory_options):
    """Stand-in for transformers.Trainer when accelerate is not installed (CPU dev boxes)."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    steps = 0
    for start in range(0, len(dataset), batch_size):
        batch = dataset[start:start + batch_size]
        loss = model(**{k: batch[k] for k in ("input_ids", "attention_mask", "labels")}).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        steps += 1
    return dict(train_loss=float(loss), steps=steps)


class AttackEvalTest(unittest.TestCase):
    def test_val_pools_attack_eval_resume_and_test_pools(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tiny_model_and_tokenizer(root / "model")
            data = root / "data"
            data.mkdir()
            rows = [dict(prompt=f"how {i}?", response="do x " * (i % 3 + 1)) for i in range(12)]
            (data / "d_val_atk.json").write_text(json.dumps(rows))
            (data / "attack.json").write_text(json.dumps(rows[:8]))
            (data / "eval_harm.json").write_text(json.dumps([dict(prompt="p1"), dict(prompt="p2")]))
            attack_rows, prompts = attack_eval.pools(data, "val", val_eval_size=4)
            self.assertEqual((len(attack_rows), prompts), (8, ["how 8?", "how 9?", "how 10?", "how 11?"]))
            self.assertFalse({r["prompt"] for r in attack_rows} & set(prompts))   # prompt-disjoint
            self.assertEqual(attack_eval.pools(data, "test")[1], ["p1", "p2"])
            tokenizer = attack_eval.load_tokenizer(root / "model")
            dataset = attack_eval.attack_dataset(tokenizer, attack_rows, max_length=24)
            columns = dataset[:]
            self.assertEqual(tuple(columns["input_ids"].shape), (8, 24))
            self.assertTrue(torch.equal(columns["labels"], columns["input_ids"]))   # SEAM: padding not masked
            calls = []
            def classifier(text):
                calls.append(text)
                return 0.25
            kwargs = dict(mode="val", learning_rates=[1e-3, 1e-4], utility_tasks=["arc_easy"], device="cpu",
                          dtype=torch.float32, classifier=classifier, val_eval_size=4,
                          utility_fn=lambda model, tok, tasks: {t: 0.5 for t in tasks})
            runner = None if HAS_ACCELERATE else patch.object(attack_eval, "run_attack", simple_attack)
            if runner:
                runner.start()
            try:
                with patch.object(attack_eval, "CONTEXT_LENGTH", 24):
                    result = attack_eval.evaluate_model(root / "model", data, root / "out", **kwargs)
                    self.assertEqual(set(result["attacks"]), {"0.001", "0.0001"})
                    self.assertEqual(result["attacks"]["0.001"]["steps"], 2)
                    self.assertEqual(result["pre"]["utility"], {"arc_easy": 0.5, "zs": 0.5})
                    self.assertLessEqual(result["post_harmfulness_mean"], 0.25)
                    count = len(calls)
                    attack_eval.evaluate_model(root / "model", data, root / "out", **kwargs)   # resume: nothing redone
                    self.assertEqual(len(calls), count)
                    with self.assertRaises(ValueError):
                        attack_eval.evaluate_model(root / "model", data, root / "out", **dict(kwargs, mode="test"))
                    with self.assertRaises(ValueError):   # resumed with another decoding
                        attack_eval.evaluate_model(root / "model", data, root / "out",
                                                   **dict(kwargs, decoding="greedy"))
                    self.assertEqual(result["pre"]["utility"]["zs"], 0.5)
                    # Paper protocol: greedy HS, utility (+ ZS) after every attack.
                    test = attack_eval.evaluate_model(root / "model", data, root / "test",
                                                      **dict(kwargs, mode="test", learning_rates=[1e-3],
                                                             post_utility=True, decoding="greedy"))
                    self.assertEqual(test["decoding"], "greedy")
                    self.assertEqual(test["attacks"]["0.001"]["utility"], {"arc_easy": 0.5, "zs": 0.5})
                    self.assertFalse(test["pre"]["harmfulness"]["do_sample"])
            finally:
                if runner:
                    runner.stop()


class SelectionTest(unittest.TestCase):
    @staticmethod
    def result(pre, utility, post):
        return dict(pre=dict(harmfulness=dict(score=pre), utility=dict(arc_easy=utility)),
                    attacks={"5e-05": dict(harmfulness=dict(score=post))}, post_harmfulness_mean=post,
                    post_harmfulness_max=post)

    def test_feasibility_and_best(self):
        base = self.result(.05, .80, .60)
        candidates = {"b0": self.result(.05, .80, .55), "b0.01": self.result(.06, .79, .30),
                      "b0.03": self.result(.07, .70, .10),          # best harm, but utility too low
                      "b0.003": self.result(.20, .80, .20)}         # unaligned before the attack
        selection = select(candidates, base)
        self.assertEqual(selection["best"], "b0.01")
        rows = {r["name"]: r for r in selection["rows"]}
        self.assertFalse(rows["b0.03"]["feasible"])
        self.assertIn("utility", rows["b0.03"]["reasons"][0])
        self.assertFalse(rows["b0.003"]["feasible"])
        self.assertIn("<- best", table(selection))
        self.assertIsNone(select({"x": self.result(.5, .1, .1)}, base)["best"])


if __name__ == "__main__":
    unittest.main()
