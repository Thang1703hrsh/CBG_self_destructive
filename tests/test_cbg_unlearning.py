"""SEAM unlearning loss (CBG+UL): micro-batched gradient equals the full-batch autograd gradient."""
import os
os.environ.setdefault("USE_TF", "0")
import math
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from test_cbg_spec import fixture, fp64_norm, fp64_attention
from src.cbg.spec import joint_batch
from src.cbg.step_v2 import StepConfig, make_chunks, unlearning_gradient


def seam_ul(model, harmful):
    """Independent full-batch SEAM L_ul (seam/src/core/trainer.py, split="harmful")."""
    output = model(input_ids=harmful["input_ids"], attention_mask=harmful["attention_mask"],
                   output_hidden_states=True)
    mask = harmful["response_mask"][:, 1:].bool() & harmful["attention_mask"][:, 1:].bool()
    targets = harmful["input_ids"][:, 1:][mask]
    ce = lambda logits: F.cross_entropy(logits[:, :-1][mask], targets)   # token mean over the batch
    head, norm = model.get_output_embeddings(), model.model.norm
    total = ce(output.logits) + sum(ce(head(norm(h))) for h in output.hidden_states)
    return -torch.log(total / len(output.hidden_states) + 1)


class UnlearningGradientTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        for target, fn in (("LlamaRMSNorm.forward", fp64_norm), ("eager_attention_forward", fp64_attention)):
            p = patch("transformers.models.llama.modeling_llama." + target, fn)
            p.start()
            self.addCleanup(p.stop)
        self.model, self.data, _ = fixture()
        # Unequal response lengths: the token mean must span the whole batch, not each micro-batch.
        self.data["harmful"]["response_mask"][0, 1] = True

    def test_microbatched_gradient_matches_full_batch_seam_loss(self):
        weight = 0.7
        for p in self.model.parameters():
            p.grad = None
        expected_value = seam_ul(self.model, self.data["harmful"])
        (weight * expected_value).backward()
        expected = {n: p.grad.clone() for n, p in self.model.named_parameters()}
        joint = joint_batch(self.data)
        for budget in (0, 8):   # one safety micro-batch, then one pair per micro-batch
            config = StepConfig(xi=.01, kappa_mu=.3, r=1e-4, alpha=1., beta=0., k=1, perturb_precision="fp32",
                                microbatch_tokens=budget)
            chunks = make_chunks(joint, budget)
            self.assertEqual(sum(1 for c in chunks if c.batch.m), 1 if budget == 0 else 2)
            for p in self.model.parameters():
                p.grad = None
            stats = unlearning_gradient(self.model, joint, config, weight=weight, chunks=chunks)
            self.assertTrue(math.isclose(stats["L_ul"], float(expected_value), rel_tol=1e-12))
            for name, p in self.model.named_parameters():
                torch.testing.assert_close(p.grad, expected[name], rtol=1e-9, atol=1e-12, msg=name)


if __name__ == "__main__":
    unittest.main()
