"""Trajectory CBG: attacker loss/direction, in-place shift and restore, geometry at theta'."""
import os
os.environ.setdefault("USE_TF", "0")
import copy
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from test_cbg_spec import fixture, fp64_norm, fp64_attention
from src.cbg.spec import joint_batch
from src.cbg.step_v2 import StepConfig, attack_sign_direction, cbg_step, harmful_sft_loss, shift_weights


class TrajectoryTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        for target, fn in (("LlamaRMSNorm.forward", fp64_norm), ("eager_attention_forward", fp64_attention)):
            p = patch("transformers.models.llama.modeling_llama." + target, fn)
            p.start()
            self.addCleanup(p.stop)
        self.model, self.data, _ = fixture()
        self.data["harmful"]["response_mask"][0, 1] = True   # unequal response lengths
        self.joint = joint_batch(self.data)
        self.cfg = StepConfig(xi=.01, kappa_mu=.3, r=1e-4, alpha=0., beta=.1, k=1, perturb_precision="fp32",
                              logprob_chunk_size=0, microbatch_tokens=8)

    def manual_loss(self, model):
        h = self.data["harmful"]
        logits = model(input_ids=h["input_ids"], attention_mask=h["attention_mask"]).logits
        mask = h["response_mask"][:, 1:].bool() & h["attention_mask"][:, 1:].bool()
        return F.cross_entropy(logits[:, :-1][mask], h["input_ids"][:, 1:][mask])

    def test_attacker_loss_and_sign_direction_match_autograd(self):
        expected = self.manual_loss(self.model)
        self.assertAlmostEqual(harmful_sft_loss(self.model, self.joint, self.cfg), float(expected), places=12)
        grads = torch.autograd.grad(expected, list(self.model.parameters()))
        signs, loss = attack_sign_direction(self.model, self.joint, self.cfg)
        self.assertAlmostEqual(loss, float(expected), places=12)
        for (p, sign), g in zip(signs, grads):
            self.assertEqual(sign.dtype, torch.int8)
            torch.testing.assert_close(sign, torch.sign(g).to(torch.int8))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))   # .grad left clean

    def test_shift_moves_toward_lower_attacker_loss_and_restores(self):
        before = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        signs, loss = attack_sign_direction(self.model, self.joint, self.cfg)
        shift_weights(signs, -1e-3)
        self.assertLess(harmful_sft_loss(self.model, self.joint, self.cfg), loss)   # theta' is further along the attack
        shift_weights(signs, 1e-3)
        for n, p in self.model.named_parameters():
            torch.testing.assert_close(p.detach(), before[n], rtol=0, atol=1e-15, msg=n)

    def test_geometry_gradient_at_shifted_point_equals_an_independent_copy(self):
        alpha = 1e-3
        signs, _ = attack_sign_direction(self.model, self.joint, self.cfg)
        reference = copy.deepcopy(self.model)
        with torch.no_grad():
            for (p, sign), q in zip(signs, reference.parameters()):
                q.sub_(alpha * sign.to(q.dtype))
        kwargs = dict(include_base=False, signs=torch.ones(1, len(self.joint.seq_to_example.unique()), dtype=torch.long),
                      fixed_samples=None)
        torch.manual_seed(0)
        cbg_step(reference, self.joint, self.cfg, sample_generator=torch.Generator().manual_seed(3), **kwargs)
        before = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        shift_weights(signs, -alpha)
        try:
            cbg_step(self.model, self.joint, self.cfg, sample_generator=torch.Generator().manual_seed(3), **kwargs)
        finally:
            shift_weights(signs, alpha)
        for (n, p), q in zip(self.model.named_parameters(), reference.parameters()):
            if q.grad is None:
                self.assertIsNone(p.grad, n)
                continue
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-9, atol=1e-12, msg=n)
            torch.testing.assert_close(p.detach(), before[n], rtol=0, atol=1e-15, msg=n)


if __name__ == "__main__":
    unittest.main()
