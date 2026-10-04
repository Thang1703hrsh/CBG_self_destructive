"""CPU regression tests for SEAM gradient boundaries and numerical updates."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
import torch.nn.functional as F
from src.core.numerics import CPUAdamW, collate_seam_batch, masked_causal_ce, parameter_perturbation


class SeamFixTests(unittest.TestCase):
    def test_first_response_token_has_correct_gradient(self):
        logits = torch.randn(1, 4, 7, requires_grad=True)
        labels = torch.tensor([[1, 2, 3, 4]])
        mask = torch.tensor([[0, 0, 1, 1]])
        loss = masked_causal_ce(logits, labels, mask)
        expected = F.cross_entropy(logits[0, 1:3], labels[0, 2:4])
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertGreater(logits.grad[0, 1].norm().item(), 0)
        self.assertEqual(logits.grad[0, 0].norm().item(), 0)

    def test_padding_and_token_zero_are_handled_without_mutation(self):
        logits = torch.randn(1, 4, 7, requires_grad=True)
        labels = torch.tensor([[1, 0, 3, -100]])
        saved = labels.clone()
        loss = masked_causal_ce(logits, labels, torch.ones_like(labels))
        self.assertTrue(torch.allclose(loss, F.cross_entropy(logits[0, :2], labels[0, 1:3])))
        self.assertTrue(torch.equal(labels, saved))
        with self.assertRaisesRegex(ValueError, 'no selected'):
            masked_causal_ce(logits, labels, torch.zeros_like(labels))

    def test_collator_preserves_real_eos_and_dataset_inputs(self):
        row = {'input_ids': torch.tensor([1, 2, 2]), 'attention_mask': torch.tensor([1, 1, 0])}
        original = row['input_ids'].clone()
        batch = collate_seam_batch([row])
        self.assertEqual(batch['labels'].tolist(), [[1, 2, -100]])
        self.assertTrue(torch.equal(row['input_ids'], original))

    def test_fp32_perturbation_is_measured_and_restored_even_on_error(self):
        model = torch.nn.Linear(32, 32, bias=False)
        original = model.weight.detach().clone()
        grad = {'weight': torch.ones_like(model.weight)}
        norm = grad['weight'].norm()
        with self.assertRaisesRegex(RuntimeError, 'inside'):
            with parameter_perturbation(model, grad, grad, norm, norm, 0., .001) as stats:
                self.assertGreater(stats['changed_fraction'], .99)
                self.assertAlmostEqual(stats['actual_norm'], .001, places=6)
                raise RuntimeError('inside')
        self.assertTrue(torch.equal(model.weight, original))

    def test_bf16_zero_perturbation_fails_and_restores(self):
        model = torch.nn.Linear(32, 32, bias=False).to(torch.bfloat16)
        with torch.no_grad(): model.weight.fill_(1.)
        original = model.weight.detach().clone()
        grad = {'weight': torch.ones_like(model.weight, dtype=torch.float32)}
        with self.assertRaisesRegex(FloatingPointError, 'rounded entirely'):
            with parameter_perturbation(model, grad, grad, 32., 32., 0., .001):
                pass
        self.assertTrue(torch.equal(model.weight, original))

    def test_cpu_adamw_matches_pytorch_and_resume(self):
        a = torch.nn.Parameter(torch.tensor([.125, -.375, .75]))
        b = torch.nn.Parameter(a.detach().clone())
        custom = CPUAdamW([a], lr=2e-5, weight_decay=.01)
        reference = torch.optim.AdamW([b], lr=2e-5, weight_decay=.01, foreach=False)
        for step in range(10):
            a.grad = b.grad = torch.tensor([.1 + step, -.2, .3])
            custom.step(); reference.step()
            self.assertTrue(torch.allclose(a, b, atol=1e-7, rtol=1e-6))
        resumed = CPUAdamW([a], lr=2e-5)
        resumed.load_state_dict(custom.state_dict())
        a.grad = b.grad = torch.tensor([.2, -.5, 1.])
        resumed.step(); reference.step()
        self.assertTrue(torch.allclose(a, b, atol=1e-7, rtol=1e-6))
        self.assertEqual(resumed.state[a]['exp_avg'].device.type, 'cpu')


if __name__ == '__main__':
    unittest.main()
