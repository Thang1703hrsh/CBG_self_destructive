"""Execution optimizations must preserve the V5 maps, eigenpair and gradient."""
import os
os.environ.setdefault("USE_TF", "0")
import unittest
from collections import OrderedDict
from unittest.mock import patch
import torch
from src.cbg.dual import build_functional_kernel, dual_rayleigh_solve
from src.cbg.functionals import make_safety_functional, make_benign_functional
from src.cbg.spectral_grad import direct_spectral_gradient
from src.cbg.tree_ops import tree_norm, tree_sub
from src.cbg.validate import fixture
from test_cbg_geometry import ShapeRecordingMode


class ExecutionTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model, self.p, self.h, self.s, self.b = fixture(torch.device("cpu"))

    def maps(self, chunk=1, q=2):
        return (make_safety_functional(self.model, self.h, self.s, chunk_size=chunk),
                make_benign_functional(self.model, self.b, chunk_size=chunk, token_blocks=q))

    def test_batched_columns_kernel_eigenpair_gradient_and_detach(self):
        for dtype in (torch.float32, torch.float64):
            self.model.to(dtype)
            for chunk in (1, 2):
                fh, fb = self.maps(chunk)
                ref = build_functional_kernel(fh, fb, self.p)
                solved = dual_rayleigh_solve(fh, fb, self.p, mu=.1)
                grad = direct_spectral_gradient(fh, fb, self.p, solved.direction, mu=.1, center=solved.center)
                for width in (2, 4):  # Nine columns: uneven final group.
                    kernel = build_functional_kernel(fh, fb, self.p, column_batch_size=width)
                    tol = 2e-5 if dtype == torch.float32 else 1e-10
                    torch.testing.assert_close(kernel.values, ref.values, atol=tol, rtol=tol)
                    self.assertIsNone(kernel.values.grad_fn)
                    result = dual_rayleigh_solve(fh, fb, self.p, mu=.1, column_batch_size=width)
                    torch.testing.assert_close(result.center.rho, solved.center.rho, rtol=tol, atol=tol)
                    self.assertLess(float(result.diagnostics.relative_residual), 1e-4)
                    torch.testing.assert_close(result.center.b, torch.ones_like(result.center.b), rtol=tol, atol=tol)
                    g = direct_spectral_gradient(fh, fb, self.p, result.direction, mu=.1, center=result.center)
                    self.assertLess(float(tree_norm(tree_sub(g.gradient, grad.gradient))/tree_norm(grad.gradient)), tol)
                    self.assertTrue(all(v.grad_fn is None for v in g.gradient.values()))

    def test_no_full_jacobian_and_no_model_mutation_on_failure(self):
        fh, fb = self.maps()
        before = {n: (v.clone(), v.data_ptr()) for n, v in self.model.named_parameters()}
        count = sum(v.numel() for v in self.p.values())
        with ShapeRecordingMode() as recorded:
            build_functional_kernel(fh, fb, self.p, column_batch_size=2)
        for shape in ((count, count), (9, count), (count, 9)):
            self.assertNotIn(shape, recorded.shapes)
        with patch("src.cbg.geometry.vmap", side_effect=RuntimeError("transform failed")):
            with self.assertRaisesRegex(RuntimeError, "transform failed"):
                build_functional_kernel(fh, fb, self.p, column_batch_size=2)
        for n, v in self.model.named_parameters():
            self.assertTrue(torch.equal(v, before[n][0]))
            self.assertEqual(v.data_ptr(), before[n][1])

    def test_column_limit_and_config_validation(self):
        from src.core.cbg_trainer import CbgArguments
        fh, fb = self.maps()
        for width in (0, -1, True, 1.5, 9):
            with self.assertRaises(ValueError):
                build_functional_kernel(fh, fb, self.p, column_batch_size=width)
        with self.assertRaises(ValueError):
            CbgArguments(dual_column_batch_size=0)

    def test_zero_safety_and_unequal_sizes(self):
        fh, _ = self.maps()
        fb = make_benign_functional(self.model, {k: v[:2] for k,v in self.b.items()}, token_blocks=3, chunk_size=1)
        torch.testing.assert_close(build_functional_kernel(fh, fb, self.p, column_batch_size=4).values,
                                   build_functional_kernel(fh, fb, self.p).values)
        zero = lambda p: p[next(iter(p))].sum().expand(3) * 0
        self.assertTrue(dual_rayleigh_solve(zero, fb, self.p, mu=.1, column_batch_size=2).diagnostics.zero_safety)

    def test_refinement_repairs_perturbed_reconstruction_without_relaxing_gate(self):
        from src.cbg.geometry import functional_vjp
        p = OrderedDict(w=torch.ones(2, dtype=torch.float64, requires_grad=True))
        fh = lambda p: p["w"] * p["w"].new_tensor([2., 1.])
        fb = lambda p: p["w"] * 0
        calls = 0
        def noisy(*args):
            nonlocal calls
            result = functional_vjp(*args)
            if calls == 0:
                result["w"][1] += 1e-4
            calls += 1
            return result
        with patch("src.cbg.dual.functional_vjp", side_effect=noisy):
            with self.assertRaisesRegex(FloatingPointError, "verification"):
                dual_rayleigh_solve(fh, fb, p, mu=.1, tolerance=1e-6)
        calls = 0
        with patch("src.cbg.dual.functional_vjp", side_effect=noisy):
            separated = dual_rayleigh_solve(
                fh, fb, p, mu=.1, tolerance=1e-6,
                residual_tolerance=1e-5,
            )
        self.assertGreater(float(separated.diagnostics.relative_residual), 1e-6)
        self.assertLess(float(separated.diagnostics.relative_residual), 1e-5)
        self.assertLess(float(separated.diagnostics.dual_value_error), 1e-6)
        self.assertEqual(separated.diagnostics.refinement_steps, 0)
        calls = 0
        with patch("src.cbg.dual.functional_vjp", side_effect=noisy):
            r = dual_rayleigh_solve(fh, fb, p, mu=.1, tolerance=1e-6, refinement_steps=2)
        self.assertEqual(r.diagnostics.refinement_steps, 1)
        self.assertLess(float(r.diagnostics.relative_residual), 1e-10)
        self.assertAlmostEqual(float(r.center.rho), 20.)
        self.assertAlmostEqual(float(r.center.b), 1.)
        self.assertTrue(all(v.grad_fn is None for v in r.direction.values()))

    def test_refinement_does_not_accept_corrupt_kernel(self):
        from src.cbg.dual import FunctionalKernel
        fh, fb = self.maps()
        kernel = build_functional_kernel(fh, fb, self.p)
        wrong = FunctionalKernel(2 * kernel.values, kernel.safety_size, kernel.benign_size)
        with patch("src.cbg.dual.build_functional_kernel", return_value=wrong):
            with self.assertRaises(FloatingPointError):
                dual_rayleigh_solve(fh, fb, self.p, mu=.1, refinement_steps=3)

    def test_tiny_qwen_transform_compatibility(self):
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from torch import nn
        torch.manual_seed(32)
        config = Qwen2Config(vocab_size=17, hidden_size=8, intermediate_size=16,
                            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                            max_position_embeddings=32, use_cache=False, attention_dropout=0.)
        config._attn_implementation = "eager"
        model = Qwen2ForCausalLM(config).double().eval().requires_grad_(False)
        class LoRA(nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base
                self.lora_A = nn.Parameter(torch.randn(2, base.in_features, dtype=torch.float64) * .1)
                self.lora_B = nn.Parameter(torch.zeros(base.out_features, 2, dtype=torch.float64))
            def forward(self, x):
                return self.base(x) + nn.functional.linear(nn.functional.linear(x, self.lora_A), self.lora_B)
        attn = model.model.layers[0].self_attn
        attn.q_proj, attn.v_proj = LoRA(attn.q_proj), LoRA(attn.v_proj)
        model.eval()
        p = OrderedDict((n,v) for n,v in model.named_parameters() if v.requires_grad)
        def batch(ids):
            return dict(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1,5,dtype=torch.long),
                        response_mask=torch.tensor([[False,False,False,True,True]]))
        fh = make_safety_functional(model, batch([1,2,3,4,5]), batch([1,2,3,6,7]), chunk_size=1)
        fb = make_benign_functional(model, batch([1,8,9,10,11]), chunk_size=1, token_blocks=2)
        reference = build_functional_kernel(fh, fb, p)
        actual = build_functional_kernel(fh, fb, p, column_batch_size=2)
        torch.testing.assert_close(actual.values, reference.values, atol=1e-12, rtol=1e-10)

    def test_trainer_retries_only_dual_verification_at_narrower_width(self):
        from src.core.cbg_trainer import CbgArguments, CbgTrainer
        fh, fb = self.maps()
        accepted = dual_rayleigh_solve(fh, fb, self.p, mu=.1, column_batch_size=2)
        trainer = object.__new__(CbgTrainer)
        trainer.cbg_args = CbgArguments(
            cbg_version="v5", geometry_solver="dual",
            spectral_derivative_backend="direct_rof", benign_token_blocks=2,
            dual_column_batch_size=4, dual_column_fallback=True,
        )
        failure = FloatingPointError(
            "dual reconstruction failed independent verification: residual=0.000106, value_error=1e-8"
        )
        with patch("src.core.cbg_trainer.dual_rayleigh_solve",
                   side_effect=(failure, accepted)) as solve:
            result = trainer._search_geometry(fh, fb, self.p, mu=.1, warm_start=None)
        self.assertIs(result, accepted)
        self.assertEqual([call.kwargs["column_batch_size"] for call in solve.call_args_list], [4, 2])
        self.assertEqual(trainer._effective_dual_column_width, 2)
        self.assertEqual(trainer._dual_column_fallbacks, 1)

        trainer.cbg_args.dual_column_batch_size = 3
        with patch("src.core.cbg_trainer.dual_rayleigh_solve",
                   side_effect=(failure, accepted)) as solve:
            result = trainer._search_geometry(fh, fb, self.p, mu=.1, warm_start=None)
        self.assertIs(result, accepted)
        self.assertEqual(
            [call.kwargs["column_batch_size"] for call in solve.call_args_list],
            [3, 2],
        )
        self.assertEqual(trainer._effective_dual_column_width, 2)
        self.assertEqual(trainer._dual_column_fallbacks, 1)

        for unsafe in (
            FloatingPointError("functional kernel is materially non-PSD"),
            FloatingPointError("non-finite functional kernel"),
        ):
            with patch("src.core.cbg_trainer.dual_rayleigh_solve", side_effect=unsafe) as solve:
                with self.assertRaises(FloatingPointError):
                    trainer._search_geometry(fh, fb, self.p, mu=.1, warm_start=None)
            self.assertEqual(solve.call_count, 1)

if __name__ == "__main__":
    unittest.main()
