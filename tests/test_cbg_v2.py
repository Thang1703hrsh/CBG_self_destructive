"""Numerical and integration checks for block benign geometry and dual solve."""
import os
os.environ.setdefault("USE_TF", "0")

import unittest
from collections import OrderedDict
from unittest.mock import patch

import torch
from torch import nn
from torch.func import jacrev

import test_cbg_optimized as optimized_tests
from test_cbg_geometry import ShapeRecordingMode
from src.cbg.dual import build_functional_kernel, dual_rayleigh_solve, FunctionalKernel
from src.cbg.functionals import make_benign_functional, make_safety_functional
from src.cbg.geometry import metric_vector_product, functional_jvp
from src.cbg.rayleigh import warm_started_rayleigh_ascent
from src.cbg.spectral_grad import centered_spectral_gradient, parameter_gradient
from src.cbg.tree_ops import tree_dot, tree_norm, tree_sub, tree_scale
from src.cbg.validate import fixture
from src.core.cbg_trainer import CbgArguments


def explicit_jacobian(function, parameters):
    jac = jacrev(function)(parameters)
    return torch.cat([leaf.reshape(leaf.shape[0], -1) for leaf in jac.values()], dim=1)


def explicit_maximum(fh, fb, parameters, mu):
    h, b = explicit_jacobian(fh, parameters), explicit_jacobian(fb, parameters)
    gh, gb = h.T @ h / h.shape[0], b.T @ b / b.shape[0]
    chol = torch.linalg.cholesky(gb + mu * torch.eye(gb.shape[0], dtype=gb.dtype))
    left = torch.linalg.solve_triangular(chol, gh, upper=False)
    sym = torch.linalg.solve_triangular(chol, left.T, upper=False).T
    return torch.linalg.eigvalsh((sym + sym.T) / 2)[-1]


class CbgV2Test(unittest.TestCase):
    def setUp(self):
        self.model, self.params, self.harmful, self.safe, self.benign = fixture(torch.device("cpu"))
        self.fh = make_safety_functional(self.model, self.harmful, self.safe, chunk_size=2)
        self.fb = make_benign_functional(self.model, self.benign, chunk_size=2)

    def test_kernel_matches_explicit_jacobian_with_unequal_batches_and_chunks(self):
        benign = {n: v[:2] for n,v in self.benign.items()}
        fb = make_benign_functional(self.model, benign, chunk_size=1, token_blocks=3)
        kernel = build_functional_kernel(self.fh, fb, self.params)
        jh, jb = explicit_jacobian(self.fh, self.params), explicit_jacobian(fb, self.params)
        j = torch.cat((jh / 3**.5, jb / 6**.5))
        torch.testing.assert_close(kernel.values, j @ j.T, rtol=1e-11, atol=1e-13)
        self.assertEqual(kernel.values.shape, (9, 9))
        self.assertIsNone(kernel.values.grad_fn)

    def test_dual_recovers_leading_eigenpair_where_ascent_stalls(self):
        mu = .0006359621327540446
        result = dual_rayleigh_solve(self.fh, self.fb, self.params, mu=mu, tolerance=1e-8)
        exact = explicit_maximum(self.fh, self.fb, self.params, mu)
        torch.testing.assert_close(result.diagnostics.rho_hat, exact, rtol=1e-10, atol=1e-10)
        self.assertLess(float(result.diagnostics.relative_residual), 1e-10)
        self.assertAlmostEqual(float(result.center.b), 1., places=12)
        self.assertGreater(float(result.diagnostics.damping_fraction), .99)
        ascent = warm_started_rayleigh_ascent(self.fh, self.fb, self.params, mu=mu, inner_steps=20)
        self.assertGreater(float(result.diagnostics.rho_hat), float(ascent.diagnostics.rho_hat))
        for tree in (result.direction, result.center.gh_direction, result.center.gb_direction):
            self.assertTrue(all(not p.requires_grad and p.grad_fn is None for p in tree.values()))

    def test_small_damping_matches_explicit_reference(self):
        for mu in (1e-6, 1e-8):
            result = dual_rayleigh_solve(self.fh, self.fb, self.params, mu=mu, tolerance=1e-6)
            exact = explicit_maximum(self.fh, self.fb, self.params, mu)
            torch.testing.assert_close(result.center.rho, exact, rtol=1e-6, atol=1e-6)

    def test_fd_matches_derivative_of_actual_leading_eigenvalue(self):
        fb = make_benign_functional(self.model, self.benign, token_blocks=2, chunk_size=1)
        mu = .05
        result = dual_rayleigh_solve(self.fh, fb, self.params, mu=mu, tolerance=1e-8)
        exact = parameter_gradient(explicit_maximum(self.fh, fb, self.params, mu), self.params)
        snapshots = {n: (p.clone(), p.data_ptr()) for n,p in self.model.named_parameters()}
        fd = centered_spectral_gradient(self.fh, fb, self.params, result.direction,
            mu=mu, tau_fd=1e-4, center=result.center)
        relative = tree_norm(tree_sub(fd.gradient, exact)) / tree_norm(exact)
        self.assertLess(float(relative), 1e-6)
        for n,p in self.model.named_parameters():
            self.assertTrue(torch.equal(p, snapshots[n][0]))
            self.assertEqual(p.data_ptr(), snapshots[n][1])

    def test_zero_safety_returns_no_arbitrary_direction(self):
        result = dual_rayleigh_solve(lambda p: p["q_proj.weight"].sum().expand(2) * 0,
                                    self.fb, self.params, mu=.1)
        self.assertTrue(result.diagnostics.zero_safety)
        self.assertIsNone(result.center)
        self.assertEqual(float(tree_norm(result.direction)), 0.)
        self.assertEqual(float(result.diagnostics.rho_hat), 0.)

    def test_repeated_eigenvalues_and_zero_benign_rank(self):
        params = OrderedDict(p=torch.tensor([.2, -.1, .8], dtype=torch.float64, requires_grad=True))
        h = lambda p: p["p"][:2]
        b = lambda p: p["p"][:1] * 0
        result = dual_rayleigh_solve(h, b, params, mu=.5, tolerance=1e-8)
        self.assertAlmostEqual(float(result.center.rho), 1.)
        self.assertAlmostEqual(float(result.diagnostics.eigengap), 0.)
        self.assertLess(float(result.diagnostics.relative_residual), 1e-10)

    def test_fp32_and_sign_aligned_warm_start(self):
        self.model.float()
        result = dual_rayleigh_solve(self.fh, self.fb, self.params, mu=.1)
        reverse = tree_scale(result.direction, -1.)
        warmed = dual_rayleigh_solve(self.fh, self.fb, self.params, mu=.1, warm_start=reverse)
        self.assertGreater(float(warmed.diagnostics.warm_start_cosine), .9999)
        self.assertTrue(all(p.dtype == torch.float32 for p in warmed.direction.values()))

    def test_rejects_autocast_and_oversized_kernel(self):
        with torch.autocast("cpu", dtype=torch.bfloat16), self.assertRaisesRegex(ValueError, "autocast"):
            dual_rayleigh_solve(self.fh, self.fb, self.params, mu=.1)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            dual_rayleigh_solve(self.fh, self.fb, self.params, mu=.1, max_size=2)

    def test_corrupt_kernel_is_rejected_not_silently_repaired(self):
        original = build_functional_kernel(self.fh, self.fb, self.params)
        for kind in ("asymmetry", "indefinite", "nonfinite", "wrong_scale"):
            matrix = original.values.clone()
            if kind == "asymmetry":
                matrix[0, 1] += 1
            elif kind == "indefinite":
                matrix[0, 0] = -10
            elif kind == "nonfinite":
                matrix[0, 0] = float("nan")
            else:
                # Symmetric and PSD, but inconsistent with the real operators:
                # independent full-space verification must catch this too.
                matrix *= 2
            corrupted = FunctionalKernel(matrix, 3, 3)
            with patch("src.cbg.dual.build_functional_kernel", return_value=corrupted):
                with self.assertRaises(FloatingPointError):
                    dual_rayleigh_solve(self.fh, self.fb, self.params, mu=.1)

    def test_no_parameter_square_or_jacobian_allocation(self):
        p = sum(v.numel() for v in self.params.values())
        with ShapeRecordingMode() as recorded:
            dual_rayleigh_solve(self.fh, self.fb, self.params, mu=.1)
        for shape in ((p, p), (6, p), (p, 6), (3, p), (p, 3)):
            self.assertNotIn(shape, recorded.shapes)

    def test_block_metric_dominates_sequence_metric(self):
        # Variable response lengths, including q > length and an uneven chunk.
        benign = {n: v.clone() for n,v in self.benign.items()}
        benign["response_mask"][0, -2] = False
        fb1 = make_benign_functional(self.model, benign, token_blocks=1)
        fb4 = make_benign_functional(self.model, benign, token_blocks=4, chunk_size=2)
        j1, j4 = explicit_jacobian(fb1, self.params), explicit_jacobian(fb4, self.params)
        difference = j4.T @ j4 / 12 - j1.T @ j1 / 3
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(difference).min().detach()), -1e-12)
        torch.testing.assert_close(fb4(self.params),
            make_benign_functional(self.model, benign, token_blocks=4)(self.params))
        # For q > length, every nonempty block is one token. Manual scalar oracle.
        logits = self.model(benign["input_ids"])[0]
        mask = benign["response_mask"][:, 1:]
        expected = []
        for row in range(3):
            selected = logits[row, :-1][mask[row]].log_softmax(-1)
            targets = benign["input_ids"][row, 1:][mask[row]]
            values = selected.gather(-1, targets[:, None]).squeeze(-1)
            expected.extend(list(values * (4 / len(values))**.5))
            expected.extend([values.new_zeros(())] * (4 - len(values)))
        torch.testing.assert_close(fb4(self.params), torch.stack(expected))

    def test_opposing_token_derivatives_no_longer_cancel(self):
        class Opposing(nn.Module):
            def __init__(self):
                super().__init__()
                self.p = nn.Parameter(torch.zeros((), dtype=torch.float64))
            def forward(self, input_ids, attention_mask=None):
                slopes = self.p * torch.tensor([[1., -1., 0.]], dtype=torch.float64)
                return (torch.stack((slopes, torch.zeros_like(slopes)), dim=-1),)
        model = Opposing().eval()
        batch = dict(input_ids=torch.zeros(1, 3, dtype=torch.long),
                     response_mask=torch.tensor([[False, True, True]]))
        params = OrderedDict(model.named_parameters())
        direction = OrderedDict(p=torch.ones((), dtype=torch.float64))
        q = []
        for blocks in (1, 2):
            fb = make_benign_functional(model, batch, token_blocks=blocks)
            q.append(tree_dot(direction, metric_vector_product(fb, params, direction, detach_result=True)))
        self.assertEqual(float(q[0]), 0.)
        self.assertAlmostEqual(float(q[1]), .25)

    def test_dual_trainer_step_and_zero_safety_branch(self):
        fixture_test = optimized_tests.OptimizedCbgTest()
        fixture_test.setUp()
        trainer, metrics = fixture_test.trainer_fixture(geometry_solver="dual", benign_token_blocks=2)
        batch = {"harmful": fixture_test.harmful, "safe": fixture_test.safe, "benign": fixture_test.benign}
        trainer.training_step(fixture_test.model, batch)
        self.assertEqual(metrics["cbg/kernel_columns"], 9)
        self.assertEqual(metrics["cbg/converged"], 1.)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in fixture_test.params.values()))
        fixture_test.model.zero_grad()
        batch["harmful"] = batch["safe"]
        with patch("src.core.cbg_trainer.centered_spectral_gradient", side_effect=AssertionError("zero safety FD")):
            trainer.training_step(fixture_test.model, batch)
        self.assertEqual(metrics["cbg/zero_safety"], 1.)
        self.assertEqual(metrics["cbg/enabled"], 1.)
        self.assertEqual(metrics["train/grad_norm_cbg"], 0.)
        self.assertIsNone(trainer.prev_u)

    def test_lora_zero_factor_initialization(self):
        params = OrderedDict(a=torch.tensor([.2, .3], dtype=torch.float64, requires_grad=True),
                             b=torch.zeros(2, dtype=torch.float64, requires_grad=True))
        h = lambda p: (p["a"] * p["b"]).sum().reshape(1)
        b = lambda p: (p["a"] * p["b"])[0:1]
        result = dual_rayleigh_solve(h, b, params, mu=.1, tolerance=1e-8)
        torch.testing.assert_close(result.direction["a"], torch.zeros_like(params["a"]))
        torch.testing.assert_close(result.center.rho, explicit_maximum(h, b, params, .1))

    def test_configuration_rejects_invalid_variant(self):
        for values in (dict(geometry_solver="unknown"), dict(benign_token_blocks=0),
                       dict(dual_tolerance=0), dict(dual_tolerance=float("nan")),
                       dict(dual_kernel_tolerance=float("inf"))):
            with self.assertRaises(ValueError):
                CbgArguments(**values)

    def test_equal_safety_benign_maps_avoid_woodbury_cancellation(self):
        params = OrderedDict(p=torch.tensor([.2, .5, -.1], dtype=torch.float64, requires_grad=True))
        function = lambda p: p["p"] * torch.tensor([1., 2., 3.], dtype=torch.float64)
        mu = 1e-10
        result = dual_rayleigh_solve(function, function, params, mu=mu, tolerance=1e-4)
        self.assertAlmostEqual(float(result.center.rho), 3 / (3 + mu), places=9)
        self.assertLess(float(result.diagnostics.relative_residual), 1e-7)

    def test_failed_kernel_call_restores_live_parameters(self):
        before = {n: (p.clone(), p.data_ptr()) for n,p in self.model.named_parameters()}
        with patch("src.cbg.dual.functional_jvp", side_effect=RuntimeError("injected")):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                dual_rayleigh_solve(self.fh, self.fb, self.params, mu=.1)
        for n,p in self.model.named_parameters():
            self.assertTrue(torch.equal(p, before[n][0]))
            self.assertEqual(p.data_ptr(), before[n][1])

    def test_v2_yaml_parser_and_resume_fingerprint(self):
        from pathlib import Path
        from dataclasses import fields, asdict
        from transformers import TrainingArguments
        from src.utils.config import MyParser, load_yaml_config
        from src.core.cbg_trainer import _config_fingerprint, CBG_STATE_VERSION
        config = load_yaml_config(Path(__file__).resolve().parents[1] / "config" / "cbg_v2_smoke_1b.yaml")
        for section, cls in (("CbgArguments", CbgArguments), ("TrainingArguments", TrainingArguments)):
            self.assertFalse(set(config[section]) - {f.name for f in fields(cls)})
        (args,) = MyParser((CbgArguments,)).parse_args_into_dataclasses(args=[], default_values=config["CbgArguments"])
        self.assertEqual(args.geometry_solver, "dual")
        self.assertEqual(args.benign_token_blocks, 2)
        self.assertFalse(config["TrainingArguments"]["bf16"])
        base = asdict(CbgArguments())
        self.assertNotEqual(_config_fingerprint(base), _config_fingerprint(asdict(args)))
        changed = dict(base, benign_token_blocks=2)
        self.assertNotEqual(_config_fingerprint(base), _config_fingerprint(changed))
        self.assertEqual(CBG_STATE_VERSION, 4)


if __name__ == "__main__":
    unittest.main()
