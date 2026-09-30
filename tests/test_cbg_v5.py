"""V5 equations tested against independent dense tiny-model oracles."""
import os
os.environ.setdefault("USE_TF", "0")

import unittest
import tempfile
from pathlib import Path
from collections import OrderedDict
from unittest.mock import patch

import torch
from torch.func import jacrev

from src.cbg.validate import fixture
from src.cbg.functionals import make_safety_functional, make_benign_functional
from src.cbg.dual import dual_rayleigh_solve
from src.cbg.bounded import bounded_rayleigh_solve, compute_complement_enclosure
from src.cbg.spectral_grad import direct_spectral_gradient, centered_spectral_gradient, parameter_gradient
from src.cbg.tree_ops import tree_dot, tree_norm, tree_sub, tree_cosine_similarity, tree_detach
from src.core.cbg_trainer import CbgArguments
import test_cbg_optimized as optimized_tests
from test_cbg_geometry import ShapeRecordingMode


def explicit_value(fh, fb, parameters, mu):
    # Explicit Jacobians and P-square matrices exist only in tiny tests.
    hsize, bsize = fh(parameters).numel(), fb(parameters).numel()
    a = torch.cat([t.reshape(hsize, -1) for t in jacrev(fh)(parameters).values()], 1) / hsize**.5
    c = torch.cat([t.reshape(bsize, -1) for t in jacrev(fb)(parameters).values()], 1) / bsize**.5
    b = c.T @ c + mu * torch.eye(c.shape[1], dtype=c.dtype)
    l = torch.linalg.cholesky(b)
    w = torch.linalg.solve_triangular(l, a.T, upper=False)
    return torch.linalg.eigvalsh(w.T @ w)[-1]


class V5Test(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model, self.p, self.h, self.s, self.b = fixture(torch.device("cpu"))
        self.mu = .0006359621327540446

    def maps(self, q=2, chunk=1):
        return (make_safety_functional(self.model, self.h, self.s, chunk_size=chunk),
                make_benign_functional(self.model, self.b, chunk_size=chunk, token_blocks=q))

    def test_direct_matches_actual_eigenvalue_gradient_fd_and_chunking(self):
        for q in (1, 2):
            gradients = []
            for chunk in (0, 1, 2):
                fh, fb = self.maps(q, chunk)
                result = dual_rayleigh_solve(fh, fb, self.p, mu=self.mu, tolerance=1e-8)
                direct = direct_spectral_gradient(fh, fb, self.p, result.direction, mu=self.mu, center=result.center)
                exact = parameter_gradient(explicit_value(fh, fb, self.p, self.mu), self.p)
                self.assertLess(float(tree_norm(tree_sub(direct.gradient, exact))/tree_norm(exact)), 1e-8)
                for radius in (1e-5, 1e-4, 1e-3):
                    fd = centered_spectral_gradient(fh, fb, self.p, result.direction, mu=self.mu,
                                                   center=result.center, tau_fd=radius)
                    self.assertGreater(float(tree_cosine_similarity(direct.gradient, fd.gradient)), .99999)
                    self.assertLess(float(tree_norm(tree_sub(direct.gradient, fd.gradient))/tree_norm(direct.gradient)), 1e-4)
                self.assertIsNone(direct.delta)
                self.assertTrue(all(t.grad_fn is None and not t.requires_grad for t in direct.gradient.values()))
                gradients.append(direct.gradient)
            self.assertLess(float(tree_norm(tree_sub(gradients[0], gradients[-1]))/tree_norm(gradients[0])), 1e-8)

    def test_direct_live_parameters_exception_and_stale_cache(self):
        fh, fb = self.maps()
        r = dual_rayleigh_solve(fh, fb, self.p, mu=self.mu)
        before = {n: (p.clone(), p.data_ptr()) for n, p in self.model.named_parameters()}
        direct_spectral_gradient(fh, fb, self.p, r.direction, mu=self.mu, center=r.center)
        with patch("src.cbg.spectral_grad.jvp", side_effect=RuntimeError("unsupported transform")):
            with self.assertRaisesRegex(RuntimeError, "unsupported"):
                direct_spectral_gradient(fh, fb, self.p, r.direction, mu=self.mu, center=r.center)
        with patch.object(self.model, "forward", side_effect=RuntimeError("inside functional substitution")):
            with self.assertRaisesRegex(RuntimeError, "inside functional"):
                direct_spectral_gradient(fh, fb, self.p, r.direction, mu=self.mu, center=r.center)
        for n, p in self.model.named_parameters():
            self.assertTrue(torch.equal(p, before[n][0]))
            self.assertEqual(p.data_ptr(), before[n][1])
        with self.assertRaisesRegex(ValueError, "stale"):
            direct_spectral_gradient(fh, fb, self.p, r.direction, mu=self.mu * 2, center=r.center)

    def test_affine_zero_hessian_and_disconnected_tangent_guard(self):
        p = OrderedDict(w=torch.tensor([1., 2.], dtype=torch.float64, requires_grad=True))
        fh, fb = lambda p: p["w"], lambda p: p["w"][:1]
        r = dual_rayleigh_solve(fh, fb, p, mu=.1)
        g = direct_spectral_gradient(fh, fb, p, r.direction, mu=.1, center=r.center, allow_constant_tangent=True)
        self.assertEqual(float(tree_norm(g.gradient)), 0.)
        with patch("src.cbg.spectral_grad.jvp", return_value=(torch.tensor(0.), torch.tensor(0.))):
            with self.assertRaisesRegex(RuntimeError, "constant tangent"):
                direct_spectral_gradient(fh, fb, p, r.direction, mu=.1, center=r.center)

    def test_hidden_dominant_mode_enclosure_and_row_enrichment(self):
        p = OrderedDict(w=torch.zeros(2, dtype=torch.float64, requires_grad=True))
        fh = lambda p: p["w"] * p["w"].new_tensor([2**.5, 200**.5])
        fb = lambda p: p["w"] * 0
        v = OrderedDict(w=torch.tensor([1., 0.], dtype=torch.float64))
        bound = compute_complement_enclosure(fh, fb, p, [v], mu=1., tolerance=1e-10)
        self.assertAlmostEqual(float(bound.lower), 1.)
        self.assertAlmostEqual(float(bound.upper), 100.)
        self.assertAlmostEqual(float(bound.eta), 0.)
        with patch("src.cbg.bounded.dual_rayleigh_solve", side_effect=AssertionError("unnecessary fallback")):
            r = bounded_rayleigh_solve(fh, fb, p, mu=1., warm_start=v, initial_size=1,
                max_size=2, max_enrichments=2, enclosure_atol=1e-9, enclosure_rtol=1e-9)
        self.assertAlmostEqual(float(r.center.rho), 100.)
        self.assertFalse(r.diagnostics.dual_fallback)
        self.assertGreater(r.diagnostics.enclosure_checks, 1)
        self.assertAlmostEqual(float(r.center.b), 1.)

    def test_random_enclosures_and_orthogonality_rejection(self):
        torch.manual_seed(51)
        for _ in range(12):
            a, c = torch.randn(3, 5, dtype=torch.float64), torch.randn(2, 5, dtype=torch.float64)
            p = OrderedDict(w=torch.randn(5, dtype=torch.float64, requires_grad=True))
            fh, fb = lambda p: a @ p["w"], lambda p: c @ p["w"]
            mu = .01
            b = c.T @ c / 2 + mu * torch.eye(5, dtype=torch.float64)
            v = torch.randn(5, 2, dtype=torch.float64)
            l = torch.linalg.cholesky(v.T @ b @ v)
            v = torch.linalg.solve_triangular(l, v.T, upper=False).T
            basis = [OrderedDict(w=v[:, i]) for i in range(2)]
            bound = compute_complement_enclosure(fh, fb, p, basis, mu=mu)
            rho = explicit_value(fh, fb, p, mu)
            self.assertLessEqual(float(bound.lower), float(rho) + 1e-9)
            self.assertGreaterEqual(float(bound.upper), float(rho) - 1e-9)
            with self.assertRaisesRegex(FloatingPointError, "orthogonality"):
                compute_complement_enclosure(fh, fb, p, basis + basis[:1], mu=mu)

    def test_same_batch_fallback_and_cap(self):
        fh, fb = self.maps()
        r = bounded_rayleigh_solve(fh, fb, self.p, mu=self.mu, max_enrichments=0)
        oracle = dual_rayleigh_solve(fh, fb, self.p, mu=self.mu)
        self.assertTrue(r.diagnostics.dual_fallback)
        torch.testing.assert_close(r.center.rho, oracle.center.rho)
        self.assertGreater(r.diagnostics.metric_evaluations, oracle.diagnostics.metric_evaluations)
        self.assertTrue(all(not t.requires_grad for v in r.warm_basis for t in v.values()))
        with self.assertRaisesRegex(ValueError, "fallback exceeds"):
            bounded_rayleigh_solve(fh, fb, self.p, mu=self.mu, dual_max_size=8)

    def test_v5_trainer_dispatch_total_objective_and_beta_zero(self):
        f = optimized_tests.OptimizedCbgTest()
        f.setUp()
        trainer, metrics = f.trainer_fixture(cbg_version="v5", geometry_solver="dual",
            spectral_derivative_backend="direct_rof", benign_token_blocks=2, beta_mode="fixed")
        batch = dict(harmful=f.harmful, safe=f.safe, benign=f.benign)
        with patch("src.core.cbg_trainer.centered_spectral_gradient", side_effect=AssertionError("V5 used FD")):
            loss = trainer.training_step(f.model, batch)
        self.assertEqual(metrics["cbg/direct_rof"], 1.)
        self.assertEqual(metrics["cbg/fd_diagnostics_applicable"], 0.)
        expected = metrics["train/l_aln"] + trainer.cbg_args.alpha*metrics["train/l_bgn"] + metrics["train/beta_effective"]*metrics["train/rho_hat"]
        self.assertAlmostEqual(float(loss)*trainer._gradient_accumulation_divisor(), float(expected))
        trainer.cbg_args.beta = 0.
        with patch.object(trainer, "_search_geometry", side_effect=AssertionError("beta zero geometry")):
            trainer.training_step(f.model, batch)
        self.assertEqual(metrics["cbg/enabled"], 0.)

    def test_v5_scalar_functional_fallback_keeps_strict_gate(self):
        import gc
        import weakref
        f = optimized_tests.OptimizedCbgTest()
        f.setUp()
        trainer, metrics = f.trainer_fixture(
            cbg_version="v5", geometry_solver="dual",
            spectral_derivative_backend="direct_rof", benign_token_blocks=2,
            beta_mode="fixed",
        )
        trainer.cbg_args.functional_chunk_size = 2
        trainer.cbg_args.dual_functional_chunk_fallback = True
        original_search = trainer._search_geometry
        calls = []
        failure_message = (
            "dual reconstruction failed independent verification: "
            "residual=0.00488, residual_tolerance=0.002, "
            "value_error=2.86e-7, value_tolerance=0.0001"
        )
        failed_tensors = []

        def fail_then_search(safety, benign, parameters, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                scratch = torch.ones(10)
                failed_tensors.append(weakref.ref(scratch))
                raise FloatingPointError(failure_message)
            gc.collect()
            self.assertIsNone(failed_tensors[0](), "failed solver traceback retains tensors during retry")
            return original_search(safety, benign, parameters, **kwargs)

        batch = dict(harmful=f.harmful, safe=f.safe, benign=f.benign)
        with patch.object(trainer, "_search_geometry", side_effect=fail_then_search):
            loss = trainer.training_step(f.model, batch)

        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(len(calls), 2)
        self.assertNotIn("dual_column_batch_size", calls[0])
        self.assertEqual(calls[1]["dual_column_batch_size"], 1)
        self.assertFalse(calls[1]["dual_column_fallback"])
        self.assertEqual(metrics["cbg/effective_functional_chunk_size"], 1.)
        self.assertEqual(metrics["cbg/functional_chunk_fallbacks"], 1.)
        self.assertEqual(metrics["cbg/dual_residual_tolerance"], 1e-4)

    def test_combined_gradient_clipping_and_diagnostics(self):
        f = optimized_tests.OptimizedCbgTest()
        f.setUp()
        trainer, metrics = f.trainer_fixture(
            cbg_version="v5", geometry_solver="dual", spectral_derivative_backend="direct_rof",
            benign_token_blocks=2, beta=10., beta_mode="fixed",
        )
        batch = dict(harmful=f.harmful, safe=f.safe, benign=f.benign)
        trainer.training_step(f.model, batch)
        raw = OrderedDict((n, p.grad.clone()) for n, p in f.params.items())
        norm = tree_norm(raw)
        torch.testing.assert_close(norm, metrics["train/grad_norm_composed"])
        self.assertGreater(float(norm), 1.)
        # The Trainer/Accelerator optimizer boundary uses this operation on the
        # accumulated .grad buffers, not on spectral components individually.
        measured = torch.nn.utils.clip_grad_norm_(list(f.params.values()), 1.)
        clipped = OrderedDict((n, p.grad) for n, p in f.params.items())
        torch.testing.assert_close(measured, norm)
        self.assertLessEqual(float(tree_norm(clipped)), 1. + 1e-6)
        self.assertGreater(float(tree_cosine_similarity(raw, clipped)), .999999)
        cosine = float(metrics["train/cbg_preservation_cosine"])
        self.assertTrue(-1. <= cosine <= 1.)
        pnorm = metrics["train/grad_norm_preservation"]
        snorm = metrics["train/grad_norm_beta_cbg"]
        torch.testing.assert_close(norm.square(), pnorm.square() + snorm.square() + 2*pnorm*snorm*cosine)

    def test_scalar_retry_failure_stops_before_gradient_injection(self):
        f = optimized_tests.OptimizedCbgTest()
        f.setUp()
        trainer, _ = f.trainer_fixture(
            cbg_version="v5", geometry_solver="dual", spectral_derivative_backend="direct_rof",
            dual_functional_chunk_fallback=True,
        )
        trainer.cbg_args.functional_chunk_size = 2
        batch = dict(harmful=f.harmful, safe=f.safe, benign=f.benign)
        for message, expected_calls in (
            ("dual reconstruction failed independent verification: residual=0.00488", 2),
            ("non-finite functional kernel", 1),
        ):
            with patch.object(trainer, "_search_geometry", side_effect=FloatingPointError(message)) as solve:
                with self.assertRaises(FloatingPointError):
                    trainer.training_step(f.model, batch)
            self.assertEqual(solve.call_count, expected_calls)
            self.assertTrue(all(p.grad is None for p in f.params.values()))

    def test_clipping_is_in_resume_runtime_signature(self):
        f = optimized_tests.OptimizedCbgTest()
        f.setUp()
        trainer, _ = f.trainer_fixture()
        trainer.args.max_grad_norm = 1.
        first = trainer._checkpoint_runtime()
        trainer.args.max_grad_norm = 0.
        self.assertNotEqual(first, trainer._checkpoint_runtime())

    def test_no_dense_parameter_matrices_and_fp32(self):
        self.model.float()
        self.p = OrderedDict((n, p) for n, p in self.model.named_parameters() if p.requires_grad)
        fh, fb = self.maps()
        recorder = ShapeRecordingMode()
        with recorder:
            r = bounded_rayleigh_solve(fh, fb, self.p, mu=.1, max_enrichments=1)
            g = direct_spectral_gradient(fh, fb, self.p, r.direction, mu=.1, center=r.center)
        count = sum(p.numel() for p in self.p.values())
        for shape in ((count, count), (9, count), (count, 9)):
            self.assertNotIn(shape, recorder.shapes)
        fd = centered_spectral_gradient(fh, fb, self.p, r.direction, mu=.1, center=r.center, tau_fd=1e-3)
        self.assertGreater(float(tree_cosine_similarity(g.gradient, fd.gradient)), .999)
        self.assertLess(float(tree_norm(tree_sub(g.gradient, fd.gradient))/tree_norm(g.gradient)), .01)

    def test_lora_zero_factor_direct_matches_full_eigen_derivative(self):
        p = OrderedDict(a=torch.tensor([.2, .3], dtype=torch.float64, requires_grad=True),
                        b=torch.zeros(2, dtype=torch.float64, requires_grad=True))
        fh = lambda p: (p["a"] * p["b"]).sum().reshape(1)
        fb = lambda p: (p["a"] * p["b"])[0:1]
        r = dual_rayleigh_solve(fh, fb, p, mu=.1, tolerance=1e-8)
        direct = direct_spectral_gradient(fh, fb, p, r.direction, mu=.1, center=r.center)
        exact = parameter_gradient(explicit_value(fh, fb, p, .1), p)
        self.assertLess(float(tree_norm(tree_sub(direct.gradient, exact))), 1e-10)

    def test_nonfinite_and_unsupported_precision_raise(self):
        fh, fb = self.maps()
        r = dual_rayleigh_solve(fh, fb, self.p, mu=self.mu)
        with patch("src.cbg.spectral_grad.jvp", return_value=(torch.tensor(float("nan")), torch.tensor(0.))):
            with self.assertRaises(FloatingPointError):
                direct_spectral_gradient(fh, fb, self.p, r.direction, mu=self.mu, center=r.center)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            with self.assertRaisesRegex(ValueError, "autocast"):
                direct_spectral_gradient(fh, fb, self.p, r.direction, mu=self.mu)

    def test_tiny_llama_eager_with_zero_initialized_lora_factors(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        from torch import nn
        from src.cbg.spectral_grad import exact_spectral_gradient, make_scalarized_functional
        torch.manual_seed(7)
        config = LlamaConfig(vocab_size=17, hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=32, use_cache=False, attention_dropout=0.)
        config._attn_implementation = "eager"
        model = LlamaForCausalLM(config).float().eval().requires_grad_(False)
        class LoRA(nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base
                self.lora_A = nn.Parameter(torch.randn(2, base.in_features) * .1)
                self.lora_B = nn.Parameter(torch.zeros(base.out_features, 2))
            def forward(self, x):
                return self.base(x) + nn.functional.linear(nn.functional.linear(x, self.lora_A), self.lora_B)
        attn = model.model.layers[0].self_attn
        attn.q_proj, attn.v_proj = LoRA(attn.q_proj), LoRA(attn.v_proj)
        model.eval()
        p = OrderedDict((n, p) for n, p in model.named_parameters() if p.requires_grad)
        def batch(ids):
            return dict(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, 5, dtype=torch.long),
                        response_mask=torch.tensor([[False, False, False, True, True]]))
        harmful, safe, benign = batch([1,2,3,4,5]), batch([1,2,3,6,7]), batch([1,8,9,10,11])
        fh = make_safety_functional(model, harmful, safe, chunk_size=1)
        fb = make_benign_functional(model, benign, chunk_size=1, token_blocks=2)
        r = dual_rayleigh_solve(fh, fb, p, mu=.1)
        g = direct_spectral_gradient(fh, fb, p, r.direction, mu=.1, center=r.center)
        exact = exact_spectral_gradient(make_scalarized_functional(fh, fb, r.center), p, r.direction, r.center.b)
        self.assertLess(float(tree_norm(tree_sub(g.gradient, exact))/tree_norm(exact)), 1e-4)
        fd = centered_spectral_gradient(fh, fb, p, r.direction, mu=.1, center=r.center, tau_fd=1e-3)
        self.assertGreater(float(tree_cosine_similarity(g.gradient, fd.gradient)), .99)
        self.assertLess(float(tree_norm(tree_sub(g.gradient, fd.gradient))/tree_norm(g.gradient)), .01)

    def test_full_safety_range_closes_bound_and_warm_restart(self):
        torch.manual_seed(15)
        a, c = torch.randn(3, 5, dtype=torch.float64), torch.randn(2, 5, dtype=torch.float64)
        p = OrderedDict(w=torch.randn(5, dtype=torch.float64, requires_grad=True))
        fh, fb = lambda p: a@p["w"], lambda p: c@p["w"]
        mu = .01
        b = c.T@c/2 + mu*torch.eye(5, dtype=torch.float64)
        v = torch.linalg.solve(b, a.T)
        l = torch.linalg.cholesky(v.T@b@v)
        v = torch.linalg.solve_triangular(l, v.T, upper=False).T
        basis = tuple(OrderedDict(w=v[:, i]) for i in range(3))
        enclosure = compute_complement_enclosure(fh, fb, p, basis, mu=mu)
        self.assertLess(float(enclosure.upper-enclosure.lower), 1e-8)
        rng = torch.random.get_rng_state().clone()
        r = bounded_rayleigh_solve(fh, fb, p, mu=mu, warm_basis=basis, initial_size=1,
            max_size=3, max_enrichments=0)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(len(r.warm_basis), 3)
        self.assertFalse(r.diagnostics.dual_fallback)

    def test_repeated_leading_value_keeps_one_active_gradient(self):
        p = OrderedDict(w=torch.ones(2, dtype=torch.float64, requires_grad=True))
        fh, fb = lambda p: p["w"]**2, lambda p: p["w"] * 0
        r = dual_rayleigh_solve(fh, fb, p, mu=1., tolerance=1e-8)
        self.assertEqual(float(r.diagnostics.eigengap), 0.)
        g = direct_spectral_gradient(fh, fb, p, r.direction, mu=1., center=r.center)
        # lambda_max = max(2*w_0^2, 2*w_1^2); one active derivative is valid.
        expected = 4 * r.direction["w"].square()
        torch.testing.assert_close(g.gradient["w"], expected)

    def test_zero_safety_and_failed_fallback_do_not_differentiate(self):
        f = optimized_tests.OptimizedCbgTest()
        f.setUp()
        trainer, metrics = f.trainer_fixture(cbg_version="v5", geometry_solver="bounded_ritz",
            spectral_derivative_backend="direct_rof", benign_token_blocks=2, ritz_max_enrichments=0)
        with patch("src.core.cbg_trainer.direct_spectral_gradient", side_effect=AssertionError("zero safety AD")):
            trainer.training_step(f.model, dict(harmful=f.safe, safe=f.safe, benign=f.benign))
        self.assertEqual(metrics["cbg/zero_safety"], 1.)
        self.assertEqual(trainer.prev_basis, ())
        f.model.zero_grad()
        with patch("src.cbg.bounded.dual_rayleigh_solve", side_effect=FloatingPointError("oracle failure")):
            with self.assertRaisesRegex(FloatingPointError, "oracle failure"):
                trainer.training_step(f.model, dict(harmful=f.harmful, safe=f.safe, benign=f.benign))
        self.assertTrue(all(p.grad is None for p in f.params.values()))

    def test_optimizer_update_metric_measures_applied_step(self):
        from src.core.cbg_trainer import _OptimizerStepMetrics
        f = optimized_tests.OptimizedCbgTest()
        f.setUp()
        trainer, metrics = f.trainer_fixture()
        callback = _OptimizerStepMetrics(trainer)
        for p in f.params.values():
            p.grad = torch.ones_like(p)
        callback.on_pre_optimizer_step(None, None, None, model=f.model)
        trainer.optimizer.step()
        callback.on_optimizer_step(None, None, None, model=f.model)
        expected = .01 * sum(p.numel() for p in f.params.values())**.5
        self.assertAlmostEqual(float(metrics["train/optimizer_step_norm"]), expected)
        self.assertAlmostEqual(metrics["train/optimizer_lr_used"], .01)
        self.assertIsNone(callback.before)

    def test_direct_response_graphs_are_released(self):
        import gc
        fh, fb = self.maps()
        r = dual_rayleigh_solve(fh, fb, self.p, mu=self.mu)
        count = dict(live=0, peak=0)
        class Saved:
            def __init__(self, tensor):
                self.tensor = tensor
                count["live"] += 1
                count["peak"] = max(count["peak"], count["live"])
            def __del__(self):
                count["live"] -= 1
        with torch.autograd.graph.saved_tensors_hooks(Saved, lambda box: box.tensor):
            direct_spectral_gradient(fh, fb, self.p, r.direction, mu=self.mu, center=r.center)
        gc.collect()
        self.assertGreater(count["peak"], 0)
        self.assertEqual(count["live"], 0)

    def test_manifest_attack_selection_checks_hash_and_overlap(self):
        import json, hashlib
        from src.data_processing.cbg_datasets import load_manifest_attack_records
        from test_cbg_datasets import unsafe_record
        with tempfile.TemporaryDirectory() as directory:
            source, manifest = Path(directory)/"data.json", Path(directory)/"manifest.json"
            records = [unsafe_record(i) for i in range(5)]
            source.write_text(json.dumps(records), encoding="utf-8")
            info = dict(safety_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                        defense_indices=[0, 2], attack_indices=[4, 1, 3])
            manifest.write_text(json.dumps(info), encoding="utf-8")
            result = load_manifest_attack_records(source, manifest, attack_size=2)
            self.assertEqual(result, [records[4], records[1]])
            with self.assertRaisesRegex(ValueError, "exceeds"):
                load_manifest_attack_records(source, manifest, attack_size=4)
            info["attack_indices"] = [0, 1]
            manifest.write_text(json.dumps(info), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_manifest_attack_records(source, manifest, attack_size=1)
            info["safety_source_sha256"] = "wrong"
            manifest.write_text(json.dumps(info), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash"):
                load_manifest_attack_records(source, manifest, attack_size=1)

    def test_v5_config_and_checkpoint_basis(self):
        from src.utils.config import load_yaml_config, MyParser
        config = load_yaml_config(Path(__file__).resolve().parents[1]/"config/cbg_v5_smoke_1b.yaml")
        (args,) = MyParser((CbgArguments,)).parse_args_into_dataclasses(args=[], default_values=config["CbgArguments"])
        self.assertEqual(args.spectral_derivative_backend, "direct_rof")
        for options in (dict(cbg_version="v5"), dict(cbg_version="v5", geometry_solver="dual")):
            with self.assertRaises(ValueError):
                CbgArguments(**options)
        f = optimized_tests.OptimizedCbgTest()
        f.setUp()
        trainer, _ = f.trainer_fixture(cbg_version="v5", geometry_solver="bounded_ritz",
            spectral_derivative_backend="direct_rof", benign_token_blocks=2, ritz_max_enrichments=0)
        trainer.training_step(f.model, dict(harmful=f.harmful, safe=f.safe, benign=f.benign))
        original = trainer.prev_basis
        trainer.args.should_save = True
        trainer._admissible_names = tuple(f.params)
        trainer._beta_candidate_sum, trainer._beta_candidate_count, trainer._calibrated_beta = 0., 0, None
        with tempfile.TemporaryDirectory() as directory:
            trainer._save_cbg_artifacts(Path(directory))
            trainer.prev_basis = ()
            trainer._load_cbg_state(Path(directory), f.model)
            trainer.args.gradient_accumulation_steps = 2
            with self.assertRaisesRegex(ValueError, "batch, accumulation or precision"):
                trainer._load_cbg_state(Path(directory), f.model)
            trainer.args.gradient_accumulation_steps = 1
        self.assertEqual(len(original), len(trainer.prev_basis))
        for x, y in zip(original, trainer.prev_basis):
            for name in x:
                torch.testing.assert_close(x[name], y[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
