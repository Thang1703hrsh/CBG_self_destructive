"""Equivalence, resource bounds, and solver tests for the optimized CBG path."""
import os
os.environ.setdefault("USE_TF", "0")

import copy
import unittest
from collections import OrderedDict
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from test_cbg_spectral_grad import TinyCausalLM, sequence_batch
from src.cbg.functionals import (
    benign_kl_from_logits, benign_preservation_loss, make_benign_functional,
    make_safety_functional, safe_alignment_loss, select_response_logits, sequence_chunks,
)
from src.cbg.geometry import metric_vector_product
from src.cbg.rayleigh import calibrate_relative_damping, normalize_b_metric, warm_started_rayleigh_ascent
from src.cbg.spectral_grad import (
    centered_spectral_gradient, compute_spectral_center, exact_spectral_gradient,
    make_scalarized_functional, parameter_gradient, streamed_loss_gradient,
    _gradient_at_perturbation,
)
from src.cbg.tree_ops import (
    select_admissible_parameters, tree_add, tree_cosine_similarity, tree_dot,
    tree_norm, tree_random_like, tree_scale, tree_sub,
)
from src.core.cbg_trainer import CbgArguments, CbgTrainer, _temporary_eval


def legacy_search(fh, fb, parameters, direction, mu, steps, lr):
    """Independent literal pre-optimization algorithm, recomputing normalization."""
    for _ in range(steps):
        gh = metric_vector_product(fh, parameters, direction, detach_result=True)
        gb = metric_vector_product(fb, parameters, direction, detach_result=True)
        bvec = tree_add(gb, direction, alpha=mu)
        denom = tree_dot(direction, bvec)
        rho = tree_dot(direction, gh) / denom
        grad = tree_scale(tree_sub(gh, tree_scale(bvec, rho)), 2 / denom)
        direction = normalize_b_metric(fb, parameters, tree_add(direction, grad, alpha=lr), mu)
    return direction


class OptimizedCbgTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.model = TinyCausalLM().double().eval()
        self.params = select_admissible_parameters(self.model, parameter_names=("q_proj.weight", "v_proj.weight"))
        self.harmful = sequence_batch([[0,1,2,3,4], [1,2,3,4,5], [2,3,4,5,6]])
        self.safe = sequence_batch([[0,1,2,5,6], [1,2,3,6,0], [2,3,4,0,1]])
        self.benign = sequence_batch([[3,2,1,0,6], [4,3,2,1,5], [5,4,3,2,4]])
        self.fh = make_safety_functional(self.model, self.harmful, self.safe)
        self.fb = make_benign_functional(self.model, self.benign)
        self.u = tree_random_like(self.params, generator=torch.Generator().manual_seed(21))

    def assert_tree_close(self, a, b, rtol=1e-9, atol=1e-11):
        self.assertEqual(tuple(a), tuple(b))
        for name in a:
            torch.testing.assert_close(a[name], b[name], rtol=rtol, atol=atol)

    def test_cached_normalization_matches_literal_old_algorithm(self):
        for steps in (1, 2, 5):
            expected = legacy_search(self.fh, self.fb, self.params, self.u, .2, steps, .01)
            actual = warm_started_rayleigh_ascent(self.fh, self.fb, self.params,
                mu=.2, inner_steps=steps, inner_lr=.01, warm_start=self.u)
            self.assert_tree_close(actual.direction, expected)
            self.assertEqual(actual.diagnostics.metric_evaluations, 2 * (steps + 1))
            for name in ("gh_direction", "gb_direction", "b_direction"):
                for value in getattr(actual.center, name).values():
                    self.assertIsNone(value.grad_fn)
                    self.assertFalse(value.requires_grad)

    def test_outer_center_reuse_needs_no_additional_metric_calls(self):
        result = warm_started_rayleigh_ascent(self.fh, self.fb, self.params, mu=.2, warm_start=self.u)
        with patch("src.cbg.spectral_grad.metric_product_with_jvp", side_effect=AssertionError("recomputed metric")):
            cached = centered_spectral_gradient(self.fh, self.fb, self.params, result.direction,
                                               mu=.2, center=result.center)
        fresh = centered_spectral_gradient(self.fh, self.fb, self.params, result.direction, mu=.2)
        self.assert_tree_close(cached.gradient, fresh.gradient, rtol=1e-7, atol=1e-9)
        self.assertIsNone(cached.grad_plus)
        self.assertIsNone(cached.grad_minus)
        with torch.no_grad():
            next(iter(self.params.values())).add_(.01)
        with self.assertRaisesRegex(ValueError, "stale"):
            centered_spectral_gradient(self.fh, self.fb, self.params, result.direction, mu=.2, center=result.center)

    def test_cache_rejects_changed_batch_or_model_mode(self):
        center = compute_spectral_center(self.fh, self.fb, self.params, self.u, mu=.2)
        self.safe["input_ids"][0, -1] = 2
        with self.assertRaisesRegex(ValueError, "stale"):
            centered_spectral_gradient(self.fh, self.fb, self.params, self.u, mu=.2, center=center)
        center = compute_spectral_center(self.fh, self.fb, self.params, self.u, mu=.2)
        self.model.train()
        with self.assertRaisesRegex(ValueError, "stale"):
            centered_spectral_gradient(self.fh, self.fb, self.params, self.u, mu=.2, center=center)

    def test_streamed_phi_reduces_peak_saved_tensor_bytes(self):
        center = compute_spectral_center(self.fh, self.fb, self.params, self.u, mu=.2)
        phi = make_scalarized_functional(self.fh, self.fb, center)
        def measure(functional):
            counts = dict(live=0, peak=0)
            class Saved:
                def __init__(self, tensor):
                    self.tensor = tensor
                    self.size = tensor.numel() * tensor.element_size()
                    counts["live"] += self.size
                    counts["peak"] = max(counts["peak"], counts["live"])
                def __del__(self):
                    counts["live"] -= self.size
            with torch.autograd.graph.saved_tensors_hooks(Saved, lambda box: box.tensor):
                gradient = _gradient_at_perturbation(functional, self.params, self.u, 1e-4)
            return gradient, counts["peak"]
        mono, mono_peak = measure(lambda p: phi(p))
        stream, stream_peak = measure(phi)
        self.assert_tree_close(mono, stream)
        self.assertLess(stream_peak, mono_peak * .6)

    def test_cached_center_rejects_changed_autocast_precision(self):
        center = compute_spectral_center(self.fh, self.fb, self.params, self.u, mu=.2)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            with self.assertRaisesRegex(ValueError, "stale"):
                centered_spectral_gradient(self.fh, self.fb, self.params, self.u, mu=.2, center=center)

    def test_uneven_chunks_preserve_safety_cross_terms_and_global_weights(self):
        chunked = make_safety_functional(self.model, self.harmful, self.safe, chunk_size=2)
        torch.testing.assert_close(chunked(self.params), self.fh(self.params))
        expected = metric_vector_product(self.fh, self.params, self.u, detach_result=True)
        actual = metric_vector_product(chunked, self.params, self.u, detach_result=True)
        self.assert_tree_close(actual, expected)
        # Compare with the monolithic margin, independently of the block protocol.
        monolithic = lambda p: self.fh(p)
        self.assert_tree_close(actual, metric_vector_product(monolithic, self.params, self.u, detach_result=True))

    def test_chunked_fd_preserves_hvp_and_bounds_forward_batch(self):
        seen = []
        hook = self.model.register_forward_pre_hook(lambda module, args, kwargs: seen.append(kwargs["input_ids"].shape[0]), with_kwargs=True)
        try:
            fh = make_safety_functional(self.model, self.harmful, self.safe, chunk_size=1)
            fb = make_benign_functional(self.model, self.benign, chunk_size=1)
            fd = centered_spectral_gradient(fh, fb, self.params, self.u, mu=.2)
        finally:
            hook.remove()
        self.assertTrue(seen)
        self.assertLessEqual(max(seen), 1)
        exact = exact_spectral_gradient(make_scalarized_functional(fh, fb, fd.center), self.params, self.u, fd.center.b)
        self.assertGreater(float(tree_cosine_similarity(fd.gradient, exact)), .9999)
        self.assertLess(float(tree_norm(tree_sub(fd.gradient, exact)) / tree_norm(exact)), 1e-5)
        full = centered_spectral_gradient(self.fh, self.fb, self.params, self.u, mu=.2)
        self.assert_tree_close(fd.gradient, full.gradient, rtol=1e-7, atol=1e-9)

    def test_damping_calibration_uses_jvp_only_and_same_quadratic_forms(self):
        generator = torch.Generator().manual_seed(10)
        costs = []
        for _ in range(5):
            u = tree_random_like(self.params, generator=generator)
            u = tree_scale(u, tree_norm(u).reciprocal())
            gu = metric_vector_product(self.fb, self.params, u, detach_result=True)
            costs.append(tree_dot(u, gu))
        with patch("src.cbg.geometry.vjp", side_effect=AssertionError("calibration ran VJP")):
            mu = calibrate_relative_damping(self.fb, self.params, num_directions=5,
                                            generator=torch.Generator().manual_seed(10))
        torch.testing.assert_close(mu, (.05 * torch.stack(costs).median()).clamp_min(1e-8))

    def test_selected_reference_and_streamed_losses_match_full_batch_gradients(self):
        with torch.no_grad():
            reference = self.model(self.benign["input_ids"])[0].detach() + torch.randn(3, 5, 7, dtype=torch.float64) * .1
        selected = select_response_logits(reference, self.benign)
        full_loss = benign_preservation_loss(self.model, self.params, self.benign, reference)
        full_gradient = parameter_gradient(full_loss, self.params)
        selected_loss = benign_preservation_loss(self.model, self.params, self.benign, selected)
        torch.testing.assert_close(full_loss, selected_loss)
        parts = list(sequence_chunks(self.benign, 2))
        def losses():
            offset = 0
            for fraction, batch in parts:
                n = batch["input_ids"].shape[0]
                ref = select_response_logits(reference[offset:offset+n], batch)
                yield fraction, benign_preservation_loss(self.model, self.params, batch, ref)
                offset += n
        loss, gradient = streamed_loss_gradient(losses(), self.params)
        torch.testing.assert_close(loss, full_loss)
        self.assert_tree_close(gradient, full_gradient)

    def test_adaptive_search_backtracks_and_reports_convergence(self):
        # Linear maps give constant, explicit diagonal metrics for a controlled solver test.
        params = OrderedDict(p=torch.tensor([.2, -.3], dtype=torch.float64, requires_grad=True))
        fh = lambda p: p["p"] * torch.tensor([2., 1.], dtype=torch.float64)
        fb = lambda p: p["p"]
        u = OrderedDict(p=torch.tensor([1., 1.], dtype=torch.float64))
        result = warm_started_rayleigh_ascent(fh, fb, params, mu=.5, inner_steps=2,
            max_steps=100, inner_lr=10., max_backtracks=12, warm_start=u,
            residual_tolerance=1e-5, rayleigh_tolerance=1e-7)
        history = torch.stack(result.diagnostics.rayleigh_values)
        self.assertTrue(bool(torch.all(history[1:] >= history[:-1] - 1e-12)))
        self.assertGreater(result.diagnostics.rejected_steps, 0)
        self.assertTrue(result.diagnostics.converged)
        self.assertAlmostEqual(float(result.diagnostics.rho_hat), 2., places=8)
        self.assertLess(result.diagnostics.inner_steps, 100)

    def test_fp16_dot_promotes_before_multiply(self):
        vector = OrderedDict(p=torch.tensor([400., 400.], dtype=torch.float16))
        self.assertEqual(float(tree_dot(vector, vector)), 320000.)

    def test_eval_context_restores_mixed_module_modes_on_exception(self):
        self.model.train()
        self.model.q_proj.eval()
        modes = [m.training for m in self.model.modules()]
        with self.assertRaises(RuntimeError), _temporary_eval(self.model):
            self.assertFalse(any(m.training for m in self.model.modules()))
            raise RuntimeError("injected")
        self.assertEqual([m.training for m in self.model.modules()], modes)

    def trainer_fixture(self, **overrides):
        # Real CbgTrainer.training_step with only the unavailable HF runtime shell stubbed.
        trainer = CbgTrainer.__new__(CbgTrainer)
        trainer.cbg_args = CbgArguments(reference_mode="separate_model", damping_mode="absolute",
                                        mu=.2, functional_chunk_size=1, preservation_chunk_size=2, **overrides)
        trainer.reference_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        trainer.accelerator = SimpleNamespace(unwrap_model=lambda m: m, backward=lambda loss: loss.backward())
        trainer.args = SimpleNamespace(device=torch.device("cpu"), bf16=False, gradient_accumulation_steps=1)
        trainer.state = SimpleNamespace(global_step=0)
        trainer.prev_u, trainer.calibrated_mu = self.u, .2
        trainer.optimizer = torch.optim.SGD(self.params.values(), lr=.01)
        trainer._prepare_inputs = lambda x: x
        trainer._admissible_parameters = lambda m: self.params
        trainer.compute_loss_context_manager = nullcontext
        metrics = {}
        trainer._record_metrics = lambda values: metrics.update(values)
        return trainer, metrics

    def test_full_trainer_step_injects_gradient_and_keeps_parameters_exact(self):
        trainer, metrics = self.trainer_fixture()
        snapshots = {n: (p.detach().clone(), p.data_ptr()) for n,p in self.model.named_parameters()}
        loss = trainer.training_step(self.model, {"harmful": self.harmful, "safe": self.safe, "benign": self.benign})
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.params.values()))
        self.assertEqual(metrics["cbg/metric_evaluations"], 6)
        for n,p in self.model.named_parameters():
            self.assertTrue(torch.equal(p, snapshots[n][0]))
            self.assertEqual(p.data_ptr(), snapshots[n][1])
        self.assertTrue(all(p.grad is None for p in trainer.reference_model.parameters()))

    def test_beta_zero_training_step_never_calls_geometry(self):
        trainer, metrics = self.trainer_fixture(beta=0.)
        _, ga, _, gb = trainer._preservation_gradients(self.model, self.params, self.safe, self.benign)
        expected = tree_add(ga, gb, alpha=trainer.cbg_args.alpha)
        with patch("src.core.cbg_trainer.warm_started_rayleigh_ascent", side_effect=AssertionError("geometry called")), \
             patch("src.core.cbg_trainer.centered_spectral_gradient", side_effect=AssertionError("FD called")), \
             patch.object(trainer, "_resolve_mu", side_effect=AssertionError("damping called")):
            trainer.training_step(self.model, {"harmful": self.harmful, "safe": self.safe, "benign": self.benign})
        self.assert_tree_close(OrderedDict((n, p.grad) for n,p in self.params.items()), expected)
        self.assertEqual(metrics["cbg/enabled"], 0.)
        self.assertEqual(metrics["train/grad_norm_beta_cbg"], 0.)

    def test_strict_nonconvergence_fails_before_outer_or_backward(self):
        trainer, _ = self.trainer_fixture(inner_fail_on_nonconvergence=True)
        with patch("src.core.cbg_trainer.centered_spectral_gradient", side_effect=AssertionError("FD called")):
            with self.assertRaisesRegex(RuntimeError, "relative_residual="):
                trainer.training_step(self.model, {"harmful": self.harmful, "safe": self.safe, "benign": self.benign})
        self.assertTrue(all(p.grad is None for p in self.params.values()))

    def test_gradient_ratio_beta_is_bounded_frozen_and_degenerate_safe(self):
        trainer, _ = self.trainer_fixture(beta_mode="grad_ratio", beta_max=.1, beta_calibration_steps=2)
        trainer._beta_candidate_sum, trainer._beta_candidate_count, trainer._calibrated_beta = 0., 0, None
        zero = tree_scale(self.u, 0.)
        self.assertEqual(trainer._effective_beta(self.u, zero), trainer.cbg_args.beta)
        self.assertEqual(trainer._beta_candidate_count, 0)
        self.assertEqual(trainer._effective_beta(self.u, self.u), .1)
        trainer.state.global_step = 2
        self.assertEqual(trainer._effective_beta(self.u, tree_scale(self.u, 100.)), .1)
        self.assertEqual(trainer._beta_candidate_count, 1)


if __name__ == "__main__":
    unittest.main()
