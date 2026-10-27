"""The scalable step (src/cbg/step_v2.py) must reproduce the verified spec step."""
import os
os.environ.setdefault("USE_TF", "0")
import copy
import itertools
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from test_cbg_spec import fixture, fp64_norm, fp64_attention, jacobian_rows
from test_cbg_method_v2 import objective
from src.cbg.functionals import select_response_logits
from src.cbg.spec import joint_batch, measurements, _forward, spec_training_gradient
from src.cbg.step_v2 import (StepConfig, cbg_step, make_chunks, make_totals, measure_geometry,
                             place_layers, curvature_estimates, curvature_gradient, plain_gradient)
from src.cbg.spec import measure_geometry as spec_measure_geometry
from src.data_processing.cbg_datasets import tokenize_paired_responses, tokenize_prompt_response


def step_config(spec, **changes):
    values = dict(xi=spec.xi, kappa_mu=spec.kappa_mu, r=spec.r, alpha=spec.alpha, beta=spec.beta,
                  k=spec.k, num_sign_draws=spec.num_sign_draws, logprob_chunk_size=0)
    values.update(changes)
    return StepConfig(**values)


class StepV2Test(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        for target, fn in (("LlamaRMSNorm.forward", fp64_norm), ("eager_attention_forward", fp64_attention)):
            p = patch("transformers.models.llama.modeling_llama." + target, fn)
            p.start()
            self.addCleanup(p.stop)
        self.model, self.data, self.cfg = fixture()
        # Unequal lengths exercise per-micro-batch width trimming and padding.
        self.data["benign"]["attention_mask"][0, -1] = 0
        self.data["benign"]["response_mask"][0, -1] = False
        self.samples = torch.tensor([1, 2, 3, 4, 5])
        with torch.no_grad():
            b = self.data["benign"]
            self.ref = select_response_logits(
                self.model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits, b) + .05

    def reference(self, draws):
        return spec_training_gradient(self.model, self.data, replace(self.cfg, num_sign_draws=draws),
                                      reference_logits=self.ref, generator=torch.Generator().manual_seed(7),
                                      fixed_samples=self.samples)

    def run_step(self, **changes):
        for p in self.model.parameters():
            p.grad = None
        diagnostics = cbg_step(self.model, self.data, step_config(self.cfg, **changes), reference=self.ref,
                               sign_generator=torch.Generator().manual_seed(7), fixed_samples=self.samples)
        return {n: p.grad.clone() for n, p in self.model.named_parameters()}, diagnostics

    def assert_same(self, actual, expected, rtol=1e-8):
        for name, value in expected.items():
            torch.testing.assert_close(actual[name], value, rtol=rtol, atol=1e-11, msg=name)

    def test_hvp_sign_average_matches_full_differentiated_paper_objective(self):
        # Independent oracle: differentiate through explicit toy Jacobian rows,
        # relative coordinates, trace damping, and both spectral sums.
        params = dict(self.model.named_parameters())
        joint = joint_batch(self.data)
        values = measurements(_forward(self.model, joint), joint, samples=self.samples,
                              reference_logits=self.ref)
        jac = jacobian_rows(values.values, params, create_graph=True)
        from src.cbg.spec_hooks import spec_linear_modules
        kernel = sum((jac[name].flatten(1) * (module.weight.square() + self.cfg.xi**2).flatten())
                     @ jac[name].flatten(1).T for name, module in spec_linear_modules(self.model).items())
        loss = values.alignment + self.cfg.alpha * values.benign_kl + self.cfg.beta * objective(kernel, 2, 2, "differentiated")
        exact = torch.autograd.grad(loss, tuple(params.values()))
        signs = torch.tensor(list(itertools.product((-1., 1.), repeat=5)))
        cbg_step(self.model, self.data, step_config(self.cfg, curvature="hvp", num_sign_draws=len(signs),
                                                  microbatch_tokens=8), reference=self.ref,
                 fixed_samples=self.samples, signs=signs)
        self.assert_same({n: p.grad for n, p in params.items()}, dict(zip(params, exact)), rtol=1e-8)

    def test_hvp_preserves_frozen_U_and_matches_fd_on_trainable_subset(self):
        self.model.model.layers[0].self_attn.q_proj.weight.requires_grad_(False)
        expected = self.reference(1).gradient
        before = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        cbg_step(self.model, self.data, step_config(self.cfg, curvature="hvp", microbatch_tokens=8),
                 reference=self.ref, fixed_samples=self.samples, sign_generator=torch.Generator().manual_seed(7))
        actual = {n: p.grad for n, p in self.model.named_parameters() if p.requires_grad}
        self.assertLess(self.relative_gap(actual, expected), 1e-6)
        for n, p in self.model.named_parameters():
            torch.testing.assert_close(p, before[n], rtol=0, atol=0)
            if n not in expected:
                self.assertFalse(p.requires_grad)
                self.assertIsNone(p.grad)

    def test_hvp_trains_only_head_while_geometry_uses_all_frozen_block_weights(self):
        self.model.requires_grad_(False)
        self.model.lm_head.requires_grad_(True)
        expected = self.reference(1).gradient
        cbg_step(self.model, self.data, step_config(self.cfg, curvature="hvp", microbatch_tokens=8),
                 reference=self.ref, fixed_samples=self.samples, sign_generator=torch.Generator().manual_seed(7))
        actual = {n: p.grad for n, p in self.model.named_parameters() if p.requires_grad}
        self.assertLess(self.relative_gap(actual, expected), 1e-6)
        self.assertEqual(set(actual), {"lm_head.weight"})
        self.assertTrue(all(p.grad is None for n, p in self.model.named_parameters() if n not in actual))

    def test_fd_restores_weights_and_modes_after_endpoint_failure(self):
        from src.cbg import step_v2
        original = step_v2._logits
        for restore in ("copy", "subtract"):
            for endpoint in (1, 2):
                before = {n: p.detach().clone() for n, p in self.model.named_parameters()}
                self.model.train()
                calls = 0
                def fail(model, batch):
                    nonlocal calls
                    calls += 1
                    # Two center chunks, then two chunks per endpoint.
                    if calls == 3 + 2 * (endpoint - 1):
                        raise RuntimeError("injected endpoint failure")
                    return original(model, batch)
                with patch.object(step_v2, "_logits", side_effect=fail):
                    with self.assertRaisesRegex(RuntimeError, "injected"):
                        self.run_step(restore=restore)
                self.assertTrue(self.model.training)
                for n, p in self.model.named_parameters():
                    torch.testing.assert_close(p, before[n], rtol=0, atol=0 if restore == "copy" else 1e-14)
                self.model.eval()

    def test_zero_shared_direction_keeps_base_and_avoids_division_by_zero(self):
        from src.cbg import step_v2
        expected = spec_training_gradient(self.model, self.data, replace(self.cfg, beta=0.),
                                          reference_logits=self.ref).gradient
        for curvature in ("fd", "hvp"):
            with patch.object(step_v2, "eigen_sign_scalarization",
                              side_effect=lambda q, signs: (torch.zeros_like(signs, dtype=torch.float64),
                                                             torch.zeros_like(signs, dtype=torch.float64))):
                actual, stats = self.run_step(curvature=curvature, detach_lambda=True, num_sign_draws=2)
            self.assert_same(actual, expected, rtol=1e-12)
            self.assertEqual(stats["epsilon"], 0.)

    def test_step_configuration_rejects_invalid_execution_options(self):
        for options in (dict(gram_max_bytes=0), dict(logprob_chunk_size=-1), dict(kl_direction="other"),
                        dict(block_path=""), dict(detach_lambda="false"), dict(deterministic=1)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                step_config(self.cfg, **options)

    def test_matches_spec_step_for_draws_budgets_and_restore_modes(self):
        before = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        for draws in (1, 2):
            expected = self.reference(draws).gradient
            for budget in (0, 4, 8):
                for restore in ("copy", "subtract"):
                    with self.subTest(draws=draws, budget=budget, restore=restore):
                        actual, diag = self.run_step(num_sign_draws=draws, microbatch_tokens=budget,
                                                     restore=restore)
                        self.assert_same(actual, expected)
                        self.assertEqual(diag["geometry_active"], 1.)
                        for n, p in self.model.named_parameters():
                            if restore == "copy":
                                self.assertTrue(torch.equal(p, before[n]), n)
                            else:
                                torch.testing.assert_close(p, before[n], rtol=0, atol=1e-14)
                                with torch.no_grad():
                                    p.copy_(before[n])

    def test_microbatches_split_rows_and_trim_padding(self):
        joint = joint_batch(self.data)
        chunks = make_chunks(joint, 8)
        self.assertEqual([(c.batch.m, c.batch.n) for c in chunks], [(1, 0), (1, 0), (0, 2), (0, 1)])
        self.assertEqual(chunks[2].batch.input_ids.shape[1], 4)
        single = make_chunks(joint, 0)
        self.assertEqual([(c.batch.m, c.batch.n) for c in single], [(2, 0), (0, 3)])
        totals = make_totals(joint)
        full = measurements(_forward(self.model, joint), joint, samples=self.samples, reference_logits=self.ref)
        values, alignment, kl = [], 0., 0.
        for chunk in chunks:
            part = measurements(_forward(self.model, chunk.batch), chunk.batch, totals=totals,
                                samples=self.samples[chunk.benign[0]:chunk.benign[1]],
                                reference_logits=self.ref[chunk.benign[0]:chunk.benign[1]])
            values.append((chunk.examples, part.values))
            alignment, kl = alignment + part.alignment, kl + part.benign_kl
        assembled = torch.zeros_like(full.values)
        for examples, value in values:
            assembled[examples] = value
        torch.testing.assert_close(assembled, full.values, rtol=1e-12, atol=1e-14)
        torch.testing.assert_close(alignment, full.alignment, rtol=1e-12, atol=1e-14)
        torch.testing.assert_close(kl, full.benign_kl, rtol=1e-12, atol=1e-14)

    def test_beta_zero_and_degenerate_geometry_are_plain_gradients(self):
        expected = spec_training_gradient(self.model, self.data, replace(self.cfg, beta=0.),
                                          reference_logits=self.ref, fixed_samples=self.samples).gradient
        actual, diag = self.run_step(beta=0., microbatch_tokens=4)
        self.assert_same(actual, expected, rtol=1e-12)
        self.assertEqual(diag["geometry_active"], 0.)
        self.data["harmful"] = copy.deepcopy(self.data["safe"])
        expected = spec_training_gradient(self.model, self.data, replace(self.cfg, beta=0.),
                                          reference_logits=self.ref, fixed_samples=self.samples).gradient
        actual, diag = self.run_step(microbatch_tokens=4)
        self.assertEqual(diag["geometry_skipped"], 1.)
        self.assert_same(actual, expected, rtol=1e-12)

    def test_geometry_matches_and_layer_split_placement_runs(self):
        expected = spec_measure_geometry(self.model, self.data, config=self.cfg, samples=self.samples)
        actual, _ = measure_geometry(self.model, self.data, step_config(self.cfg, microbatch_tokens=8),
                                     samples=self.samples)
        for key in ("log_eta_k", "rho_mu", "tr_Kbb", "c_mu", "nu_1k"):
            self.assertAlmostEqual(actual[key], float(expected[key]), delta=1e-10 * max(1, abs(actual[key])))
        expected_grad = self.reference(1).gradient
        assignment = place_layers(self.model, ["cpu", "cpu"])
        self.assertEqual(len(assignment), 2)
        actual_grad, _ = self.run_step(microbatch_tokens=8)
        self.assert_same(actual_grad, expected_grad)

    def test_geometry_only_plus_plain_gradient_is_the_full_step(self):
        # lazy_mode reuse splits the step: beta * grad log eta (include_base=False) + plain base gradient.
        full, _ = self.run_step(microbatch_tokens=8)
        for p in self.model.parameters():
            p.grad = None
        diag = cbg_step(self.model, self.data, step_config(self.cfg, microbatch_tokens=8), reference=None,
                        sign_generator=torch.Generator().manual_seed(7), fixed_samples=self.samples,
                        include_base=False)
        self.assertEqual(diag["geometry_active"], 1.)
        geometry = {n: p.grad.clone() for n, p in self.model.named_parameters()}
        plain_gradient(self.model, joint_batch(self.data), step_config(self.cfg, microbatch_tokens=8),
                       reference=self.ref)
        # The full step averages the base gradient over theta +- eps w: equal up to O(eps^2) (2e-8 here).
        gap = sum(float((p.grad - full[n]).norm()) ** 2 for n, p in self.model.named_parameters()) ** .5
        self.assertLess(gap / sum(float(v.norm()) ** 2 for v in full.values()) ** .5, 1e-6)
        self.assertGreater(sum(float(g.norm()) for g in geometry.values()), 0.)
        self.data["harmful"] = copy.deepcopy(self.data["safe"])   # degenerate: nothing at all
        for p in self.model.parameters():
            p.grad = None
        diag = cbg_step(self.model, self.data, step_config(self.cfg), include_base=False,
                        fixed_samples=self.samples)
        self.assertEqual(diag["geometry_skipped"], 1.)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def relative_gap(self, actual, expected):
        gap = sum(float((actual[n] - expected[n]).norm()) ** 2 for n in expected) ** .5
        return gap / sum(float(v.norm()) ** 2 for v in expected.values()) ** .5

    def test_hvp_matches_fp64_spec_step_and_finite_difference_converges_to_it(self):
        before = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        hvp, diag = self.run_step(curvature="hvp", microbatch_tokens=8)
        self.assertEqual(diag["geometry_active"], 1.)
        self.assertEqual(diag["curvature_hvp"], 1.)
        for n, p in self.model.named_parameters():   # nothing is perturbed or restored
            self.assertTrue(torch.equal(p, before[n]), n)
        # The verified fp64 spec step is a central difference at the fixture radius: O(r^2) apart.
        self.assertLess(self.relative_gap(hvp, self.reference(1).gradient), 1e-6)
        # Finite differences converge to the HVP at second order: r -> r/10 shrinks the gap ~100x.
        gaps = [self.relative_gap(self.run_step(r=radius, microbatch_tokens=8)[0], hvp) for radius in (1e-2, 1e-3)]
        self.assertLess(gaps[1], gaps[0] / 30, gaps)
        self.assertGreater(gaps[0], 1e-10, gaps)   # the comparison is not vacuous

    def test_hvp_is_invariant_to_microbatching_draws_and_logprob_checkpointing(self):
        expected, _ = self.run_step(curvature="hvp")
        for changes in (dict(microbatch_tokens=4), dict(microbatch_tokens=8, hvp_microbatch_tokens=4),
                        dict(microbatch_tokens=0, hvp_microbatch_tokens=8),
                        dict(microbatch_tokens=8, logprob_chunk_size=3)):   # checkpointed scores, double backward
            with self.subTest(**changes):
                actual, _ = self.run_step(curvature="hvp", **changes)
                self.assert_same(actual, expected, rtol=1e-10)
        two, _ = self.run_step(curvature="hvp", num_sign_draws=2, microbatch_tokens=8)
        self.assertLess(self.relative_gap(two, self.reference(2).gradient), 1e-6)

    def test_hvp_geometry_only_plus_plain_is_exactly_the_full_hvp_step(self):
        full, _ = self.run_step(curvature="hvp", microbatch_tokens=8)
        for p in self.model.parameters():
            p.grad = None
        cbg_step(self.model, self.data, step_config(self.cfg, curvature="hvp", microbatch_tokens=8),
                 sign_generator=torch.Generator().manual_seed(7), fixed_samples=self.samples, include_base=False)
        plain_gradient(self.model, joint_batch(self.data), step_config(self.cfg, microbatch_tokens=8),
                       reference=self.ref)
        # Unlike FD (base averaged over theta +- eps w), the HVP takes the base at theta: exact split.
        self.assert_same({n: p.grad for n, p in self.model.named_parameters()}, full, rtol=1e-10)

    def test_hvp_runs_layer_split_and_degenerate_geometry_falls_back(self):
        expected, _ = self.run_step(curvature="hvp", microbatch_tokens=8)
        place_layers(self.model, ["cpu", "cpu"])
        actual, _ = self.run_step(curvature="hvp", microbatch_tokens=8)
        self.assert_same(actual, expected, rtol=1e-10)
        self.data["harmful"] = copy.deepcopy(self.data["safe"])
        plain = spec_training_gradient(self.model, self.data, replace(self.cfg, beta=0.),
                                       reference_logits=self.ref, fixed_samples=self.samples).gradient
        actual, diag = self.run_step(curvature="hvp", microbatch_tokens=4)
        self.assertEqual(diag["geometry_skipped"], 1.)
        self.assert_same(actual, plain, rtol=1e-12)

    def test_curvature_estimates_compare_fd_and_hvp_on_one_center(self):
        estimates = {}
        def sink(kind, value, seconds, epsilon):
            self.assertGreaterEqual(seconds, 0.)
            self.assertEqual(epsilon is None, kind == "hvp")
            estimates[(kind, value)] = {n: p.grad.clone() for n, p in self.model.named_parameters()}
        curvature_estimates(self.model, self.data, step_config(self.cfg, microbatch_tokens=8, hvp_microbatch_tokens=4),
                            plan=[("hvp", "fp32"), ("fd", 1e-4), ("fd", 1e-3)],
                            sample_generator=torch.Generator().manual_seed(3),
                            sign_generator=torch.Generator().manual_seed(4), sink=sink)
        exact = estimates[("hvp", "fp32")]
        near, far = (self.relative_gap(estimates[("fd", r)], exact) for r in (1e-4, 1e-3))
        self.assertLess(near, 1e-6)
        self.assertLess(near, far / 30)
        with self.assertRaises(ValueError):
            step_config(self.cfg, curvature="exact")

    def test_curvature_radius_sweep_is_consistent(self):
        estimates = {}
        def sink(radius, epsilon):
            estimates[radius] = torch.cat([p.grad.flatten().clone() for p in self.model.parameters()])
        curvature_gradient(self.model, self.data, step_config(self.cfg, microbatch_tokens=8),
                           radii=[1e-3, 1e-4], sample_generator=torch.Generator().manual_seed(3),
                           sign_generator=torch.Generator().manual_seed(4), sink=sink)
        a, b = estimates[1e-3], estimates[1e-4]
        self.assertLess(float((a - b).norm() / b.norm()), 1e-4)


class TruncationTest(unittest.TestCase):
    class Tok:
        bos_token_id, eos_token_id, pad_token_id = 1, 2, 0
        def encode(self, text, add_special_tokens=False):
            return [3 + ord(c) % 50 for c in text]

    def test_min_response_tokens_keeps_prompt_and_cuts_response_tail(self):
        tok = self.Tok()
        prompt, response = "p" * 10, "r" * 40
        old = tokenize_prompt_response(tok, prompt, response, max_length=32)
        self.assertEqual(int(old["response_mask"].sum()), 31)          # prompt cut to one token
        new = tokenize_prompt_response(tok, prompt, response, max_length=32, min_response_tokens=8)
        self.assertEqual(int(new["response_mask"].sum()), 21)          # BOS + 10 prompt tokens kept
        long_prompt = tokenize_prompt_response(tok, "p" * 60, response, max_length=32, min_response_tokens=8)
        self.assertEqual(int(long_prompt["response_mask"].sum()), 8)
        h, s = tokenize_paired_responses(tok, prompt, response, "r" * 5, max_length=32, min_response_tokens=8)
        torch.testing.assert_close(h["input_ids"][:11], s["input_ids"][:11])
        self.assertEqual(int(s["response_mask"].sum()), 5)


if __name__ == "__main__":
    unittest.main()
