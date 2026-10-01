"""Independent numerical oracles for the authoritative implementation spec."""
import os
os.environ.setdefault("USE_TF", "0")

import copy
import itertools
from collections import OrderedDict
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch
from torch.func import functional_call, grad, vmap
from transformers import LlamaConfig, LlamaForCausalLM

from src.cbg.spec import (SpecConfig, _center, _forward, joint_batch, measurements,
                          measure_geometry, spec_precision, spec_training_gradient)
from src.cbg.spec_hooks import SpecLinearHooks, spec_linear_modules
from src.cbg.spec_spectral import solve_spec_spectrum, sign_scalarization
from src.cbg.functionals import select_response_logits


def fp64_norm(module, hidden_states):
    # HF casts even double models to FP32 in RMSNorm. The required FP64 oracle
    # must preserve double arithmetic; the production FP32 formula is identical.
    return module.weight * hidden_states * torch.rsqrt(
        hidden_states.square().mean(-1, keepdim=True) + module.variance_epsilon)


def fp64_attention(module, query, key, value, attention_mask, scaling, dropout=0., **kwargs):
    from transformers.models.llama.modeling_llama import repeat_kv
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)
    scores = (query @ key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, :key.shape[-2]]
    weights = scores.softmax(-1)
    return (weights @ value).transpose(1, 2).contiguous(), weights


def fixture(dtype=torch.float64):
    torch.manual_seed(123)
    cfg = LlamaConfig(vocab_size=13, hidden_size=8, intermediate_size=12,
                      num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                      max_position_embeddings=32, use_cache=False, attention_dropout=0.)
    cfg._attn_implementation = "eager"
    model = LlamaForCausalLM(cfg).to(dtype).eval()
    def batch(rows):
        ids = torch.tensor(rows)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        mask[:, -2:] = True
        return dict(input_ids=ids, attention_mask=torch.ones_like(ids), response_mask=mask)
    data = dict(harmful=batch([[1,2,3,4], [2,3,4,5]]),
                safe=batch([[1,2,6,7], [2,3,8,9]]),
                benign=batch([[1,3,5,7], [2,4,6,8], [3,5,7,9]]))
    config = SpecConfig(xi=.01, kappa_mu=.3, r=1e-4, alpha=1., beta=.1, k=2)
    return model, data, config


def jacobian_rows(values, parameters, create_graph=False):
    rows = [torch.autograd.grad(v, tuple(parameters.values()), retain_graph=True,
                               create_graph=create_graph, allow_unused=True) for v in values]
    return OrderedDict((name, torch.stack([torch.zeros_like(p) if row[i] is None else row[i]
                                          for row in rows]))
                       for i, (name, p) in enumerate(parameters.items()))


def flatten(tree):
    return torch.cat([v.reshape(-1) for v in tree.values()])


class SpecTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        # Test-only precision corrections to HF's unconditional FP32 casts.
        for target, replacement in (("LlamaRMSNorm.forward", fp64_norm),
                                     ("eager_attention_forward", fp64_attention)):
            patched = patch("transformers.models.llama.modeling_llama." + target, replacement)
            patched.start()
            self.addCleanup(patched.stop)
        self.model, self.data, self.cfg = fixture()
        self.joint = joint_batch(self.data)
        self.modules = spec_linear_modules(self.model)
        self.params = OrderedDict(self.model.named_parameters())
        self.samples = torch.tensor([1, 2, 3, 4, 5, 6])

    def values(self, parameters=None):
        return measurements(_forward(self.model, self.joint, parameters), self.joint,
                            samples=self.samples).values

    def kernel(self):
        jac = jacobian_rows(self.values(), self.params)
        K = sum((jac[name].flatten(1) * (p.weight.detach().square() + self.cfg.xi**2).flatten())
                @ jac[name].flatten(1).T for name, p in self.modules.items())
        return K, jac

    def test_T1_hooks_match_individual_gradients_and_vmap(self):
        # Different response lengths and padding exercise masks and weighting.
        self.data["benign"]["attention_mask"][0, -1] = 0
        self.data["safe"]["response_mask"][1, -2] = False
        self.joint = joint_batch(self.data)
        self.samples = self.samples[:5]
        expected, jac = self.kernel()
        # Independent vmap(grad) oracle for the safety margins.
        # These safety sequences have no padding. Build their shared additive
        # causal mask OUTSIDE vmap: Transformers 4.49's 2D-mask conversion uses
        # an in-place write into an unbatched tensor when its mask is vmapped.
        # Passing the equivalent 4D mask avoids that framework limitation while
        # retaining the same model, gradients, vmap and numerical tolerances.
        width = self.data["harmful"]["input_ids"].shape[1]
        parameter = next(self.model.parameters())
        causal_mask = torch.full((width, width), torch.finfo(parameter.dtype).min,
                                 dtype=parameter.dtype, device=parameter.device).triu(1)[None, None]
        def one(p, harmful, safe, hm, sm):
            def score(ids, mask):
                logits = functional_call(self.model, p, (), {"input_ids": ids[None],
                    "attention_mask": causal_mask}).logits[0, :-1]
                scores = logits.log_softmax(-1).gather(1, ids[1:, None]).squeeze(1)
                return (scores * mask[1:]).sum() / mask[1:].sum()
            return (score(harmful, hm) - score(safe, sm)) / 2**.5
        batched = vmap(grad(one), in_dims=(None, 0, 0, 0, 0))(
            self.params, self.data["harmful"]["input_ids"], self.data["safe"]["input_ids"],
            self.data["harmful"]["response_mask"], self.data["safe"]["response_mask"])
        for name in self.modules:
            torch.testing.assert_close(jac[name][:2], batched[name], rtol=1e-10, atol=1e-12)
        for budget in (16384, 1024*1024):
            with SpecLinearHooks(self.modules, self.joint.seq_to_example, self.joint.attention_mask,
                                 xi=self.cfg.xi, max_bytes=budget) as hooks:
                v = self.values()
                self.assertIsNotNone(hooks.backward_target)
                torch.autograd.grad(v.sum(), hooks.backward_target, retain_graph=True)
                torch.testing.assert_close(hooks.K, expected, rtol=1e-10, atol=1e-12)
                q = solve_spec_spectrum(hooks.K, 2, 2, kappa_mu=.3, mu_mode="detached")
                cw, _ = sign_scalarization(q, torch.tensor([1, -1]), torch.tensor([-1, -1]))
                hooks.materialize(cw, q.A)
                torch.autograd.grad(v.sum(), hooks.backward_target)
                for name, module in self.modules.items():
                    C = jac[name].flatten(1)
                    lam = module.weight.detach().square() + self.cfg.xi**2
                    torch.testing.assert_close(hooks.w[name], (cw @ C).reshape_as(lam) * lam,
                                               rtol=1e-10, atol=1e-12)
                    expected_dl = -2 * module.weight.detach() * (C * (q.A @ C)).sum(0).reshape_as(lam)
                    torch.testing.assert_close(hooks.d_lambda[name], expected_dl, rtol=1e-10, atol=1e-12)

    def test_T2_reduced_solve_and_T4_full_damping_correction(self):
        torch.manual_seed(9)
        theta = torch.randn(9, dtype=torch.float64, requires_grad=True)
        C = torch.randn(5, 9, dtype=torch.float64)
        lam_inv = theta.square() + .01
        K = (C * lam_inv.detach()) @ C.T
        q = solve_spec_spectrum(K, 2, 2, kappa_mu=.3, mu_mode="detached")
        B = C[2:].T @ C[2:] + q.mu * torch.diag(lam_inv.detach().reciprocal())
        Gh = C[:2].T @ C[:2]
        chol = torch.linalg.cholesky(B)
        left = torch.linalg.solve_triangular(chol, Gh, upper=False)
        whitened = torch.linalg.solve_triangular(chol, left.T, upper=False).T
        expected = torch.linalg.eigvalsh((whitened + whitened.T)/2).flip(0)[:2]
        torch.testing.assert_close(q.rho[:2], expected, rtol=1e-8, atol=1e-10)
        u = lam_inv.detach()[:, None] * (C.T @ q.c)
        v = lam_inv.detach()[:, None] * (C.T @ q.reference_c)
        torch.testing.assert_close(u.T @ B @ u, torch.eye(2, dtype=B.dtype), rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(v.T @ (v / lam_inv.detach()[:, None]), torch.eye(2, dtype=B.dtype))
        scalar = -(q.mu/q.S) * ((u.square()/lam_inv[:, None]).sum(0)*q.rho[:2]).sum()
        scalar += (((v.square()/lam_inv[:, None]).sum(0)*q.nu[:2]).sum()/q.N)
        expected_dl = torch.autograd.grad(scalar, theta)[0]
        actual_dl = -2*theta.detach()*(C*(q.A @ C)).sum(0)
        torch.testing.assert_close(actual_dl, expected_dl, rtol=1e-10, atol=1e-12)

    def test_T3_all_signs_against_recomputed_central_difference(self):
        # Small nonlinear map permits FD of every coordinate, including one
        # coordinate OUTSIDE U, without a parameter-square model allocation.
        torch.manual_seed(12)
        theta = torch.randn(7, dtype=torch.float64, requires_grad=True)
        a = torch.randn(5, 7, dtype=torch.float64)
        def values(p):
            return torch.sin(a @ p) + .2 * (a @ p).square()
        C = torch.autograd.functional.jacobian(values, theta)[:, :6]
        lam = theta[:6].detach().square() + .01
        q = solve_spec_spectrum((C*lam)@C.T, 2, 2, kappa_mu=.3, mu_mode="detached")
        dl = torch.cat((-2*theta[:6].detach()*(C*(q.A@C)).sum(0), theta.new_zeros(1)))
        estimate = torch.zeros_like(theta)
        for signs in itertools.product((-1., 1.), repeat=4):
            cw, g = sign_scalarization(q, theta.new_tensor(signs[:2]), theta.new_tensor(signs[2:]))
            w = torch.cat((lam * (C.T @ cw), theta.new_zeros(1)))
            hessian = torch.autograd.functional.hessian(lambda p: g @ values(p), theta)
            estimate += (2*hessian@w + dl) / 16
        def objective(p):
            j = torch.autograd.functional.jacobian(values, p)[:, :6]
            K = (j * (p[:6].square()+.01)) @ j.T
            return solve_spec_spectrum(K, 2, 2, kappa_mu=.3, fixed_mu=q.mu).log_eta
        eps = 1e-5
        expected = torch.stack([(objective(theta.detach()+eps*d) - objective(theta.detach()-eps*d))/(2*eps)
                                for d in torch.eye(7, dtype=theta.dtype)])
        self.assertLess(float((estimate-expected).norm()/expected.norm()), 1e-6)
        self.assertGreater(float(estimate[-1].abs()), 1e-5)

    def test_T5_swiglu_invariance_and_T7_safety_scaling(self):
        config = replace(self.cfg, xi=0.)
        before = measure_geometry(self.model, self.data, config=config, samples=self.samples)
        before_e = measure_geometry(self.model, self.data, config=replace(config, damping="euclidean"), samples=self.samples)
        logits = _forward(self.model, self.joint).detach()
        with torch.no_grad():
            mlp = self.model.model.layers[0].mlp
            mlp.up_proj.weight[0].mul_(3.)
            mlp.down_proj.weight[:, 0].div_(3.)
        torch.testing.assert_close(_forward(self.model, self.joint), logits, rtol=1e-12, atol=1e-12)
        after = measure_geometry(self.model, self.data, config=config, samples=self.samples)
        torch.testing.assert_close(before["eta_k"], after["eta_k"], rtol=1e-8, atol=1e-12)
        after_e = measure_geometry(self.model, self.data, config=replace(config, damping="euclidean"), samples=self.samples)
        self.assertGreater(float((after_e["eta_k"]-before_e["eta_k"]).abs()), 1e-8)
        K, _ = self.kernel()
        scale = K.new_tensor([7.,7.,1.,1.,1.])
        q = solve_spec_spectrum(K, 2, 2, kappa_mu=.3)
        scaled = solve_spec_spectrum(K*scale[:, None]*scale[None, :], 2, 2, kappa_mu=.3)
        torch.testing.assert_close(q.log_eta, scaled.log_eta, rtol=1e-10, atol=1e-12)
        self.assertTrue(0 < float(q.log_eta.exp()) <= 1)

    def test_T6_endpoint_gradient_matches_exact_hvp_including_outside_U(self):
        with spec_precision(self.model):
            q, samples, g, w, dl = _center(self.model, self.joint, self.cfg, self.modules,
                generator=torch.Generator().manual_seed(5), samples=self.samples)
        full_w = OrderedDict((n, w.get(n, torch.zeros_like(p))) for n,p in self.params.items())
        def omega(p):
            return g @ measurements(_forward(self.model, self.joint, p), self.joint, samples=samples).values
        scalar = omega(self.params)
        first = torch.autograd.grad(scalar, tuple(self.params.values()), create_graph=True)
        dot = sum((v*full_w[n]).sum() for (n,_),v in zip(self.params.items(), first))
        exact = torch.cat([v.reshape(-1) for v in torch.autograd.grad(dot, tuple(self.params.values()))])*2
        norm = sum((v.square()/(self.params[n].detach().square()+self.cfg.xi**2)).sum() for n,v in w.items()).sqrt()
        errors = []
        for radius in (1e-3, 1e-4, 1e-5):
            epsilon = radius * sum(v.numel() for v in w.values())**.5 / norm
            endpoints = []
            for sign in (1, -1):
                p = OrderedDict((n, (v.detach()+sign*epsilon*full_w[n]).requires_grad_()) for n,v in self.params.items())
                endpoints.append(torch.cat([v.reshape(-1) for v in torch.autograd.grad(omega(p), tuple(p.values()))]))
            fd = (endpoints[0]-endpoints[1])/epsilon
            errors.append(float((fd-exact).norm()/exact.norm()))
        self.assertLess(errors[-1], 1e-6, errors)
        self.assertLess(errors[-1], errors[0], errors)

    def test_measurements_sum_fisher_global_token_losses_and_reference_kl(self):
        logits = _forward(self.model, self.joint)
        ref = select_response_logits(_forward(self.model, joint_batch(self.data))[4:], self.data["benign"]).detach() + .1
        values = measurements(logits, self.joint, samples=self.samples, reference_logits=ref)
        mask = self.joint.response_mask[:, 1:]
        benign = logits[4:, :-1][mask[4:]].log_softmax(-1)
        expected = benign.gather(1, self.samples[:, None]).reshape(3,2).sum(1)/3**.5
        torch.testing.assert_close(values.values[2:], expected)
        torch.testing.assert_close(values.benign_kl, torch.zeros_like(values.benign_kl), atol=1e-14, rtol=0)
        # Sampling has no gradient and remains exactly fixed in perturbed calls.
        sampled = measurements(logits, self.joint, generator=torch.Generator().manual_seed(3))
        again = measurements(logits+.2, self.joint, samples=sampled.samples)
        self.assertTrue(torch.equal(sampled.samples, again.samples))
        self.assertFalse(sampled.samples.requires_grad)

    def test_step_preserves_live_state_and_updates_all_trainable_coordinates(self):
        self.model.train()
        self.model.model.layers[0].eval()
        modes = [m.training for m in self.model.modules()]
        before = {n: (p.detach().clone(), p.data_ptr()) for n,p in self.params.items()}
        with torch.no_grad():
            ref = select_response_logits(self.model(**{k:v for k,v in self.data["benign"].items() if k != "response_mask"}).logits,
                                         self.data["benign"])
        result = spec_training_gradient(self.model, self.data, self.cfg, reference_logits=ref,
                                        generator=torch.Generator().manual_seed(3))
        self.assertEqual(tuple(result.gradient), tuple(self.params))
        self.assertGreater(float(result.gradient["model.embed_tokens.weight"].norm()), 0.)
        for n,p in self.params.items():
            self.assertTrue(torch.equal(p, before[n][0]))
            self.assertEqual(p.data_ptr(), before[n][1])
            self.assertIsNone(p.grad)
        self.assertEqual(modes, [m.training for m in self.model.modules()])
        with patch("src.cbg.spec._forward", side_effect=RuntimeError("injected")):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                spec_training_gradient(self.model, self.data, self.cfg, reference_logits=ref)
        self.assertEqual(modes, [m.training for m in self.model.modules()])
        self.assertTrue(all(not m._forward_hooks for m in self.modules.values()))

    def test_full_model_estimator_matches_differentiated_log_eta(self):
        config = replace(self.cfg, r=1e-5)
        with torch.no_grad():
            ref = select_response_logits(self.model(input_ids=self.data["benign"]["input_ids"]).logits,
                                         self.data["benign"])
        current = measurements(_forward(self.model, self.joint), self.joint,
                               samples=self.samples, reference_logits=ref)
        jac = jacobian_rows(current.values, self.params, create_graph=True)
        K = sum((jac[n].flatten(1)*(module.weight.square()+config.xi**2).flatten()) @ jac[n].flatten(1).T
                for n,module in self.modules.items())
        mu = .3*K[2:,2:].trace()/3
        M = (K[:2,:2] - K[:2,2:] @ torch.linalg.solve(K[2:,2:]+mu*torch.eye(3), K[2:,:2]))/mu
        objective = (mu*torch.linalg.eigvalsh((M+M.T)/2)[-2:].sum()).log() - torch.linalg.eigvalsh(K[:2,:2])[-2:].sum().log()
        loss = current.alignment + config.alpha*current.benign_kl + config.beta*objective
        exact = torch.cat([g.flatten() for g in torch.autograd.grad(loss, tuple(self.params.values()))])
        average = torch.zeros_like(exact)
        original = measurements
        def fixed_samples(*args, **kwargs):
            kwargs["samples"] = self.samples
            return original(*args, **kwargs)
        for signs in itertools.product((0, 1), repeat=5):
            with patch("src.cbg.spec.measurements", side_effect=fixed_samples), patch(
                "src.cbg.spec.torch.randint", return_value=torch.tensor(signs)):
                result = spec_training_gradient(self.model, self.data, config, reference_logits=ref)
                average += flatten(result.gradient)/32
        self.assertLess(float((average-exact).norm()/exact.norm()), 1e-6)

    def test_T8_twenty_steps_decrease_geometry_with_bounded_losses(self):
        reference_model = copy.deepcopy(self.model).requires_grad_(False)
        ref = select_response_logits(reference_model(input_ids=self.data["benign"]["input_ids"]).logits,
                                     self.data["benign"])
        config = replace(self.cfg, beta=1., num_sign_draws=4)
        before = measure_geometry(self.model, self.data, config=config, samples=self.samples)["log_eta_k"]
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-4, weight_decay=0.)
        generator = torch.Generator().manual_seed(76)
        for _ in range(20):
            result = spec_training_gradient(self.model, self.data, config, reference_logits=ref, generator=generator)
            for name, p in self.params.items():
                p.grad = result.gradient[name]
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            self.assertLess(float(result.diagnostics["L_aln"]), 5.)
            self.assertLess(float(result.diagnostics["L_bgn"]), .1)
        after = measure_geometry(self.model, self.data, config=config, samples=self.samples)["log_eta_k"]
        self.assertLess(float(after), float(before))

    def test_fixed_batch_geometry_only_uses_frozen_fisher_samples(self):
        config = replace(self.cfg, alpha=0., drop_aln=True, num_sign_draws=4)
        expected = config.beta * measure_geometry(
            self.model, self.data, config=config, samples=self.samples)["log_eta_k"]
        with patch("src.cbg.spec.torch.multinomial", side_effect=AssertionError("resampled Fisher")):
            result = spec_training_gradient(self.model, self.data, config,
                generator=torch.Generator().manual_seed(76), fixed_samples=self.samples)
        torch.testing.assert_close(result.loss, expected, rtol=1e-10, atol=1e-12)
        self.assertEqual(result.diagnostics["num_sign_draws"], 4)
        self.assertEqual(result.diagnostics["ablation_drop_aln"], 1.)


if __name__ == "__main__":
    unittest.main()
