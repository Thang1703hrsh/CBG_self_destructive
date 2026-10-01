"""Independent v2 checks (not the older, unrelated cbg_version='v2' path)."""
import copy
import itertools
import unittest
from collections import OrderedDict
from dataclasses import asdict, replace
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path

import torch
from test_cbg_spec import fixture, fp64_norm, fp64_attention, jacobian_rows, flatten
from src.cbg.spec import (_center, _forward, joint_batch, measurements,
                          spec_training_gradient, measure_geometry)
from src.cbg.spec_hooks import spec_linear_modules
from src.cbg.spec_spectral import solve_spec_spectrum, eigen_sign_scalarization
from src.core.cbg_trainer import CbgArguments, CbgTrainer, _config_fingerprint


def objective(K, m, k, mode):
    K = (K+K.T)/2
    hh, hb, bb = K[:m, :m], K[:m, m:], K[m:, m:]
    mu = .3*bb.trace()/len(bb)
    if mode == "detached":
        mu = mu.detach()
    S = hh-hb@torch.linalg.solve(bb+mu*torch.eye(len(bb), dtype=K.dtype), hb.T)
    return torch.linalg.eigvalsh((S+S.T)/2)[-k:].sum().log()-torch.linalg.eigvalsh(hh)[-k:].sum().log()


class MethodV2Test(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        for target, fn in (("LlamaRMSNorm.forward", fp64_norm), ("eager_attention_forward", fp64_attention)):
            p = patch("transformers.models.llama.modeling_llama."+target, fn)
            p.start()
            self.addCleanup(p.stop)
        self.model, self.data, self.cfg = fixture()
        self.cfg = replace(self.cfg, alpha=0.)
        self.samples = torch.tensor([1, 2, 3, 4, 5, 6])

    def test_G_and_all_signs_match_autodiff_both_mu_modes(self):
        torch.manual_seed(15)
        theta = torch.randn(7, dtype=torch.float64, requires_grad=True)
        V = torch.randn(5, 7, dtype=torch.float64)
        F = lambda t: torch.sin(V@t)+.2*(V@t).square()
        C = torch.autograd.functional.jacobian(F, theta, create_graph=True)[:, :6]
        lam = theta[:6].square()+.01
        K = (C*lam)@C.T
        for mode in ("differentiated", "detached"):
            f = objective(K, 2, 2, mode)
            exact_G, = torch.autograd.grad(f, K, retain_graph=True)
            exact, = torch.autograd.grad(f, theta, retain_graph=True)
            q = solve_spec_spectrum(K, 2, 2, kappa_mu=.3, mu_mode=mode)
            torch.testing.assert_close(q.G, exact_G, rtol=1e-10, atol=1e-12)
            estimate = torch.zeros_like(theta)
            covariance = torch.zeros_like(K)
            dl = torch.cat((2*theta[:6].detach()*(C.detach()*(q.G@C.detach())).sum(0), theta.new_zeros(1)))
            for signs in itertools.product((-1., 1.), repeat=5):
                b, a = eigen_sign_scalarization(q, theta.new_tensor(signs))
                covariance += a[:, None]*b[None, :]/32
                w = torch.cat((lam.detach()*(C.detach().T@b), theta.new_zeros(1)))
                H = torch.autograd.functional.hessian(lambda t: a@F(t), theta)
                estimate += (2*H@w+dl)/32
            torch.testing.assert_close(covariance, q.G, rtol=1e-10, atol=1e-12)
            torch.testing.assert_close(estimate, exact, rtol=1e-9, atol=1e-11)

    def test_benign_inflation_has_zero_differentiated_derivative(self):
        torch.manual_seed(21)
        C = torch.randn(5, 9, dtype=torch.float64)
        scale = torch.tensor(1., dtype=torch.float64, requires_grad=True)
        rows = torch.cat((C[:2], C[2:]*scale))
        K = rows@rows.T
        derivative, = torch.autograd.grad(objective(K, 2, 2, "differentiated"), scale, retain_graph=True)
        detached, = torch.autograd.grad(objective(K, 2, 2, "detached"), scale)
        self.assertLess(abs(float(derivative)), 1e-12)
        self.assertLess(float(detached), -1e-6)

    def test_cached_directions_radius_and_single_center_for_multiple_draws(self):
        joint = joint_batch(self.data)
        modules = spec_linear_modules(self.model)
        q, _, hooks = _center(self.model, joint, self.cfg, modules, samples=self.samples, return_capture=True)
        self.assertIsNone(hooks.backward_target)
        b, _ = eigen_sign_scalarization(q, torch.ones(5))
        with patch("torch.autograd.grad", side_effect=AssertionError("unexpected direction backward")):
            hooks.materialize(b, q.A)
        params = OrderedDict(self.model.named_parameters())
        jac = jacobian_rows(measurements(_forward(self.model, joint), joint, samples=self.samples).values, params)
        norm = 0.
        for name, module in modules.items():
            C = jac[name].flatten(1)
            lam = module.weight.detach().square()+self.cfg.xi**2
            torch.testing.assert_close(hooks.w[name], (b@C).reshape_as(lam)*lam)
            torch.testing.assert_close(hooks.d_lambda[name], 2*module.weight.detach()*(C*(q.G@C)).sum(0).reshape_as(lam))
            norm += (hooks.w[name].square()/lam).sum()
        torch.testing.assert_close(norm, b@q.kernel@b)
        with patch("src.cbg.spec._center", wraps=_center) as center:
            result = spec_training_gradient(self.model, self.data, replace(self.cfg, num_sign_draws=4), fixed_samples=self.samples)
            self.assertEqual(center.call_count, 1)
        self.assertEqual(result.diagnostics["num_sign_draws"], 4)

    def test_beta_zero_is_exact_plain_gradient_without_center(self):
        joint = joint_batch(self.data)
        loss = measurements(_forward(self.model, joint), joint, samples=self.samples).alignment
        exact = torch.autograd.grad(loss, tuple(self.model.parameters()))
        with patch("src.cbg.spec._center", side_effect=AssertionError("beta zero used geometry")):
            result = spec_training_gradient(self.model, self.data, replace(self.cfg, beta=0.), fixed_samples=self.samples)
        torch.testing.assert_close(flatten(result.gradient), torch.cat([g.flatten() for g in exact]), rtol=1e-12, atol=1e-14)

    def test_padding_checkpoint_setting_and_offload_preserve_geometry(self):
        base = measure_geometry(self.model, self.data, config=self.cfg, samples=self.samples)
        self.model.gradient_checkpointing_enable()
        self.model.train()
        padded = copy.deepcopy(self.data)
        for group in padded.values():
            for field, value in group.items():
                group[field] = torch.nn.functional.pad(value, (0, 3), value=0)
        other = measure_geometry(self.model, padded, config=replace(self.cfg, capture_offload=True), samples=self.samples)
        self.assertTrue(self.model.training)
        self.assertTrue(self.model.is_gradient_checkpointing)
        for key in ("eta_k", "rho_mu", "tr_Kbb", "c_mu"):
            torch.testing.assert_close(base[key], other[key], rtol=1e-10, atol=1e-12)

    def test_degenerate_geometry_skips_only_cbg(self):
        data = copy.deepcopy(self.data)
        data["harmful"] = copy.deepcopy(data["safe"])
        result = spec_training_gradient(self.model, data, self.cfg, fixed_samples=self.samples)
        plain = spec_training_gradient(self.model, data, replace(self.cfg, beta=0.), fixed_samples=self.samples)
        self.assertEqual(result.diagnostics["geometry_skipped"], 1.)
        torch.testing.assert_close(flatten(result.gradient), flatten(plain.gradient))

    def test_frozen_trained_subset_does_not_shrink_U(self):
        expected = measure_geometry(self.model, self.data, config=self.cfg, samples=self.samples)
        full = spec_training_gradient(self.model, self.data, self.cfg, fixed_samples=self.samples,
                                      generator=torch.Generator().manual_seed(98))
        trained = "model.layers.1.mlp.down_proj.weight"
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(name == trained)
        actual = measure_geometry(self.model, self.data, config=self.cfg, samples=self.samples)
        torch.testing.assert_close(actual["eta_k"], expected["eta_k"])
        result = spec_training_gradient(self.model, self.data, self.cfg, fixed_samples=self.samples,
                                       generator=torch.Generator().manual_seed(98))
        self.assertEqual(list(result.gradient), [trained])
        torch.testing.assert_close(result.gradient[trained], full.gradient[trained], rtol=1e-9, atol=1e-11)

    def test_chunked_full_vocab_scores_and_gradients(self):
        joint = joint_batch(self.data)
        logits = _forward(self.model, joint).detach().requires_grad_(True)
        reference = torch.randn(6, logits.shape[-1], dtype=logits.dtype)
        for direction in ("reference_to_current", "current_to_reference"):
            def evaluate(chunk):
                values = measurements(logits, joint, samples=self.samples, reference_logits=reference,
                                      kl_direction=direction, logprob_chunk_size=chunk)
                scalar = values.values.square().sum()+values.alignment+values.benign_kl
                return scalar.detach(), torch.autograd.grad(scalar, logits)[0]
            expected = evaluate(0)
            for chunk in (1, 3, 128):
                actual = evaluate(chunk)
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-12)

    def test_gradient_component_diagnostics_do_not_change_update(self):
        ordinary = spec_training_gradient(self.model, self.data, self.cfg, fixed_samples=self.samples,
                                         generator=torch.Generator().manual_seed(5))
        measured = spec_training_gradient(self.model, self.data, replace(self.cfg, log_gradient_components=True),
                                         fixed_samples=self.samples, generator=torch.Generator().manual_seed(5))
        torch.testing.assert_close(flatten(measured.gradient), flatten(ordinary.gradient), rtol=0, atol=0)
        self.assertEqual(measured.diagnostics["gradient_components_measured"], 1.)
        self.assertGreater(measured.diagnostics["base_gradient_norm"], 0.)
        self.assertGreater(measured.diagnostics["beta_curvature_norm"], 0.)
        self.assertGreater(measured.diagnostics["beta_geometry_norm"], 0.)
        self.assertAlmostEqual(
            measured.diagnostics["geometry_to_base_grad_ratio"],
            measured.diagnostics["beta_geometry_norm"] / measured.diagnostics["base_gradient_norm"],
        )

    def test_L1_fp32_hook_gram_matches_per_example_autograd(self):
        model, data, config = fixture(torch.float32)
        joint = joint_batch(data)
        params = OrderedDict(model.named_parameters())
        modules = spec_linear_modules(model)
        values = measurements(_forward(model, joint), joint, samples=self.samples).values
        jac = jacobian_rows(values, params)
        expected = sum((jac[n].flatten(1).double()*(p.weight.detach().square()+config.xi**2).flatten().double())
                       @jac[n].flatten(1).double().T for n, p in modules.items())
        q, _ = _center(model, joint, config, modules, samples=self.samples, materialize=False)
        torch.testing.assert_close(q.kernel, expected, rtol=3e-5, atol=1e-10)

    def test_compressed_capture_keeps_fp64_oracle_and_stores_bf16_on_fp32(self):
        model, data, config = fixture(torch.float32)
        joint = joint_batch(data)
        q, _, captures = _center(model, joint, replace(config, capture_dtype="bf16", capture_offload=True),
                                 spec_linear_modules(model), samples=self.samples, return_capture=True)
        for x, delta in captures.captures.values():
            self.assertEqual(x.dtype, torch.bfloat16)
            self.assertEqual(delta.dtype, torch.bfloat16)
            self.assertEqual(x.device.type, "cpu")
        self.assertTrue(torch.isfinite(q.log_eta))

    def test_v2_config_roundtrip_and_legacy_checkpoint_rejection(self):
        from src.utils.config import load_yaml_config
        root = Path(__file__).resolve().parents[1]
        for filename in ("cbg_spec_pilot.yaml", "cbg_method_v2_pilot.yaml"):
            raw = load_yaml_config(root/"config"/filename)["CbgArguments"]
            args = CbgArguments(**raw)
            self.assertEqual(args.spec_config().mu_mode, "differentiated")
            self.assertEqual(args.spec_config().estimator, "shared_sign")
            current = asdict(args)
            old = dict(current)
            old.pop("spec_method_revision")
            self.assertNotEqual(_config_fingerprint(current), _config_fingerprint(old))
            changed = dict(current, mu_mode="detached")
            self.assertNotEqual(_config_fingerprint(current), _config_fingerprint(changed))
        with self.assertRaises(NotImplementedError):
            replace(self.cfg, estimator="per_example")

    def test_lazy_schedule_uses_optimizer_step_and_explicit_rescaling(self):
        from src.cbg.spec import SpecStepResult
        for rescale in (True, False):
            config = replace(self.cfg, lazy_period=3, lazy_rescale=rescale)
            records = []
            trainer = SimpleNamespace(
                optimizer=None,
                cbg_args=SimpleNamespace(safety_batch_size=2, benign_batch_size=3,
                                         alpha=0., spec_config=lambda: config),
                state=SimpleNamespace(global_step=0),
                accelerator=SimpleNamespace(unwrap_model=lambda model: model),
                _prepare_inputs=lambda batch: batch,
                _validate_batch=CbgTrainer._validate_batch,
                _gradient_accumulation_divisor=lambda: 2,
                _backward_explicit_gradient=lambda *args, **kwargs: None,
                _record_metrics=records.append)
            def step(model, batch, chosen, **kwargs):
                return SpecStepResult(OrderedDict(), torch.tensor(0.), {})
            with patch("src.cbg.spec.spec_training_gradient", side_effect=step) as execute:
                for iteration in (0, 0, 1, 1, 2, 2, 3, 3):
                    trainer.state.global_step = iteration
                    CbgTrainer._spec_training_step(trainer, self.model, self.data)
                active_beta = config.beta*(3 if rescale else 1)
                self.assertEqual([call.args[2].beta for call in execute.call_args_list],
                                 [active_beta, active_beta, 0., 0., 0., 0., active_beta, active_beta])


if __name__ == "__main__":
    unittest.main()
