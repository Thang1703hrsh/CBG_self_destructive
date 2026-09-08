from __future__ import annotations

import copy
import unittest
from collections import OrderedDict

import torch
from torch import nn

from src.cbg.functionals import (
    benign_preservation_loss,
    make_benign_functional,
    make_safety_functional,
    safe_alignment_loss,
    token_normalized_response_log_probs,
)
from src.cbg.rayleigh import warm_started_rayleigh_ascent
from src.cbg.spectral_grad import (
    centered_spectral_gradient,
    choose_physical_delta,
    compose_training_gradient,
    exact_spectral_gradient,
    make_scalarized_functional,
    parameter_gradient,
)
from src.cbg.tree_ops import (
    select_admissible_parameters,
    tree_add,
    tree_cosine_similarity,
    tree_dot,
    tree_norm,
    tree_random_like,
    tree_scale,
    tree_sub,
)


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 7, hidden_size: int = 3) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        embeddings = self.embedding(input_ids)
        hidden = torch.tanh(self.q_proj(embeddings) + self.v_proj(embeddings))
        return (self.lm_head(hidden),)


def sequence_batch(rows: list[list[int]]) -> dict[str, torch.Tensor]:
    input_ids = torch.tensor(rows, dtype=torch.long)
    response_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    response_mask[:, -2:] = True
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "response_mask": response_mask,
    }


class SpectralGradientTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(31)
        self.model = TinyCausalLM().to(dtype=torch.float64)
        self.model.eval()
        self.parameters = select_admissible_parameters(
            self.model,
            parameter_names=("q_proj.weight", "v_proj.weight"),
        )
        self.harmful = sequence_batch(
            [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5], [2, 3, 4, 5, 6]]
        )
        self.safe = sequence_batch(
            [[0, 1, 2, 5, 6], [1, 2, 3, 6, 0], [2, 3, 4, 0, 1]]
        )
        self.benign = sequence_batch(
            [[3, 2, 1, 0, 6], [4, 3, 2, 1, 5], [5, 4, 3, 2, 4]]
        )
        self.fh = make_safety_functional(self.model, self.harmful, self.safe)
        self.fb = make_benign_functional(self.model, self.benign)
        generator = torch.Generator(device="cpu").manual_seed(32)
        warm_start = tree_random_like(self.parameters, generator=generator)
        self.escape = warm_started_rayleigh_ascent(
            self.fh,
            self.fb,
            self.parameters,
            mu=0.2,
            inner_steps=3,
            inner_lr=1e-3,
            warm_start=warm_start,
            eps_norm=1e-15,
        ).direction

    def test_centered_fd_matches_exact_hvp(self) -> None:
        snapshots = OrderedDict(
            (name, (parameter.detach().clone(), parameter.data_ptr()))
            for name, parameter in self.model.named_parameters()
        )
        result = centered_spectral_gradient(
            self.fh,
            self.fb,
            self.parameters,
            self.escape,
            mu=0.2,
            tau_fd=1e-4,
            theta_floor=1e-3,
            u_floor=1e-12,
        )
        phi = make_scalarized_functional(self.fh, self.fb, result.center)
        exact = exact_spectral_gradient(
            phi,
            self.parameters,
            result.center.direction,
            result.center.b,
        )

        cosine = tree_cosine_similarity(result.gradient, exact)
        relative_error = tree_norm(tree_sub(result.gradient, exact)) / torch.clamp_min(
            tree_norm(exact),
            1e-15,
        )
        self.assertGreater(cosine.item(), 0.95)
        self.assertLess(relative_error.item(), 1e-2)

        expected_delta = (
            1e-4
            * torch.clamp_min(tree_norm(self.parameters), 1e-3)
            / torch.clamp_min(tree_norm(self.escape), 1e-12)
        )
        torch.testing.assert_close(result.delta, expected_delta, rtol=0, atol=0)
        for tensor in (
            result.center.b,
            result.center.rho,
            result.center.qh,
            result.center.qb,
            result.center.ch,
            result.center.cb,
        ):
            self.assertFalse(tensor.requires_grad)
            self.assertIsNone(tensor.grad_fn)

        for name, parameter in self.model.named_parameters():
            before, pointer = snapshots[name]
            self.assertTrue(torch.equal(parameter.detach(), before))
            self.assertEqual(parameter.data_ptr(), pointer)

    def test_preservation_losses_are_separate_and_reference_is_frozen(self) -> None:
        safe_logits = self.model(
            self.safe["input_ids"],
            attention_mask=self.safe["attention_mask"],
        )[0]
        expected_alignment = -token_normalized_response_log_probs(
            safe_logits,
            self.safe["input_ids"],
            self.safe["response_mask"],
            attention_mask=self.safe["attention_mask"],
        ).mean()
        alignment = safe_alignment_loss(self.model, self.parameters, self.safe)
        torch.testing.assert_close(alignment, expected_alignment, rtol=1e-12, atol=1e-12)

        reference_model = copy.deepcopy(self.model).eval()
        for parameter in reference_model.parameters():
            parameter.requires_grad_(False)
        with torch.no_grad():
            reference_logits = reference_model(
                self.benign["input_ids"],
                attention_mask=self.benign["attention_mask"],
            )[0]
        zero_kl = benign_preservation_loss(
            self.model,
            self.parameters,
            self.benign,
            reference_logits,
        )
        self.assertLess(abs(zero_kl.item()), 1e-12)

        changed = tree_add(self.parameters, tree_scale(self.escape, 0.05))
        positive_kl = benign_preservation_loss(
            self.model,
            changed,
            self.benign,
            reference_logits,
        )
        self.assertGreater(positive_kl.item(), 0.0)
        changed_gradient = parameter_gradient(positive_kl, changed)
        self.assertTrue(all(torch.isfinite(value).all() for value in changed_gradient.values()))
        self.assertTrue(all(parameter.grad is None for parameter in reference_model.parameters()))

    def test_beta_zero_exactly_disables_spectral_tree(self) -> None:
        generator = torch.Generator(device="cpu").manual_seed(33)
        alignment = tree_random_like(self.parameters, generator=generator)
        benign = tree_random_like(self.parameters, generator=generator)
        spectral = OrderedDict(
            (name, torch.full_like(value, torch.nan)) for name, value in self.parameters.items()
        )
        expected = tree_add(alignment, benign, alpha=0.37)
        actual = compose_training_gradient(
            alignment,
            benign,
            spectral,
            alpha=0.37,
            beta=0.0,
        )
        for name in expected:
            self.assertTrue(torch.equal(actual[name], expected[name]))

    def test_inner_search_graph_is_not_retained(self) -> None:
        generator = torch.Generator(device="cpu").manual_seed(34)
        warm_start = tree_random_like(self.parameters, generator=generator)
        warm_start = OrderedDict(
            (name, value.requires_grad_(True)) for name, value in warm_start.items()
        )
        result = warm_started_rayleigh_ascent(
            self.fh,
            self.fb,
            self.parameters,
            mu=0.2,
            inner_steps=5,
            inner_lr=1e-3,
            warm_start=warm_start,
        )
        for value in result.direction.values():
            self.assertFalse(value.requires_grad)
            self.assertIsNone(value.grad_fn)
        diagnostics = result.diagnostics
        diagnostic_tensors = (
            diagnostics.rho_hat,
            diagnostics.qh,
            diagnostics.qb,
            diagnostics.qb_damped,
            diagnostics.eig_residual_norm,
            diagnostics.u_euclidean_norm,
            diagnostics.b_mu_norm,
            diagnostics.mu,
            *diagnostics.rayleigh_values,
        )
        if diagnostics.warm_start_cosine is not None:
            diagnostic_tensors += (diagnostics.warm_start_cosine,)
        for value in diagnostic_tensors:
            self.assertFalse(value.requires_grad)
            self.assertIsNone(value.grad_fn)

    def test_parameters_remain_exact_after_exception_during_fd(self) -> None:
        snapshots = OrderedDict(
            (name, (parameter.detach().clone(), parameter.data_ptr()))
            for name, parameter in self.model.named_parameters()
        )
        calls = 0

        def raising_safety(parameters):
            nonlocal calls
            calls += 1
            if calls >= 4:
                raise RuntimeError("intentional finite-difference failure")
            return self.fh(parameters)

        with self.assertRaises(RuntimeError):
            centered_spectral_gradient(
                raising_safety,
                self.fb,
                self.parameters,
                self.escape,
                mu=0.2,
            )
        for name, parameter in self.model.named_parameters():
            before, pointer = snapshots[name]
            self.assertTrue(torch.equal(parameter.detach(), before))
            self.assertEqual(parameter.data_ptr(), pointer)

    def test_plus_and_minus_use_common_rng_state(self) -> None:
        parameter = OrderedDict(
            weight=torch.tensor([0.3, -0.2, 0.5], dtype=torch.float64, requires_grad=True)
        )
        direction = OrderedDict(weight=torch.tensor([0.2, 0.4, -0.1], dtype=torch.float64))
        states: list[torch.Tensor] = []
        safety_matrix = torch.tensor(
            [[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]], dtype=torch.float64
        )
        benign_matrix = torch.tensor(
            [[0.5, 1.0, 0.0], [1.0, -0.5, 0.5]], dtype=torch.float64
        )

        def stochastic_functional(matrix):
            def functional(parameters):
                states.append(torch.random.get_rng_state().clone())
                noise = 0.01 * torch.rand(2, dtype=torch.float64)
                return matrix @ parameters["weight"] + noise

            return functional

        torch.manual_seed(35)
        centered_spectral_gradient(
            stochastic_functional(safety_matrix),
            stochastic_functional(benign_matrix),
            parameter,
            direction,
            mu=0.2,
            same_randomness=True,
        )
        self.assertGreaterEqual(len(states), 4)
        self.assertTrue(torch.equal(states[-4], states[-2]))
        self.assertTrue(torch.equal(states[-3], states[-1]))


if __name__ == "__main__":
    unittest.main()
