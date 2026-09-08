from __future__ import annotations

import unittest
from collections import OrderedDict

import torch
from torch import nn
from torch.func import jacrev
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from src.cbg.functionals import (
    make_benign_functional,
    make_safety_functional,
    safety_functional_map,
    token_normalized_response_log_probs,
)
from src.cbg.geometry import (
    damped_benign_metric_vector_product,
    metric_vector_product,
    uniform_functional_weights,
)
from src.cbg.rayleigh import (
    calibrate_relative_damping,
    normalize_b_metric,
    warm_started_rayleigh_ascent,
)
from src.cbg.tree_ops import (
    select_admissible_parameters,
    select_lora_parameters,
    tree_cosine_similarity,
    tree_dot,
    tree_random_like,
)


class TinyCausalLM(nn.Module):
    """Small differentiable causal model with no stochastic state."""

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


class ShapeRecordingMode(TorchDispatchMode):
    """Record tensor output shapes produced while a matrix-free operation runs."""

    def __init__(self) -> None:
        super().__init__()
        self.shapes: list[tuple[int, ...]] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        del types
        output = func(*args, **(kwargs or {}))
        leaves, _ = tree_flatten(output)
        self.shapes.extend(tuple(leaf.shape) for leaf in leaves if isinstance(leaf, torch.Tensor))
        return output


def make_sequence_batch(rows: list[list[int]]) -> dict[str, torch.Tensor]:
    input_ids = torch.tensor(rows, dtype=torch.long)
    response_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    response_mask[:, -2:] = True
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "response_mask": response_mask,
    }


class CbgGeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(23)
        cls.model = TinyCausalLM().to(dtype=torch.float64)
        cls.model.eval()
        cls.parameters = select_admissible_parameters(
            cls.model,
            parameter_names=("q_proj.weight", "v_proj.weight"),
        )
        cls.harmful_batch = make_sequence_batch(
            [
                [0, 1, 2, 3, 4],
                [1, 2, 3, 4, 5],
                [2, 3, 4, 5, 6],
            ]
        )
        cls.safe_batch = make_sequence_batch(
            [
                [0, 1, 2, 5, 6],
                [1, 2, 3, 6, 0],
                [2, 3, 4, 0, 1],
            ]
        )
        cls.benign_batch = make_sequence_batch(
            [
                [3, 2, 1, 0, 6],
                [4, 3, 2, 1, 5],
                [5, 4, 3, 2, 4],
            ]
        )
        cls.safety_functional = staticmethod(
            make_safety_functional(
                cls.model,
                cls.harmful_batch,
                cls.safe_batch,
            )
        )
        cls.benign_functional = staticmethod(
            make_benign_functional(cls.model, cls.benign_batch)
        )
        cls.parameter_count = sum(parameter.numel() for parameter in cls.parameters.values())

    @classmethod
    def random_direction(cls, seed: int) -> OrderedDict[str, torch.Tensor]:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return tree_random_like(cls.parameters, generator=generator)

    @classmethod
    def flatten_tree(cls, tree) -> torch.Tensor:
        return torch.cat([tree[name].reshape(-1) for name in cls.parameters])

    def explicit_jacobian(self, functional) -> torch.Tensor:
        jacobian_tree = jacrev(functional)(self.parameters)
        batch_size = functional(self.parameters).numel()
        return torch.cat(
            [jacobian_tree[name].reshape(batch_size, -1) for name in self.parameters],
            dim=1,
        )

    def test_functional_maps_use_token_normalized_response_log_probs(self) -> None:
        parameters = self.parameters
        functional_values = safety_functional_map(
            self.model,
            parameters,
            self.harmful_batch,
            self.safe_batch,
        )
        harmful_logits = self.model(
            self.harmful_batch["input_ids"],
            attention_mask=self.harmful_batch["attention_mask"],
        )[0]
        safe_logits = self.model(
            self.safe_batch["input_ids"],
            attention_mask=self.safe_batch["attention_mask"],
        )[0]
        expected = token_normalized_response_log_probs(
            harmful_logits,
            self.harmful_batch["input_ids"],
            self.harmful_batch["response_mask"],
            attention_mask=self.harmful_batch["attention_mask"],
        ) - token_normalized_response_log_probs(
            safe_logits,
            self.safe_batch["input_ids"],
            self.safe_batch["response_mask"],
            attention_mask=self.safe_batch["attention_mask"],
        )
        torch.testing.assert_close(functional_values, expected, rtol=1e-12, atol=1e-12)
        self.assertEqual(tuple(self.benign_functional(parameters).shape), (3,))

    def test_uniform_weights_are_identity_over_batch_size(self) -> None:
        values = torch.tensor([2.0, -4.0, 8.0], dtype=torch.float64)
        torch.testing.assert_close(
            uniform_functional_weights(values),
            values / 3,
            rtol=0,
            atol=0,
        )

    def test_a_matrix_free_metric_vector_product_matches_explicit_jacobian(self) -> None:
        direction = self.random_direction(seed=101)
        for name, functional in (
            ("safety", self.safety_functional),
            ("benign", self.benign_functional),
        ):
            with self.subTest(functional=name):
                jacobian = self.explicit_jacobian(functional)
                explicit_gram = jacobian.T @ jacobian / jacobian.shape[0]
                expected = explicit_gram @ self.flatten_tree(direction)

                actual_tree = metric_vector_product(
                    functional,
                    self.parameters,
                    direction,
                )
                actual = self.flatten_tree(actual_tree)
                relative_error = torch.linalg.vector_norm(actual - expected) / torch.clamp_min(
                    torch.linalg.vector_norm(expected),
                    1e-15,
                )
                self.assertLess(relative_error.item(), 1e-10)

    def test_b_metric_is_symmetric(self) -> None:
        v = self.random_direction(seed=102)
        w = self.random_direction(seed=103)
        gv = metric_vector_product(self.benign_functional, self.parameters, v)
        gw = metric_vector_product(self.benign_functional, self.parameters, w)
        lhs = tree_dot(v, gw)
        rhs = tree_dot(w, gv)
        torch.testing.assert_close(lhs, rhs, rtol=1e-10, atol=1e-12)

    def test_c_metric_is_positive_semidefinite(self) -> None:
        for seed in range(104, 112):
            direction = self.random_direction(seed=seed)
            metric_direction = metric_vector_product(
                self.safety_functional,
                self.parameters,
                direction,
            )
            quadratic_form = tree_dot(direction, metric_direction)
            self.assertGreaterEqual(quadratic_form.item(), -1e-12)

    def test_d_b_mu_normalization(self) -> None:
        mu = 0.2
        direction = self.random_direction(seed=112)
        normalized = normalize_b_metric(
            self.benign_functional,
            self.parameters,
            direction,
            mu,
            eps_norm=1e-15,
        )
        gb_normalized = metric_vector_product(
            self.benign_functional,
            self.parameters,
            normalized,
        )
        b_cost = tree_dot(normalized, gb_normalized) + mu * tree_dot(normalized, normalized)
        torch.testing.assert_close(
            b_cost,
            torch.ones_like(b_cost),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_damped_benign_operator_is_gb_plus_mu_identity(self) -> None:
        mu = 0.2
        direction = self.random_direction(seed=115)
        gb_direction = metric_vector_product(
            self.benign_functional,
            self.parameters,
            direction,
        )
        damped_direction = damped_benign_metric_vector_product(
            self.benign_functional,
            self.parameters,
            direction,
            mu,
        )
        for name in direction:
            torch.testing.assert_close(
                damped_direction[name],
                gb_direction[name] + mu * direction[name],
                rtol=1e-12,
                atol=1e-12,
            )

    def test_relative_damping_calibration_is_positive_and_reproducible(self) -> None:
        first = calibrate_relative_damping(
            self.benign_functional,
            self.parameters,
            damping_fraction=0.05,
            num_directions=4,
            generator=torch.Generator(device="cpu").manual_seed(116),
        )
        second = calibrate_relative_damping(
            self.benign_functional,
            self.parameters,
            damping_fraction=0.05,
            num_directions=4,
            generator=torch.Generator(device="cpu").manual_seed(116),
        )
        self.assertGreater(first.item(), 0.0)
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_e_rayleigh_ascent_increases_objective(self) -> None:
        warm_start = self.random_direction(seed=113)
        result = warm_started_rayleigh_ascent(
            self.safety_functional,
            self.benign_functional,
            self.parameters,
            mu=0.2,
            inner_steps=4,
            inner_lr=1e-3,
            warm_start=warm_start,
            eps_norm=1e-15,
        )
        history = torch.stack(result.diagnostics.rayleigh_values)
        increments = history[1:] - history[:-1]
        self.assertTrue(torch.all(increments >= -1e-12), msg=f"history={history.tolist()}")
        self.assertGreater(history[-1].item(), history[0].item())
        diagnostics = result.diagnostics
        torch.testing.assert_close(
            diagnostics.rho_hat,
            diagnostics.qh / diagnostics.qb_damped,
            rtol=1e-12,
            atol=1e-12,
        )
        torch.testing.assert_close(
            diagnostics.qb_damped,
            diagnostics.qb + 0.2 * diagnostics.u_euclidean_norm.square(),
            rtol=1e-10,
            atol=1e-10,
        )
        self.assertTrue(torch.isfinite(diagnostics.eig_residual_norm).item())
        torch.testing.assert_close(
            diagnostics.b_mu_norm,
            torch.ones_like(diagnostics.b_mu_norm),
            rtol=1e-10,
            atol=1e-10,
        )
        expected_cosine = tree_cosine_similarity(result.direction, warm_start)
        self.assertIsNotNone(diagnostics.warm_start_cosine)
        torch.testing.assert_close(
            diagnostics.warm_start_cosine,
            expected_cosine,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_f_matrix_free_path_allocates_no_parameter_square_tensor(self) -> None:
        direction = self.random_direction(seed=114)
        recorder = ShapeRecordingMode()
        with recorder:
            result = metric_vector_product(
                self.safety_functional,
                self.parameters,
                direction,
            )
        self.assertEqual(tuple(result), tuple(self.parameters))
        forbidden_shape = (self.parameter_count, self.parameter_count)
        self.assertNotIn(
            forbidden_shape,
            recorder.shapes,
            msg=f"matrix-free path allocated forbidden tensor shape {forbidden_shape}",
        )


class LoRASelectionTest(unittest.TestCase):
    def test_only_configured_lora_target_parameters_are_selected(self) -> None:
        class FakePeftModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_proj = nn.Module()
                self.q_proj.lora_A = nn.Linear(2, 2, bias=False)
                self.q_proj.lora_B = nn.Linear(2, 2, bias=False)
                self.v_proj = nn.Module()
                self.v_proj.lora_A = nn.Linear(2, 2, bias=False)
                self.k_proj = nn.Module()
                self.k_proj.lora_A = nn.Linear(2, 2, bias=False)
                self.base_weight = nn.Parameter(torch.ones(2, 2))

        model = FakePeftModel()
        selected = select_lora_parameters(model, target_modules=("q_proj", "v_proj"))
        self.assertTrue(selected)
        self.assertTrue(all("lora_" in name for name in selected))
        self.assertTrue(all("q_proj" in name or "v_proj" in name for name in selected))
        self.assertTrue(all("k_proj" not in name for name in selected))
        self.assertNotIn("base_weight", selected)


if __name__ == "__main__":
    unittest.main()
