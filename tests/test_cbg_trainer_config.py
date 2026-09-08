import os

# The active development environment has an unrelated Keras 3 installation;
# CBG training is PyTorch-only.
os.environ.setdefault("USE_TF", "0")

import json
import tempfile
import unittest
from collections import OrderedDict
from dataclasses import asdict, fields
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from transformers import TrainingArguments

from src.core.cbg_trainer import (
    CBG_CONFIG_NAME,
    CBG_STATE_NAME,
    CbgArguments,
    CbgTrainer,
    _config_fingerprint,
)
from src.utils.config import MyParser, load_yaml_config


class _LocalBackward:
    @staticmethod
    def backward(loss):
        loss.backward()

    @staticmethod
    def unwrap_model(model):
        return model


class _ToyLoRAProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(2))


class _ToyLoRAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = _ToyLoRAProjection()


class CbgTrainerConfigTest(unittest.TestCase):
    def test_guide_defaults_and_validation(self):
        config = CbgArguments()
        self.assertEqual(config.lora_target_modules, ["q_proj", "v_proj"])
        self.assertEqual(config.inner_steps, 2)
        self.assertEqual(config.damping_mode, "relative")
        self.assertEqual(config.damping_fraction, 0.05)
        self.assertEqual(config.tau_fd, 1e-4)
        self.assertEqual(config.beta, 1e-3)

        with self.assertRaisesRegex(ValueError, "lora_dropout=0"):
            CbgArguments(lora_dropout=0.1)
        with self.assertRaisesRegex(ValueError, "beta_calibration_steps"):
            CbgArguments(beta_mode="grad_ratio", beta_calibration_steps=0)

    def test_checkpoint_fingerprint_ignores_only_run_local_fields(self):
        first = asdict(CbgArguments())
        second = dict(first)
        second["wandb_run_name"] = "resumed-name"
        second["cbg_resume_from_checkpoint"] = "checkpoint-10"
        self.assertEqual(_config_fingerprint(first), _config_fingerprint(second))

        second["tau_fd"] = 1e-3
        self.assertNotEqual(_config_fingerprint(first), _config_fingerprint(second))

    def test_candidate_yaml_settings_match_actual_dataclass_fields(self):
        for name in ("cbg_smoke_1b.yaml", "cbg_llama32_3b.yaml"):
            configuration = load_yaml_config(Path(__file__).resolve().parents[1] / "config" / name)
            for section, cls in (("CbgArguments", CbgArguments), ("TrainingArguments", TrainingArguments)):
                unknown = set(configuration[section]) - {field.name for field in fields(cls)}
                self.assertFalse(unknown, f"unknown {section} keys in {name}: {unknown}")
            (cbg,) = MyParser((CbgArguments,)).parse_args_into_dataclasses(
                args=[], default_values=configuration["CbgArguments"],
            )
            self.assertEqual(cbg.functional_chunk_size, 1)
            self.assertEqual(cbg.inner_policy, "adaptive")
            self.assertEqual(cbg.beta_mode, "grad_ratio")
            self.assertEqual(cbg.inner_fail_on_nonconvergence, "3b" in name)

    def test_joint_training_and_cbg_argument_parser_has_no_collisions(self):
        # Constructing the real joint parser detects duplicate CLI options.
        # Parse the CBG half independently because the lightweight test
        # environment intentionally does not install Accelerate, which
        # TrainingArguments.__post_init__ requires even on CPU.
        MyParser((TrainingArguments, CbgArguments))
        parser = MyParser((CbgArguments,))
        (cbg,) = parser.parse_args_into_dataclasses(
            args=[],
            default_values={
                "split_seed": 13,
                "cbg_resume_from_checkpoint": "checkpoint-3",
                "lora_target_modules": ["q_proj", "v_proj"],
            },
        )
        self.assertEqual(cbg.split_seed, 13)
        self.assertEqual(cbg.cbg_resume_from_checkpoint, "checkpoint-3")
        self.assertEqual(cbg.lora_target_modules, ["q_proj", "v_proj"])
        (cli,) = parser.parse_args_into_dataclasses(
            args=["--lora_target_modules", "q_proj", "k_proj"], default_values={},
        )
        self.assertEqual(cli.lora_target_modules, ["q_proj", "k_proj"])

    def test_zero_proxy_uses_amp_hook_and_accumulates_gradient(self):
        trainer = CbgTrainer.__new__(CbgTrainer)
        trainer.accelerator = _LocalBackward()
        parameter = nn.Parameter(torch.tensor([2.0, -3.0], dtype=torch.float32))
        parameter.grad = torch.tensor([0.5, 0.25])
        parameters = OrderedDict(adapter=parameter)
        explicit = OrderedDict(adapter=torch.tensor([4.0, -2.0]))

        trainer._backward_explicit_gradient(parameters, explicit, divisor=2)

        torch.testing.assert_close(
            parameter.grad,
            torch.tensor([2.5, -0.75]),
            rtol=0,
            atol=0,
        )

    def test_fixed_beta_zero_disables_the_entire_geometry_path(self):
        trainer = CbgTrainer.__new__(CbgTrainer)
        trainer.cbg_args = CbgArguments(beta=0.0, beta_mode="fixed")
        self.assertFalse(trainer._should_compute_geometry())

        trainer.cbg_args = CbgArguments(beta=1e-3, beta_mode="fixed")
        self.assertTrue(trainer._should_compute_geometry())

        trainer.cbg_args = CbgArguments(
            beta=0.0,
            beta_mode="grad_ratio",
            beta_calibration_steps=1,
        )
        self.assertTrue(trainer._should_compute_geometry())

    def test_cbg_state_and_configuration_are_saved_without_live_graphs(self):
        trainer = CbgTrainer.__new__(CbgTrainer)
        trainer.args = SimpleNamespace(should_save=True)
        trainer.state = SimpleNamespace(global_step=7)
        trainer.cbg_args = CbgArguments()
        trainer.data_manifest = {"safety_source_sha256": "toy-source", "defense_indices": [0], "attack_indices": [1]}
        trainer._admissible_names = ("q_proj.lora_A",)
        live = torch.tensor([1.0, -2.0], requires_grad=True)
        trainer.prev_u = OrderedDict({"q_proj.lora_A": live * 2})
        trainer.calibrated_mu = 0.125
        trainer._beta_candidate_sum = 0.0
        trainer._beta_candidate_count = 0
        trainer._calibrated_beta = None

        with tempfile.TemporaryDirectory() as directory:
            trainer._save_cbg_artifacts(Path(directory))
            config_path = Path(directory) / CBG_CONFIG_NAME
            state_path = Path(directory) / CBG_STATE_NAME
            self.assertTrue(config_path.is_file())
            self.assertTrue(state_path.is_file())
            with config_path.open("r", encoding="utf-8") as handle:
                saved_config = json.load(handle)
            self.assertEqual(saved_config["method"], "cbg")
            state = torch.load(state_path, map_location="cpu", weights_only=True)
            self.assertEqual(state["global_step"], 7)
            self.assertEqual(state["calibrated_mu"], 0.125)
            saved_direction = state["prev_u"]["q_proj.lora_A"]
            self.assertEqual(saved_direction.dtype, torch.float32)
            self.assertFalse(saved_direction.requires_grad)
            torch.testing.assert_close(saved_direction, torch.tensor([2.0, -4.0]))

            restored = CbgTrainer.__new__(CbgTrainer)
            restored.args = SimpleNamespace(should_save=True)
            restored.cbg_args = CbgArguments()
            restored.data_manifest = dict(trainer.data_manifest)
            restored._admissible_names = ("q_proj.lora_A",)
            restored.accelerator = _LocalBackward()
            restored.prev_u = None
            restored.calibrated_mu = None
            restored._beta_candidate_sum = 0.0
            restored._beta_candidate_count = 0
            restored._calibrated_beta = None
            restored._load_cbg_state(Path(directory), _ToyLoRAModel())
            self.assertEqual(restored.calibrated_mu, 0.125)
            torch.testing.assert_close(
                restored.prev_u["q_proj.lora_A"],
                torch.tensor([2.0, -4.0]),
            )
            restored.data_manifest["attack_indices"] = [0]
            with self.assertRaisesRegex(ValueError, "manifest"):
                restored._load_cbg_state(Path(directory), _ToyLoRAModel())


if __name__ == "__main__":
    unittest.main()
