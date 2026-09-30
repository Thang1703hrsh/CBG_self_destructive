"""Offline Trainer, batching, configuration and checkpoint integration."""
import os
os.environ.setdefault("USE_TF", "0")

from collections import OrderedDict
import copy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import torch
from transformers import TrainingArguments
from transformers.utils import is_accelerate_available

from src.core.cbg_trainer import CbgArguments, CbgTrainer, _config_fingerprint
from src.data_processing.cbg_datasets import CbgJointBatchDataset, collate_cbg_joint_batch
from src.utils.config import MyParser, load_yaml_config
from src.cbg.spec import spec_training_gradient
from test_cbg_spec import fixture


def arguments(**overrides):
    options = dict(cbg_version="spec", geometry_solver="exact", parameter_space="full",
                   reference_mode="separate_model", safety_batch_size=2, benign_batch_size=3,
                   spectral_modes=2, xi=.01, kappa_mu=.3, relative_radius=.001,
                   alpha=1., beta=.1)
    options.update(overrides)
    return CbgArguments(**options)


class SpecIntegrationTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_joint_batches_support_unequal_counts_and_independent_lengths(self):
        _, data, _ = fixture()
        safety = [dict(harmful={k:v[i] for k,v in data["harmful"].items()},
                       safe={k:v[i] for k,v in data["safe"].items()}, pair_id=i) for i in range(2)]
        benign = [{k:v[i] for k,v in data["benign"].items()} for i in range(3)]
        ds = CbgJointBatchDataset(safety, benign, m=2, n=3, seed=3)
        batch = collate_cbg_joint_batch([ds[0]])
        self.assertEqual(batch["harmful"]["input_ids"].shape[0], 2)
        self.assertEqual(batch["benign"]["input_ids"].shape[0], 3)
        self.assertEqual(len(ds), 1)
        with self.assertRaises(ValueError):
            collate_cbg_joint_batch([ds[0], ds[0]])
        with self.assertRaises(ValueError):
            CbgJointBatchDataset(safety, benign, m=2, n=4)

    def test_yaml_cli_precedence_and_checkpoint_fingerprints(self):
        config = load_yaml_config(Path(__file__).resolve().parents[1]/"config/cbg_spec_pilot.yaml")
        parser = MyParser((CbgArguments,))
        (args,) = parser.parse_args_into_dataclasses(args=["--relative_radius=0.002", "--beta", "0.2"],
                                                    default_values=config["CbgArguments"])
        self.assertEqual(args.relative_radius, .002)
        self.assertEqual(args.beta, .2)
        (plain,) = parser.parse_args_into_dataclasses(args=[])
        self.assertEqual(plain.cbg_version, "v2")
        with self.assertRaisesRegex(ValueError, "Unknown configuration"):
            parser.parse_args_into_dataclasses(args=[], default_values={"kapppa_mu": .1})
        first = asdict(args)
        self.assertNotEqual(_config_fingerprint(first), _config_fingerprint(dict(first, xi=.02)))
        old = asdict(CbgArguments())
        self.assertEqual(_config_fingerprint(old), _config_fingerprint(dict(old, xi=.02)))
        for invalid in (dict(xi=None), dict(kappa_mu=0.), dict(relative_radius=0.),
                        dict(spectral_modes=3), dict(parameter_space="lora"),
                        dict(beta_mode="grad_ratio"), dict(geometry_solver="dual")):
            with self.assertRaises(ValueError):
                arguments(**invalid)

    @unittest.skipUnless(is_accelerate_available(), "Trainer integration requires accelerate")
    def test_real_trainer_gradients_accumulation_and_checkpoint_resume(self):
        # Unmodified HF kernels and FP32 weights: also a real-model smoke test.
        model, data, _ = fixture(torch.float32)
        initial = copy.deepcopy(model)
        cbg = arguments()
        with tempfile.TemporaryDirectory() as directory:
            def make_trainer(out):
                return CbgTrainer(model=copy.deepcopy(initial), reference_model=copy.deepcopy(initial),
                    cbg_args=cbg, args=TrainingArguments(output_dir=out, use_cpu=True,
                        per_device_train_batch_size=1, gradient_accumulation_steps=2,
                        max_steps=2, learning_rate=2e-4, lr_scheduler_type="constant",
                        warmup_steps=0, save_steps=1, logging_strategy="no", report_to=[],
                        remove_unused_columns=False, disable_tqdm=True, seed=18,
                        dataloader_pin_memory=False), train_dataset=[data]*4,
                    data_collator=collate_cbg_joint_batch)
            trainer = make_trainer(directory)
            self.assertNotIn("model.embed_tokens.weight", trainer._admissible_names)
            self.assertTrue(trainer.model.model.embed_tokens.weight.requires_grad)
            # Verify the accumulation divisor exactly, with identical RNG.
            reference = trainer._reference_logits(trainer.model, data["benign"])
            torch.manual_seed(81)
            expected = spec_training_gradient(trainer.model, data, cbg.spec_config(), reference_logits=reference)
            torch.manual_seed(81)
            trainer.training_step(trainer.model, data)
            for name,p in trainer.model.named_parameters():
                torch.testing.assert_close(p.grad, expected.gradient[name]/2, rtol=1e-6, atol=1e-7)
            trainer.model.zero_grad(set_to_none=True)
            trainer.train()
            expected_weights = OrderedDict((n,p.detach().clone()) for n,p in trainer.model.named_parameters())
            resumed = make_trainer(str(Path(directory)/"resumed"))
            resumed.train(resume_from_checkpoint=str(Path(directory)/"checkpoint-1"))
            self.assertEqual(resumed.state.global_step, 2)
            for name,p in resumed.model.named_parameters():
                torch.testing.assert_close(p, expected_weights[name], rtol=0, atol=0)
            self.assertTrue(all(p.grad is None for p in resumed.reference_model.parameters()))
            self.assertTrue((Path(directory)/"checkpoint-2"/"cbg_config.json").exists())


if __name__ == "__main__":
    unittest.main()
