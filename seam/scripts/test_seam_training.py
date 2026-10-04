"""Offline integration: actual SeamTrainer + tiny random Llama; no GPU or HF downloads."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ['WANDB_MODE'] = 'disabled'
os.environ['USE_TF'] = '0'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import LlamaConfig, LlamaForCausalLM, TrainingArguments
from src.core.trainer import SeamTrainer
from src.core.numerics import CPUAdamW, collate_seam_batch


class TrainingIntegrationTests(unittest.TestCase):
    def test_accumulation_does_not_scale_each_component_twice(self):
        outcomes = []
        with tempfile.TemporaryDirectory() as directory:
            for accumulation in (1, 2):
                torch.manual_seed(123)
                model = LlamaForCausalLM(LlamaConfig(vocab_size=16, hidden_size=8,
                                                    intermediate_size=16, num_hidden_layers=1,
                                                    num_attention_heads=2, num_key_value_heads=2,
                                                    use_cache=False))
                def rows(ids):
                    return [{'input_ids': torch.tensor(ids), 'attention_mask': torch.ones(len(ids), dtype=torch.long)}
                            for _ in range(2)]
                output = Path(directory) / str(accumulation)
                args = TrainingArguments(output_dir=str(output), max_steps=1,
                                         per_device_train_batch_size=2 // accumulation,
                                         gradient_accumulation_steps=accumulation, use_cpu=True,
                                         report_to='none', save_strategy='no', learning_rate=2e-5,
                                         max_grad_norm=0., disable_tqdm=True)
                seam_args = SimpleNamespace(alpha=1., beta=.01, epsilon=.001,
                                            gradient_storage='cpu', optimizer_cpu_offload=True)
                trainer = SeamTrainer(model=model, args=args, seam_args=seam_args,
                                      train_dataset=rows([1, 3, 4, 5, 2]),
                                      align_dataset=rows([1, 3, 6, 7, 2]),
                                      benign_dataset=rows([1, 8, 9, 10, 2]),
                                      data_collator=collate_seam_batch)
                trainer.train()
                record = json.loads((output / 'loss_components.jsonl').read_text().splitlines()[0])
                outcomes.append(record)
        for key in ('unlearning_loss', 'utility_loss', 'harmful_gradient_norm', 'benign_gradient_norm'):
            self.assertAlmostEqual(outcomes[0][key], outcomes[1][key], places=4, msg=key)

    def test_real_trainer_two_steps_with_fp32_cpu_offload(self):
        torch.manual_seed(42)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=16, hidden_size=8,
                                            intermediate_size=16, num_hidden_layers=1,
                                            num_attention_heads=2, num_key_value_heads=2,
                                            max_position_embeddings=32, use_cache=False))
        def rows(ids):
            return [{'input_ids': torch.tensor(ids), 'attention_mask': torch.ones(len(ids), dtype=torch.long)}
                    for _ in range(4)]
        harmful, refusal, benign = rows([1, 3, 4, 5, 2]), rows([1, 3, 6, 7, 2]), rows([1, 8, 9, 10, 2])
        original = model.model.embed_tokens.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            args = TrainingArguments(output_dir=directory, max_steps=2,
                                     per_device_train_batch_size=2, use_cpu=True,
                                     report_to='none', save_strategy='no', logging_steps=1,
                                     learning_rate=2e-5, max_grad_norm=0., disable_tqdm=True)
            seam_args = SimpleNamespace(alpha=1., beta=.01, epsilon=.001,
                                        gradient_storage='cpu', optimizer_cpu_offload=True)
            trainer = SeamTrainer(model=model, args=args, seam_args=seam_args,
                                  train_dataset=harmful, align_dataset=refusal,
                                  benign_dataset=benign, data_collator=collate_seam_batch)
            result = trainer.train()
            self.assertEqual(result.global_step, 2)
            optimizer = getattr(trainer.optimizer, 'optimizer', trainer.optimizer)
            self.assertIsInstance(optimizer, CPUAdamW)
            self.assertFalse(torch.equal(model.model.embed_tokens.weight, original))
            records = [json.loads(line) for line in (Path(directory) / 'loss_components.jsonl').read_text().splitlines()]
            self.assertEqual(len(records), 2)
            for record in records:
                self.assertEqual(len(record['perturbations']), 2)
                self.assertGreater(record['weighted_sd_gradient_norm'], 0)
                self.assertTrue(all(p['actual_norm'] > 0 for p in record['perturbations']))


if __name__ == '__main__':
    unittest.main()
