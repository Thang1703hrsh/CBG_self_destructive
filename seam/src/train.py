import argparse
import os
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union

os.environ.setdefault("USE_TF", "0")  # This training entrypoint uses PyTorch only.

import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from transformers.utils import requires_backends

from src.core.trainer import SeamTrainer
from src.core.numerics import collate_seam_batch
from src.data_processing._datasets import construct_alpaca_dataset, construct_beavertails_dataset
from src.utils.config import load_yaml_config, MyParser


@dataclass
class SeamArguments:
    # Model configuration
    model_name: str = field(
        default=None, 
        metadata={"help": "Path to pretrained model"}
    )
    tokenizer_name: str = field(
        default=None, 
        metadata={"help": "Name or path of the tokenizer"}
    )
    wandb_project_name: str = field(
        default= "seam",
        metadata={"help": "WandB project name"}
    )

    wandb_run_name: Optional[str] = field(
        default=None,
        metadata={"help": "WandB run name (optional)"}
    )
    attack_dataset: bool = field(
        default=True,
        metadata={"help": "Whether to use attack dataset"}
    )
    defense_size: int = field(
        default=4000,
        metadata={"help": "Number of examples for defense"}
    )
    data_path: str = field(
        default="data/beavertails_with_refusals_train.json",
        metadata={"help": "Path to defense BeaverTails JSON"},
    )
    benign_seed: int = field(
        default=42,
        metadata={"help": "Seed for selecting benign Alpaca examples"},
    )
    weight_dtype: str = field(default="float32", metadata={"help": "float32 or bfloat16 model weights"})
    gradient_storage: str = field(default="cpu", metadata={"help": "cpu or gpu extra gradient buffers"})
    optimizer_cpu_offload: bool = field(default=True, metadata={"help": "FP32 AdamW moments on CPU"})
    
    # Optimization parameters
    epsilon: float = field(
        default=1e-3,
        metadata={"help": "Parameter pertubation radius"}
    )
    alpha: float = field(
        default=1.0,
        metadata={"help": "Weight for L_up"}
    )
    beta: float = field(
        default=0.01,
        metadata={"help": "Weight for L_sd"}
    )

@dataclass
class MyTrainingArguments(TrainingArguments):
    @property
    def n_gpu(self):
        """
        The number of GPUs used for data paralleling.

        Note:
            Reserve two GPUs for gradient storage when available. With exactly two
            GPUs, train on cuda:0 and store both gradient buffers on cuda:1.
        """
        requires_backends(self, ["torch"])
        if not hasattr(self, "_n_gpu"):
            _ = self._setup_devices
        return max(1, self._n_gpu - 2)

def main():
    parser = argparse.ArgumentParser(description="Simple config path parser")
    parser.add_argument('--config', type=str, default='config/train.yaml', help='Path to YAML config file')
    config_path, remaining_args = parser.parse_known_args()

    yaml_config = load_yaml_config(config_path.config)
    training_yaml_args = yaml_config.get('TrainingArguments', {})
    unlearn_yaml_args = yaml_config.get('UnlearnArguments', {})
    yaml_defaults = {**training_yaml_args, **unlearn_yaml_args}

    hf_parser = MyParser((MyTrainingArguments, SeamArguments))
    training_args, seam_args = hf_parser.parse_args_into_dataclasses(
        args=remaining_args,
        default_values=yaml_defaults,
    )
    if seam_args.weight_dtype not in {"float32", "bfloat16"} or seam_args.gradient_storage not in {"cpu", "gpu"}:
        raise ValueError("Invalid SEAM weight_dtype or gradient_storage")
    if seam_args.weight_dtype == "float32" and (training_args.bf16 or training_args.fp16):
        raise ValueError("FP32 SEAM finite differences require --bf16 false --fp16 false")
    if seam_args.optimizer_cpu_offload and seam_args.weight_dtype != "float32":
        raise ValueError("CPU AdamW requires FP32 model weights")
    if training_args.tf32:
        raise ValueError("Disable TF32 for SEAM finite-difference training")

    wandb.init(project=seam_args.wandb_project_name, name=seam_args.wandb_run_name, group="train")
    print(f"Using config from: {config_path.config}")
    wandb.config.update(asdict(training_args)) 
    wandb.config.update(asdict(seam_args))

    try:
        tokenizer = AutoTokenizer.from_pretrained(seam_args.model_name)
    except Exception as e:
        tokenizer = AutoTokenizer.from_pretrained(seam_args.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(seam_args.model_name,
                                               torch_dtype=getattr(torch, seam_args.weight_dtype))
    model.config.use_cache = False

    harmful_dataset, harmless_dataset, test_dataset = construct_beavertails_dataset(
        tokenizer,
        attack=seam_args.attack_dataset,
        defence_size=seam_args.defense_size,
        train_path=seam_args.data_path,
    )
    benign_dataset = construct_alpaca_dataset(
        tokenizer, retain_size=seam_args.defense_size, seed=seam_args.benign_seed
    )

    trainer = SeamTrainer(
        model=model,
        args=training_args,
        seam_args=seam_args,
        train_dataset=harmful_dataset,
        tokenizer=tokenizer,
        align_dataset=harmless_dataset,
        benign_dataset=benign_dataset,
        data_collator=collate_seam_batch,
    )
    trainer.train()
    wandb.log({"Output dir": training_args.output_dir})
    wandb.finish()

if __name__ == "__main__":
    main()
