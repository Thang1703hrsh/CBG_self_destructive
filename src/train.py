import argparse
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from transformers.utils import requires_backends

from src.core.trainer import SeamTrainer
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
        default=1000,
        metadata={"help": "Number of examples for defense"}
    )
    
    # Optimization parameters
    epsilon: float = field(
        default=1e-3,
        metadata={"help": "Parameter pertubation radius"}
    )
    alpha: float = field(
        default=1.0,
        metadata={"help": "Weight for L_sd"}
    )
    beta: float = field(
        default=0.001,
        metadata={"help": "Weight for L_up"}
    )

@dataclass
class MyTrainingArguments(TrainingArguments):
    @property
    def n_gpu(self):
        """
        The number of GPUs used for data paralleling.

        Note:
            In default Trainer, this is the number of visiable GPUs. Here we use visiable GPUs-2 for data paralleling because we use 2 GPUs to store extra gradients.
            if smaller base models are adopted, we can use 1 GPU for extra gradient transfer. And you can revise 2 to 1 in the code below.
        """
        requires_backends(self, ["torch"])
        if not hasattr(self, "_n_gpu"):
            _ = self._setup_devices
        return self._n_gpu - 2


def run_cbg(yaml_config, remaining_args, config_path, selected_method):
    """Run the isolated CBG path without changing legacy SEAM construction."""

    from peft import LoraConfig, TaskType, get_peft_model
    import json
    import hashlib
    from pathlib import Path
    from transformers import set_seed

    from src.core.cbg_trainer import CbgArguments, CbgTrainer
    from src.data_processing.cbg_datasets import (
        collate_cbg_defense_batch,
        construct_cbg_alpaca_dataset,
        construct_cbg_beavertails_datasets,
        construct_cbg_defense_dataset,
    )

    training_yaml_args = yaml_config.get('TrainingArguments', {})
    cbg_yaml_args = yaml_config.get('CbgArguments', {})
    yaml_defaults = {**training_yaml_args, **cbg_yaml_args}
    hf_parser = MyParser((TrainingArguments, CbgArguments))
    training_args, cbg_args = hf_parser.parse_args_into_dataclasses(
        args=remaining_args,
        default_values=yaml_defaults,
    )
    cbg_args.method = selected_method
    print(json.dumps({"TrainingArguments": asdict(training_args), "CbgArguments": asdict(cbg_args)},
                     indent=2, default=str))
    if not Path(cbg_args.beavertails_path).is_file():
        raise FileNotFoundError(f"Missing refusal dataset: {cbg_args.beavertails_path}")
    set_seed(training_args.seed)

    wandb.init(
        project=cbg_args.wandb_project_name,
        name=cbg_args.wandb_run_name,
        group="train",
    )
    print(f"Using config from: {config_path}")
    wandb.config.update(asdict(training_args))
    wandb.config.update(asdict(cbg_args))

    tokenizer_source = cbg_args.tokenizer_name or cbg_args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_dtype = torch.float32
    if training_args.bf16:
        model_dtype = torch.bfloat16
    elif training_args.fp16:
        model_dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        cbg_args.model_name,
        torch_dtype=model_dtype,
        attn_implementation=cbg_args.cbg_attention_implementation,
    )
    model.config.use_cache = False
    lora_config = LoraConfig(
        inference_mode=False,
        task_type=TaskType.CAUSAL_LM,
        r=cbg_args.lora_rank,
        lora_alpha=cbg_args.lora_alpha,
        lora_dropout=cbg_args.lora_dropout,
        bias=cbg_args.lora_bias,
        target_modules=list(cbg_args.lora_target_modules),
    )
    model = get_peft_model(model, lora_config)
    # Only adapters are trainable, and their coordinate arithmetic stays FP32.
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.dtype != torch.float32:
            parameter.data = parameter.data.float()
    print(f"CBG trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    reference_model = None
    if cbg_args.reference_mode == "separate_model":
        reference_model = AutoModelForCausalLM.from_pretrained(
            cbg_args.model_name,
            torch_dtype=model_dtype,
            attn_implementation=cbg_args.cbg_attention_implementation,
        )
        reference_model.config.use_cache = False
        reference_model.eval()
        for parameter in reference_model.parameters():
            parameter.requires_grad_(False)

    safety_dataset, _attack_dataset, split_indices = construct_cbg_beavertails_datasets(
        tokenizer,
        path=cbg_args.beavertails_path,
        defense_size=cbg_args.defense_size,
        attack_size=cbg_args.attack_size,
        seed=cbg_args.split_seed,
        shuffle=cbg_args.data_shuffle,
        max_length=cbg_args.context_length,
    )
    benign_dataset = construct_cbg_alpaca_dataset(
        tokenizer,
        retain_size=cbg_args.defense_size,
        max_length=cbg_args.context_length,
    )
    defense_dataset = construct_cbg_defense_dataset(safety_dataset, benign_dataset)
    with Path(cbg_args.beavertails_path).open("rb") as source:
        source_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    data_manifest = {
        "safety_source_sha256": source_sha256,
        "defense_indices": list(split_indices.defense),
        "attack_indices": list(split_indices.attack),
        "benign_source": "yahma/alpaca-cleaned",
        "benign_examples": len(benign_dataset),
    }
    output_path = Path(training_args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / "cbg_data_split.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != data_manifest:
        raise ValueError("output directory already contains a different CBG data split")
    manifest_path.write_text(json.dumps(data_manifest, indent=2) + "\n", encoding="utf-8")
    wandb.config.update(
        {
            "cbg/defense_source_indices": len(split_indices.defense),
            "cbg/attack_source_indices": len(split_indices.attack),
        }
    )

    trainer = CbgTrainer(
        model=model,
        reference_model=reference_model,
        data_manifest=data_manifest,
        args=training_args,
        cbg_args=cbg_args,
        train_dataset=defense_dataset,
        tokenizer=tokenizer,
        data_collator=collate_cbg_defense_batch,
    )
    trainer.train(resume_from_checkpoint=cbg_args.cbg_resume_from_checkpoint)
    wandb.log({"Output dir": training_args.output_dir})
    wandb.finish()

def main():
    parser = argparse.ArgumentParser(description="Simple config path parser")
    parser.add_argument('--config', type=str, default='config/train.yaml', help='Path to YAML config file')
    parser.add_argument('--method', choices=('seam', 'cbg'), default=None, help='Defense method')
    config_path, remaining_args = parser.parse_known_args()

    yaml_config = load_yaml_config(config_path.config)
    configured_method = yaml_config.get('CbgArguments', {}).get(
        'method',
        yaml_config.get('UnlearnArguments', {}).get('method', 'seam'),
    )
    selected_method = config_path.method or configured_method
    if selected_method == 'cbg':
        return run_cbg(yaml_config, remaining_args, config_path.config, selected_method)
    if selected_method != 'seam':
        raise ValueError(f"Unsupported defense method: {selected_method!r}")

    training_yaml_args = yaml_config.get('TrainingArguments', {})
    unlearn_yaml_args = yaml_config.get('UnlearnArguments', {})
    yaml_defaults = {**training_yaml_args, **unlearn_yaml_args}

    hf_parser = MyParser((MyTrainingArguments, SeamArguments))
    training_args, seam_args = hf_parser.parse_args_into_dataclasses(
        args=remaining_args,
        default_values=yaml_defaults,
    )

    wandb.init(project=seam_args.wandb_project_name, name=seam_args.wandb_run_name, group="train")
    print(f"Using config from: {config_path.config}")
    wandb.config.update(asdict(training_args)) 
    wandb.config.update(asdict(seam_args))

    try:
        tokenizer = AutoTokenizer.from_pretrained(seam_args.model_name)
    except Exception as e:
        tokenizer = AutoTokenizer.from_pretrained(seam_args.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(seam_args.model_name, torch_dtype=torch.bfloat16)
    # model.config.use_cache = False

    harmful_dataset, harmless_dataset, test_dataset = construct_beavertails_dataset(tokenizer, attack=seam_args.attack_dataset, defence_size=seam_args.defense_size)
    benign_dataset = construct_alpaca_dataset(tokenizer, retain_size=seam_args.defense_size)

    def collate_fn(batch):
        def pad_id(tensor):
            tensor[tensor == tokenizer.pad_token_id] = -100
            return tensor
        return {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'labels': torch.stack([pad_id(item['input_ids']) for item in batch]),
        }

    trainer = SeamTrainer(
        model=model,
        args=training_args,
        seam_args=seam_args,
        train_dataset=harmful_dataset,
        tokenizer=tokenizer,
        align_dataset=harmless_dataset,
        benign_dataset=benign_dataset,
        data_collator=collate_fn,
    )
    trainer.train()
    wandb.log({"Output dir": training_args.output_dir})
    wandb.finish()

if __name__ == "__main__":
    main()
