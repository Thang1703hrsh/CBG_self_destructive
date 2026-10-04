"""Adapter around the repository's original CTRAPTrainer.

The original train.py has a hard-coded local token file and UltraChat path.
This adapter supplies equivalent 5k/5k/5k datasets while leaving the
method's three-gradient training_step in CTRAP/trainer.py unchanged. Long
prompts are shortened so each pair has answer tokens within 256 tokens.
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

sys.path.insert(0, "/root/seam/CTRAP")
from trainer import CTRAPTrainer  # noqa: E402
from utils import PROMPT_DICT  # noqa: E402

MODEL = "meta-llama/Llama-2-7b-chat-hf"
IGNORE_INDEX = -100


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, tokenizer):
        self.items = []
        self.truncated_prompts = 0
        max_length = tokenizer.model_max_length
        for prompt, answer in pairs:
            source = PROMPT_DICT["prompt_no_input"].format(instruction=prompt)
            source_ids = tokenizer(source, add_special_tokens=True).input_ids
            if len(source_ids) > max_length - 32:
                prompt_tokens = tokenizer(prompt, add_special_tokens=False).input_ids
                keep = min(len(prompt_tokens), max_length - 96)
                while keep > 0:
                    short_prompt = tokenizer.decode(prompt_tokens[:keep], skip_special_tokens=True)
                    source = PROMPT_DICT["prompt_no_input"].format(instruction=short_prompt)
                    source_ids = tokenizer(source, add_special_tokens=True).input_ids
                    if len(source_ids) <= max_length - 32:
                        break
                    keep -= 8
                self.truncated_prompts += 1
            if len(source_ids) > max_length - 32:
                raise ValueError("Could not fit CTRAP prompt and answer within 256 tokens")
            target_ids = tokenizer(
                answer + tokenizer.eos_token, add_special_tokens=False,
            ).input_ids[:max_length - len(source_ids)]
            if not target_ids:
                raise ValueError("CTRAP pair has no trainable answer tokens")
            ids = source_ids + target_ids
            labels = [IGNORE_INDEX] * len(source_ids) + target_ids
            self.items.append({
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })
        if len(self.items) != 5000:
            raise ValueError(f"Expected 5,000 usable pairs, got {len(self.items)}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collator(batch, pad_id):
    ids = torch.nn.utils.rnn.pad_sequence(
        [row["input_ids"] for row in batch], batch_first=True, padding_value=pad_id,
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        [row["labels"] for row in batch], batch_first=True, padding_value=IGNORE_INDEX,
    )
    return {"input_ids": ids, "labels": labels, "attention_mask": ids.ne(pad_id)}


def load_pairs(path: Path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    if len(rows) != 5000:
        raise ValueError(f"CTRAP defense split must be exactly 5,000 rows, got {len(rows)}")
    safe, harmful = [], []
    for row in rows:
        refusal = row["refusal"]
        if "\nAnswer: " not in refusal:
            raise ValueError("CTRAP original data format requires '\\nAnswer: ' in refusals")
        prefix, answer = refusal.split("\nAnswer: ", 1)
        question = prefix.replace("Question: ", "", 1).strip()
        if not question or not answer.strip():
            raise ValueError("Empty CTRAP safe pair")
        safe.append((question, answer.strip()))
        harmful.append((row["prompt"].strip(), row["response"].strip()))
    return safe, harmful


def helpful_pairs():
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
    pairs = []
    for row in dataset:
        messages = row["messages"]
        if len(messages) >= 2 and messages[0]["content"] and messages[1]["content"]:
            pairs.append((messages[0]["content"], messages[1]["content"]))
        if len(pairs) == 5000:
            return pairs
    raise ValueError(f"Need 5,000 UltraChat first-turn pairs, got {len(pairs)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--defense", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--direct-gpu-load", action="store_true",
                        help="Stream model weights to CUDA to reduce CPU RAM during loading")
    parser.add_argument("--smoke", action="store_true", help="Validate two steps and skip model export")
    parser.add_argument("--smoke-export", action="store_true",
                        help="In smoke mode, also exercise full checkpoint export")
    args = parser.parse_args()
    if args.max_steps != -1 and args.max_steps <= 0:
        raise ValueError("--max-steps must be -1 or a positive integer")
    if args.smoke and args.max_steps != 2:
        raise ValueError("CTRAP smoke mode requires --max-steps 2")
    if args.smoke_export and not args.smoke:
        raise ValueError("--smoke-export requires --smoke")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    safe, harmful = load_pairs(args.defense)
    helpful = helpful_pairs()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, model_max_length=256, padding_side="right")
    new_tokens = 0
    if tokenizer.pad_token is None:
        new_tokens = tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    safe_dataset = PairDataset(safe, tokenizer)
    harmful_dataset = PairDataset(harmful, tokenizer)
    helpful_dataset = PairDataset(helpful, tokenizer)
    model_load_kwargs = {"torch_dtype": torch.bfloat16}
    if args.direct_gpu_load:
        if not torch.cuda.is_available():
            raise RuntimeError("--direct-gpu-load requires CUDA")
        model_load_kwargs.update({"low_cpu_mem_usage": True, "device_map": {"": 0}})
    model = AutoModelForCausalLM.from_pretrained(MODEL, **model_load_kwargs)
    model.resize_token_embeddings(len(tokenizer))
    if new_tokens:
        with torch.no_grad():
            for embedding in (model.get_input_embeddings(), model.get_output_embeddings()):
                if embedding is not None:
                    embedding.weight[-new_tokens:] = embedding.weight[:-new_tokens].mean(
                        dim=0, keepdim=True,
                    )
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=32, lora_alpha=4, target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0, bias="none", task_type="CAUSAL_LM",
    ))
    args.output.mkdir(parents=True, exist_ok=False)
    training_args = TrainingArguments(
        output_dir=str(args.output / "state"), num_train_epochs=20,
        max_steps=args.max_steps,
        per_device_train_batch_size=5, gradient_accumulation_steps=1,
        learning_rate=5e-4, weight_decay=0.1, lr_scheduler_type="constant",
        warmup_ratio=0, bf16=True, tf32=True, optim="adamw_torch",
        logging_steps=1 if args.smoke else 100,
        logging_nan_inf_filter=False,
        save_strategy="no" if args.smoke else "steps",
        save_steps=min(1000, args.max_steps) if args.max_steps > 0 else 1000,
        save_total_limit=2, evaluation_strategy="no",
        report_to="none", seed=args.seed, data_seed=args.seed,
        remove_unused_columns=False,
    )
    training_args.alpha = 0.1
    training_args.lamb = 0.1
    training_args.guide_data_num = 5000
    data_collator = lambda batch: collator(batch, tokenizer.pad_token_id)
    trainer = CTRAPTrainer(
        model=model, args=training_args,
        train_dataset=safe_dataset,
        data_collator=data_collator, tokenizer=tokenizer,
    )
    trainer.init(harmful_dataset, helpful_dataset)
    result = trainer.train()
    if args.max_steps > 0 and trainer.state.global_step != args.max_steps:
        raise RuntimeError(
            f"Expected {args.max_steps} CTRAP steps, got {trainer.state.global_step}"
        )
    losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
    if args.smoke and (len(losses) != 2 or not all(math.isfinite(float(value)) for value in losses)):
        raise RuntimeError(f"CTRAP smoke loss is non-finite or missing: {losses}")
    if not args.smoke or args.smoke_export:
        model.save_pretrained(str(args.output / "adapter"))
        merged = model.merge_and_unload()
        merged.config.use_cache = True
        merged.save_pretrained(str(args.output / "model"), safe_serialization=True)
        tokenizer.save_pretrained(str(args.output / "model"))
        model_dir = args.output / "model"
        if (not (model_dir / "config.json").exists()
                or not (model_dir / "tokenizer_config.json").exists()
                or not list(model_dir.glob("*.safetensors"))):
            raise FileNotFoundError("CTRAP merged model, tokenizer, or safetensors not exported")
    train_metrics = dict(result.metrics)
    train_metrics.update({
        "global_step": trainer.state.global_step,
        "requested_max_steps": args.max_steps,
        "direct_gpu_load": args.direct_gpu_load,
        "safe_usable_pairs": len(safe_dataset),
        "harmful_usable_pairs": len(harmful_dataset),
        "ultrachat_usable_pairs": len(helpful_dataset),
        "safe_truncated_prompts": safe_dataset.truncated_prompts,
        "harmful_truncated_prompts": harmful_dataset.truncated_prompts,
        "ultrachat_truncated_prompts": helpful_dataset.truncated_prompts,
        "step_losses": losses,
        "smoke": args.smoke,
        "smoke_export": args.smoke_export,
    })
    (args.output / "train_metrics.json").write_text(
        json.dumps(train_metrics, indent=2, default=str), encoding="utf-8",
    )


if __name__ == "__main__":
    main()
