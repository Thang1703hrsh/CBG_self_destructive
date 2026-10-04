"""Paper-specified SDD SFT objective, reconstructed because SDD/train is absent.

This is not the authors' LLaMA-Factory training implementation. We use the
Llama-2 chat template, response-only CE, full-model BF16 fine-tuning, 500
optimizer steps, global batch 24, and learning rate 5e-7.
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

MODEL = "meta-llama/Llama-2-7b-chat-hf"
IGNORE_INDEX = -100


class SddDataset(torch.utils.data.Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int = 512, min_pairs: int = 8000):
        self.examples = []
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                prompt = row["prompt"].strip()
                answer = row["completion"].strip()
                if not prompt or not answer:
                    continue
                def prompt_ids(text):
                    return tokenizer.apply_chat_template(
                        [{"role": "user", "content": text}],
                        tokenize=True, add_generation_prompt=True,
                    )

                source_ids = prompt_ids(prompt)
                if len(source_ids) > max_length - 32:
                    short_prompt = tokenizer.decode(
                        tokenizer(prompt, add_special_tokens=False).input_ids[:max_length - 96],
                        skip_special_tokens=True,
                    )
                    source_ids = prompt_ids(short_prompt)
                if len(source_ids) > max_length - 32:
                    raise ValueError("Could not fit SDD prompt into the 512-token context")
                target_ids = tokenizer(
                    answer + tokenizer.eos_token, add_special_tokens=False,
                ).input_ids[:max_length - len(source_ids)]
                if not target_ids:
                    raise ValueError("SDD pair has no trainable answer tokens")
                self.examples.append({
                    "input_ids": source_ids + target_ids,
                    "labels": [IGNORE_INDEX] * len(source_ids) + target_ids,
                })
        if len(self.examples) < min_pairs:
            raise ValueError(f"SDD requires {min_pairs} usable pairs; got {len(self.examples)}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def collator(batch, pad_id):
    width = max(len(row["input_ids"]) for row in batch)
    ids = [row["input_ids"] + [pad_id] * (width - len(row["input_ids"])) for row in batch]
    labels = [row["labels"] + [IGNORE_INDEX] * (width - len(row["labels"])) for row in batch]
    masks = [[1] * len(row["input_ids"]) + [0] * (width - len(row["input_ids"])) for row in batch]
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--smoke", action="store_true", help="Validate two steps and skip the large checkpoint")
    parser.add_argument("--smoke-export", action="store_true",
                        help="In smoke mode, also exercise full checkpoint export")
    args = parser.parse_args()
    if args.max_steps <= 0 or (args.smoke and args.max_steps != 2):
        raise ValueError("Smoke mode requires exactly 2 positive optimizer steps")
    if args.smoke_export and not args.smoke:
        raise ValueError("--smoke-export requires --smoke")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = SddDataset(args.pairs, tokenizer, min_pairs=48 if args.smoke else 8000)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    model.config.use_cache = False
    training_args = TrainingArguments(
        output_dir=str(args.output),
        max_steps=args.max_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=12,
        learning_rate=5e-7,
        lr_scheduler_type="constant",
        warmup_steps=0,
        bf16=True,
        optim="adamw_bnb_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        logging_nan_inf_filter=False,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=dataset,
        data_collator=lambda batch: collator(batch, tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    result = trainer.train()
    if result.global_step != args.max_steps:
        raise RuntimeError(f"Expected {args.max_steps} SDD steps, got {result.global_step}")
    losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
    if args.smoke and (len(losses) != 2 or not all(math.isfinite(float(value)) for value in losses)):
        raise RuntimeError(f"SDD smoke loss is non-finite or missing: {losses}")
    if not args.smoke or args.smoke_export:
        trainer.save_model(str(args.output / "model"))
        tokenizer.save_pretrained(str(args.output / "model"))
        model_dir = args.output / "model"
        if (not (model_dir / "config.json").exists()
                or not (model_dir / "tokenizer_config.json").exists()
                or not list(model_dir.glob("*.safetensors"))):
            raise FileNotFoundError("SDD model, tokenizer, or safetensors not exported")
    train_metrics = dict(result.metrics)
    train_metrics.update({"global_step": trainer.state.global_step, "usable_pairs": len(dataset),
                          "step_losses": losses, "smoke": args.smoke,
                          "smoke_export": args.smoke_export})
    (args.output / "train_metrics.json").write_text(
        json.dumps(train_metrics, indent=2, default=str), encoding="utf-8",
    )


if __name__ == "__main__":
    main()
