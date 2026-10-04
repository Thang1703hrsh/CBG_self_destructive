"""Read-only CPU audit of the current SEAM training code and source data."""
import ast
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
source = ROOT / "src/core/trainer.py"
tree = ast.parse(source.read_text(encoding="utf-8"))
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SeamTrainer")
namespace = {"torch": torch, "contextmanager": contextmanager}
for name in ("masked_token_ce_loss", "temporary_parameter_update"):
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)

torch.manual_seed(42)
logits = torch.randn(1, 4, 12, requires_grad=True)
labels = torch.tensor([[1, 2, 3, 4]])
mask = torch.tensor([[0., 0., 1., 1.]])
loss = namespace["masked_token_ce_loss"](None, logits, labels, mask)
loss.backward()
old_boundary = float(logits.grad[0, 1].norm())
fixed_logits = logits.detach().clone().requires_grad_(True)
fixed_targets = labels[:, 1:].masked_fill(~mask[:, 1:].bool(), -100)
fixed_loss = torch.nn.functional.cross_entropy(fixed_logits[:, :-1].reshape(-1, 12), fixed_targets.reshape(-1))
fixed_loss.backward()
empty_loss = namespace["masked_token_ce_loss"](None, logits.detach(), labels, torch.zeros_like(mask))

tokenizer = AutoTokenizer.from_pretrained(OUT / "checkpoint-metadata", local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
rows = json.loads((ROOT / "defense.json").read_text(encoding="utf-8"))
harmful = tokenizer([f"Question: {r['prompt']}\nAnswer:\n{r['response']}" for r in rows],
                    padding="max_length", truncation=True, max_length=256, return_tensors="pt")
refusal = tokenizer([r["refusal"] for r in rows], padding="max_length", truncation=True,
                    max_length=256, return_tensors="pt")
selected = harmful.input_ids != refusal.input_ids
target_valid = selected[:, 1:] & refusal.attention_mask[:, 1:].bool() & (refusal.input_ids[:, 1:] != tokenizer.eos_token_id)
lost = target_valid & ~selected[:, :-1]
first_indices = target_valid.long().argmax(dim=1)
first_lost = lost.gather(1, first_indices[:, None]).squeeze(1) & target_valid.any(dim=1)

# Use the exact collator body without importing training dependencies.
train_tree = ast.parse((ROOT / "src/train.py").read_text(encoding="utf-8"))
main = next(n for n in train_tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
collator = next(n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == "collate_fn")
collator_ns = {"torch": torch, "tokenizer": SimpleNamespace(pad_token_id=2)}
exec(compile(ast.Module(body=[collator], type_ignores=[]), "src/train.py", "exec"), collator_ns)
example = {"input_ids": torch.tensor([1, 7, 2, 2]), "attention_mask": torch.tensor([1, 1, 0, 0])}
original = example["input_ids"].clone()
collator_ns["collate_fn"]([example])

# Illustrative numerical test, not a measurement on the 7B model's gradients.
perturbation = {}
for dtype in (torch.bfloat16, torch.float32):
    model = torch.nn.Linear(1024, 1024, bias=False, dtype=dtype)
    owner = SimpleNamespace(grad_device1="cpu")
    before = model.weight.detach().clone()
    g1 = {"weight": torch.ones_like(before)}
    g2 = {"weight": torch.zeros_like(before)}
    norm = torch.tensor(1024., dtype=dtype)
    with namespace["temporary_parameter_update"](owner, model, g1, g2, norm, norm, torch.tensor(0., dtype=dtype), 0.001):
        changed = int((model.weight != before).sum())
    perturbation[str(dtype)] = {"parameters": before.numel(), "changed": changed,
                               "restored_exactly": bool(torch.equal(model.weight, before))}

trainer = json.loads((OUT / "checkpoint-metadata/trainer_state.json").read_text())
logs = [r for r in trainer["log_history"] if "loss" in r]
report = {
    "scope": "Current source, exact cached checkpoint metadata, local defense.json; no GPU training",
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "boundary_test": {"actual_method_gradient": old_boundary, "target_only_mask_gradient": float(fixed_logits.grad[0, 1].norm()),
                      "zero_mask_loss_is_nan": bool(torch.isnan(empty_loss))},
    "defense": {"rows": len(rows), "valid_refusal_targets": int(target_valid.sum()),
                "targets_with_zeroed_predicting_logits": int(lost.sum()),
                "rows_with_lost_targets": int(lost.any(dim=1).sum()),
                "rows_with_first_selected_refusal_target_lost": int(first_lost.sum()),
                "rows_without_selected_refusal_targets": int((~target_valid.any(dim=1)).sum())},
    "collator": {"mutates_sample_input_ids": not torch.equal(example["input_ids"], original)},
    "perturbation_toy_test": perturbation,
    "checkpoint_training": {"global_step": trainer["global_step"], "epoch": trainer["epoch"],
                            "batch": trainer["train_batch_size"], "first_5_mean_logged_loss": sum(x["loss"] for x in logs[:5])/5,
                            "last_5_mean_logged_loss": sum(x["loss"] for x in logs[-5:])/5,
                            "note": "Logged loss excludes the self-destruct gradient contribution; no component logs in checkpoint metadata."},
}
(OUT / "training-audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
