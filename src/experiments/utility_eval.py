"""Utility benchmarks (HellaSwag, MMLU, ...) with the SEAM protocol, on one GPU.

SEAM (seam/src/eval.py + seam/src/utils/metrics.py::eval_utility): bf16 model, tokenizer with pad = eos,
lm_eval.simple_evaluate on an HFLM wrapper, default (zero-shot) prompts, batch size 24. Same here.

  python -m src.experiments.utility_eval --model results/.../final --output results/utility/cbg-hvp
  python -m src.experiments.utility_eval --model ducthang1703/cbg-hvp-llama2-7b --output results/utility/cbg-hvp
  python -m src.experiments.utility_eval --model meta-llama/Llama-2-7b-chat-hf --output results/utility/base

--model is a local folder or a Hugging Face repo (private repos need HF_TOKEN). Results go to
<output>/utility.json (headline numbers) and <output>/lm_eval_results.json (full lm-eval output).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

# The number reported per task. HellaSwag: SEAM reports lm-eval "acc"; acc_norm is stored too.
METRICS = ("acc,none", "acc_norm,none", "acc_stderr,none", "acc_norm_stderr,none")


def headline(results, tasks):
    out = {}
    for task in tasks:
        row = results.get(task, {})
        out[task] = {k.split(",")[0]: row[k] for k in METRICS if k in row}
    return out


def table(scores, tasks, base=None):
    """Markdown table; ``scores``: {name: utility.json["scores"]}; adds a delta row per model vs ``base``."""
    cols = [(t, m) for t in tasks for m in (("acc", "acc_norm") if t == "hellaswag" else ("acc",))]
    cell = lambda s, t, m: f"{s[t][m]:.4f} +/- {s[t].get(m + '_stderr', 0):.4f}" if m in s.get(t, {}) else ""
    lines = ["| model | " + " | ".join(f"{t} {m}" for t, m in cols) + " |", "|---|" + "---|" * len(cols)]
    lines += [f"| {name} | " + " | ".join(cell(s, t, m) for t, m in cols) + " |" for name, s in scores.items()]
    if base in scores:
        for name, s in scores.items():
            if name != base:
                lines.append(f"| delta {name} - {base} | " + " | ".join(
                    f"{s[t][m] - scores[base][t][m]:+.4f}" if m in s.get(t, {}) and m in scores[base].get(t, {})
                    else "" for t, m in cols) + " |")
    return "\n".join(lines)


def evaluate(model_path, tasks, *, tokenizer_path=None, batch_size=24, device="cuda:0", limit=None):
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path)
    tokenizer.pad_token = tokenizer.eos_token   # SEAM eval.py
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device).eval()
    results = lm_eval.simple_evaluate(model=HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size),
                                      tasks=list(tasks), batch_size=batch_size, limit=limit)   # lm-eval default seeds, as SEAM
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="local model folder or HF repo id")
    parser.add_argument("--tokenizer", default=None, help="defaults to --model")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", default="hellaswag,mmlu")
    parser.add_argument("--batch-size", type=int, default=24, help="SEAM uses 24")
    parser.add_argument("--limit", type=float, default=None, help="examples per task, for a quick smoke test only")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    tasks = [t for t in args.tasks.split(",") if t]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    results = evaluate(args.model, tasks, tokenizer_path=args.tokenizer, batch_size=args.batch_size,
                       device=args.device, limit=args.limit)
    summary = dict(model=args.model, tasks=tasks, batch_size=args.batch_size, limit=args.limit, dtype="bfloat16",
                   num_fewshot=0, seconds=round(time.time() - start, 1),
                   scores=headline(results["results"], tasks))
    (out / "utility.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "lm_eval_results.json").write_text(
        json.dumps({k: results.get(k) for k in ("results", "groups", "n-shot", "versions")}, indent=2, default=str),
        encoding="utf-8")
    print(json.dumps(summary["scores"], indent=2))
    if args.limit:
        print("NOTE: --limit was set; these numbers are a smoke test, not reportable.")


if __name__ == "__main__":
    main()
