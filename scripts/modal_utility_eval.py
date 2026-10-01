"""Modal launcher: HellaSwag + MMLU utility (SEAM protocol) on ONE L40S (48 GB) per model.

SEAM protocol (seam/src/utils/metrics.py::eval_utility): bf16 model, pad = eos, lm-eval zero-shot, batch 24.
Implementation: src/experiments/utility_eval.py (the same code runs on a local GPU).

  modal run scripts/modal_utility_eval.py                                   # base + CBG-HVP (HF repo), in parallel
  modal run scripts/modal_utility_eval.py --limit 50                        # smoke test (~10 min, not reportable)
  modal run scripts/modal_utility_eval.py --models "cbg-hvp=run:hvp-tf32"   # a run on the cbg-v2-7b-output volume
  modal run --detach scripts/modal_utility_eval.py                          # keeps running if the PC shuts down

--models: comma-separated name=source; source is an HF repo id, or run:<run-name> for
<output volume>/<run-name>/train/final. The tokenizer always comes from --tokenizer (the base model).
Results: volume cbg-v2-7b-output at utility/<name>/, copied locally to results/utility/modal/ (not with --detach;
then fetch them with: modal volume get cbg-v2-7b-output utility results/utility/modal).
A model whose utility.json already exists with the same model/tasks/limit is skipped unless --force.
The HF_TOKEN secret needs Llama-2 access and read access to private model repos.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/workspace"
OUTPUT = "/vol/output"
CACHE = "/vol/hf"
GPU = os.environ.get("CBG_UTILITY_GPU", "L40S")   # 7B bf16 = ~14 GB weights; 48 GB is enough
TASKS = "hellaswag,mmlu"   # fixed: only these two benchmarks
BASE = "meta-llama/Llama-2-7b-chat-hf"
DEFAULT_MODELS = f"base={BASE},cbg-hvp=ducthang1703/cbg-hvp-llama2-7b"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(ROOT / "requirements.txt"))
    .env({"PYTHONPATH": REMOTE, "USE_TF": "0", "TRANSFORMERS_NO_TF": "1", "HF_HOME": CACHE,
          "HF_DATASETS_CACHE": CACHE + "/datasets", "TOKENIZERS_PARALLELISM": "false",
          "OMP_NUM_THREADS": "4", "WANDB_MODE": "disabled", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_dir(str(ROOT / "src"), REMOTE + "/src")
)
app = modal.App("cbg-utility-eval")
output_volume = modal.Volume.from_name("cbg-v2-7b-output", create_if_missing=True)
cache_volume = modal.Volume.from_name("cbg-spec-hf", create_if_missing=True)
hf_secret = modal.Secret.from_name("HF_TOKEN")


def _name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value):
        raise ValueError(f"invalid name: {value!r}")
    return value


def _parse_models(spec):
    models = {}
    for item in filter(None, (s.strip() for s in spec.split(","))):
        name, sep, source = item.partition("=")
        if not sep or not source:
            raise ValueError(f"--models entry {item!r} is not name=source")
        models[_name(name)] = source
    return models


# RAM: an fp32 upload of a 7B model is ~27 GB on disk; it is loaded straight to bf16 (~14 GB) -> 48 GiB.
@app.function(image=image, gpu=GPU, cpu=4, memory=48 * 1024, timeout=4 * 3600,
              volumes={OUTPUT: output_volume, CACHE: cache_volume}, secrets=[hf_secret])
def evaluate(name: str, source: str, tokenizer: str, tasks: str, batch_size: int, limit: float, force: bool):
    os.chdir(REMOTE)
    out = Path(OUTPUT) / "utility" / _name(name)
    try:
        if source.startswith("run:"):
            model = Path(OUTPUT) / _name(source[4:]) / "train" / "final"
            if not (model / "config.json").exists():
                raise FileNotFoundError(f"{model} not found on the cbg-v2-7b-output volume")
            model = str(model)
        else:
            model = source
        done = out / "utility.json"
        if done.exists() and not force:
            old = json.loads(done.read_text())
            if (old.get("model"), old.get("tasks"), old.get("limit")) == (model, tasks.split(","), limit or None):
                return dict(name=name, skipped=True, **old)
        out.mkdir(parents=True, exist_ok=True)
        argv = [sys.executable, "-m", "src.experiments.utility_eval", "--model", model, "--tokenizer", tokenizer,
                "--output", str(out), "--tasks", tasks, "--batch-size", str(batch_size)]
        if limit:
            argv += ["--limit", str(limit)]
        started = time.time()
        with open(out / "console.log", "w", encoding="utf-8") as log:
            process = subprocess.Popen(argv, cwd=REMOTE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1)
            for line in process.stdout:
                print(f"[{name}] {line}", end="", flush=True)
                log.write(line)
            status = process.wait()
        output_volume.commit()
        cache_volume.commit()   # model + lm-eval datasets are reused by the next run
        if status:
            raise RuntimeError(f"utility_eval exited with status {status}; see utility/{name}/console.log")
        return dict(name=name, hours=(time.time() - started) / 3600, **json.loads(done.read_text()))
    except Exception as error:   # returned, not raised: no billed retries
        return dict(name=name, status="failed", error=repr(error))


@app.local_entrypoint()
def main(models: str = DEFAULT_MODELS, tokenizer: str = BASE, batch_size: int = 24, limit: float = 0.0,
         force: bool = False, base_name: str = "base"):
    from src.experiments.utility_eval import table
    tasks = TASKS
    chosen = _parse_models(models)
    print(f"GPU {GPU} per model | tasks {tasks} | batch {batch_size}" + (f" | limit {limit} (smoke)" if limit else ""))
    for name, source in chosen.items():
        print(f"  {name}: {source}")
    calls = [(name, source, tokenizer, tasks, batch_size, limit, force) for name, source in chosen.items()]
    results = list(evaluate.starmap(calls))   # one L40S container per model, in parallel

    failed = [r for r in results if r.get("status") == "failed"]
    local = ROOT / "results" / "utility" / "modal" / (f"smoke-{limit:g}" if limit else "full")
    local.mkdir(parents=True, exist_ok=True)
    scores = {}
    for r in results:
        if r.get("status") == "failed":
            print(f"FAILED {r['name']}: {r['error']}")
            continue
        scores[r["name"]] = r["scores"]
        (local / f"{r['name']}.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
        print(f"{r['name']}: " + ("skipped (already done)" if r.get("skipped") else f"{r['hours']:.2f} h"))
    if scores:
        text = table(scores, tasks.split(","), base=base_name)
        (local / "summary.md").write_text(text + "\n", encoding="utf-8")
        print("\n" + text + f"\n\nsaved: {local}")
        if limit:
            print("NOTE: --limit was set; smoke-test numbers, not reportable.")
    if failed:
        raise SystemExit(1)
