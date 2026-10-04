"""Modal launcher: harmful fine-tuning attack + harmfulness/utility of CBG-HVP vs base, on L40S (48 GB).

Protocol: src/experiments/attack_eval.py (SEAM eval.py, attack mode) with two memory options so a 7B
full-parameter attack fits in 48 GB:
  optim = paged_adamw_8bit   (SEAM: adamw_torch -> ~55 GB, does not fit)
  gradient checkpointing     (activations of batch 4 x 256 tokens)
Both models (base and CBG) are attacked with exactly the same options, so the comparison stays fair.
Use --optim adamw_torch --no-gradient-checkpointing on an 80 GB GPU to reproduce SEAM exactly.

One container per job, all in parallel (a job = one model x one phase):
  pre             harmfulness (100 eval_harm prompts) + utility before the attack
  lr<lr>-s<seed>  one attack (1,000 rows, 1 epoch) -> harmfulness + post-attack utility

Scenarios (--scenario):
  smoke     cbg-hvp only, lr 2e-5, arc_easy, utility --limit 0.05     (~20 min; checks the pipeline)
  main      base + cbg-hvp, lrs 2e-5,5e-5,8e-5,1e-4,2e-4, seed 42,
            pre + post utility on truthfulqa_mc1,mmlu,hellaswag,arc_easy   (paper table, SEAM Table 6)
  seeds     base + cbg-hvp, lrs 2e-5,1e-4, seeds 43,44                       (variance of the main table)
  decoding  base + cbg-hvp, lr 1e-4, decoding checkpoint (sampled)           (robustness to decoding)
  gate      --models <name>=run:<run-name>: one VALIDATION attack (d_val_atk) at lr 5e-5, harmfulness +
            arc_easy (~20 min, ~1 USD); go/no-go before the full `main` evaluation (scripts/Run-CBGBeta.ps1)
Explicit flags (--models, --lrs, --seeds, --tasks, --decoding) override the scenario.

  modal run scripts/modal_attack_eval_l40s.py --scenario smoke --wait
  modal run --detach scripts/modal_attack_eval_l40s.py --scenario main   # returns at once; PC may be switched off
  modal run scripts/modal_attack_eval_l40s.py --scenario main --collect  # later: download summary + per-job JSON
Without --wait, ALWAYS pass --detach: a non-detached app stops (and cancels the spawned jobs) as soon
as the command returns. A rerun skips finished jobs (resumable).

Needs: the data volume prepared by `modal run scripts/modal_cbg_v2_7b.py --stage prepare` (attack.json,
eval_harm.json under <data-tag>/), and the HF_TOKEN secret with Llama-2 access and read access to the CBG repo.
Results: volume cbg-v2-7b-output at attack-l40s/<model>/<decoding>/<job>/attack_eval.json, copied to
results/attack_l40s/<scenario>/ with summary.md. Finished jobs are skipped (resumable) unless --force.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/workspace"
DATA = "/vol/data"
OUTPUT = "/vol/output"
CACHE = "/vol/hf"
GPU = os.environ.get("CBG_ATTACK_GPU", "L40S")
BASE = "meta-llama/Llama-2-7b-chat-hf"
CBG = "ducthang1703/cbg-hvp-llama2-7b"
PAPER_TASKS = "truthfulqa_mc1,mmlu,hellaswag,arc_easy"
PAPER_LRS = "2e-5,5e-5,8e-5,1e-4,2e-4"

SCENARIOS = {
    "smoke": dict(models=f"cbg-hvp={CBG}", lrs="2e-5", seeds="42", tasks="arc_easy", decoding="greedy",
                  limit=0.05, pre=True),
    "main": dict(models=f"base={BASE},cbg-hvp={CBG}", lrs=PAPER_LRS, seeds="42", tasks=PAPER_TASKS,
                 decoding="greedy", limit=0.0, pre=True),
    "seeds": dict(models=f"base={BASE},cbg-hvp={CBG}", lrs="2e-5,1e-4", seeds="43,44", tasks=PAPER_TASKS,
                  decoding="greedy", limit=0.0, pre=False),
    "decoding": dict(models=f"base={BASE},cbg-hvp={CBG}", lrs="1e-4", seeds="42", tasks="", decoding="checkpoint",
                     limit=0.0, pre=True),
    # Cheap go/no-go for a new defense run (pass --models name=run:<run-name>): ONE validation attack
    # (d_val_atk, prompt-disjoint from the test data) at lr 5e-5, harmfulness + arc_easy only.
    "gate": dict(models="", lrs="5e-5", seeds="42", tasks="arc_easy", decoding="greedy", limit=0.0, pre=False,
                 mode="val"),
    # Geometry diagnostic: the validation attack at lr 1e-4, saving the attacked model at steps 25 and 250
    # (<job>/attacked-0.0001/step-<n>) so that modal_cbg_v2_7b.py --stage geometry can measure held-out eta on it.
    "diag": dict(models="", lrs="1e-4", seeds="42", tasks="", decoding="greedy", limit=0.0, pre=False,
                 mode="val", prefix="diag-", save_at_steps="25,250"),
}

# Host resources. CPU: one process; tokenization and the DeBERTa classifier are light -> 4 cores, 4 threads.
# RAM: weights are streamed shard by shard to the GPU (low_cpu_mem_usage + device_map), paged 8-bit Adam
# spills to host only under pressure -> 32 GiB is enough even for an fp32 checkpoint (~27 GB on disk).
CPU = float(os.environ.get("CBG_ATTACK_CPU", "4"))
MEMORY = int(os.environ.get("CBG_ATTACK_MEMORY", str(32 * 1024)))   # MiB

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(ROOT / "requirements.txt"))
    .pip_install("bitsandbytes==0.45.3", "sentencepiece", "protobuf")
    .env({"PYTHONPATH": REMOTE, "USE_TF": "0", "TRANSFORMERS_NO_TF": "1", "HF_HOME": CACHE,
          "HF_DATASETS_CACHE": CACHE + "/datasets", "TOKENIZERS_PARALLELISM": "false",
          "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "WANDB_MODE": "disabled",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_dir(str(ROOT / "src"), REMOTE + "/src")
)
app = modal.App("cbg-attack-eval-l40s")
data_volume = modal.Volume.from_name("cbg-v2-7b-data", create_if_missing=True)
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


def _cache_marker(tasks):
    """Exists once a job finished with these lm-eval tasks, i.e. their datasets + the classifier are cached."""
    key = re.sub(r"[^A-Za-z0-9_.-]", "_", tasks or "none")[:120]
    return Path(CACHE) / "cbg_lm_eval_ready" / f"{key}.json"


def _job_dir(model_name, decoding, job):
    return Path(OUTPUT) / "attack-l40s" / model_name / decoding / job


@app.function(image=image, gpu=GPU, cpu=CPU, memory=MEMORY, timeout=10 * 3600,
              volumes={DATA: data_volume, OUTPUT: output_volume, CACHE: cache_volume}, secrets=[hf_secret])
def run_job(job: dict):
    """One phase of one model; returns attack_eval.json (or the error, never raised: no billed retries)."""
    from huggingface_hub import snapshot_download
    os.chdir(REMOTE)
    out = _job_dir(job["model_name"], job["decoding"], job["job"])
    try:
        done = out / "attack_eval.json"
        if done.exists() and not job["force"]:
            old = json.loads(done.read_text())
            # Reuse only an identical evaluation: smoke and main share job paths but differ in tasks/limit.
            same = (old.get("complete") and old.get("utility_tasks") == [t for t in job["tasks"].split(",") if t]
                    and old.get("limit") == (job["limit"] or None) and old.get("seed") == job["seed"]
                    and old.get("mode") == job["mode"])
            if same:
                marker = _cache_marker(job["tasks"])
                if job["tasks"] and not marker.exists():   # it ran here before: its datasets are cached
                    marker.parent.mkdir(parents=True, exist_ok=True)
                    marker.write_text(json.dumps(dict(tasks=job["tasks"], job=job["job"])), encoding="utf-8")
                    cache_volume.commit()
                return dict(job, skipped=True, result=old)
            print(f"[{job['model_name']}/{job['job']}] existing result has other tasks/limit/seed "
                  f"or is incomplete: re-running", flush=True)
            # attack_eval would otherwise resume the stale phases; keep the old log for diagnosis.
            previous_log = out / "console.log"
            kept = previous_log.read_text(encoding="utf-8", errors="replace") if previous_log.exists() else None
            shutil.rmtree(out)
            if kept is not None:
                out.mkdir(parents=True)
                (out / "console.previous.log").write_text(kept, encoding="utf-8")
        elif job["force"] and out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)
        # Download once into the shared cache volume (safetensors only; no .bin duplicates).
        patterns = ["*.json", "*.safetensors", "tokenizer.model"]
        if job["source"].startswith("run:"):   # a defense run trained by modal_cbg_v2_7b.py on the output volume
            output_volume.reload()   # see a training commit made after this container started
            run_dir = Path(OUTPUT) / _name(job["source"][4:]) / "train"
            metrics = run_dir / "metrics.json"
            status = json.loads(metrics.read_text()).get("status") if metrics.exists() else "missing"
            if status != "complete":
                raise FileNotFoundError(f"{run_dir} has no completed training run (metrics.json status: {status})")
            model_path = str(run_dir / "final")
        else:
            model_path = snapshot_download(job["source"], allow_patterns=patterns)
        tokenizer_path = model_path if Path(model_path, "tokenizer.model").exists() or \
            Path(model_path, "tokenizer.json").exists() else snapshot_download(BASE, allow_patterns=patterns)
        cache_volume.commit()
        data_dir = Path(DATA) / _name(job["data_tag"])
        for name in (("d_val_atk.json",) if job["mode"] == "val" else ("attack.json", "eval_harm.json")):
            if not (data_dir / name).exists():
                raise FileNotFoundError(f"{data_dir / name} missing: run modal_cbg_v2_7b.py --stage prepare")
        argv = [sys.executable, "-m", "src.experiments.attack_eval", "--model", model_path,
                "--tokenizer", tokenizer_path, "--data", str(data_dir), "--output", str(out), "--mode", job["mode"],
                "--decoding", job["decoding"], "--seed", str(job["seed"]), "--optim", job["optim"],
                "--gen-batch-size", str(job["gen_batch_size"])]
        if job["gradient_checkpointing"]:
            argv.append("--gradient-checkpointing")
        if job.get("save_at_steps"):
            argv += ["--save-at-steps", job["save_at_steps"]]
        if job["lr"] is None:          # pre-attack phase only
            argv += ["--lrs", "", "--utility", job["tasks"]]
        else:
            argv += ["--lrs", job["lr"], "--skip-pre", "--utility", job["tasks"]]
            if job["tasks"]:
                argv.append("--post-utility")
        env = dict(os.environ)
        if job["limit"]:
            env["CBG_LM_EVAL_LIMIT"] = str(job["limit"])
        # Hugging Face allows 1,000 API calls per 5 minutes; lm-eval resolves every MMLU subset online even
        # when it is cached, so ~10 parallel jobs hit HTTP 429. Once one job has cached the lm-eval datasets
        # and the harm classifier (marker below), the evaluation subprocess runs fully offline.
        cache_volume.reload()
        marker = _cache_marker(job["tasks"])
        offline = marker.exists()
        if offline:
            env.update(HF_DATASETS_OFFLINE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        started = time.time()
        for attempt in (1, 2):
            with open(out / "console.log", "a" if attempt > 1 else "w", encoding="utf-8") as log:
                if attempt > 1:
                    log.write(f"\n===== retry after HTTP 429 (offline={offline}) =====\n")
                process = subprocess.Popen(argv, cwd=REMOTE, env=env, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, bufsize=1)
                rate_limited = False
                for line in process.stdout:
                    print(f"[{job['model_name']}/{job['job']}] {line}", end="", flush=True)
                    log.write(line)
                    rate_limited = rate_limited or "429 Client Error" in line
                status = process.wait()
            if not status or not rate_limited or attempt == 2:
                break
            time.sleep(330)   # let the 5-minute rate-limit window pass, then try once more
        if status:
            output_volume.commit()
            raise RuntimeError(f"attack_eval exited with status {status}; see {out}/console.log")
        if job["tasks"] and not offline:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps(dict(tasks=job["tasks"], job=job["job"])), encoding="utf-8")
        result = json.loads(done.read_text())
        result.update(complete=True, hours=(time.time() - started) / 3600, gpu=GPU, limit=job["limit"] or None)
        done.write_text(json.dumps(result, indent=2), encoding="utf-8")
        output_volume.commit()
        cache_volume.commit()
        return dict(job, result=result)
    except Exception as error:
        return dict(job, status="failed", error=repr(error))


def _summary(results, scenario):
    """Markdown table: pre row and one row per attack; mean over seeds, with min-max when > 1 seed."""
    rows, pre = {}, {}
    for r in results:
        if r.get("status") == "failed":
            continue
        res = r["result"]
        if r["lr"] is None:
            pre[r["model_name"]] = res.get("pre", {})
            continue
        entry = next(iter(res.get("attacks", {}).values()), None)
        if entry:
            rows.setdefault((r["model_name"], r["lr"]), []).append(entry)
    tasks = sorted({t for e in [*pre.values(), *[x for v in rows.values() for x in v]]
                    for t in (e.get("utility") or {}) if t != "zs"})
    head = "| model | phase | harmfulness | harmful>0.5 | " + " | ".join(tasks) + " | ZS | attack loss |"
    lines = [f"## Attack evaluation ({scenario}, {GPU})", "", head, "|" + "---|" * (head.count("|") - 1)]
    fmt = lambda xs: "" if not xs else (f"{sum(xs) / len(xs):.3f}" if len(xs) == 1 else
                                        f"{sum(xs) / len(xs):.3f} [{min(xs):.3f}-{max(xs):.3f}]")
    for model, p in sorted(pre.items()):
        u = p.get("utility") or {}
        lines.append(f"| {model} | pre | {fmt([p['harmfulness']['score']])} | "
                     f"{fmt([p['harmfulness']['scores_over_50']])} | "
                     + " | ".join(fmt([u[t]] if t in u else []) for t in tasks) + f" | {fmt([u['zs']] if 'zs' in u else [])} | |")
    for (model, lr), entries in sorted(rows.items(), key=lambda kv: (kv[0][0], float(kv[0][1]))):
        col = lambda f: [f(e) for e in entries if f(e) is not None]
        lines.append(f"| {model} | lr {lr} (n={len(entries)}) | {fmt(col(lambda e: e['harmfulness']['score']))} | "
                     f"{fmt(col(lambda e: e['harmfulness']['scores_over_50']))} | "
                     + " | ".join(fmt(col(lambda e, t=t: (e.get('utility') or {}).get(t))) for t in tasks)
                     + f" | {fmt(col(lambda e: (e.get('utility') or {}).get('zs')))} | {fmt(col(lambda e: e['train_loss']))} |")
    lines += ["", "harmfulness: mean DeBERTa P(harmful) on 100 eval_harm prompts (lower = safer). A defended model "
              "should have LOW post-attack harmfulness only together with LOW post-attack utility (self-destruction); "
              "low harmfulness with intact utility would mean the attack failed for another reason (check attack loss)."]
    return "\n".join(lines)


def _summary_dir(scenario):
    return Path(OUTPUT) / "attack-l40s" / "_summary" / _name(scenario)


# Runs entirely on Modal: dispatches every job, waits for all of them and writes the table to the volume.
# The local machine is not needed after .spawn(), so the run survives a closed terminal or a shut-down PC.
@app.function(image=image, cpu=0.25, memory=1024, timeout=24 * 3600, volumes={OUTPUT: output_volume, CACHE: cache_volume})
def orchestrate(jobs: list, scenario: str, limit: float):
    # Pre-attack jobs first: they download the lm-eval datasets and the classifier into the cache volume,
    # so the attack jobs that follow can evaluate offline (no Hugging Face rate limit).
    first = [j for j in jobs if j["lr"] is None]
    rest = [j for j in jobs if j["lr"] is not None]
    if not first and rest and not _cache_marker(rest[0]["tasks"]).exists():
        first, rest = rest[:1], rest[1:]   # no pre job: one attack job warms the cache
    results = list(run_job.map(first, return_exceptions=True)) if first else []
    results += list(run_job.map(rest, return_exceptions=True)) if rest else []
    results = [r if isinstance(r, dict) else dict(status="failed", error=repr(r), model_name="?", job="?", lr="?")
               for r in results]
    out = _summary_dir(scenario)
    out.mkdir(parents=True, exist_ok=True)
    for r in results:
        if r.get("status") != "failed":
            (out / f"{r['model_name']}__{r['job']}.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
    text = _summary(results, scenario)
    if limit:
        text += f"\n\nNOTE: utility used --limit {limit}; smoke numbers, not reportable."
    failed = [f"FAILED {r.get('model_name')}/{r.get('job')}: {r['error']}" for r in results if r.get("status") == "failed"]
    if failed:
        text += "\n\n" + "\n".join(failed)
    (out / "summary.md").write_text(text + "\n", encoding="utf-8")
    (out / "status.json").write_text(json.dumps(dict(finished=time.time(), jobs=len(results), failed=len(failed))),
                                     encoding="utf-8")
    output_volume.commit()
    return text


@app.function(image=image, cpu=0.25, memory=1024, volumes={OUTPUT: output_volume})
def read_summary(scenario: str):
    """Files of the last finished run of ``scenario`` (empty dict while it is still running)."""
    output_volume.reload()
    out = _summary_dir(scenario)
    if not (out / "status.json").exists():
        return {}
    return {f.name: f.read_text(encoding="utf-8") for f in out.iterdir() if f.is_file()}


def _save_local(files, local):
    local.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (local / name).write_text(text, encoding="utf-8")


@app.local_entrypoint()
def main(scenario: str = "smoke", models: str = "", lrs: str = "", seeds: str = "", tasks: str = "-",
         decoding: str = "", data_tag: str = "seam4k-v1", optim: str = "paged_adamw_8bit",
         gradient_checkpointing: bool = True, gen_batch_size: int = 16, force: bool = False,
         wait: bool = False, collect: bool = False):
    """Default: start the run on Modal and return at once (use with --detach; the PC may be switched off).
    --wait: stay attached and copy the results when done.  --collect: only download the finished results."""
    if scenario not in SCENARIOS:
        raise ValueError(f"--scenario must be one of {sorted(SCENARIOS)}")
    local = ROOT / "results" / "attack_l40s" / scenario
    if collect:
        files = read_summary.remote(scenario)
        if not files:
            print(f"{scenario}: no finished run on the volume yet (still running, or never started)")
            raise SystemExit(1)
        _save_local(files, local)
        print(files["summary.md"] + f"\nsaved: {local}")
        return
    plan = dict(SCENARIOS[scenario])
    for key, value in dict(models=models, lrs=lrs, seeds=seeds, decoding=decoding).items():
        if value:
            plan[key] = value
    if tasks != "-":
        plan["tasks"] = tasks
    if plan["decoding"] not in {"greedy", "checkpoint"}:
        raise ValueError("--decoding must be greedy or checkpoint")
    chosen = _parse_models(plan["models"])
    if not plan["models"]:
        raise ValueError(f"--scenario {scenario} needs --models name=source (source: HF repo or run:<run-name>)")
    mode = plan.get("mode", "test")
    prefix = plan.get("prefix", "val-" if mode == "val" else "")
    common = dict(decoding=plan["decoding"], tasks=plan["tasks"], data_tag=data_tag, optim=optim, mode=mode,
                  save_at_steps=plan.get("save_at_steps", ""),
                  gradient_checkpointing=gradient_checkpointing, gen_batch_size=gen_batch_size,
                  force=force, limit=plan["limit"])
    jobs = []
    for name, source in chosen.items():
        if plan["pre"]:
            jobs.append(dict(common, model_name=name, source=source, job=prefix + "pre", lr=None, seed=42))
        for seed in [int(s) for s in plan["seeds"].split(",") if s]:
            for lr in [v for v in plan["lrs"].split(",") if v]:
                jobs.append(dict(common, model_name=name, source=source, job=f"{prefix}lr{float(lr):g}-s{seed}",
                                 lr=f"{float(lr):g}", seed=seed))
    print(f"{scenario}: {len(jobs)} jobs on {GPU} (one container each, in parallel) | optim {optim} | "
          f"grad ckpt {gradient_checkpointing} | decoding {plan['decoding']} | tasks {plan['tasks'] or '-'}")
    for j in jobs:
        print(f"  {j['model_name']:10s} {j['job']}")
    if not wait:
        print("\nNOTE: this mode needs `modal run --detach`; without it the app stops when this command returns.")
        call = orchestrate.spawn(jobs, scenario, plan["limit"])
        print(f"\nStarted on Modal (call {call.object_id}). With --detach you can now close the terminal"
              f" and switch off the PC.\nProgress: Modal dashboard -> app cbg-attack-eval-l40s.\n"
              f"Results when done: modal run scripts/modal_attack_eval_l40s.py --scenario {scenario} --collect")
        return
    print("\n" + orchestrate.remote(jobs, scenario, plan["limit"]))
    _save_local(read_summary.remote(scenario), local)
    print(f"\nsaved: {local}")
