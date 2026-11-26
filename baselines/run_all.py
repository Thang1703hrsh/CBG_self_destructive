"""Train and evaluate the Base/SEAM/SDD/CTRAP baselines on a multi-GPU server, then print the results table.

For every model and seed: prepare the shared data split, train SEAM (2 GPUs), SDD and CTRAP (1 GPU each),
push each checkpoint to a PUBLIC Hugging Face repo, and evaluate Base and the three defenses on the
robustness suite (clean + harmful fine-tuning at 2e-5, 5e-5, 8e-5, 1e-4, 2e-4; one GPU per case).
Jobs are scheduled on the free GPUs; finished jobs are skipped, so rerunning the same command resumes.

  python baselines/run_all.py                       # Qwen2.5-7B + Mistral-7B, seeds 42 43 44
  python baselines/run_all.py --models qwen --seeds 42 --gpus 0,1
  python baselines/run_all.py --smoke               # ~30 min pipeline check on Qwen2.5-0.5B (no HF push)

Outputs (default outputs/baselines): data/, ckpt/, results/, logs/<job>.log, status.json, RESULTS.md
(mean ± std over seeds), results.csv. At the end RESULTS.md, results.csv and the per-case metrics are also
uploaded to the public HF dataset <hf-user>/self-destruct-baselines-results (unless --no-push).
Stop a run with `kill <PID>` (running jobs are stopped too); rerun the same command to resume.
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
BASE_DIR = REPO / "baselines"
MODELS = {
    "qwen": {"hf": "Qwen/Qwen2.5-7B-Instruct", "short": "qwen25-7b-instruct", "utility_batch_size": 16},
    "mistral": {"hf": "mistralai/Mistral-7B-Instruct-v0.3", "short": "mistral-7b-instruct-v03",
                "utility_batch_size": 64},
    "llama2": {"hf": "meta-llama/Llama-2-7b-chat-hf", "short": "llama2-7b-chat", "utility_batch_size": 64},
}
SMOKE_MODEL = {"hf": "Qwen/Qwen2.5-0.5B-Instruct", "short": "qwen25-0p5b-instruct", "utility_batch_size": 16}
METHODS = ("base", "seam", "sdd", "ctrap")
CASES = ("clean", "harmful-lr2e-5", "harmful-lr5e-5", "harmful-lr8e-5", "harmful-lr1e-4", "harmful-lr2e-4")
# Same evaluation protocol as the earlier Qwen/Mistral transfer runs (greedy HS as in the paper's
# other-models section); the Llama-2 tables used sampled decoding.
PROTOCOL = {"test_size": 100, "attack_size": 1000, "benign_size": 1000, "steps": 250, "train_batch_size": 4,
            "generation_batch_size": 16, "context_length": 256, "max_new_tokens": 50,
            "label_scope": "seam_padded", "data_selection": "seam_original", "harmful_label": "LABEL_0",
            "save_attacked_models": False, "attack_optim": "adamw_torch"}


@dataclass
class Job:
    name: str
    cmd: list
    gpus: int
    deps: list
    done: Path                       # the job is complete when this file exists (written last by the job)
    priority: int
    cwd: Path = REPO
    env: dict = field(default_factory=dict)
    clean: Path = None               # partial output removed before a (re)try
    timeout: float = 4 * 3600        # seconds; a hung job is killed and retried
    attempts: int = 0


def log(message):
    print(time.strftime("[%Y-%m-%d %H:%M:%S] ") + message, flush=True)


def die_with_parent():
    """Linux: a job receives SIGTERM if run_all.py dies, so no orphan keeps writing into a run directory."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)   # PR_SET_PDEATHSIG
    except Exception:
        pass


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def visible_gpus(arg):
    if arg:
        return [g.strip() for g in arg.split(",") if g.strip()]
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return [g.strip() for g in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if g.strip()]
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
    return [str(i) for i, line in enumerate(out.splitlines()) if line.startswith("GPU ")]


def eval_config(model, seed, label, steps, decoding, key):
    sources = {"base": {"kind": "hf", "repo_id": model["hf"]},
               "seam": {"kind": "volume", "volume": "ckpt", "path": f"{key}/seam-seed{seed}/checkpoint-{steps}"},
               "sdd": {"kind": "volume", "volume": "ckpt", "path": f"{key}/sdd-seed{seed}/model"},
               "ctrap": {"kind": "volume", "volume": "ckpt", "path": f"{key}/ctrap-seed{seed}/model"}}
    defense = {"seam": "seam_defense.json", "sdd": "sdd_pairs.jsonl", "ctrap": "ctrap_defense.json"}
    entry = {"source": sources[label]}
    if label in defense:
        entry["defense_data"] = [{"kind": "volume", "volume": "data", "path": f"seed{seed}/{defense[label]}"}]
    return {
        "models": {label: entry},
        "data": {"attack": {"kind": "volume", "volume": "data", "path": f"seed{seed}/attack.json"},
                 "test": {"kind": "dataset", "repo_id": "PKU-Alignment/BeaverTails", "split": "30k_test"},
                 "benign": {"kind": "dataset", "repo_id": "yahma/alpaca-cleaned", "split": "train",
                            "exclude_first": 4000, "exclude_seed": seed}},
        "protocol": {**PROTOCOL, "seed": seed, "decoding": decoding,
                     "utility_batch_size": model["utility_batch_size"]},
    }


def build_jobs(args, models, out):
    py, steps, jobs = sys.executable, args.steps, []
    data, ckpt, results = out / "data", out / "ckpt", out / "results"
    common_env = {"WANDB_MODE": "disabled", "TOKENIZERS_PARALLELISM": "false", "USE_TF": "0",
                  "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                  "DEFENSE_EVAL_SOURCES": str(out)}
    smoke = ["--smoke", "--smoke-export"] if args.smoke else []
    for seed in args.seeds:
        jobs.append(Job(f"data-seed{seed}", [py, str(BASE_DIR / "prepare_data.py"), "--seed", str(seed),
                                             "--out", str(data)], 0, [], data / f"seed{seed}/manifest.json", 0,
                        timeout=3 * 3600))
    seam_layout = (["--gradient_storage", "gpu", "--perturbation_storage", "cpu"] if args.seam_gpus >= 2
                   else ["--gradient_storage", "gpu", "--perturbation_storage", "gpu"])
    for key, model in models.items():
        for seed in args.seeds:
            d, c = data / f"seed{seed}", ckpt / key
            trained = {}
            if "seam" in args.methods:
                run = c / f"seam-seed{seed}"
                jobs.append(Job(f"train-seam-{key}-seed{seed}", [
                    py, "-m", "src.train", "--model_name", model["hf"], "--tokenizer_name", model["hf"],
                    "--data_path", str(d / "seam_defense.json"), "--output_dir", str(run),
                    "--defense_size", "4000", "--benign_seed", str(seed), "--seed", str(seed),
                    "--max_steps", str(steps), "--per_device_train_batch_size", "8",
                    "--save_strategy", "steps", "--save_steps", str(steps), "--report_to", "none",
                    "--wandb_run_name", f"seam-{key}-seed{seed}", "--weight_dtype", "bfloat16",
                    "--optimizer_cpu_offload", "false", "--bf16", "true", *seam_layout],
                    # trainer_state.json is the last file HF Trainer writes into a checkpoint.
                    args.seam_gpus, [f"data-seed{seed}"], run / f"checkpoint-{steps}/trainer_state.json", 1,
                    cwd=BASE_DIR / "seam", env={"PYTHONPATH": ""}, clean=run, timeout=14 * 3600))
                trained["seam"] = (run / f"checkpoint-{steps}", None, [run / "loss_components.jsonl"])
            if "sdd" in args.methods:
                run = c / f"sdd-seed{seed}"
                jobs.append(Job(f"train-sdd-{key}-seed{seed}", [
                    py, str(BASE_DIR / "sdd_train.py"), "--model", model["hf"], "--pairs", str(d / "sdd_pairs.jsonl"),
                    "--output", str(run), "--seed", str(seed), "--max-steps", str(steps), *smoke],
                    # train_metrics.json is written after the model export.
                    1, [f"data-seed{seed}"], run / "train_metrics.json", 2, clean=run, timeout=6 * 3600))
                trained["sdd"] = (run / "model", None, [run / "train_metrics.json"])
            if "ctrap" in args.methods:
                run = c / f"ctrap-seed{seed}"
                jobs.append(Job(f"train-ctrap-{key}-seed{seed}", [
                    str(args.ctrap_python), str(BASE_DIR / "ctrap/ctrap_train.py"), "--model", model["hf"],
                    "--defense", str(d / "ctrap_defense.json"), "--output", str(run), "--seed", str(seed),
                    "--max-steps", str(steps), "--direct-gpu-load", *smoke],
                    1, [f"data-seed{seed}"], run / "train_metrics.json", 2, clean=run, timeout=6 * 3600))
                trained["ctrap"] = (run / "model", run / "adapter", [run / "train_metrics.json"])
            for method, (model_dir, adapter, extra) in trained.items():
                if args.push:
                    repo = f"{args.hf_namespace}/{method}-{model['short']}-seed{seed}-step{steps}"
                    marker = model_dir.parent / f"pushed-{method}.json"
                    cmd = [py, str(BASE_DIR / "push_hf.py"), "--model-dir", str(model_dir), "--repo", repo,
                           "--method", method.upper(), "--base", model["hf"], "--seed", str(seed),
                           "--steps", str(steps), "--marker", str(marker), "--extra", *map(str, extra)]
                    if adapter:
                        cmd += ["--adapter-dir", str(adapter)]
                    jobs.append(Job(f"push-{method}-{key}-seed{seed}", cmd, 0,
                                    [f"train-{method}-{key}-seed{seed}"], marker, 5, timeout=8 * 3600))
            for label in args.methods:
                run_dir = results / key / f"seed{seed}" / label
                config = run_dir / "config.yaml"
                if not args.dry_run:
                    run_dir.mkdir(parents=True, exist_ok=True)
                    config.write_text(yaml.safe_dump(eval_config(model, seed, label, steps, args.decoding, key),
                                                     sort_keys=False), encoding="utf-8")
                deps = [f"data-seed{seed}"] + ([f"train-{label}-{key}-seed{seed}"] if label != "base" else [])
                prep = f"evalprep-{label}-{key}-seed{seed}"
                jobs.append(Job(prep, [py, str(BASE_DIR / "eval/prepare_local.py"), "--config", str(config),
                                       "--suite", args.suite, "--models", label, "--run-dir", str(run_dir)],
                                0, deps, run_dir / "manifest.json", 3, timeout=3 * 3600))
                for case in (("clean", "harmful-lr5e-5") if args.smoke else CASES):
                    jobs.append(Job(f"eval-{label}-{key}-seed{seed}-{case}", [
                        py, str(BASE_DIR / "eval/defense_eval_worker.py"), "--manifest", str(run_dir / "manifest.json"),
                        "--model", label, "--case", case], 1, [prep],
                        run_dir / "cases" / label / case / "result.json", 4, timeout=4 * 3600))
    for job in jobs:
        job.env = {**common_env, **job.env}
    return jobs


def schedule(jobs, gpus, out, retries, cpu_slots, max_eval):
    by_name = {job.name: job for job in jobs}
    state = {job.name: ("done" if job.done.is_file() else "pending") for job in jobs}
    free, running = list(gpus), {}
    (out / "logs").mkdir(parents=True, exist_ok=True)

    def write_status():
        counts = {s: sum(v == s for v in state.values()) for s in ("done", "running", "pending", "failed")}
        (out / "status.json").write_text(json.dumps({"counts": counts, "jobs": state}, indent=1), encoding="utf-8")

    log(f"{len(jobs)} jobs ({sum(s == 'done' for s in state.values())} already done) on GPUs {','.join(gpus)}")
    try:
        while True:
            for job in jobs:   # a failed dependency fails its dependents
                if state[job.name] == "pending" and any(state[d] == "failed" for d in job.deps):
                    state[job.name] = "failed"
                    log(f"SKIP {job.name}: a dependency failed")
            ready = sorted((j for j in jobs if state[j.name] == "pending" and all(state[d] == "done" for d in j.deps)),
                           key=lambda j: j.priority)
            cpu_busy = sum(by_name[n].gpus == 0 for n in running)
            eval_busy = sum(n.startswith("eval-") for n in running)
            reserved = False
            for job in ready:
                if job.gpus == 0:
                    if cpu_busy >= cpu_slots:
                        continue
                    cpu_busy += 1
                    assigned = []
                else:
                    if reserved or job.gpus > len(free) or (job.name.startswith("eval-") and eval_busy >= max_eval):
                        reserved = reserved or job.gpus > 1   # keep freed GPUs for a waiting multi-GPU job
                        continue
                    assigned, free[:] = free[:job.gpus], free[job.gpus:]
                    eval_busy += job.name.startswith("eval-")
                if job.clean is not None and job.clean.exists():
                    shutil.rmtree(job.clean)
                env = {**os.environ, **job.env, "CUDA_VISIBLE_DEVICES": ",".join(assigned)}
                handle = open(out / "logs" / f"{job.name}.log", "a", encoding="utf-8")
                handle.write(f"\n===== attempt {job.attempts + 1}, GPUs [{','.join(assigned)}] =====\n")
                handle.flush()
                process = subprocess.Popen(job.cmd, cwd=job.cwd, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                           preexec_fn=die_with_parent if sys.platform.startswith("linux") else None)
                running[job.name] = (process, assigned, handle, time.time())
                state[job.name] = "running"
                log(f"START {job.name} gpus=[{','.join(assigned)}]")
            write_status()
            if not running:
                break
            time.sleep(10)
            for name, (process, assigned, handle, started) in list(running.items()):
                code = process.poll()
                if code is None and time.time() - started > by_name[name].timeout:
                    log(f"TIMEOUT {name} after {by_name[name].timeout / 3600:.1f} h: killing it")
                    process.kill()
                    code = process.wait()
                if code is None:
                    continue
                handle.close()
                free.extend(assigned)
                del running[name]
                job = by_name[name]
                if code == 0 and job.done.is_file():
                    state[name] = "done"
                    log(f"DONE  {name}")
                else:
                    job.attempts += 1
                    state[name] = "pending" if job.attempts <= retries else "failed"
                    log(f"FAIL  {name} (exit {code}, attempt {job.attempts}) -> {state[name]}; see logs/{name}.log")
    except KeyboardInterrupt:
        log("interrupted: stopping running jobs (rerun the same command to resume)")
        for process, _, _, _ in running.values():
            process.terminate()
        for process, _, _, _ in running.values():
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
        raise
    finally:
        write_status()
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default="qwen,mistral", help=f"comma list of {','.join(MODELS)}")
    parser.add_argument("--seeds", default=None, help="default 42,43,44 (42 with --smoke)")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--gpus", default="", help="GPU ids to use (default: all visible)")
    parser.add_argument("--seam-gpus", type=int, default=0,
                        help="GPUs per SEAM run: 3 (model | gradients | perturbed gradients, most memory headroom), "
                             "2 (both gradient sets on the second GPU; ~5 GB headroom for Qwen2.5-7B on 80 GB). "
                             "Default: 3 if at least 3 GPUs are visible, else 2")
    parser.add_argument("--out", type=Path, default=REPO / "outputs" / "baselines")
    parser.add_argument("--ctrap-python", type=Path, default=REPO / ".venv-ctrap" / "bin" / "python")
    parser.add_argument("--decoding", default="greedy", choices=("greedy", "sample"))
    parser.add_argument("--no-push", action="store_true", help="do not upload checkpoints to Hugging Face")
    parser.add_argument("--hf-namespace", default="", help="HF user/org for the public repos (default: token owner)")
    parser.add_argument("--results-repo", default="",
                        help="public HF dataset for the results (default: <hf-namespace>/self-destruct-baselines-results)")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--cpu-slots", type=int, default=2)
    parser.add_argument("--max-eval-parallel", type=int, default=6, help="caps HF Hub traffic of lm-eval")
    parser.add_argument("--smoke", action="store_true", help="tiny end-to-end check (Qwen2.5-0.5B, 2 steps)")
    parser.add_argument("--dry-run", action="store_true", help="list the jobs and exit")
    args = parser.parse_args()
    args.seeds = [int(s) for s in (args.seeds or ("42" if args.smoke else "42,43,44")).split(",")]
    args.methods = [m for m in args.methods.split(",") if m]
    if set(args.methods) - set(METHODS):
        raise SystemExit(f"--methods must be from {METHODS}")
    if args.smoke:
        models, args.steps, args.suite = {"smoke": SMOKE_MODEL}, 2, "smoke"
        args.out = args.out.parent / (args.out.name + "-smoke")
        args.no_push = True
    else:
        models = {k: MODELS[k] for k in args.models.split(",")}
        args.steps, args.suite = 500, "robustness"
    args.out = args.out.resolve()
    args.push = not args.no_push
    if not args.seam_gpus:
        try:
            count = len(visible_gpus(args.gpus))
        except FileNotFoundError:   # no nvidia-smi (e.g. --dry-run on a laptop)
            count = 2
        args.seam_gpus = 3 if count >= 3 else 2
    if args.dry_run:
        args.hf_namespace = args.hf_namespace or "<hf-user>"
        jobs = build_jobs(args, models, args.out)
        for job in jobs:
            status = "done" if job.done.is_file() else "todo"
            print(f"{status:4} gpus={job.gpus} {job.name:48} deps={','.join(job.deps)}")
        print(f"{len(jobs)} jobs")
        return
    if not os.environ.get("HF_TOKEN") and not (Path.home() / ".cache/huggingface/token").is_file():
        raise SystemExit("Set HF_TOKEN (needed for downloads and the public uploads)")
    if args.push and not args.hf_namespace:
        from huggingface_hub import HfApi
        args.hf_namespace = HfApi().whoami()["name"]
    if "ctrap" in args.methods and not args.ctrap_python.is_file():
        raise SystemExit(f"CTRAP environment missing: {args.ctrap_python} (run scripts/setup_baselines.sh)")
    gpus = visible_gpus(args.gpus)
    if not gpus:
        raise SystemExit("No GPU found")
    if "seam" in args.methods and len(gpus) < args.seam_gpus:
        raise SystemExit(f"SEAM needs {args.seam_gpus} GPUs; only {len(gpus)} available")
    args.results_repo = args.results_repo or f"{args.hf_namespace or 'hf-user'}/self-destruct-baselines-results"
    log(f"models={list(models)} seeds={args.seeds} methods={args.methods} out={args.out} "
        f"push={'public HF repos under ' + args.hf_namespace if args.push else 'off'}")
    args.out.mkdir(parents=True, exist_ok=True)
    lock = args.out / ".run.lock"
    if lock.is_file() and pid_alive(int(lock.read_text().strip() or 0)):
        raise SystemExit(f"Another run_all.py (PID {lock.read_text().strip()}) is using {args.out}")
    lock.write_text(str(os.getpid()))
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        state = schedule(build_jobs(args, models, args.out), gpus, args.out, args.retries,
                         args.cpu_slots, args.max_eval_parallel)
    finally:
        lock.unlink(missing_ok=True)
    failed = sorted(name for name, s in state.items() if s == "failed")
    subprocess.run([sys.executable, str(BASE_DIR / "aggregate.py"), "--out", str(args.out),
                    "--models", ",".join(models), "--seeds", ",".join(map(str, args.seeds)),
                    "--methods", ",".join(args.methods)], check=False)
    # Results (table + per-case metrics) to a public HF dataset; a dry run in --smoke checks the file selection.
    push_results = [sys.executable, str(BASE_DIR / "push_results.py"), "--out", str(args.out),
                    "--repo", args.results_repo] + ([] if args.push else ["--dry-run"])
    if subprocess.run(push_results, check=False).returncode:
        log("results upload failed; retry with: python baselines/push_results.py "
            f"--out {args.out} --repo {args.results_repo}")
    log(f"results: {args.out / 'RESULTS.md'} (mean ± std), {args.out / 'results.csv'}"
        + (f", https://huggingface.co/datasets/{args.results_repo}" if args.push else ""))
    if failed:
        log(f"{len(failed)} job(s) failed: {', '.join(failed)}. Fix the cause (logs/) and rerun the same command.")
        raise SystemExit(1)
    log("all jobs done")


if __name__ == "__main__":
    main()
