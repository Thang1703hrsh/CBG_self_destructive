"""Modal launcher: CBG v2 defense training of Llama-2-7B-Chat on one 2×H200 container.

Stages (run in this order; each is explicit):
  modal run scripts/modal_cbg_v2_7b.py --stage prepare            # CPU: pinned data, pools, model download
  modal run scripts/modal_cbg_v2_7b.py --stage verify             # CPU: reference + equivalence tests
  modal run scripts/modal_cbg_v2_7b.py --stage radius --run-name r1     # 2×H200, ~10 min
  modal run scripts/modal_cbg_v2_7b.py --stage hvp-check --run-name h1 \
      --config config/cbg_v2_llama2_7b_2xh200_fast_hvp.yaml        # 2×H200: exact HVP vs FD
  modal run scripts/modal_cbg_v2_7b.py --stage smoke  --run-name s1     # 2×H200, 3 steps
  modal run --detach scripts/modal_cbg_v2_7b.py --stage train --run-name cbg-7b-s42 --hf-repo <user>/<name>
  modal run scripts/modal_cbg_v2_7b.py --stage upload --run-name cbg-7b-s42 --hf-repo <user>/<name>  # CPU, retry

`train` resumes automatically from the run's last checkpoint when the same run
name is launched again with the same config (also after a preemption retry); a
finished run is never retrained. With --hf-repo, the GPU container uploads
final/ (+ training records and a model card) to the Hub as soon as training
ends, so the upload also happens for --detach runs. Repos are private unless
--hf-public; --hf-dtype bfloat16 halves the upload (float32 = exactly as
trained). The HF_TOKEN secret must have write access.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/workspace"
DATA = "/vol/data"
OUTPUT = "/vol/output"
CACHE = "/vol/hf"
# Own variable on purpose: CBG_MODAL_GPU (S0 scripts) is usually a single GPU, e.g. "H200".
GPU = os.environ.get("CBG_V2_MODAL_GPU", "H200:2")


def _gpu_count(spec):
    count = spec.rsplit(":", 1)[1] if ":" in spec else "1"
    return int(count) if count.isdigit() else 1


# Host resources of the GPU containers, sized to what the stages use (the GPUs do the work):
#   CPU  one Python process; data is pre-tokenized in RAM; OMP_NUM_THREADS=4.
#   RAM  train/smoke: fp32 model load (~27 GB, once per model, sequential), checkpoint
#        save/resume streamed (torch.save per tensor, torch.load mmap) -> 64 GiB.
#        radius/hvp-check: two fp32 gradient snapshots on CPU (2 x 27 GB) + load -> 96 GiB.
GPU_CPU = float(os.environ.get("CBG_V2_CPU", "4"))
TRAIN_MEMORY = int(os.environ.get("CBG_V2_TRAIN_MEMORY", str(64 * 1024)))   # MiB
CHECK_MEMORY = int(os.environ.get("CBG_V2_CHECK_MEMORY", str(96 * 1024)))   # MiB
CHECK_STAGES = {"radius", "hvp-check"}
# Attack evaluation follows SEAM's eval.py: bf16 model + full-parameter attack on ONE GPU (~55 GB).
EVAL_GPU = os.environ.get("CBG_V2_EVAL_GPU", "H200")
BASE_RUN = "base-llama2-7b-chat"   # --run-name for evaluating the undefended base model

def _image(*extra_packages):
    """Base environment; local sources are added last (Modal forbids build steps after add_local_*)."""
    base = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git")
        .pip_install_from_requirements(str(ROOT / "requirements.txt"))
    )
    if extra_packages:
        base = base.pip_install(*extra_packages)
    return (
        base.env({"PYTHONPATH": REMOTE, "USE_TF": "0", "TRANSFORMERS_NO_TF": "1", "HF_HOME": CACHE,
                  "HF_DATASETS_CACHE": CACHE + "/datasets", "TOKENIZERS_PARALLELISM": "false",
                  "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "WANDB_MODE": "disabled",
                  "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                  # Deterministic cuBLAS: the two perturbed passes must differ only by the perturbation.
                  "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
        .add_local_dir(str(ROOT / "src"), REMOTE + "/src")
        .add_local_dir(str(ROOT / "tests"), REMOTE + "/tests")
        .add_local_dir(str(ROOT / "config"), REMOTE + "/config")
        .add_local_file(str(ROOT / "cbg_reference.py"), REMOTE + "/cbg_reference.py")
    )


image = _image()
# Attacks on one L40S use paged 8-bit AdamW (bitsandbytes), as scripts/modal_attack_eval_l40s.py.
attack_image = _image("bitsandbytes==0.45.3", "sentencepiece", "protobuf")
app = modal.App("cbg-v2-7b")
data_volume = modal.Volume.from_name("cbg-v2-7b-data", create_if_missing=True)
output_volume = modal.Volume.from_name("cbg-v2-7b-output", create_if_missing=True)
cache_volume = modal.Volume.from_name("cbg-spec-hf", create_if_missing=True)
volumes = {DATA: data_volume, OUTPUT: output_volume, CACHE: cache_volume}
hf_secret = modal.Secret.from_name("HF_TOKEN")


def _name(value, label):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value):
        raise ValueError(f"invalid {label}: {value!r}")
    return value


def _stream(argv, log_path, commit_every=600):
    """Run a child process, tee its output, and commit the output volume periodically."""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    def committer():
        while not stop.wait(commit_every):
            try:
                output_volume.commit()
            except Exception as error:  # a failed periodic commit must not kill training
                print(f"[launcher] volume commit failed: {error}", flush=True)
    thread = threading.Thread(target=committer, daemon=True)
    thread.start()
    try:
        with open(log_path, "a", encoding="utf-8") as log:
            process = subprocess.Popen(argv, cwd=REMOTE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1)
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            status = process.wait()
    finally:
        stop.set()
        output_volume.commit()
    if status:
        raise RuntimeError(f"{argv[2:4]} exited with status {status}; see {log_path}")


@app.function(image=image, cpu=4, memory=16384, timeout=4 * 3600, volumes=volumes, secrets=[hf_secret])
def prepare(config_text: str, data_tag: str):
    import yaml
    from src.experiments.prepare_v2 import prepare_data
    os.chdir(REMOTE)
    try:
        # Same code as local servers: python -m src.experiments.prepare_v2
        return prepare_data(yaml.safe_load(config_text), Path(DATA) / _name(data_tag, "data tag"))
    finally:
        data_volume.commit()
        cache_volume.commit()


@app.function(image=image, cpu=4, memory=8192, timeout=3600)
def verify():
    os.chdir(REMOTE)
    results = {}
    for name, argv in (("cbg_reference", [sys.executable, "cbg_reference.py"]),
                       ("unit_tests", [sys.executable, "-m", "unittest", "-v", "test_cbg_spec",
                                       "test_cbg_method_v2", "test_cbg_step_v2", "test_cbg_train_v2", "test_cbg_s01",
                                       "test_hf_upload", "test_attack_eval"])):
        cwd = REMOTE + "/tests" if name == "unit_tests" else REMOTE
        done = subprocess.run(argv, cwd=cwd, capture_output=True, text=True)
        text = done.stdout + done.stderr
        print(text[-6000:], flush=True)
        results[name] = "passed" if done.returncode == 0 else "failed"
    if any(v != "passed" for v in results.values()):
        raise RuntimeError(f"verification failed: {results}")
    return results


def _upload(run_dir, hf_repo, hf_public, hf_dtype, data_dir):
    from src.experiments.hf_upload import upload_run
    from src.experiments.protocol import read_json
    pinned = read_json(Path(data_dir) / "model_revision.json")
    record = upload_run(run_dir, hf_repo, private=not hf_public, dtype=hf_dtype, base_model=pinned["model"],
                        base_revision=pinned["revision"], token=os.environ.get("HF_TOKEN"),
                        staging_dir=str(run_dir))   # a bf16 cast is staged on the volume, not container disk
    output_volume.commit()
    print(f"[launcher] uploaded to {record['url']} ({record['commit']})", flush=True)
    return record


# float32 uploads stream the files; a bf16 cast holds one safetensors shard (<= 5 GB) at a time.
@app.function(image=image, cpu=2, memory=16384, timeout=4 * 3600, volumes=volumes, secrets=[hf_secret])
def upload(run_name: str, data_tag: str, hf_repo: str, hf_public: bool = False, hf_dtype: str = "float32"):
    os.chdir(REMOTE)
    output_volume.reload()
    run_dir = Path(OUTPUT) / _name(run_name, "run name") / "train"
    return _upload(run_dir, hf_repo, hf_public, hf_dtype, Path(DATA) / _name(data_tag, "data tag"))


@app.function(image=image, gpu=GPU, cpu=GPU_CPU, memory=TRAIN_MEMORY, timeout=24 * 3600, volumes=volumes,
              secrets=[hf_secret], retries=modal.Retries(max_retries=2, initial_delay=30.0))
def gpu_run(stage: str, config_text: str, data_tag: str, run_name: str, smoke_steps: int = 6,
            hf_repo: str = "", hf_public: bool = False, hf_dtype: str = "float32", model_override: str = ""):
    """train / smoke / geometry."""
    return _gpu_entry(stage, config_text, data_tag, run_name, smoke_steps, hf_repo, hf_public, hf_dtype,
                      model_override)


@app.function(image=image, gpu=GPU, cpu=GPU_CPU, memory=CHECK_MEMORY, timeout=4 * 3600, volumes=volumes,
              secrets=[hf_secret])
def gpu_check(stage: str, config_text: str, data_tag: str, run_name: str):
    """radius / hvp-check: hold fp32 gradient snapshots in host RAM."""
    return _gpu_entry(stage, config_text, data_tag, run_name, 0, "", False, "float32")


@app.function(image=image, gpu=EVAL_GPU, cpu=GPU_CPU, memory=TRAIN_MEMORY, timeout=8 * 3600, volumes=volumes,
              secrets=[hf_secret])
def gpu_eval(run_name: str, data_tag: str, mode: str, lrs: str, utility_tasks: str, post_utility: bool = False):
    """SEAM-protocol attack + harmfulness/utility of <run>/train/final (or the base model for BASE_RUN)."""
    from huggingface_hub import snapshot_download
    from src.experiments.protocol import read_json
    os.chdir(REMOTE)
    try:
        data_dir = Path(DATA) / _name(data_tag, "data tag")
        if run_name == BASE_RUN:
            pinned = read_json(data_dir / "model_revision.json")
            model_path = snapshot_download(pinned["model"], revision=pinned["revision"], local_files_only=True,
                                           allow_patterns=["*.json", "*.safetensors", "tokenizer.model"])
        else:
            model_path = str(Path(OUTPUT) / _name(run_name, "run name") / "train" / "final")
            metrics = Path(OUTPUT) / run_name / "train" / "metrics.json"
            if not metrics.exists() or read_json(metrics).get("status") != "complete":
                raise FileNotFoundError(f"{run_name} has no completed training run (train/metrics.json)")
        run_dir = Path(OUTPUT) / _name(run_name, "run name") / f"attack-eval-{mode}"
        argv = [sys.executable, "-m", "src.experiments.attack_eval", "--model", model_path, "--data", str(data_dir),
                "--output", str(run_dir), "--mode", mode, "--lrs", lrs, "--utility", utility_tasks]
        if post_utility:
            argv.append("--post-utility")
        started = time.time()
        _stream(argv, run_dir / "console.log")
        cache_volume.commit()   # classifier + lm-eval datasets downloaded on first use
        return dict(stage="attack-eval", output=str(run_dir), hours=(time.time() - started) / 3600,
                    result=read_json(run_dir / "attack_eval.json"))
    except Exception as error:   # returned, not raised: no retries for evaluation
        return dict(status="failed", stage="attack-eval", error=repr(error))


def _gpu_entry(stage, config_text, data_tag, run_name, smoke_steps, hf_repo, hf_public, hf_dtype, model_override=""):
    import torch
    import yaml
    from huggingface_hub import snapshot_download
    from src.experiments.protocol import read_json
    os.chdir(REMOTE)
    # Configuration errors are returned, not raised: a raise would be retried (and billed) twice more.
    required, visible = yaml.safe_load(config_text)["execution"]["devices"], torch.cuda.device_count()
    if isinstance(required, int) and visible < required:
        return dict(status="failed", stage=stage,
                    error=f"{required} GPUs required, {visible} visible (container GPU spec {GPU!r})")
    if stage != "train":   # radius/smoke: a retry reproduces the same failure; train keeps retries (preemption)
        try:
            return _gpu_stage(stage, config_text, data_tag, run_name, smoke_steps, hf_repo, hf_public, hf_dtype,
                              model_override)
        except Exception as error:
            return dict(status="failed", stage=stage, error=repr(error))
    return _gpu_stage(stage, config_text, data_tag, run_name, smoke_steps, hf_repo, hf_public, hf_dtype)


def _gpu_stage(stage, config_text, data_tag, run_name, smoke_steps, hf_repo, hf_public, hf_dtype, model_override=""):
    from huggingface_hub import snapshot_download
    from src.experiments.protocol import read_json
    data_dir = Path(DATA) / _name(data_tag, "data tag")
    if not (data_dir / "manifest.json").exists():
        raise FileNotFoundError("run --stage prepare first")
    pinned = read_json(data_dir / "model_revision.json")
    model_path = snapshot_download(pinned["model"], revision=pinned["revision"], local_files_only=True,
                                   allow_patterns=["*.json", "*.safetensors", "tokenizer.model"])
    if model_override:   # a checkpoint on the output volume (path relative to it), e.g. an attacked model
        model_path = str(Path(OUTPUT) / model_override)
        if ".." in Path(model_override).parts or not (Path(model_path) / "config.json").exists():
            raise FileNotFoundError(f"no model at {model_path} on the output volume")
    run_dir = Path(OUTPUT) / _name(run_name, "run name") / stage
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.yaml"
    if config_path.exists() and config_path.read_text(encoding="utf-8") != config_text:
        raise ValueError(f"{run_dir} was started with a different config; use a new --run-name")
    config_path.write_text(config_text, encoding="utf-8")
    argv = [sys.executable, "-m", "src.cbg.train_v2", "--config", str(config_path), "--data", str(data_dir),
            "--output", str(run_dir), "--model", model_path, "--stage", stage, "--smoke-steps", str(smoke_steps)]
    if stage == "train" and (run_dir / "identity.json").exists():
        argv.append("--resume")   # same run name + same config: continue from the last checkpoint
    started = time.time()
    _stream(argv, run_dir / "console.log")
    result = dict(stage=stage, output=str(run_dir), hours=(time.time() - started) / 3600)
    for name in ("smoke_summary.json", "radius_check.json", "hvp_check.json", "metrics.json"):
        if (run_dir / name).exists():
            result[name] = read_json(run_dir / name)
    if stage == "train" and hf_repo and result.get("metrics.json", {}).get("status") != "complete":
        print("[launcher] training did not complete (e.g. early-stopped): no HF upload", flush=True)
    elif stage == "train" and hf_repo:
        if (run_dir / "hf_upload.json").exists() and read_json(run_dir / "hf_upload.json")["repo_id"] == hf_repo:
            result["hf_upload"] = read_json(run_dir / "hf_upload.json")   # a retry after a finished upload
        else:
            try:
                result["hf_upload"] = _upload(run_dir, hf_repo, hf_public, hf_dtype, data_dir)
            except Exception as error:   # the trained model is safe on the volume; never retrain for this
                print(f"[launcher] HF upload failed: {error!r}; retry with --stage upload", flush=True)
                result["hf_upload"] = dict(status="failed", error=repr(error))
    return result


# ---------------------------------------------------------------------------
# C2 test pipeline: runs entirely on Modal (the local machine may be switched off after the launch).
# ---------------------------------------------------------------------------
C2_RUN = "cbg-pilot-c2"
C2_DIR = "c2-pipeline"


def _c2_write(name, value):
    path = Path(OUTPUT) / C2_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) if not isinstance(value, str) else value,
                    encoding="utf-8")
    output_volume.commit()


@app.function(image=attack_image, gpu="L40S", cpu=4, memory=32 * 1024, timeout=6 * 3600, volumes=volumes,
              secrets=[hf_secret])
def l40s_attack(label: str, model_rel: str, data_tag: str):
    """Validation attack at lr 1e-4 (attack_eval protocol of modal_attack_eval_l40s.py): HS + arc_easy, model
    snapshots at attack step 25 and at the end. model_rel: path on the output volume, or "" for the base model."""
    from huggingface_hub import snapshot_download
    from src.experiments.protocol import read_json
    os.chdir(REMOTE)
    cache_volume.reload()
    output_volume.reload()
    data_dir = Path(DATA) / _name(data_tag, "data tag")
    pinned = read_json(data_dir / "model_revision.json")
    base = snapshot_download(pinned["model"], revision=pinned["revision"], local_files_only=True,
                             allow_patterns=["*.json", "*.safetensors", "tokenizer.model"])
    model = str(Path(OUTPUT) / model_rel) if model_rel else base
    out = Path(OUTPUT) / C2_DIR / "attacks" / _name(label, "label")
    argv = [sys.executable, "-m", "src.experiments.attack_eval", "--model", model, "--tokenizer", base,
            "--data", str(data_dir), "--output", str(out), "--mode", "val", "--lrs", "1e-4", "--decoding", "greedy",
            "--utility", "arc_easy", "--post-utility", "--skip-pre", "--optim", "paged_adamw_8bit",
            "--gradient-checkpointing", "--gen-batch-size", "16", "--save-at-steps", "25"]
    _stream(argv, out / "console.log")
    output_volume.commit()
    cache_volume.commit()
    return read_json(out / "attack_eval.json")


@app.function(image=image, cpu=0.25, memory=1024, timeout=24 * 3600, volumes=volumes)
def pipeline_c2(config_text: str, data_tag: str):
    """prepare (if needed) -> [base attack || C2 training] -> C2 attack -> post-attack geometry -> summary."""
    from src.experiments.protocol import read_json
    status = dict(stage="start", started=time.time())
    def stage(name, **extra):
        status.update(stage=name, **extra)
        _c2_write("status.json", status)
        print(f"[c2] {name}", flush=True)
    try:
        data_volume.reload()
        if not (Path(DATA) / data_tag / "model_revision.json").exists():
            stage("prepare")
            prepare.remote(config_text, data_tag)
            data_volume.reload()
        stage("train C2 + base attack")
        base_attack = l40s_attack.spawn("base", "", data_tag)
        train = gpu_run.remote("train", config_text, data_tag, C2_RUN)
        metrics = train.get("metrics.json", {}) if isinstance(train, dict) else {}
        if metrics.get("status") != "complete":
            raise RuntimeError(f"C2 training did not complete: {train}")
        stage("C2 attack")
        c2_attack = l40s_attack.remote("cbg-pilot-c2", f"{C2_RUN}/train/final", data_tag)
        base_attack = base_attack.get()
        stage("post-attack geometry")
        geometry = {}
        calls = {label: gpu_run.spawn("geometry", config_text, data_tag, f"geo-c2pipe-{label}", 6, "", False,
                                      "float32", f"{C2_DIR}/attacks/{label}/attacked-0.0001/step-25")
                 for label in ("base", "cbg-pilot-c2")}
        for label, call in calls.items():
            result = call.get()
            output_volume.reload()
            path = Path(OUTPUT) / f"geo-c2pipe-{label}" / "geometry" / "geometry.json"
            geometry[label] = read_json(path)["geo/log_eta_k"] if path.exists() else f"failed: {result}"
        log = [json.loads(line) for line in (Path(OUTPUT) / C2_RUN / "train" / "train_log.jsonl")
               .read_text(encoding="utf-8").splitlines() if line.strip()]
        held_out = {r["step"]: r["geo/log_eta_k"] for r in log if "geo/log_eta_k" in r}
        rows = []
        for label, res in (("base", base_attack), ("cbg-pilot-c2", c2_attack)):
            entry = res["attacks"]["0.0001"]
            rows.append(dict(model=label, HS=entry["harmfulness"]["score"],
                             harmful_rate=entry["harmfulness"]["scores_over_50"],
                             arc_easy_after=entry.get("utility", {}).get("arc_easy"), attack_loss=entry["train_loss"],
                             log_eta_after_25_attack_steps=geometry.get(label)))
        summary = dict(held_out_log_eta_during_training=held_out, attack_lr=1e-4, split="val", rows=rows)
        _c2_write("summary.json", summary)
        lines = ["| model | HS | HS>0.5 | arc_easy after | attack loss | log eta after 25 attack steps |",
                 "|---|---|---|---|---|---|"]
        lines += [f"| {r['model']} | {r['HS']:.3f} | {r['harmful_rate']:.3f} | {r['arc_easy_after']} | "
                  f"{r['attack_loss']:.3f} | {r['log_eta_after_25_attack_steps']} |" for r in rows]
        lines += ["", f"C2 held-out log eta during training: {held_out}"]
        _c2_write("summary.md", "\n".join(lines) + "\n")
        stage("complete", finished=time.time())
        return summary
    except Exception as error:
        stage("failed", error=repr(error))
        raise


@app.function(image=image, cpu=0.25, memory=1024, volumes=volumes)
def c2_status():
    output_volume.reload()
    out = Path(OUTPUT) / C2_DIR
    parts = [(out / name).read_text(encoding="utf-8") for name in ("status.json", "summary.md") if (out / name).exists()]
    return "\n".join(parts) or "C2 pipeline has not written anything yet"


# ---------------------------------------------------------------------------
# Checkpoint-selection pipeline: train once (models kept at steps 100/250/500), validation attacks at lr 1e-4,
# selection, then the test evaluation of the selected model. Runs entirely on Modal.
# ---------------------------------------------------------------------------
SELECT_DIR = "select-pipeline"
PAPER_TASKS = "truthfulqa_mc1,mmlu,hellaswag,arc_easy"
TEST_LRS = ("2e-5", "5e-5", "8e-5", "1e-4", "2e-4")
# Base model on the same protocol (results/attack_l40s, measured on two accounts): pre-attack MMLU 46.4, ARC-e 73.8.
BASE_PRE_MMLU_ARC = (0.464 + 0.738) / 2
BASE_VAL_LR1E4 = dict(HS=0.568, attack_loss=0.869)


def _select_write(name, value):
    path = Path(OUTPUT) / SELECT_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value, indent=2, default=str), encoding="utf-8")
    output_volume.commit()


@app.function(image=attack_image, gpu="L40S", cpu=4, memory=32 * 1024, timeout=8 * 3600, volumes=volumes,
              secrets=[hf_secret])
def l40s_eval(out_rel: str, model_rel: str, data_tag: str, mode: str, lrs: str, tasks: str, pre: bool,
              offline: bool = False):
    """One attack_eval job on one L40S (protocol of modal_attack_eval_l40s.py: paged 8-bit AdamW, gradient
    checkpointing, greedy HS). lrs "" = pre-attack only. offline: lm-eval datasets/classifier from the cache
    (avoids Hugging Face's 1,000 requests / 5 min limit when several jobs run at once)."""
    from huggingface_hub import snapshot_download
    from src.experiments.protocol import read_json
    os.chdir(REMOTE)
    cache_volume.reload()
    output_volume.reload()
    data_dir = Path(DATA) / _name(data_tag, "data tag")
    pinned = read_json(data_dir / "model_revision.json")
    base = snapshot_download(pinned["model"], revision=pinned["revision"], local_files_only=True,
                             allow_patterns=["*.json", "*.safetensors", "tokenizer.model"])
    model = str(Path(OUTPUT) / model_rel) if model_rel else base
    out = Path(OUTPUT) / out_rel
    if ".." in Path(out_rel).parts or ".." in Path(model_rel).parts:
        raise ValueError("relative paths must stay on the output volume")
    done = out / "attack_eval.json"
    if done.exists() and read_json(done).get("complete"):
        return read_json(done)   # resumable: a finished job is reused
    argv = [sys.executable, "-m", "src.experiments.attack_eval", "--model", model, "--tokenizer", base,
            "--data", str(data_dir), "--output", str(out), "--mode", mode, "--lrs", lrs, "--decoding", "greedy",
            "--utility", tasks, "--optim", "paged_adamw_8bit", "--gradient-checkpointing", "--gen-batch-size", "16"]
    if lrs:
        argv += ["--post-utility"] if tasks else []
        if not pre:
            argv.append("--skip-pre")
    if offline:
        os.environ.update(HF_DATASETS_OFFLINE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    _stream(argv, out / "console.log")
    result = read_json(done)
    result["complete"] = True
    done.write_text(json.dumps(result, indent=2), encoding="utf-8")
    output_volume.commit()
    cache_volume.commit()
    return result


def _mean(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return sum(values) / len(values) if values else None


@app.function(image=image, cpu=0.25, memory=1024, timeout=24 * 3600, volumes=volumes)
def pipeline_select(config_text: str, data_tag: str, run_name: str):
    """prepare -> train (models at 100/250/500) -> val attack lr 1e-4 per model -> select -> test evaluation."""
    status = dict(stage="start", started=time.time(), run=run_name)
    def stage(name, **extra):
        status.update(stage=name, **extra)
        _select_write("status.json", status)
        print(f"[select] {name}", flush=True)
    try:
        data_volume.reload()
        if not (Path(DATA) / data_tag / "model_revision.json").exists():
            stage("prepare")
            prepare.remote(config_text, data_tag)
            data_volume.reload()
        stage("train")
        train = gpu_run.remote("train", config_text, data_tag, run_name)
        output_volume.reload()
        train_dir = Path(OUTPUT) / run_name / "train"
        metrics = json.loads((train_dir / "metrics.json").read_text()) if (train_dir / "metrics.json").exists() else {}
        status["train_status"] = metrics.get("status", repr(train))
        candidates = {name: f"{run_name}/train/{name}" for name in ("model-step-100", "model-step-250", "final")
                      if (train_dir / name / "config.json").exists()}
        if not candidates:
            raise RuntimeError(f"training produced no model: {metrics or train}")
        stage("validation attacks", candidates=list(candidates))
        names = list(candidates)
        val = {}
        # The first job downloads MMLU / ARC-e and the classifier into the cache; the others then run offline.
        first = names[0]
        val[first] = l40s_eval.remote(f"{SELECT_DIR}/val/{first}", candidates[first], data_tag, "val", "1e-4",
                                      "mmlu,arc_easy", True)
        calls = {name: l40s_eval.spawn(f"{SELECT_DIR}/val/{name}", candidates[name], data_tag, "val", "1e-4",
                                       "mmlu,arc_easy", True, True) for name in names[1:]}
        for name, call in calls.items():
            val[name] = call.get()
        rows = []
        for name, res in val.items():
            attack = res["attacks"]["0.0001"]
            pre = res["pre"]["utility"]
            rows.append(dict(model=name, pre_mmlu=pre.get("mmlu"), pre_arc=pre.get("arc_easy"),
                             pre_mean=_mean([pre.get("mmlu"), pre.get("arc_easy")]),
                             pre_HS=res["pre"]["harmfulness"]["score"], HS=attack["harmfulness"]["score"],
                             attack_loss=attack["train_loss"], post_mmlu=attack.get("utility", {}).get("mmlu"),
                             post_arc=attack.get("utility", {}).get("arc_easy")))
        eligible = [r for r in rows if r["pre_mean"] is not None and r["pre_mean"] >= BASE_PRE_MMLU_ARC - 0.02]
        ranked = sorted(eligible, key=lambda r: (round(r["HS"], 2), -r["attack_loss"]))
        selection = dict(base_pre_mean=BASE_PRE_MMLU_ARC, base_val_lr1e4=BASE_VAL_LR1E4, candidates=rows,
                         eligible=[r["model"] for r in eligible], selected=ranked[0]["model"] if ranked else None,
                         rule="pre mean(MMLU, ARC-e) >= base - 0.02; lowest HS (2 decimals), then highest attack loss")
        _select_write("selection.json", selection)
        if not ranked:
            stage("no eligible model", selection=selection)
            return selection
        chosen = ranked[0]["model"]
        stage("test evaluation", selected=chosen)
        model_rel = candidates[chosen]
        test = {"pre": l40s_eval.remote(f"{SELECT_DIR}/test/{chosen}/pre", model_rel, data_tag, "test", "",
                                        PAPER_TASKS, True)}
        calls = {lr: l40s_eval.spawn(f"{SELECT_DIR}/test/{chosen}/lr{lr}", model_rel, data_tag, "test", lr,
                                     PAPER_TASKS, False, True) for lr in TEST_LRS}
        for lr, call in calls.items():
            test[lr] = call.get()
        lines = [f"# CBG checkpoint selection ({run_name})", "", "## Validation (lr 1e-4 attack, val split)", "",
                 "| model | pre MMLU | pre ARC-e | pre HS | HS after | attack loss | post MMLU | post ARC-e |",
                 "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(f"| {r['model']} | {r['pre_mmlu']:.3f} | {r['pre_arc']:.3f} | {r['pre_HS']:.3f} | "
                         f"{r['HS']:.3f} | {r['attack_loss']:.3f} | {r['post_mmlu']} | {r['post_arc']} |")
        lines += [f"| base (reference) | 0.464 | 0.738 | | {BASE_VAL_LR1E4['HS']:.3f} | "
                  f"{BASE_VAL_LR1E4['attack_loss']:.3f} | | |", "", f"Selected: **{chosen}** ({selection['rule']})",
                  "", "## Test (selected model)", "",
                  "| phase | HS | HS>0.5 | TruthfulQA | MMLU | HellaSwag | ARC-e | ZS | attack loss |",
                  "|---|---|---|---|---|---|---|---|---|"]
        def table_row(label, part, loss=""):
            u = part.get("utility") or {}
            h = part["harmfulness"]
            cells = [u.get(t) for t in ("truthfulqa_mc1", "mmlu", "hellaswag", "arc_easy")]
            fmt = lambda v: f"{v:.3f}" if isinstance(v, (int, float)) else ""
            return (f"| {label} | {h['score']:.3f} | {h['scores_over_50']:.3f} | " + " | ".join(fmt(c) for c in cells)
                    + f" | {fmt(_mean(cells))} | {loss} |")
        lines.append(table_row("pre", test["pre"]["pre"]))
        for lr in TEST_LRS:
            attack = test[lr]["attacks"][f"{float(lr):g}"]
            lines.append(table_row(f"lr {lr}", attack, f"{attack['train_loss']:.3f}"))
        _select_write("summary.md", "\n".join(lines) + "\n")
        _select_write("test.json", test)
        stage("complete", finished=time.time())
        return selection
    except Exception as error:
        stage("failed", error=repr(error))
        raise


@app.function(image=image, cpu=0.25, memory=1024, volumes=volumes)
def select_status():
    output_volume.reload()
    out = Path(OUTPUT) / SELECT_DIR
    parts = [(out / name).read_text(encoding="utf-8") for name in ("status.json", "summary.md") if (out / name).exists()]
    return "\n".join(parts) or "selection pipeline has not written anything yet"


# ---------------------------------------------------------------------------
# Train once and push the models kept at steps 100 / 250 / 500 to Hugging Face. Runs entirely on Modal.
# ---------------------------------------------------------------------------
PUSH_SNAPSHOTS = (("model-step-100", "step100"), ("model-step-250", "step250"), ("final", "step500"))


@app.function(image=image, cpu=2, memory=16384, timeout=4 * 3600, volumes=volumes, secrets=[hf_secret])
def upload_model(run_name: str, data_tag: str, subdir: str, hf_repo: str, hf_public: bool, hf_dtype: str):
    """Push <run>/train/<subdir> (model + tokenizer, training records, model card) to hf_repo."""
    from src.experiments.hf_upload import upload_run
    from src.experiments.protocol import read_json
    os.chdir(REMOTE)
    output_volume.reload()
    run_dir = Path(OUTPUT) / _name(run_name, "run name") / "train"
    record_path = run_dir / ("hf_upload.json" if subdir == "final" else f"hf_upload_{subdir}.json")
    if record_path.exists() and read_json(record_path).get("repo_id") == hf_repo:
        return read_json(record_path)   # already pushed (a retry)
    pinned = read_json(Path(DATA) / _name(data_tag, "data tag") / "model_revision.json")
    record = upload_run(run_dir, hf_repo, private=not hf_public, dtype=hf_dtype, base_model=pinned["model"],
                        base_revision=pinned["revision"], token=os.environ.get("HF_TOKEN"),
                        staging_dir=str(run_dir), model_subdir=subdir, require_complete=False)
    output_volume.commit()
    print(f"[push] {subdir} -> {record['url']} ({record['commit']})", flush=True)
    return record


@app.function(image=image, cpu=0.25, memory=1024, timeout=24 * 3600, volumes=volumes)
def pipeline_train_push(config_text: str, data_tag: str, run_name: str, hf_repo: str, hf_public: bool,
                        hf_dtype: str):
    """prepare (if needed) -> train -> push every kept model to <hf_repo>-step100 / -step250 / -step500."""
    status_path = Path(OUTPUT) / _name(run_name, "run name") / "train_push_status.json"
    status = dict(stage="start", started=time.time(), run=run_name)
    def stage(name, **extra):
        status.update(stage=name, **extra)
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
        output_volume.commit()
        print(f"[train-push] {name}", flush=True)
    try:
        data_volume.reload()
        if not (Path(DATA) / data_tag / "model_revision.json").exists():
            stage("prepare")
            prepare.remote(config_text, data_tag)
            data_volume.reload()
        stage("train")
        result = gpu_run.remote("train", config_text, data_tag, run_name)
        output_volume.reload()
        train_dir = Path(OUTPUT) / run_name / "train"
        metrics_path = train_dir / "metrics.json"
        status["train_status"] = (json.loads(metrics_path.read_text())["status"] if metrics_path.exists()
                                  else repr(result))
        kept = [(subdir, suffix) for subdir, suffix in PUSH_SNAPSHOTS if (train_dir / subdir / "config.json").exists()]
        if not kept:
            raise RuntimeError(f"training produced no model to push: {result}")
        stage("push", models=[subdir for subdir, _ in kept])
        pushed = {}
        for subdir, suffix in kept:   # one at a time: each upload stages a cast copy on the volume
            pushed[subdir] = upload_model.remote(run_name, data_tag, subdir, f"{hf_repo}-{suffix}", hf_public,
                                                 hf_dtype)["url"]
        stage("complete", pushed=pushed, finished=time.time())
        return pushed
    except Exception as error:
        stage("failed", error=repr(error))
        raise


@app.local_entrypoint()
def main(stage: str = "prepare", run_name: str = "", config: str = "config/cbg_v2_llama2_7b_2xh200.yaml",
         data_tag: str = "seam4k-v1", smoke_steps: int = 6, hf_repo: str = "", hf_public: bool = False,
         hf_dtype: str = "float32", eval_mode: str = "val", attack_lrs: str = "2e-5,5e-5,1e-4",
         utility_tasks: str = "arc_easy", post_utility: bool = False, model_path: str = ""):
    config_text = (ROOT / config).read_text(encoding="utf-8") if not Path(config).is_absolute() \
        else Path(config).read_text(encoding="utf-8")
    if hf_repo:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", hf_repo):
            raise ValueError(f"--hf-repo must be <user or org>/<name>, got {hf_repo!r}")
        if hf_dtype not in {"float32", "bfloat16", "float16"}:
            raise ValueError("--hf-dtype must be float32, bfloat16 or float16")
    if stage == "push-model":   # push one model already on the volume (no training)
        if not (run_name and hf_repo):
            raise ValueError("--stage push-model needs --run-name and --hf-repo (and --model-path <subdir>, default final)")
        record = upload_model.remote(_name(run_name, "run name"), data_tag, model_path or "final", hf_repo, hf_public,
                                     hf_dtype)
        print(json.dumps(record, indent=2, default=str))
        return
    if stage == "train-push":
        if not (run_name and hf_repo):
            raise ValueError("--stage train-push needs --run-name and --hf-repo (models go to <repo>-step100/250/500)")
        call = pipeline_train_push.spawn(config_text, data_tag, _name(run_name, "run name"), hf_repo, hf_public,
                                         hf_dtype)
        print(f"[launcher] train + push for {run_name!r} started on Modal (call {call.object_id}); "
              f"the PC may be switched off.\n"
              f"Status: modal volume get cbg-v2-7b-output {run_name}/train_push_status.json .", flush=True)
        return
    if stage == "select-pipeline":
        run = run_name or "cbg-select"
        call = pipeline_select.spawn(config_text, data_tag, _name(run, "run name"))
        print(f"[launcher] selection pipeline for {run!r} started on Modal (call {call.object_id}); "
              f"the PC may be switched off.\n"
              f"Status : modal run scripts/modal_cbg_v2_7b.py --stage select-status\n"
              f"Results: modal volume get cbg-v2-7b-output {SELECT_DIR}/summary.md .", flush=True)
        return
    if stage == "select-status":
        print(select_status.remote())
        return
    if stage == "c2-pipeline":
        call = pipeline_c2.spawn(config_text, data_tag)
        print(f"[launcher] C2 pipeline started on Modal (call {call.object_id}); the PC may be switched off.\n"
              f"Status : modal run scripts/modal_cbg_v2_7b.py --stage c2-status\n"
              f"Results: modal volume get cbg-v2-7b-output {C2_DIR}/summary.md .", flush=True)
        return
    if stage == "c2-status":
        result = c2_status.remote()
        print(result)
        return
    if stage == "attack-eval":
        if not run_name:
            raise ValueError(f"--stage attack-eval needs --run-name (a trained run, or {BASE_RUN} for the base model)")
        if eval_mode not in {"val", "test"}:
            raise ValueError("--eval-mode must be val (configuration selection) or test (reporting)")
        print(f"[launcher] attack-eval {run_name} ({eval_mode}) on {EVAL_GPU}, lrs {attack_lrs}", flush=True)
        result = gpu_eval.remote(run_name, data_tag, eval_mode, attack_lrs, utility_tasks, post_utility)
    elif stage == "upload":
        if not (run_name and hf_repo):
            raise ValueError("--stage upload needs --run-name and --hf-repo")
        result = upload.remote(run_name, data_tag, hf_repo, hf_public, hf_dtype)
    elif stage == "prepare":
        result = prepare.remote(config_text, data_tag)
    elif stage == "verify":
        result = verify.remote()
    elif stage in {"radius", "hvp-check", "smoke", "train", "geometry", "calibrate"}:
        if not run_name:
            raise ValueError("--run-name is required for GPU stages")
        import yaml
        required = yaml.safe_load(config_text)["execution"]["devices"]
        if isinstance(required, int) and _gpu_count(GPU) < required:
            raise ValueError(f"the config needs {required} GPUs but the container requests {GPU!r}; "
                             f"unset CBG_V2_MODAL_GPU or set it to e.g. 'H200:{required}'")
        memory = CHECK_MEMORY if stage in CHECK_STAGES else TRAIN_MEMORY
        print(f"[launcher] GPU stage {stage!r} on {GPU}, {GPU_CPU:g} CPU, {memory // 1024} GiB RAM", flush=True)
        if stage in CHECK_STAGES:
            result = gpu_check.remote(stage, config_text, data_tag, run_name)
        else:
            result = gpu_run.remote(stage, config_text, data_tag, run_name, smoke_steps,
                                    hf_repo if stage == "train" else "", hf_public, hf_dtype,
                                    model_path if stage in {"geometry", "calibrate"} else "")
    else:
        raise ValueError("stage must be prepare, verify, radius, hvp-check, smoke, train, attack-eval or upload")
    print(json.dumps(result, indent=2, default=str))
    if isinstance(result, dict) and result.get("status") == "failed":
        raise SystemExit(f"stage {stage} failed: {result['error']}")
