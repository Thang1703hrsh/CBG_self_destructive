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

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install_from_requirements(str(ROOT / "requirements.txt"))
    .env({"PYTHONPATH": REMOTE, "USE_TF": "0", "TRANSFORMERS_NO_TF": "1", "HF_HOME": CACHE,
          "HF_DATASETS_CACHE": CACHE + "/datasets", "TOKENIZERS_PARALLELISM": "false",
          "OMP_NUM_THREADS": "4", "WANDB_MODE": "disabled",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
          # Deterministic cuBLAS: the two perturbed passes must differ only by the perturbation.
          "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    .add_local_dir(str(ROOT / "src"), REMOTE + "/src")
    .add_local_dir(str(ROOT / "tests"), REMOTE + "/tests")
    .add_local_dir(str(ROOT / "config"), REMOTE + "/config")
    .add_local_file(str(ROOT / "cbg_reference.py"), REMOTE + "/cbg_reference.py")
)
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
            hf_repo: str = "", hf_public: bool = False, hf_dtype: str = "float32"):
    """train / smoke."""
    return _gpu_entry(stage, config_text, data_tag, run_name, smoke_steps, hf_repo, hf_public, hf_dtype)


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


def _gpu_entry(stage, config_text, data_tag, run_name, smoke_steps, hf_repo, hf_public, hf_dtype):
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
            return _gpu_stage(stage, config_text, data_tag, run_name, smoke_steps, hf_repo, hf_public, hf_dtype)
        except Exception as error:
            return dict(status="failed", stage=stage, error=repr(error))
    return _gpu_stage(stage, config_text, data_tag, run_name, smoke_steps, hf_repo, hf_public, hf_dtype)


def _gpu_stage(stage, config_text, data_tag, run_name, smoke_steps, hf_repo, hf_public, hf_dtype):
    from huggingface_hub import snapshot_download
    from src.experiments.protocol import read_json
    data_dir = Path(DATA) / _name(data_tag, "data tag")
    if not (data_dir / "manifest.json").exists():
        raise FileNotFoundError("run --stage prepare first")
    pinned = read_json(data_dir / "model_revision.json")
    model_path = snapshot_download(pinned["model"], revision=pinned["revision"], local_files_only=True,
                                   allow_patterns=["*.json", "*.safetensors", "tokenizer.model"])
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
    if stage == "train" and hf_repo:
        if (run_dir / "hf_upload.json").exists() and read_json(run_dir / "hf_upload.json")["repo_id"] == hf_repo:
            result["hf_upload"] = read_json(run_dir / "hf_upload.json")   # a retry after a finished upload
        else:
            try:
                result["hf_upload"] = _upload(run_dir, hf_repo, hf_public, hf_dtype, data_dir)
            except Exception as error:   # the trained model is safe on the volume; never retrain for this
                print(f"[launcher] HF upload failed: {error!r}; retry with --stage upload", flush=True)
                result["hf_upload"] = dict(status="failed", error=repr(error))
    return result


@app.local_entrypoint()
def main(stage: str = "prepare", run_name: str = "", config: str = "config/cbg_v2_llama2_7b_2xh200.yaml",
         data_tag: str = "seam4k-v1", smoke_steps: int = 6, hf_repo: str = "", hf_public: bool = False,
         hf_dtype: str = "float32", eval_mode: str = "val", attack_lrs: str = "2e-5,5e-5,1e-4",
         utility_tasks: str = "arc_easy", post_utility: bool = False):
    config_text = (ROOT / config).read_text(encoding="utf-8") if not Path(config).is_absolute() \
        else Path(config).read_text(encoding="utf-8")
    if hf_repo:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", hf_repo):
            raise ValueError(f"--hf-repo must be <user or org>/<name>, got {hf_repo!r}")
        if hf_dtype not in {"float32", "bfloat16", "float16"}:
            raise ValueError("--hf-dtype must be float32, bfloat16 or float16")
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
    elif stage in {"radius", "hvp-check", "smoke", "train"}:
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
                                    hf_repo if stage == "train" else "", hf_public, hf_dtype)
    else:
        raise ValueError("stage must be prepare, verify, radius, hvp-check, smoke, train, attack-eval or upload")
    print(json.dumps(result, indent=2, default=str))
    if isinstance(result, dict) and result.get("status") == "failed":
        raise SystemExit(f"stage {stage} failed: {result['error']}")
