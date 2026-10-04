"""CTRAP L40S variants: defense-only, or defense plus LoRA-attack evaluation.

These variants do not replace the H100 full-model-attack comparison. Each worker
requests two physical CPU cores, 16 GiB RAM and one L40S. A small CPU-only
coordinator remains alive while it waits for the workers.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import modal

repo = Path(__file__).resolve().parent.parent
app = modal.App("ctrap-l40s-comparison")
prep_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("requests==2.32.3")
    .add_local_dir(str(repo / "scripts"), remote_path="/root/seam/scripts")
)
train_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.1.0", "transformers==4.45.2", "peft==0.5.0",
        "accelerate==1.0.1", "datasets==2.14.7", "pyarrow==14.0.1",
        "numpy==1.24.1", "huggingface-hub==0.25.2", "wandb==0.16.1",
        "sentencepiece==0.1.99", "scipy==1.11.3", "packaging==23.2",
    )
    .add_local_dir(str(repo / "CTRAP"), remote_path="/root/seam/CTRAP")
    .add_local_dir(str(repo / "scripts"), remote_path="/root/seam/scripts")
)
eval_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(repo / "requirements.txt"))
    .pip_install("wandb==0.19.8", "requests==2.32.3")
    .add_local_dir(str(repo / "src"), remote_path="/root/seam/src")
    .add_local_dir(str(repo / "config"), remote_path="/root/seam/config")
    .add_local_dir(str(repo / "scripts"), remote_path="/root/seam/scripts")
)
hub_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub==0.36.2", "hf_xet")
    .add_local_dir(str(repo / "scripts"), remote_path="/root/seam/scripts")
)
data_volume = modal.Volume.from_name("defense-compare-data", create_if_missing=True)
hf_volume = modal.Volume.from_name("seam-main-hf-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("defense-compare-results", create_if_missing=True)


def _paths(seed: int, run_id: str):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", run_id):
        raise ValueError("run_id must contain 1-64 letters, digits, _ or -")
    name = f"ctrap-l40s-seed{seed}-{run_id}"
    return Path("/data") / name, Path("/results") / name


@app.function(
    image=prep_image, cpu=2, memory=16384, timeout=24 * 60 * 60,
    volumes={"/data": data_volume},
)
def prepare(seed: int, run_id: str):
    sys.path.insert(0, "/root/seam/scripts")
    from modal_compare_common import prepare_split

    data_dir, _ = _paths(seed, run_id)
    split = prepare_split(seed, data_dir)
    defense_rows = json.loads((data_dir / "ctrap_defense.json").read_text(encoding="utf-8"))
    if any("\nAnswer: " not in row["refusal"] for row in defense_rows):
        raise ValueError("RepNoise refusals do not match CTRAP's Question/Answer format")
    data_volume.commit()
    return split


@app.function(
    image=train_image, gpu="L40S", cpu=2, memory=16384, timeout=24 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/data": data_volume, "/hf": hf_volume, "/results": results_volume},
)
def train(seed: int, run_id: str, max_steps: int):
    data_dir, output = _paths(seed, run_id)
    os.environ.update({"HF_HOME": "/hf", "WANDB_MODE": "disabled", "PYTHONUNBUFFERED": "1"})
    subprocess.run([
        sys.executable, "/root/seam/scripts/ctrap_train.py",
        "--defense", str(data_dir / "ctrap_defense.json"),
        "--output", str(output / "defense"), "--seed", str(seed),
        "--max-steps", str(max_steps), "--direct-gpu-load",
    ], cwd="/root/seam", check=True)
    if not (output / "defense" / "model" / "config.json").is_file():
        raise FileNotFoundError("CTRAP merged checkpoint missing")
    metrics = json.loads((output / "defense" / "train_metrics.json").read_text(encoding="utf-8"))
    if metrics["global_step"] != max_steps:
        raise RuntimeError(f"Expected {max_steps} steps, got {metrics['global_step']}")
    results_volume.commit()
    hf_volume.commit()


@app.function(
    image=eval_image, gpu="L40S", cpu=2, memory=16384, timeout=24 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/data": data_volume, "/hf": hf_volume, "/results": results_volume},
)
def evaluate_lora(seed: int, run_id: str, test_size: int):
    sys.path.insert(0, "/root/seam/scripts")
    from modal_compare_common import evaluate_sweep, write_json

    data_dir, output = _paths(seed, run_id)
    os.environ["HF_HOME"] = "/hf"

    def stage(name):
        write_json(output / "eval_status.json", {"stage": name, "state": "running"})
        results_volume.commit()

    evaluate_sweep(
        "CTRAP", seed, test_size, output, data_dir / "attack.json",
        str(output / "defense" / "model"), stage,
        lora_attack_config="ctrap", direct_gpu_load=True,
    )
    write_json(output / "eval_status.json", {"stage": "complete", "state": "complete"})
    results_volume.commit()
    hf_volume.commit()


@app.function(
    image=hub_image, cpu=1, memory=2048, timeout=30 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def hf_repo_preflight(repo_id: str):
    sys.path.insert(0, "/root/seam/scripts")
    from hf_checkpoint_push import ensure_private_empty_repo

    return ensure_private_empty_repo(repo_id)


@app.function(
    image=hub_image, cpu=2, memory=4096, timeout=24 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/results": results_volume},
)
def push_ctrap_hf(seed: int, run_id: str, repo_id: str, expected_steps: int):
    sys.path.insert(0, "/root/seam/scripts")
    from hf_checkpoint_push import push_checkpoint

    _, output = _paths(seed, run_id)
    return push_checkpoint(repo_id, output, adapter=True, expected_steps=expected_steps)


@app.function(
    image=prep_image, cpu=0.125, memory=512, timeout=24 * 60 * 60,
    volumes={"/results": results_volume},
)
def pipeline(seed: int, run_id: str, max_steps: int, test_size: int,
             mode: str, hf_repo: str = ""):
    sys.path.insert(0, "/root/seam/scripts")
    from modal_compare_common import ATTACK_LRS, TASKS, write_json

    if max_steps <= 0 or test_size < 0 or mode not in {"train_only", "full_lora", "train_push"}:
        raise ValueError("Invalid max_steps, test_size, or mode")
    if mode == "train_push" and not hf_repo:
        raise ValueError("HF push mode requires --hf-repo")
    if mode != "train_push" and hf_repo:
        raise ValueError("--hf-repo is only allowed in train_push mode")
    _, output = _paths(seed, run_id)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    status = {"method": "CTRAP", "mode": mode, "run_id": run_id,
              "seed": seed, "state": "running"}

    def stage(name):
        status["stage"] = name
        write_json(output / "status.json", status)
        results_volume.commit()
        print("CTRAP_L40S_STAGE", name, flush=True)

    try:
        stage("prepare_common_split")
        split = prepare.remote(seed, run_id)
        write_json(output / "manifest.json", {
            "method": "CTRAP", "mode": mode, "seed": seed, "run_id": run_id,
            "protocol": ("train only; no attack/evaluation" if mode != "full_lora"
                         else "LoRA attack on L40S; not comparable to full-model attack runs"),
            "split": split, "defense_safe_pairs": 5000,
            "defense_harmful_pairs": 5000, "defense_ultrachat_pairs": 5000,
            "defense_max_steps": max_steps, "defense_lr": 5e-4,
            "defense_lora_rank": 32, "defense_lora_alpha": 4,
            "defense_lora_modules": ["q_proj", "k_proj", "v_proj"],
            "attack_type": "LoRA" if mode == "full_lora" else "none",
            "attack_lora_rank": 32 if mode == "full_lora" else None,
            "attack_lora_alpha": 4 if mode == "full_lora" else None,
            "attack_steps": 250 if mode == "full_lora" else 0,
            "attack_lrs": ATTACK_LRS if mode == "full_lora" else [],
            "eval_tasks": TASKS if mode == "full_lora" else [],
            "test_size": test_size if mode == "full_lora" else 0,
            "gpu": "one L40S per GPU stage", "worker_cpu": 2,
            "worker_memory_mib": 16384,
            "coordinator_cpu": 0.125, "coordinator_memory_mib": 512,
            "hf_repo": hf_repo or None,
            "training_provenance": "Original CTRAPTrainer with data-loading adapter",
        })
        if mode == "train_push":
            stage("hf_private_repo_preflight")
            hf_repo_preflight.remote(hf_repo)
        stage(f"defense_{max_steps}_steps")
        train.remote(seed, run_id, max_steps)
        results_volume.reload()
        if mode == "full_lora":
            stage("lora_attack_evaluation_sweep")
            evaluate_lora.remote(seed, run_id, test_size)
            results_volume.reload()
        if mode == "train_push":
            stage("hf_upload")
            status["hf_url"] = push_ctrap_hf.remote(seed, run_id, hf_repo, max_steps)
            results_volume.reload()
        status.update({"state": "complete", "stage": "complete"})
        write_json(output / "status.json", status)
        results_volume.commit()
        return str(output)
    except Exception as exc:
        status.update({"state": "failed", "error": repr(exc)})
        write_json(output / "status.json", status)
        results_volume.commit()
        raise


@app.local_entrypoint()
def train_main(seed: int = 42, run_id: str = "", max_steps: int = 500,
               hf_repo: str = ""):
    if not run_id or max_steps <= 0:
        raise ValueError("Pass a unique --run-id and positive --max-steps")
    mode = "train_push" if hf_repo else "train_only"
    call = pipeline.spawn(seed, run_id, max_steps, 0, mode, hf_repo)
    print(f"Submitted Modal call: {call.object_id}")
    print(f"Results: defense-compare-results/ctrap-l40s-seed{seed}-{run_id}")
    if hf_repo:
        print(f"Private HF repo: https://huggingface.co/{hf_repo}")


@app.local_entrypoint()
def full_main(seed: int = 42, run_id: str = "", max_steps: int = 500,
              test_size: int = 100):
    if not run_id or max_steps <= 0 or test_size < 0:
        raise ValueError("Pass a unique --run-id, positive --max-steps, and nonnegative --test-size")
    call = pipeline.spawn(seed, run_id, max_steps, test_size, "full_lora", "")
    print(f"Submitted Modal call: {call.object_id}")
    print(f"Results: defense-compare-results/ctrap-l40s-seed{seed}-{run_id}")


@app.function(
    image=prep_image, cpu=0.125, memory=512, timeout=24 * 60 * 60,
    volumes={"/results": results_volume},
)
def retry_ctrap_push(seed: int, run_id: str, repo_id: str):
    sys.path.insert(0, "/root/seam/scripts")
    from modal_compare_common import write_json

    _, output = _paths(seed, run_id)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("mode") != "train_push" or manifest.get("seed") != seed
            or manifest.get("run_id") != run_id or manifest.get("hf_repo") != repo_id):
        raise ValueError("CTRAP train-and-push manifest does not match this upload retry")
    expected_steps = manifest.get("defense_max_steps")
    if not isinstance(expected_steps, int) or expected_steps <= 0:
        raise ValueError("CTRAP manifest has no valid defense_max_steps")
    url = push_ctrap_hf.remote(seed, run_id, repo_id, expected_steps)
    results_volume.reload()
    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    status.update({"state": "complete", "stage": "complete", "hf_url": url})
    status.pop("error", None)
    write_json(output / "status.json", status)
    results_volume.commit()
    return url


@app.local_entrypoint()
def retry_ctrap_push_main(seed: int = 42, run_id: str = "", hf_repo: str = ""):
    if not run_id or not hf_repo:
        raise ValueError("Pass prior train-and-push --run-id and matching --hf-repo")
    call = retry_ctrap_push.spawn(seed, run_id, hf_repo)
    print(f"Submitted HF upload retry: {call.object_id}")
