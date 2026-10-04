"""Detached CTRAP comparison run, using CTRAP/trainer.py's original loss."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import modal

repo = Path(__file__).resolve().parent.parent
app = modal.App("ctrap-full-comparison")
prep_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("requests==2.32.3")
    .add_local_dir(str(repo / "scripts"), remote_path="/root/seam/scripts")
)
# CTRAP code targets Transformers 4.45.2 and PEFT 0.5.0. Keep its trainer in a
# separate image from SEAM's newer evaluator environment.
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
data_volume = modal.Volume.from_name("defense-compare-data", create_if_missing=True)
hf_volume = modal.Volume.from_name("seam-main-hf-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("defense-compare-results", create_if_missing=True)


def _paths(seed, run_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", run_id):
        raise ValueError("run_id must contain 1-64 letters, digits, _ or -")
    return (Path("/data") / f"ctrap-seed{seed}-{run_id}",
            Path("/results") / f"ctrap-seed{seed}-{run_id}")


@app.function(
    image=prep_image, cpu=2, memory=4096, timeout=24 * 60 * 60,
    volumes={"/data": data_volume, "/results": results_volume},
)
def pipeline(seed: int, run_id: str, test_size: int, max_steps: int):
    sys.path.insert(0, "/root/seam/scripts")
    from modal_compare_common import prepare_split, write_json, ATTACK_LRS, TASKS

    if max_steps != -1 and max_steps <= 0:
        raise ValueError("max_steps must be -1 or positive")

    data_dir, output = _paths(seed, run_id)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    status = {"method": "CTRAP", "run_id": run_id, "seed": seed, "state": "running"}

    def stage(name):
        status["stage"] = name
        write_json(output / "status.json", status)
        results_volume.commit()
        print("CTRAP_STAGE", name, flush=True)

    try:
        stage("prepare_common_split")
        split = prepare_split(seed, data_dir)
        defense_rows = json.loads((data_dir / "ctrap_defense.json").read_text(encoding="utf-8"))
        if any("\nAnswer: " not in row["refusal"] for row in defense_rows):
            raise ValueError("RepNoise refusals do not match CTRAP's required Question/Answer format")
        data_volume.commit()
        write_json(output / "manifest.json", {
            "method": "CTRAP", "seed": seed, "run_id": run_id,
            "protocol": "SEAM-style shared attack/eval, not CTRAP paper table",
            "split": split, "defense_safe_pairs": 5000,
            "defense_harmful_pairs": 5000, "defense_ultrachat_pairs": 5000,
            "defense_epochs": 20, "defense_max_steps": max_steps,
            "training_budget": (f"{max_steps}_optimizer_steps" if max_steps > 0
                                else "20_epochs"),
            "nominal_batch": 10,
            "alignment_microbatch": 5, "harmful_batch": 10,
            "defense_lr": 5e-4, "alpha": 0.1, "lambda": 0.1,
            "lora_rank": 32, "lora_alpha": 4, "lora_modules": ["q_proj", "k_proj", "v_proj"],
            "attack_steps": 250, "attack_lrs": ATTACK_LRS,
            "eval_tasks": TASKS, "test_size": test_size,
            "gpu": "one H100 at a time",
            "training_provenance": "Original CTRAPTrainer with data-loading adapter",
        })
        stage(f"defense_{max_steps}_steps" if max_steps > 0 else "defense_20_epochs")
        train_ctrap.remote(seed, run_id, max_steps)
        results_volume.reload()
        stage("evaluation_sweep")
        evaluate_ctrap.remote(seed, run_id, test_size)
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


@app.function(
    image=train_image, gpu="H100", cpu=8, memory=65536, timeout=24 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/data": data_volume, "/hf": hf_volume, "/results": results_volume},
)
def train_ctrap(seed: int, run_id: str, max_steps: int):
    data_dir, output = _paths(seed, run_id)
    os.environ.update({"HF_HOME": "/hf", "WANDB_MODE": "disabled", "PYTHONUNBUFFERED": "1"})
    subprocess.run([
        sys.executable, "/root/seam/scripts/ctrap_train.py",
        "--defense", str(data_dir / "ctrap_defense.json"),
        "--output", str(output / "defense"), "--seed", str(seed),
        "--max-steps", str(max_steps),
    ], cwd="/root/seam", check=True)
    if not (output / "defense" / "model" / "config.json").exists():
        raise FileNotFoundError("CTRAP merged checkpoint missing")
    results_volume.commit()
    hf_volume.commit()


@app.function(
    image=eval_image, gpu="H100", cpu=8, memory=65536, timeout=24 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/data": data_volume, "/hf": hf_volume, "/results": results_volume},
)
def evaluate_ctrap(seed: int, run_id: str, test_size: int):
    sys.path.insert(0, "/root/seam/scripts")
    from modal_compare_common import evaluate_sweep, write_json

    data_dir, output = _paths(seed, run_id)
    os.environ["HF_HOME"] = "/hf"

    def stage(name):
        write_json(output / "eval_status.json", {"stage": name, "state": "running"})
        results_volume.commit()

    evaluate_sweep("CTRAP", seed, test_size, output, data_dir / "attack.json",
                   str(output / "defense" / "model"), stage)
    write_json(output / "eval_status.json", {"stage": "complete", "state": "complete"})
    results_volume.commit()
    hf_volume.commit()


@app.local_entrypoint()
def main(seed: int = 42, run_id: str = "", test_size: int = 100,
         max_steps: int = -1):
    if not run_id or test_size < 0 or (max_steps != -1 and max_steps <= 0):
        raise ValueError("Pass unique --run-id, nonnegative --test-size, and max_steps=-1 or positive")
    call = pipeline.spawn(seed, run_id, test_size, max_steps)
    print(f"Submitted Modal call: {call.object_id}")
    print(f"Results: defense-compare-results/ctrap-seed{seed}-{run_id}")


@app.function(
    image=prep_image, cpu=2, memory=4096, timeout=2 * 60 * 60,
    volumes={"/data": data_volume, "/results": results_volume},
)
def smoke_pipeline(seed: int, run_id: str, smoke_export: bool = False):
    sys.path.insert(0, "/root/seam/scripts")
    from modal_compare_common import prepare_split, write_json

    data_dir, output = _paths(seed, run_id)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    status = {"method": "CTRAP", "mode": "smoke", "seed": seed, "run_id": run_id,
              "smoke_export": smoke_export,
              "stage": "prepare_data", "state": "running"}
    write_json(output / "status.json", status)
    results_volume.commit()
    try:
        prepare_split(seed, data_dir)
        defense_rows = json.loads((data_dir / "ctrap_defense.json").read_text(encoding="utf-8"))
        if any("\nAnswer: " not in row["refusal"] for row in defense_rows):
            raise ValueError("RepNoise refusals do not match CTRAP's Question/Answer format")
        data_volume.commit()
        status["stage"] = "train_2_steps"
        write_json(output / "status.json", status)
        results_volume.commit()
        metrics = smoke_train_ctrap.remote(seed, run_id, smoke_export)
        results_volume.reload()
        status.update({"stage": "complete", "state": "complete", "step_losses": metrics["step_losses"]})
        write_json(output / "status.json", status)
        results_volume.commit()
        return {"result": str(output), "metrics": metrics}
    except Exception as exc:
        status.update({"state": "failed", "error": repr(exc)})
        write_json(output / "status.json", status)
        results_volume.commit()
        raise


@app.function(
    image=train_image, gpu="H100", cpu=8, memory=65536, timeout=60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/data": data_volume, "/hf": hf_volume, "/results": results_volume},
)
def smoke_train_ctrap(seed: int, run_id: str, smoke_export: bool = False):
    data_dir, output = _paths(seed, run_id)
    os.environ.update({"HF_HOME": "/hf", "WANDB_MODE": "disabled", "PYTHONUNBUFFERED": "1"})
    train_output = Path("/tmp/ctrap-smoke-export") if smoke_export else output / "defense"
    command = [
        sys.executable, "/root/seam/scripts/ctrap_train.py",
        "--defense", str(data_dir / "ctrap_defense.json"),
        "--output", str(train_output), "--seed", str(seed),
        "--max-steps", "2", "--smoke",
    ]
    if smoke_export:
        command.append("--smoke-export")
    subprocess.run(command, cwd="/root/seam", check=True)
    metrics = json.loads((train_output / "train_metrics.json").read_text(encoding="utf-8"))
    results_volume.commit()
    hf_volume.commit()
    return metrics


@app.local_entrypoint()
def smoke_main(seed: int = 42, run_id: str = "", smoke_export: bool = False):
    if not run_id:
        raise ValueError("Pass a unique --run-id")
    result = smoke_pipeline.remote(seed, run_id, smoke_export)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
