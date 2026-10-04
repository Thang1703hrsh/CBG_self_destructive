"""One-seed SEAM main-result sweep on Modal; see Run-SEAMFull.ps1.

This follows the public-data option in the official SEAM README. The public
RepNoise refusals are not verified to be the GPT-4o refusals described in the
paper. The published 20-seed mean/std requires 20 separate runs.
"""

import hashlib
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

import modal

app = modal.App("seam-full-main-result")
repo = Path('/root/seam') if os.environ.get('MODAL_IS_REMOTE') == '1' else Path(__file__).resolve().parent.parent
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(repo / "requirements.txt"))
    .pip_install("wandb==0.19.8", "requests==2.32.3")
    .add_local_file(str(repo / "requirements.txt"), remote_path="/root/seam/requirements.txt")
    .add_local_dir(str(repo / "src"), remote_path="/root/seam/src")
    .add_local_dir(str(repo / "config"), remote_path="/root/seam/config")
)
data_volume = modal.Volume.from_name("seam-full-data", create_if_missing=True)
hf_volume = modal.Volume.from_name("seam-main-hf-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("seam-full-results", create_if_missing=True)

PUBLIC_DATA_URL = (
    "https://raw.githubusercontent.com/domenicrosati/representation-noising/"
    "main/data/beavertails_with_refusals_train.json"
)
MODEL = "meta-llama/Llama-2-7b-chat-hf"
ATTACK_LRS = ("2e-5", "5e-5", "8e-5", "1e-4", "2e-4")
TASKS = ("arc_easy", "mmlu", "truthfulqa_mc2", "hellaswag")


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=327680,  # FP32 gradients, snapshots and AdamW moments on CPU.
    timeout=24 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/data": data_volume, "/hf": hf_volume, "/results": results_volume},
)
def run_full(seed: int, run_id: str, test_size: int = 100,
             train_only: bool = False, train_steps: int = 500):
    import requests

    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", run_id):
        raise ValueError("run_id must contain only letters, digits, _ or -")
    if test_size < 0:
        raise ValueError("test_size must be nonnegative")
    if not 1 <= train_steps <= 500:
        raise ValueError("train_steps must be between 1 and 500")
    if train_steps != 500 and not train_only:
        raise ValueError("Short training is diagnostic only; use train_only")

    output = Path("/results") / f"seam-full-seed{seed}-{run_id}"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {output}")
    output.mkdir(parents=True)
    data_dir = Path("/data") / f"seam-full-seed{seed}-{run_id}"
    data_dir.mkdir(parents=True)

    env = os.environ.copy()
    env.update({
        "HF_HOME": "/hf",
        "WANDB_MODE": "disabled",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
    })
    status = {"run_id": run_id, "seed": seed, "stage": "prepare_data", "state": "running"}

    def stage(name):
        status["stage"] = name
        write_json(output / "status.json", status)
        results_volume.commit()
        print("SEAM_STAGE", name, flush=True)

    def invoke(arguments, visible_devices):
        call_env = env.copy()
        call_env["CUDA_VISIBLE_DEVICES"] = visible_devices
        print("SEAM_COMMAND", " ".join(arguments), flush=True)
        subprocess.run(arguments, cwd="/root/seam", env=call_env, check=True)

    try:
        stage("prepare_data")
        response = requests.get(PUBLIC_DATA_URL, timeout=120)
        response.raise_for_status()
        raw_sha256 = hashlib.sha256(response.content).hexdigest()
        records = [
            row for row in response.json()
            if row.get("is_safe") is False and row.get("prompt")
            and row.get("response") and row.get("refusal")
        ]
        random.Random(seed).shuffle(records)
        if len(records) < 5000:
            raise ValueError(f"Need 5,000 eligible disjoint rows, found {len(records)}")
        defense_path = data_dir / "defense.json"
        attack_path = data_dir / "attack.json"
        write_json(defense_path, records[:4000])
        write_json(attack_path, records[4000:5000])
        data_volume.commit()
        write_json(output / "manifest.json", {
            "seed": seed, "run_id": run_id, "source": PUBLIC_DATA_URL,
            "source_sha256": raw_sha256, "refusal_provenance": "RepNoise public data; GPT-4o not verified",
            "defense_count": 4000, "attack_count": 1000,
            "defense_sha256": hashlib.sha256(defense_path.read_bytes()).hexdigest(),
            "attack_sha256": hashlib.sha256(attack_path.read_bytes()).hexdigest(),
            "train_steps": train_steps, "train_batch_size": 8,
            "mode": "train_only" if train_only else "full",
            "weight_dtype": "float32", "gradient_storage": "cpu",
            "optimizer": "AdamW FP32, CPU moments", "autocast": False,
            "code_hashes": {str(p): hashlib.sha256((repo / p).read_bytes()).hexdigest()
                            for p in ("src/core/trainer.py", "src/core/numerics.py", "src/train.py",
                                      "src/data_processing/_datasets.py", "config/train.yaml", "requirements.txt")},
            "attack_steps": 250, "attack_batch_size": 4,
            "attack_learning_rates": ATTACK_LRS,
            "eval_tasks": TASKS, "test_size": test_size,
            "gpu": "H100 (FP32 defense with CPU offload)", "cpu_memory_mib": 327680,
        })
        results_volume.commit()

        stage(f"defense_{train_steps}_steps")
        invoke([
            sys.executable, "-m", "src.train",
            "--data_path", str(defense_path),
            "--output_dir", str(output / "defense"),
            "--defense_size", "4000", "--benign_seed", str(seed),
            "--seed", str(seed), "--max_steps", str(train_steps),
            "--per_device_train_batch_size", "8",
            "--save_strategy", "steps", "--save_steps", str(train_steps),
            "--report_to", "none", "--wandb_run_name", f"seam-full-{run_id}",
        ], "0")
        checkpoint = output / "defense" / f"checkpoint-{train_steps}"
        if not (checkpoint / "config.json").exists():
            raise FileNotFoundError(f"Missing defense checkpoint: {checkpoint}")
        results_volume.commit()

        if train_only:
            status.update(stage="complete", state="complete", checkpoint=str(checkpoint))
            write_json(output / "status.json", status)
            results_volume.commit()
            hf_volume.commit()
            print("SEAM_TRAIN_COMPLETE", checkpoint, flush=True)
            return str(checkpoint)

        summary = {"manifest": str(output / "manifest.json"), "metrics": {}}
        for model_label, model_path in (("base", MODEL), ("seam", str(checkpoint))):
            for lr in (None, *ATTACK_LRS):
                key = f"{model_label}-pre" if lr is None else f"{model_label}-attack-lr{lr}"
                stage(key)
                metric_path = output / "metrics" / f"{key}.json"
                args = [
                    sys.executable, "-m", "src.eval",
                    "--model_name", model_path,
                    "--tokenizer_name", MODEL,
                    "--data_path", str(attack_path),
                    "--output_dir", str(output / "attack_work" / key),
                    "--seed", str(seed), "--report_to", "none",
                    "--pre_utility", "false", "--post_utility", "true",
                    "--gpt_eval", "false", "--test_size", str(test_size),
                    "--attack", "false" if lr is None else "true",
                    "--attack_dataset", "true", "--attack_size", "1000",
                    "--per_device_train_batch_size", "4", "--utility_batch_size", "8",
                    "--gradient_checkpointing", "true",
                    "--max_steps", "250", "--save_strategy", "no",
                    "--result_path", str(metric_path),
                    "--utility_tasks", *TASKS,
                    "--wandb_run_name", key,
                ]
                if lr is not None:
                    args.extend(("--learning_rate", lr))
                invoke(args, "0")
                summary["metrics"][key] = json.loads(metric_path.read_text(encoding="utf-8"))
                results_volume.commit()

        write_json(output / "summary.json", summary)
        status.update({"stage": "complete", "state": "complete"})
        write_json(output / "status.json", status)
        results_volume.commit()
        hf_volume.commit()
        print("SEAM_COMPLETE", output, flush=True)
        return str(output)
    except Exception as exc:
        status.update({"state": "failed", "error": repr(exc)})
        write_json(output / "status.json", status)
        results_volume.commit()
        raise


@app.local_entrypoint()
def main(seed: int = 42, run_id: str = "", test_size: int = 100,
         train_only: bool = False, train_steps: int = 500):
    if not run_id:
        raise ValueError("Pass a unique --run-id from Run-SEAMFull.ps1")
    call = run_full.spawn(seed, run_id, test_size, train_only, train_steps)
    print(f"Submitted Modal function call: {call.object_id}")
    print(f"Results volume: seam-full-results/seam-full-seed{seed}-{run_id}")
