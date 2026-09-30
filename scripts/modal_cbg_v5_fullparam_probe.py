"""Run one exact full-parameter CBG V5 feasibility step on strictly two H100s."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/workspace"
HF_ROOT = "/vol/hf"
DATA_ROOT = "/vol/data"
OUTPUT_ROOT = "/vol/output"
MODEL_SOURCE = Path("/vol/hf/models/Qwen2.5-3B-Instruct")
REFUSAL_SOURCE = Path("/vol/data/beavertails_with_refusals_train.json")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install_from_requirements(str(ROOT / "requirements.txt"))
    .env({
        "PYTHONPATH": REMOTE,
        "USE_TF": "0",
        "TRANSFORMERS_NO_TF": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "WANDB_MODE": "offline",
        "CBG_SINGLE_MODEL_GPU": "1",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
    })
    .add_local_dir(str(ROOT / "src"), f"{REMOTE}/src")
    .add_local_file(
        str(ROOT / "config" / "cbg_v5_qwen25_3b_fullparam_probe.yaml"),
        f"{REMOTE}/config/cbg_v5_qwen25_3b_fullparam_probe.yaml",
    )
)

app = modal.App("cbg-v5-qwen25-3b-fullparam-probe")
hf_volume = modal.Volume.from_name("cbg-v5-hf-cache", create_if_missing=False)
data_volume = modal.Volume.from_name("cbg-v5-data", create_if_missing=False)
output_volume = modal.Volume.from_name("cbg-v5-fullparam-probes", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100:2",
    cpu=2.0,
    memory=64 * 1024,
    timeout=4 * 60 * 60,
    volumes={HF_ROOT: hf_volume, DATA_ROOT: data_volume, OUTPUT_ROOT: output_volume},
)
def run_probe() -> dict[str, object]:
    import torch

    if torch.cuda.device_count() != 2:
        raise RuntimeError(f"Expected two H100s, received {torch.cuda.device_count()} GPUs")
    gpu_info = []
    for index in range(2):
        props = torch.cuda.get_device_properties(index)
        gpu_info.append({"name": props.name, "memory_gib": props.total_memory / 1024**3})
        # Modal may satisfy an H100 request with a newer accelerator. This
        # experiment requires an apples-to-apples H100 measurement, so reject
        # substitutions before copying assets or constructing either model.
        normalized_name = props.name.upper()
        if "H100" not in normalized_name or "H200" in normalized_name:
            raise RuntimeError(
                f"Strict H100-only probe rejected GPU {index}: {props.name!r}. "
                "No training work was started."
            )
        if props.total_memory < 75 * 1024**3:
            raise RuntimeError("Full-parameter probe requires two 80GB GPUs")
    if not MODEL_SOURCE.is_dir() or not REFUSAL_SOURCE.is_file():
        raise FileNotFoundError("Prepared CBG assets are missing")

    local_model = Path("/tmp/Qwen2.5-3B-Instruct")
    shutil.copytree(MODEL_SOURCE, local_model, dirs_exist_ok=True)
    data_dir = Path(REMOTE) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REFUSAL_SOURCE, data_dir / REFUSAL_SOURCE.name)
    output = Path(OUTPUT_ROOT) / "qwen25-3b-seed42"
    output.mkdir(parents=True, exist_ok=True)
    command = [
        "python", "-m", "src.train", "--method", "cbg",
        "--config", "config/cbg_v5_qwen25_3b_fullparam_probe.yaml",
        "--model_name", str(local_model), "--tokenizer_name", str(local_model),
        "--output_dir", str(output), "--max_steps", "1",
    ]
    print(json.dumps({"gpus": gpu_info, "command": command}, indent=2), flush=True)
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": "0,1",
        "CBG_SINGLE_MODEL_GPU": "1",
        "HF_HOME": HF_ROOT,
        "HF_DATASETS_CACHE": f"{HF_ROOT}/datasets",
    })
    completed = subprocess.run(command, cwd=REMOTE, env=environment, check=False)
    output_volume.commit()
    if completed.returncode:
        raise RuntimeError(f"Full-parameter CBG probe exited with status {completed.returncode}")
    return {"status": "completed", "output": str(output)}


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(run_probe.remote(), indent=2))
