"""Run the original SEAM method on Qwen2.5-3B-Instruct with Modal.

Examples, from the repository root::

    modal run seam/scripts/modal_seam_qwen25_3b.py --prepare-only
    modal run seam/scripts/modal_seam_qwen25_3b.py --steps 1 --run-name seam-qwen3b-smoke
    modal run --detach seam/scripts/modal_seam_qwen25_3b.py \
        --steps 500 --run-name seam-qwen3b-main-seed42

The two-GPU layout follows the smaller-model option documented by SEAM:
GPU 0 owns the model/optimizer and GPU 1 owns auxiliary full-model gradients.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from urllib.request import urlopen

import modal


APP_NAME = "seam-qwen25-3b"
REMOTE_REPO = "/workspace/seam"
HF_ROOT = "/vol/hf"
DATA_ROOT = "/vol/data"
OUTPUT_ROOT = "/vol/output"
MODEL_PATH = "/vol/hf/models/Qwen2.5-3B-Instruct"
REFUSAL_PATH = "/vol/data/beavertails_with_refusals_train.json"

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
ALPACA_ID = "yahma/alpaca-cleaned"
REFUSAL_URL = (
    "https://raw.githubusercontent.com/domenicrosati/representation-noising/"
    "main/data/beavertails_with_refusals_train.json"
)
# A timing comparison is ambiguous if Modal selects A100 or H100 at will.
# The runtime check below also rejects an H200 substitution.
GPU_SPEC = "H100:2"

SEAM_ROOT = Path(__file__).resolve().parents[1]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install_from_requirements(str(SEAM_ROOT / "requirements.txt"))
    .pip_install("wandb>=0.19,<1")
    .env(
        {
            "PYTHONPATH": REMOTE_REPO,
            "USE_TF": "0",
            "TRANSFORMERS_NO_TF": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "WANDB_MODE": "offline",
            "SEAM_GRADIENT_GPUS": "1",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
        }
    )
    .add_local_dir(str(SEAM_ROOT / "src"), f"{REMOTE_REPO}/src")
    .add_local_file(
        str(SEAM_ROOT / "config" / "train_qwen25_3b_modal.yaml"),
        f"{REMOTE_REPO}/config/train_qwen25_3b_modal.yaml",
    )
)

app = modal.App(APP_NAME)
hf_volume = modal.Volume.from_name("cbg-v5-hf-cache", create_if_missing=True)
data_volume = modal.Volume.from_name("cbg-v5-data", create_if_missing=True)
output_volume = modal.Volume.from_name("seam-qwen25-3b-checkpoints", create_if_missing=True)


def _download(url: str, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    digest = hashlib.sha256()
    with urlopen(url, timeout=120) as response, temporary.open("wb") as stream:
        while chunk := response.read(1024 * 1024):
            stream.write(chunk)
            digest.update(chunk)
    temporary.replace(destination)
    return digest.hexdigest()


@app.function(
    image=image,
    cpu=1.0,
    memory=8 * 1024,
    timeout=4 * 60 * 60,
    volumes={HF_ROOT: hf_volume, DATA_ROOT: data_volume},
)
def prepare_assets() -> dict[str, object]:
    from datasets import load_dataset
    from huggingface_hub import HfApi, snapshot_download

    refusal = Path(REFUSAL_PATH)
    if refusal.exists():
        refusal_sha = hashlib.sha256(refusal.read_bytes()).hexdigest()
    else:
        refusal_sha = _download(REFUSAL_URL, refusal)
    records = json.loads(refusal.read_text(encoding="utf-8"))
    eligible = sum(
        isinstance(row, dict)
        and row.get("is_safe") is not True
        and bool(str(row.get("response", "")).strip())
        and bool(str(row.get("refusal", "")).strip())
        for row in records
    )
    if eligible < 4000:
        raise ValueError(f"Only {eligible} eligible BeaverTails pairs; 4,000 required")
    data_volume.commit()

    revision = HfApi().model_info(MODEL_ID).sha
    snapshot_download(MODEL_ID, revision=revision, local_dir=MODEL_PATH)
    load_dataset(ALPACA_ID, split="train", cache_dir=f"{HF_ROOT}/datasets")
    manifest = {
        "model": MODEL_ID,
        "revision": revision,
        "refusal_sha256": refusal_sha,
        "eligible_refusal_pairs": eligible,
        "benign_dataset": ALPACA_ID,
    }
    Path(f"{HF_ROOT}/seam_asset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    hf_volume.commit()
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def _latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints: list[tuple[int, Path]] = []
    if output_dir.exists():
        for path in output_dir.iterdir():
            match = re.fullmatch(r"checkpoint-(\d+)", path.name)
            if match and (path / "trainer_state.json").is_file():
                checkpoints.append((int(match.group(1)), path))
    return max(checkpoints, default=(0, None), key=lambda item: item[0])[1]


@app.function(
    image=image,
    gpu=GPU_SPEC,
    cpu=2.0,
    memory=32 * 1024,
    timeout=24 * 60 * 60,
    volumes={HF_ROOT: hf_volume, DATA_ROOT: data_volume, OUTPUT_ROOT: output_volume},
)
def train(steps: int, run_name: str, batch_size: int = 8, save_steps: int = 100) -> dict[str, object]:
    import torch

    if steps < 1 or batch_size < 1 or save_steps < 1:
        raise ValueError("steps, batch_size and save_steps must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name):
        raise ValueError("run_name contains unsupported characters")
    if torch.cuda.device_count() != 2:
        raise RuntimeError(f"Expected two GPUs, received {torch.cuda.device_count()}")
    gpu_info = []
    for index in range(2):
        props = torch.cuda.get_device_properties(index)
        gpu_info.append({"index": index, "name": props.name, "memory_gib": props.total_memory / 1024**3})
        normalized_gpu_name = props.name.upper()
        if "H100" not in normalized_gpu_name or "H200" in normalized_gpu_name:
            raise RuntimeError(
                f"requested H100-only execution but GPU {index} is {props.name}"
            )
        if props.total_memory < 75 * 1024**3:
            raise RuntimeError(f"GPU {index} has only {props.total_memory / 1024**3:.1f} GiB")

    model_path = Path(MODEL_PATH)
    refusal_path = Path(REFUSAL_PATH)
    if not model_path.is_dir() or not refusal_path.is_file():
        raise FileNotFoundError("Assets missing; run --prepare-only first")

    work_data = Path(REMOTE_REPO) / "data"
    work_data.mkdir(parents=True, exist_ok=True)
    destination = work_data / refusal_path.name
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(refusal_path)

    output_dir = Path(OUTPUT_ROOT) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = _latest_checkpoint(output_dir)
    command = [
        "python", "-m", "src.train",
        "--config", "config/train_qwen25_3b_modal.yaml",
        "--model_name", str(model_path),
        "--tokenizer_name", str(model_path),
        "--output_dir", str(output_dir),
        "--max_steps", str(steps),
        "--per_device_train_batch_size", str(batch_size),
        "--save_steps", str(min(save_steps, steps)),
        "--wandb_run_name", run_name,
    ]
    if checkpoint is not None:
        command.extend(["--resume_from_checkpoint", str(checkpoint)])

    resolved = {
        "method": "seam",
        "model": MODEL_ID,
        "data": {"harmful": 4000, "refusal": 4000, "alpaca": 4000, "length": 256},
        "training": {"steps": steps, "batch_size": batch_size, "lr": 2e-5, "precision": "bf16"},
        "seam": {"epsilon": 1e-3, "alpha": 1.0, "beta": 0.01},
        "gpus": gpu_info,
        "resume": str(checkpoint) if checkpoint else None,
        "command": command,
    }
    (output_dir / "resolved_run.json").write_text(json.dumps(resolved, indent=2) + "\n")
    print(json.dumps(resolved, indent=2), flush=True)

    environment = os.environ.copy()
    environment.update(
        {
            "HF_HOME": HF_ROOT,
            "HF_DATASETS_CACHE": f"{HF_ROOT}/datasets",
            "TRANSFORMERS_CACHE": f"{HF_ROOT}/transformers",
            "CUDA_VISIBLE_DEVICES": "0,1",
            "SEAM_GRADIENT_GPUS": "1",
        }
    )
    completed = subprocess.run(command, cwd=REMOTE_REPO, env=environment, check=False)
    output_volume.commit()
    if completed.returncode:
        raise RuntimeError(f"SEAM exited with status {completed.returncode}")
    return {"run_name": run_name, "steps": steps, "output": str(output_dir)}


@app.local_entrypoint()
def main(
    steps: int = 500,
    run_name: str = "seam-qwen3b-main-seed42",
    batch_size: int = 8,
    save_steps: int = 100,
    prepare_only: bool = False,
    skip_prepare: bool = False,
) -> None:
    if not skip_prepare:
        prepare_assets.remote()
    if prepare_only:
        return
    result = train.remote(steps=steps, run_name=run_name, batch_size=batch_size, save_steps=save_steps)
    print(json.dumps(result, indent=2))
