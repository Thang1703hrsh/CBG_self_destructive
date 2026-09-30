"""Qualify or train CBG V5 Llama2-7B on one strict Modal H100 80GB.

The Meta Llama 2 checkpoint is gated. Before running this launcher, accept the
model license on Hugging Face and create a Modal Secret named
``huggingface-secret`` containing the key ``HF_TOKEN``.

Prepare gated assets without allocating a GPU:

    modal run scripts/modal_cbg_v5_llama2_7b.py --prepare-only

Run the required five-step qualification with a fresh run name:

    modal run scripts/modal_cbg_v5_llama2_7b.py \
        --steps 5 --run-name llama2-7b-h100-qual5-seed42 \
        --benchmark-no-warmup

Only after qualification passes, launch a resumable detached main run:

    modal run --detach scripts/modal_cbg_v5_llama2_7b.py \
        --steps 500 --run-name llama2-7b-h100-main-seed42 --background
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from urllib.request import urlopen

import modal


APP_NAME = "cbg-v5-llama2-7b"
REMOTE_ROOT = "/workspace"
HF_VOLUME_ROOT = "/vol/hf"
DATA_VOLUME_ROOT = "/vol/data"
OUTPUT_VOLUME_ROOT = "/vol/output"
MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
MODEL_VOLUME_PATH = "/vol/hf/models/Llama-2-7b-chat-hf"
REFUSAL_VOLUME_PATH = "/vol/data/beavertails_with_refusals_train.json"
ALPACA_ID = "yahma/alpaca-cleaned"
REFUSAL_URL = (
    "https://raw.githubusercontent.com/domenicrosati/representation-noising/"
    "main/data/beavertails_with_refusals_train.json"
)

PREP_CPU_CORES = 0.5
PREP_MEMORY_MIB = 4 * 1024
TRAIN_CPU_CORES = 1.0
# Loading the FP32 7B checkpoint can transiently require substantially more
# host memory than the final GPU-resident LoRA training state.
TRAIN_MEMORY_MIB = 48 * 1024
GPU_TYPE = "H100"
MIN_GPU_MEMORY_GIB = 78

REPO_ROOT = Path(__file__).resolve().parents[1]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install_from_requirements(str(REPO_ROOT / "requirements.txt"))
    .env(
        {
            "PYTHONPATH": REMOTE_ROOT,
            "USE_TF": "0",
            "TRANSFORMERS_NO_TF": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "MALLOC_ARENA_MAX": "2",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_MAX_THREADS": "1",
            "WANDB_MODE": "offline",
        }
    )
    .add_local_dir(str(REPO_ROOT / "src"), f"{REMOTE_ROOT}/src")
    .add_local_file(
        str(REPO_ROOT / "config" / "cbg_v5_llama2_7b_main.yaml"),
        f"{REMOTE_ROOT}/config/cbg_v5_llama2_7b_main.yaml",
    )
)

app = modal.App(APP_NAME)
hf_volume = modal.Volume.from_name("cbg-v5-hf-cache", create_if_missing=True)
data_volume = modal.Volume.from_name("cbg-v5-data", create_if_missing=True)
output_volume = modal.Volume.from_name("cbg-v5-checkpoints", create_if_missing=True)
hf_secret = modal.Secret.from_name(
    "huggingface-secret", required_keys=["HF_TOKEN"]
)


def _download_stream(url: str, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    digest = hashlib.sha256()
    with urlopen(url, timeout=120) as response, temporary.open("wb") as target:
        while chunk := response.read(1024 * 1024):
            target.write(chunk)
            digest.update(chunk)
    temporary.replace(destination)
    return digest.hexdigest()


def _validate_refusal_json(path: Path) -> int:
    with path.open("r", encoding="utf-8") as stream:
        records = json.load(stream)
    required = {"prompt", "response", "refusal"}
    eligible = sum(
        1
        for record in records
        if isinstance(record, dict)
        and required.issubset(record)
        and record.get("is_safe") is not True
        and str(record["response"]).strip()
        and str(record["refusal"]).strip()
    )
    if eligible < 5000:
        raise ValueError(f"only {eligible} eligible harmful/refusal pairs; need 5,000")
    print(f"Validated {eligible} eligible pairs from {len(records)} source records")
    return len(records)


@app.function(
    image=image,
    cpu=PREP_CPU_CORES,
    memory=PREP_MEMORY_MIB,
    timeout=60 * 60 * 6,
    secrets=[hf_secret],
    volumes={HF_VOLUME_ROOT: hf_volume, DATA_VOLUME_ROOT: data_volume},
)
def prepare_assets() -> dict[str, object]:
    """Download the gated model and public datasets without a paid GPU."""

    from datasets import load_dataset
    from huggingface_hub import HfApi, snapshot_download

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Modal Secret huggingface-secret is missing HF_TOKEN")

    refusal_path = Path(REFUSAL_VOLUME_PATH)
    model_path = Path(MODEL_VOLUME_PATH)
    hf_root = Path(HF_VOLUME_ROOT)
    refusal_sha = (
        hashlib.sha256(refusal_path.read_bytes()).hexdigest()
        if refusal_path.exists()
        else _download_stream(REFUSAL_URL, refusal_path)
    )
    refusal_count = _validate_refusal_json(refusal_path)
    data_volume.commit()

    try:
        revision = HfApi(token=token).model_info(MODEL_ID).sha
        snapshot_download(
            repo_id=MODEL_ID,
            revision=revision,
            local_dir=model_path,
            token=token,
        )
    except Exception as error:
        raise RuntimeError(
            "Unable to download gated Llama2-7B. Confirm that the Hugging "
            "Face account behind HF_TOKEN has accepted the Meta license."
        ) from error

    load_dataset(ALPACA_ID, split="train", cache_dir=str(hf_root / "datasets"))
    manifest = {
        "model_id": MODEL_ID,
        "model_revision": revision,
        "refusal_url": REFUSAL_URL,
        "refusal_sha256": refusal_sha,
        "refusal_records": refusal_count,
        "benign_dataset": ALPACA_ID,
    }
    (hf_root / "asset_manifest_llama2_7b.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    hf_volume.commit()
    print(json.dumps(manifest, indent=2))
    return manifest


def _latest_checkpoint(output_dir: Path) -> tuple[int, Path] | None:
    candidates: list[tuple[int, Path]] = []
    if output_dir.exists():
        for path in output_dir.iterdir():
            match = re.fullmatch(r"checkpoint-(\d+)", path.name)
            if match and path.is_dir() and (path / "cbg_state.pt").is_file():
                candidates.append((int(match.group(1)), path))
    return max(candidates, default=None, key=lambda item: item[0])


@app.function(
    image=image,
    gpu=GPU_TYPE,
    cpu=TRAIN_CPU_CORES,
    memory=TRAIN_MEMORY_MIB,
    timeout=60 * 60 * 24,
    volumes={
        HF_VOLUME_ROOT: hf_volume,
        DATA_VOLUME_ROOT: data_volume,
        OUTPUT_VOLUME_ROOT: output_volume,
    },
)
def train(
    steps: int,
    run_name: str,
    seed: int = 42,
    save_steps: int = 250,
    beta_mode: str = "grad_ratio",
    beta: float = 0.00002,
    functional_chunk_size: int = 1,
    preservation_chunk_size: int = 4,
    dual_column_batch_size: int = 2,
    dual_refinement_steps: int = 3,
    dual_residual_tolerance: float = 0.0005,
    benchmark_no_warmup: bool = False,
) -> dict[str, object]:
    if steps < 1 or save_steps < 1:
        raise ValueError("steps and save_steps must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name):
        raise ValueError("invalid run_name")
    if beta_mode not in {"fixed", "grad_ratio"} or not 0 <= beta <= 1:
        raise ValueError("invalid beta configuration")
    if functional_chunk_size < 1 or preservation_chunk_size < 1:
        raise ValueError("Llama2-7B qualification requires positive chunk sizes")
    if dual_column_batch_size < 1 or dual_refinement_steps < 0:
        raise ValueError("invalid dual execution configuration")
    if not math.isfinite(dual_residual_tolerance) or dual_residual_tolerance <= 0:
        raise ValueError("dual_residual_tolerance must be finite and positive")
    if benchmark_no_warmup and steps > 10:
        raise ValueError("no-warmup mode is limited to at most ten qualification steps")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Modal did not attach CUDA")
    properties = torch.cuda.get_device_properties(0)
    if "H100" not in properties.name.upper() or "H200" in properties.name.upper():
        raise RuntimeError(f"requested H100-only execution but received {properties.name}")
    if properties.total_memory < MIN_GPU_MEMORY_GIB * 1024**3:
        raise RuntimeError(
            f"expected H100 80GB; received {properties.name} "
            f"with {properties.total_memory / 1024**3:.1f} GiB"
        )

    refusal_path = Path(REFUSAL_VOLUME_PATH)
    model_volume_path = Path(MODEL_VOLUME_PATH)
    if not refusal_path.is_file() or not model_volume_path.is_dir():
        raise FileNotFoundError("assets missing; prepare_assets must complete first")

    output_dir = Path(OUTPUT_VOLUME_ROOT) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    latest = _latest_checkpoint(output_dir)
    if latest is not None and latest[0] >= steps:
        return {"status": "already_complete", "checkpoint": str(latest[1])}

    local_model = Path("/tmp/models/Llama-2-7b-chat-hf")
    if not local_model.exists():
        local_model.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(model_volume_path, local_model)
    os.environ.update(
        {
            "HF_HOME": HF_VOLUME_ROOT,
            "HF_DATASETS_CACHE": f"{HF_VOLUME_ROOT}/datasets",
            "WANDB_DIR": str(output_dir),
            "CUDA_VISIBLE_DEVICES": "0",
        }
    )

    command = [
        "python", "-m", "src.train", "--method", "cbg",
        "--config", "config/cbg_v5_llama2_7b_main.yaml",
        "--model_name", str(local_model), "--tokenizer_name", str(local_model),
        "--beavertails_path", str(refusal_path),
        "--output_dir", str(output_dir), "--max_steps", str(steps),
        "--save_steps", str(min(save_steps, steps)),
        "--seed", str(seed), "--data_seed", str(seed),
        "--wandb_run_name", run_name,
        "--beta_mode", beta_mode, "--beta", str(beta),
        "--functional_chunk_size", str(functional_chunk_size),
        "--preservation_chunk_size", str(preservation_chunk_size),
        "--dual_column_batch_size", str(dual_column_batch_size),
        "--dual_refinement_steps", str(dual_refinement_steps),
        "--dual_residual_tolerance", str(dual_residual_tolerance),
    ]
    if benchmark_no_warmup:
        if latest is not None:
            raise ValueError("use a fresh run name for no-warmup qualification")
        command.extend(["--warmup_ratio", "0", "--warmup_steps", "0"])
    if latest is not None:
        command.extend(["--cbg_resume_from_checkpoint", str(latest[1])])

    started = time.monotonic()
    print("Executing:", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=REMOTE_ROOT, check=False)
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        output_volume.commit()
        raise subprocess.CalledProcessError(completed.returncode, command)
    output_volume.commit()
    latest_after = _latest_checkpoint(output_dir)
    result = {
        "status": "complete",
        "run_name": run_name,
        "target_steps": steps,
        "elapsed_seconds": elapsed,
        "gpu": properties.name,
        "gpu_memory_gib": properties.total_memory / 1024**3,
        "functional_chunk_size": functional_chunk_size,
        "preservation_chunk_size": preservation_chunk_size,
        "dual_column_batch_size": dual_column_batch_size,
        "dual_residual_tolerance": dual_residual_tolerance,
        "latest_checkpoint": str(latest_after[1]) if latest_after else None,
    }
    print(json.dumps(result, indent=2))
    return result


@app.local_entrypoint()
def main(
    steps: int = 5,
    run_name: str = "llama2-7b-h100-qual5-seed42",
    seed: int = 42,
    save_steps: int = 250,
    beta_mode: str = "grad_ratio",
    beta: float = 0.00002,
    functional_chunk_size: int = 1,
    preservation_chunk_size: int = 4,
    dual_column_batch_size: int = 2,
    dual_refinement_steps: int = 3,
    dual_residual_tolerance: float = 0.0005,
    benchmark_no_warmup: bool = False,
    background: bool = False,
    prepare_only: bool = False,
) -> None:
    manifest = prepare_assets.remote()
    if prepare_only:
        print(json.dumps(manifest, indent=2))
        return
    kwargs = {
        "steps": steps,
        "run_name": run_name,
        "seed": seed,
        "save_steps": min(save_steps, steps),
        "beta_mode": beta_mode,
        "beta": beta,
        "functional_chunk_size": functional_chunk_size,
        "preservation_chunk_size": preservation_chunk_size,
        "dual_column_batch_size": dual_column_batch_size,
        "dual_refinement_steps": dual_refinement_steps,
        "dual_residual_tolerance": dual_residual_tolerance,
        "benchmark_no_warmup": benchmark_no_warmup,
    }
    if background:
        call = train.spawn(**kwargs)
        print(json.dumps({"status": "spawned", "function_call_id": call.object_id}, indent=2))
        return
    print(json.dumps(train.remote(**kwargs), indent=2))
