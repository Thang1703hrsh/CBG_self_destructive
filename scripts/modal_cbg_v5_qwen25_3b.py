"""Run CBG V5 Qwen2.5-3B training on one verified Modal L40S.

The app deliberately prepares public model/data assets on CPU before it ever
allocates the paid GPU.  Checkpoints, data, and the Hugging Face model snapshot
live in separate Modal Volumes.

Examples (from the repository root):

    modal run scripts/modal_cbg_v5_qwen25_3b.py --prepare-only
    modal run scripts/modal_cbg_v5_qwen25_3b.py --steps 1 --run-name qwen3b-probe
    modal run --detach scripts/modal_cbg_v5_qwen25_3b.py \
        --steps 500 --run-name qwen3b-main-seed42 --push-to-hub --background
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


APP_NAME = "cbg-v5-qwen25-3b"
REMOTE_ROOT = "/workspace"
HF_VOLUME_ROOT = "/vol/hf"
DATA_VOLUME_ROOT = "/vol/data"
OUTPUT_VOLUME_ROOT = "/vol/output"
MODEL_VOLUME_PATH = "/vol/hf/models/Qwen2.5-3B-Instruct"
REFUSAL_VOLUME_PATH = "/vol/data/beavertails_with_refusals_train.json"

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_HF_REPO_ID = "ducthang1703/cbg-v5-qwen25-3b-l40s"
ALPACA_ID = "yahma/alpaca-cleaned"
REFUSAL_URL = (
    "https://raw.githubusercontent.com/domenicrosati/representation-noising/"
    "main/data/beavertails_with_refusals_train.json"
)

# Modal's cpu unit is a physical core (normally two vCPUs), while memory uses
# MiB. CBG is GPU-bound; one physical core is sufficient once tensors are
# materialized. Keep 16 GiB for FP32 checkpoint loading rather than risking a
# host OOM to save only a few cents per hour.
# Modal's ephemeral-disk override has a 512 GiB minimum, so requesting the
# user's smaller 150 GiB target is invalid; leaving it unset uses Modal's
# default local-SSD quota and avoids an artificial larger-disk request.
PREP_CPU_CORES = 0.5
PREP_MEMORY_MIB = 4 * 1024
TRAIN_CPU_CORES = 1.0
TRAIN_MEMORY_MIB = 16 * 1024
# Keep qualification timings tied to the requested 48 GB device rather than
# allowing an automatic accelerator substitution.
GPU_TYPE = "L40S"
MIN_GPU_MEMORY_GIB = 44

REPO_ROOT = Path(__file__).resolve().parents[1]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install_from_requirements(str(REPO_ROOT / "requirements.txt"))
    .env(
        {
            "PYTHONPATH": str(REMOTE_ROOT),
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
    # Modal requires all local-file additions to be the final image operations.
    .add_local_dir(str(REPO_ROOT / "src"), f"{REMOTE_ROOT}/src")
    .add_local_file(
        str(REPO_ROOT / "config" / "cbg_v5_qwen25_3b_main.yaml"),
        f"{REMOTE_ROOT}/config/cbg_v5_qwen25_3b_main.yaml",
    )
)

app = modal.App(APP_NAME)
hf_volume = modal.Volume.from_name("cbg-v5-hf-cache", create_if_missing=True)
data_volume = modal.Volume.from_name("cbg-v5-data", create_if_missing=True)
output_volume = modal.Volume.from_name("cbg-v5-checkpoints", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])


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
    if not isinstance(records, list) or len(records) < 5000:
        raise ValueError("refusal JSON must contain at least 5,000 records")
    # Match the production loader: the public source contains a few unusable
    # rows, which ``filter_beavertails_records`` intentionally skips.  Validate
    # that enough eligible harmful/refusal pairs remain for the 4K/1K split.
    required = {"prompt", "response", "refusal"}
    eligible = 0
    for record in records:
        if not isinstance(record, dict) or not required.issubset(record):
            continue
        if record.get("is_safe") is True:
            continue
        if not str(record["response"]).strip() or not str(record["refusal"]).strip():
            continue
        eligible += 1
    if eligible < 5000:
        raise ValueError(
            f"refusal JSON has only {eligible} eligible harmful/refusal pairs; need 5,000"
        )
    print(f"Validated {eligible} eligible pairs from {len(records)} source records")
    return len(records)


@app.function(
    image=image,
    cpu=PREP_CPU_CORES,
    memory=PREP_MEMORY_MIB,
    timeout=60 * 60 * 4,
    volumes={
        HF_VOLUME_ROOT: hf_volume,
        DATA_VOLUME_ROOT: data_volume,
    },
)
def prepare_assets() -> dict[str, object]:
    """Download and verify all public assets without allocating a GPU."""

    from datasets import load_dataset
    from huggingface_hub import HfApi, snapshot_download

    refusal_path = Path(REFUSAL_VOLUME_PATH)
    model_path = Path(MODEL_VOLUME_PATH)
    hf_root = Path(HF_VOLUME_ROOT)
    if refusal_path.exists():
        refusal_sha = hashlib.sha256(refusal_path.read_bytes()).hexdigest()
    else:
        refusal_sha = _download_stream(REFUSAL_URL, refusal_path)
    refusal_count = _validate_refusal_json(refusal_path)
    data_volume.commit()

    model_info = HfApi().model_info(MODEL_ID)
    model_revision = model_info.sha
    snapshot_download(
        repo_id=MODEL_ID,
        revision=model_revision,
        local_dir=model_path,
    )

    # Populate the exact datasets cache used by the training process.  Loading
    # only the train split validates availability without retaining Python data.
    load_dataset(
        ALPACA_ID,
        split="train",
        cache_dir=str(hf_root / "datasets"),
    )
    manifest = {
        "model_id": MODEL_ID,
        "model_revision": model_revision,
        "refusal_url": REFUSAL_URL,
        "refusal_sha256": refusal_sha,
        "refusal_records": refusal_count,
        "benign_dataset": ALPACA_ID,
    }
    (hf_root / "asset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    hf_volume.commit()
    print(json.dumps(manifest, indent=2))
    return manifest


def _latest_checkpoint(output_dir: Path) -> tuple[int, Path] | None:
    candidates: list[tuple[int, Path]] = []
    if not output_dir.exists():
        return None
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
    secrets=[hf_secret],
)
def train(
    steps: int,
    run_name: str,
    seed: int = 42,
    save_steps: int = 50,
    solver: str = "dual",
    beta_mode: str = "grad_ratio",
    beta: float = 0.00002,
    functional_chunk_size: int = 4,
    preservation_chunk_size: int = 8,
    dual_column_batch_size: int = 3,
    dual_refinement_steps: int = 3,
    dual_residual_tolerance: float = 0.002,
    max_grad_norm: float = 1.0,
    benchmark_no_warmup: bool = False,
    push_to_hub: bool = False,
    hf_repo_id: str = DEFAULT_HF_REPO_ID,
    hf_private: bool = True,
) -> dict[str, object]:
    """Train or resume one CBG run and persist its complete checkpoints."""

    if steps < 1:
        raise ValueError("steps must be positive")
    if save_steps < 1:
        raise ValueError("save_steps must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name):
        raise ValueError("run_name may contain only letters, numbers, dot, dash and underscore")
    if solver not in {"dual", "bounded_ritz"}:
        raise ValueError("solver must be dual or bounded_ritz")
    if beta_mode not in {"fixed", "grad_ratio"}:
        raise ValueError("beta_mode must be fixed or grad_ratio")
    if not 0 <= beta <= 1:
        raise ValueError("beta must be between zero and one")
    if functional_chunk_size < 0 or preservation_chunk_size < 0:
        raise ValueError("chunk sizes must be non-negative; zero means full batch")
    if dual_column_batch_size < 1 or dual_refinement_steps < 0:
        raise ValueError("column width must be positive and refinement budget nonnegative")
    if not math.isfinite(dual_residual_tolerance) or dual_residual_tolerance <= 0:
        raise ValueError("dual_residual_tolerance must be finite and positive")
    if not math.isfinite(max_grad_norm) or max_grad_norm < 0:
        raise ValueError("max_grad_norm must be finite and nonnegative; zero disables clipping")
    if push_to_hub and not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", hf_repo_id):
        raise ValueError("hf_repo_id must look like owner/repository")
    if benchmark_no_warmup and steps > 10:
        raise ValueError("benchmark_no_warmup is restricted to qualification runs of at most 10 steps")
    refusal_path = Path(REFUSAL_VOLUME_PATH)
    model_volume_path = Path(MODEL_VOLUME_PATH)
    output_root = Path(OUTPUT_VOLUME_ROOT)
    hf_root = Path(HF_VOLUME_ROOT)
    if not refusal_path.is_file() or not model_volume_path.is_dir():
        raise FileNotFoundError("assets are missing; run prepare_assets first")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Modal did not attach the requested CUDA GPU")
    properties = torch.cuda.get_device_properties(0)
    total_memory = properties.total_memory
    normalized_gpu_name = properties.name.upper()
    if "L40S" not in normalized_gpu_name:
        raise RuntimeError(
            f"requested L40S-only execution but Modal attached {properties.name}"
        )
    if total_memory < MIN_GPU_MEMORY_GIB * 1024**3:
        raise RuntimeError(
            f"expected {GPU_TYPE} with at least {MIN_GPU_MEMORY_GIB} GiB but received {properties.name} with "
            f"{total_memory / 1024**3:.1f} GiB"
        )

    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    latest = _latest_checkpoint(output_dir)
    if latest is not None and latest[0] >= steps:
        result = {
            "status": "already_complete",
            "run_name": run_name,
            "checkpoint_step": latest[0],
            "checkpoint": str(latest[1]),
        }
        print(json.dumps(result, indent=2))
        return result

    # Stage weights for the initial load. Once loaded, forwards read GPU memory,
    # not this directory; copying does not accelerate each JVP/VJP.
    local_model = Path("/tmp/models/Qwen2.5-3B-Instruct")
    if not local_model.exists():
        local_model.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(model_volume_path, local_model)

    os.environ.update(
        {
            "HF_HOME": str(hf_root),
            "HF_DATASETS_CACHE": str(hf_root / "datasets"),
            "WANDB_DIR": str(output_dir),
            "CUDA_VISIBLE_DEVICES": "0",
        }
    )

    command = [
        "python",
        "-m",
        "src.train",
        "--method",
        "cbg",
        "--config",
        "config/cbg_v5_qwen25_3b_main.yaml",
        "--model_name",
        str(local_model),
        "--tokenizer_name",
        str(local_model),
        "--beavertails_path",
        str(refusal_path),
        "--output_dir",
        str(output_dir),
        "--max_steps",
        str(steps),
        "--save_steps",
        str(min(save_steps, steps)),
        "--seed",
        str(seed),
        "--data_seed",
        str(seed),
        "--wandb_run_name",
        run_name,
        "--geometry_solver",
        solver,
        "--beta_mode",
        beta_mode,
        "--beta",
        str(beta),
        "--functional_chunk_size",
        str(functional_chunk_size),
        "--preservation_chunk_size",
        str(preservation_chunk_size),
        "--dual_column_batch_size",
        str(dual_column_batch_size),
        "--dual_refinement_steps",
        str(dual_refinement_steps),
        "--dual_residual_tolerance",
        str(dual_residual_tolerance),
        "--max_grad_norm",
        str(max_grad_norm),
    ]
    if benchmark_no_warmup:
        if latest is not None:
            raise ValueError("use a fresh run name for a no-warmup benchmark")
        command.extend(["--warmup_ratio", "0", "--warmup_steps", "0"])
    if latest is not None:
        command.extend(["--cbg_resume_from_checkpoint", str(latest[1])])

    started = time.monotonic()
    print("Executing:", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=REMOTE_ROOT, check=False)
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        # Persist any complete checkpoint produced before the failure so an
        # automatic Modal retry can continue from it.
        output_volume.commit()
        raise subprocess.CalledProcessError(completed.returncode, command)

    output_volume.commit()
    latest_after = _latest_checkpoint(output_dir)
    hub_url = None
    if push_to_hub:
        from huggingface_hub import HfApi

        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("push_to_hub requires HF_TOKEN in Modal Secret huggingface-secret")
        if latest_after is None:
            raise RuntimeError("training completed without a checkpoint to upload")
        api = HfApi(token=token)
        api.create_repo(
            repo_id=hf_repo_id,
            repo_type="model",
            private=hf_private,
            exist_ok=True,
        )
        api.upload_folder(
            folder_path=str(latest_after[1]),
            repo_id=hf_repo_id,
            repo_type="model",
            commit_message=f"CBG V5 {run_name} checkpoint-{latest_after[0]}",
        )
        hub_url = f"https://huggingface.co/{hf_repo_id}/tree/main"
    result = {
        "status": "complete",
        "run_name": run_name,
        "target_steps": steps,
        "solver": solver,
        "beta_mode": beta_mode,
        "beta": beta,
        "functional_chunk_size": functional_chunk_size,
        "preservation_chunk_size": preservation_chunk_size,
        "dual_column_batch_size": dual_column_batch_size,
        "dual_refinement_steps": dual_refinement_steps,
        "dual_residual_tolerance": dual_residual_tolerance,
        "max_grad_norm": max_grad_norm,
        "benchmark_no_warmup": benchmark_no_warmup,
        "elapsed_seconds": elapsed,
        "gpu": properties.name,
        "requested_gpu": GPU_TYPE,
        "gpu_memory_gib": total_memory / 1024**3,
        "latest_checkpoint": str(latest_after[1]) if latest_after else None,
        "push_to_hub": push_to_hub,
        "hf_repo_id": hf_repo_id if push_to_hub else None,
        "hf_private": hf_private if push_to_hub else None,
        "hf_url": hub_url,
        "latest_checkpoint_step": latest_after[0] if latest_after else None,
    }
    print(json.dumps(result, indent=2))
    return result


@app.local_entrypoint()
def main(
    steps: int = 1,
    run_name: str = "qwen3b-probe",
    seed: int = 42,
    save_steps: int = 50,
    solver: str = "dual",
    beta_mode: str = "grad_ratio",
    beta: float = 0.00002,
    functional_chunk_size: int = 4,
    preservation_chunk_size: int = 8,
    dual_column_batch_size: int = 3,
    dual_refinement_steps: int = 3,
    dual_residual_tolerance: float = 0.002,
    max_grad_norm: float = 1.0,
    benchmark_no_warmup: bool = False,
    push_to_hub: bool = False,
    hf_repo_id: str = DEFAULT_HF_REPO_ID,
    hf_private: bool = True,
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
        "solver": solver,
        "beta_mode": beta_mode,
        "beta": beta,
        "functional_chunk_size": functional_chunk_size,
        "preservation_chunk_size": preservation_chunk_size,
        "dual_column_batch_size": dual_column_batch_size,
        "dual_refinement_steps": dual_refinement_steps,
        "dual_residual_tolerance": dual_residual_tolerance,
        "max_grad_norm": max_grad_norm,
        "benchmark_no_warmup": benchmark_no_warmup,
        "push_to_hub": push_to_hub,
        "hf_repo_id": hf_repo_id,
        "hf_private": hf_private,
    }
    if background:
        call = train.spawn(**kwargs)
        print(json.dumps({"status": "spawned", "function_call_id": call.object_id}, indent=2))
        return
    result = train.remote(**kwargs)
    print(json.dumps(result, indent=2))
