"""Run a fixed-batch, geometry-only descent check for an existing S0 pilot."""
from __future__ import annotations

import os
from pathlib import Path
import re

import modal


ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/workspace"
OUTPUT = "/vol/output"
CACHE = "/vol/hf"
GPU = os.environ.get("CBG_MODAL_GPU", "H200")
GPU_STAGE_MEMORY = int(os.environ.get("CBG_MODAL_MEMORY", "65536"))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install_from_requirements(str(ROOT / "requirements.txt"))
    .env({"PYTHONPATH": REMOTE, "USE_TF": "0", "TRANSFORMERS_NO_TF": "1",
          "HF_HOME": CACHE, "HF_HUB_OFFLINE": "1",
          "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1",
          "MKL_NUM_THREADS": "1",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir(str(ROOT / "src"), REMOTE + "/src")
    .add_local_file(str(ROOT / "scripts" / "cbg_fixed_batch_descent.py"),
                    REMOTE + "/scripts/cbg_fixed_batch_descent.py")
)
app = modal.App("cbg-spec-s0-fixed-batch-descent")
outputs = modal.Volume.from_name("cbg-spec-s0-output")
cache = modal.Volume.from_name("cbg-spec-hf")


@app.function(image=image, gpu=GPU, cpu=4, memory=GPU_STAGE_MEMORY,
              timeout=3 * 3600, volumes={OUTPUT: outputs, CACHE: cache})
def descent(run_name: str, attempt: str):
    os.chdir(REMOTE)
    from huggingface_hub import snapshot_download
    from src.experiments.protocol import read_json
    from scripts.cbg_fixed_batch_descent import run

    base = Path(OUTPUT) / run_name
    if not (base / "prepared_protocol.yaml").is_file():
        raise FileNotFoundError("prepared S0 run is missing")
    if not (base / "choices-train.yaml").is_file():
        raise FileNotFoundError("the S0 training choices are missing")
    pinned = read_json(base / "model_revision.json")
    snapshot = snapshot_download(pinned["model"], revision=pinned["revision"],
                                 local_files_only=True)
    output = base / "diagnostics" / f"fixed_batch_descent_20x4_{attempt}"
    try:
        return run(snapshot, pinned["model"], pinned["revision"],
                   base / "bank.pt", base / "prepared_protocol.yaml",
                   base / "choices-train.yaml", output, steps=20, sign_draws=4)
    finally:
        outputs.commit()


@app.local_entrypoint()
def main(run_name: str, attempt: str = "b"):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_name):
        raise ValueError("invalid S0 run name")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,20}", attempt):
        raise ValueError("invalid diagnostic attempt")
    print(descent.remote(run_name, attempt))
