"""Run exactly five CBG-only updates with held-out geometry measurements."""
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
    .add_local_file(str(ROOT / "scripts" / "cbg_five_step_probe.py"),
                    REMOTE + "/scripts/cbg_five_step_probe.py")
)
app = modal.App("cbg-spec-s0-five-step")
outputs = modal.Volume.from_name("cbg-spec-s0-output")
cache = modal.Volume.from_name("cbg-spec-hf")


@app.function(image=image, gpu=GPU, cpu=4, memory=GPU_STAGE_MEMORY,
              timeout=2 * 3600, volumes={OUTPUT: outputs, CACHE: cache})
def probe(run_name: str, attempt: str):
    os.chdir(REMOTE)
    from huggingface_hub import snapshot_download
    from src.experiments.protocol import read_json
    from scripts.cbg_five_step_probe import run

    base = Path(OUTPUT) / run_name
    if not (base / "prepared_protocol.yaml").is_file():
        raise FileNotFoundError("prepared S0 run is missing")
    if not (base / "choices-train.yaml").is_file():
        raise FileNotFoundError("S0 training choices are missing")
    pinned = read_json(base / "model_revision.json")
    snapshot = snapshot_download(pinned["model"], revision=pinned["revision"],
                                 local_files_only=True)
    output = base / "diagnostics" / f"five_step_holdout_{attempt}"
    try:
        return run(snapshot, pinned["model"], pinned["revision"],
                   base / "bank.pt", base / "prepared_protocol.yaml",
                   base / "choices-train.yaml", output)
    finally:
        outputs.commit()


@app.local_entrypoint()
def main(run_name: str, attempt: str = "a"):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_name):
        raise ValueError("invalid S0 run name")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,20}", attempt):
        raise ValueError("invalid diagnostic attempt")
    print(probe.remote(run_name, attempt))
