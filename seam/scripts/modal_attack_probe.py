"""Re-run one prepared eval case with a different attack max_grad_norm (diagnostic).

Reuses an existing run's manifest, inputs and pinned checkpoints, and the unchanged
defense_eval_worker.run_case; only TrainingArguments.max_grad_norm is overridden.
Results go to /results/probes/<probe_id>, never into the original run.

  python -m modal run --detach scripts/modal_attack_probe.py::main --run-id RUN \
      --label base --case harmful-lr2e-4 --max-grad-norm 1.0
"""

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import modal

REPO = Path(__file__).resolve().parent.parent
REMOTE = PurePosixPath("/root/defense_eval")
app = modal.App("defense-eval-attack-probe")
# Same packages as modal_defense_eval.py so only the clipping differs.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(REPO / "requirements.txt"))
    .pip_install("wandb==0.19.8", "requests==2.32.3", "sentencepiece==0.2.0",
                 "bitsandbytes==0.45.2")
    .env({"HF_HOME": "/hf", "WANDB_MODE": "disabled", "USE_TF": "0",
          "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"})
)
for filename in ("defense_eval_common.py", "defense_eval_worker.py"):
    image = image.add_local_file(str(REPO / "scripts" / filename), str(REMOTE / "scripts" / filename))
results_volume = modal.Volume.from_name("defense-eval-results")
hf_volume = modal.Volume.from_name("defense-eval-hf-cache")


@app.function(image=image, gpu="L40S", cpu=4, memory=32768, timeout=4 * 3600,
              volumes={"/results": results_volume, "/hf": hf_volume})
def probe(run_id: str, label: str, case_id: str, max_grad_norm: float, probe_id: str):
    sys.path.insert(0, str(REMOTE / "scripts"))
    import transformers
    from defense_eval_worker import run_case

    source = Path("/results") / run_id
    root = Path("/results") / "probes" / probe_id
    if root.exists():
        raise FileExistsError(f"Probe exists: {root}")
    (root / "inputs").mkdir(parents=True)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    manifest["probe"] = {"source_run": run_id, "max_grad_norm": max_grad_norm}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for name in manifest["input_sha256"]:
        shutil.copy2(source / "inputs" / name, root / "inputs" / name)
    results_volume.commit()

    base = transformers.TrainingArguments

    class ClippedArguments(base):
        def __post_init__(self):
            self.max_grad_norm = max_grad_norm
            super().__post_init__()

    transformers.TrainingArguments = ClippedArguments
    try:
        run_case(root / "manifest.json", label, case_id)
    finally:
        transformers.TrainingArguments = base
        results_volume.commit()
    result = json.loads((root / "cases" / label / case_id / "result.json").read_text(encoding="utf-8"))
    return {key: result[key] for key in ("HS_percent", "harmful_rate_percent", "ZS_percent", "utility_percent")}


@app.local_entrypoint()
def main(run_id: str, label: str = "base", case: str = "harmful-lr2e-4", max_grad_norm: float = 1.0):
    probe_id = f"{label}-{case}-clip{max_grad_norm:g}-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    call = probe.spawn(run_id, label, case, max_grad_norm, probe_id)
    print(f"Submitted probe {call.object_id}")
    print(f"Results: defense-eval-results/probes/{probe_id}/cases/{label}/{case}/")
