"""Restore the pinned HF snapshots of a prepared eval run into this workspace's cache.

Used to resume a defense-eval run on another Modal workspace: resume re-hashes
every checkpoint file at the manifest's resolved_path, so the same revisions must
exist in defense-eval-hf-cache under the same HF_HOME layout. CPU only.
"""

import hashlib
import json
from pathlib import Path

import modal

app = modal.App("defense-eval-hf-cache-warm")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("huggingface_hub==0.36.2", "hf_xet")
         .env({"HF_HOME": "/hf"}))
hf_volume = modal.Volume.from_name("defense-eval-hf-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("defense-eval-results")
# Same patterns as modal_defense_eval.prepare; the file set must match exactly.
MODEL_PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt"]
CLASSIFIER_PATTERNS = MODEL_PATTERNS + ["pytorch_model*.bin"]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@app.function(image=image, cpu=2, memory=4096, timeout=3 * 3600,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/hf": hf_volume, "/results": results_volume})
def warm(run_id: str):
    from huggingface_hub import snapshot_download

    manifest = json.loads(Path(f"/results/{run_id}/manifest.json").read_text(encoding="utf-8"))
    entries = [(label, info, MODEL_PATTERNS) for label, info in manifest["models"].items()]
    entries.append(("classifier", manifest["classifier"], CLASSIFIER_PATTERNS))
    for label, info, patterns in entries:
        if info.get("kind", "hf") != "hf":
            raise ValueError(f"{label}: only HF sources can be restored, got {info.get('kind')}")
        local = snapshot_download(info["repo_id"], revision=info["revision"], allow_patterns=patterns)
        if local != info["resolved_path"]:
            raise ValueError(f"{label}: snapshot path {local} != manifest {info['resolved_path']}")
        for name, expected in info["files"].items():
            if _sha256(Path(local) / name) != expected["sha256"]:
                raise ValueError(f"{label}: file hash differs from the prepared run: {name}")
        print("WARM_OK", label, info["revision"], len(info["files"]), "files", flush=True)
    hf_volume.commit()


@app.local_entrypoint()
def main(run_id: str):
    warm.remote(run_id)
