"""Safely create a private HF model repo and upload a verified checkpoint."""

import json
import os
import re
from pathlib import Path

from huggingface_hub import HfApi


_REPO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_PREFLIGHT_MARKER = ".train-push-preflight"
_IGNORABLE_EXISTING = {".gitattributes", "README.md"}


def _api() -> HfApi:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Modal secret huggingface-secret must provide HF_TOKEN")
    return HfApi(token=token)


def ensure_private_empty_repo(repo_id: str) -> str:
    """Validate write access before renting a GPU; never overwrite model files."""
    if not _REPO_ID.fullmatch(repo_id):
        raise ValueError("HF repo must be namespace/name using letters, digits, ., _ or -")
    api = _api()
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    info = api.repo_info(repo_id=repo_id, repo_type="model")
    if not info.private:
        raise ValueError(f"Refusing to upload to public HF repo: {repo_id}")
    existing = set(api.list_repo_files(repo_id=repo_id, repo_type="model"))
    unexpected = existing - _IGNORABLE_EXISTING
    if unexpected:
        raise FileExistsError(
            f"Refusing to overwrite nonempty HF repo {repo_id}; existing files: "
            + ", ".join(sorted(unexpected)[:8])
        )
    # An existing empty repo can be readable with a read-only token. Probe an
    # actual tiny write now, before the expensive GPU training begins.
    api.upload_file(
        repo_id=repo_id, repo_type="model", path_or_fileobj=b"Private checkpoint upload preflight.\n",
        path_in_repo=_PREFLIGHT_MARKER, commit_message="Verify checkpoint upload access",
    )
    return f"https://huggingface.co/{repo_id}"


def push_checkpoint(repo_id: str, output: Path, *, adapter: bool = False,
                    expected_steps: int = 500) -> str:
    """Upload merged/full model at repo root; optional CTRAP adapter in adapter/."""
    if not _REPO_ID.fullmatch(repo_id):
        raise ValueError("Invalid HF repo id")
    api = _api()
    info = api.repo_info(repo_id=repo_id, repo_type="model")
    if not info.private:
        raise ValueError(f"Refusing to upload to public HF repo: {repo_id}")
    model_dir = output / "defense" / "model"
    adapter_dir = output / "defense" / "adapter"
    metrics_path = output / "defense" / "train_metrics.json"
    manifest_path = output / "manifest.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if expected_steps <= 0:
        raise ValueError("expected_steps must be positive")
    if metrics.get("global_step") != expected_steps:
        raise ValueError(
            f"Expected {expected_steps} training steps, got {metrics.get('global_step')}"
        )
    if not (model_dir / "config.json").is_file() or not (model_dir / "tokenizer_config.json").is_file():
        raise FileNotFoundError("Full model config/tokenizer missing")
    if not list(model_dir.glob("*.safetensors")):
        raise FileNotFoundError("Full-model safetensors missing")
    if adapter and not (adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError("CTRAP LoRA adapter missing")
    if not manifest_path.is_file():
        raise FileNotFoundError("Training manifest missing")

    api.upload_folder(
        repo_id=repo_id, repo_type="model", folder_path=str(model_dir),
        commit_message="Upload verified 500-step model checkpoint",
    )
    if adapter:
        api.upload_folder(
            repo_id=repo_id, repo_type="model", folder_path=str(adapter_dir),
            path_in_repo="adapter", commit_message="Upload CTRAP LoRA adapter",
        )
    api.upload_file(
        repo_id=repo_id, repo_type="model", path_or_fileobj=str(metrics_path),
        path_in_repo="train_metrics.json", commit_message="Record training metrics",
    )
    api.upload_file(
        repo_id=repo_id, repo_type="model", path_or_fileobj=str(manifest_path),
        path_in_repo="manifest.json", commit_message="Record training setup",
    )
    files = set(api.list_repo_files(repo_id=repo_id, repo_type="model"))
    required = {"config.json", "tokenizer_config.json", "train_metrics.json", "manifest.json"}
    if adapter:
        required.add("adapter/adapter_config.json")
    if not required <= files or not any(name.endswith(".safetensors") and "/" not in name for name in files):
        raise RuntimeError("HF upload completed but expected files are not visible in the repository")
    return f"https://huggingface.co/{repo_id}"
