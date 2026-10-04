"""Safely diagnose Hugging Face authentication inside Modal."""

from __future__ import annotations

import json
import os

import modal


# Defaults: the model and revision pinned by `modal_cbg_v2_7b.py --stage prepare` (config.json of that commit).
MODEL = "meta-llama/Llama-2-7b-chat-hf"
REVISION = "f5db02db724555f92da89c216ac04704f23d4590"

app = modal.App("cbg-hf-access-check")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "huggingface_hub==0.29.3"
)
secret = modal.Secret.from_name("HF_TOKEN")


def error_summary(exc: Exception) -> dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc)[:1000],
    }


@app.function(image=image, secrets=[secret], cpu=1, memory=1024, timeout=300)
def check_access(model: str = MODEL, revision: str = REVISION) -> dict:
    from huggingface_hub import HfApi, hf_hub_download

    token = os.environ.get("HF_TOKEN")
    report = {
        "model": model,
        "revision": revision,
        "token_present": bool(token),
        "token_prefix_ok": bool(token and token.startswith("hf_")),
        "token_length": len(token or ""),
    }
    if not token:
        report["status"] = "missing_hf_token"
        return report

    try:
        identity = HfApi(token=token).whoami()
        report["hf_username"] = identity.get("name")
        report["token_role"] = identity.get("auth", {}).get("accessToken", {}).get("role")
    except Exception as exc:
        report["status"] = "token_invalid"
        report["identity_error"] = error_summary(exc)
        return report

    try:
        path = hf_hub_download(
            model,
            "config.json",
            revision=revision,
            token=token,
            force_download=True,
        )
        report["downloaded_file"] = os.path.basename(path)
        report["status"] = "access_ok"
    except Exception as exc:
        report["status"] = "repo_access_denied"
        report["download_error"] = error_summary(exc)
    return report


@app.local_entrypoint()
def main(model: str = MODEL, revision: str = REVISION):
    print(json.dumps(check_access.remote(model, revision), indent=2, ensure_ascii=False))
