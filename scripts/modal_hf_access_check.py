"""Safely diagnose Hugging Face authentication inside Modal."""

from __future__ import annotations

import json
import os

import modal


MODEL = "meta-llama/Llama-3.2-1B-Instruct"
REVISION = "9213176726f574b556790deb65791e0c5aa438b6"

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
def check_access() -> dict:
    from huggingface_hub import HfApi, hf_hub_download

    token = os.environ.get("HF_TOKEN")
    report = {
        "model": MODEL,
        "revision": REVISION,
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
            MODEL,
            ".gitattributes",
            revision=REVISION,
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
def main():
    print(json.dumps(check_access.remote(), indent=2, ensure_ascii=False))
