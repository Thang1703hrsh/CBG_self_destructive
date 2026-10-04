"""Data pools + pinned model for CBG v2 (SEAM main-run recipe), shared by Modal and local servers.

Downloads the pinned RepNoise JSON (SHA-256 checked), BeaverTails 30k_test / 330k_train and
Alpaca at pinned revisions, builds the pools (src/experiments/seam_data.py), downloads the model
at a pinned revision and writes the truncation report.

Usage (local server):
  python -m src.experiments.prepare_v2 --config <yaml> --data <dir>        # prepare if missing
  python -m src.experiments.prepare_v2 --data <dir> --model-path           # print the local model path
HF_TOKEN must give access to the configured model (gated: Llama-2, Mistral).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import urllib.request

import yaml

from src.experiments.protocol import file_digest, read_json, write_json

ALPACA_REVISION = "12567cabf869d7c92e573c7c783905fc160e9639"
BEAVERTAILS_REVISION = "325c10601deaf84618333d9624a0637543e9ae62"
MODEL_PATTERNS = ["*.json", "*.safetensors", "tokenizer.model", "merges.txt"]   # merges.txt: Qwen BPE
# Mistral repos also ship consolidated.safetensors (mistral-inference format, ~14.5 GB), unused by transformers.
MODEL_IGNORE = ["consolidated*"]


def prepare_data(config, target):
    """Build every pool under ``target``; refuses to overwrite a prepared directory."""
    target = Path(target)
    if (target / "manifest.json").exists():
        if (target / "model_revision.json").exists():
            raise ValueError(f"{target} already prepared; use a new directory to rebuild")
        # The pools were written but the model step failed (e.g. a 401 from Hugging Face):
        # keep the pools (pinned, deterministic) and finish only the model download.
        return _prepare_model(config, target, load_pools(target))
    from datasets import load_dataset
    from huggingface_hub import HfApi, hf_hub_download
    from src.experiments.seam_data import build_pools, write_pools
    target.mkdir(parents=True, exist_ok=True)
    data = config["data"]
    # SEAM main run: PUBLIC_DATA_URL of scripts/modal_seam_full.py (same bytes as the HF mirror).
    repnoise_path = target / "source_beavertails_with_refusals_train.json"
    urllib.request.urlretrieve(data["safety_source"], repnoise_path)
    if file_digest(repnoise_path) != data["safety_source_sha256"]:
        raise ValueError("downloaded RepNoise JSON differs from the pinned SHA-256 of the SEAM run")
    repnoise = json.loads(repnoise_path.read_text(encoding="utf-8"))
    def parquet(name):
        path = hf_hub_download("PKU-Alignment/BeaverTails", f"default/{name}/0000.parquet",
                               repo_type="dataset", revision=BEAVERTAILS_REVISION)
        return [dict(r) for r in load_dataset("parquet", data_files={"x": path}, split="x")]
    evaluation, extra = parquet("30k_test"), parquet("330k_train")
    alpaca_path = hf_hub_download("yahma/alpaca-cleaned", "alpaca_data_cleaned.json", repo_type="dataset",
                                  revision=ALPACA_REVISION)
    alpaca = json.loads(Path(alpaca_path).read_text(encoding="utf-8"))
    pools = build_pools(repnoise, extra, evaluation, alpaca, data["sizes"], seed=config["seeds"]["data_split"],
                        benign_shuffle=data["benign_shuffle"], test_size=data["test_size"])
    model = config["model"]
    revision = config.get("model_revision") or HfApi().model_info(model).sha
    sources = dict(repnoise=dict(url=data["safety_source"], sha256=data["safety_source_sha256"]),
                   seam_split=dict(seed=config["seeds"]["data_split"], shuffle="random.Random(seed).shuffle",
                                   benign_shuffle=data["benign_shuffle"], test_size=data["test_size"]),
                   beavertails=dict(revision=BEAVERTAILS_REVISION, eval="30k_test", extra="330k_train"),
                   alpaca=dict(repo="yahma/alpaca-cleaned", revision=ALPACA_REVISION),
                   model=dict(name=model, revision=revision))
    write_pools(pools, target, sources)
    return _prepare_model(config, target, pools, revision)


def load_pools(target):
    """The pools listed in an existing manifest (name -> rows)."""
    manifest = read_json(Path(target) / "manifest.json")
    return {name: read_json(Path(target) / f"{name}.json") for name in manifest["files"]}


def _prepare_model(config, target, pools, revision=None):
    """Download the pinned model, then write model_revision.json and the truncation report (last step)."""
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoTokenizer
    from src.experiments.seam_data import truncation_report
    model = config["model"]
    revision = revision or config.get("model_revision") or \
        read_json(target / "manifest.json").get("sources", {}).get("model", {}).get("revision") or \
        HfApi().model_info(model).sha
    snapshot_download(model, revision=revision, allow_patterns=MODEL_PATTERNS, ignore_patterns=MODEL_IGNORE)
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    if config["lengths"]["safety"] != config["lengths"]["benign"]:
        print("NOTE: SEAM truncates every branch at one CONTEXT_LENGTH; lengths differ here", flush=True)
    report = truncation_report(tokenizer, pools, max_length=config["lengths"]["safety"])
    write_json(target / "truncation_report.json", report)
    # Written last: its presence marks a complete prepare (checked by the launchers).
    write_json(target / "model_revision.json", dict(model=model, revision=revision))
    manifest = read_json(target / "manifest.json")
    return dict(data=str(target), files=manifest["files"], overlaps=manifest.get("overlap_report"),
                truncation=report)


def model_path(data_dir, local_files_only=True):
    """Local snapshot of the model pinned at prepare time."""
    from huggingface_hub import snapshot_download
    pinned = read_json(Path(data_dir) / "model_revision.json")
    return snapshot_download(pinned["model"], revision=pinned["revision"], local_files_only=local_files_only,
                             allow_patterns=MODEL_PATTERNS, ignore_patterns=MODEL_IGNORE)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-path", action="store_true", help="print the pinned model's local path and exit")
    args = parser.parse_args(argv)
    if args.model_path:
        print(model_path(args.data))
        return
    if (Path(args.data) / "model_revision.json").exists():   # written last by a complete prepare
        print(f"{args.data} already prepared", flush=True)
        return
    if not args.config:
        parser.error("--config is required to prepare")
    result = prepare_data(yaml.safe_load(Path(args.config).read_text(encoding="utf-8")), args.data)
    print(json.dumps(dict(data=result["data"], files=result["files"]), indent=2))


if __name__ == "__main__":
    main()
