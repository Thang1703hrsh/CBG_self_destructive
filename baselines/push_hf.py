"""Upload one trained baseline checkpoint to a PUBLIC Hugging Face model repo.

Creates <namespace>/<repo> if needed. A repo that already holds a model (config.json + safetensors) is
left untouched, so reruns are safe. Uploads the full model directory, the CTRAP LoRA adapter if present,
training metrics/logs, and a short model card.

  python baselines/push_hf.py --model-dir outputs/baselines/ckpt/qwen/sdd-seed42/model \
      --repo user/sdd-qwen25-7b-instruct-seed42-step500 --method SDD --base Qwen/Qwen2.5-7B-Instruct --seed 42
"""
import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi

CARD = """---
base_model: {base}
library_name: transformers
tags: [safety, harmful-fine-tuning, self-destructive-defense, {tag}]
---

# {method} defense on {base} (seed {seed})

Research checkpoint: the **{method}** baseline defense against harmful fine-tuning, trained from
`{base}` for {steps} steps with seed {seed}. It was produced for the baseline comparison of
*No Cheap Escape: Collateral Barrier Geometry for Self-Destructive Language Models* with
`baselines/run_all.py` of that repository. It is not a safety-certified model.

Usage and license follow the base model `{base}`.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--extra", type=Path, nargs="*", default=[], help="additional files (metrics, logs)")
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--marker", type=Path, default=None, help="written after a successful upload")
    args = parser.parse_args()
    result = {"repo": args.repo, "url": f"https://huggingface.co/{args.repo}", "public": True}
    if not (args.model_dir / "config.json").is_file() or not list(args.model_dir.glob("*.safetensors")):
        raise FileNotFoundError(f"No full safetensors checkpoint in {args.model_dir}")
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(args.repo, repo_type="model", private=False, exist_ok=True)
    info = api.repo_info(args.repo, repo_type="model")
    if info.private:
        api.update_repo_settings(args.repo, private=False)
    files = set(api.list_repo_files(args.repo))
    if "config.json" in files and any(name.endswith(".safetensors") and "/" not in name for name in files):
        print(f"already uploaded: https://huggingface.co/{args.repo}")
        if args.marker:
            args.marker.write_text(json.dumps(result), encoding="utf-8")
        return
    api.upload_folder(repo_id=args.repo, folder_path=str(args.model_dir),
                      commit_message=f"{args.method} seed {args.seed}: model")
    if args.adapter_dir and (args.adapter_dir / "adapter_config.json").is_file():
        api.upload_folder(repo_id=args.repo, folder_path=str(args.adapter_dir), path_in_repo="adapter",
                          commit_message="LoRA adapter")
    for extra in args.extra:
        if extra.is_file():
            api.upload_file(repo_id=args.repo, path_or_fileobj=str(extra), path_in_repo=extra.name,
                            commit_message=f"Add {extra.name}")
    card = CARD.format(base=args.base, method=args.method, seed=args.seed, steps=args.steps,
                       tag=args.method.lower())
    api.upload_file(repo_id=args.repo, path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md",
                    commit_message="Model card")
    if args.marker:
        args.marker.write_text(json.dumps(result), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
