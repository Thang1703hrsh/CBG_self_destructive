"""Upload the results table and per-case metrics to a PUBLIC Hugging Face dataset repo.

Uploads RESULTS.md (also as the repo README, so the mean ± std table shows on the repo page), results.csv,
and for every case its result.json / train_metrics.json plus each run's config.yaml. Generated responses
(responses.jsonl) are not uploaded.

  python baselines/push_results.py --out outputs/baselines --repo user/self-destruct-baselines-results
"""
import argparse
import os
from pathlib import Path

PATTERNS = ["RESULTS.md", "results.csv", "results/**/config.yaml",
            "results/**/cases/**/result.json", "results/**/cases/**/train_metrics.json"]
HEADER = "---\nlicense: mit\npretty_name: Self-destructive defense baselines (Base, SEAM, SDD, CTRAP)\n---\n\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--dry-run", action="store_true", help="list the files that would be uploaded")
    args = parser.parse_args()
    files = sorted({p for pattern in PATTERNS for p in args.out.glob(pattern) if p.is_file()})
    if not (args.out / "RESULTS.md").is_file():
        raise SystemExit(f"No RESULTS.md in {args.out}; run baselines/aggregate.py first")
    print(f"{len(files)} files -> https://huggingface.co/datasets/{args.repo}")
    if args.dry_run:
        for path in files[:20]:
            print("  ", path.relative_to(args.out))
        return
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(args.repo, repo_type="dataset", private=False, exist_ok=True)
    if api.repo_info(args.repo, repo_type="dataset").private:
        api.update_repo_settings(args.repo, private=False, repo_type="dataset")
    api.upload_folder(repo_id=args.repo, repo_type="dataset", folder_path=str(args.out),
                      allow_patterns=PATTERNS, commit_message="Update baseline results")
    readme = HEADER + (args.out / "RESULTS.md").read_text(encoding="utf-8")
    api.upload_file(repo_id=args.repo, repo_type="dataset", path_or_fileobj=readme.encode("utf-8"),
                    path_in_repo="README.md", commit_message="Results table")
    print(f"uploaded: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
