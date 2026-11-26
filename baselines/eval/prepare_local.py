"""Prepare one evaluation run on a plain GPU server (port of modal_defense_eval.prepare).

Selects the test/attack/benign rows exactly as the Modal harness does, pins Hub snapshots (base model and
harm classifier) in the local HF cache, and writes <run>/manifest.json + <run>/inputs/. Local checkpoints
(kind: volume, resolved under $DEFENSE_EVAL_SOURCES) are used in place instead of being copied.
Cases are then run with defense_eval_worker.py (unchanged), one GPU each.

  DEFENSE_EVAL_SOURCES=outputs/baselines python baselines/eval/prepare_local.py \
      --config run/config.yaml --suite robustness --models seam --run-dir run
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

from defense_eval_common import (CLASSIFIER, canonical_hash, harmful_rows, prompt_key, read_rows,
                                 resolve_config, seam_attack_rows, seam_test_rows, select_rows, sha256,
                                 volume_path, write_json)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def code_hashes():
    return {name: sha256(HERE / name) for name in
            ("defense_eval_common.py", "defense_eval_worker.py", "prepare_local.py")} | {
                "requirements.txt": sha256(REPO / "requirements.txt")}


def model_files(path, allow_bin=False):
    path = Path(path)
    weights = list(path.glob("*.safetensors"))
    if allow_bin:
        weights += list(path.glob("pytorch_model*.bin"))
    if not (path / "config.json").is_file() or not weights:
        raise FileNotFoundError(f"Need a full/merged safetensors checkpoint, not only an adapter: {path}")
    for index in path.glob("*.index.json"):
        if index.name.endswith(".bin.index.json") and not allow_bin:
            continue
        for filename in set(json.loads(index.read_text(encoding="utf-8")).get("weight_map", {}).values()):
            if Path(filename).name != filename or not (path / filename).is_file():
                raise FileNotFoundError(f"Checkpoint index refers to a missing/invalid shard: {filename}")
    # Sizes only: hashing 15 GB per local checkpoint is slow and the files are not modified afterwards.
    return {str(f.relative_to(path)): {"bytes": f.stat().st_size} for f in sorted(path.iterdir())
            if f.is_file() and f.suffix in {".json", ".safetensors", ".model", ".txt", ".bin"}}


def load_source(source, metadata, key):
    if source["kind"] == "volume":
        path = volume_path(source)
        metadata[key] = {**source, "file_sha256": sha256(path)}
        return read_rows(path)
    from datasets import load_dataset
    from huggingface_hub import HfApi

    revision = HfApi().dataset_info(source["repo_id"], revision=source.get("revision", "main")).sha
    rows = load_dataset(source["repo_id"], split=source["split"], revision=revision)
    metadata[key] = {**source, "revision": revision, "fingerprint": rows._fingerprint, "source_rows": len(rows)}
    return rows


def prepare(config, run_dir):
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoConfig, AutoTokenizer

    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["config_hash"] != canonical_hash(config):
            raise ValueError(f"{run_dir} was prepared with a different config")
        return manifest
    (run_dir / "inputs").mkdir(parents=True, exist_ok=True)
    sources, excluded = {}, set()
    for label, model in config["models"].items():
        for index, source in enumerate(model.get("defense_data", [])):
            path = volume_path(source)
            rows = read_rows(path)
            keys = {prompt_key(row.get("prompt", row.get("instruction", ""))) for row in rows}
            if "" in keys or not keys:
                raise ValueError(f"Invalid/empty defense prompt file for {label}: {path}")
            excluded.update(keys)
            sources[f"defense/{label}/{index}"] = {**source, "sha256": sha256(path),
                                                   "rows": len(rows), "unique_prompts": len(keys)}
    protocol = config["protocol"]
    test_source = load_source(config["data"]["test"], sources, "test")
    all_test_keys = {prompt_key(row["prompt"]) for row in test_source}
    attack_source = load_source(config["data"]["attack"], sources, "attack")
    if protocol["data_selection"] == "seam_original":
        test = seam_test_rows(test_source, protocol["test_size"])
        attack = seam_attack_rows(attack_source, protocol["attack_size"])
    else:
        test = select_rows(harmful_rows(test_source), protocol["test_size"], protocol["seed"], excluded)
        attack = select_rows(harmful_rows(attack_source), protocol["attack_size"], protocol["seed"],
                             excluded | all_test_keys)
    if any(not row["response"].strip() for row in attack):
        raise ValueError("Selected attack contains empty harmful responses")
    benign_source = config["data"]["benign"]
    raw_benign = load_source(benign_source, sources, "benign")
    raw_benign = raw_benign.shuffle(seed=int(benign_source.get("exclude_seed", protocol["seed"])))
    skip = int(benign_source.get("exclude_first", 0))
    benign_rows, seen = [], set()
    for row in raw_benign.select(range(skip, len(raw_benign))):
        prompt = (str(row.get("instruction", "")) + "\n" + str(row.get("input", ""))).strip()
        answer = str(row.get("output", "")).strip()
        key = prompt_key(prompt)
        if prompt and answer and key not in seen:
            benign_rows.append({"prompt": prompt, "response": answer, "prompt_id": canonical_hash(key)})
            seen.add(key)
    benign = select_rows(benign_rows, protocol["benign_size"], protocol["seed"],
                         excluded | all_test_keys | {prompt_key(row["prompt"]) for row in attack})
    for filename, rows in (("attack.json", attack), ("benign.json", benign), ("test.json", test)):
        write_json(run_dir / "inputs" / filename, rows)

    pinned = {}
    for label, model in config["models"].items():
        source = model["source"]
        if source["kind"] == "hf":
            revision = HfApi().model_info(source["repo_id"], revision=source.get("revision", "main")).sha
            local = snapshot_download(source["repo_id"], revision=revision,
                                      allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"])
            info = {**source, "revision": revision, "resolved_path": local}
        else:
            local = volume_path(source)
            info = {**source, "resolved_path": str(local)}
        info["files"] = model_files(local)
        tokenizer = AutoTokenizer.from_pretrained(local, local_files_only=True, trust_remote_code=False)
        model_config = AutoConfig.from_pretrained(local, local_files_only=True, trust_remote_code=False)
        if tokenizer.eos_token_id is None:
            raise ValueError(f"Missing tokenizer EOS: {label}")
        if protocol["context_length"] + protocol["max_new_tokens"] > model_config.max_position_embeddings:
            raise ValueError(f"Generation budget exceeds model context: {label}")
        info["model_type"] = model_config.model_type
        info["tokenizer_class"] = type(tokenizer).__name__
        pinned[label] = info
    revision = HfApi().model_info(CLASSIFIER).sha
    local = snapshot_download(CLASSIFIER, revision=revision,
                              allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "pytorch_model*.bin"])
    classifier = {"repo_id": CLASSIFIER, "revision": revision, "resolved_path": local,
                  "files": model_files(local, allow_bin=True)}
    classifier_config = AutoConfig.from_pretrained(local, local_files_only=True, trust_remote_code=False)
    if protocol["harmful_label"] not in classifier_config.id2label.values():
        raise ValueError("Configured harmful_label is missing from classifier id2label")
    counts = Counter(category for row in test for category in row["categories"])
    manifest = {
        "run_id": run_dir.name, "config": config, "config_hash": canonical_hash(config),
        "code_sha256": code_hashes(), "sources": sources, "models": pinned, "classifier": classifier,
        "input_sha256": {name: sha256(run_dir / "inputs" / name) for name in ("attack.json", "benign.json", "test.json")},
        "counts": {"attack": len(attack), "benign": len(benign), "test": len(test),
                   "excluded_defense_prompts": len(excluded), "test_categories": dict(counts),
                   "test_defense_overlap": sum(prompt_key(r["prompt"]) in excluded for r in test),
                   "attack_defense_overlap": sum(prompt_key(r["prompt"]) in excluded for r in attack),
                   "attack_test_overlap": sum(prompt_key(r["prompt"]) in all_test_keys for r in attack)},
        "metric": "MC1 and raw acc,none for all four utility tasks; zero-shot",
    }
    manifest["protocol_hash"] = canonical_hash(manifest)
    write_json(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--suite", default="robustness")
    parser.add_argument("--models", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    config = resolve_config(yaml.safe_load(args.config.read_text(encoding="utf-8")), args.suite, args.models)
    manifest = prepare(config, args.run_dir)
    print("PREPARED", args.run_dir, json.dumps(manifest["counts"]))


if __name__ == "__main__":
    main()
