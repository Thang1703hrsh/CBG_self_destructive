"""Detached, resumable evaluation of existing Base/SDD/CTRAP/SEAM checkpoints.

Use Run-DefenseEval.ps1 to select configuration/suite/models before importing
this module. Source volumes are read-only; all new artifacts have their own volume.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path, PurePosixPath

import modal
import yaml

REMOTE = PurePosixPath("/root/defense-eval")
# Modal hydrates this module as /root/modal_defense_eval.py; auxiliary files
# are mounted under /root/defense-eval/scripts by the Image definition below.
REPO = Path(REMOTE) if os.environ.get("MODAL_IS_REMOTE") == "1" else Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from defense_eval_common import (CLASSIFIER, canonical_hash, harmful_rows,
                                 markdown_summary, prompt_key, read_rows,
                                 resolve_config, seam_attack_rows, seam_test_rows,
                                 select_rows, sha256,
                                 source_volumes, volume_path, write_json)

CONFIG_PATH = Path(os.environ.get("DEFENSE_EVAL_CONFIG", str(REPO / "config" / "defense_eval.yaml"))).resolve()
SUITE = os.environ.get("DEFENSE_EVAL_SUITE", "core")
MODELS = os.environ.get("DEFENSE_EVAL_MODELS", "sdd,ctrap,seam")
PARALLEL_ENV = os.environ.get("DEFENSE_EVAL_PARALLEL", "6")
CONFIG = resolve_config(yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")), SUITE, MODELS)
app = modal.App("defense-checkpoint-evaluation")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(REPO / "requirements.txt"))
    .pip_install("wandb==0.19.8", "requests==2.32.3", "sentencepiece==0.2.0",
                 "bitsandbytes==0.45.2")
    .env({"HF_HOME": "/hf", "WANDB_MODE": "disabled", "USE_TF": "0",
          "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1",
          "DEFENSE_EVAL_CONFIG": str(REMOTE / "config.yaml"),
          "DEFENSE_EVAL_SUITE": SUITE, "DEFENSE_EVAL_MODELS": MODELS,
          "DEFENSE_EVAL_PARALLEL": str(PARALLEL_ENV)})
    .add_local_file(str(CONFIG_PATH), str(REMOTE / "config.yaml"))
    .add_local_file(str(REPO / "requirements.txt"), str(REMOTE / "requirements.txt"))
)
for filename in ("defense_eval_common.py", "defense_eval_worker.py", "modal_defense_eval.py"):
    image = image.add_local_file(str(REPO / "scripts" / filename), str(REMOTE / "scripts" / filename))
results_volume = modal.Volume.from_name("defense-eval-results", create_if_missing=True)
hf_volume = modal.Volume.from_name("defense-eval-hf-cache", create_if_missing=True)
source_mounts = {
    f"/sources/{name}": modal.Volume.from_name(name).with_mount_options(read_only=True)
    for name in source_volumes(CONFIG)
}
VOLUMES = {"/results": results_volume, "/hf": hf_volume, **source_mounts}
SECRET = [modal.Secret.from_name("huggingface-secret")]
# Max concurrent L40S case containers; set DEFENSE_EVAL_PARALLEL=1 for sequential.
PARALLEL = max(1, int(os.environ.get("DEFENSE_EVAL_PARALLEL", "6")))


def root_for(run_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id):
        raise ValueError("Run ID must contain 1-80 letters, digits, _ or -")
    return Path("/results") / run_id


def code_hashes():
    return {name: sha256(REMOTE / "scripts" / name)
            for name in ("defense_eval_common.py", "defense_eval_worker.py", "modal_defense_eval.py")} | {
                "requirements.txt": sha256(REMOTE / "requirements.txt")}


def model_files(path, allow_bin=False):
    path = Path(path)
    weights = list(path.glob("*.safetensors"))
    if allow_bin:
        weights += list(path.glob("pytorch_model*.bin"))
    if not (path / "config.json").is_file() or not weights:
        raise FileNotFoundError(f"Need a full/merged safetensors checkpoint, not only an adapter: {path}")
    for index in path.glob("*.index.json"):
        # Hub repos like Llama-2 ship both formats; .bin shards are not downloaded.
        if index.name.endswith(".bin.index.json") and not allow_bin:
            continue
        for filename in set(json.loads(index.read_text(encoding="utf-8")).get("weight_map", {}).values()):
            if Path(filename).name != filename or not (path / filename).is_file():
                raise FileNotFoundError(f"Checkpoint index refers to a missing/invalid shard: {filename}")
    return {str(file.relative_to(path)): {"sha256": sha256(file), "bytes": file.stat().st_size}
            for file in sorted(path.iterdir()) if file.is_file()
            and file.suffix in {".json", ".safetensors", ".model", ".txt", ".bin"}}


def load_source(source, metadata, key):
    if source["kind"] == "volume":
        path = volume_path(source)
        metadata[key] = {**source, "file_sha256": sha256(path)}
        return read_rows(path)
    from datasets import load_dataset
    from huggingface_hub import HfApi

    revision = HfApi().dataset_info(source["repo_id"], revision=source.get("revision", "main")).sha
    rows = load_dataset(source["repo_id"], split=source["split"], revision=revision)
    metadata[key] = {**source, "revision": revision, "fingerprint": rows._fingerprint,
                     "source_rows": len(rows)}
    return rows


@app.function(image=image, cpu=2, memory=8192, timeout=24 * 3600,
              secrets=SECRET, volumes=VOLUMES)
def prepare(run_id: str, resume: bool):
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoConfig, AutoTokenizer

    results_volume.reload()
    hf_volume.reload()
    root = root_for(run_id)
    manifest_path = root / "manifest.json"
    if resume:
        if not manifest_path.is_file():
            raise FileNotFoundError("No completed preparation to resume; use a new run ID")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["config_hash"] != canonical_hash(CONFIG) or manifest["code_sha256"] != code_hashes():
            raise ValueError("Resume requires the same resolved config, suite, models, and code")
        for name, digest in manifest["input_sha256"].items():
            if sha256(root / "inputs" / name) != digest:
                raise ValueError(f"Prepared input changed: {name}")
        for label, info in manifest["models"].items():
            if model_files(info["resolved_path"]) != info["files"]:
                raise ValueError(f"Prepared checkpoint changed: {label}")
        if model_files(manifest["classifier"]["resolved_path"], allow_bin=True) != manifest["classifier"]["files"]:
            raise ValueError("Prepared classifier changed")
        return manifest
    if root.exists():
        raise FileExistsError("Run exists; use -Resume with identical options, or a new run ID")
    (root / "inputs").mkdir(parents=True)
    status = {"state": "running", "stage": "cpu_preparation", "run_id": run_id}
    write_json(root / "status.json", status)
    results_volume.commit()
    try:
        sources, excluded = {}, set()
        for label, model in CONFIG["models"].items():
            for index, source in enumerate(model.get("defense_data", [])):
                path = volume_path(source)
                rows = read_rows(path)
                keys = {prompt_key(row.get("prompt", row.get("instruction", ""))) for row in rows}
                if "" in keys or not keys:
                    raise ValueError(f"Invalid/empty defense prompt file for {label}: {path}")
                excluded.update(keys)
                sources[f"defense/{label}/{index}"] = {**source, "sha256": sha256(path),
                                                       "rows": len(rows), "unique_prompts": len(keys)}
        protocol = CONFIG["protocol"]
        test_source = load_source(CONFIG["data"]["test"], sources, "test")
        # Exclude *all* test prompts from attack training, including safe QA rows.
        all_test_keys = {prompt_key(row["prompt"]) for row in test_source}
        attack_source = load_source(CONFIG["data"]["attack"], sources, "attack")
        if protocol["data_selection"] == "seam_original":
            # Paper evaluator: no shuffling or overlap exclusion; overlaps are reported.
            test = seam_test_rows(test_source, protocol["test_size"])
            attack = seam_attack_rows(attack_source, protocol["attack_size"])
        else:
            test = select_rows(harmful_rows(test_source), protocol["test_size"],
                               protocol["seed"], excluded)
            attack = select_rows(harmful_rows(attack_source), protocol["attack_size"],
                                 protocol["seed"], excluded | all_test_keys)
        if any(not row["response"].strip() for row in attack):
            raise ValueError("Selected attack contains empty harmful responses")
        benign_source = CONFIG["data"]["benign"]
        raw_benign = load_source(benign_source, sources, "benign")
        raw_benign = raw_benign.shuffle(seed=int(benign_source.get("exclude_seed", protocol["seed"])))
        skip = int(benign_source.get("exclude_first", 0))
        if skip < 0 or skip >= len(raw_benign):
            raise ValueError("Invalid benign exclude_first")
        benign_rows = []
        seen = set()
        for row in raw_benign.select(range(skip, len(raw_benign))):
            prompt = (str(row.get("instruction", "")) + "\n" + str(row.get("input", ""))).strip()
            answer = str(row.get("output", "")).strip()
            key = prompt_key(prompt)
            if prompt and answer and key not in seen:
                benign_rows.append({"prompt": prompt, "response": answer,
                                    "prompt_id": canonical_hash(key)})
                seen.add(key)
        benign = select_rows(benign_rows, protocol["benign_size"], protocol["seed"],
                             excluded | all_test_keys | {prompt_key(row["prompt"]) for row in attack})
        for filename, rows in (("attack.json", attack), ("benign.json", benign), ("test.json", test)):
            write_json(root / "inputs" / filename, rows)
        status["stage"] = "pin_and_download_checkpoints"
        write_json(root / "status.json", status)
        results_volume.commit()
        pinned = {}
        for label, model in CONFIG["models"].items():
            source = model["source"]
            if source["kind"] == "hf":
                revision = HfApi().model_info(source["repo_id"], revision=source.get("revision", "main")).sha
                local = snapshot_download(source["repo_id"], revision=revision,
                                          allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"])
                info = {**source, "revision": revision, "resolved_path": local}
            else:
                # A frozen copy keeps subsequent source-volume edits out of this run.
                original = volume_path(source)
                model_files(original)
                local = root / "inputs" / "models" / label
                shutil.copytree(original, local)
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
            print("PINNED_MODEL", label, info.get("revision", "volume snapshot"), flush=True)
        revision = HfApi().model_info(CLASSIFIER).sha
        local = snapshot_download(CLASSIFIER, revision=revision,
                                  allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "pytorch_model*.bin"])
        classifier = {"repo_id": CLASSIFIER, "revision": revision,
                      "resolved_path": local, "files": model_files(local, allow_bin=True)}
        AutoTokenizer.from_pretrained(local, local_files_only=True, trust_remote_code=False)
        classifier_config = AutoConfig.from_pretrained(local, local_files_only=True, trust_remote_code=False)
        if protocol["harmful_label"] not in classifier_config.id2label.values():
            raise ValueError("Configured harmful_label is missing from classifier id2label")
        counts = Counter(category for row in test for category in row["categories"])
        manifest = {
            "run_id": run_id, "config": CONFIG, "config_hash": canonical_hash(CONFIG),
            "code_sha256": code_hashes(), "sources": sources, "models": pinned,
            "classifier": classifier,
            "input_sha256": {name: sha256(root / "inputs" / name)
                             for name in ("attack.json", "benign.json", "test.json")},
            "counts": {"attack": len(attack), "benign": len(benign), "test": len(test),
                       "excluded_defense_prompts": len(excluded), "test_categories": dict(counts),
                       "test_defense_overlap": sum(prompt_key(r["prompt"]) in excluded for r in test),
                       "attack_defense_overlap": sum(prompt_key(r["prompt"]) in excluded for r in attack),
                       "attack_test_overlap": sum(prompt_key(r["prompt"]) in all_test_keys for r in attack)},
            "generation": {"do_sample": False, "num_beams": 1,
                           "max_new_tokens": protocol["max_new_tokens"]},
            "attack_provenance": "new common split; excludes selected defense and all test prompts",
            "metric": "MC1 and raw acc,none for all four utility tasks; zero-shot",
        }
        manifest["protocol_hash"] = canonical_hash(manifest)
        write_json(manifest_path, manifest)
        status.update(state="prepared", stage="cpu_preparation_complete")
        write_json(root / "status.json", status)
        results_volume.commit()
        hf_volume.commit()
        return manifest
    except Exception as exc:
        status.update(state="failed", error=repr(exc))
        write_json(root / "status.json", status)
        results_volume.commit()
        raise


@app.function(image=image, gpu="L40S", cpu=4, memory=16384, timeout=12 * 3600,
              secrets=SECRET, volumes={"/results": results_volume, "/hf": hf_volume})
def evaluate_case(run_id: str, label: str, case_id: str):
    results_volume.reload()
    hf_volume.reload()
    output = root_for(run_id) / "cases" / label / case_id
    output.mkdir(parents=True, exist_ok=True)
    status = {"state": "running", "model": label, "scenario": case_id}
    write_json(output / "status.json", status)
    results_volume.commit()
    try:
        subprocess.run([
            sys.executable, str(REMOTE / "scripts" / "defense_eval_worker.py"),
            "--manifest", str(root_for(run_id) / "manifest.json"),
            "--model", label, "--case", case_id,
        ], check=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
        result = json.loads((output / "result.json").read_text(encoding="utf-8"))
        status["state"] = "complete"
        write_json(output / "status.json", status)
        return result
    except Exception as exc:
        status.update(state="failed", error=repr(exc))
        write_json(output / "status.json", status)
        raise
    finally:
        results_volume.commit()
        hf_volume.commit()


@app.function(image=image, cpu=0.125, memory=512, timeout=24 * 3600,
              volumes={"/results": results_volume})
def pipeline(run_id: str, resume: bool = False, prepare_only: bool = False):
    root = root_for(run_id)
    # prepare owns its own failure status and validates immutable artifacts on resume.
    manifest = prepare.remote(run_id, resume)
    results_volume.reload()
    if prepare_only:
        return {"state": "prepared", "run_id": run_id, "counts": manifest["counts"]}
    summary = {"run_id": run_id, "suite": CONFIG["suite"],
               "protocol_version": CONFIG["protocol_version"],
               "protocol_hash": manifest["protocol_hash"], "metrics": {}}
    status = {"run_id": run_id, "state": "running", "stage": "evaluation"}
    try:
        keys = [(label, case["id"]) for label in CONFIG["models"] for case in CONFIG["cases"]]
        results, pending = {}, []
        for label, case_id in keys:
            path = root / "cases" / label / case_id / "result.json"
            if resume and path.is_file():
                result = json.loads(path.read_text(encoding="utf-8"))
                if (result.get("state") != "complete"
                        or result.get("protocol_hash") != manifest["protocol_hash"]
                        or result.get("model") != label or result.get("scenario") != case_id):
                    raise ValueError(f"Existing result does not match this run: {label}/{case_id}")
                print("RESUME_SKIP", f"{label}/{case_id}", flush=True)
                results[(label, case_id)] = result
            else:
                pending.append((label, case_id))

        def publish():
            # Table rows keep model/case order regardless of completion order.
            summary["metrics"] = {f"{label}/{case_id}": results[(label, case_id)]
                                  for label, case_id in keys if (label, case_id) in results}
            write_json(root / "summary.json", summary)
            (root / "comparison.md").write_text(markdown_summary(summary), encoding="utf-8")
            results_volume.commit()

        publish()
        # Independent cases, one L40S each; PARALLEL caps concurrent GPUs.
        failures = []
        for start in range(0, len(pending), PARALLEL):
            chunk = pending[start:start + PARALLEL]
            status["stage"] = ", ".join(f"{label}/{case_id}" for label, case_id in chunk)
            write_json(root / "status.json", status)
            results_volume.commit()
            outputs = evaluate_case.starmap([(run_id, label, case_id) for label, case_id in chunk],
                                            return_exceptions=True)
            for item, output in zip(chunk, outputs):
                if isinstance(output, BaseException):
                    failures.append(f"{item[0]}/{item[1]}: {output!r}")
                else:
                    results[item] = output
            results_volume.reload()
            publish()
        if failures:
            raise RuntimeError("Failed cases (rerun with -Resume): " + "; ".join(failures))
        status.update(state="complete", stage="complete", completed_cases=len(summary["metrics"]))
        write_json(root / "status.json", status)
        results_volume.commit()
        return {"run_id": run_id, "state": "complete", "completed_cases": len(summary["metrics"])}
    except Exception as exc:
        results_volume.reload()
        status.update(state="failed", error=repr(exc))
        write_json(root / "status.json", status)
        results_volume.commit()
        raise


@app.local_entrypoint()
def main(run_id: str, resume: bool = False, prepare_only: bool = False):
    root_for(run_id)
    call = pipeline.spawn(run_id, resume, prepare_only)
    print(f"Submitted Modal call: {call.object_id}")
    print(f"Suite: {CONFIG['suite']}; models: {','.join(CONFIG['models'])}")
    print(f"Results: defense-eval-results/{run_id}")
