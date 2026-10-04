"""Detached SDD comparison run on Modal. Does not claim exact paper reproduction."""

import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import modal

repo = Path(__file__).resolve().parent.parent
app = modal.App("sdd-full-comparison")
prep_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("datasets==3.3.1", "sentence-transformers==3.4.1", "requests==2.32.3")
    .add_local_dir(str(repo / "scripts"), remote_path="/root/seam/scripts")
)
work_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(repo / "requirements.txt"))
    .pip_install("wandb==0.19.8", "requests==2.32.3", "bitsandbytes==0.45.2")
    .add_local_dir(str(repo / "src"), remote_path="/root/seam/src")
    .add_local_dir(str(repo / "config"), remote_path="/root/seam/config")
    .add_local_dir(str(repo / "scripts"), remote_path="/root/seam/scripts")
)
hub_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub==0.36.2", "hf_xet")
    .add_local_dir(str(repo / "scripts"), remote_path="/root/seam/scripts")
)
data_volume = modal.Volume.from_name("defense-compare-data", create_if_missing=True)
hf_volume = modal.Volume.from_name("seam-main-hf-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("defense-compare-results", create_if_missing=True)


def _paths(seed, run_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", run_id):
        raise ValueError("run_id must contain 1-64 letters, digits, _ or -")
    return (Path("/data") / f"sdd-seed{seed}-{run_id}",
            Path("/results") / f"sdd-seed{seed}-{run_id}")


def _restarted_status(output, **expected):
    """Status of an unfinished attempt of this exact run, else None.

    Modal restarts a preempted input with the same arguments; such a restart may
    rebuild its own partial outputs instead of refusing them.
    """
    try:
        previous = json.loads((output / "status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if previous.get("state") == "complete":
        return None
    if any(previous.get(key) != value for key, value in expected.items()):
        return None
    return previous


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_preflight(seed, run_id, answer_pool, data_dir, output,
                        expected_pairs=8000, restarted_train=False):
    """Fail before renting a GPU unless a complete, matching CPU preflight exists."""
    status_path = output / "status.json"
    manifest_path = data_dir / "preflight_manifest.json"
    if not status_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("SDD preflight is missing; run Run-SDDPreflight.ps1 first")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # A restarted train-push attempt has already replaced the preflight status.
    preflight_done = (status.get("mode") == "data_preflight" and status.get("state") == "complete") or (
        restarted_train and status.get("mode") == "train_push")
    if (not preflight_done
            or status.get("seed") != seed or status.get("run_id") != run_id
            or status.get("answer_pool") != answer_pool):
        raise ValueError("SDD preflight status does not match this seed, run ID, and answer pool")
    if (manifest.get("seed") != seed or manifest.get("run_id") != run_id
            or manifest.get("answer_pool") != answer_pool):
        raise ValueError("SDD preflight manifest does not match this run")
    split = manifest.get("split") or {}
    if (split.get("attack_count") != 1000 or not split.get("source_sha256")):
        raise ValueError("SDD preflight has no valid shared attack split")
    data_manifest_path = data_dir / "sdd_data_manifest.json"
    if not data_manifest_path.is_file():
        raise FileNotFoundError("SDD data manifest is missing")
    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    if (data_manifest.get("pairs") != expected_pairs
            or data_manifest.get("target_pairs") != expected_pairs
            or data_manifest.get("paper_equivalent_answer_pool") != (answer_pool == "AlpacaLima")):
        raise ValueError("SDD preflight did not produce the requested 8,000-pair answer pool")
    paths = {
        "attack.json": data_dir / "attack.json",
        "ctrap_defense.json": data_dir / "ctrap_defense.json",
        "sdd_pairs.jsonl": data_dir / "sdd_pairs.jsonl",
        "sdd_data_manifest.json": data_manifest_path,
    }
    for name, path in paths.items():
        if not path.is_file() or _sha256(path) != manifest.get("sha256", {}).get(name):
            raise ValueError(f"SDD preflight file missing or changed: {name}")
    with paths["sdd_pairs.jsonl"].open(encoding="utf-8") as stream:
        if sum(1 for line in stream if line.strip()) != expected_pairs:
            raise ValueError("SDD pair file does not contain exactly 8,000 nonempty rows")
    return manifest


def _match_irrelevant_answers(selected, answers, prompt_embeddings, sentence_model,
                              seed, max_similarity=0.25, max_resample=50):
    """Uniform random answer draws, embedding only answers actually sampled."""
    import numpy as np

    sample_rng = np.random.default_rng(seed)
    pairs = [None] * len(selected)
    pending = list(range(len(selected)))
    answer_embedding_cache = {}
    for attempt in range(1, max_resample + 1):
        sampled_indices = sample_rng.integers(0, len(answers), size=len(pending))
        missing = list(dict.fromkeys(
            int(index) for index in sampled_indices if int(index) not in answer_embedding_cache
        ))
        if missing:
            embeddings = sentence_model.encode(
                [answers[index] for index in missing], batch_size=128,
                normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
            ).astype(np.float32)
            answer_embedding_cache.update(zip(missing, embeddings))
        still_pending = []
        for position, index in zip(pending, sampled_indices):
            index = int(index)
            similarity = float(np.dot(prompt_embeddings[position], answer_embedding_cache[index]))
            if similarity <= max_similarity:
                prompt, category = selected[position]
                pairs[position] = {"prompt": prompt, "completion": answers[index],
                                   "category": category, "cosine_sim": similarity,
                                   "resample_tries": attempt}
            else:
                still_pending.append(position)
        pending = still_pending
        print(f"SDD_MATCH round={attempt} remaining={len(pending)} "
              f"encoded_answers={len(answer_embedding_cache)}", flush=True)
        if not pending:
            break
    if pending:
        raise ValueError(f"Could not find irrelevant answers for {len(pending)} prompts "
                         f"in {max_resample} tries")
    return pairs, len(answer_embedding_cache)


def _build_sdd_pairs(seed, data_dir, attack_path, include_lima=True, target_pairs=8000):
    """Build categorized pairs; reduced Alpaca-only data is for smoke diagnostics."""
    import numpy as np
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from sentence_transformers import SentenceTransformer

    sys.path.insert(0, "/root/seam/scripts")
    from modal_compare_common import write_json

    attack = json.loads(attack_path.read_text(encoding="utf-8"))
    attack_prompts = {row["prompt"].strip().casefold() for row in attack}
    beaver = load_dataset("PKU-Alignment/BeaverTails", split="330k_train")
    beaver_test = load_dataset("PKU-Alignment/BeaverTails", split="30k_test")
    test_prompts = {str(row["prompt"]).strip().casefold() for row in beaver_test}
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    # Full runs intentionally stop before renting a GPU unless LIMA is accessible.
    lima = None
    if include_lima:
        # GAIR/lima is script-backed; datasets 3.x no longer executes dataset
        # scripts. Download the gated JSONL through the authenticated Hub API.
        lima_file = hf_hub_download(
            repo_id="GAIR/lima", filename="train.jsonl", repo_type="dataset",
        )
        lima = load_dataset("json", data_files={"train": lima_file}, split="train")
    answers = [row["output"].strip() for row in alpaca if row.get("output")]
    if lima is not None:
        answers += [
            row["conversations"][1].strip() for row in lima
            if len(row.get("conversations", [])) > 1 and row["conversations"][1]
        ]
    answers = list(dict.fromkeys(answer for answer in answers if answer))
    if len(answers) < (8000 if target_pairs == 8000 else 1024):
        raise ValueError(f"High-quality answer pool too small: {len(answers)}")
    if target_pairs < 8000:
        answers = random.Random(seed).sample(answers, 1024)

    candidates = {}
    seen = set()
    for row in beaver:
        prompt = str(row["prompt"]).strip()
        key = prompt.casefold()
        if (row["is_safe"] is not False or not prompt or key in attack_prompts
                or key in test_prompts or key in seen):
            continue
        seen.add(key)
        categories = row.get("category") or {}
        active = [name for name, yes in categories.items() if yes] if isinstance(categories, dict) else []
        if active:
            candidates[key] = (prompt, active)
    if len(candidates) < 8000:
        raise ValueError(f"Need 8,000 disjoint categorized BeaverTails prompts, got {len(candidates)}")
    print(f"SDD_ELIGIBLE_CANDIDATES {len(candidates)}", flush=True)

    rng = random.Random(seed)
    category_names = sorted({cat for _, active in candidates.values() for cat in active})
    if len(category_names) != 14:
        raise ValueError(f"Paper expects 14 harmful categories; found {len(category_names)}")
    buckets = {cat: [key for key, (_, active) in candidates.items() if cat in active] for cat in category_names}
    candidate_category_counts = {cat: len(keys) for cat, keys in buckets.items()}
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected = []
    selected_set = set()
    counts = Counter()
    while len(selected) < target_pairs:
        progress = False
        for cat in sorted(category_names, key=lambda name: (counts[name], name)):
            while buckets[cat] and buckets[cat][-1] in selected_set:
                buckets[cat].pop()
            if not buckets[cat]:
                continue
            key = buckets[cat].pop()
            selected.append((candidates[key][0], cat))
            selected_set.add(key)
            counts[cat] += 1
            progress = True
            if len(selected) == target_pairs:
                break
        if not progress:
            raise ValueError(f"Could only balance {len(selected)} unique SDD prompts")

    sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    prompt_embeddings = sentence_model.encode(
        [prompt for prompt, _ in selected], batch_size=128,
        normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
    ).astype(np.float32)
    max_similarity = 0.25
    pairs, encoded_answers = _match_irrelevant_answers(
        selected, answers, prompt_embeddings, sentence_model, seed,
        max_similarity=max_similarity,
    )

    pairs_path = data_dir / "sdd_pairs.jsonl"
    with pairs_path.open("w", encoding="utf-8") as stream:
        for pair in pairs:
            stream.write(json.dumps(pair, ensure_ascii=False) + "\n")
    write_json(data_dir / "sdd_data_manifest.json", {
        "prompt_dataset": "PKU-Alignment/BeaverTails:330k_train",
        "answer_datasets": (["tatsu-lab/alpaca:train", "GAIR/lima:train"]
                            if include_lima else ["tatsu-lab/alpaca:train"]),
        "prompt_fingerprint": beaver._fingerprint,
        "test_fingerprint": beaver_test._fingerprint,
        "alpaca_fingerprint": alpaca._fingerprint,
        "lima_fingerprint": lima._fingerprint if lima is not None else None,
        "paper_equivalent_answer_pool": include_lima,
        "target_pairs": target_pairs,
        "answer_pool_size": len(answers),
        "encoded_answer_count": encoded_answers,
        "eligible_candidates": len(candidates),
        "pairs": len(pairs), "category_counts": dict(counts),
        "candidate_category_counts": candidate_category_counts,
        "category_balance_spread": max(counts.values()) - min(counts.values()),
        "sbert_model": "sentence-transformers/all-MiniLM-L6-v2",
        "cosine_threshold": max_similarity, "max_resample": 50,
        "disjoint_from_shared_attack_and_beavertails_test": True,
    })
    return pairs_path


@app.function(
    image=prep_image, cpu=8, memory=32768, timeout=24 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/data": data_volume, "/hf": hf_volume, "/results": results_volume},
)
def pipeline(seed: int, run_id: str, test_size: int, answer_pool: str):
    sys.path.insert(0, "/root/seam/scripts")
    from modal_compare_common import write_json, ATTACK_LRS, TASKS

    if answer_pool not in {"AlpacaOnly", "AlpacaLima"}:
        raise ValueError(f"Unsupported SDD answer pool: {answer_pool}")
    data_dir, output = _paths(seed, run_id)
    prepared = _validate_preflight(seed, run_id, answer_pool, data_dir, output)
    os.environ["HF_HOME"] = "/hf"
    status = {"method": "SDD", "run_id": run_id, "seed": seed,
              "mode": "full", "answer_pool": answer_pool, "state": "running"}

    def stage(name):
        status["stage"] = name
        write_json(output / "status.json", status)
        results_volume.commit()
        print("SDD_STAGE", name, flush=True)

    try:
        stage("preflight_validated")
        write_json(output / "manifest.json", {
            "method": "SDD", "seed": seed, "run_id": run_id,
            "protocol": "SEAM-style shared attack/eval, not SDD paper Table 1",
            "answer_pool": answer_pool,
            "paper_equivalent_answer_pool": answer_pool == "AlpacaLima",
            "split": prepared["split"], "defense_pairs": 8000, "defense_steps": 500,
            "preflight_run_id": run_id,
            "preflight_file_sha256": prepared["sha256"],
            "defense_global_batch": 24, "defense_lr": 5e-7,
            "attack_steps": 250, "attack_lrs": ATTACK_LRS,
            "eval_tasks": TASKS, "test_size": test_size,
            "gpu": "one H100 at a time",
            "training_provenance": "Reconstructed SFT; official SDD/train missing",
        })
        stage("defense_500_steps")
        train_sdd.remote(seed, run_id)
        results_volume.reload()
        stage("evaluation_sweep")
        evaluate_sdd.remote(seed, run_id, test_size)
        results_volume.reload()
        status.update({"state": "complete", "stage": "complete"})
        write_json(output / "status.json", status)
        results_volume.commit()
        return str(output)
    except Exception as exc:
        status.update({"state": "failed", "error": repr(exc)})
        write_json(output / "status.json", status)
        results_volume.commit()
        raise


@app.function(
    image=work_image, gpu="H100", cpu=8, memory=65536, timeout=24 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/data": data_volume, "/hf": hf_volume, "/results": results_volume},
)
def train_sdd(seed: int, run_id: str):
    data_dir, output = _paths(seed, run_id)
    os.environ.update({"HF_HOME": "/hf", "WANDB_MODE": "disabled", "PYTHONUNBUFFERED": "1"})
    subprocess.run([
        sys.executable, "/root/seam/scripts/sdd_train.py",
        "--pairs", str(data_dir / "sdd_pairs.jsonl"),
        "--output", str(output / "defense"), "--seed", str(seed),
        "--max-steps", "500",
    ], cwd="/root/seam", check=True)
    if not (output / "defense" / "model" / "config.json").exists():
        raise FileNotFoundError("SDD full-model checkpoint missing")
    results_volume.commit()
    hf_volume.commit()


@app.function(
    image=work_image, gpu="H100", cpu=8, memory=65536, timeout=24 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/data": data_volume, "/hf": hf_volume, "/results": results_volume},
)
def evaluate_sdd(seed: int, run_id: str, test_size: int):
    sys.path.insert(0, "/root/seam/scripts")
    from modal_compare_common import evaluate_sweep, write_json

    data_dir, output = _paths(seed, run_id)
    os.environ["HF_HOME"] = "/hf"

    def stage(name):
        write_json(output / "eval_status.json", {"stage": name, "state": "running"})
        results_volume.commit()

    evaluate_sweep("SDD", seed, test_size, output, data_dir / "attack.json",
                   str(output / "defense" / "model"), stage)
    write_json(output / "eval_status.json", {"stage": "complete", "state": "complete"})
    results_volume.commit()
    hf_volume.commit()


@app.local_entrypoint()
def main(seed: int = 42, run_id: str = "", test_size: int = 100,
         answer_pool: str = "AlpacaOnly"):
    if not run_id or test_size < 0:
        raise ValueError("Pass unique --run-id and nonnegative --test-size")
    call = pipeline.spawn(seed, run_id, test_size, answer_pool)
    print(f"Submitted Modal call: {call.object_id}")
    print(f"Results: defense-compare-results/sdd-seed{seed}-{run_id}")


@app.function(
    image=hub_image, cpu=1, memory=2048, timeout=30 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def hf_repo_preflight(repo_id: str):
    sys.path.insert(0, "/root/seam/scripts")
    from hf_checkpoint_push import ensure_private_empty_repo

    return ensure_private_empty_repo(repo_id)


@app.function(
    image=hub_image, cpu=2, memory=4096, timeout=24 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/results": results_volume},
)
def push_sdd_hf(seed: int, run_id: str, repo_id: str):
    sys.path.insert(0, "/root/seam/scripts")
    from hf_checkpoint_push import push_checkpoint

    _, output = _paths(seed, run_id)
    return push_checkpoint(repo_id, output)


@app.function(
    image=prep_image, cpu=0.125, memory=512, timeout=24 * 60 * 60,
    volumes={"/data": data_volume, "/results": results_volume},
)
def train_push_pipeline(seed: int, run_id: str, answer_pool: str, repo_id: str):
    sys.path.insert(0, "/root/seam/scripts")
    from modal_compare_common import write_json

    if answer_pool not in {"AlpacaOnly", "AlpacaLima"}:
        raise ValueError(f"Unsupported SDD answer pool: {answer_pool}")
    data_dir, output = _paths(seed, run_id)
    previous = _restarted_status(output, mode="train_push", seed=seed, run_id=run_id,
                                 answer_pool=answer_pool, hf_repo=repo_id)
    prepared = _validate_preflight(seed, run_id, answer_pool, data_dir, output,
                                   restarted_train=bool(previous))
    if (output / "defense").exists():
        if not previous:
            raise FileExistsError(f"Refusing to overwrite existing SDD defense: {output / 'defense'}")
        print("SDD_TRAIN_PUSH_RESTART retraining from scratch", flush=True)
        shutil.rmtree(output / "defense")
    status = {"method": "SDD", "mode": "train_push", "run_id": run_id,
              "seed": seed, "answer_pool": answer_pool, "hf_repo": repo_id,
              "state": "running"}

    def stage(name):
        status["stage"] = name
        write_json(output / "status.json", status)
        results_volume.commit()
        print("SDD_TRAIN_PUSH_STAGE", name, flush=True)

    try:
        # A restarted attempt already reserved the repo; preflight would now refuse it.
        if not previous or previous.get("stage") in {None, "hf_private_repo_preflight"}:
            stage("hf_private_repo_preflight")
            hf_repo_preflight.remote(repo_id)
        write_json(output / "manifest.json", {
            "method": "SDD", "mode": "train_push", "seed": seed, "run_id": run_id,
            "protocol": "defense training only; no attack/evaluation",
            "answer_pool": answer_pool,
            "paper_equivalent_answer_pool": answer_pool == "AlpacaLima",
            "split": prepared["split"], "defense_pairs": 8000,
            "defense_steps": 500, "defense_global_batch": 24,
            "defense_lr": 5e-7, "preflight_run_id": run_id,
            "preflight_file_sha256": prepared["sha256"],
            "gpu": "one H100", "hf_repo": repo_id,
            "training_provenance": "Reconstructed SFT; official SDD/train missing",
        })
        stage("defense_500_steps")
        train_sdd.remote(seed, run_id)
        results_volume.reload()
        stage("hf_upload")
        hf_url = push_sdd_hf.remote(seed, run_id, repo_id)
        results_volume.reload()
        status.update({"state": "complete", "stage": "complete", "hf_url": hf_url})
        write_json(output / "status.json", status)
        results_volume.commit()
        return hf_url
    except Exception as exc:
        status.update({"state": "failed", "error": repr(exc)})
        write_json(output / "status.json", status)
        results_volume.commit()
        raise


@app.local_entrypoint()
def train_push_main(seed: int = 42, run_id: str = "", answer_pool: str = "AlpacaOnly",
                    hf_repo: str = ""):
    if not run_id or not hf_repo:
        raise ValueError("Pass completed preflight --run-id and private --hf-repo")
    call = train_push_pipeline.spawn(seed, run_id, answer_pool, hf_repo)
    print(f"Submitted Modal call: {call.object_id}")
    print(f"Results: defense-compare-results/sdd-seed{seed}-{run_id}")
    print(f"Private HF repo: https://huggingface.co/{hf_repo}")


@app.function(
    image=prep_image, cpu=0.125, memory=512, timeout=24 * 60 * 60,
    volumes={"/results": results_volume},
)
def retry_sdd_push(seed: int, run_id: str, repo_id: str):
    sys.path.insert(0, "/root/seam/scripts")
    from modal_compare_common import write_json

    _, output = _paths(seed, run_id)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("mode") != "train_push" or manifest.get("seed") != seed
            or manifest.get("run_id") != run_id or manifest.get("hf_repo") != repo_id):
        raise ValueError("SDD train-and-push manifest does not match this upload retry")
    url = push_sdd_hf.remote(seed, run_id, repo_id)
    results_volume.reload()
    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    status.update({"state": "complete", "stage": "complete", "hf_url": url})
    status.pop("error", None)
    write_json(output / "status.json", status)
    results_volume.commit()
    return url


@app.local_entrypoint()
def retry_sdd_push_main(seed: int = 42, run_id: str = "", hf_repo: str = ""):
    if not run_id or not hf_repo:
        raise ValueError("Pass prior train-and-push --run-id and matching --hf-repo")
    call = retry_sdd_push.spawn(seed, run_id, hf_repo)
    print(f"Submitted HF upload retry: {call.object_id}")


@app.function(
    image=prep_image, cpu=8, memory=32768, timeout=24 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/data": data_volume, "/hf": hf_volume, "/results": results_volume},
)
def data_preflight(seed: int, run_id: str, answer_pool: str):
    sys.path.insert(0, "/root/seam/scripts")
    from modal_compare_common import prepare_split, write_json

    if answer_pool not in {"AlpacaOnly", "AlpacaLima"}:
        raise ValueError(f"Unsupported SDD answer pool: {answer_pool}")
    data_dir, output = _paths(seed, run_id)
    if output.exists():
        if not _restarted_status(output, mode="data_preflight", seed=seed,
                                 run_id=run_id, answer_pool=answer_pool):
            raise FileExistsError(f"Refusing to overwrite {output}")
        print("SDD_PREFLIGHT_RESTART rebuilding partial outputs", flush=True)
        shutil.rmtree(output)
        shutil.rmtree(data_dir, ignore_errors=True)
    output.mkdir(parents=True)
    os.environ["HF_HOME"] = "/hf"
    status = {"method": "SDD", "mode": "data_preflight", "seed": seed,
              "run_id": run_id, "answer_pool": answer_pool,
              "stage": "prepare_common_split", "state": "running"}
    write_json(output / "status.json", status)
    results_volume.commit()
    try:
        split = prepare_split(seed, data_dir)
        data_volume.commit()
        status["stage"] = "prepare_sdd_pairs"
        write_json(output / "status.json", status)
        results_volume.commit()
        _build_sdd_pairs(seed, data_dir, data_dir / "attack.json",
                         include_lima=answer_pool == "AlpacaLima")
        data_manifest = json.loads((data_dir / "sdd_data_manifest.json").read_text(encoding="utf-8"))
        status["stage"] = "validate_preflight"
        write_json(output / "status.json", status)
        results_volume.commit()
        names = ("attack.json", "ctrap_defense.json", "sdd_pairs.jsonl", "sdd_data_manifest.json")
        manifest = {
            "seed": seed, "run_id": run_id, "answer_pool": answer_pool,
            "split": split, "pairs": data_manifest["pairs"],
            "sha256": {name: _sha256(data_dir / name) for name in names},
        }
        write_json(data_dir / "preflight_manifest.json", manifest)
        data_volume.commit()
        status.update({"state": "complete", "stage": "complete", "pairs": manifest["pairs"]})
        write_json(output / "status.json", status)
        write_json(output / "data_manifest.json", data_manifest)
        write_json(output / "preflight_manifest.json", manifest)
        results_volume.commit()
        return manifest
    except Exception as exc:
        status.update({"state": "failed", "error": repr(exc)})
        write_json(output / "status.json", status)
        results_volume.commit()
        raise


@app.local_entrypoint()
def preflight_main(seed: int = 42, run_id: str = "",
                   answer_pool: str = "AlpacaOnly"):
    if not run_id:
        raise ValueError("Pass a unique --run-id")
    call = data_preflight.spawn(seed, run_id, answer_pool)
    print(f"Submitted Modal preflight call: {call.object_id}")
    print(f"Status: defense-compare-results/sdd-seed{seed}-{run_id}/status.json")


@app.function(
    image=prep_image, cpu=8, memory=32768, timeout=4 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/data": data_volume, "/hf": hf_volume, "/results": results_volume},
)
def smoke_pipeline(seed: int, run_id: str, full_data: bool = False,
                   smoke_export: bool = False):
    sys.path.insert(0, "/root/seam/scripts")
    from modal_compare_common import prepare_split, write_json

    data_dir, output = _paths(seed, run_id)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    os.environ["HF_HOME"] = "/hf"
    status = {"method": "SDD", "mode": "smoke", "seed": seed, "run_id": run_id,
              "answer_pool": "AlpacaOnly", "full_data": full_data,
              "smoke_export": smoke_export,
              "stage": "prepare_data", "state": "running"}
    write_json(output / "status.json", status)
    results_volume.commit()
    try:
        prepare_split(seed, data_dir)
        _build_sdd_pairs(seed, data_dir, data_dir / "attack.json",
                         include_lima=False, target_pairs=8000 if full_data else 48)
        data_volume.commit()
        status["stage"] = "train_2_steps"
        write_json(output / "status.json", status)
        results_volume.commit()
        metrics = smoke_train_sdd.remote(seed, run_id, smoke_export)
        results_volume.reload()
        status.update({"stage": "complete", "state": "complete", "step_losses": metrics["step_losses"]})
        write_json(output / "status.json", status)
        results_volume.commit()
        return {"result": str(output), "metrics": metrics}
    except Exception as exc:
        status.update({"state": "failed", "error": repr(exc)})
        write_json(output / "status.json", status)
        results_volume.commit()
        raise


@app.function(
    image=work_image, gpu="H100", cpu=8, memory=65536, timeout=60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/data": data_volume, "/hf": hf_volume, "/results": results_volume},
)
def smoke_train_sdd(seed: int, run_id: str, smoke_export: bool = False):
    data_dir, output = _paths(seed, run_id)
    os.environ.update({"HF_HOME": "/hf", "WANDB_MODE": "disabled", "PYTHONUNBUFFERED": "1"})
    train_output = Path("/tmp/sdd-smoke-export") if smoke_export else output / "defense"
    command = [
        sys.executable, "/root/seam/scripts/sdd_train.py",
        "--pairs", str(data_dir / "sdd_pairs.jsonl"),
        "--output", str(train_output), "--seed", str(seed),
        "--max-steps", "2", "--smoke",
    ]
    if smoke_export:
        command.append("--smoke-export")
    subprocess.run(command, cwd="/root/seam", check=True)
    metrics = json.loads((train_output / "train_metrics.json").read_text(encoding="utf-8"))
    results_volume.commit()
    hf_volume.commit()
    return metrics


@app.local_entrypoint()
def smoke_main(seed: int = 42, run_id: str = "", full_data: bool = False,
               smoke_export: bool = False):
    if not run_id:
        raise ValueError("Pass a unique --run-id")
    result = smoke_pipeline.remote(seed, run_id, full_data, smoke_export)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
