"""Model-independent defense/attack data for one seed (same code as the Llama-2 runs).

Writes <out>/seed<S>/:
  attack.json          shared attack split (RepNoise rows 4000:5000 after random.Random(seed).shuffle)
  ctrap_defense.json   CTRAP defense rows 0:4000 + 5000:6000
  seam_defense.json    SEAM defense rows 0:4000 (= ctrap_defense.json[:4000])
  sdd_pairs.jsonl      8,000 SDD pairs (AlpacaOnly answer pool), cosine <= 0.25
  manifest.json        sha256 of every file, and whether it matches the Llama-2 runs (data_hashes.json)

  python baselines/prepare_data.py --seed 42 --out outputs/baselines/data
"""
import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import requests

PUBLIC_DATA_URL = (
    "https://raw.githubusercontent.com/domenicrosati/representation-noising/"
    "main/data/beavertails_with_refusals_train.json"
)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def prepare_split(seed: int, data_dir: Path):
    """Match SEAM's 4k/1k split; reserve a further 1k for CTRAP's 5k."""
    response = requests.get(PUBLIC_DATA_URL, timeout=120)
    response.raise_for_status()
    source_sha256 = hashlib.sha256(response.content).hexdigest()
    rows = [
        row for row in response.json()
        if row.get("is_safe") is False and row.get("prompt")
        and row.get("response") and row.get("refusal")
    ]
    random.Random(seed).shuffle(rows)
    if len(rows) < 6000:
        raise ValueError(f"Need at least 6,000 eligible RepNoise rows, got {len(rows)}")
    data_dir.mkdir(parents=True, exist_ok=False)
    attack = rows[4000:5000]
    defense = rows[:4000] + rows[5000:6000]
    write_json(data_dir / "attack.json", attack)
    write_json(data_dir / "ctrap_defense.json", defense)
    return {
        "source": PUBLIC_DATA_URL,
        "source_sha256": source_sha256,
        "refusal_provenance": "RepNoise public data; GPT-4o not verified",
        "eligible_rows": len(rows),
        "seam_defense_indices": "0:4000",
        "shared_attack_indices": "4000:5000",
        "ctrap_extra_defense_indices": "5000:6000",
        "attack_count": len(attack),
        "ctrap_defense_count": len(defense),
    }


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

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data_dir = args.out / f"seed{args.seed}"
    if (data_dir / "manifest.json").is_file():
        print(f"already prepared: {data_dir}")
        return
    if data_dir.exists():   # an interrupted attempt: rebuild from scratch
        for path in data_dir.iterdir():
            path.unlink()
        data_dir.rmdir()
    split = prepare_split(args.seed, data_dir)
    defense = json.loads((data_dir / "ctrap_defense.json").read_text(encoding="utf-8"))
    write_json(data_dir / "seam_defense.json", defense[:4000])
    _build_sdd_pairs(args.seed, data_dir, data_dir / "attack.json", include_lima=False)
    names = ("attack.json", "ctrap_defense.json", "seam_defense.json", "sdd_pairs.jsonl")
    hashes = {name: sha256(data_dir / name) for name in names}
    reference = json.loads((Path(__file__).with_name("data_hashes.json")).read_text(encoding="utf-8"))
    expected = reference.get(str(args.seed), {})
    same = {name: (expected.get(name) == digest if name in expected else None) for name, digest in hashes.items()}
    for name, ok in same.items():
        if ok is False:
            print(f"WARNING: {name} differs from the Llama-2 runs of seed {args.seed}", file=sys.stderr)
    write_json(data_dir / "manifest.json", {"seed": args.seed, "split": split, "sha256": hashes,
                                            "matches_llama2_runs": same})
    print(f"prepared {data_dir}: {same}")


if __name__ == "__main__":
    main()
