#!/usr/bin/env python3

"""
Dataset Construction
- Load harmful instructions from AdvBench
- Load high-quality answers from Alpaca
- Pair <harmful_instruction, random_answer>
- Filter by irrelevance via SentenceBERT cosine similarity (resample if too similar)
- Optionally balance by harmful categories if the dataset provides category labels
"""

import argparse
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer


# -----------------------------
# Helpers: robust column access
# -----------------------------
def _first_existing_key(d: Dict, keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in d:
            return k
    return None


def _pick_column_name(columns: List[str], candidates: List[str]) -> Optional[str]:
    s = set(columns)
    for c in candidates:
        if c in s:
            return c
    return None


def _safe_strip(x: Optional[str]) -> str:
    if x is None:
        return ""
    return str(x).strip()


# -----------------------------
# Load Alpaca answers
# -----------------------------
def load_alpaca_answers(
    alpaca_id: str = "tatsu-lab/alpaca",
    split: str = "train",
    min_chars: int = 1,
) -> List[str]:
    ds = load_dataset(alpaca_id, split=split)
    # Alpaca has columns: instruction, input, output, text (viewer shows these)  :contentReference[oaicite:2]{index=2}
    if "output" not in ds.column_names:
        raise ValueError(f"[Alpaca] expected column 'output' but got {ds.column_names}")
    answers = []
    for ex in ds:
        out = _safe_strip(ex.get("output"))
        if len(out) >= min_chars:
            answers.append(out)
    if not answers:
        raise RuntimeError("[Alpaca] got empty answer pool after filtering.")
    return answers


# -----------------------------
# Load AdvBench harmful instructions (+ optional categories)
# -----------------------------
def load_advbench(
    preferred_ids: List[str],
    split: str = "train",
) -> Tuple[Dataset, str]:
    """
    Try dataset ids in order, return (dataset, used_id).
    Note: some AdvBench variants on HF are gated and may fail without prior acceptance. :contentReference[oaicite:3]{index=3}
    """
    last_err = None
    for did in preferred_ids:
        try:
            ds = load_dataset(did, split=split)
            return ds, did
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to load any AdvBench dataset from {preferred_ids}. Last error: {last_err}")


def extract_harmful_instructions_and_categories(
    ds: Dataset,
) -> Tuple[List[str], Optional[List[str]]]:
    """
    Returns:
      instructions: List[str]
      categories: Optional[List[str]] aligned to instructions (same length), if available
    """
    cols = ds.column_names

    # Common candidates seen across variants:
    # - AlignmentResearch/AdvBench has 'instructions' (often a list/sequence) :contentReference[oaicite:4]{index=4}
    instr_col = _pick_column_name(cols, ["instruction", "instructions", "prompt", "query", "goal"])
    if instr_col is None:
        raise ValueError(f"[AdvBench] cannot find instruction column in {cols}")

    # Optional category candidates (not always present)
    cat_col = _pick_column_name(cols, ["category", "harmful_category", "behavior_category", "topic", "type"])

    instructions: List[str] = []
    categories: Optional[List[str]] = [] if cat_col is not None else None

    for ex in ds:
        raw = ex.get(instr_col)

        # If it's a list/sequence (as in AlignmentResearch/AdvBench viewer), take first element :contentReference[oaicite:5]{index=5}
        if isinstance(raw, (list, tuple)):
            if len(raw) == 0:
                continue
            inst = _safe_strip(raw[0])
        else:
            inst = _safe_strip(raw)

        if not inst:
            continue

        instructions.append(inst)

        if cat_col is not None and categories is not None:
            categories.append(_safe_strip(ex.get(cat_col)) or "unknown")

    if not instructions:
        raise RuntimeError("[AdvBench] extracted 0 instructions (after filtering empty).")

    if categories is not None and len(categories) != len(instructions):
        raise RuntimeError("[AdvBench] categories length mismatch with instructions.")

    return instructions, categories


# -----------------------------
# Balanced sampling by category
# -----------------------------
def balanced_sample_indices(
    categories: List[str],
    total: int,
    seed: int,
) -> List[int]:
    rng = random.Random(seed)
    cat2idx: Dict[str, List[int]] = {}
    for i, c in enumerate(categories):
        cat2idx.setdefault(c, []).append(i)

    cats = sorted(cat2idx.keys())
    if len(cats) == 0:
        return []

    per = total // len(cats)
    remainder = total - per * len(cats)

    chosen: List[int] = []

    # sample per category
    for c in cats:
        pool = cat2idx[c][:]
        rng.shuffle(pool)
        take = min(per, len(pool))
        chosen.extend(pool[:take])

    # distribute remainder from categories with leftover
    if remainder > 0:
        leftovers = []
        chosen_set = set(chosen)
        for c in cats:
            for i in cat2idx[c]:
                if i not in chosen_set:
                    leftovers.append(i)
        rng.shuffle(leftovers)
        chosen.extend(leftovers[:remainder])

    rng.shuffle(chosen)
    return chosen[:total]


def random_sample_indices(n: int, total: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    return idx[: min(total, n)]


# -----------------------------
# Irrelevance selection via SentenceBERT cosine similarity
# -----------------------------
def build_sdd_pairs(
    harmful_instructions: List[str],
    benign_answers: List[str],
    model_name: str,
    threshold: float,
    max_resample: int,
    seed: int,
    batch_size: int,
) -> Dict[str, List]:
    """
    Returns dict of columns: instruction, output, text, cosine_sim, resample_tries
    """
    # Reproducibility
    rng = np.random.default_rng(seed)

    model = SentenceTransformer(model_name)

    # Precompute benign answer embeddings (normalize => cosine = dot)
    benign_emb = model.encode(
        benign_answers,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    out_instruction = []
    out_output = []
    out_text = []
    out_sim = []
    out_tries = []

    for inst in harmful_instructions:
        inst_emb = model.encode(
            [inst],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0].astype(np.float32)

        best = None  # (sim, answer_idx, tries)
        tries = 0

        while True:
            idx = int(rng.integers(0, len(benign_answers)))
            sim = float(np.dot(inst_emb, benign_emb[idx]))
            tries += 1

            # Keep the least similar among attempts as a fallback
            if best is None or sim < best[0]:
                best = (sim, idx, tries)

            if sim <= threshold:
                chosen_sim, chosen_idx, chosen_tries = sim, idx, tries
                break

            if tries >= max_resample:
                chosen_sim, chosen_idx, chosen_tries = best
                break

        answer = benign_answers[chosen_idx]

        out_instruction.append(inst)
        out_output.append(answer)
        out_text.append(f"### Instruction:\n{inst}\n\n### Response:\n{answer}")
        out_sim.append(float(chosen_sim))
        out_tries.append(int(chosen_tries))

    return {
        "prompt": out_instruction,
        "completion": out_output,
        "text": out_text,
        "cosine_sim": out_sim,
        "resample_tries": out_tries,
    }


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpaca_id", type=str, default="tatsu-lab/alpaca")
    ap.add_argument(
        "--advbench_ids",
        type=str,
        nargs="+",
        default=["walledai/AdvBench", "AlignmentResearch/AdvBench"],
        help="Try these dataset IDs in order. Some may be gated on HF.",
    )
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--total_size", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument(
        "--sbert_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceBERT model on HF/ sentence-transformers.",
    )
    ap.add_argument("--threshold", type=float, default=0.25, help="cosine similarity threshold; higher => stricter filtering")
    ap.add_argument("--max_resample", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)

    ap.add_argument("--min_answer_chars", type=int, default=1)
    ap.add_argument("--save_dir", type=str, default="sdd_out")
    ap.add_argument("--save_name", type=str, default="sdd_advbench_alpaca")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # 1) Load benign answers
    benign_answers = load_alpaca_answers(
        alpaca_id=args.alpaca_id,
        split=args.split,
        min_chars=args.min_answer_chars,
    )

    # 2) Load harmful dataset
    adv_ds, used_adv_id = load_advbench(args.advbench_ids, split=args.split)
    harmful_instructions_all, categories = extract_harmful_instructions_and_categories(adv_ds)

    # 3) Sample to total_size (balanced if categories available)
    if categories is not None:
        sample_idx = balanced_sample_indices(categories, args.total_size, args.seed)
    else:
        sample_idx = random_sample_indices(len(harmful_instructions_all), args.total_size, args.seed)

    harmful_instructions = [harmful_instructions_all[i] for i in sample_idx]
    sampled_categories = [categories[i] for i in sample_idx] if categories is not None else None

    # 4) Pair + irrelevance selection
    cols = build_sdd_pairs(
        harmful_instructions=harmful_instructions,
        benign_answers=benign_answers,
        model_name=args.sbert_model,
        threshold=args.threshold,
        max_resample=args.max_resample,
        seed=args.seed,
        batch_size=args.batch_size,
    )

    if sampled_categories is not None:
        cols["category"] = sampled_categories

    out_ds = Dataset.from_dict(cols)

    # 5) Save
    out_path = os.path.join(args.save_dir, args.save_name)
    out_ds.save_to_disk(out_path)

    # Also save JSONL for convenience
    jsonl_path = os.path.join(args.save_dir, f"{args.save_name}.jsonl")
    out_ds.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)

    # Minimal summary (avoid printing content)
    print(f"[OK] AdvBench used: {used_adv_id}")
    print(f"[OK] Output rows: {len(out_ds)}")
    print(f"[OK] Saved to: {out_path}")
    print(f"[OK] JSONL: {jsonl_path}")
    print(f"[Info] cosine_sim stats: min={np.min(cols['cosine_sim']):.4f}, mean={np.mean(cols['cosine_sim']):.4f}, max={np.max(cols['cosine_sim']):.4f}")


if __name__ == "__main__":
    main()

