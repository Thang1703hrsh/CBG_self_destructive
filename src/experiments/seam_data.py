"""Legacy SEAM data pools and the paper's prompt-disjoint preparation policy.

The default ``split_policy="seam"`` preserves the historical row recipe below.
``split_policy="paper"`` excludes overlapping prompts, filters test categories
on the corresponding rows, and reserves benign fine-tuning data separately.

SEAM main run (scripts/modal_seam_full.py in the SEAM project, Run-SEAMFull.ps1, seed 42):
  1. keep RepNoise ``beavertails_with_refusals_train.json`` rows with ``is_safe is False``
     and non-empty ``prompt``, ``response`` and ``refusal``;
  2. ``random.Random(seed).shuffle(records)``;
  3. defense = first 4,000 rows, attack = next 1,000 rows (no prompt de-duplication);
  4. benign = ``yahma/alpaca-cleaned`` train shuffled with ``benign_seed = seed``, first 4,000 rows;
  5. harmfulness test = BeaverTails ``30k_test`` through SEAM's filter, first ``test_size=100`` prompts.

In legacy mode these SEAM pools are reproduced row for row. CBG needs a
few pools that SEAM does not have (D_geo, D_val_atk, a 10K attack pool); they
are drawn after row 5,000 of the same shuffled list (then BeaverTails
330k_train), prompt-disjoint from every SEAM pool and from the test split.
"""
from __future__ import annotations

from pathlib import Path
import random
import statistics

import numpy as np

from src.experiments.protocol import digest, file_digest, prompt_key, write_json

# seam/src/data_processing/_datasets.py — order matters for the test filter.
BT_CATEGORIES = {
    "half": ["violence,aiding_and_abetting,incitement", "hate_speech,offensive_language",
             "discrimination,stereotype,injustice", "misinformation_regarding_ethics,laws_and_safety",
             "privacy_violation"],
    "other_half": ["financial_crime,property_crime,theft", "drug_abuse,weapons,banned_substance",
                   "terrorism,organized_crime", "sexually_explicit,adult_content"],
    "violence": ["violence,aiding_and_abetting,incitement"],
    "toxic": ["hate_speech,offensive_language", "discrimination,stereotype,injustice"],
    "privacy_misinformation": ["misinformation_regarding_ethics,laws_and_safety", "privacy_violation"],
    "crime_unethical": ["financial_crime,property_crime,theft", "drug_abuse,weapons,banned_substance",
                        "non_violent_unethical_behavior", "terrorism,organized_crime"],
    "sexually_explicit": ["sexually_explicit,adult_content"],
    "self_harm": ["self_harm"],
}


def seam_valid_rows(rows):
    """Step 1: unsafe rows with non-empty prompt, response and refusal (original order)."""
    return [row for row in rows
            if row.get("is_safe") is False and row.get("prompt") and row.get("response") and row.get("refusal")]


def seam_test_prompts(test_rows, test_size=100):
    """SEAM's _test_dataset_tokenizer, reproduced literally (batched over the whole split).

    The first *true* category of each row is collected only for rows that have
    one, then zipped with all rows; safe rows (no true category) therefore shift
    the pairing and truncate the zip. That behavior is kept on purpose.
    """
    processed = []
    for category in (row["category"] for row in test_rows):
        for key, value in category.items():
            if value:
                processed.append(key)
                break
    harmful_categories = [c for values in BT_CATEGORIES.values() for c in values]
    prompts = []
    for row, category in zip(test_rows, processed):
        if row["is_safe"] or category not in harmful_categories:
            continue
        prompts.append(row["prompt"])
    return prompts[:test_size]


def alpaca_order(count, seed, method):
    """Permutation used for SEAM's benign shuffle.

    ``hf_dataset`` reproduces ``datasets.Dataset.shuffle(seed=seed)`` (NumPy
    default_rng permutation); ``python_random`` is ``random.Random(seed).shuffle``.
    """
    if method == "hf_dataset":
        return np.random.default_rng(seed).permutation(count).tolist()
    if method == "python_random":
        order = list(range(count))
        random.Random(seed).shuffle(order)
        return order
    raise ValueError("benign_shuffle must be hf_dataset or python_random")


def _record(row, source, index):
    return dict(prompt=row["prompt"], response=row["response"], refusal=row.get("refusal", ""),
                category=row.get("category"), is_safe=row.get("is_safe"), source=source, source_index=index)


def build_pools(repnoise, beavertails_extra, test_rows, alpaca, sizes, *, seed=42, benign_shuffle="hf_dataset",
                test_size=100, split_policy="seam", eval_excluded_prompts=()):
    """Build legacy row splits or the paper's prompt-disjoint splits.

    ``eval_excluded_prompts`` supplies any additional baseline-training prompts
    that must be excluded from the common harmfulness test set.
    """
    if split_policy not in {"seam", "paper"}:
        raise ValueError("split_policy must be seam or paper")
    valid = seam_valid_rows(repnoise)
    index_of = {id(row): i for i, row in enumerate(repnoise)}
    shuffled = list(valid)
    random.Random(seed).shuffle(shuffled)
    need = sizes["defense"] + sizes["attack"]
    if len(shuffled) < need:
        raise ValueError(f"SEAM requires at least {need} valid rows, found {len(shuffled)}")
    rows = [_record(row, "repnoise", index_of[id(row)]) for row in shuffled]
    pools = dict(defense=rows[:sizes["defense"]], attack=rows[sizes["defense"]:need])

    if split_policy == "paper":
        defense_prompts = {prompt_key(r["prompt"]) for r in pools["defense"]}
        candidates = [r for r in rows[sizes["defense"]:] if prompt_key(r["prompt"]) not in defense_prompts]
        pools["attack"] = candidates[:sizes["attack"]]
        if len(pools["attack"]) != sizes["attack"]:
            raise ValueError("not enough prompt-disjoint harmful attack rows")
        selected = {r["source_index"] for r in pools["defense"] + pools["attack"]}
        rest = [r for r in rows if r["source_index"] not in selected]
    else:
        rest = rows[need:]

    # CBG-only pools: after row 5,000 of the same shuffled list, prompt-disjoint.
    excluded = {prompt_key(r["prompt"]) for r in test_rows}
    used = {prompt_key(r["prompt"]) for r in pools["defense"] + pools["attack"]}
    extra = [_record(r, "beavertails_330k_train", i) for i, r in enumerate(beavertails_extra)
             if r.get("is_safe") is False and r.get("prompt") and r.get("response")]

    def take(size, candidates, need_refusal, repeat_prompts=False):
        """Rows whose prompt is in no earlier pool; repeat_prompts allows repeats within this pool."""
        result, before, seen_rows = [], set(used), set()
        for row in candidates:
            if len(result) == size:
                break
            key = prompt_key(row["prompt"])
            blocked = key in before if repeat_prompts else key in used
            identity = (row["source"], row["source_index"])
            if blocked or key in excluded or identity in seen_rows or (need_refusal and not row["refusal"]):
                continue
            used.add(key)
            seen_rows.add(identity)
            result.append(row)
        if len(result) != size:
            raise ValueError(f"only {len(result)} eligible rows for a pool of {size}")
        return result

    pools["d_geo_safety"] = take(sizes["d_geo_safety"], rest, True)
    pools["d_val_atk"] = take(sizes["d_val_atk"], rest + extra, False)
    # Like SEAM's own pools, the attack pool may contain several responses per prompt.
    pools["d_atk_10k"] = pools["attack"] + take(sizes["d_atk_10k"] - sizes["attack"], rest + extra, False,
                                                repeat_prompts=True)

    order = alpaca_order(len(alpaca), seed, benign_shuffle)
    benign = [dict(instruction=alpaca[i]["instruction"], input=alpaca[i]["input"], output=alpaca[i]["output"],
                   source_index=i) for i in order]
    if len(benign) < sizes["benign"] + sizes["d_geo_benign"]:
        raise ValueError("not enough benign training and geometry rows")
    pools["benign"] = benign[:sizes["benign"]]
    pools["d_geo_benign"] = benign[sizes["benign"]:sizes["benign"] + sizes["d_geo_benign"]]
    if split_policy == "paper":
        # Exclude prompts rather than only row IDs; duplicate instructions are
        # common in instruction datasets. Reserve benign FT before D_geo.
        benign_used = {_benign_key(r) for r in pools["benign"]}
        candidates = [r for r in benign[sizes["benign"]:] if _benign_key(r) not in benign_used]
        cursor = 0
        for name, size in (("benign_attack", sizes.get("benign_attack", 1000)),
                           ("d_geo_benign", sizes["d_geo_benign"])):
            chosen = []
            while cursor < len(candidates) and len(chosen) < size:
                row = candidates[cursor]
                cursor += 1
                key = _benign_key(row)
                if key not in benign_used:
                    chosen.append(row)
                    benign_used.add(key)
            if len(chosen) != size:
                raise ValueError(f"not enough prompt-disjoint rows for {name}")
            pools[name] = chosen
        blocked = used | {prompt_key(p) for p in eval_excluded_prompts}
        prompts = []
        categories = {c for values in BT_CATEGORIES.values() for c in values}
        for row in test_rows:
            prompt = row.get("prompt", "")
            key = prompt_key(prompt)
            if (not prompt or row.get("is_safe") is not False or key in blocked
                or not any(value and category in categories for category, value in row.get("category", {}).items())):
                continue
            prompts.append(dict(prompt=prompt))
            blocked.add(key)
            if len(prompts) == test_size:
                break
        if len(prompts) != test_size:
            raise ValueError("not enough prompt-disjoint harmfulness test prompts")
        pools["eval_harm"] = prompts
    else:
        pools["eval_harm"] = [dict(prompt=p) for p in seam_test_prompts(test_rows, test_size)]
    return pools


def _benign_key(row):
    return prompt_key(f"{row['instruction']}\n{row.get('input', '')}")


def write_pools(pools, directory, sources):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, rows in pools.items():
        path = directory / f"{name}.json"
        write_json(path, rows)
        files[name] = dict(count=len(rows), sha256=file_digest(path))
    keys = {name: {prompt_key(r["prompt"]) for r in rows} for name, rows in pools.items() if rows and "prompt" in rows[0]}
    # SEAM's own pools are reproduced as-is: report their overlaps, never "fix" them.
    report = dict(defense_unique_prompts=len(keys["defense"]), attack_unique_prompts=len(keys["attack"]),
                  defense_attack_shared_prompts=len(keys["defense"] & keys["attack"]),
                  defense_eval_shared_prompts=len(keys["defense"] & keys["eval_harm"]))
    policy = sources.get("seam_split", {}).get("split_policy", "seam")
    if policy == "paper":
        for left, right in (("defense", "attack"), ("defense", "eval_harm"), ("attack", "eval_harm")):
            if keys[left] & keys[right]:
                raise ValueError(f"paper split leaks prompts between {left} and {right}")
        for left, right in (("benign", "benign_attack"), ("benign", "d_geo_benign"),
                            ("benign_attack", "d_geo_benign")):
            if {_benign_key(r) for r in pools[left]} & {_benign_key(r) for r in pools[right]}:
                raise ValueError(f"paper split leaks prompts between {left} and {right}")
    cbg_only = {"d_geo_safety": ("defense", "attack", "eval_harm", "d_val_atk", "d_atk_10k"),
                "d_val_atk": ("defense", "attack", "eval_harm", "d_atk_10k")}
    for name, others in cbg_only.items():
        for other in others:
            other_keys = keys[other] if not (name == "d_val_atk" and other == "d_atk_10k") else \
                {prompt_key(r["prompt"]) for r in pools["d_atk_10k"][len(pools["attack"]):]}
            if keys[name] & other_keys:
                raise ValueError(f"CBG-only pool {name} leaks prompts into {other}")
    benign_keys = lambda rows: {r["source_index"] for r in rows}
    if benign_keys(pools["benign"]) & benign_keys(pools["d_geo_benign"]):
        raise ValueError("D_geo benign overlaps the benign training rows")
    manifest = dict(schema="cbg_paper_1" if policy == "paper" else "cbg_v2_seam_main_run_1",
                    sources=sources, files=files, overlap_report=report,
                    seam_pools=["defense", "attack", "benign", "eval_harm"],
                    cbg_only_pools=["d_geo_safety", "d_val_atk", "d_atk_10k", "d_geo_benign"],
                    d_atk_10k="attack.json (1,000 rows, SEAM) followed by 9,000 CBG-only rows")
    manifest["sha256"] = digest(manifest)
    write_json(directory / "manifest.json", manifest)
    return manifest


def truncation_report(tokenizer, pools, *, max_length):
    """SEAM truncation (whole string <= max_length, right-truncated) on the training pools."""
    from src.data_processing.cbg_datasets import (_prompt_token_ids, seam_benign_texts, seam_safety_texts)
    def summarize(rows):
        full = [r[0] for r in rows]
        kept = [r[1] for r in rows]
        return dict(examples=len(rows), truncated_fraction=sum(f > max_length for f in full) / len(rows),
                    no_response_after_truncation_last_token_used=sum(k <= 0 for k in kept),
                    prefix_mismatch=sum(r[2] for r in rows),
                    median_full_tokens=statistics.median(full), median_response_kept=statistics.median(kept))
    def measure(prompt, text):
        full = _prompt_token_ids(tokenizer, text, add_bos=True)
        prefix = _prompt_token_ids(tokenizer, prompt, add_bos=True)
        return len(full), min(len(full), max_length) - len(prefix), full[:len(prefix)] != prefix
    harmful, safe, benign = [], [], []
    for row in pools["defense"]:
        prompt, h, s = seam_safety_texts(row)
        harmful.append(measure(prompt, h))
        safe.append(measure(prompt, s))
    for row in pools["benign"]:
        benign.append(measure(*seam_benign_texts(row)))
    return dict(max_length=max_length, harmful=summarize(harmful), refusal=summarize(safe), benign=summarize(benign))
