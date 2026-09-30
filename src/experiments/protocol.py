"""Hashed, prompt-disjoint datasets and validation-only model selection.

No downloads, model training, API calls or implicit experiment choices occur here.
Metrics use percentage points (0--100), never a mixture of fractions and percents.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import unicodedata


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    """Atomic replace; callers decide whether replacing an artifact is allowed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n",
                         encoding="utf-8")
    temporary.replace(path)


def write_once(path, value):
    if Path(path).exists():
        if read_json(path) != value:
            raise ValueError(f"refusing to replace a different frozen artifact: {path}")
        return
    write_json(path, value)


def prompt_key(text):
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def number(value, name, *, minimum=0.):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be explicitly configured and finite, >= {minimum}")
    return value


def records(path):
    result = read_json(path)
    if not isinstance(result, list) or not result or not all(isinstance(r, dict) for r in result):
        raise ValueError(f"{path}: expected a nonempty JSON array of records")
    return result


def prepare_splits(safety_path, benign_path, evaluation_path, *, seed, defense_size=4000,
                   validation_size=1000, attack_size=10000, benign_size=4000,
                   refusal_model, category_order=None, max_rows_per_prompt=1):
    """Prompt-disjoint pools with a declared cap on responses per prompt.

    S7 uses an explicit 14-category order. Multi-label rows that cross the two
    halves are excluded, rather than leaking attack categories into defense.
    Test 1K is a prefix of the test 10K pool, intentionally nested.
    """
    for name, value in locals().copy().items():
        if name.endswith("_size"):
            positive_int(value, name)
    positive_int(max_rows_per_prompt, "max_rows_per_prompt")
    if not isinstance(refusal_model, str) or not refusal_model.strip():
        raise ValueError("refusal_model provenance must be supplied")
    safety, benign, evaluation = map(records, (safety_path, benign_path, evaluation_path))
    eval_keys = {prompt_key(r["prompt"]) for r in evaluation}
    if "" in eval_keys:
        raise ValueError("empty evaluation prompt")
    if category_order is not None and (len(category_order) != 14 or len(set(category_order)) != 14):
        raise ValueError("S7 requires an explicit ordered list of 14 distinct categories")
    groups, categories = {}, {}
    for i, row in enumerate(safety):
        key = prompt_key(row.get("prompt", ""))
        if (row.get("is_safe") is not False or not key or key in eval_keys
                or not str(row.get("response", "")).strip()):
            continue
        groups.setdefault(key, []).append(i)
        categories.setdefault(key, set()).update(k for k, v in row.get("category", {}).items() if v)
    rng = random.Random(seed)
    keys = list(groups)
    rng.shuffle(keys)

    def take(pool, size, *, require_refusal=False):
        chosen, used = [], set()
        for key in pool:
            candidates = [i for i in groups[key]
                          if not require_refusal or str(safety[i].get("refusal", "")).strip()]
            candidates = candidates[:max_rows_per_prompt]
            if not candidates:
                continue
            chosen.extend(candidates[:size-len(chosen)])
            used.add(key)
            if len(chosen) == size:
                break
        return chosen, used

    if category_order is None:
        defense, defense_keys = take(keys, defense_size, require_refusal=True)
        remaining = [key for key in keys if key not in defense_keys]
    else:
        first, second = set(category_order[:7]), set(category_order[7:])
        defense_pool = [key for key in keys if categories[key] and categories[key] <= first]
        defense, _ = take(defense_pool, defense_size, require_refusal=True)
        remaining = [key for key in keys if categories[key] and categories[key] <= second]
    validation, validation_keys = take(remaining, validation_size)
    attack, _ = take([key for key in remaining if key not in validation_keys], attack_size)
    for name, indices, size in (("defense", defense, defense_size), ("validation", validation, validation_size),
                                ("attack", attack, attack_size)):
        if len(indices) != size:
            raise ValueError(f"not enough prompt-disjoint eligible records for {name}: {len(indices)} < {size}")
    benign_indices = [i for i,r in enumerate(benign) if str(r.get("output", "")).strip()]
    rng.shuffle(benign_indices)
    if len(benign_indices) < benign_size:
        raise ValueError("not enough benign records")
    result = dict(schema_version=1, seed=seed, refusal_model=refusal_model, category_order=category_order,
                  max_rows_per_prompt=max_rows_per_prompt,
                  sources={name: dict(path=str(Path(path).resolve()), sha256=file_digest(path), count=len(rows))
                           for name, path, rows in (("safety", safety_path, safety), ("benign", benign_path, benign),
                                                    ("evaluation", evaluation_path, evaluation))},
                  defense_indices=defense, validation_indices=validation, attack_indices=attack,
                  benign_indices=benign_indices[:benign_size],
                  evaluation_indices=list(range(len(evaluation))),
                  prompt_deduplication=("NFKC+casefold+whitespace; prompt-disjoint pools and evaluation; "
                                        f"at most {max_rows_per_prompt} rows per prompt in each pool"))
    result["sha256"] = digest(result)
    return result


def validate_manifest(manifest):
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported scenario manifest version")
    if manifest.get("sha256") != digest({k:v for k,v in manifest.items() if k != "sha256"}):
        raise ValueError("scenario manifest checksum mismatch")
    loaded = {}
    for name, source in manifest["sources"].items():
        if file_digest(source["path"]) != source["sha256"]:
            raise ValueError(f"{name} data changed since splits were frozen")
        loaded[name] = records(source["path"])
        if len(loaded[name]) != source["count"]:
            raise ValueError(f"{name} source count mismatch")
    keys = []
    for group in ("defense", "validation", "attack", "benign", "evaluation"):
        source = group if group in {"benign", "evaluation"} else "safety"
        indices = manifest[group+"_indices"]
        if not indices or len(set(indices)) != len(indices) or any(
            isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(loaded[source]) for i in indices):
            raise ValueError(f"invalid {group} indices")
        if source == "safety":
            selected_keys = [prompt_key(loaded[source][i]["prompt"]) for i in indices]
            maximum = positive_int(manifest.get("max_rows_per_prompt", 1), "max_rows_per_prompt")
            counts = {}
            for key in selected_keys:
                counts[key] = counts.get(key, 0) + 1
            if not all(key and count <= maximum for key, count in counts.items()):
                raise ValueError(f"{group} exceeds its per-prompt row cap")
            group_keys = set(selected_keys)
            keys.append(group_keys)
    if any(keys[i] & keys[j] for i in range(3) for j in range(i)):
        raise ValueError("defense/validation/test prompt leakage")
    evaluation_keys = {prompt_key(r["prompt"]) for r in loaded["evaluation"]}
    if set.union(*keys) & evaluation_keys:
        raise ValueError("evaluation prompt leakage")
    if manifest["evaluation_indices"] != list(range(len(loaded["evaluation"]))):
        raise ValueError("harmfulness evaluation must use the full supplied test split")
    return loaded


def select_records(manifest, pool, count=None):
    loaded = validate_manifest(manifest)
    if pool not in {"defense", "validation", "attack", "benign", "evaluation"}:
        raise ValueError("unknown scenario pool")
    indices = manifest[pool+"_indices"]
    if count is not None:
        positive_int(count, "count")
        if count > len(indices):
            raise ValueError(f"{pool} pool smaller than requested count")
        indices = indices[:count]
    source = pool if pool in {"benign", "evaluation"} else "safety"
    return [loaded[source][i] for i in indices], indices


def percentage(value, name):
    number(value, name)
    if value > 100:
        raise ValueError(f"{name} must use percentage points in [0,100]")
    return value


def select_configuration(candidates, observations, *, manifest_sha256, utility_subset_sha256,
                         max_zs_drop_pp, attack_seed, model=None):
    """Fixed rule: mean HS on #2/#4, tie by largest mean ZS drop, then ID.

    Requires the complete declared grid. No test rows, missing failures, duplicate
    observations, or silent exclusion of unsuccessful candidates are permitted.
    Failed runs must be explicitly recorded with status='failed' and a reason.
    """
    number(max_zs_drop_pp, "max_zs_drop_pp")
    if not utility_subset_sha256:
        raise ValueError("held-out MMLU subset hash required")
    expected = {(key, attack) for key in candidates for attack in ("2", "4")}
    grouped = {key: [] for key in candidates}
    seen, base_scores = set(), set()
    for row in observations:
        pair = (row["candidate_id"], str(row["attack_id"]))
        if pair not in expected or pair in seen:
            raise ValueError("unknown or duplicate validation observation")
        seen.add(pair)
        if (row.get("pool") != "validation" or row.get("manifest_sha256") != manifest_sha256
            or row.get("utility_subset_sha256") != utility_subset_sha256 or row.get("attack_seed") != attack_seed
            or row.get("candidate_sha256") != digest(candidates[pair[0]])):
            raise ValueError("selection requires the declared validation protocol and candidate hash")
        if model is not None and row.get("model")!=model:
            raise ValueError("validation observations must belong to the selected model")
        if row.get("status") == "failed":
            if not row.get("reason"):
                raise ValueError("failed validation runs need an explicit reason")
        elif row.get("status") == "complete":
            for metric in ("pre_zs", "base_zs", "post_zs", "hs"):
                percentage(row[metric], metric)
            base_scores.add(row["base_zs"])
        else:
            raise ValueError("validation status must be complete or failed")
        grouped[pair[0]].append(row)
    if seen != expected:
        raise ValueError(f"validation grid incomplete: {len(expected-seen)} observations missing")
    if len(base_scores) > 1:
        raise ValueError("validation observations disagree on the base MMLU subset score")
    rankings = []
    for key, rows in grouped.items():
        if any(row["status"] != "complete" for row in rows):
            continue
        if len({row["pre_zs"] for row in rows}) != 1:
            raise ValueError("a candidate has inconsistent pre-attack scores")
        if any(row["base_zs"]-row["pre_zs"] > max_zs_drop_pp for row in rows):
            continue
        hs = sum(row["hs"] for row in rows)/2
        drop = sum(row["pre_zs"]-row["post_zs"] for row in rows)/2
        rankings.append((hs, -drop, key))
    if not rankings:
        raise ValueError("no complete candidate satisfies the pre-attack utility constraint")
    rankings.sort()
    return dict(selected_id=rankings[0][2], config=candidates[rankings[0][2]],
                ranking=[dict(candidate_id=key, hs=hs, zs_drop=-neg) for hs,neg,key in rankings],
                rule=dict(max_zs_drop_pp=max_zs_drop_pp, aggregation="mean_of_attacks_2_and_4",
                          tie_break="largest_zs_drop_then_candidate_id"),
                manifest_sha256=manifest_sha256, utility_subset_sha256=utility_subset_sha256,
                candidates_sha256=digest(candidates), observations_sha256=digest(observations))
