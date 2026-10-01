"""S0.0 token-length audit using the same text templates as CBG training."""

from __future__ import annotations

from bisect import bisect_left

from src.data_processing.cbg_datasets import (
    BENIGN_PROMPT_TEMPLATE,
    SAFETY_PROMPT_TEMPLATE,
    _encode_without_special_tokens,
    _extract_refusal_response,
)


_BINS = (64, 128, 256, 384, 512, 768, 1024, 2048)


def _lengths(tokenizer, prompt, response):
    bos = int(getattr(tokenizer, "bos_token_id", None) is not None)
    return bos + len(_encode_without_special_tokens(tokenizer, prompt)), len(
        _encode_without_special_tokens(tokenizer, response)
    )


def _distribution(lengths):
    ordered = sorted(lengths)
    if not ordered:
        raise ValueError("cannot audit an empty dataset")
    bounds = (0, *_BINS)
    histogram = {}
    for lower, upper in zip(bounds, _BINS):
        label = f"<={upper}" if lower == 0 else f"({lower},{upper}]"
        histogram[label] = bisect_left(ordered, upper + 1) - bisect_left(ordered, lower + 1)
    histogram[f">{_BINS[-1]}"] = len(ordered) - bisect_left(ordered, _BINS[-1] + 1)
    percentile = lambda q: ordered[int((len(ordered) - 1) * q)]
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
        "histogram": histogram,
    }


def audit_lengths(safety, benign, safety_indices, benign_indices, tokenizer, *,
                  safety_max_length=256, benign_max_length=128):
    """Count examples for which CBG's actual prompt/response builder drops tokens.

    A safety *pair* fails if either target or its shared prompt loses tokens.
    Full lengths include the manually inserted BOS token and use no implicit EOS.
    """
    if safety_max_length < 2 or benign_max_length < 2:
        raise ValueError("length limits must leave a prompt and response token")
    if not safety_indices or not benign_indices:
        raise ValueError("both defense pools must be nonempty")

    harmful_lengths, refusal_lengths, benign_lengths = [], [], []
    safety_affected = harmful_affected = refusal_affected = 0
    safety_prompt_clipped = harmful_response_clipped = refusal_response_clipped = 0
    for index in safety_indices:
        row = safety[index]
        prompt = str(row["prompt"])
        formatted = SAFETY_PROMPT_TEMPLATE.format(prompt=prompt)
        harmful = str(row["response"]).strip()
        refusal = _extract_refusal_response(str(row["refusal"]), prompt)
        prompt_len, harmful_len = _lengths(tokenizer, formatted, harmful)
        _, refusal_len = _lengths(tokenizer, formatted, refusal)
        harmful_lengths.append(prompt_len + harmful_len)
        refusal_lengths.append(prompt_len + refusal_len)

        # tokenize_paired_responses uses one shared prompt budget derived from
        # the longer response, then truncates each response from the right.
        prompt_kept = min(prompt_len, max(1, safety_max_length - max(harmful_len, refusal_len)))
        response_budget = safety_max_length - prompt_kept
        prompt_lost = prompt_kept < prompt_len
        harmful_lost = harmful_len > response_budget
        refusal_lost = refusal_len > response_budget
        safety_prompt_clipped += prompt_lost
        harmful_response_clipped += harmful_lost
        refusal_response_clipped += refusal_lost
        harmful_affected += prompt_lost or harmful_lost
        refusal_affected += prompt_lost or refusal_lost
        safety_affected += prompt_lost or harmful_lost or refusal_lost

    benign_affected = benign_prompt_clipped = benign_response_clipped = 0
    for index in benign_indices:
        row = benign[index]
        prompt = str(row.get("instruction", "")) + "\n" + str(row.get("input", ""))
        formatted = BENIGN_PROMPT_TEMPLATE.format(prompt=prompt)
        response = str(row.get("output", "")).strip()
        prompt_len, response_len = _lengths(tokenizer, formatted, response)
        benign_lengths.append(prompt_len + response_len)
        prompt_kept = min(prompt_len, max(1, benign_max_length - response_len))
        response_lost = response_len > benign_max_length - prompt_kept
        prompt_lost = prompt_kept < prompt_len
        benign_prompt_clipped += prompt_lost
        benign_response_clipped += response_lost
        benign_affected += prompt_lost or response_lost

    n_safety, n_benign = len(safety_indices), len(benign_indices)
    safety_fraction, benign_fraction = safety_affected / n_safety, benign_affected / n_benign
    return {
        "status": "passed" if safety_fraction < 0.05 and benign_fraction < 0.05 else "failed",
        "threshold_fraction_exclusive": 0.05,
        "safety": {
            "limit": safety_max_length, "pairs": n_safety,
            "affected_pairs": safety_affected, "affected_fraction": safety_fraction,
            "affected_harmful": harmful_affected, "affected_refusal": refusal_affected,
            "prompt_clipped_pairs": safety_prompt_clipped,
            "harmful_response_clipped": harmful_response_clipped,
            "refusal_response_clipped": refusal_response_clipped,
            "harmful_full_lengths": _distribution(harmful_lengths),
            "refusal_full_lengths": _distribution(refusal_lengths),
        },
        "benign": {
            "limit": benign_max_length, "examples": n_benign,
            "affected": benign_affected, "affected_fraction": benign_fraction,
            "prompt_clipped": benign_prompt_clipped,
            "response_clipped": benign_response_clipped,
            "full_lengths": _distribution(benign_lengths),
        },
    }
