"""Hyperparameter ablation helpers for scripts/run_ablation.sh.

  override  write a config with dotted-path overrides:
            python -m src.experiments.ablation override --base <yaml> --out <yaml> method.beta=0.003 method.mu_mode=detached
  summary   one table over all runs: training geometry/losses (train_log.jsonl) + SEAM-protocol validation
            attack results (attack-eval-val/attack_eval.json), with the selection rule of select_config.py:
            python -m src.experiments.ablation summary --root <runs dir> --base base-llama2-7b-chat --out <dir>
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics

import yaml

from src.experiments.select_config import select


def apply_overrides(config, overrides):
    """``overrides``: "a.b.c=value" strings; values parsed as YAML; every key must already exist."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} is not key=value")
        path, raw = item.split("=", 1)
        keys = path.split(".")
        node = config
        for key in keys[:-1]:
            if not isinstance(node, dict) or key not in node:
                raise KeyError(f"unknown config section {path!r}")
            node = node[key]
        if keys[-1] not in node:
            raise KeyError(f"unknown config key {path!r}")
        value = yaml.safe_load(raw)
        old = node[keys[-1]]
        if isinstance(old, float) and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)   # keep float keys float (alpha: 3 -> 3.0)
        if isinstance(old, int) and not isinstance(old, bool) and isinstance(value, float):
            if not value.is_integer():
                raise ValueError(f"{path} is an integer, got {raw}")
            value = int(value)
        node[keys[-1]] = value
    return config


def train_summary(run_dir, tail=50):
    log = Path(run_dir) / "train" / "train_log.jsonl"
    if not log.exists():
        return {}
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    steps = [r for r in rows if "L_aln" in r]
    geo = [r for r in rows if "geo/log_eta_k" in r]
    last = steps[-tail:]
    cbg = [r for r in steps if r.get("cbg_step") == 1. and "grad_ratio" in r]
    summary = dict(steps=steps[-1]["step"] if steps else 0,
                   L_aln_last=statistics.mean(r["L_aln"] for r in last) if last else None,
                   L_bgn_last=statistics.mean(r["L_bgn"] for r in last) if last else None,
                   L_bgn_max=max((r["L_bgn"] for r in steps), default=None),
                   clipped_fraction=(sum(r.get("geometry_clip_scale", 1.) < 1. for r in cbg) / len(cbg)) if cbg else None,
                   grad_ratio_median=statistics.median(r["grad_ratio"] for r in cbg) if cbg else None)
    if geo:
        summary.update(geo_log_eta_k_start=geo[0]["geo/log_eta_k"], geo_log_eta_k_end=geo[-1]["geo/log_eta_k"],
                       geo_delta=geo[-1]["geo/log_eta_k"] - geo[0]["geo/log_eta_k"])
    metrics = Path(run_dir) / "train" / "metrics.json"
    summary["complete"] = metrics.exists() and json.loads(metrics.read_text(encoding="utf-8")).get("status") == "complete"
    return summary


def summarize(root, base, scenarios=None, **rule):
    root = Path(root)
    names = [d.name for d in sorted(root.iterdir()) if d.is_dir() and d.name != base]
    load = lambda name: json.loads((root / name / "attack-eval-val" / "attack_eval.json").read_text(encoding="utf-8"))
    evaluated = {n: load(n) for n in names if (root / n / "attack-eval-val" / "attack_eval.json").exists()}
    base_file = root / base / "attack-eval-val" / "attack_eval.json"
    selection = select(evaluated, load(base), **rule) if evaluated and base_file.exists() else None
    by_name = {row["name"]: row for row in (selection["rows"] if selection else [])}
    rows = []
    for name in names:
        row = dict(name=name, overrides=" ".join((scenarios or {}).get(name, [])), **train_summary(root / name))
        evaluation = by_name.get(name, {})
        row.update({k: evaluation.get(k) for k in ("pre_harmfulness", "utility", "post_harmfulness_mean",
                                                    "post_harmfulness_max", "feasible")})
        rows.append(row)
    return dict(rows=rows, best=selection["best"] if selection else None,
                base=selection["base"] if selection else None, criteria=selection["criteria"] if selection else None)


def markdown(summary):
    columns = ["name", "overrides", "geo_delta", "L_aln_last", "L_bgn_last", "L_bgn_max", "clipped_fraction",
               "pre_harmfulness", "utility", "post_harmfulness_mean", "post_harmfulness_max", "feasible"]
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else ("" if v is None else str(v))
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    if summary["base"]:
        base = summary["base"]
        lines.append("| (base, no defense) | | | | | | | " + " | ".join(
            fmt(base.get(k)) for k in ("pre_harmfulness", "utility", "post_harmfulness_mean",
                                       "post_harmfulness_max")) + " | |")
    for row in summary["rows"]:
        name = row["name"] + (" **(best)**" if row["name"] == summary["best"] else "")
        lines.append("| " + " | ".join([name] + [fmt(row.get(c)) for c in columns[1:]]) + " |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    o = sub.add_parser("override")
    o.add_argument("--base", required=True)
    o.add_argument("--out", required=True)
    o.add_argument("overrides", nargs="*")
    s = sub.add_parser("summary")
    s.add_argument("--root", required=True)
    s.add_argument("--base", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--scenarios", help="scenarios file written by run_ablation.sh (name<TAB>overrides)")
    s.add_argument("--utility-task", default="arc_easy")
    s.add_argument("--max-utility-drop", type=float, default=0.02)
    s.add_argument("--max-harm-increase", type=float, default=0.05)
    args = parser.parse_args(argv)
    if args.command == "override":
        config = apply_overrides(yaml.safe_load(Path(args.base).read_text(encoding="utf-8")), args.overrides)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return
    scenarios = {}
    if args.scenarios and Path(args.scenarios).exists():
        for line in Path(args.scenarios).read_text(encoding="utf-8").splitlines():
            if line.strip():
                name, _, rest = line.partition("\t")
                scenarios[name] = rest.split()
    summary = summarize(args.root, args.base, scenarios, utility_task=args.utility_task,
                        max_utility_drop=args.max_utility_drop, max_harm_increase=args.max_harm_increase)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    table = markdown(summary)
    (out / "summary.md").write_text(table + "\n", encoding="utf-8")
    if summary["rows"]:
        with (out / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            fields = list(dict.fromkeys(key for row in summary["rows"] for key in row))   # runs may lack geo keys
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summary["rows"])
    print(table)
    print(f"\nbest (validation rule): {summary['best'] or 'n/a'}")


if __name__ == "__main__":
    main()
