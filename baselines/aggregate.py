"""Collect HS/ZS of every evaluated case and write <out>/RESULTS.md (mean ± sample std over seeds) and
<out>/results.csv (one row per model, method, case, seed).

  python baselines/aggregate.py --out outputs/baselines
"""
import argparse
import csv
import json
import statistics
from pathlib import Path

CASES = ("clean", "harmful-lr2e-5", "harmful-lr5e-5", "harmful-lr8e-5", "harmful-lr1e-4", "harmful-lr2e-4")


def cell(values):
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.2f}"
    return f"{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--methods", default="base,seam,sdd,ctrap")
    args = parser.parse_args()
    results = args.out / "results"
    models = [m for m in args.models.split(",") if m] or sorted(p.name for p in results.iterdir() if p.is_dir())
    methods = [m for m in args.methods.split(",") if m]
    rows, lines = [], ["# Baseline results (HS / ZS, %)", "",
                       "Mean ± sample std over seeds. HS: harmfulness on 100 BeaverTails test prompts (lower is safer). "
                       "ZS: mean 0-shot accuracy on MMLU, TruthfulQA-MC1, HellaSwag, ARC-easy (higher keeps utility). "
                       "Attack: full fine-tuning on 1,000 harmful rows, 250 steps, AdamW.", ""]
    for model in models:
        seeds = ([int(s) for s in args.seeds.split(",") if s] or
                 sorted(int(p.name[4:]) for p in (results / model).glob("seed*")))
        table = {}
        for method in methods:
            for case in CASES:
                for seed in seeds:
                    path = results / model / f"seed{seed}" / method / "cases" / method / case / "result.json"
                    if path.is_file():
                        result = json.loads(path.read_text(encoding="utf-8"))
                        table.setdefault((method, case), []).append((seed, result["HS_percent"], result["ZS_percent"]))
                        rows.append({"model": model, "method": method, "case": case, "seed": seed,
                                     "HS": round(result["HS_percent"], 3), "ZS": round(result["ZS_percent"], 3)})
        for metric, index in (("HS", 1), ("ZS", 2)):
            lines += [f"## {model}: {metric} (seeds {', '.join(map(str, seeds))})", "",
                      "| Case | " + " | ".join(m.upper() if m != "base" else "Base" for m in methods) + " |",
                      "|---|" + "---|" * len(methods)]
            for case in CASES:
                lines.append(f"| {case} | " + " | ".join(cell([v[index] for v in table.get((m, case), [])])
                                                         for m in methods) + " |")
            lines.append("")
        missing = [f"{m}/{c}/seed{s}" for m in methods for c in CASES for s in seeds
                   if s not in {v[0] for v in table.get((m, c), [])}]
        if missing:
            lines += [f"Missing for {model}: {', '.join(missing)}", ""]
    (args.out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (args.out / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "method", "case", "seed", "HS", "ZS"])
        writer.writeheader()
        writer.writerows(rows)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
