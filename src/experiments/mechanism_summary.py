"""Tables of scripts/run_mechanism.sh: the beta = 0 ablation and the attack-trajectory diagnostics.

  summary_mechanism.md / .json  (in --out)
    1. Ablation: clean, LoRA (q,v) 1e-4 and mixed10 5e-5 for Base, CBG and CBG beta = 0 (HS / ZS, attack loss),
       all attacked on the same server with the same protocol.
    2. Trajectory: held-out log eta_k and the cosine between the attack displacement and the top safety direction
       of the starting model, at every saved attack step.

Usage: python -m src.experiments.mechanism_summary --root results/mechanism/runs --out results/mechanism
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.experiments.protocol import write_json

MODELS = (("base", "Base"), ("cbg", "CBG"), ("beta0", "CBG, beta = 0"))
ATTACKS = (("lora", "attack-eval-lora-lr1e-4", "LoRA (q,v) 1e-4"),
           ("mixed10", "attack-eval-mixed10-lr5e-5", "mixed10 5e-5"))


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else None


def _pct(value):
    return "--" if value is None else f"{100 * value:.1f}"


def build(root):
    root = Path(root)
    ablation, trajectory = [], []
    for model, label in MODELS:
        row = dict(model=model, label=label)
        for attack, folder, _ in ATTACKS:
            result = _read(root / model / folder / "attack_eval.json")
            if not result:
                continue
            if "pre" in result:
                row["clean_hs"] = result["pre"]["harmfulness"]["score"]
                row["clean_zs"] = result["pre"]["utility"].get("zs")
            entry = next(iter(result.get("attacks", {}).values()), None)
            if entry:
                row[f"{attack}_hs"] = entry["harmfulness"]["score"]
                row[f"{attack}_rate"] = entry["harmfulness"]["scores_over_50"]
                row[f"{attack}_zs"] = entry.get("utility", {}).get("zs")
                row[f"{attack}_loss"] = entry["train_loss"]
        metrics = _read(root / model / "train" / "metrics.json")
        if metrics:
            row["training"] = metrics.get("status")
        ablation.append(row)
        traj = _read(root / model / "trajectory.json")
        if traj:
            start = next(r for r in traj["geometry"] if r["attack"] == "start")
            trajectory.append(dict(model=model, label=label, attack="start", step=0, log_eta_k=start["geo/log_eta_k"],
                                   eff_rank_Khh=start["geo/eff_rank_Khh"], cos=None, cos2=None))
            for item in traj["trajectory"]:
                geo = next(r for r in traj["geometry"] if r["attack"] == item["attack"] and r["step"] == item["step"])
                trajectory.append(dict(model=model, label=label, attack=item["attack"], step=item["step"],
                                       log_eta_k=item["log_eta_k"], eff_rank_Khh=geo["geo/eff_rank_Khh"],
                                       cos=item["cos_top_safety_mean"], cos2=item["cos2_mean"]))
    return dict(ablation=ablation, trajectory=trajectory)


def markdown(summary):
    lines = ["# Mechanism experiments (scripts/run_mechanism.sh)", "",
             "## 1. Ablation: is the geometry term responsible?", "",
             "HS / ZS in %, attack loss in parentheses. Same server, same protocol for every row "
             "(attack_eval: SEAM padded labels, adamw_torch, bf16, seed 42; HS: checkpoint sampling).", "",
             "| Model | clean | LoRA (q,v) 1e-4 | HS>0.5 | mixed10 5e-5 | HS>0.5 |", "|---|---|---|---|---|---|"]
    for row in summary["ablation"]:
        cells = [f"{_pct(row.get('clean_hs'))} / {_pct(row.get('clean_zs'))}"]
        for attack, _, _ in ATTACKS:
            loss = row.get(f"{attack}_loss")
            cells.append(f"{_pct(row.get(f'{attack}_hs'))} / {_pct(row.get(f'{attack}_zs'))}"
                         + (f" ({loss:.3f})" if loss is not None else ""))
            cells.append(_pct(row.get(f"{attack}_rate")))
        note = "" if row.get("training") in (None, "complete") else f" (training: {row['training']})"
        lines.append(f"| {row['label']}{note} | " + " | ".join(cells) + " |")
    lines += ["", "## 2. Attack trajectory", "",
              "Held-out log eta_k on D_geo; cos: mean over the D_geo batches of the cosine (relative metric) between the "
              "attack displacement and the top safety direction of the starting model (positive = toward the harmful "
              "responses); cos^2: fraction of the displacement on that direction.", "",
              "| Model | Attack | Step | log eta_k | erank K_hh | cos | cos^2 |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in summary["trajectory"]:
        cos = "--" if row["cos"] is None else f"{row['cos']:+.3f}"
        cos2 = "--" if row["cos2"] is None else f"{row['cos2']:.3f}"
        lines.append(f"| {row['label']} | {row['attack']} | {row['step']} | {row['log_eta_k']:.3f} | "
                     f"{row['eff_rank_Khh']:.2f} | {cos} | {cos2} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    summary = build(args.root)
    out = Path(args.out)
    write_json(out / "summary_mechanism.json", summary)
    (out / "summary_mechanism.md").write_text(markdown(summary), encoding="utf-8")
    print(markdown(summary))


if __name__ == "__main__":
    main()
