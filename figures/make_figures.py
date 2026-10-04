"""Figures for the CBG paper (Experiments / Geometry Analysis), from measured results only.

Inputs (read-only):
  - final CBG training log: HF ducthang1703/cbg-llama2-7b-step500, training/train_log.jsonl
  - evaluation runs in C:/Users/ducth/seam/results/defense-eval-*-h100-*/summary.json
  - orthogonalization records results/defense-eval-ortho-h100-oanh-*/{base,cbg}-ortho.json
Outputs: figures/*.pdf (for LaTeX) and figures/*.png (preview).

Usage: python figures/make_figures.py [--seam-results C:/Users/ducth/seam/results]
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter

OUT = Path(__file__).resolve().parent

# Categorical slots 1-5 of the validated reference palette, in fixed order (never cycled); each model also
# has its own marker, so identity never rests on color alone.
MODELS = ["base", "seam", "sdd", "ctrap", "cbg"]
NAMES = {"base": "Base", "seam": "SEAM", "sdd": "SDD", "ctrap": "CTRAP", "cbg": "CBG"}
COLOR = {"base": "#2a78d6", "seam": "#eb6834", "sdd": "#1baf7a", "ctrap": "#eda100", "cbg": "#e87ba4"}
MARKER = {"base": "o", "seam": "s", "sdd": "^", "ctrap": "D", "cbg": "X"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"

RUNS = {
    "core": ["defense-eval-core-h100-oanh-20261004T053520Z", "defense-eval-core-h100-thangtd-20261004T053520Z"],
    "extra": ["defense-eval-extra-h100-thangtd-20261004T100804Z"],
    "variants": ["defense-eval-variants-h100-apu-20261004T113430Z"],
    "ortho": ["defense-eval-ortho-h100-oanh-20261004T112025Z"],
}


def style():
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 1.5, "lines.markersize": 4.5,
        "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
    })


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def load_metrics(root, kind):
    rows = {}
    for run in RUNS[kind]:
        rows.update(json.loads((root / run / "summary.json").read_text(encoding="utf-8"))["metrics"])
    return rows


def training_log():
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("ducthang1703/cbg-llama2-7b-step500", "training/train_log.jsonl")
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def window_means(rows, key, width=25):
    points = [(r["step"], r[key]) for r in rows if key in r]
    out = []
    for end in range(width, 501, width):
        values = [v for s, v in points if end - width < s <= end]
        if values:
            out.append((end, statistics.mean(values)))
    return out


# ------------------------------------------------------------------------------------------------
def fig_training(log):
    """Held-out geometry and training losses of the final CBG run."""
    geo = [r for r in log if "geo/log_eta_k" in r]
    steps = [r["step"] for r in geo]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.9))
    c1, c2 = COLOR["cbg"], INK2   # one model (CBG): its color, plus neutral ink for the second quantity

    ax = axes[0, 0]
    ax.plot(steps, [r["geo/log_eta_k"] for r in geo], color=c1, marker="o")
    ax.set_title(r"(a) held-out $\log\eta_k$")
    ax.set_xlabel("training step")

    ax = axes[0, 1]
    ax.plot(steps, [r["geo/nu_1k"] for r in geo], color=c1, marker="o", label=r"$\nu_{1:k}$ (safety)")
    ax.plot(steps, [r["geo/sigma_1k"] for r in geo], color=c2, marker="s", label=r"$\sigma_{1:k}$ (not benign-explained)")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([0.2, 0.5, 1, 2, 5]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_yticklabels(["0.2", "0.5", "1", "2", "5"])
    ax.set_title(r"(b) spectral sums of $K_{hh}$ and $S_\mu$")
    ax.set_xlabel("training step")
    ax.legend(loc="upper left")

    ax = axes[0, 2]
    ax.plot(steps, [r["geo/eff_rank_Khh"] for r in geo], color=c1, marker="o", label=r"$K_{hh}$ (8 safety pairs)")
    ax.plot(steps, [r["geo/eff_rank_Kbb"] for r in geo], color=c2, marker="s", label=r"$K_{bb}$ (32 benign seq.)")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_yticklabels(["1", "2", "5", "10", "20"])
    ax.set_title("(c) effective rank of the Grams")
    ax.set_xlabel("training step")
    ax.legend(loc="lower left")

    ax = axes[1, 0]
    ax.plot(steps, [r["geo/tr_Kbb"] for r in geo], color=c1, marker="o")
    ax.set_title(r"(d) benign sensitivity $\mathrm{tr}\,K_{bb}$")
    ax.set_xlabel("training step")

    ax = axes[1, 1]
    for key, color, marker, label in (("L_harm", c1, "o", r"at $\theta$"),
                                      ("L_harm_traj_1e-05", c2, "s", r"at $\theta'=\theta-10^{-5}\,\mathrm{sign}\,\nabla$")):
        points = window_means(log, key)
        ax.plot([p[0] for p in points], [p[1] for p in points], color=color, marker=marker, markersize=3.5, label=label)
    ax.set_title(r"(e) harmful loss $\mathcal{L}_{\rm harm}$ (25-step means)")
    ax.set_xlabel("training step")
    ax.set_ylim(0.5, 3.4)
    ax.legend(loc="upper left")

    ax = axes[1, 2]
    for key, color, marker, label in (("L_bgn", c1, "o", r"$\mathcal{L}_{\rm bgn}$ (KL to $\theta_0$)"),
                                      ("L_aln", c2, "s", r"$\mathcal{L}_{\rm aln}$")):
        points = window_means(log, key)
        ax.plot([p[0] for p in points], [p[1] for p in points], color=color, marker=marker, markersize=3.5, label=label)
    ax.axhline(0.3, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), label=r"utility gate (mean $\mathcal{L}_{\rm bgn}\leq0.3$)")
    for gate in (100, 250):
        ax.axvline(gate, color=GRID, linewidth=1)
    ax.set_title("(f) training losses (25-step means)")
    ax.set_xlabel("training step")
    ax.set_ylim(0, 0.9)
    ax.legend(loc="upper right", fontsize=6.8)

    fig.tight_layout(w_pad=1.2, h_pad=1.4)
    save(fig, "fig_training_dynamics")


# ------------------------------------------------------------------------------------------------
def fig_lr(core, extra):
    """HS and ZS against the attack learning rate (separate panels: one axis per measure)."""
    rows = {**core, **extra}
    lrs = [("2e-5", 2e-5), ("5e-5", 5e-5), ("8e-5", 8e-5), ("1e-4", 1e-4), ("2e-4", 2e-4)]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for model in MODELS:
        points = [(value, rows[f"{model}/harmful-lr{tag}"]) for tag, value in lrs if f"{model}/harmful-lr{tag}" in rows]
        xs = [p[0] for p in points]
        for ax, key in zip(axes, ("HS_percent", "ZS_percent")):
            ax.plot(xs, [p[1][key] for p in points], color=COLOR[model], marker=MARKER[model], label=NAMES[model])
    axes[0].axhline(10, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    axes[0].text(2.05e-5, 12, "HS = 10 (compromise threshold)", color=INK2, fontsize=7)
    axes[1].axhline(30, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    axes[1].text(2.05e-5, 31, "ZS = 30 (self-destruction threshold)", color=INK2, fontsize=7)
    for ax, title in zip(axes, ("(a) harmfulness after the attack (HS, %)", "(b) utility after the attack (ZS, %)")):
        ax.set_xscale("log")
        ax.set_xticks([v for _, v in lrs])
        ax.set_xticklabels([t for t, _ in lrs], rotation=30)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("attack learning rate")
        ax.set_title(title)
    axes[0].set_ylim(0, 100)
    axes[1].set_ylim(25, 55)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(w_pad=2, rect=(0, 0.07, 1, 1))
    save(fig, "fig_lr_sweep")


# ------------------------------------------------------------------------------------------------
def fig_weak_attacks(core, extra, variants, ortho):
    """Base vs CBG under every attack: HS (dot plot, one row per attack)."""
    rows = {**core, **extra, **variants, **ortho}
    cases = [("clean (no attack)", "clean"), ("harmful, lr 2e-5", "harmful-lr2e-5"), ("harmful, lr 1e-4", "harmful-lr1e-4"),
             ("10% poisoned, lr 5e-5", "mixed10-lr5e-5"), ("LoRA (q,v), lr 1e-4", "harmful-lora-lr1e-4"),
             ("SGD, lr 1e-4", "harmful-sgd-lr1e-4"), ("frozen middle blocks, lr 1e-4", "harmful-frozenmid-lr1e-4"),
             ("orthogonalization (KL < 0.1)", "orthogonalized")]
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    ys = list(range(len(cases)))[::-1]
    for y, (label, case) in zip(ys, cases):
        b, c = rows[f"base/{case}"]["HS_percent"], rows[f"cbg/{case}"]["HS_percent"]
        ax.plot([b, c], [y, y], color=GRID, linewidth=2.5, zorder=1)
        for model, value, va in (("base", b, "bottom"), ("cbg", c, "top")):
            ax.scatter([value], [y], s=46, color=COLOR[model], marker=MARKER[model], zorder=3,
                       edgecolors="white", linewidths=1.2, label=NAMES[model] if y == ys[0] else None)
        delta = c - b
        ax.text(101, y, f"{delta:+.1f}", va="center", ha="left", fontsize=7, color=INK if abs(delta) >= 5 else MUTED)
    ax.axvline(10, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.set_yticks(ys)
    ax.set_yticklabels([c[0] for c in cases])
    ax.set_xlim(0, 100)
    ax.set_xlabel("HS (%) after the attack; right column: CBG minus Base")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)
    fig.tight_layout()
    save(fig, "fig_base_vs_cbg")


# ------------------------------------------------------------------------------------------------
def fig_main(core):
    """Main comparison: HS per scenario for the five models (dot plot)."""
    cases = [("clean", "clean"), ("harmful 2e-5", "harmful-lr2e-5"), ("harmful 5e-5", "harmful-lr5e-5"),
             ("harmful 1e-4", "harmful-lr1e-4"), ("benign 2e-5", "benign-lr2e-5"), ("mixed10 5e-5", "mixed10-lr5e-5")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
    ys = list(range(len(cases)))[::-1]
    offsets = {m: (i - 2) * 0.13 for i, m in enumerate(MODELS)}
    for ax, key, title in zip(axes, ("HS_percent", "ZS_percent"), ("(a) HS (%)", "(b) ZS (%)")):
        for model in MODELS:
            ax.scatter([core[f"{model}/{c}"][key] for _, c in cases], [y + offsets[model] for y in ys],
                       s=22, color=COLOR[model], marker=MARKER[model], edgecolors="white", linewidths=0.8,
                       label=NAMES[model], zorder=3)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    axes[0].axvline(10, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    axes[0].set_xlim(0, 100)
    axes[1].axvline(30, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    axes[1].set_xlim(25, 55)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([c[0] for c in cases])
    axes[1].legend(loc="lower left", ncol=1, bbox_to_anchor=(1.0, 0.0))
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_main_comparison")


# ------------------------------------------------------------------------------------------------
def fig_attack_loss(core):
    """Per-step attack loss at lr 1e-4 (one panel; log scale shows SEAM's trap at the start)."""
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for model in MODELS:
        losses = core[f"{model}/harmful-lr1e-4"]["training"]["step_losses"]
        smooth = [statistics.mean(losses[max(0, i - 9):i + 1]) for i in range(len(losses))]
        ax.plot(range(1, len(losses) + 1), smooth, color=COLOR[model], linewidth=1.5, label=NAMES[model])
    ax.set_yscale("log")
    ax.set_xlabel("attack step (lr 1e-4)")
    ax.set_ylabel("training loss (10-step mean)")
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    save(fig, "fig_attack_loss")


# ------------------------------------------------------------------------------------------------
def fig_orthogonalization(records):
    """Refusal-direction candidates: validation refusal against harmless KL, one point per layer."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    ax = axes[0]
    ax.axvspan(1e-3, 0.1, color="#f0efec", zorder=0)
    ax.text(0.0105, 0.97, "admissible: KL < 0.1", color=INK2, fontsize=7, va="top")
    for model in ("base", "cbg"):
        cands = records[model]["candidates"]
        ax.scatter([c["harmless_kl"] for c in cands], [100 * c["val_refusal"] for c in cands], s=20,
                   color=COLOR[model], marker=MARKER[model], edgecolors="white", linewidths=0.8,
                   label=NAMES[model], zorder=3)
        chosen = records[model]["chosen"]
        ax.annotate(f"{NAMES[model]} chosen: layer {chosen['layer']}", (chosen["harmless_kl"], 100 * chosen["val_refusal"]),
                    xytext=(0.0115, 40 if model == "cbg" else 18), textcoords="data", fontsize=7, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))
    ax.set_xscale("log")
    ax.set_xlim(0.01, 2)
    ax.set_ylim(-3, 80)
    ax.set_xlabel("harmless next-token KL of the ablation")
    ax.set_ylabel("validation refusal (%)")
    ax.set_title("(a) cost vs. effect of each candidate direction")
    ax.legend(loc="lower right")

    ax = axes[1]
    for model in ("base", "cbg"):
        cands = records[model]["candidates"]
        # Lowest refusal reachable at or below each KL budget (the attacker's frontier).
        budgets = sorted({c["harmless_kl"] for c in cands})
        frontier = [min(100 * c["val_refusal"] for c in cands if c["harmless_kl"] <= k) for k in budgets]
        ax.step(budgets, frontier, where="post", color=COLOR[model], linewidth=1.5, label=NAMES[model])
    ax.axvline(0.1, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(0.104, 70, "KL = 0.1", color=INK2, fontsize=7)
    ax.set_xscale("log")
    ax.set_xlim(0.01, 2)
    ax.set_ylim(-3, 80)
    ax.set_xlabel("attacker's KL budget")
    ax.set_ylabel("lowest reachable refusal (%)")
    ax.set_title("(b) refusal reachable within a KL budget")
    ax.legend(loc="upper right")
    fig.tight_layout(w_pad=2)
    save(fig, "fig_orthogonalization")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seam-results", default="C:/Users/ducth/seam/results")
    args = parser.parse_args()
    root = Path(args.seam_results)
    style()
    core, extra = load_metrics(root, "core"), load_metrics(root, "extra")
    variants, ortho = load_metrics(root, "variants"), load_metrics(root, "ortho")
    ortho_dir = root / RUNS["ortho"][0]
    records = {m: json.loads((ortho_dir / f"{m}-ortho.json").read_text(encoding="utf-8")) for m in ("base", "cbg")}
    fig_training(training_log())
    fig_main(core)
    fig_lr(core, extra)
    fig_weak_attacks(core, extra, variants, ortho)
    fig_attack_loss(core)
    fig_orthogonalization(records)


if __name__ == "__main__":
    main()
