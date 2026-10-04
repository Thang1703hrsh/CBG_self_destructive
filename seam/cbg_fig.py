from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle, FancyArrowPatch

BLUE = "#2F80ED"
BLUE_FILL = "#8FBCEB"
RED = "#D62828"
RED_FILL = "#F28B82"
ORANGE = "#FF7F0E"
PURPLE = "#3F2DA8"

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "mathtext.fontset": "stix",
})

def draw_top_panel(save_png="cbg_local_geometry_top.png", save_pdf="cbg_local_geometry_top.pdf"):
    fig = plt.figure(figsize=(12, 4.2), dpi=180)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    ax.text(6.0, 3.95, r"Local functional geometry (near $\theta$)",
            ha="center", va="top", fontsize=22, fontweight="semibold")

    center = (4.7, 1.75)
    widths = [4.9, 4.0, 3.1, 2.2, 1.3]
    heights = [2.9, 2.3, 1.8, 1.3, 0.8]
    for i, (w, h) in enumerate(zip(widths, heights)):
        if i == 0:
            ax.add_patch(Ellipse(center, w, h, facecolor=BLUE_FILL, alpha=0.28, edgecolor="none"))
        else:
            ax.add_patch(Ellipse(center, w, h, facecolor="none", edgecolor=BLUE,
                                 alpha=0.35 + 0.1*i, lw=1.6))

    red_center = (7.7, 2.75)
    angle = 20
    red_widths = [4.0, 3.2, 2.5, 1.8, 1.15]
    red_heights = [1.65, 1.30, 1.00, 0.72, 0.48]
    for i, (w, h) in enumerate(zip(red_widths, red_heights)):
        if i == 0:
            ax.add_patch(Ellipse(red_center, w, h, angle=angle, facecolor=RED_FILL, alpha=0.20, edgecolor="none"))
        else:
            ax.add_patch(Ellipse(red_center, w, h, angle=angle, facecolor="none", edgecolor=RED,
                                 alpha=0.25 + 0.12*i, lw=1.5))

    ax.add_patch(Circle((4.0, 1.8), 0.12, color="black", zorder=5))
    ax.text(4.0, 1.35, r"$\theta$", ha="center", va="center", fontsize=28)

    ax.add_patch(FancyArrowPatch((4.1, 1.9), (7.15, 2.78),
                                 arrowstyle='Simple,tail_width=1.4,head_width=12,head_length=16',
                                 color=RED, lw=0, alpha=0.98, zorder=6))
    ax.add_patch(Circle((7.62, 2.87), 0.10, facecolor=RED, edgecolor=RED, zorder=7))

    ax.add_patch(FancyArrowPatch((4.05, 1.72), (8.55, 1.95),
                                 connectionstyle="arc3,rad=0.42",
                                 arrowstyle='Simple,tail_width=1.0,head_width=10,head_length=12',
                                 linestyle="--", color=ORANGE, lw=2.0, alpha=0.98, zorder=6))
    ax.add_patch(Circle((8.82, 1.86), 0.09, facecolor=ORANGE, edgecolor=ORANGE, zorder=7))

    ax.annotate("coupled harmful\ntrajectory\n(high collateral)",
                xy=(7.65, 2.95), xytext=(9.0, 3.15),
                fontsize=19, color=RED, fontweight="semibold",
                ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=RED, lw=1.5))
    ax.text(8.85, 1.80, "alternative\nlow-collateral escape",
            fontsize=20, color=ORANGE, fontweight="semibold", ha="left", va="center")
    ax.text(8.9, 0.77, "another update direction\nmay remain cheap",
            fontsize=19, color="#222222", ha="left", va="center")
    ax.text(0.42, 0.62, "benign\ndegradation $Q_b(u)$",
            fontsize=22, color=BLUE, fontweight="semibold", ha="left", va="center")

    fig.savefig(save_png, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(save_pdf, bbox_inches="tight", facecolor="white")

def draw_bottom_panel(save_png="cbg_local_geometry_bottom.png", save_pdf="cbg_local_geometry_bottom.pdf"):
    fig = plt.figure(figsize=(7.6, 5.2), dpi=180)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 7.6)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    ax.text(3.95, 4.95, r"Local functional geometry (near $\theta$)",
            ha="center", va="top", fontsize=22, fontweight="semibold")

    ax.annotate("", xy=(1.05, 4.0), xytext=(1.05, 0.9),
                arrowprops=dict(arrowstyle="simple", color="black", lw=0, mutation_scale=16))
    ax.annotate("", xy=(6.2, 0.95), xytext=(3.15, 0.95),
                arrowprops=dict(arrowstyle="simple", color="black", lw=0, mutation_scale=16))
    ax.text(0.78, 4.08, r"$u_2$", fontsize=24, ha="center", va="bottom")
    ax.text(6.3, 0.79, r"$u_1$", fontsize=24, ha="left", va="center")

    center = (3.85, 2.35)
    blue_angle = -25
    blue_ws = [3.9, 3.25, 2.6, 2.0]
    blue_hs = [2.6, 2.1, 1.65, 1.15]
    for i, (w, h) in enumerate(zip(blue_ws, blue_hs)):
        if i == 0:
            ax.add_patch(Ellipse(center, w, h, angle=blue_angle, facecolor=BLUE_FILL, alpha=0.25, edgecolor="none"))
        else:
            ax.add_patch(Ellipse(center, w, h, angle=blue_angle, facecolor="none", edgecolor=BLUE,
                                 lw=1.6, alpha=0.45 + 0.1*i))

    red_angle = 25
    red_ws = [3.0, 2.45, 1.95, 1.45, 1.0]
    red_hs = [1.55, 1.25, 0.95, 0.70, 0.45]
    for i, (w, h) in enumerate(zip(red_ws, red_hs)):
        if i == 0:
            ax.add_patch(Ellipse(center, w, h, angle=red_angle, facecolor=RED_FILL, alpha=0.22, edgecolor="none"))
        else:
            ax.add_patch(Ellipse(center, w, h, angle=red_angle, facecolor="none", edgecolor=RED,
                                 lw=1.5, alpha=0.30 + 0.12*i))

    ax.add_patch(Circle((3.55, 2.10), 0.12, color="black", zorder=5))
    ax.text(3.55, 1.58, r"$\theta$", ha="center", va="center", fontsize=28)

    ax.add_patch(FancyArrowPatch((3.58, 2.12), (5.42, 3.83),
                                 arrowstyle='Simple,tail_width=1.6,head_width=12,head_length=16',
                                 color=PURPLE, lw=0, alpha=0.98, zorder=6))
    ax.add_patch(Circle((4.95, 3.33), 0.04, facecolor=PURPLE, edgecolor=PURPLE))

    ax.text(5.12, 4.06, r"$u^*$ (weakest escape)",
            fontsize=22, color=PURPLE, fontweight="semibold", ha="left", va="center")
    ax.text(5.88, 2.95, "safety-sensitive\nchange $Q_h(u)$",
            fontsize=21, color=RED, fontweight="semibold", ha="left", va="center")
    ax.text(0.16, 0.67, "benign\ndegradation $Q_b(u)$",
            fontsize=21, color=BLUE, fontweight="semibold", ha="left", va="center")

    fig.savefig(save_png, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(save_pdf, bbox_inches="tight", facecolor="white")

if __name__ == "__main__":
    draw_top_panel()
    draw_bottom_panel()
