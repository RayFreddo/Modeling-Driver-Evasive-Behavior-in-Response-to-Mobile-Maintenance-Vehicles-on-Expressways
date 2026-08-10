from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "figure" / "development" / "figure7"
OUT = WORK / "figure7_image_only_final"
OUT.mkdir(parents=True, exist_ok=True)
SOURCE = ROOT / "figure" / "Figure5_four_scenarios_photorealistic_4K_final.png"

ROAD = "#85898B"
WHITE = "#FFFFFF"
INK = "#17253D"
BLUE = "#17669D"
BLUE_SOFT = "#77B7D9"
PURPLE = "#6C5CC7"
PURPLE_SOFT = "#B8AEEB"
RED = "#C8443F"
GREEN = "#2B9B61"
AMBER = "#D08C00"
RULE = "#D6DDE1"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 10,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def sprites() -> tuple[Image.Image, Image.Image]:
    source = Image.open(SOURCE).convert("RGB")
    car = source.crop((480, 1400, 800, 1850)).crop((65, 35, 300, 430))
    mmv = source.crop((470, 250, 800, 870)).crop((45, 40, 300, 510))
    return car, mmv


def curve(ax, points, color, lw=2.3, ls="-", zorder=5):
    verts = [points[0], points[1], points[2], points[3]]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
    p = PathPatch(
        MplPath(verts, codes),
        facecolor="none",
        edgecolor=color,
        linewidth=lw,
        linestyle=ls,
        capstyle="round",
        joinstyle="round",
        zorder=zorder,
    )
    ax.add_patch(p)
    return p


def panel_base(ax, number, title, color, car, mmv):
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 10.5)
    ax.set_axis_off()
    ax.add_patch(Rectangle((0, 0), 4, 9.15, facecolor=ROAD, edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((0, 0), 4, 10.5, facecolor="none", edgecolor="#58636A", lw=1.0, zorder=20))
    ax.plot([2, 2], [0, 9.15], color=WHITE, lw=2.0, ls=(0, (5, 5)), zorder=1)
    ax.add_patch(Circle((0.38, 9.82), 0.25, facecolor=color, edgecolor="none", zorder=21))
    ax.text(0.38, 9.82, str(number), ha="center", va="center", color=WHITE,
            fontsize=10, fontweight="bold", zorder=22)
    ax.text(0.78, 9.82, title, ha="left", va="center", color=color,
            fontsize=11, fontweight="bold", zorder=22)

    # Passenger car and MMV at an approximately 1:1.45 length ratio.
    ax.imshow(car, extent=(2.64, 3.36, 0.82, 2.12), zorder=10, interpolation="lanczos")
    ax.imshow(mmv, extent=(2.50, 3.50, 7.00, 8.88), zorder=9, interpolation="lanczos")


def horizon_points(ax, xs, ys, color, edge=WHITE, size=31, zorder=9):
    ax.scatter(xs, ys, s=size, facecolor=color, edgecolor=edge, linewidth=0.8, zorder=zorder)


def build():
    car, mmv = sprites()
    fig = plt.figure(figsize=(7.20, 2.80), facecolor="white")
    axes = [
        fig.add_axes([0.025, 0.06, 0.29, 0.88]),
        fig.add_axes([0.355, 0.06, 0.29, 0.88]),
        fig.add_axes([0.685, 0.06, 0.29, 0.88]),
    ]

    # 1. CV anchor: straight extrapolation from recent motion.
    ax = axes[0]
    panel_base(ax, 1, "CV anchor", BLUE, car, mmv)
    history_y = np.array([0.20, 0.40, 0.62, 0.88, 1.18])
    history_alpha = np.linspace(0.25, 0.85, len(history_y))
    for y, a in zip(history_y, history_alpha):
        ax.add_patch(Circle((3.0, y), 0.055, facecolor=BLUE, edgecolor="none", alpha=a, zorder=4))
    cv_y = np.array([2.25, 3.15, 4.05, 4.95, 5.85, 6.75])
    cv_x = np.full_like(cv_y, 3.0)
    ax.plot(cv_x, cv_y, color=BLUE, lw=2.2, ls=(0, (4, 3)), zorder=5)
    horizon_points(ax, cv_x, cv_y, BLUE)
    # A warning ring makes the uncorrected straight anchor's conflict tendency visible.
    ax.add_patch(Circle((3.0, 6.75), 0.26, facecolor="none", edgecolor=RED, lw=1.8, zorder=11))
    ax.add_patch(FancyArrowPatch((2.70, 6.40), (2.96, 6.67), arrowstyle="-|>",
                                 mutation_scale=10, color=RED, lw=1.4, zorder=11))

    # 2. HGB: horizon-specific residual vectors deform the CV anchor.
    ax = axes[1]
    panel_base(ax, 2, "HGB residuals", PURPLE, car, mmv)
    ax.plot(cv_x, cv_y, color=BLUE_SOFT, lw=1.7, ls=(0, (4, 3)), zorder=3)
    horizon_points(ax, cv_x, cv_y, BLUE_SOFT, size=25, zorder=4)
    hgb_x = np.array([2.98, 2.92, 2.77, 2.50, 2.15, 1.72])
    hgb_y = np.array([2.23, 3.02, 3.77, 4.44, 5.02, 5.48])
    for x0, y0, x1, y1 in zip(cv_x, cv_y, hgb_x, hgb_y):
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-|>",
                mutation_scale=8,
                color=PURPLE_SOFT,
                linewidth=1.3,
                zorder=6,
            )
        )
    curve(ax, [(3.0, 2.25), (2.92, 3.50), (2.45, 4.78), (1.72, 5.48)], PURPLE, lw=2.5, zorder=7)
    horizon_points(ax, hgb_x, hgb_y, PURPLE, size=31, zorder=9)

    # Compact histogram-to-tree icon: visual shorthand for HGB learning without prose.
    bx, by = 0.36, 7.45
    for i, h in enumerate([0.30, 0.55, 0.85, 0.60]):
        ax.add_patch(Rectangle((bx + i * 0.20, by), 0.13, h, facecolor=PURPLE_SOFT,
                               edgecolor=PURPLE, lw=0.7, zorder=12))
    ax.plot([1.33, 1.62, 1.91], [8.18, 7.75, 8.18], color=PURPLE, lw=1.0, zorder=12)
    for x, y in [(1.33, 8.18), (1.62, 7.75), (1.91, 8.18)]:
        ax.add_patch(Circle((x, y), 0.10, facecolor=WHITE, edgecolor=PURPLE, lw=1.0, zorder=13))
    ax.add_patch(FancyArrowPatch((1.15, 7.88), (1.27, 7.96), arrowstyle="-|>",
                                 mutation_scale=8, color=PURPLE, lw=1.0, zorder=13))

    # 3. ECR: endpoint-only shift, shorter terminal error, and acceptance mark.
    ax = axes[2]
    panel_base(ax, 3, "ECR correction", RED, car, mmv)
    curve(ax, [(3.0, 2.25), (2.90, 3.55), (2.40, 5.05), (1.72, 6.20)], PURPLE, lw=2.0, zorder=4)
    curve(ax, [(3.0, 2.25), (2.86, 3.60), (2.26, 5.35), (1.24, 6.78)], RED, lw=2.7, zorder=7)
    curve(ax, [(3.0, 2.25), (2.82, 3.68), (2.18, 5.48), (1.12, 6.92)],
          "#222222", lw=1.6, ls=(0, (4, 2)), zorder=9)

    base = (1.72, 6.20)
    corrected = (1.24, 6.78)
    observed = (1.12, 6.92)
    horizon_points(ax, [base[0]], [base[1]], PURPLE, size=38, zorder=10)
    horizon_points(ax, [corrected[0]], [corrected[1]], RED, size=40, zorder=11)
    horizon_points(ax, [observed[0]], [observed[1]], "#222222", size=40, zorder=12)
    ax.add_patch(FancyArrowPatch(base, corrected, arrowstyle="-|>", mutation_scale=11,
                                 color=RED, lw=2.0, zorder=13))

    # Long pre-correction error versus short accepted post-correction error.
    ax.plot([base[0], observed[0]], [base[1], observed[1]], color=PURPLE, lw=1.2,
            ls=(0, (3, 2)), zorder=8)
    ax.plot([corrected[0], observed[0]], [corrected[1], observed[1]], color=RED, lw=2.2, zorder=14)
    ax.add_patch(Circle((0.63, 6.55), 0.26, facecolor=WHITE, edgecolor=GREEN, lw=1.6, zorder=15))
    ax.plot([0.50, 0.60, 0.77], [6.54, 6.43, 6.67], color=GREEN, lw=2.0,
            solid_capstyle="round", zorder=16)

    # Sequence arrows between panels.
    for x0, x1 in [(0.316, 0.352), (0.646, 0.682)]:
        fig.add_artist(
            FancyArrowPatch(
                (x0, 0.50),
                (x1, 0.50),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=14,
                color="#5F6A70",
                lw=1.4,
                zorder=30,
            )
        )

    stem = OUT / "Figure7_HGB_ECR_principle"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    build()
