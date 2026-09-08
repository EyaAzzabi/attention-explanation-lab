"""Figures for the README, rendered from `results/` in both light and dark.

Design decisions, and why each one is what it is:

  * **Form.** The job is comparing a magnitude across an ordered set of encoders,
    with an uncertainty attached to each. That is a dot plot with error bars, not a
    bar chart: bars imply an area proportional to the value and a meaningful zero,
    and they make the error bar the least visible thing in the figure when it is the
    most important thing here.

  * **Colour by job.** Figure 1 carries two series, this reproduction and the value
    the paper reports, so the palette is categorical and the two slots are the first
    two of a validated order. Figure 2 also carries two, the gradient and
    leave-one-out measures. Encoders are the axis, never a colour: colour follows
    the series, not the rank.

  * **Two renders, not one flip.** A PNG cannot respond to the reader's theme, so
    each figure is rendered twice against its own surface, and the README selects
    between them with `<picture>`. The dark render is stepped for the dark surface rather
    than being an inverted copy.

  * **No hover layer.** These are static images in a README. Every number is
    therefore direct-labelled, and the results tables above them are the table view.

Palette validated with the data-viz checker: adjacent CVD delta-E 15.8 light and
15.5 dark against a >= 8 target, normal-vision 17.8 and 16.8 against a >= 15 floor,
and both slots clear 3:1 contrast on their surface. The de-emphasis grey is
deliberately achromatic and is not a categorical slot.

    python -m src.figures
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .report import PAPER_TAU_G, aggregate, load_results

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"

ENCODER_ORDER = ["average", "cnn", "bilstm", "transformer"]
ENCODER_LABEL = {
    "average": "Average\n(no context)",
    "cnn": "CNN\n(local)",
    "bilstm": "BiLSTM\n(sequential)",
    "transformer": "Transformer\n(all-to-all)",
}

THEMES = {
    "light": {
        "surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e",
        "muted": "#8a8a85", "grid": "#e4e3df",
        "series": ["#2a78d6", "#eb6834"],
    },
    "dark": {
        "surface": "#1a1a19", "primary": "#ffffff", "secondary": "#c3c2b7",
        "muted": "#8a8a85", "grid": "#333331",
        "series": ["#3987e5", "#d95926"],
    },
}


def _style(theme: dict) -> None:
    plt.rcParams.update({
        "figure.facecolor": theme["surface"],
        "axes.facecolor": theme["surface"],
        "savefig.facecolor": theme["surface"],
        "text.color": theme["primary"],
        "axes.labelcolor": theme["secondary"],
        "xtick.color": theme["secondary"],
        "ytick.color": theme["secondary"],
        "font.size": 10,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.edgecolor": theme["grid"],
        "grid.color": theme["grid"],
        "grid.linewidth": 0.8,
    })


def _series(grouped, dataset: str, field: str):
    means, stds, present = [], [], []
    for enc in ENCODER_ORDER:
        runs = grouped.get((dataset, enc))
        if not runs:
            continue
        a = aggregate(runs, field)
        present.append(enc)
        means.append(a["mean"])
        stds.append(0.0 if np.isnan(a["std"]) else a["std"])
    return present, np.array(means), np.array(stds)


def figure_encoders(grouped, dataset: str, mode: str) -> Path:
    """Kendall tau against gradient importance, by encoder, with the paper's value."""
    theme = THEMES[mode]
    _style(theme)
    present, means, stds = _series(grouped, dataset, "tau_gradient")
    y = np.arange(len(present))[::-1]

    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    ax.xaxis.grid(True, linewidth=0.8)
    ax.set_axisbelow(True)

    # The paper's reported values first, so the measurement sits on top of them.
    paper_y, paper_x = [], []
    for yi, enc in zip(y, present):
        ref = PAPER_TAU_G.get((dataset, enc))
        if ref:
            paper_y.append(yi)
            paper_x.append(float(np.mean(ref)))
    if paper_x:
        ax.scatter(paper_x, paper_y, s=90, facecolors="none", linewidths=2.0,
                   edgecolors=theme["muted"], zorder=3,
                   label="reported in the paper (class mean)")

    ax.errorbar(means, y, xerr=stds, fmt="o", markersize=9, linewidth=2.0,
                capsize=5, color=theme["series"][0], ecolor=theme["series"][0],
                markeredgecolor=theme["surface"], markeredgewidth=2.0,
                zorder=4, label="this reproduction (5 seeds, mean +/- sd)")

    for yi, m, s in zip(y, means, stds):
        ax.annotate(f"{m:.3f}", (m, yi), textcoords="offset points",
                    xytext=(0, 13), ha="center", fontsize=9,
                    color=theme["primary"], fontweight="bold")

    ax.set_yticks(y, [ENCODER_LABEL[e] for e in present], fontsize=9)
    ax.set_xlabel("Kendall tau between attention weights and gradient importance")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(present) - 0.4)
    ax.set_title("Attention tracks gradients only without contextualisation",
                 fontsize=12, fontweight="bold", color=theme["primary"], loc="left", pad=14)
    leg = ax.legend(loc="center right", frameon=False, fontsize=9,
                    bbox_to_anchor=(1.0, 0.30))
    for text in leg.get_texts():
        text.set_color(theme["secondary"])

    fig.tight_layout()
    out = FIGURES / f"tau_by_encoder_{dataset}_{mode}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def figure_two_measures(grouped, dataset: str, mode: str) -> Path:
    """Gradient against leave-one-out: do the two importance measures agree?"""
    theme = THEMES[mode]
    _style(theme)
    present, g_mean, g_std = _series(grouped, dataset, "tau_gradient")
    _, l_mean, l_std = _series(grouped, dataset, "tau_loo")
    x = np.arange(len(present))
    offset = 0.16

    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    ax.yaxis.grid(True, linewidth=0.8)
    ax.set_axisbelow(True)

    for dx, mean, std, colour, label in (
        (-offset, g_mean, g_std, theme["series"][0], "vs gradient importance"),
        (+offset, l_mean, l_std, theme["series"][1], "vs leave-one-out importance"),
    ):
        ax.errorbar(x + dx, mean, yerr=std, fmt="o", markersize=9, linewidth=2.0,
                    capsize=5, color=colour, ecolor=colour,
                    markeredgecolor=theme["surface"], markeredgewidth=2.0,
                    label=label, zorder=4)
        for xi, m in zip(x + dx, mean):
            ax.annotate(f"{m:.2f}", (xi, m), textcoords="offset points",
                        xytext=(0, 12), ha="center", fontsize=8.5,
                        color=theme["primary"])

    ax.set_xticks(x, [ENCODER_LABEL[e] for e in present], fontsize=9)
    ax.set_ylabel("Kendall tau")
    ax.set_ylim(0, 1)
    ax.set_title("The two importance measures rank the Transformer differently",
                 fontsize=12, fontweight="bold", color=theme["primary"], loc="left", pad=14)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=9)
    for text in leg.get_texts():
        text.set_color(theme["secondary"])

    fig.tight_layout()
    out = FIGURES / f"two_measures_{dataset}_{mode}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    grouped = load_results()
    if not grouped:
        print("no results yet; run src.experiment first")
        return
    datasets = sorted({d for d, _ in grouped})
    for dataset in datasets:
        for mode in THEMES:
            print("wrote", figure_encoders(grouped, dataset, mode).name)
            print("wrote", figure_two_measures(grouped, dataset, mode).name)


if __name__ == "__main__":
    main()
