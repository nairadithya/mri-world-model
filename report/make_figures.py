#!/usr/bin/env python3
"""Build report figures from the canonical research records; no patient data used."""

from pathlib import Path
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
for name in ("IBMPlexSans-Regular.ttf", "IBMPlexSerif-Regular.ttf"):
    files = list(Path.home().glob(f".local/share/fonts/**/{name}"))
    if files:
        font_manager.fontManager.addfont(str(files[0]))
plt.rcParams.update(
    {
        "font.family": "IBM Plex Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)
BLUE, RED, GRAY = "#0F4D92", "#B64342", "#6c6c6c"


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(
            OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight", facecolor="white"
        )
    plt.close(fig)


def box(ax, x, y, w, h, text, color="#e8f0f8"):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        facecolor=color,
        edgecolor="#424242",
        linewidth=0.8,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops={"arrowstyle": "->", "lw": 1.1, "color": "#333333"},
    )


# The two branches are separate examples, not consecutive model components.
fig, ax = plt.subplots(figsize=(8.5, 3.8))
ax.set(xlim=(0, 1), ylim=(0, 1))
ax.axis("off")
ax.axvline(0.70, color=RED, ls="--", lw=1.4)
ax.text(0.69, 0.97, "Prediction cutoff", ha="right", va="top", color=RED, fontsize=9)
ax.text(0.02, 0.89, "AVAILABLE AT CUTOFF", weight="bold", fontsize=9)
ax.text(0.73, 0.89, "TARGET / EVALUATION", weight="bold", fontsize=9)
box(ax, 0.02, 0.39, 0.14, 0.20, "MRI sequences\n+ available\nclinical inputs")
box(ax, 0.19, 0.39, 0.13, 0.20, "Preprocessing")
arrow(ax, 0.16, 0.49, 0.19, 0.49)
box(ax, 0.36, 0.65, 0.13, 0.16, "JEPA image\nencoder")
box(ax, 0.53, 0.65, 0.14, 0.16, "JEPA temporal\npredictor")
box(ax, 0.36, 0.19, 0.13, 0.16, "Spatial state\nencoder")
box(ax, 0.53, 0.19, 0.14, 0.16, "Spatial dynamics\npredictor")
for y in (0.73, 0.27):
    arrow(ax, 0.32, 0.49, 0.36, y)
    arrow(ax, 0.49, y, 0.53, y)
    arrow(ax, 0.67, y, 0.74, 0.63)
box(
    ax,
    0.74,
    0.55,
    0.23,
    0.16,
    "Predicted representation\nor anatomical readout",
    "#f0f0f0",
)
box(ax, 0.74, 0.34, 0.23, 0.12, "Compare for scoring", "#f0f0f0")
box(ax, 0.74, 0.09, 0.23, 0.17, "Future MRI / masks\nfor target and scoring", "#f7e9e9")
arrow(ax, 0.855, 0.55, 0.855, 0.46)
arrow(ax, 0.855, 0.26, 0.855, 0.34)
ax.text(
    0.46,
    0.47,
    "Separate experimental branches",
    fontsize=8,
    style="italic",
    ha="center",
    va="center",
)
save(fig, "figure_4_1_architecture")

fig, ax = plt.subplots(figsize=(8.5, 3.2))
ax.set(xlim=(0, 1), ylim=(0, 1))
ax.axis("off")
ax.plot([0.64, 0.64], [0.46, 0.91], color=RED, ls="--", lw=1.4)
ax.text(0.64, 0.95, "Prediction cutoff", ha="center", color=RED, fontsize=9)
ax.plot([0.08, 0.91], [0.65, 0.65], color="#444444", lw=1.4)
for x, label in [
    (0.14, "Earlier visit"),
    (0.45, "Current visit"),
    (0.86, "Next observed visit"),
]:
    ax.plot(x, 0.65, "o", color=BLUE if x < 0.64 else RED, ms=7)
    ax.text(x, 0.72, label, ha="center", fontsize=9)
ax.text(0.29, 0.53, "Available history + past timing", ha="center", fontsize=9)
ax.text(0.86, 0.53, "Target for scoring", ha="center", fontsize=9)
ax.text(0.05, 0.33, "History model", weight="bold")
ax.text(0.27, 0.33, "Earlier + current scans")
ax.text(0.05, 0.21, "Latest-only", weight="bold")
ax.text(0.27, 0.21, "Current scan; matched covariates where specified")
ax.text(0.05, 0.09, "Persistence", weight="bold")
ax.text(0.27, 0.09, "Copies current measurement or MRI-decoded state")
ax.text(
    0.99,
    0.01,
    "Future interval: experiment-specific; R37/R39 exclude it.",
    ha="right",
    fontsize=8,
    style="italic",
)
save(fig, "figure_4_2_timeline")

p1 = json.loads((ROOT / "info/p1_anatomy_reconciled.json").read_text())["cohorts"]
labels = ["Persistence", "Current image", "Mean image history", "JEPA state"]
rows = [
    tuple(
        p1[cohort][arm]["patient_uniform_relative_mae"]
        for cohort in ("LUMIERE", "SAILOR")
    )
    for arm in ("persistence", "current_image", "history_mean", "jepa_state")
]
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3), sharey=True, layout="constrained")
for ax, idx, cohort in zip(axes, (0, 1), ("LUMIERE", "SAILOR")):
    ax.axvline(1, color=GRAY, ls="--", lw=1)
    for y, (label, vals) in enumerate(zip(labels, rows)):
        value = vals[idx]
        ax.plot(
            value,
            y,
            marker="s" if y == 0 else "o",
            color=GRAY if y == 0 else BLUE,
            ms=6,
        )
        ax.annotate(
            f"{value:.3f}",
            (value, y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )
    ax.set(
        yticks=range(4),
        yticklabels=labels,
        title=cohort,
        xlim=(0.7, 5.5),
        xlabel="Persistence-relative MAE (lower is better)",
    )
    ax.tick_params(axis="y", length=0)
axes[0].invert_yaxis()
save(fig, "figure_7_1_p1_anatomy")


def forest(name, labels, data, title, subtitle, width=8.2):
    fig, ax = plt.subplots(
        figsize=(width, max(2.2, 1.4 + len(labels) * 0.4)), layout="constrained"
    )
    ax.axvline(0, color=GRAY, ls="--", lw=1)
    for i, (d, lo, hi) in enumerate(data):
        ax.plot([lo, hi], [i, i], color=BLUE, lw=1.7)
        ax.plot([lo, hi], [i, i], "|", color=BLUE, ms=8)
        ax.plot(d, i, "o", color=BLUE, ms=5)
    ax.set(
        yticks=range(len(labels)),
        yticklabels=labels,
        xlabel="Paired CRPS difference, log1p(mm³) scale (lower is better)",
        title=title,
    )
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.set_title(title + "\n" + subtitle, fontsize=10, pad=9)
    save(fig, name)


r37 = json.loads((ROOT / "info/r37_full_metrics.json").read_text())[
    "primary_comparisons"
]
keys = ["H-L", "H-P", "H-E", "H-R", "H-G"]
forest(
    "figure_7_2_r37_history",
    [
        "History − latest scan",
        "History − decoded persistence",
        "History − empirical change",
        "History − ridge dynamics",
        "History − pooled earlier maps",
    ],
    [(r37[k]["difference"], *r37[k]["ci"]) for k in keys],
    "R37: history comparisons",
    "Points: differences; lines: adjusted 99% patient-bootstrap intervals",
)
r39 = json.loads((ROOT / "info/r39_treatment_metrics.json").read_text())["comparisons"][
    "HT_vs_H"
]
forest(
    "figure_7_3_r39_treatment",
    ["History + treatment − history"],
    [(r39["difference"], *r39["ci99"])],
    "R39: incremental treatment status",
    "Point: difference; line: 99% patient-bootstrap interval",
)
