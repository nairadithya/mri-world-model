"""Build the full reviewer-facing plot set from the recorded metrics.

Usage:
    python3 info/plots/make_plots_review.py

Style: figures4papers house style (github.com/ChenLiu-1996/figures4papers,
CC BY-NC 4.0) — semantic blue/green/red palette, no top/right spines,
print-safe bars (black edges; hatches for sub-groups), 5-pt error caps,
y-limits tightened to the data range, dpi-300 png + pdf per panel.

All numbers come from metrics.json (the single source of truth — edit
numbers there, not here); every ratio/percentage stated in a title or
annotation is computed from it, never hardcoded.

Reads info/plots/metrics.json and writes a .png and .pdf for each of the
original 23 figures to info/plots/review_plots/. The data and figure set are
preserved; chart text and filenames use reviewer-facing descriptions.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "review_plots")

# ---------------------------------------------------------------- house style
plt.rcParams.update({
    "font.family": ["IBM Plex Sans", "DejaVu Sans", "sans-serif"],
    "font.size": 15,                 # compact watch-figures
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 2.0,
    "hatch.linewidth": 0.8,          # lighter print-safe hatches
    "legend.frameon": False,
    "svg.fonttype": "none",          # editable text in vector exports
})

PALETTE = {
    "blue_main": "#0F4D92",       # proposed / key method
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE", "green_2": "#AADCA9", "green_3": "#8BCF8B",
    "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342",
    "neutral": "#CFCECE",         # no-change reference
    "highlight": "#FFD700",       # single callout only
    "teal": "#42949E", "violet": "#9A4D8E",
}
DEFAULT_COLORS = ["#0F4D92", "#8BCF8B", "#B64342", "#42949E", "#9A4D8E", "#CFCECE"]

# Series semantics used throughout (keep consistent across panels):
#   blue_main    the proposed / showcased method
#   green_3      the improvement over a baseline in the same panel
#   red_strong   contrast model
#   neutral      no-change reference forecast
#   teal/violet  additional distinct arms (3rd/4th series)
BAR_EDGE, BAR_LW = "black", 1.5
INKD = dict(capsize=5, ecolor="black")
# white-framed legend for panels where it must sit over data / point clouds
LEGEND_BOX = dict(frameon=True, facecolor="white", framealpha=0.9,
                  edgecolor="0.85")
BOXED = dict(boxstyle="square,pad=0.15", fc="white", ec="none")

ANNO = 10.5          # value labels above bars
TICK_DENSE = 12.5    # tick label size for busy categorical axes


def save(fig, name, dpi=300, rect=None):
    """Write PNG and PDF versions under review_plots/."""
    os.makedirs(OUT, exist_ok=True)
    stem = os.path.join(OUT, name)
    if rect is None:
        fig.tight_layout(pad=2)
    else:
        fig.tight_layout(rect=rect, pad=2)
    fig.savefig(stem + ".png", dpi=dpi)
    fig.savefig(stem + ".pdf")
    plt.close(fig)


def load_metrics():
    with open(os.path.join(HERE, "metrics.json")) as f:
        return json.load(f)


# ------------------------------------------------------ training trajectory
def plot_full_cohort_training_curve(m, name):
    ep, val = m["epochs"], m["val"]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(ep, val, "o-", color=PALETTE["blue_main"], lw=2.5, ms=6,
            markeredgecolor=BAR_EDGE, markeredgewidth=1.0)
    ax.scatter([m["best_epoch"]], [m["best_val"]], s=160, zorder=5,
               color=PALETTE["green_3"], edgecolor=BAR_EDGE, linewidth=1.4)
    ax.axvline(m["best_epoch"], color=PALETTE["green_3"], ls="--", lw=1.4)
    ax.text(m["best_epoch"] + 0.8, max(val) * 0.72,
            f"lowest loss: epoch {m['best_epoch']}\n{m['best_val']:.4f}",
            color=PALETTE["green_3"], fontsize=ANNO, va="top", weight="bold")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Validation embedding-prediction loss")
    ax.set_title("MRI-history model: validation loss", fontsize=13)
    ax.set_xticks(ep)
    ax.set_ylim(0, max(val) * 1.18)
    fig.text(0.5, 0.025,
             f"Held-out patient test loss: {m['test_val']:.4f}",
             ha="center", fontsize=10.5, style="italic")
    save(fig, name, rect=[0, 0.08, 1, 1])


def plot_initial_pilot(m, name):
    """Show pilot objective learning and its no-change comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.5))
    epochs, vals = m["epochs"], m["validation_loss"]
    ax1.plot(epochs, vals, "o-", color=PALETTE["blue_main"], lw=2.5, ms=6,
             markeredgecolor=BAR_EDGE, markeredgewidth=1.0)
    ax1.scatter([epochs[-1]], [vals[-1]], s=140, zorder=5,
                color=PALETTE["green_3"], edgecolor=BAR_EDGE, linewidth=1.2)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("Validation embedding-prediction loss")
    ax1.set_title("The model learned its training objective")
    ax1.set_xticks(epochs)
    ax1.set_ylim(0, max(vals) * 1.2)
    ax1.text(0.04, 0.95,
             f"Feature variation stayed healthy\n"
             f"(spread {m['target_std']:.3f}; complexity "
             f"{m['target_effective_rank']:.1f})",
             transform=ax1.transAxes, va="top", fontsize=10.5,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8"))

    labels = [f"MRI-history model (n={m['evaluation_patients']})",
              f"no-change forecast (n={m['evaluation_patients']})"]
    values = [m["jepa_eval_error"], m["persistence_error"]]
    ax2.bar([0, 1], values,
            color=[PALETTE["red_strong"], PALETTE["neutral"]], width=0.56,
            edgecolor=BAR_EDGE, linewidth=BAR_LW)
    for x, v in enumerate(values):
        ax2.text(x, v + max(values) * 0.03, f"{v:.4f}", ha="center",
                 fontsize=ANNO)
    ax2.set_xticks([0, 1], labels)
    ax2.set_ylabel("Mean embedding prediction error")
    ax2.set_title("The pilot did not beat the no-change forecast")
    ax2.set_ylim(0, max(values) * 1.3)
    fig.suptitle("Small-cohort pilot: feasibility is not forecasting performance",
                 fontsize=14)
    save(fig, name, rect=[0, 0, 1, 0.91])


def plot_existing_anatomy_forecast(m, name):
    """Compare the original MRI-history representation with a no-change forecast."""
    cohorts = m["cohorts"]
    fig, axes = plt.subplots(1, len(cohorts), figsize=(10.4, 4.6), squeeze=False)
    for ax, cohort in zip(axes[0], cohorts):
        vals = [cohort["persistence_mae"], cohort["jepa_state_mae"]]
        bars = ax.bar([0, 1], vals,
                      color=[PALETTE["neutral"], PALETTE["red_strong"]],
                      width=0.58, edgecolor=BAR_EDGE, linewidth=BAR_LW)
        for bar, value, color in zip(bars, vals,
                                     [PALETTE["neutral"], PALETTE["red_strong"]]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    value - max(vals) * 0.07, f"{value:.3f}",
                    ha="center", va="top", fontsize=ANNO,
                    color="white" if color != PALETTE["neutral"] else "black",
                    weight="bold")
        ratio = cohort["jepa_state_mae"] / cohort["persistence_mae"]
        lo, hi = cohort["paired_difference_ci"]
        delta = cohort["paired_difference"]
        ax.text(0.5, 0.94,
                f"Error relative to no-change = {ratio:.2f}×\n"
                f"Difference = {delta:+.3f} [{lo:+.3f}, {hi:+.3f}]",
                transform=ax.transAxes, ha="center", va="top", fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8"))
        ax.set_xticks([0, 1], ["no-change\nforecast", "MRI-history\nrepresentation"])
        ax.set_ylabel("Patient-averaged log-volume error" if ax is axes[0] else "")
        ax.set_title(f"{cohort['name']}\n{cohort['patients']} patients, "
                     f"{cohort['eligible_pairs']} pairs", fontsize=12)
        ax.set_ylim(0, max(vals) * 1.55)
    fig.suptitle("Existing MRI-history model vs. no-change forecast", fontsize=14)
    fig.text(0.5, 0.015,
             "Lower is better. Intervals are 95% patient-resampling intervals.",
             ha="center", fontsize=9.5, style="italic")
    save(fig, name, rect=[0, 0.06, 1, 0.90])


def plot_lesion_repeat_relative_errors(m, name):
    """Compare each training repeat with a no-change forecast on both cohorts."""
    cohorts = m["cohorts"]
    fig, axes = plt.subplots(1, len(cohorts), figsize=(10.6, 4.8), sharey=True)
    for ax, cohort in zip(axes, cohorts):
        seeds = cohort["seeds"]
        x = np.arange(len(seeds))
        vals = [seed["relative_mae"] for seed in seeds]
        color = PALETTE["blue_main"] if cohort["name"] == "LUMIERE" \
            else PALETTE["red_strong"]
        ax.axhline(1.0, color=PALETTE["neutral"], ls="--", lw=1.8,
                   label="same error as no-change forecast")
        ax.plot(x, vals, "o-", color=color, lw=2.2, ms=8,
                markeredgecolor=BAR_EDGE, markeredgewidth=1)
        for idx, (xi, value) in enumerate(zip(x, vals)):
            # Keep the first/last labels inside their panel boundaries.
            if idx == 0:
                offset, align = (7, 10), "left"
            elif idx == len(vals) - 1:
                offset, align = (-7, 10), "right"
            else:
                offset, align = (0, 10), "center"
            ax.annotate(f"{value:.3f}", (xi, value), xytext=offset,
                        textcoords="offset points", ha=align, fontsize=10)
        ax.set_xticks(x, [f"Repeat {i + 1}" for i in x])
        ax.set_xlabel("Training repeat")
        ax.set_title(f"{cohort['name']} (n={cohort['patients']} patients)")
        ax.set_ylim(m["ratio_ylim"][0], m["ratio_ylim"][1])
        ax.grid(axis="y", color="0.88", lw=0.8)
    axes[0].set_ylabel("Error relative to no-change forecast\n(1.0 = same error; lower is better)")
    axes[-1].legend(fontsize=10, loc="upper right")
    fig.suptitle("Lesion-aware forecast error across three training repeats", fontsize=14)
    fig.text(0.5, 0.015,
             "All repeats use the same previously examined patient groups; they are not new validation cohorts.",
             ha="center", fontsize=9.5, style="italic")
    save(fig, name, rect=[0, 0.06, 1, 0.90])


def plot_lesion_repeat_intervals(m, name):
    """Show paired patient-level intervals for model-minus-baseline error."""
    cohorts = m["cohorts"]
    fig, axes = plt.subplots(1, len(cohorts), figsize=(10.6, 4.8), sharex=True)
    for ax, cohort in zip(axes, cohorts):
        seeds = cohort["seeds"]
        y = np.arange(len(seeds))
        color = PALETTE["blue_main"] if cohort["name"] == "LUMIERE" \
            else PALETTE["red_strong"]
        ax.axvline(0, color=PALETTE["neutral"], ls="--", lw=1.8)
        for yi, seed in zip(y, seeds):
            lo, hi = seed["difference_ci"]
            delta = seed["difference"]
            ax.errorbar(delta, yi,
                        xerr=[[delta - lo], [hi - delta]],
                        fmt="o", color=color, ecolor=color,
                        markersize=7, capsize=4, elinewidth=1.8,
                        markeredgecolor=BAR_EDGE, markeredgewidth=0.8)
            ax.annotate(f"{delta:+.3f} [{lo:+.3f}, {hi:+.3f}]",
                        (hi, yi), xytext=(7, 0), textcoords="offset points",
                        va="center", fontsize=9)
        ax.set_yticks(y, [f"Repeat {i + 1}" for i in y])
        ax.invert_yaxis()
        ax.set_title(f"{cohort['name']} (n={cohort['patients']} patients)")
        ax.grid(axis="x", color="0.88", lw=0.8)
    axes[0].set_xlim(m["difference_xlim"])
    axes[0].set_xlabel("Lesion-aware error − no-change error")
    axes[1].set_xlabel("Lesion-aware error − no-change error")
    fig.supylabel("Training repeat")
    fig.suptitle("Paired error differences with 95% patient-level intervals", fontsize=14)
    fig.text(0.5, 0.015,
             "Negative values favor the lesion-aware model. An interval crossing zero does not resolve a difference.",
             ha="center", fontsize=9.5, style="italic")
    save(fig, name, rect=[0.04, 0.06, 1, 0.90])


def plot_training_continuation(m, name):
    ep, val = m["epochs"], m["val"]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(ep, val, "o-", color=PALETTE["red_strong"], lw=2.5, ms=6,
            markeredgecolor=BAR_EDGE, markeredgewidth=1.0, label="continued training")
    ax.axhline(m["champion_val"], color=PALETTE["blue_main"], ls="--", lw=1.6,
               label=f"earlier best model {m['champion_val']:.4f}")
    ax.scatter([m["leg_best_epoch"]], [m["leg_best_val"]], s=150, zorder=5,
               color=PALETTE["highlight"], edgecolor=BAR_EDGE, linewidth=1.2)
    ax.text(2.0, 0.0295,
            f"best in this continuation: epoch {m['leg_best_epoch']}\nloss = {m['leg_best_val']:.4f}",
            color=PALETTE["red_strong"], fontsize=ANNO, va="top")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Validation embedding-prediction loss")
    ax.set_title("Continued training increased validation error")
    ax.set_xticks(ep)
    ax.set_ylim(0.004, 0.033)
    ax.legend(fontsize=11, loc="lower right", **LEGEND_BOX)
    save(fig, name)


# -------------------------------------------- response-training comparison
def plot_response_training_effect(m, name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.4))
    # Left: compare the original model with response-supervised training.
    ax1.bar([0, 1], [m["dynamics_champion"], m["dynamics_aux"]],
            color=[PALETTE["blue_main"], PALETTE["red_strong"]],
            width=0.55, edgecolor=BAR_EDGE, linewidth=BAR_LW)
    ax1.axhline(m["persistence_ref"], color=PALETTE["neutral"], ls="--", lw=1.6,
                label=f"no-change forecast {m['persistence_ref']:.3f}")
    ax1.set_xticks([0, 1], ["original model", "with response training"])
    ax1.set_ylabel("Embedding-prediction error")
    ax1.set_title("Prediction error rises with response training", fontsize=11.5)
    ax1.set_ylim(0, 0.028)
    for x, v in [(0, m["dynamics_champion"]), (1, m["dynamics_aux"])]:
        ax1.text(x, v + 0.0007, f"{v:.4f}", ha="center", fontsize=ANNO)
    ax1.legend(fontsize=10.5, loc="upper left", **LEGEND_BOX)
    # Right: compare one patient split with patient-separated cross-validation.
    pre = [m["f1_hero_champion"], m["f1_cv_champion"]]
    post = [m["f1_hero_aux"], m["f1_cv_aux"]]
    pre_err = [0, m["f1_cv_champion_sd"]]
    post_err = [0, m["f1_cv_aux_sd"]]
    x = list(range(2))
    w = 0.38
    ax2.bar([i - w / 2 for i in x], pre, w, yerr=pre_err, label="original image encoder",
            color=PALETTE["blue_main"], edgecolor=BAR_EDGE, linewidth=BAR_LW, **INKD)
    ax2.bar([i + w / 2 for i in x], post, w, yerr=post_err,
            label="response-trained image encoder", color=PALETTE["red_strong"], hatch="/",
            edgecolor=BAR_EDGE, linewidth=BAR_LW, **INKD)
    ax2.axhline(m["sota_f1"], color=PALETTE["neutral"], ls=":", lw=1.6,
                label=f"reported comparison score {m['sota_f1']:.2f}")
    ax2.set_xticks(x, ["single patient split", "held-out patients"])
    ax2.set_ylabel("Class-balanced F1 score")
    ax2.set_title("Single-split gain fades across patients", fontsize=11.5)
    ax2.set_ylim(0, 0.70)        # headroom so the legend clears the labels
    for i in x:
        ax2.text(i - w / 2, pre[i] + pre_err[i] + 0.02, f"{pre[i]:.3f}",
                 ha="center", fontsize=ANNO - 1)
        ax2.text(i + w / 2, post[i] + post_err[i] + 0.02, f"{post[i]:.3f}",
                 ha="center", fontsize=ANNO - 1)
    ax2.legend(fontsize=10, loc="upper right", **LEGEND_BOX)
    fig.suptitle("Response training: no consistent classification gain", fontsize=13)
    save(fig, name, rect=[0, 0, 1, 0.94])


# -------------------------------------------------------------- cross-site
def plot_second_cohort_transfer(m, name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.4))
    # Left: compare the MRI-history model with the no-change forecast.
    dj, dp = m["dynamics_jepa"], m["dynamics_persist"]
    ax1.bar([0, 1], [dj, dp],
            color=[PALETTE["red_strong"], PALETTE["neutral"]],
            width=0.55, edgecolor=BAR_EDGE, linewidth=BAR_LW)
    ax1.set_xticks([0, 1], ["MRI-history model", "no-change forecast"])
    ax1.set_ylabel("Mean embedding-prediction error")
    ax1.set_title(f"Next-scan prediction: no-change is {dj / dp:.1f}× better")
    ax1.set_ylim(0, max(dj, dp) * 1.35)
    for x, v in [(0, dj), (1, dp)]:
        ax1.text(x, v + 0.001, f"{v:.4f}", ha="center", fontsize=ANNO)
    # Right: readouts transfer ~= in-domain (zero SAILOR training) -> green.
    f1 = [m["f1_lumiere_cv"], m["f1_sailor_transfer"]]
    ax2.bar([0, 1], f1, color=[PALETTE["blue_main"], PALETTE["green_3"]],
            width=0.55, edgecolor=BAR_EDGE, linewidth=BAR_LW)
    ax2.axhline(m["majority_f1"], color=PALETTE["neutral"], ls="--", lw=1.6,
                label=f"Most-common-class baseline {m['majority_f1']:.2f}")
    ax2.set_xticks([0, 1], ["LUMIERE\nin-domain", "SAILOR\ntransfer"])
    ax2.set_ylabel("Class-balanced F1 score")
    ax2.set_title("Exploratory response-category scores")
    ax2.set_ylim(0, 0.5)
    for i, v in enumerate(f1):
        ax2.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=ANNO)
    ax2.text(1, 0.42,
             f"Error discrimination\nROC area {m['surprise_auc_sailor']:.3f}\n({m['surprise_n']} pairs)",
             ha="center", va="bottom", fontsize=9.5, color="black",
             weight="bold")
    ax2.legend(fontsize=10.5, loc="upper left", **LEGEND_BOX)
    fig.suptitle("Transfer to a second cohort: exploratory representation results",
                 fontsize=13)
    save(fig, name, rect=[0, 0, 1, 0.94])


# ------------------------------------------------------- gap-stratified bins
def plot_second_cohort_interval_groups(m, name):
    bins, pairs, jepa, persist = m["bins"], m["pairs"], m["jepa"], m["persist"]
    x = list(range(len(bins)))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.bar([i - w / 2 for i in x], jepa, w, label="MRI-history predictor",
           color=PALETTE["red_strong"], edgecolor=BAR_EDGE, linewidth=BAR_LW)
    ax.bar([i + w / 2 for i in x], persist, w, label="no-change forecast",
           color=PALETTE["neutral"], edgecolor=BAR_EDGE, linewidth=BAR_LW)
    ax.set_xticks(x, [f"{b}\n(n={p})" for b, p in zip(bins, pairs)])
    ax.tick_params(axis="x", labelsize=TICK_DENSE)
    ax.set_ylabel("Mean embedding-prediction error")
    ax.set_title(f"Second-cohort prediction by recorded visit interval "
                 f"(n={m['n_pairs']}, median interval {m['gap_median']} days)",
                 fontsize=13)
    for i in x:
        ax.text(i - w / 2, jepa[i] + 0.0009, f"{jepa[i]:.4f}",
                ha="center", fontsize=ANNO - 1)
        ax.text(i + w / 2, persist[i] + 0.0009, f"{persist[i]:.4f}",
                ha="center", fontsize=ANNO - 1)
    ax.set_ylim(0, max(jepa) * 1.3)
    ax.legend(fontsize=11, loc="upper right")
    save(fig, name)


def plot_interval_aware_model(m, name):
    bins, pairs, champ, gap = m["bins"], m["pairs"], m["jepa"], m["gap_head"]
    persist = m["persist"]
    x = list(range(len(bins)))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    ax.bar([i - w / 2 for i in x], champ, w, label="original MRI-history predictor",
           color=PALETTE["red_strong"], edgecolor=BAR_EDGE, linewidth=BAR_LW)
    ax.bar([i + w / 2 for i in x], gap, w, label="interval-aware predictor",
           color=PALETTE["blue_main"], edgecolor=BAR_EDGE, linewidth=BAR_LW)
    ax.plot(x, persist, "D-", color="black", ms=6, lw=1.4,
            label="no-change forecast")
    ax.set_xticks(x, [f"{b}\n(n={p})" for b, p in zip(bins, pairs)])
    ax.tick_params(axis="x", labelsize=TICK_DENSE)
    ax.set_ylabel("Mean embedding-prediction error")
    ax.set_title("Using the visit interval helps, but no-change remains stronger",
                 fontsize=13)
    for i in x:
        pct = (champ[i] - gap[i]) / champ[i] * 100
        ax.text(i - w / 2, champ[i] + 0.0009, f"{champ[i]:.4f}",
                ha="center", fontsize=ANNO - 1)
        ax.text(i + w / 2, gap[i] + 0.0009, f"{gap[i]:.4f}",
                ha="center", fontsize=ANNO - 1)
        ax.text(i, max(champ[i], gap[i]) + 0.003, f"-{pct:.0f}%",
                ha="center", fontsize=ANNO, color=PALETTE["blue_main"],
                weight="bold")
        ax.text(i, persist[i] + 0.0012, f"{persist[i]:.4f}", ha="center",
                fontsize=9, color="black")
    ax.set_ylim(0, max(champ) * 1.5)   # headroom so the legend clears -27%
    ax.legend(fontsize=11, loc="upper right")
    save(fig, name)


def plot_velocity_model(m, name):
    bins, pairs = m["bins"], m["pairs"]
    champ, persist, cond = m["jepa"], m["persist"], m["field_cond"]
    x = list(range(len(bins)))
    w = 0.3
    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    ax.bar([i - w for i in x], champ, w, label="original MRI-history predictor",
           color=PALETTE["red_strong"], edgecolor=BAR_EDGE, linewidth=BAR_LW)
    ax.bar(x, cond, w, label="velocity-based predictor",
           color=PALETTE["blue_main"], edgecolor=BAR_EDGE, linewidth=BAR_LW)
    mid = [i + w for i in x]        # right of both bars -> labels stay clear
    ax.plot(mid, persist, "D-", color="black", ms=6, lw=1.4,
            label="no-change forecast")
    ax.set_xticks(x, [f"{b}\n(n={p})" for b, p in zip(bins, pairs)])
    ax.tick_params(axis="x", labelsize=TICK_DENSE)
    ax.set_ylabel("Mean embedding-prediction error")
    ratio = np.median(np.array(champ) / np.array(cond))
    ax.set_title(f"Velocity-based predictor reduces error ~{ratio:.0f}×; "
                 "no-change is still competitive", fontsize=13)
    for i in x:
        ax.text(i - w, champ[i] + 0.0009, f"{champ[i]:.4f}",
                ha="center", fontsize=ANNO - 1)
        ax.text(i, max(cond[i], persist[i]) + 0.0012, f"{cond[i]:.4f}",
                ha="center", fontsize=ANNO - 1, color=PALETTE["blue_main"])
        ax.text(i + w, persist[i] + 0.0009, f"{persist[i]:.4f}",
                ha="center", fontsize=9, color="black")
    ax.set_ylim(0, max(champ) * 1.32)
    ax.legend(fontsize=11, loc="upper right")
    save(fig, name)


def plot_treatment_context_comparison(m, name):
    bins, pairs = m["bins"], m["pairs"]
    persist, cond, uncond = m["persist"], m["field_cond"], m["field_uncond"]
    x = list(range(len(bins)))
    w = 0.22
    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    ax.bar([i - w for i in x], cond, w, label="using recorded treatment phase",
           color=PALETTE["blue_main"], edgecolor=BAR_EDGE, linewidth=BAR_LW)
    ax.bar(x, uncond, w, label="without treatment information", hatch=".",
           color=PALETTE["blue_secondary"], edgecolor=BAR_EDGE, linewidth=BAR_LW)
    ax.bar([i + w for i in x], persist, w, label="no-change forecast",
           color=PALETTE["neutral"], edgecolor=BAR_EDGE, linewidth=BAR_LW)
    ax.set_xticks(x, [f"{b}\n(n={p})" for b, p in zip(bins, pairs)])
    ax.tick_params(axis="x", labelsize=TICK_DENSE)
    ax.set_ylabel("Mean embedding-prediction error")
    ax.set_title("Treatment information made little difference",
                 fontsize=13)
    for i in x:
        if abs(cond[i] - uncond[i]) < 0.0005:
            ax.text(i - w / 2, max(cond[i], uncond[i]) + 0.00025,
                    f"{cond[i]:.4f} (tie)", ha="center", fontsize=ANNO - 1.5,
                    color=PALETTE["blue_main"])
        else:
            ax.text(i - w, cond[i] + 0.00025, f"{cond[i]:.4f}",
                    ha="center", fontsize=ANNO - 1.5)
            ax.text(i, uncond[i] + 0.00025, f"{uncond[i]:.4f}",
                    ha="center", fontsize=ANNO - 1.5)
        ax.text(i + w, persist[i] + 0.00025, f"{persist[i]:.4f}",
                ha="center", fontsize=ANNO - 1.5)
    ax.set_ylim(0, 0.0085)       # headroom so the legend clears the labels
    ax.legend(fontsize=10.5, loc="upper right")
    save(fig, name)


# ------------------------------------------------------------ latent clouds
def plot_cohort_feature_distribution(m, name):
    lum = np.array(m["lumiere_xy"])
    sai = np.array(m["sailor_xy"])
    e1, e2 = m["explained_pct"]
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.scatter(lum[:, 0], lum[:, 1], s=16, alpha=0.5, color=PALETTE["blue_main"],
               label=f"LUMIERE visits (n={m['n_lumiere']})")
    ax.scatter(sai[:, 0], sai[:, 1], s=20, alpha=0.6, color=PALETTE["red_strong"],
               label=f"SAILOR visits (n={m['n_sailor']})")
    ax.set_xlabel(f"Feature projection axis 1 ({e1}% of variation)")
    ax.set_ylabel(f"Feature projection axis 2 ({e2}% of variation)")
    ax.set_title("Cohort feature distributions differ", fontsize=13)
    ax.legend(fontsize=11, loc="upper left", **LEGEND_BOX)
    save(fig, name)


def plot_aligned_feature_distribution(m, name):
    lum = np.array(m["lumiere_xy"])
    sai = np.array(m["sailor_xy"])
    e1, e2 = m["explained_pct"]
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.scatter(lum[:, 0], lum[:, 1], s=16, alpha=0.5, color=PALETTE["blue_main"],
               label=f"LUMIERE visits (n={m['n_lumiere']})")
    ax.scatter(sai[:, 0], sai[:, 1], s=20, alpha=0.6, color=PALETTE["green_3"],
               label=f"SAILOR visits, distribution-aligned (n={m['n_sailor']})")
    ax.set_xlabel(f"Feature projection axis 1 ({e1}% of variation)")
    ax.set_ylabel(f"Feature projection axis 2 ({e2}% of variation)")
    ax.set_title("Feature alignment leaves cohort differences", fontsize=13)
    ax.legend(fontsize=11, loc="upper left", **LEGEND_BOX)
    save(fig, name)


def plot_feature_distribution_matching(m, name):
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    cats = ["MRI-history predictor", "no-change forecast"]
    x = [0, 1]
    w = 0.26
    ax.bar(x[0] - w, m["head_unaligned"], w, label="predictor, original features",
           color=PALETTE["red_strong"], edgecolor=BAR_EDGE, linewidth=BAR_LW)
    ax.bar(x[0], m["head_aligned"], w, label="predictor, distribution-aligned",
           color=PALETTE["green_3"], edgecolor=BAR_EDGE, linewidth=BAR_LW)
    ax.bar(x[0] + w, m["head_transductive"], w, label="predictor, evaluation-distribution diagnostic",
           color=PALETTE["blue_secondary"], edgecolor=BAR_EDGE, linewidth=BAR_LW)
    ax.bar(x[1] - w / 2, m["persist_unaligned"], w, label="no-change, original features",
           color=PALETTE["neutral"], edgecolor=BAR_EDGE, linewidth=BAR_LW)
    ax.bar(x[1] + w / 2, m["persist_aligned"], w, label="no-change, aligned",
           hatch=".", color=PALETTE["neutral"], edgecolor=BAR_EDGE,
           linewidth=BAR_LW)
    ax.set_xticks(x, cats)
    ax.set_ylabel("Mean embedding-prediction error")
    ratio = m["head_unaligned"] / m["head_aligned"]
    ax.set_title(f"Feature matching reduces error {ratio:.1f}×\nwithout retraining",
                 fontsize=13)
    for v, xp in [(m["head_unaligned"], x[0] - w), (m["head_aligned"], x[0]),
                  (m["head_transductive"], x[0] + w)]:
        ax.text(xp, v + 0.0008, f"{v:.4f}", ha="center", fontsize=ANNO - 1)
    for v, xp in [(m["persist_unaligned"], x[1] - w / 2),
                  (m["persist_aligned"], x[1] + w / 2)]:
        ax.text(xp, v + 0.0008, f"{v:.4f}", ha="center", fontsize=ANNO - 1)
    pct = 100 * (1 - m["head_aligned"] / m["head_unaligned"])
    ax.annotate(f"-{pct:.0f}% ({ratio:.1f}x)", xy=(x[0], m["head_aligned"]),
                xytext=(0.5, 0.029), ha="center", fontsize=11,
                color=PALETTE["green_3"], weight="bold",
                arrowprops=dict(arrowstyle="->", color=PALETTE["green_3"],
                                lw=1.5))
    ax.text(0.5, 0.021, "model weights unchanged\nremaining cohort differences",
            ha="center", fontsize=9.5, color="dimgray")
    ax.set_ylim(0, 0.034)
    ax.legend(fontsize=9.5, loc="upper right")
    save(fig, name)


def plot_image_processing_effect(m, name):
    slots, n = m["slots"], m["n"]
    raw, mni = m["raw_med"], m["mni_med"]
    x = list(range(len(slots)))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    ax.bar([i - w / 2 for i in x], raw, w, label="before standard-space processing",
           color=PALETTE["red_strong"], edgecolor=BAR_EDGE, linewidth=BAR_LW)
    ax.bar([i + w / 2 for i in x], mni, w, label="after standard-space processing",
           color=PALETTE["blue_main"], edgecolor=BAR_EDGE, linewidth=BAR_LW)
    ax.set_xticks(x, [f"{s}\n(n={p})" for s, p in zip(slots, n)])
    ax.tick_params(axis="x", labelsize=TICK_DENSE)
    ax.set_ylabel("Median voxel-wise image change")
    ratio = np.median(np.array(raw) / np.array(mni))
    ax.set_title(f"Standard-space processing reduces measured image change by ~{ratio:.1f}×",
                 fontsize=13)
    for i in x:
        ax.text(i - w / 2, raw[i] + 0.015, f"{raw[i]:.3f}",
                ha="center", fontsize=ANNO - 1)
        ax.text(i + w / 2, mni[i] + 0.015, f"{mni[i]:.3f}",
                ha="center", fontsize=ANNO - 1)
        ax.text(i, max(raw[i], mni[i]) + 0.06, f"{raw[i] / mni[i]:.1f}x",
                ha="center", fontsize=ANNO - 0.5, color=PALETTE["blue_main"],
                weight="bold")
    ax.set_ylim(0, max(raw) * 1.3)
    ax.legend(fontsize=11, loc="upper right")
    save(fig, name)


# ------------------------------------------- reprocessed-SAILOR comparisons
def _four_gate_bars(ax, d, labels, ylabel, title, ymax=None):
    """Compare original and reprocessed scans with a no-change reference."""
    x = [0, 1, 3, 4]
    vals = [d["deriv_jepa"], d["deriv_persist"],
            d["dynamics_jepa"], d["dynamics_persist"]]
    cols = [PALETTE["red_strong"], PALETTE["neutral"],
            PALETTE["red_strong"], PALETTE["neutral"]]
    hts = [None, None, "/", "."]
    for xi, v, c, h in zip(x, vals, cols, hts):
        ax.bar(xi, v, width=0.55, color=c, hatch=h,
               edgecolor=BAR_EDGE, linewidth=BAR_LW)
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", labelsize=TICK_DENSE)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12.5)
    if ymax is None:
        ymax = max(vals) * 1.3
    ax.set_ylim(0, ymax)
    for xi, v in zip(x, vals):
        ax.text(xi, v + ymax * 0.018, f"{v:.4f}", ha="center", fontsize=ANNO - 1)
    return vals


def plot_reprocessed_second_cohort(m, name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.4))
    _four_gate_bars(
        ax1, m, ["original scans\nmodel", "original scans\nno-change",
                 "reprocessed scans\nmodel", "reprocessed scans\nno-change"],
        "Mean embedding-prediction error",
        "The no-change forecast remains stronger in both conditions")
    f1 = [m["deriv_transfer_f1"], m["transfer_f1"]]
    ax2.bar([0, 1], f1,
            color=[PALETTE["blue_main"], PALETTE["red_strong"]],
            width=0.55, hatch=[None, "/"], edgecolor=BAR_EDGE,
            linewidth=BAR_LW)
    ax2.set_xticks([0, 1], ["original scans", "reprocessed scans"])
    ax2.tick_params(axis="x", labelsize=TICK_DENSE)
    ax2.set_ylabel("Class-balanced F1 score")
    maj = m["transfer_majority"]
    ax2.set_title(f"Exploratory response score (majority {maj:.2f}; "
                  f"error area-under-curve {m['deriv_surprise_auc']:.2f} → {m['surprise_auc']:.2f})",
                  fontsize=12.5)
    ax2.set_ylim(0, 0.5)
    for i, v in enumerate(f1):
        ax2.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=ANNO)
    fig.suptitle("Reprocessing the second cohort changes model behavior",
                 fontsize=13)
    fig.text(0.5, 0.01,
             "Response-category mapping in this cohort has not been independently verified.",
             ha="center", fontsize=8.5, style="italic")
    save(fig, name, rect=[0, 0, 1, 0.94])


def plot_reprocessed_forecast(m, name):
    r = m["sailor_reprocessed"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.4))
    _four_gate_bars(
        ax1, r, ["original scans\nmodel", "original scans\nno-change",
                 "reprocessed scans\nmodel", "reprocessed scans\nno-change"],
        f"Mean error (all pairs, n={r['n_pairs']})",
        f"Model error falls {r['deriv_jepa']:.4f} → {r['dynamics_jepa']:.4f}")
    dj, dp = r["deriv_jepa"] / r["deriv_persist"], \
        r["dynamics_jepa"] / r["dynamics_persist"]
    ax2.bar([0, 1], [dj, dp],
            color=[PALETTE["blue_secondary"], PALETTE["blue_main"]],
            width=0.55, hatch=[None, "/"], edgecolor=BAR_EDGE,
            linewidth=BAR_LW)
    ax2.set_xticks([0, 1], ["derivatives", "reprocessed"])
    ax2.set_ylabel("Model error / no-change error")
    ax2.set_title(f"Relative error increases {dp / dj:.1f}× after reprocessing",
                  fontsize=12.5)
    ax2.set_ylim(0, max(dj, dp) * 1.3)
    for i, v in enumerate([dj, dp]):
        ax2.text(i, v + max(dj, dp) * 0.05, f"{v:.1f}x", ha="center",
                 fontsize=ANNO)
    fig.suptitle("Reprocessing lowers embedding error but does not improve forecasting",
                 fontsize=13)
    save(fig, name, rect=[0, 0, 1, 0.94])


def plot_processing_groups(m, name):
    lum = np.array(m["lumiere_xy"])
    der = np.array(m["deriv_xy"])
    rep = np.array(m["reproc_xy"])
    e1, e2 = m["explained_pct"]
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    ax.scatter(lum[:, 0], lum[:, 1], s=16, alpha=0.5, color=PALETTE["blue_main"],
               label=f"LUMIERE (n={m['n_lumiere']})")
    ax.scatter(der[:, 0], der[:, 1], s=20, alpha=0.6, color=PALETTE["red_strong"],
               label=f"SAILOR original derivatives (n={m['n_deriv']})")
    ax.scatter(rep[:, 0], rep[:, 1], s=20, alpha=0.6, color=PALETTE["violet"],
               label=f"SAILOR reprocessed, skull-stripped (n={m['n_reproc']})")
    ax.set_xlabel(f"Feature projection axis 1 ({e1}% of variation)")
    ax.set_ylabel(f"Feature projection axis 2 ({e2}% of variation)")
    ax.set_title("Reprocessing changes cohort features", fontsize=13)
    ax.legend(fontsize=10.5, loc="upper left", **LEGEND_BOX)
    save(fig, name)


def plot_four_group_processing_comparison(m, t, name):
    lum = np.array(t["lumiere_xy"])
    bet = np.array(t["betfix_xy"])
    der = np.array(t["deriv_xy"])
    rep = np.array(t["repro_xy"])
    e1, e2 = t["explained_pct"]
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    ax.scatter(lum[:, 0], lum[:, 1], s=16, alpha=0.5, color=PALETTE["blue_main"],
               label=f"LUMIERE (n={t['n_lumiere']})")
    ax.scatter(der[:, 0], der[:, 1], s=18, alpha=0.55, color=PALETTE["red_strong"],
               label=f"SAILOR original derivatives (n={t['n_deriv']})")
    ax.scatter(rep[:, 0], rep[:, 1], s=18, alpha=0.55, color=PALETTE["violet"],
               label=f"reprocessed, skull-stripped (n={t['n_repro']})")
    ax.scatter(bet[:, 0], bet[:, 1], s=22, alpha=0.65, color=PALETTE["teal"],
               label=f"reprocessed, skull retained (n={t['n_betfix']})")
    ax.set_xlabel(f"Feature projection axis 1 ({e1}% of variation)")
    ax.set_ylabel(f"Feature projection axis 2 ({e2}% of variation)")
    ax.set_title("Keeping skull tissue does not remove cohort differences", fontsize=13)
    ax.legend(fontsize=9.5, loc="upper left", **LEGEND_BOX)
    save(fig, name)


def plot_skull_tissue_sensitivity(m, name):
    d = m
    x = list(range(len(d["bins"])))
    w = 0.3
    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    ax.bar([i - w for i in x], d["jepa"], w, label="MRI-history predictor",
           color=PALETTE["red_strong"], edgecolor=BAR_EDGE, linewidth=BAR_LW)
    ax.plot(x, d["persist"], "D-", color="black", ms=6, lw=1.4,
            label="no-change forecast")
    ax.set_xticks(x, [f"{b}\n(n={p})" for b, p in zip(d["bins"], d["pairs"])])
    ax.tick_params(axis="x", labelsize=TICK_DENSE)
    ax.set_ylabel("Mean embedding-prediction error")
    ax.set_title("Keeping skull tissue does not close the second-cohort gap",
                 fontsize=13)
    for i in x:
        ax.text(i - w, d["jepa"][i] + 0.0009, f"{d['jepa'][i]:.4f}",
                ha="center", fontsize=ANNO - 1)
        ax.text(i, d["persist"][i] + 0.0009, f"{d['persist'][i]:.4f}",
                ha="center", fontsize=9, color="black")
    ax.set_ylim(0, max(d["jepa"]) * 1.3)
    ax.legend(fontsize=11, loc="upper right")
    save(fig, name)


# ---------------------------------------------------------------- gauntlet
def plot_response_feature_comparison(g, name):
    methods = g["methods"]
    labels = {
        "true": "Observed follow-up (upper bound)",
        "champ": "MRI-history forecast",
        "gap": "Visit-interval forecast",
        "field": "Treatment-aware forecast",
    }
    colors = {"true": PALETTE["neutral"], "champ": PALETTE["red_strong"],
              "gap": PALETTE["blue_main"], "field": PALETTE["teal"]}
    hts = {"true": ".", "champ": "/", "gap": "\\", "field": "x"}
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(16, 6.0))
    fig.suptitle("MRI-history features and clinical response classification",
                 fontsize=14, fontweight="bold")

    # A. bars with patient-level cross-validation spread and exploratory transfer.
    groups = [("LUMIERE\nheld-out patients", g["lum_cv_f1"], g["lum_cv_sd"]),
              ("SAILOR\nheld-out patients", g["sai_cv_f1"], g["sai_cv_sd"]),
              ("SAILOR\ntransfer", g["sai_transfer_f1"], [0.0] * 4)]
    x = np.arange(len(groups))
    w = 0.19
    for i, m in enumerate(methods):
        means = [grp[1][i] for grp in groups]
        errs = [grp[2][i] for grp in groups]
        axA.bar(x + (i - 1.5) * w, means, w, yerr=errs,
                label=labels[m], color=colors[m], hatch=hts[m],
                edgecolor=BAR_EDGE, linewidth=BAR_LW, **INKD)
    axA.set_xticks(x)
    axA.set_xticklabels([grp[0] for grp in groups])
    axA.tick_params(axis="x", labelsize=9)
    axA.set_ylabel("Class-balanced F1 score")
    axA.set_title("Response classification by feature type", fontsize=12.5)
    axA.set_ylim(0, 0.58)
    axA.hlines(g["a9_states_cv"], -0.4, 0.4, colors="k", ls="--", lw=1)
    axA.text(0.0, g["a9_states_cv"] + 0.011, f"{g['a9_states_cv']:.2f}",
             ha="center", fontsize=9, bbox=BOXED)
    axA.hlines(g["a12b_states_transfer"], 1.6, 2.4, colors="k", ls="--", lw=1)
    axA.text(2.0, g["a12b_states_transfer"] + 0.011,
             f"{g['a12b_states_transfer']:.2f}", ha="center", fontsize=9, bbox=BOXED)
    axA.legend(fontsize=10, loc="upper left")

    # B. dissociation scatter: error ratio (log) vs signal.
    # "true" has no JEPA error (it is the endpoint) -> parity ratio 1.0.
    EVALS = [("LUMIERE folds", "o", g["lum_cv_f1"], g["lum_cv_sd"], g["lum_err"]),
             ("SAILOR folds", "s", g["sai_cv_f1"], g["sai_cv_sd"], g["sai_err"]),
             ("SAILOR transfer", "^", g["sai_transfer_f1"], [None] * 4,
              g["sai_err"])]
    for tag, mk, f1s, sds, errs in EVALS:
        for j, m in enumerate(methods):
            e = errs[m]
            r = 1.0 if e is None else e / errs["persist"]
            sd = sds[j]
            axB.errorbar(r, f1s[j], yerr=sd if sd else 0.0, fmt=mk,
                         color=colors[m], ecolor=colors[m],
                         markersize=8, capsize=3, elinewidth=1,
                         label=f"{labels[methods[j]]} · {tag}" if tag == "LUMIERE folds"
                         else None)
    import matplotlib.lines as mlines
    meth = [mlines.Line2D([], [], color=colors[m], marker="o",
                           linestyle="None", markersize=7, label=labels[m])
            for j, m in enumerate(methods)]
    ev = [mlines.Line2D([], [], color="k", marker=mk, linestyle="None",
                         markersize=7, label=tag)
          for tag, mk, _, _, _ in EVALS]
    axB.legend(handles=meth + ev, frameon=True, fontsize=8.5, loc="upper left",
               ncol=2)
    axB.set_xscale("log")
    axB.set_xticks([0.7, 1, 2, 4, 7])       # explicit ticks: log minors collide
    axB.set_xticklabels(["0.7", "1", "2", "4", "7"])
    axB.minorticks_off()
    axB.axvline(1.0, color="k", linestyle=":", linewidth=1)
    axB.text(1.0, 0.075, "same error as no-change forecast", ha="center", fontsize=8.5,
             transform=axB.get_xaxis_transform())
    axB.set_xlabel("Embedding-error ratio: model / no-change forecast (log scale)")
    axB.set_ylabel("Class-balanced F1 score")
    axB.set_title("Prediction error compared with response signal", fontsize=12.5)
    axB.set_ylim(0.05, 0.48)
    fig.text(0.5, 0.015,
             "Dashed lines show the MRI-history reference. SAILOR response labels are not independently verified; transfer scores are exploratory.",
             ha="center", fontsize=9, style="italic")
    save(fig, name, dpi=600, rect=[0.02, 0.12, 0.98, 0.9])


# ----------------------------------------------------------- cross-site K
def plot_second_cohort_adaptation(g, name):
    ks = [0] + list(g["ks"])
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    style = {"JEPA": (PALETTE["blue_main"], "o", 2.6),
             "CNN-3D": (PALETTE["red_strong"], "s", 1.8),
             "CNN-2D": (PALETTE["red_2"], "^", 1.8),
             "CNN-2D-ROI": (PALETTE["violet"], "D", 1.8),
             "RAD": (PALETTE["neutral"], "v", 1.8)}
    display_names = {
        "JEPA": "Pretrained MRI-history model",
        "CNN-3D": "3D model from scratch",
        "CNN-2D": "2D model from scratch",
        "CNN-2D-ROI": "2D model with lesion crops",
        "RAD": "Volume and growth features",
    }
    for enc_name, enc in g["encoders"].items():
        ys = [enc["zero_shot"]] + list(enc["ks"])
        col, mk, lw = style.get(enc_name, ("#777777", "x", 1.2))
        ax.plot(ks, ys, mk + "-", color=col, lw=lw, ms=6,
                label=display_names.get(enc_name, enc_name))
    ax.axhline(g["majority_f1"], color=PALETTE["neutral"], ls="--", lw=1.6,
               label=f"Always predict the most common class ({g['majority_f1']:.2f})")
    ax.set_xlabel("SAILOR patients used for adaptation (0 = none)")
    ax.set_ylabel("Response-category class-balanced F1, patient-separated")
    ax.set_title("Exploratory second-cohort adaptation comparison")
    ax.set_xticks(ks)
    ax.set_ylim(0.08, 0.52)      # keep the 0.479 majority line inside
    ax.legend(fontsize=10, loc="upper left", ncol=2)
    ax.text(0.02, 0.145,
            "Scratch-trained image models and measured-volume features "
            "remain near the majority-class score.",
            transform=ax.transAxes, fontsize=8.5, style="italic",
            color="dimgray", va="top")
    ax.text(0.02, 0.085,
            "The patient groups are small; treat these curves as exploratory.",
            transform=ax.transAxes, fontsize=8.5, style="italic",
            color="dimgray", va="top")
    save(fig, name)


def main():
    m = load_metrics()
    plots = [
        (plot_initial_pilot, m["jepa_pilot"], "pilot_training_and_baseline"),
        (plot_full_cohort_training_curve, m["hero_leg1"], "full_cohort_training_curve"),
        (plot_existing_anatomy_forecast, m["jepa_anatomy_forecast"],
         "existing_model_anatomy_forecast"),
        (plot_lesion_repeat_relative_errors, m["lesion_seed_sensitivity"],
         "lesion_model_training_repeats"),
        (plot_lesion_repeat_intervals, m["lesion_seed_sensitivity"],
         "lesion_model_patient_intervals"),
        (plot_training_continuation, m["hero_leg2"], "continued_training_curve"),
        (plot_response_training_effect, m["run6"], "response_training_comparison"),
        (plot_second_cohort_transfer, m["sailor_transfer"], "second_cohort_transfer"),
        (plot_second_cohort_interval_groups, m["sailor_gap_bins"], "second_cohort_visit_intervals"),
        (plot_interval_aware_model, m["sailor_gap_bins"], "interval_aware_prediction"),
        (plot_velocity_model, m["sailor_gap_bins"], "velocity_model_comparison"),
        (plot_treatment_context_comparison, m["sailor_gap_bins"], "treatment_context_comparison"),
        (plot_cohort_feature_distribution, m["site_shift"], "cohort_feature_distributions"),
        (plot_aligned_feature_distribution, m["site_shift_aligned"],
         "aligned_cohort_feature_distributions"),
        (plot_feature_distribution_matching, m["coral"], "feature_distribution_matching"),
        (plot_image_processing_effect, m["raw_mni"], "image_processing_effect"),
        (plot_reprocessed_second_cohort, m["sailor_reprocessed"],
         "second_cohort_reprocessing"),
        (plot_reprocessed_forecast, m, "reprocessed_image_forecasting"),
        (plot_processing_groups, m["three_way_shift"], "processing_comparison_three_groups"),
        (plot_skull_tissue_sensitivity, m["betfix_decider"], "skull_tissue_sensitivity"),
        (plot_four_group_processing_comparison, m["three_way_shift"], m["fourth_cloud"],
         "processing_comparison_four_groups"),
        (plot_response_feature_comparison, m["gauntlet_probe"], "response_feature_comparison"),
        (plot_second_cohort_adaptation, m["cross_site_adapt"], "second_cohort_adaptation"),
    ]
    for spec in plots:
        if len(spec) == 4:
            fn, a, b, name = spec
            fn(a, b, name)
        else:
            fn, a, name = spec
            fn(a, name)
    print(f"Wrote {len(plots)} reviewer-facing plots (PNG + PDF) to {OUT}")


if __name__ == "__main__":
    main()
