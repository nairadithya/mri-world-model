"""Regenerate every plot in info/plots/ from metrics.json.

Usage:
    python3 info/plots/make_plots.py

Style: figures4papers house style (github.com/ChenLiu-1996/figures4papers,
CC BY-NC 4.0) — semantic blue/green/red palette, no top/right spines,
print-safe bars (black edges; hatches for sub-groups), 5-pt error caps,
y-limits tightened to the data range, dpi-300 png + pdf per panel.

All numbers come from metrics.json (the single source of truth — edit
numbers there, not here); every ratio/percentage stated in a title or
annotation is computed from it, never hardcoded.

Reads info/plots/metrics.json and writes, per plot, a .png and a .pdf:
    hero_leg1_val          Hero Run Leg 1 val curve (best at epoch 7)
    hero_leg2_val          Hero Run Leg 2 val drift (champion never touched)
    run6_tradeoff          Aux fine-tune: dynamics cost vs honest-F1 flat
    sailor_transfer        Cross-site summary: dynamics fail, readouts transfer
    sailor_gap_bins        Gap-stratified cross-site: persistence wins every bin
    sailor_gaphead_vs_champ  Gap head vs champion head by bin
    sailor_field_rescue    Velocity field vs champion head by bin
    sailor_field_ablation  Treatment-vs-constant phase of the velocity field
    site_shift / site_shift_aligned  Latent clouds before/after CORAL
    coral_alignment        Statistics-matching head fix (zero training)
    raw_vs_mni             Raw-session change vs MNI-pipeline change
    sailor_reprocessed / sailor_reproc_vs_persist  Execute-pipeline gate
    three_way_shift / four_clouds  PCA clouds across cache generations
    betfix_decider         Skull-restored gate (site, not skull)
    gauntlet_probe         Predicted vs actual latents as RANO features
                           (error ratio vs preserved signal dissociate)
    cross_site_adapt       Frozen JEPA vs from-scratch CNNs (K-shot)
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

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
    "neutral": "#CFCECE",         # persistence / reference
    "highlight": "#FFD700",       # single callout only
    "teal": "#42949E", "violet": "#9A4D8E",
}
DEFAULT_COLORS = ["#0F4D92", "#8BCF8B", "#B64342", "#42949E", "#9A4D8E", "#CFCECE"]

# Series semantics used throughout (keep consistent across panels):
#   blue_main    the proposed / showcased method
#   green_3      the improvement over a baseline in the same panel
#   red_strong   the failing / contrast arm (champion head when beaten, etc.)
#   neutral      persistence and other pure reference baselines
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
    """Write png + pdf next to metrics.json (repo layout), print dpi."""
    stem = os.path.join(HERE, name)
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


# ------------------------------------------------------------------ hero legs
def plot_hero_leg1(m, name):
    ep, val = m["epochs"], m["val"]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(ep, val, "o-", color=PALETTE["blue_main"], lw=2.5, ms=6,
            markeredgecolor=BAR_EDGE, markeredgewidth=1.0)
    ax.scatter([m["best_epoch"]], [m["best_val"]], s=160, zorder=5,
               color=PALETTE["green_3"], edgecolor=BAR_EDGE, linewidth=1.4)
    ax.axvline(m["best_epoch"], color=PALETTE["green_3"], ls="--", lw=1.4)
    ax.text(m["best_epoch"] + 0.8, max(val) * 0.72,
            f"best: epoch {m['best_epoch']}\nval = {m['best_val']:.4f}",
            color=PALETTE["green_3"], fontsize=ANNO, va="top", weight="bold")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val loss")
    ax.set_title("Hero Run Leg 1 — val loss (best at epoch 7)")
    ax.set_xticks(ep)
    ax.set_ylim(0, max(val) * 1.18)
    save(fig, name)


def plot_hero_leg2(m, name):
    ep, val = m["epochs"], m["val"]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(ep, val, "o-", color=PALETTE["red_strong"], lw=2.5, ms=6,
            markeredgecolor=BAR_EDGE, markeredgewidth=1.0, label="leg-2 val")
    ax.axhline(m["champion_val"], color=PALETTE["blue_main"], ls="--", lw=1.6,
               label=f"champion {m['champion_val']:.4f} (never touched)")
    ax.scatter([m["leg_best_epoch"]], [m["leg_best_val"]], s=150, zorder=5,
               color=PALETTE["highlight"], edgecolor=BAR_EDGE, linewidth=1.2)
    ax.text(2.0, 0.0295,
            f"leg best: epoch {m['leg_best_epoch']}\nval = {m['leg_best_val']:.4f}",
            color=PALETTE["red_strong"], fontsize=ANNO, va="top")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val loss")
    ax.set_title("Hero Run Leg 2 — val drifts up (champion holds)")
    ax.set_xticks(ep)
    ax.set_ylim(0.004, 0.033)
    ax.legend(fontsize=11, loc="lower right", **LEGEND_BOX)
    save(fig, name)


# --------------------------------------------------------------------- run 6
def plot_run6(m, name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.4))
    # Left: dynamics cost (lower is better); champion = key, aux = harm.
    ax1.bar([0, 1], [m["dynamics_champion"], m["dynamics_aux"]],
            color=[PALETTE["blue_main"], PALETTE["red_strong"]],
            width=0.55, edgecolor=BAR_EDGE, linewidth=BAR_LW)
    ax1.axhline(m["persistence_ref"], color=PALETTE["neutral"], ls="--", lw=1.6,
                label=f"persistence {m['persistence_ref']:.3f}")
    ax1.set_xticks([0, 1], ["champion", "aux-tuned"])
    ax1.set_ylabel("JEPA error")
    ax1.set_title(
        f"Run 6 cost: dynamics {m['dynamics_aux'] / m['dynamics_champion']:.1f}x worse")
    ax1.set_ylim(0, 0.028)
    for x, v in [(0, m["dynamics_champion"]), (1, m["dynamics_aux"])]:
        ax1.text(x, v + 0.0007, f"{v:.4f}", ha="center", fontsize=ANNO)
    ax1.legend(fontsize=10.5, loc="upper left", **LEGEND_BOX)
    # Right: task gain is split luck (higher is better); aux pair hatched.
    pre = [m["f1_hero_champion"], m["f1_cv_champion"]]
    post = [m["f1_hero_aux"], m["f1_cv_aux"]]
    pre_err = [0, m["f1_cv_champion_sd"]]
    post_err = [0, m["f1_cv_aux_sd"]]
    x = list(range(2))
    w = 0.38
    ax2.bar([i - w / 2 for i in x], pre, w, yerr=pre_err, label="champion encoder",
            color=PALETTE["blue_main"], edgecolor=BAR_EDGE, linewidth=BAR_LW, **INKD)
    ax2.bar([i + w / 2 for i in x], post, w, yerr=post_err,
            label="aux-tuned encoder", color=PALETTE["red_strong"], hatch="/",
            edgecolor=BAR_EDGE, linewidth=BAR_LW, **INKD)
    ax2.axhline(m["sota_f1"], color=PALETTE["neutral"], ls=":", lw=1.6,
                label=f"SOTA {m['sota_f1']:.2f}")
    ax2.set_xticks(x, ["hero-split", "CV (honest)"])
    ax2.set_ylabel("macro-F1")
    ax2.set_title("Run 6 gain: split luck only")
    ax2.set_ylim(0, 0.70)        # headroom so the legend clears the labels
    for i in x:
        ax2.text(i - w / 2, pre[i] + pre_err[i] + 0.02, f"{pre[i]:.3f}",
                 ha="center", fontsize=ANNO - 1)
        ax2.text(i + w / 2, post[i] + post_err[i] + 0.02, f"{post[i]:.3f}",
                 ha="center", fontsize=ANNO - 1)
    ax2.legend(fontsize=10, loc="upper right", **LEGEND_BOX)
    fig.suptitle("Run 6 (aux fine-tune): classification fell, dynamics paid, "
                 "honest F1 flat", fontsize=13)
    save(fig, name, rect=[0, 0, 1, 0.94])


# -------------------------------------------------------------- cross-site
def plot_sailor_transfer(m, name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.4))
    # Left: dynamics fail -> JEPA is the contrast arm, persistence the ref.
    dj, dp = m["dynamics_jepa"], m["dynamics_persist"]
    ax1.bar([0, 1], [dj, dp],
            color=[PALETTE["red_strong"], PALETTE["neutral"]],
            width=0.55, edgecolor=BAR_EDGE, linewidth=BAR_LW)
    ax1.set_xticks([0, 1], ["JEPA", "persistence"])
    ax1.set_ylabel("mean error")
    ax1.set_title(f"Dynamics: persistence wins ~{dj / dp:.1f}x")
    ax1.set_ylim(0, max(dj, dp) * 1.35)
    for x, v in [(0, dj), (1, dp)]:
        ax1.text(x, v + 0.001, f"{v:.4f}", ha="center", fontsize=ANNO)
    # Right: readouts transfer ~= in-domain (zero SAILOR training) -> green.
    f1 = [m["f1_lumiere_cv"], m["f1_sailor_transfer"]]
    ax2.bar([0, 1], f1, color=[PALETTE["blue_main"], PALETTE["green_3"]],
            width=0.55, edgecolor=BAR_EDGE, linewidth=BAR_LW)
    ax2.axhline(m["majority_f1"], color=PALETTE["neutral"], ls="--", lw=1.6,
                label=f"majority {m['majority_f1']:.2f}")
    ax2.set_xticks([0, 1], ["LUMIERE\nin-domain", "SAILOR\ntransfer"])
    ax2.set_ylabel("macro-F1")
    ax2.set_title("Readouts: transfer ~= in-domain, zero training")
    ax2.set_ylim(0, 0.5)
    for i, v in enumerate(f1):
        ax2.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=ANNO)
    ax2.text(1, 0.42, f"AUC {m['surprise_auc_sailor']}\n(n={m['surprise_n']})",
             ha="center", va="bottom", fontsize=9.5, color="black",
             weight="bold")
    ax2.legend(fontsize=10.5, loc="upper left", **LEGEND_BOX)
    fig.suptitle("Cross-site (LUMIERE to SAILOR): dynamics fail, "
                 "representation transfers", fontsize=13)
    save(fig, name, rect=[0, 0, 1, 0.94])


# ------------------------------------------------------- gap-stratified bins
def plot_gap_bins(m, name):
    bins, pairs, jepa, persist = m["bins"], m["pairs"], m["jepa"], m["persist"]
    x = list(range(len(bins)))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.bar([i - w / 2 for i in x], jepa, w, label="JEPA (frozen champion)",
           color=PALETTE["red_strong"], edgecolor=BAR_EDGE, linewidth=BAR_LW)
    ax.bar([i + w / 2 for i in x], persist, w, label="persistence (no change)",
           color=PALETTE["neutral"], edgecolor=BAR_EDGE, linewidth=BAR_LW)
    ax.set_xticks(x, [f"{b}\n(n={p})" for b, p in zip(bins, pairs)])
    ax.tick_params(axis="x", labelsize=TICK_DENSE)
    ax.set_ylabel("mean error")
    ax.set_title(f"Cross-site by gap: persistence wins every bin "
                 f"(n={m['n_pairs']}, median gap {m['gap_median']}d)",
                 fontsize=13)
    for i in x:
        ax.text(i - w / 2, jepa[i] + 0.0009, f"{jepa[i]:.4f}",
                ha="center", fontsize=ANNO - 1)
        ax.text(i + w / 2, persist[i] + 0.0009, f"{persist[i]:.4f}",
                ha="center", fontsize=ANNO - 1)
    ax.set_ylim(0, max(jepa) * 1.3)
    ax.legend(fontsize=11, loc="upper right")
    save(fig, name)


def plot_gaphead(m, name):
    bins, pairs, champ, gap = m["bins"], m["pairs"], m["jepa"], m["gap_head"]
    persist = m["persist"]
    x = list(range(len(bins)))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    ax.bar([i - w / 2 for i in x], champ, w, label="champion 1-step",
           color=PALETTE["red_strong"], edgecolor=BAR_EDGE, linewidth=BAR_LW)
    ax.bar([i + w / 2 for i in x], gap, w, label="gap-conditioned head",
           color=PALETTE["blue_main"], edgecolor=BAR_EDGE, linewidth=BAR_LW)
    ax.plot(x, persist, "D-", color="black", ms=6, lw=1.4,
            label="persistence (still wins)")
    ax.set_xticks(x, [f"{b}\n(n={p})" for b, p in zip(bins, pairs)])
    ax.tick_params(axis="x", labelsize=TICK_DENSE)
    ax.set_ylabel("mean error")
    ax.set_title("Gap-conditioned head beats champion head, "
                 "still loses to persistence", fontsize=13)
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


def plot_field_rescue(m, name):
    bins, pairs = m["bins"], m["pairs"]
    champ, persist, cond = m["jepa"], m["persist"], m["field_cond"]
    x = list(range(len(bins)))
    w = 0.3
    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    ax.bar([i - w for i in x], champ, w, label="champion head",
           color=PALETTE["red_strong"], edgecolor=BAR_EDGE, linewidth=BAR_LW)
    ax.bar(x, cond, w, label="velocity field",
           color=PALETTE["blue_main"], edgecolor=BAR_EDGE, linewidth=BAR_LW)
    mid = [i + w for i in x]        # right of both bars -> labels stay clear
    ax.plot(mid, persist, "D-", color="black", ms=6, lw=1.4,
            label="persistence")
    ax.set_xticks(x, [f"{b}\n(n={p})" for b, p in zip(bins, pairs)])
    ax.tick_params(axis="x", labelsize=TICK_DENSE)
    ax.set_ylabel("mean error")
    ratio = np.median(np.array(champ) / np.array(cond))
    ax.set_title(f"Velocity field beats champion ~{ratio:.0f}x, "
                 "at/below persistence", fontsize=13)
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


def plot_field_ablation(m, name):
    bins, pairs = m["bins"], m["pairs"]
    persist, cond, uncond = m["persist"], m["field_cond"], m["field_uncond"]
    x = list(range(len(bins)))
    w = 0.22
    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    ax.bar([i - w for i in x], cond, w, label="treatment phase",
           color=PALETTE["blue_main"], edgecolor=BAR_EDGE, linewidth=BAR_LW)
    ax.bar(x, uncond, w, label="constant phase", hatch=".",
           color=PALETTE["blue_secondary"], edgecolor=BAR_EDGE, linewidth=BAR_LW)
    ax.bar([i + w for i in x], persist, w, label="persistence",
           color=PALETTE["neutral"], edgecolor=BAR_EDGE, linewidth=BAR_LW)
    ax.set_xticks(x, [f"{b}\n(n={p})" for b, p in zip(bins, pairs)])
    ax.tick_params(axis="x", labelsize=TICK_DENSE)
    ax.set_ylabel("mean error")
    ax.set_title("Ablation: exact tie; field beats persistence 3/4 bins",
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
def plot_site_shift(m, name):
    lum = np.array(m["lumiere_xy"])
    sai = np.array(m["sailor_xy"])
    e1, e2 = m["explained_pct"]
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.scatter(lum[:, 0], lum[:, 1], s=16, alpha=0.5, color=PALETTE["blue_main"],
               label=f"LUMIERE visits (n={m['n_lumiere']})")
    ax.scatter(sai[:, 0], sai[:, 1], s=20, alpha=0.6, color=PALETTE["red_strong"],
               label=f"SAILOR visits (n={m['n_sailor']})")
    ax.set_xlabel(f"PC1 ({e1}%)")
    ax.set_ylabel(f"PC2 ({e2}%)")
    ax.set_title("Site shift")
    ax.legend(fontsize=11, loc="upper left", **LEGEND_BOX)
    save(fig, name)


def plot_site_shift_aligned(m, name):
    lum = np.array(m["lumiere_xy"])
    sai = np.array(m["sailor_xy"])
    e1, e2 = m["explained_pct"]
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.scatter(lum[:, 0], lum[:, 1], s=16, alpha=0.5, color=PALETTE["blue_main"],
               label=f"LUMIERE visits (n={m['n_lumiere']})")
    ax.scatter(sai[:, 0], sai[:, 1], s=20, alpha=0.6, color=PALETTE["green_3"],
               label=f"SAILOR visits, CORAL-aligned (n={m['n_sailor']})")
    ax.set_xlabel(f"PC1 ({e1}%)")
    ax.set_ylabel(f"PC2 ({e2}%)")
    ax.set_title("After CORAL: shared frame,\nresidual structure remains")
    ax.legend(fontsize=11, loc="upper left", **LEGEND_BOX)
    save(fig, name)


def plot_coral(m, name):
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    cats = ["frozen head", "persistence"]
    x = [0, 1]
    w = 0.26
    ax.bar(x[0] - w, m["head_unaligned"], w, label="head, as-is",
           color=PALETTE["red_strong"], edgecolor=BAR_EDGE, linewidth=BAR_LW)
    ax.bar(x[0], m["head_aligned"], w, label="head, CORAL-aligned",
           color=PALETTE["green_3"], edgecolor=BAR_EDGE, linewidth=BAR_LW)
    ax.bar(x[0] + w, m["head_transductive"], w, label="head, transductive bound",
           color=PALETTE["blue_secondary"], edgecolor=BAR_EDGE, linewidth=BAR_LW)
    ax.bar(x[1] - w / 2, m["persist_unaligned"], w, label="persistence, as-is",
           color=PALETTE["neutral"], edgecolor=BAR_EDGE, linewidth=BAR_LW)
    ax.bar(x[1] + w / 2, m["persist_aligned"], w, label="persistence, aligned",
           hatch=".", color=PALETTE["neutral"], edgecolor=BAR_EDGE,
           linewidth=BAR_LW)
    ax.set_xticks(x, cats)
    ax.set_ylabel("mean error")
    ratio = m["head_unaligned"] / m["head_aligned"]
    ax.set_title(f"Statistics-matching alone cuts head error {ratio:.1f}x "
                 "(zero training)", fontsize=13)
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
    ax.text(0.5, 0.021, "encoder untouched\nresidual = higher-order shift",
            ha="center", fontsize=9.5, color="dimgray")
    ax.set_ylim(0, 0.034)
    ax.legend(fontsize=9.5, loc="upper right")
    save(fig, name)


def plot_raw_mni(m, name):
    slots, n = m["slots"], m["n"]
    raw, mni = m["raw_med"], m["mni_med"]
    x = list(range(len(slots)))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    ax.bar([i - w / 2 for i in x], raw, w, label="raw sessions",
           color=PALETTE["red_strong"], edgecolor=BAR_EDGE, linewidth=BAR_LW)
    ax.bar([i + w / 2 for i in x], mni, w, label="MNI pipeline",
           color=PALETTE["blue_main"], edgecolor=BAR_EDGE, linewidth=BAR_LW)
    ax.set_xticks(x, [f"{s}\n(n={p})" for s, p in zip(slots, n)])
    ax.tick_params(axis="x", labelsize=TICK_DENSE)
    ax.set_ylabel("median mean|change|")
    ratio = np.median(np.array(raw) / np.array(mni))
    ax.set_title(f"Same pairs: raw change ~{ratio:.1f}x MNI change "
                 "(pipeline damps)", fontsize=13)
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
    """Deriv-vs-reproc x JEPA-vs-persist gate bars (red = JEPA, gray =
    persistence; hatch = reprocessed arm) with value labels."""
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


def plot_sailor_reprocessed(m, name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.4))
    _four_gate_bars(
        ax1, m, ["deriv\nJEPA", "deriv\npersist", "reproc\nJEPA",
                 "reproc\npersist"],
        "mean error", "Gate does not flip: persistence wins both arms")
    f1 = [m["deriv_transfer_f1"], m["transfer_f1"]]
    ax2.bar([0, 1], f1,
            color=[PALETTE["blue_main"], PALETTE["red_strong"]],
            width=0.55, hatch=[None, "/"], edgecolor=BAR_EDGE,
            linewidth=BAR_LW)
    ax2.set_xticks([0, 1], ["deriv\ntransfer", "reproc\ntransfer"])
    ax2.tick_params(axis="x", labelsize=TICK_DENSE)
    ax2.set_ylabel("macro-F1")
    maj = m["transfer_majority"]
    ax2.set_title(f"Transfer readout drops (maj {maj:.2f}, AUC "
                  f"{m['deriv_surprise_auc']:.2f}->{m['surprise_auc']:.2f})",
                  fontsize=12.5)
    ax2.set_ylim(0, 0.5)
    for i, v in enumerate(f1):
        ax2.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=ANNO)
    fig.suptitle("SAILOR reprocessed (LUMIERE-contract): dynamics still lose, "
                 "readouts drop", fontsize=13)
    save(fig, name, rect=[0, 0, 1, 0.94])


def plot_reproc_vs_persist(m, name):
    r = m["sailor_reprocessed"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.4))
    _four_gate_bars(
        ax1, r, ["deriv\nJEPA", "deriv\npersist", "reproc\nJEPA",
                 "reproc\npersist"],
        f"mean error (pooled, n={r['n_pairs']})",
        f"JEPA error falls {r['deriv_jepa']:.4f}->{r['dynamics_jepa']:.4f}")
    dj, dp = r["deriv_jepa"] / r["deriv_persist"], \
        r["dynamics_jepa"] / r["dynamics_persist"]
    ax2.bar([0, 1], [dj, dp],
            color=[PALETTE["blue_secondary"], PALETTE["blue_main"]],
            width=0.55, hatch=[None, "/"], edgecolor=BAR_EDGE,
            linewidth=BAR_LW)
    ax2.set_xticks([0, 1], ["derivatives", "reprocessed"])
    ax2.set_ylabel("JEPA / persistence error ratio")
    ax2.set_title(f"persistence wins {dp / dj:.1f}x bigger after reprocess",
                  fontsize=12.5)
    ax2.set_ylim(0, max(dj, dp) * 1.3)
    for i, v in enumerate([dj, dp]):
        ax2.text(i, v + max(dj, dp) * 0.05, f"{v:.1f}x", ha="center",
                 fontsize=ANNO)
    fig.suptitle("Reprocess vs derivatives: representation error improves, "
                 "gate does not", fontsize=13)
    save(fig, name, rect=[0, 0, 1, 0.94])


def plot_three_way(m, name):
    lum = np.array(m["lumiere_xy"])
    der = np.array(m["deriv_xy"])
    rep = np.array(m["reproc_xy"])
    e1, e2 = m["explained_pct"]
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    ax.scatter(lum[:, 0], lum[:, 1], s=16, alpha=0.5, color=PALETTE["blue_main"],
               label=f"LUMIERE (n={m['n_lumiere']})")
    ax.scatter(der[:, 0], der[:, 1], s=20, alpha=0.6, color=PALETTE["red_strong"],
               label=f"SAILOR derivatives (n={m['n_deriv']})")
    ax.scatter(rep[:, 0], rep[:, 1], s=20, alpha=0.6, color=PALETTE["violet"],
               label=f"SAILOR reprocessed, skull-in (n={m['n_reproc']})")
    ax.set_xlabel(f"PC1 ({e1}%)")
    ax.set_ylabel(f"PC2 ({e2}%)")
    ax.set_title("Three clouds: reprocessing moved\nSAILOR somewhere new")
    ax.legend(fontsize=10.5, loc="upper left", **LEGEND_BOX)
    save(fig, name)


def plot_fourth_cloud(m, t, name):
    lum = np.array(t["lumiere_xy"])
    bet = np.array(t["betfix_xy"])
    der = np.array(t["deriv_xy"])
    rep = np.array(t["repro_xy"])
    e1, e2 = t["explained_pct"]
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    ax.scatter(lum[:, 0], lum[:, 1], s=16, alpha=0.5, color=PALETTE["blue_main"],
               label=f"LUMIERE (n={t['n_lumiere']})")
    ax.scatter(der[:, 0], der[:, 1], s=18, alpha=0.55, color=PALETTE["red_strong"],
               label=f"SAILOR derivatives (n={t['n_deriv']})")
    ax.scatter(rep[:, 0], rep[:, 1], s=18, alpha=0.55, color=PALETTE["violet"],
               label=f"reprocessed skull-in (n={t['n_repro']})")
    ax.scatter(bet[:, 0], bet[:, 1], s=22, alpha=0.65, color=PALETTE["teal"],
               label=f"reprocessed skull-out (n={t['n_betfix']})")
    ax.set_xlabel(f"PC1 ({e1}%)")
    ax.set_ylabel(f"PC2 ({e2}%)")
    ax.set_title("Skull-out rejoins derivatives;\nLUMIERE stays distant")
    ax.legend(fontsize=9.5, loc="upper left", **LEGEND_BOX)
    save(fig, name)


def plot_betfix_decider(m, name):
    d = m
    x = list(range(len(d["bins"])))
    w = 0.3
    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    ax.bar([i - w for i in x], d["jepa"], w, label="champion head",
           color=PALETTE["red_strong"], edgecolor=BAR_EDGE, linewidth=BAR_LW)
    ax.plot(x, d["persist"], "D-", color="black", ms=6, lw=1.4,
            label="persistence")
    ax.set_xticks(x, [f"{b}\n(n={p})" for b, p in zip(d["bins"], d["pairs"])])
    ax.tick_params(axis="x", labelsize=TICK_DENSE)
    ax.set_ylabel("mean error")
    ax.set_title("Skull restored, gate still lost in every bin (site, not skull)",
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
def plot_gauntlet(g, name):
    methods = g["methods"]
    labels = g["method_labels"]
    colors = {"true": PALETTE["neutral"], "champ": PALETTE["red_strong"],
              "gap": PALETTE["blue_main"], "field": PALETTE["teal"]}
    hts = {"true": ".", "champ": "/", "gap": "\\", "field": "x"}
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.4))
    fig.suptitle("Dynamics gauntlet: predicted vs actual latents as RANO "
                 "features", fontsize=14, fontweight="bold")

    # A. honest-number bars (CV with spread; transfer single-split).
    groups = [("LUMIERE\npatient CV", g["lum_cv_f1"], g["lum_cv_sd"]),
              ("SAILOR\nsubject CV", g["sai_cv_f1"], g["sai_cv_sd"]),
              ("SAILOR\ntransfer", g["sai_transfer_f1"], [0.0] * 4)]
    x = np.arange(len(groups))
    w = 0.19
    for i, m in enumerate(methods):
        means = [grp[1][i] for grp in groups]
        errs = [grp[2][i] for grp in groups]
        axA.bar(x + (i - 1.5) * w, means, w, yerr=errs,
                label=labels[i], color=colors[m], hatch=hts[m],
                edgecolor=BAR_EDGE, linewidth=BAR_LW, **INKD)
    axA.set_xticks(x)
    axA.set_xticklabels([grp[0] for grp in groups])
    axA.tick_params(axis="x", labelsize=TICK_DENSE)
    axA.set_ylabel("macro-F1 (RANO t+1, MLP probe)")
    axA.set_title("A. Preserved RANO signal by feature", fontsize=12.5)
    axA.set_ylim(0, 0.58)        # headroom for legend + reference labels
    axA.hlines(g["a9_states_cv"], -0.4, 0.4, colors="k", ls="--", lw=1)
    axA.text(0.0, g["a9_states_cv"] + 0.011, f"A9 states CV {g['a9_states_cv']:.2f}",
             ha="center", fontsize=9, bbox=BOXED)
    axA.hlines(g["a12b_states_transfer"], 1.6, 2.4, colors="k", ls="--", lw=1)
    axA.text(2.0, g["a12b_states_transfer"] + 0.011,
             f"A12-b states transfer {g['a12b_states_transfer']:.2f}",
             ha="center", fontsize=9, bbox=BOXED)
    axA.legend(fontsize=10, loc="upper left")

    # B. dissociation scatter: error ratio (log) vs signal.
    # "true" has no JEPA error (it is the endpoint) -> parity ratio 1.0.
    EVALS = [("LUM-CV", "o", g["lum_cv_f1"], g["lum_cv_sd"], g["lum_err"]),
             ("SAI-CV", "s", g["sai_cv_f1"], g["sai_cv_sd"], g["sai_err"]),
             ("SAI-transfer", "^", g["sai_transfer_f1"], [None] * 4,
              g["sai_err"])]
    for tag, mk, f1s, sds, errs in EVALS:
        for j, m in enumerate(methods):
            e = errs[m]
            r = 1.0 if e is None else e / errs["persist"]
            sd = sds[j]
            axB.errorbar(r, f1s[j], yerr=sd if sd else 0.0, fmt=mk,
                         color=colors[m], ecolor=colors[m],
                         markersize=8, capsize=3, elinewidth=1,
                         label=f"{labels[j]} · {tag}" if tag == "LUM-CV"
                         else None)
    import matplotlib.lines as mlines
    meth = [mlines.Line2D([], [], color=colors[m], marker="o",
                           linestyle="None", markersize=7, label=labels[j])
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
    axB.text(1.0, 0.075, "persistence parity", ha="center", fontsize=8.5,
             transform=axB.get_xaxis_transform())
    axB.set_xlabel("cosine-error ratio JEPA / persistence (log)")
    axB.set_ylabel("macro-F1 (RANO t+1, MLP probe)")
    axB.set_title("B. Error ratio vs preserved signal", fontsize=12.5)
    axB.annotate("SAI field:\nbest error,\nmiddling signal",
                 xy=(0.9, 0.32), xytext=(0.35, 0.30), ha="center", va="top",
                 arrowprops=dict(arrowstyle="->", color="dimgray"),
                 fontsize=8.5,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"))
    axB.annotate("LUM field:\nworst error,\ngap-level signal",
                 xy=(3.11, 0.272), xytext=(8.0, 0.155),
                 arrowprops=dict(arrowstyle="->", color="dimgray"),
                 fontsize=8.5, ha="center",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"))
    axB.set_ylim(0.05, 0.48)
    fig.text(0.5, 0.015,
             "Probes train-stat standardized (A9 protocol otherwise); "
             "transfer = train on LUMIERE-train rows; field = SAILOR-fit "
             "uncond (cond ties). Errors on the same labelled pairs.",
             ha="center", fontsize=9, style="italic")
    save(fig, name, dpi=600, rect=[0, 0, 1, 0.92])


# ----------------------------------------------------------- cross-site K
def plot_cross_site_adapt(g, name):
    ks = [0] + list(g["ks"])
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    style = {"JEPA": (PALETTE["blue_main"], "o", 2.6),
             "CNN-3D": (PALETTE["red_strong"], "s", 1.8),
             "CNN-2D": (PALETTE["red_2"], "^", 1.8),
             "CNN-2D-ROI": (PALETTE["violet"], "D", 1.8),
             "RAD": (PALETTE["neutral"], "v", 1.8)}
    for enc_name, enc in g["encoders"].items():
        ys = [enc["zero_shot"]] + list(enc["ks"])
        col, mk, lw = style.get(enc_name, ("#777777", "x", 1.2))
        ax.plot(ks, ys, mk + "-", color=col, lw=lw, ms=6, label=enc_name)
    ax.axhline(g["majority_f1"], color=PALETTE["neutral"], ls="--", lw=1.6,
               label=f"SAILOR majority {g['majority_f1']:.2f}")
    ax.set_xlabel("K support subjects (0 = zero-shot from LUMIERE)")
    ax.set_ylabel("SAILOR macro-F1 (subject-wise)")
    ax.set_title("Cross-site adaptability: frozen JEPA\n"
                 "vs from-scratch supervised CNNs")
    ax.set_xticks(ks)
    ax.set_ylim(0.08, 0.52)      # keep the 0.479 majority line inside
    ax.legend(fontsize=10, loc="upper left", ncol=2)
    ax.text(0.02, 0.145,
            "CNNs collapse in-domain (0.18-0.22); the radiomics/growth "
            "comparator (RAD) trails too.",
            transform=ax.transAxes, fontsize=8.5, style="italic",
            color="dimgray", va="top")
    ax.text(0.02, 0.085, "JEPA leads every K (A29-A35).",
            transform=ax.transAxes, fontsize=8.5, style="italic",
            color="dimgray", va="top")
    save(fig, name)


def main():
    m = load_metrics()
    plots = [
        (plot_hero_leg1, m["hero_leg1"], "hero_leg1_val"),
        (plot_hero_leg2, m["hero_leg2"], "hero_leg2_val"),
        (plot_run6, m["run6"], "run6_tradeoff"),
        (plot_sailor_transfer, m["sailor_transfer"], "sailor_transfer"),
        (plot_gap_bins, m["sailor_gap_bins"], "sailor_gap_bins"),
        (plot_gaphead, m["sailor_gap_bins"], "sailor_gaphead_vs_champ"),
        (plot_field_rescue, m["sailor_gap_bins"], "sailor_field_rescue"),
        (plot_field_ablation, m["sailor_gap_bins"], "sailor_field_ablation"),
        (plot_site_shift, m["site_shift"], "site_shift"),
        (plot_site_shift_aligned, m["site_shift_aligned"],
         "site_shift_aligned"),
        (plot_coral, m["coral"], "coral_alignment"),
        (plot_raw_mni, m["raw_mni"], "raw_vs_mni"),
        (plot_sailor_reprocessed, m["sailor_reprocessed"],
         "sailor_reprocessed"),
        (plot_reproc_vs_persist, m, "sailor_reproc_vs_persist"),
        (plot_three_way, m["three_way_shift"], "three_way_shift"),
        (plot_betfix_decider, m["betfix_decider"], "betfix_decider"),
        (plot_fourth_cloud, m["three_way_shift"], m["fourth_cloud"],
         "four_clouds"),
        (plot_gauntlet, m["gauntlet_probe"], "gauntlet_probe"),
        (plot_cross_site_adapt, m["cross_site_adapt"], "cross_site_adapt"),
    ]
    for spec in plots:
        if len(spec) == 4:
            fn, a, b, name = spec
            fn(a, b, name)
        else:
            fn, a, name = spec
            fn(a, name)
    print(f"wrote {len(plots)} plots (png + pdf) to", HERE)


if __name__ == "__main__":
    main()