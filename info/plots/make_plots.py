"""Regenerate every plot in info/plots/ from metrics.json.

Usage:
    python3 info/plots/make_plots.py

Reads info/plots/metrics.json (the single source of truth — edit numbers
there, not in this script) and writes:
    hero_leg1_val.png   Hero Run Leg 1 val curve (best at epoch 7)
    hero_leg2_val.png   Hero Run Leg 2 val drift (champion never touched)
    run6_tradeoff.png   Aux fine-tune: dynamics cost vs honest-F1 flat
    sailor_transfer.png Cross-site summary: dynamics fail, readouts transfer
    sailor_gap_bins.png Gap-stratified cross-site: persistence wins every bin
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def load_metrics():
    with open(os.path.join(HERE, "metrics.json")) as f:
        return json.load(f)


def plot_hero_leg1(m, out):
    ep, val = m["epochs"], m["val"]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(ep, val, "o-", lw=2)
    ax.scatter([m["best_epoch"]], [m["best_val"]], s=120, color="green", zorder=5)
    ax.axvline(m["best_epoch"], color="green", ls="--", lw=1.2)
    ax.text(m["best_epoch"] + 0.5, 0.06,
            f"best: epoch {m['best_epoch']}\nval = {m['best_val']:.4f}",
            color="green", fontsize=10, va="top")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val loss")
    ax.set_title("Hero Run Leg 1 — val loss (best at epoch 7)")
    ax.set_xticks(ep)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_hero_leg2(m, out):
    ep, val = m["epochs"], m["val"]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(ep, val, "o-", lw=2, label="leg-2 val")
    ax.axhline(m["champion_val"], color="green", ls="--", lw=1.2,
               label=f"champion {m['champion_val']:.4f} (never touched)")
    ax.scatter([m["leg_best_epoch"]], [m["leg_best_val"]], s=120,
               color="orange", zorder=5)
    ax.text(2.2, 0.024,
            f"leg best: epoch {m['leg_best_epoch']}\nval = {m['leg_best_val']:.4f}",
            color="#B26A00", fontsize=10, va="top")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val loss")
    ax.set_title("Hero Run Leg 2 — val drifts up (champion holds)")
    ax.set_xticks(ep)
    ax.set_ylim(0.005, 0.033)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_run6(m, out):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))
    # Left: dynamics cost (lower is better)
    ax1.bar([0, 1], [m["dynamics_champion"], m["dynamics_aux"]],
            color=["green", "red"], width=0.55)
    ax1.axhline(m["persistence_ref"], color="gray", ls="--", lw=1.2,
                label=f"persistence ~{m['persistence_ref']:.3f}")
    ax1.set_xticks([0, 1], ["champion", "aux-tuned"])
    ax1.set_ylabel("JEPA error")
    ax1.set_title("Run 6 cost: dynamics 2.5x worse")
    ax1.set_ylim(0, 0.026)
    for x, v in [(0, m["dynamics_champion"]), (1, m["dynamics_aux"])]:
        ax1.text(x, v + 0.0007, f"{v:.4f}", ha="center", fontsize=10)
    ax1.legend(fontsize=8)
    # Right: task gain is split luck (higher is better)
    pre = [m["f1_hero_champion"], m["f1_cv_champion"]]
    post = [m["f1_hero_aux"], m["f1_cv_aux"]]
    pre_err = [0, m["f1_cv_champion_sd"]]
    post_err = [0, m["f1_cv_aux_sd"]]
    x = list(range(2))
    w = 0.38
    ax2.bar([i - w / 2 for i in x], pre, w, yerr=pre_err, capsize=4,
            label="champion encoder", color="#1f77b4")
    ax2.bar([i + w / 2 for i in x], post, w, yerr=post_err, capsize=4,
            label="aux-tuned encoder", color="#ff7f0e")
    ax2.axhline(m["sota_f1"], color="gray", ls=":", lw=1.2,
                label=f"SOTA {m['sota_f1']:.2f}")
    ax2.set_xticks(x, ["hero-split", "CV (honest)"])
    ax2.set_ylabel("macro-F1")
    ax2.set_title("Run 6 gain: split luck only")
    ax2.set_ylim(0, 0.62)
    for i in x:
        ax2.text(i - w / 2, pre[i] + pre_err[i] + 0.02, f"{pre[i]:.3f}",
                 ha="center", fontsize=9)
        ax2.text(i + w / 2, post[i] + post_err[i] + 0.02, f"{post[i]:.3f}",
                 ha="center", fontsize=9)
    ax2.legend(fontsize=8)
    fig.suptitle("Run 6 (aux fine-tune): classification fell, dynamics paid, honest F1 flat",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_sailor_transfer(m, out):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))
    ax1.bar([0, 1], [m["dynamics_jepa"], m["dynamics_persist"]],
            color=["red", "gray"], width=0.55)
    ax1.set_xticks([0, 1], [f"JEPA\n{m['dynamics_jepa']:.4f}",
                            f"persistence\n{m['dynamics_persist']:.4f}"])
    ax1.set_ylabel("mean error")
    ax1.set_title("Dynamics: persistence wins ~5x")
    ax1.set_ylim(0, 0.034)
    for x, v in [(0, m["dynamics_jepa"]), (1, m["dynamics_persist"])]:
        ax1.text(x, v + 0.001, f"{v:.4f}", ha="center", fontsize=10)
    f1 = [m["f1_lumiere_cv"], m["f1_sailor_transfer"]]
    ax2.bar([0, 1], f1, color=["#1f77b4", "#2ca02c"], width=0.55)
    ax2.axhline(m["majority_f1"], color="gray", ls="--", lw=1.2,
                label=f"majority ~{m['majority_f1']:.2f}")
    ax2.set_xticks([0, 1], ["LUMIERE\nin-domain", "SAILOR\ntransfer"])
    ax2.set_ylabel("macro-F1")
    ax2.set_title("Readouts: transfer ~= in-domain, zero training")
    ax2.set_ylim(0, 0.5)
    for i, v in enumerate(f1):
        ax2.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.text(1, 0.30, f"AUC {m['surprise_auc_sailor']}\n(n={m['surprise_n']})",
             ha="center", fontsize=9, color="#2ca02c")
    fig.suptitle("Cross-site (LUMIERE to SAILOR): dynamics fail, representation transfers",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_gap_bins(m, out):
    bins, pairs, jepa, persist = m["bins"], m["pairs"], m["jepa"], m["persist"]
    x = list(range(len(bins)))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([i - w / 2 for i in x], jepa, w, label="JEPA (frozen champion)", color="red")
    ax.bar([i + w / 2 for i in x], persist, w, label="persistence (no change)", color="gray")
    ax.set_xticks(x, [f"{b}\n(n={p})" for b, p in zip(bins, pairs)])
    ax.set_ylabel("mean error")
    ax.set_title(f"Cross-site by gap: persistence wins every bin "
                 f"(n={m['n_pairs']}, median gap {m['gap_median']}d)")
    for i in x:
        ax.text(i - w / 2, jepa[i] + 0.0009, f"{jepa[i]:.4f}", ha="center", fontsize=9)
        ax.text(i + w / 2, persist[i] + 0.0009, f"{persist[i]:.4f}", ha="center", fontsize=9)
    ax.set_ylim(0, max(jepa) * 1.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_gaphead(m, out):
    bins, pairs, champ, gap = m["bins"], m["pairs"], m["jepa"], m["gap_head"]
    persist = m["persist"]
    x = list(range(len(bins)))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar([i - w / 2 for i in x], champ, w, label="champion 1-step", color="#D55E00")
    ax.bar([i + w / 2 for i in x], gap, w, label="gap-conditioned head", color="#0072B2")
    ax.plot(x, persist, "D-", color="black", ms=6, lw=1.2, label="persistence (still wins)")
    ax.set_xticks(x, [f"{b}\n(n={p})" for b, p in zip(bins, pairs)])
    ax.set_ylabel("mean error")
    ax.set_title("Gap-conditioned head beats champion head, still loses to persistence")
    for i in x:
        pct = (champ[i] - gap[i]) / champ[i] * 100
        ax.text(i - w / 2, champ[i] + 0.0009, f"{champ[i]:.4f}", ha="center", fontsize=9)
        ax.text(i + w / 2, gap[i] + 0.0009, f"{gap[i]:.4f}", ha="center", fontsize=9)
        ax.text(i, max(champ[i], gap[i]) + 0.003, f"-{pct:.0f}%",
                ha="center", fontsize=9, color="#0072B2", weight="bold")
        ax.text(i, persist[i] + 0.0012, f"{persist[i]:.4f}", ha="center",
                fontsize=8, color="black")
    ax.set_ylim(0, max(champ) * 1.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_field_rescue(m, out):
    bins, pairs = m["bins"], m["pairs"]
    champ, persist, cond = m["jepa"], m["persist"], m["field_cond"]
    x = list(range(len(bins)))
    w = 0.3
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar([i - w for i in x], champ, w, label="champion head", color="#D55E00")
    ax.bar(x, cond, w, label="velocity field", color="#0072B2")
    mid = [i - w / 2 for i in x]
    ax.plot(mid, persist, "D-", color="black", ms=6, lw=1.2, label="persistence")
    ax.set_xticks(x, [f"{b}\n(n={p})" for b, p in zip(bins, pairs)])
    ax.set_ylabel("mean error")
    ax.set_title("Velocity field beats champion ~8x, at/below persistence")
    for i in x:
        ax.text(i - w, champ[i] + 0.0009, f"{champ[i]:.4f}", ha="center", fontsize=9)
        ax.text(i, max(cond[i], persist[i]) + 0.0012, f"{cond[i]:.4f}",
                ha="center", fontsize=9, color="#0072B2")
        ax.text(i - w / 2, persist[i] - 0.0022, f"{persist[i]:.4f}", ha="center",
                fontsize=8, color="black")
    ax.set_ylim(0, max(champ) * 1.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_field_ablation(m, out):
    bins, pairs = m["bins"], m["pairs"]
    persist, cond, uncond = m["persist"], m["field_cond"], m["field_uncond"]
    x = list(range(len(bins)))
    w = 0.22
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar([i - w for i in x], cond, w, label="treatment phase", color="#0072B2")
    ax.bar(x, uncond, w, label="constant phase", color="#56B4E9")
    ax.bar([i + w for i in x], persist, w, label="persistence", color="black")
    ax.set_xticks(x, [f"{b}\n(n={p})" for b, p in zip(bins, pairs)])
    ax.set_ylabel("mean error")
    ax.set_title("Ablation: exact tie; field beats persistence 3/4 bins")
    for i in x:
        if abs(cond[i] - uncond[i]) < 0.00005:
            ax.text(i - w / 2, max(cond[i], uncond[i]) + 0.00022,
                    f"{cond[i]:.4f} (tie)", ha="center", fontsize=9,
                    color="#0072B2")
        else:
            ax.text(i - w, cond[i] + 0.00022, f"{cond[i]:.4f}", ha="center", fontsize=9)
            ax.text(i, uncond[i] + 0.00022, f"{uncond[i]:.4f}", ha="center", fontsize=9)
        ax.text(i + w, persist[i] + 0.00022, f"{persist[i]:.4f}", ha="center", fontsize=9)
    ax.set_ylim(0, 0.0075)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_site_shift(m, out):
    import numpy as np
    lum = np.array(m["lumiere_xy"])
    sai = np.array(m["sailor_xy"])
    e1, e2 = m["explained_pct"]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(lum[:, 0], lum[:, 1], s=10, alpha=0.45, color="#1f77b4",
               label=f"LUMIERE visits (n={m['n_lumiere']})")
    ax.scatter(sai[:, 0], sai[:, 1], s=14, alpha=0.6, color="#D55E00",
               label=f"SAILOR visits (n={m['n_sailor']})")
    ax.set_xlabel(f"PC1 ({e1}%)")
    ax.set_ylabel(f"PC2 ({e2}%)")
    ax.set_title("Site shift")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_coral(m, out):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    cats = ["frozen head", "persistence"]
    x = [0, 1]
    w = 0.26
    ax.bar(x[0] - w, m["head_unaligned"], w, color="#D55E00", label="head, as-is")
    ax.bar(x[0], m["head_aligned"], w, color="#0072B2", label="head, CORAL-aligned")
    ax.bar(x[0] + w, m["head_transductive"], w, color="#56B4E9",
           label="head, transductive bound")
    ax.bar(x[1] - w / 2, m["persist_unaligned"], w, color="black", label="persistence, as-is")
    ax.bar(x[1] + w / 2, m["persist_aligned"], w, color="#777777",
           label="persistence, aligned")
    ax.set_xticks(x, cats)
    ax.set_ylabel("mean error")
    ax.set_title("Statistics-matching alone cuts head error 2.4x (zero training)")
    for v, xp in [(m["head_unaligned"], x[0] - w), (m["head_aligned"], x[0]),
                  (m["head_transductive"], x[0] + w)]:
        ax.text(xp, v + 0.0007, f"{v:.4f}", ha="center", fontsize=9)
    for v, xp in [(m["persist_unaligned"], x[1] - w / 2),
                  (m["persist_aligned"], x[1] + w / 2)]:
        ax.text(xp, v + 0.0007, f"{v:.4f}", ha="center", fontsize=9)
    ax.annotate("-59% (2.4x)", xy=(x[0], m["head_aligned"]),
                xytext=(0.5, 0.022), ha="center", fontsize=10, color="#0072B2",
                weight="bold",
                arrowprops=dict(arrowstyle="->", color="#0072B2", lw=1.5))
    ax.text(0.5, 0.016, "encoder untouched\nresidual = higher-order shift",
            ha="center", fontsize=9, color="dimgray")
    ax.set_ylim(0, 0.034)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_raw_mni(m, out):
    slots, n = m["slots"], m["n"]
    raw, mni = m["raw_med"], m["mni_med"]
    x = list(range(len(slots)))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar([i - w / 2 for i in x], raw, w, label="raw sessions", color="#D55E00")
    ax.bar([i + w / 2 for i in x], mni, w, label="MNI pipeline", color="#0072B2")
    ax.set_xticks(x, [f"{s}\n(n={p})" for s, p in zip(slots, n)])
    ax.set_ylabel("median mean|change|")
    ax.set_title("Same pairs: raw change ~4x MNI change (pipeline damps)")
    for i in x:
        ax.text(i - w / 2, raw[i] + 0.015, f"{raw[i]:.3f}", ha="center", fontsize=9)
        ax.text(i + w / 2, mni[i] + 0.015, f"{mni[i]:.3f}", ha="center", fontsize=9)
        ax.text(i, max(raw[i], mni[i]) + 0.06, f"{raw[i] / mni[i]:.1f}x",
                ha="center", fontsize=9, color="#0072B2", weight="bold")
    ax.set_ylim(0, max(raw) * 1.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_site_shift_aligned(m, out):
    import numpy as np
    lum = np.array(m["lumiere_xy"])
    sai = np.array(m["sailor_xy"])
    e1, e2 = m["explained_pct"]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(lum[:, 0], lum[:, 1], s=10, alpha=0.45, color="#1f77b4",
               label=f"LUMIERE visits (n={m['n_lumiere']})")
    ax.scatter(sai[:, 0], sai[:, 1], s=14, alpha=0.6, color="#D55E00",
               label=f"SAILOR visits, CORAL-aligned (n={m['n_sailor']})")
    ax.set_xlabel(f"PC1 ({e1}%)")
    ax.set_ylabel(f"PC2 ({e2}%)")
    ax.set_title("After CORAL: shared frame, residual structure remains")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_sailor_reprocessed(m, out):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))
    x = [0, 1, 3, 4]
    vals = [m["deriv_jepa"], m["deriv_persist"],
            m["dynamics_jepa"], m["dynamics_persist"]]
    cols = ["red", "gray", "red", "gray"]
    ax1.bar(x, vals, color=cols, width=0.55)
    ax1.set_xticks(x, ["deriv\nJEPA", "deriv\npersist", "reproc\nJEPA",
                       "reproc\npersist"])
    ax1.set_ylabel("mean error")
    ax1.set_title("Gate does not flip: persistence wins both arms")
    ax1.set_ylim(0, max(vals) * 1.3)
    for xi, v in zip(x, vals):
        ax1.text(xi, v + 0.0007, f"{v:.4f}", ha="center", fontsize=9)
    f1 = [m["deriv_transfer_f1"], m["transfer_f1"]]
    ax2.bar([0, 1], f1, color=["#2ca02c", "#d62728"], width=0.55)
    ax2.set_xticks([0, 1], ["deriv\ntransfer", "reproc\ntransfer"])
    ax2.set_ylabel("macro-F1")
    ax2.set_title("Transfer readout drops (maj 0.48, AUC "
                  f"{m['deriv_surprise_auc']:.2f}->{m['surprise_auc']:.2f})")
    ax2.set_ylim(0, 0.5)
    for i, v in enumerate(f1):
        ax2.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=10)
    fig.suptitle("SAILOR reprocessed (LUMIERE-contract): dynamics still lose, "
                 "readouts drop",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_reproc_vs_persist(m, out):
    r = m["sailor_reprocessed"]
    g = m["sailor_gap_bins"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    x = [0, 1, 3, 4]
    vals = [r["deriv_jepa"], r["deriv_persist"],
            r["dynamics_jepa"], r["dynamics_persist"]]
    ax1.bar(x, vals, color=["red", "gray", "red", "gray"], width=0.55)
    ax1.set_xticks(x, ["deriv\nJEPA", "deriv\npersist", "reproc\nJEPA",
                       "reproc\npersist"])
    ax1.set_ylabel("mean error (pooled, n=243)")
    ax1.set_title("JEPA error falls 0.0290->0.0245, gap widens ~5x->~6.6x")
    ax1.set_ylim(0, max(vals) * 1.3)
    for xi, v in zip(x, vals):
        ax1.text(xi, v + 0.0007, f"{v:.4f}", ha="center", fontsize=9)
    bins = g["bins"]
    xi = list(range(len(bins)))
    ax2.plot(xi, g["jepa"], "ro-", label="deriv JEPA", ms=5)
    ax2.plot(xi, r["jepa"], "r^--", label="reproc JEPA", ms=5)
    ax2.plot(xi, g["persist"], "ko-", label="deriv persist", ms=5)
    ax2.plot(xi, r["persist"], "k^--", label="reproc persist", ms=5)
    ax2.set_xticks(xi, [f"{b}\n(n={p})" for b, p in zip(bins, g["pairs"])])
    ax2.set_ylabel("mean error (log)")
    ax2.set_yscale("log")
    ax2.set_title("Every bin: JEPA down a little, persistence down more")
    ax2.legend(fontsize=8)
    fig.suptitle("Reprocess vs derivatives, same 243 pairs: representation "
                 "error improves, gate does not",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    m = load_metrics()
    plot_hero_leg1(m["hero_leg1"], os.path.join(HERE, "hero_leg1_val.png"))
    plot_hero_leg2(m["hero_leg2"], os.path.join(HERE, "hero_leg2_val.png"))
    plot_run6(m["run6"], os.path.join(HERE, "run6_tradeoff.png"))
    plot_sailor_transfer(m["sailor_transfer"], os.path.join(HERE, "sailor_transfer.png"))
    plot_gap_bins(m["sailor_gap_bins"], os.path.join(HERE, "sailor_gap_bins.png"))
    plot_gaphead(m["sailor_gap_bins"], os.path.join(HERE, "sailor_gaphead_vs_champ.png"))
    plot_field_rescue(m["sailor_gap_bins"], os.path.join(HERE, "sailor_field_rescue.png"))
    plot_field_ablation(m["sailor_gap_bins"], os.path.join(HERE, "sailor_field_ablation.png"))
    plot_site_shift(m["site_shift"], os.path.join(HERE, "site_shift.png"))
    plot_site_shift_aligned(m["site_shift_aligned"],
                            os.path.join(HERE, "site_shift_aligned.png"))
    plot_coral(m["coral"], os.path.join(HERE, "coral_alignment.png"))
    plot_raw_mni(m["raw_mni"], os.path.join(HERE, "raw_vs_mni.png"))
    plot_sailor_reprocessed(m["sailor_reprocessed"],
                            os.path.join(HERE, "sailor_reprocessed.png"))
    plot_reproc_vs_persist(m, os.path.join(HERE, "sailor_reproc_vs_persist.png"))
    print("wrote 14 plots to", HERE)


if __name__ == "__main__":
    main()
