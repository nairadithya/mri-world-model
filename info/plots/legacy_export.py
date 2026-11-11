"""Export historical figures with their current evidential status attached."""

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.markers import MarkerStyle


WITHDRAWN = {
    "lesion_seed_relative_mae",
    "lesion_seed_delta_ci",
    "lesion_model_training_repeats",
    "lesion_model_patient_intervals",
}
SUPERSEDED = {"cross_site_adapt", "second_cohort_adaptation"}


def save_legacy(fig, stem, dpi=300, rect=None):
    stem = Path(stem)
    for ax in fig.axes:
        clouds = [
            c
            for c in ax.collections
            if isinstance(c, PathCollection) and not c.get_label().startswith("_")
        ]
        if len(clouds) > 1:
            legend = ax.get_legend()
            handles = legend.legend_handles if legend else []
            for collection, shape in zip(clouds, ("o", "s", "^", "D")):
                marker = MarkerStyle(shape)
                path = marker.get_path().transformed(marker.get_transform())
                collection.set_paths([path])
                for handle in handles:
                    if (
                        isinstance(handle, PathCollection)
                        and handle.get_label() == collection.get_label()
                    ):
                        handle.set_paths([path])
        title = ax.get_title()
        ax.set_title(
            "\n".join(textwrap.fill(line, 58) for line in title.splitlines()),
            fontsize=11,
        )
    if stem.name in WITHDRAWN:
        notice = "WITHDRAWN IN A40: future-visit inputs leaked into P2.4. Historical scores only."
    elif stem.name in SUPERSEDED:
        notice = "SUPERSEDED: corrected-cache adaptation results differ. See info/EXPERIMENTS.md."
    else:
        notice = "Historical development result. Scope and corrections: info/plots/REVIEW.md."
    if stem.name in {
        "run6_tradeoff",
        "response_training_comparison",
        "gauntlet_probe",
        "response_feature_comparison",
    }:
        notice += "\nCross-validation error bars show recorded fold SD, not confidence intervals."
    fig.tight_layout(pad=2, rect=rect)
    fig.text(
        0.5,
        -0.025,
        notice,
        ha="center",
        va="top",
        fontsize=9,
        color="#B64342" if stem.name in WITHDRAWN | SUPERSEDED else "#444444",
    )
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{extension}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
