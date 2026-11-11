"""Render the later research record from public aggregates, without patient data.

Run with .venv/bin/python -B info/plots/make_current_plots.py.
"""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INFO = Path(__file__).resolve().parents[1]
OUT = INFO / "plots" / "current"
BLUE, GREEN, RED, GRAY = "#0F4D92", "#8BCF8B", "#B64342", "#CFCECE"
RECORDS = []
SOURCES = {}


def load(name):
    path = INFO / name
    SOURCES[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def save(fig, name, title, caption, sources, values):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.suptitle(title, fontfamily="IBM Plex Serif", fontsize=12)
    for extension in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{extension}", dpi=300, facecolor="white")
    plt.close(fig)
    RECORDS.append(
        dict(name=name, title=title, caption=caption, sources=sources, values=values)
    )


def panel(title, xlabel, rows, interval="ci95"):
    return dict(title=title, xlabel=xlabel, rows=rows, interval=interval)


def forest(name, title, panels, caption, sources):
    height = max(3.2, 1.9 + 0.48 * max(len(p["rows"]) for p in panels))
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(7.2, height * len(panels)),
        layout="constrained",
        squeeze=False,
    )
    values = []
    for ax, p in zip(axes[:, 0], panels):
        ax.axvline(0, color="#666666", ls="--", lw=0.8)
        labels = []
        for y, (label, row) in enumerate(p["rows"]):
            value = row["difference"]
            lo, hi = row[p["interval"]]
            if not np.isfinite([value, lo, hi]).all() or lo > hi:
                raise ValueError(f"Invalid interval in {name}: {label}")
            ax.plot([lo, hi], [y, y], color=BLUE, lw=1.5)
            ax.plot([lo, hi], [y, y], "|", color=BLUE, ms=7)
            ax.plot(value, y, "o", color=BLUE, mec="black", mew=0.5, ms=5)
            labels.append(label)
            values.append(
                dict(
                    panel=p["title"],
                    label=label,
                    difference=value,
                    interval=[lo, hi],
                    interval_key=p["interval"],
                )
            )
        ax.set(
            yticks=range(len(labels)),
            yticklabels=labels,
            title=p["title"],
            xlabel=p["xlabel"],
            ylim=(len(labels) - 0.5, -0.5),
        )
        ax.margins(x=0.12)
        ax.tick_params(axis="y", length=0)
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    save(fig, name, title, caption, sources, values)


def comparison_figures():
    f = "plots/reviewed_reference_metrics.json"
    d = load(f)
    forest(
        "r35_state_and_change",
        "R35: state learning does not establish added paired information",
        [
            panel(
                "Current state; different controls by cohort",
                "MAE difference on log1p(mm³); negative favors spatial state",
                list(d["r35"]["state"].items()),
            ),
            panel(
                "Observed change: paired head minus independent state subtraction",
                "MAE difference on log1p(mm³); negative favors paired head",
                list(d["r35"]["paired_minus_independent"].items()),
                "ci_family12",
            ),
        ],
        "State intervals are ordinary 95% patient bootstraps. Observed-change intervals are family-12 "
        "adjusted 99.583% intervals. Scores average three fitted seeds. LUMIERE has 149 visits from 26 "
        "patients and 117 pairs from 23; SAILOR has 269 visits and 242 pairs from 27. The spatial model "
        "also receives dense mask supervision, so the state gain does not isolate architecture. "
        "Both scans are observed for paired change. Cohort-specific controls and panel scales differ. "
        "The JSON cites the reviewed numerical receipts, including the corrected independent-state comparison.",
        [f],
    )
    a = d["corrected_adaptation"]
    fig, ax = plt.subplots(figsize=(7.2, 3.5), layout="constrained")
    for arm, marker, color in [("JEPA", "o", BLUE), ("CNN control", "s", RED)]:
        ax.plot(a["support_patients"], a[arm], marker=marker, color=color, label=arm)
    ax.set(
        xlabel="Support patients; 0 means zero-shot",
        ylabel="Macro-F1; higher is better",
        xticks=a["support_patients"],
        ylim=(0, 0.35),
    )
    ax.legend()
    save(
        fig,
        "corrected_adaptation",
        "Corrected-cache adaptation reverses the old zero-shot ordering",
        "Recorded corrected SAILOR response-category scores. Lines connect evaluated support sizes; "
        "no uncertainty estimates are supplied. The CNN column is the recorded control, not the "
        "entire historical CNN sweep. SAILOR is reused development data and the response-code "
        "mapping lacks independent confirmation. These curves supersede the legacy adaptation "
        "plots as a description of the corrected cache; they are not anatomical transfer evidence.",
        [f],
        a,
    )

    f = "r36_phase1_metrics.json"
    d = load(f)
    forest(
        "r36_observed_change",
        "R36: added encoder accessibility remains unresolved",
        [
            panel(
                "Matched frozen readouts; both scans observed",
                "Local occupancy MAE difference; negative favors candidate",
                [
                    (label, d["comparisons"][key])
                    for key, label in [
                        ("C-A", "Change-balanced − state-only"),
                        ("C-B", "Change-balanced − unbalanced"),
                        ("D-C", "Nuisance-trained − change-balanced"),
                    ]
                ],
                "family_adjusted_ci",
            )
        ],
        "Points show paired patient-uniform error differences across three fitted seeds. Lines are "
        "family-three adjusted 98.33% patient-bootstrap intervals on 117 observed pairs from 23 LUMIERE "
        "development patients. A is state-only, B is unbalanced change, C is change-balanced, and D adds "
        "nuisance training. This is observed change, not forecasting. Negative values favor the first arm.",
        [f],
    )

    f = "r38_full_metrics.json"
    d = load(f)
    forest(
        "r38_joint_training",
        "R38: joint training did not establish a forecasting gain",
        [
            panel(
                "Forecast contrasts",
                "CRPS difference on log1p(mm³); negative favors joint history",
                [
                    (label, d["primary_contrasts"][key])
                    for key, label in [
                        ("JH-JL", "Joint history − joint latest"),
                        ("JH-OH", "Joint history − detached history"),
                        ("JH-P", "Joint history − persistence"),
                        ("JH-E", "Joint history − empirical change"),
                    ]
                ],
                "ci_99_5",
            )
        ],
        "Lines are adjusted 99.5% patient-bootstrap intervals for the selected forecast contrasts. "
        "The fixed comparison uses 75 pairs from 18 reused LUMIERE patients and averages fitted seed scores. "
        "Joint history also failed current-state retention. Intervals crossing zero do not establish equivalence.",
        [f],
    )

    f = "r40_full_metrics.json"
    d = load(f)
    forest(
        "r40_supplied_horizon",
        "R40: a supplied horizon did not establish useful history",
        [
            panel(
                "History with supplied horizon minus each control",
                "CRPS difference on log1p(mm³); negative favors history",
                [
                    (label, row)
                    for label, row in zip(
                        [
                            "Latest + horizon",
                            "Persistence",
                            "Empirical change",
                            "Ridge + horizon",
                        ],
                        d["comparisons"].values(),
                    )
                ],
            )
        ],
        "Points are paired patient-uniform CRPS differences; lines are the recorded ordinary 95% "
        "patient-bootstrap intervals, not multiplicity-adjusted intervals. The assessment has 75 pairs "
        "from 18 reused LUMIERE patients. The realized week-bin horizon is supplied retrospectively. "
        "This does not test prospective prediction at a known appointment time.",
        [f],
    )

    f = "preaudit_revisit_metrics.json"
    units = load(f)["units"]

    def aggregate(study, component):
        return units[study]["components"][component]["aggregate"]

    rows = [
        (
            "R41: native crop − whole image",
            aggregate("R41", "cnn")["contrasts"]["native-H_minus_whole-H"],
        ),
        (
            "R42: pretrained − random ViT",
            aggregate("R42", "vit")["contrasts"]["pretrained-H_minus_randomvit-H"],
        ),
        ("R44: postoperative context", aggregate("R44", "postop")["contrast"]),
        ("R46: scanner context", aggregate("R46", "scanner")["contrast"]),
    ]
    forest(
        "r41_r44_r46_forecast_controls",
        "Forecasting controls on reused LUMIERE patients",
        [
            panel(
                "Separate exploratory pilots; 18 scoring patients each",
                "CRPS difference on log1p(mm³); negative favors candidate",
                rows,
            )
        ],
        "Lines are ordinary 95% patient-bootstrap intervals. Each row tests its own matched control: "
        "native versus whole-image history, pretrained versus random ViT history, and inclusion versus "
        "omission of postoperative or scanner context. These adaptive pilots are not independent "
        "replications or a pooled estimate. Small pilot budgets limit conclusions about convergence.",
        [f],
    )
    horizons = aggregate("R43", "dynamics")["horizons"]
    forest(
        "r43_rollout",
        "R43: continuous dynamics versus matched discrete transitions",
        [
            panel(
                "Propagated rollout",
                "CRPS difference on log1p(mm³); negative favors continuous dynamics",
                [
                    (
                        f"Step {step}; {h['contrasts']['continuous_minus_discrete']['patients']} patients",
                        h["contrasts"]["continuous_minus_discrete"],
                    )
                    for step, h in horizons.items()
                ],
            )
        ],
        "Points are continuous-minus-discrete CRPS differences and lines are ordinary 95% "
        "patient-bootstrap intervals. The same fitted models propagate their predictions through "
        "successive steps; the eligible patient subset changes with horizon. Time is nominal LUMIERE "
        "week-bin time, not exact scan-day time. There is no pooling across horizons.",
        [f],
    )
    forest(
        "r45_r46_response_controls",
        "R45/R46: response assessment and perfusion forecasting",
        [
            panel(
                "R45: observed binary response; 16 LUMIERE patients",
                "Log-loss difference; negative favors candidate",
                [
                    (
                        "R45: paired − separate encoders",
                        aggregate("R45", "response")["contrasts"][
                            "paired_minus_separate"
                        ],
                    ),
                    (
                        "R45: paired − later image",
                        aggregate("R45", "response")["contrasts"][
                            "paired_minus_current"
                        ],
                    ),
                ],
            ),
            panel(
                "R46: anatomical forecasting; 27 SAILOR patients",
                "CRPS difference on log1p(mm³); negative favors perfusion",
                [
                    (
                        "R46: perfusion − structural",
                        aggregate("R46", "perfusion")["contrasts"][
                            "perfusion_minus_structural"
                        ],
                    ),
                    (
                        "R46: perfusion − missingness",
                        aggregate("R46", "perfusion")["contrasts"][
                            "perfusion_minus_missingness"
                        ],
                    ),
                ],
            ),
        ],
        "Lines are ordinary 95% patient-bootstrap intervals. R45 assesses binary response after "
        "both scans are observed, on 48 pairs from 16 reused LUMIERE patients. R46 forecasts anatomy "
        "from cutoff structural measurements with or without perfusion, on 238 pairs from 27 "
        "SAILOR development patients. Panels have different units, tasks, and scales. Neither "
        "comparison establishes the tested incremental benefit.",
        [f],
    )

    f = "r47_anatomical_signal_metrics.json"
    d = load(f)
    panels = []
    for group, title, prefix in [
        ("source", "LUMIERE: 48 development patients", "forecast/"),
        ("transfer", "SAILOR: 27 development patients", ""),
    ]:
        panels.append(
            panel(
                title,
                "CRPS difference on log1p(mm³); negative favors first arm",
                [
                    (
                        "Current + horizon − persistence",
                        d[group]["contrasts"][prefix + "Lq_minus_P"],
                    ),
                    (
                        "History + horizon − current + horizon",
                        d[group]["contrasts"][prefix + "Hq_minus_Lq"],
                    ),
                ],
                "ci97_5",
            )
        )
    forest(
        "r47_current_state_transfer",
        "R47: current-state gain in-domain, no established history gain",
        panels,
        "Lines are recorded adjusted 97.5% patient-bootstrap intervals. Current measured anatomy and "
        "the retrospectively supplied interval improve the in-domain distributional score against "
        "persistence. The history increment is unresolved and the current-state advantage does not "
        "transfer. Panels use independent axis limits. Source results average two repeated nested "
        "partitions; locked source models supply developmental transfer predictions.",
        [f],
    )

    f = "r49_pretraining_state_metrics.json"
    d = load(f)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), layout="constrained")
    values = []
    for ax, metric, label in zip(
        axes,
        ["dice", "mae"],
        ["Nonempty Dice; higher is better", "Log1p(mm³) MAE; lower is better"],
    ):
        row = d["contrasts"][metric]
        for y, arm, marker, color in [
            (0, "pretrained", "o", BLUE),
            (1, "random", "s", GRAY),
        ]:
            value = row["arm_seed_mean_scores"][arm]
            ax.plot(value, y, marker, color=color, mec="black", ms=7)
            ax.annotate(
                f"{value:.4f}",
                (value, y),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
            )
            values.append(dict(metric=metric, arm=arm, value=value))
        ax.set(
            yticks=[0, 1],
            yticklabels=["Pretrained", "Random"],
            ylim=(1.6, -0.6),
            xlabel=label,
        )
        ax.margins(x=0.55)
    save(
        fig,
        "r49_pretraining_scores",
        "R49: matched pretraining improves overlap, absolute Dice stays low",
        "Points are means of patient-uniform scores over two fitted seeds on 60 visits from 13 reused "
        "LUMIERE scoring patients. No arm-wise uncertainty is supplied here. Architecture, supervision, "
        "patient draws, and selection opportunities are matched. Dice is unitless with 1 indicating "
        "perfect overlap. The next figure gives the paired uncertainty and volume-retention criterion.",
        [f],
        values,
    )
    forest(
        "r49_pretraining_contrasts",
        "R49: paired evidence for an initialization benefit",
        [
            panel(
                "Nonempty mask overlap",
                "Pretrained − random Dice; positive favors pretraining",
                [("Nonempty Dice", d["contrasts"]["dice"])],
                "ci97_5",
            ),
            panel(
                "Current volume error",
                "Pretrained − random MAE on log1p(mm³); negative favors pretraining",
                [("Volume MAE", d["contrasts"]["mae"])],
                "ci97_5",
            ),
        ],
        "Lines are adjusted 97.5% patient-bootstrap intervals over 13 scoring patients. Both seeds "
        "favor pretraining for Dice. The volume interval crosses zero but its upper limit passes "
        "the specified +0.05 retention allowance. This supports initialization under the matched "
        "recipe, not clinical utility or temporal understanding. Panels have different units.",
        [f],
    )

    files = [
        "r50_r51_clinical_metrics.json",
        "r50_r51_v2_clinical_metrics.json",
        "r50_r51_v3_clinical_metrics.json",
    ]
    panels = []
    for version, f in zip(["v1", "v2", "v3"], files):
        d = load(f)
        panels.append(
            panel(
                version,
                "Calibrated log-loss difference; negative favors first arm",
                [
                    (label, d["contrasts"][key])
                    for key, label in [
                        ("direct_minus_densenet", "Direct BrainIAC − DenseNet"),
                        ("jepa_minus_direct", "JEPA − direct supervision"),
                        ("jepa_minus_shuffled", "JEPA − shuffled teacher"),
                    ]
                ],
                "ci_family3",
            )
        )
    forest(
        "r50_r51_clinical_objectives",
        "R50/R51: no accepted primary JEPA gain across the three recipes",
        panels,
        "Lines are family-three adjusted 98.33% patient-bootstrap intervals for observed four-class "
        "response log loss on 10 reused scoring patients. Versions change bundled training choices "
        "and are not a controlled single-factor ablation. v3 recovers fitting readiness; v1 and v2 "
        "retain their readiness limitations. Independent panel scales preserve visible uncertainty. "
        "The observed-response task is not future forecasting.",
        files,
    )

    f = "r52_spatial_token_encoder_metrics.json"
    d = load(f)
    forest(
        "r52_token_interface",
        "R52: a token-interface replacement was not established",
        [
            panel(
                "Fresh token-only anatomical readouts",
                "CNN − BrainIAC nonempty Dice; positive favors CNN",
                [("Nonempty Dice", d["contrasts"]["probe"]["dice"])],
                "ci_family3",
            ),
            panel(
                "Volume retention",
                "CNN − BrainIAC MAE on log1p(mm³); negative favors CNN",
                [("Volume MAE", d["contrasts"]["probe"]["mae"])],
                "ci_family3",
            ),
        ],
        "The candidate is the compact CNN, so all signs are CNN minus BrainIAC. Lines are family-three "
        "adjusted 98.33% patient-bootstrap intervals on 13 reused LUMIERE patients. Both encoders use "
        "the same 216 by 64 token interface and fresh readouts. Mean Dice is about 0.0666 for both. "
        "The figure does not identify whether information was lost in the backbone, interface, or readout.",
        [f],
    )

    f = "r53_flair_prediction_metrics.json"
    d = load(f)
    forest(
        "r53_image_prediction",
        "R53: small completion gain, no useful image-forecast gain",
        [
            panel(
                "Standardized FLAIR completion",
                "Image MAE difference; negative favors pretraining",
                [("Pretrained − random", d["contrasts"]["completion_pretraining"])],
                "ci_family5",
            ),
            panel(
                "Standardized future FLAIR",
                "Image MAE difference; negative favors first arm",
                [
                    (label, d["contrasts"][key])
                    for key, label in [
                        ("forecast_pretraining", "Pretrained − random"),
                        ("forecast_history", "History − latest"),
                        ("forecast_persistence", "History − persistence"),
                        ("forecast_mean_change", "History − mean change"),
                    ]
                ],
                "ci_family5",
            ),
        ],
        "Lines are family-five adjusted 99% patient-bootstrap intervals. Completion uses 13 scoring "
        "patients and forecasting uses 10. Errors are in standardized image-intensity units, not "
        "physical tissue units. The small completion improvement misses the fixed useful-gain "
        "criterion. Separate panel scales avoid hiding the near-zero forecast contrasts.",
        [f],
    )

    files = ["r56_regional_state_metrics.json", "r57_regional_graph_metrics.json"]
    a, b = [load(f) for f in files]
    forest(
        "r56_r57_regional_context",
        "Regional context helps one current-state endpoint",
        [
            panel(
                "R56: current-state assimilation",
                "Overall regional log-count MAE difference; negative favors history",
                [("History filter − latest", a["primary"])],
            ),
            panel(
                "R57: context, adjacency, and history",
                "Nonempty regional log-count MAE difference; negative favors graph history",
                [
                    (label, b["contrasts"][key])
                    for key, label in [
                        ("SH", "Graph history − self-only"),
                        ("RH", "Graph history − shuffled adjacency"),
                        ("GL", "Graph history − graph latest"),
                    ]
                ],
                "ci_family3",
            ),
        ],
        "R56 uses ordinary 95% patient-bootstrap intervals; R57 uses family-three adjusted 98.33% "
        "intervals. Both score 10 reused patients. R56 scores all regions; R57's primary endpoint "
        "scores nonempty regions. Context improves R57 against self-only messages, but adjacency "
        "and history increments remain unresolved. These coarse log-count targets predate R60's "
        "source-volume correction and require separate confirmation. Panels have different endpoints.",
        files,
    )

    f = "r58_spatial_latent_forecast_metrics.json"
    d = load(f)
    forest(
        "r58_future_features",
        "R58: future-feature alignment did not improve anatomical forecasting",
        [
            panel(
                "JEPA history minus each control",
                "Regional log-count MAE difference; negative favors JEPA history",
                [
                    (label, d["contrasts"][key])
                    for key, label in [
                        ("DH", "Direct history"),
                        ("SH", "Shuffled teacher history"),
                        ("JL", "JEPA latest"),
                    ]
                ],
                "ci_family3",
            )
        ],
        "Lines are family-three adjusted 98.33% patient-bootstrap intervals on eight reused scoring "
        "patients. Current automated measurements are supplied at inference; future inputs are "
        "excluded. Teacher and encoder are frozen. Targets are the coarse regional log-count "
        "measurements later questioned by R59 and corrected in R60.",
        [f],
    )


def diagnostic_figures():
    f = "r48_learning_investigation_metrics.json"
    d = load(f)
    fig, ax = plt.subplots(figsize=(7.2, 3.5), layout="constrained")
    values = []
    for arm, marker, color in [("pretrained", "o", BLUE), ("random", "s", RED)]:
        curve = d["duration_intervention"]["arms"][arm]["curve"]
        x, y = [r["total_updates"] for r in curve], [r["mean_dice"] for r in curve]
        ax.plot(x, y, marker=marker, color=color, label=arm.capitalize())
        values.append(dict(arm=arm, updates=x, mean_dice=y))
    ax.set(
        xlabel="Optimizer updates",
        ylabel="Mean fitting Dice; higher is better",
        ylim=(0, 1),
    )
    ax.legend(loc="upper left")
    save(
        fig,
        "r48_fitting_duration",
        "R48: the short fitting check stopped too early",
        "Recorded checkpoints on the same two fitting examples. Lines connect measurements, with "
        "no uncertainty band or interpolated observations. Both arms pass the full original fitting "
        "criterion by 512 updates. This diagnoses optimization duration; it is not held-out "
        "generalization or evidence of pretrained superiority. Dice is unitless.",
        [f],
        values,
    )

    f = "r54_flair_learnability_metrics.json"
    d = load(f)
    fig, ax = plt.subplots(figsize=(7.2, 4.5), layout="constrained")
    labels = {
        "completion-F": "Completion: frozen",
        "completion-T": "Completion: trainable",
        "forecast-F-H": "Forecast: frozen history",
        "forecast-F-L": "Forecast: frozen latest",
        "forecast-T-H": "Forecast: trainable history",
        "forecast-T-L": "Forecast: trainable latest",
    }
    values = []
    for y, (key, label) in enumerate(labels.items()):
        reductions = np.array(d["decisions"][key]["relative_mae_reductions"]) * 100
        for i, marker, color, offset in [(0, "o", BLUE, -0.1), (1, "s", RED, 0.1)]:
            ax.plot(
                reductions[i],
                y + offset,
                marker=marker,
                color=color,
                label=f"Fitting example {i + 1}" if y == 0 else None,
                linestyle="none",
            )
        values.append(dict(arm=key, percent_mae_reduction=reductions.tolist()))
    ax.axvline(0, color="#666666", ls="--", lw=0.8)
    ax.set(
        yticks=range(len(labels)),
        yticklabels=list(labels.values()),
        ylim=(len(labels) - 0.5, -0.7),
        xlabel="MAE reduction versus best per-example control (%); positive is better",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    save(
        fig,
        "r54_individual_fitting",
        "R54: fitting remains weak even on two examples",
        "Points are the two examples' relative MAE reductions against their best image control, "
        "multiplied by 100. Shape and color identify examples; they are not bootstrap samples or "
        "error bars. The best control can differ between completion and forecasting examples. "
        "Frozen and trainable describe the MRI encoder. None meets the substantial-fit criterion. "
        "These are fitting-only diagnostics, not held-out performance.",
        [f],
        values,
    )

    f = "r55_decoder_capacity_metrics.json"
    d = load(f)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), layout="constrained")
    values = []
    for key, label, marker, color in [
        ("Z", "Codes only", "o", BLUE),
        ("ZD", "Codes + decoder", "s", RED),
        ("MRI", "MRI continuation", "^", "#555555"),
    ]:
        curve = d["cells"][key]["curve"]
        for ax, metric in zip(axes, ["mae", "gradient_mae"]):
            x = [r["step"] for r in curve]
            y = [r["scores"]["mean"][metric] for r in curve]
            ax.plot(x, y, marker=marker, color=color, label=label, ms=4)
            values.append(dict(arm=key, metric=metric, updates=x, values=y))
    axes[0].set(xlabel="Additional updates", ylabel="Image MAE; lower is better")
    axes[1].set(
        xlabel="Additional updates", ylabel="Image-gradient MAE; lower is better"
    )
    axes[0].legend(fontsize=8)
    save(
        fig,
        "r55_decoder_fitting",
        "R55: bypassing the MRI encoder did not resolve image fidelity",
        "Recorded mean scores on two current fitting images, in standardized intensity units. "
        "Codes-only and codes-plus-decoder runs fit target-specific spatial codes. MRI continuation "
        "uses the matched additional duration. Lines connect recorded checkpoints; no uncertainty "
        "is supplied. None met the full individual-image fitting criterion. These curves do not "
        "measure forecasting or generalization. R54's preceding fitting diagnosis also failed its "
        "substantial-fit criterion.",
        [f],
        values,
    )

    f = "r59_measured_history_metrics.json"
    d = load(f)
    forest(
        "r59_coarse_measured_history",
        "R59: measured-history comparison before target correction",
        [
            panel(
                "Historical coarse targets; eight scoring patients",
                "Regional log-count MAE difference; negative favors history",
                [
                    ("History − current only", d["contrasts"]["L"]),
                    ("History − persistence", d["contrasts"]["P"]),
                ],
                "ci_family2",
            )
        ],
        "Lines are family-two adjusted 97.5% patient-bootstrap intervals on 30 history pairs from "
        "eight reused patients. This comparison uses the old coarse-grid log-count targets. "
        "R59's separate inspection of six deliberately selected examples found erased burden "
        "and a change-direction disagreement; those examples cannot estimate prevalence. R60 "
        "quantifies the defect and R61 repeats the comparison with source-volume targets. "
        "Do not compare absolute R59 and R61 errors as a single-factor sampling correction: "
        "the logarithm's volume units also changed.",
        [f],
    )

    f = "r60_regional_source_target_metrics.json"
    d = load(f)
    groups = list(d["groups"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), layout="constrained")
    values = []
    for ax, numerator, denominator, title in [
        (
            axes[0],
            "positive_source_cells_erased_by_coarse_grid",
            "regional_compartment_cells",
            "Positive burden erased",
        ),
        (
            axes[1],
            "change_direction_disagreements",
            "adjacent_change_cells",
            "Opposite change direction",
        ),
    ]:
        ns = [d["groups"][g][numerator] for g in groups]
        ds = [d["groups"][g][denominator] for g in groups]
        rates = 100 * np.array(ns) / ds
        ax.bar(
            range(len(groups)),
            rates,
            color=RED,
            edgecolor="black",
            hatch="//",
            width=0.6,
        )
        for i, (n, denom, rate) in enumerate(zip(ns, ds, rates)):
            ax.annotate(
                f"{n}/{denom}",
                (i, rate),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        ax.set(
            xticks=range(len(groups)),
            xticklabels=[g.capitalize() for g in groups],
            ylabel="Share of all compared cells (%)",
            title=title,
            ylim=(0, max(rates) * 1.3),
        )
        values.append(
            dict(
                numerator=numerator,
                denominator=denominator,
                groups=groups,
                counts=ns,
                totals=ds,
            )
        )
    save(
        fig,
        "r60_target_correction",
        "R60: coarse sampling erased burden and reversed some changes",
        "Bars show endpoint-pooled counts divided by all compared region-compartment cells in each "
        "partition. Labels give numerator/denominator; these are not percentages among positive "
        "cells or patient-uniform forecast scores. No sampling uncertainty is supplied. Source-volume "
        "aggregation conserves physical volume. This is a target diagnostic across 322 visits, not "
        "a model comparison or proof of biological mask accuracy.",
        [f],
        values,
    )


def measured_history_figures():
    files = ["r61_corrected_history_metrics.json", "r62_mu_history_metrics.json"]
    studies = [load(f) for f in files]
    titles = ["R61: LUMIERE; 8 patients, 30 pairs", "R62: MU; 18 patients, 33 triples"]
    names = {
        "P": "Persistence",
        "E": "Mean change",
        "T": "Damped trend",
        "L": "Current only",
        "H": "History",
        "S": "Shuffled history",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4), layout="constrained")
    values = []
    for ax, d, title in zip(axes, studies, titles):
        for y, (key, label) in enumerate(names.items()):
            value = d["scores"][key]["mae"]
            color, marker = (
                (BLUE, "o") if key == "H" else (GRAY, "s") if key == "P" else (RED, "^")
            )
            ax.plot(value, y, marker=marker, color=color, mec="black", ms=6)
            ax.annotate(
                f"{value:.5f}",
                (value, y),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
            )
            values.append(dict(study=d["study"], arm=key, mae=value))
        ax.set(
            yticks=range(len(names)),
            yticklabels=list(names.values()),
            ylim=(len(names) - 0.5, -0.5),
            xlabel="Regional log1p-mL MAE",
            title=title,
        )
        ax.margins(x=0.65)
        ax.tick_params(axis="y", length=0)
        ax.ticklabel_format(axis="x", useOffset=False)
    caption = (
        "Points show patient-uniform regional MAE of log(1 + volume in mL), not raw mL. "
        "Both earlier and current measurements are supplied to history regressions. Persistence "
        "copies current measured burden. Trend selects zero and coincides with persistence. "
        "No arm-wise uncertainty is supplied. Panels use independent scales and different "
        "anatomical tasks: three automated compartments in R61, four reviewed-mask compartments "
        "including cavity in R62. Absolute errors are not a cross-cohort leaderboard. All scoring "
        "patients are development evidence; MU's forty reserved patients remain unopened."
    )
    save(
        fig,
        "r61_r62_measured_history",
        "Corrected measured history still loses to persistence",
        caption,
        files,
        values,
    )
    forest(
        "r61_r62_history_intervals",
        "Small history increments do not beat persistence",
        [
            panel(
                title,
                "MAE difference on log1p-mL; negative favors history",
                [
                    ("History − current only", d["contrasts"]["L"]),
                    ("History − persistence", d["contrasts"]["P"]),
                ],
                "ci_family2",
            )
            for title, d in zip(titles, studies)
        ],
        "Lines are family-two adjusted 97.5% paired patient-bootstrap intervals. R61's history-minus-current "
        "interval excludes zero but the gain misses the fixed 5% practical threshold. R62's corresponding "
        "interval crosses zero. Both history-minus-persistence intervals favor persistence. Panels have "
        "independent scales and different compartment definitions; do not pool their estimates.",
        files,
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9), layout="constrained")
    values = []
    metrics = [
        ("mae", "Overall"),
        ("nonempty_mae", "Nonempty"),
        ("empty_mae", "Empty"),
        ("changed_mae", "Changed"),
    ]
    for ax, d, title in zip(axes, studies, titles):
        for key, offset, marker, color in [
            ("P", -0.1, "s", GRAY),
            ("H", 0.1, "o", BLUE),
        ]:
            scores = [d["scores"][key][m] for m, _ in metrics]
            ax.scatter(
                scores,
                np.arange(len(metrics)) + offset,
                marker=marker,
                color=color,
                edgecolor="black",
                label=names[key],
            )
            values.append(
                dict(
                    study=d["study"],
                    arm=key,
                    metrics=[m for m, _ in metrics],
                    values=scores,
                )
            )
        ax.set(
            yticks=range(len(metrics)),
            yticklabels=[label for _, label in metrics],
            ylim=(3.6, -0.6),
            xlabel="Regional log1p-mL MAE",
            title=title,
            xlim=(0, None),
        )
        ax.legend(loc="upper right", fontsize=8)
    save(
        fig,
        "r61_r62_error_strata",
        "Learned corrections help changed cells but add empty-cell error",
        "Descriptive patient-uniform error strata for the same R61 and R62 scoring cohorts. Empty "
        "and nonempty refer to the future target; changed cells exceed the fixed 0.1 log1p-mL "
        "absolute-change threshold. Strata overlap and must not be added. No stratum-specific intervals "
        "are supplied. Each panel has its own scale. The overall criterion still favors persistence.",
        files,
        values,
    )

    f = files[1]
    d = studies[1]
    fig, ax = plt.subplots(figsize=(7.2, 3.5), layout="constrained")
    groups = ["fitting", "selection", "pilot_scoring"]
    assigned = [d["cohorts"][g]["patients"] for g in groups]
    eligible = [
        d["metadata_eligibility"][g]["metadata_eligible_patients"] for g in groups
    ]
    x = np.arange(len(groups))
    ax.bar(
        x - 0.18,
        assigned,
        width=0.36,
        color=GRAY,
        edgecolor="black",
        label="Assigned development patients",
    )
    ax.bar(
        x + 0.18,
        eligible,
        width=0.36,
        color=BLUE,
        edgecolor="black",
        hatch="//",
        label="History-eligible patients",
    )
    for i, (n, k) in enumerate(zip(assigned, eligible)):
        for offset, value in [(-0.18, n), (0.18, k)]:
            ax.annotate(
                str(value),
                (i + offset, value),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
            )
    ax.set(
        xticks=x,
        xticklabels=["Fitting", "Selection", "Pilot scoring"],
        ylabel="Patients (count)",
        ylim=(0, max(assigned) * 1.4),
    )
    ax.legend(fontsize=8)
    save(
        fig,
        "r62_development_cohort",
        "MU development allocation and history eligibility",
        "Assigned and history-eligible patient counts are overlapping categories, not stacked groups. "
        "The eligible partitions provide 107, 47, and 33 triples respectively. Metadata anomalies and "
        "missing released files exclude affected comparisons. Forty separately reserved patients "
        "remain unopened and are not represented by image or mask measurements in this plot.",
        [f],
        dict(groups=groups, assigned=assigned, eligible=eligible),
    )


def main():
    RECORDS.clear()
    SOURCES.clear()
    plt.rcParams.update(
        {
            "font.family": ["IBM Plex Sans", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "savefig.dpi": 300,
        }
    )
    comparison_figures()
    diagnostic_figures()
    measured_history_figures()
    lines = [
        "# Research figures through R62",
        "",
        "Generated by `info/plots/make_current_plots.py` from public aggregate JSON. "
        "Figures are 7.2 inches wide, with vector PDF and 300-dpi PNG exports. "
        "[Review and coverage](../REVIEW.md) explains the older figures and remaining gaps.",
        "",
    ]
    for r in RECORDS:
        lines.extend(
            [
                f"## {r['title']}",
                "",
                f"![{r['title']}]({r['name']}.png)",
                "",
                r["caption"],
                "",
                f"[PDF]({r['name']}.pdf) · "
                + " · ".join(f"[Source aggregate](../../{f})" for f in r["sources"]),
                "",
            ]
        )
    (OUT / "README.md").write_text("\n".join(lines))
    (OUT / "plotted_values.json").write_text(
        json.dumps(dict(source_sha256=SOURCES, figures=RECORDS), indent=2) + "\n"
    )
    print(
        f"Wrote {len(RECORDS)} current figures, captions, and plotted-value provenance to {OUT}"
    )


if __name__ == "__main__":
    main()
