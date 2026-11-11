# Report figures and tables

These assets are intended for a printed report at approximately 8 inches wide. Figure numbering is provisional. Aggregate figures are supplied as PDF and 300-dpi PNG; regenerate with `.venv/bin/python report/make_figures.py`. They contain no participant-level data. The separate [case figures and captions](CASE_FIGURES.md) describe private exports under `outputs/report-cases/`.

Later results through R62 are available in the [current figure gallery](../info/plots/current/README.md). The [figure review](../info/plots/REVIEW.md) records the full inventory. The November 2 P1 reconciliation resolved the ratio discrepancy as mixed weighting. Figure 7.1 now uses patient-uniform ratios from the [reconciled aggregate](../info/p1_anatomy_reconciled.json), including SAILOR JEPA at 2.883.

## Figure 1.1: longitudinal MRI

`outputs/report-cases/figure_1_1_longitudinal_mri.png` and its PDF show one LUMIERE patient at week bins 0, 44, and 56. All three panels use the released atlas FLAIR at the same axial coordinate. The [case-figure record](CASE_FIGURES.md#figure-11-longitudinal-mri) defines the compartments, selection, source, CC0 permission, and timing limits.

## Figure captions

**Figure 4.1. Forecasting architecture.** MRI and, where used, cutoff-available clinical inputs are preprocessed and enter one of two separate experimental branches: the original JEPA representation predictor or a later spatial-state/dynamics predictor. The branches are conceptual summaries, not a single chained model; input availability and output heads differ by experiment. The vertical dashed line denotes the prediction-time cutoff; future MRI and masks are used only as targets or for evaluation. Predictions are compared with experiment-specific baselines. Source: [Methods](../info/METHODS.md).

**Figure 4.2. Forecasting timeline and baselines.** Earlier and current visits are available at the cutoff; the next observed visit is the target. A history model uses earlier scans as well as the current scan. A latest-only model omits earlier MRI, with covariates matched where the experiment specifies. Persistence copies a current measurement or an MRI-decoded state, depending on the experiment. R37 and R39 exclude the realized future interval from predictor inputs; other experiments may use a retrospectively known interval, which must be stated explicitly. Source: [Methods](../info/METHODS.md).

**Figure 7.1. P1 next-visit anatomy forecast error relative to persistence.** Points show patient-uniform MAE divided by the corresponding cohort's persistence MAE; the dashed line marks persistence at 1.0 and lower is better. P1 evaluates three automated lesion-compartment log-volumes. LUMIERE includes 117 pairs from 23 encoder-unseen patients; SAILOR includes 242 pairs from 27 subjects and is developmental transfer, not an independent external test. No uncertainty is shown. Persistence uses measured current volumes, unlike MRI-based models. Source: [P1 results](../info/EXPERIMENTS.md#current-rano-free-anatomy-evidence).

**Figure 7.2. R37 history model compared with matched controls.** Points are paired patient-uniform CRPS differences and lines are Bonferroni-adjusted 99% patient-bootstrap intervals, on the native log1p(mm³) volume scale; negative values favor history. The assessment contains 75 pairs from 18 reused LUMIERE development patients. Scores average seeds, then observations within patients, then patients equally. The intervals do not establish a history benefit or equivalence. Decoded-state persistence copies MRI-derived volumes rather than measured masks. Source: [R37 results](../info/EXPERIMENTS.md#r37-full-pilot), [aggregate](../info/r37_full_metrics.json).

**Figure 7.3. R39 incremental treatment-status comparison.** The point is the paired patient-uniform Gaussian CRPS difference between history plus current treatment and history alone; the line is a 99% patient-bootstrap interval on the log1p(mm³) volume scale. Lower is better. The assessment comprises 242 SAILOR pairs from 27 reused development patients. Treatment status was coarse and observational; this is not a causal treatment effect. The interval does not establish added benefit or equivalence. Source: [R39 results](../info/EXPERIMENTS.md#r39-treatment-status-forecasting-october-30-2026), [aggregate](../info/r39_treatment_metrics.json).

**Figures 7.4 and 7.5. Patient examples.** [Detailed captions](CASE_FIGURES.md) cover the retained R49 anatomical montage and stable/changing R39 forecast cases. Figure 7.4 is explicitly partial because its 3D arrays and checkpoints were deleted. Figure 7.5 uses saved patient-held-out predictions and their fitted Gaussian scales, with input cutoff and retrospective case selection stated.

## Table 3.1 — Datasets and evaluation subsets

| Dataset or subset | Visits | Pairs | Patients or subjects |
|---|---:|---:|---|
| LUMIERE audited anatomical set | 599 | 496 usable consecutive pairs | 91 in full research cohort |
| LUMIERE encoder-unseen assessment | 149 | 117 | 26 patients with visits; 23 with pairs |
| SAILOR developmental transfer set | 269 | 242 | 27 subjects |
| R37 history-eligible assessment | — | 75 | 18 reused LUMIERE development patients |

These rows describe overlapping or different subsets and must not be added together. Confirm the 91-patient cohort description against the report's final dataset description. Sources: [Methods](../info/METHODS.md), [Experiments](../info/EXPERIMENTS.md).

## Table 5.1 — Models and baselines

| Method | Inputs | Output | Purpose | Experiment |
|---|---|---|---|---|
| Measured-volume persistence | Current automated mask volumes | Same volumes at next visit | Privileged measurement reference | P1; R39 reference |
| Decoded-state persistence | Current MRI-derived volume state | Same decoded state at next visit | MRI-only persistence control | R37; R39 |
| Latest-scan-only | Current MRI/state and specified past covariates | Next-visit volume distribution | Matched no-earlier-MRI control | R37; R39 |
| MRI-history model | Current and earlier MRI-derived states; available past timing | Next-visit volume distribution | Test added historical information | R37; R39 |
| JEPA-state forecast | Cutoff-available MRI history / representation | Predicted state and anatomical readout | Test original representation forecast | P1 |
| Empirical change | Current decoded state; source-fitted change distribution and observation noise | Volume forecast distribution | Simple stochastic change control | R37 |
| Ridge dynamics | Cutoff-available MRI feature maps and past information | Volume forecast distribution | Regularized linear dynamics control | R37 |
| History ± treatment | MRI-derived history and past timing, with or without current coarse treatment status | Next-visit volume distribution | Test incremental observational treatment information | R39 |

Measured-current-volume and MRI-decoded persistence do not receive the same information and must not be interpreted as interchangeable. See [R37](../info/EXPERIMENTS.md#r37-full-pilot), [R39](../info/EXPERIMENTS.md#r39-treatment-status-forecasting-october-30-2026), and [P1](../info/EXPERIMENTS.md#current-rano-free-anatomy-evidence).

## Table 6.1 — Evaluation protocol

| Aspect | P1 | R37 | R39 |
|---|---|---|---|
| Target | Next observed visit; three automated compartment log-volumes | Same, distributional forecast | Same, distributional forecast |
| Cutoff | No future MRI, mask, volume or ending gap in predictor | No future MRI, mask, volume, ending gap or treatment | No future MRI, mask, volume, actual next interval or future treatment label |
| Cohort and split | Patient-level encoder-unseen LUMIERE; SAILOR development transfer | Patient-disjoint source/selection/assessment; reused LUMIERE development patients | Five nested patient-held-out folds; reused SAILOR development patients |
| Metric and aggregation | MAE; patient-uniform relative to persistence shown | CRPS and MAE; average seeds, then within patient, then patients equally | Gaussian CRPS and MAE; compartments, visits within patient, then patients equally |
| Uncertainty | P1 figure shows point scores only | Paired patient bootstrap; five primary contrasts use adjusted 99% intervals | Paired patient bootstrap; primary comparison uses 99% interval, conditional on fitted folds |

Lower error is better. Automated masks are development proxies, not expert-verified biological truth. Cohorts and protocols are not interchangeable. Sources: [Methods](../info/METHODS.md), [Experiments](../info/EXPERIMENTS.md).

## Table 7.1 — Main findings

| Question | Finding |
|---|---|
| Did the original JEPA forecast beat persistence? | Not in the P1 anatomical evaluation. |
| Did MRI history improve the R37 forecast over latest scan? | Not established; the adjusted interval crosses zero. |
| Did joint state and dynamics training establish a benefit? | Not in the tested R38 setup. |
| Did current treatment status improve the history model? | Not established in R39; the result is not causal. |
| Was the 1.006 result produced by CORAL-aligned forecasting? | No. It was a corrected history-only rerun before the corrected alignment protocol. |

Sources: [P1 and corrected rerun](../info/EXPERIMENTS.md#current-rano-free-anatomy-evidence), [R37](../info/EXPERIMENTS.md#r37-full-pilot), [R38](../info/EXPERIMENTS.md#r38-completed), [R39](../info/EXPERIMENTS.md#r39-treatment-status-forecasting-october-30-2026). These findings are scoped to automated targets and reused development cohorts; inconclusive results do not establish equivalence.
