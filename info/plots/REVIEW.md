# Figure review and coverage

Reviewed November 2, 2026 against the [research summary](../RESEARCH_SUMMARY.md), [experiment record](../EXPERIMENTS.md), and [protocols](../METHODS.md).

The [current figure gallery](current/README.md) contains the added aggregate-based figures through R62, captions, and PDF links. Each figure reads public JSON. [Plotted values](current/plotted_values.json) record the displayed numbers, interval keys, and source-file hashes. No patient images, reserved MU measurements, or new model fits enter this build.

## Findings and corrections

| Finding | Correction |
|---|---|
| The two P2.4 seed figures and their reviewer versions presented leaked forecasts without the A40 withdrawal. | All four exports carry a visible withdrawal notice. Seed points are unconnected because seed order is not a temporal trend. Historical numbers remain historical. |
| Legacy adaptation figures showed the pre-correction zero-shot ordering. | Both carry a superseded notice. The new corrected-cache figure uses the canonical table's recorded values. |
| The response-feature scatter assigned the observed follow-up an invented error ratio of 1. | The observed follow-up remains in the classification panel but is omitted from the prediction-error scatter and its legend. It is a diagnostic, not a guaranteed upper bound. |
| A literature comparison line mixed the historical reported score into a macro-F1 plot despite A32's metric correction. | Removed the external comparison line from both Run 6 figures. |
| The original processing-cloud legends reversed skull-retained and skull-stripped labels. | Matched their labels to the corrected reviewer versions. Removed the causal claim that the failure was attributable to site rather than skull. |
| Several titles, category labels, and legends overlapped; one anatomy plot had no y-axis label because of an axes-array identity comparison. | Reduced the historical base font to 11 points, wrapped panel titles, added the missing MAE label, and used tight export bounds. |
| Cohort clouds relied on color alone; error-bar semantics were absent on some historical figures. | Added distinct marker shapes and explicit fold-SD notes. New plots distinguish controls with shapes or hatches and describe every interval. |
| The report architecture pointed the future-target arrow directly into the prediction box. | Added a separate comparison box receiving the prediction and future target. |
| The numerical figure collection largely stopped before the later spatial and measured-state work. | Added figures for R35, R36, R38, R40 through R62, and corrected adaptation. Existing report figures cover P1, R37, and R39. |
| P1's SAILOR table combined patient-uniform MAEs with pair-weighted relative MAEs. | Replayed the original aggregate on November 2. The patient-uniform JEPA ratio is 2.882931; 2.873313 is the valid pair-weighted ratio. Corrected every SAILOR relative score in the current table and report Figure 7.1. |

The review inspected all 46 existing numerical exports across the two legacy sets, the five report figures, and the R35 evidence-table screenshot. Archived presentations and handbook conceptual diagrams are historical or schematic sources, not additional current numerical plots. Their immutable files remain cited in the research record.

## Legacy figure inventory

Each row names the original PNG/PDF stem and its counterpart under `review_plots/`. Both versions use `metrics.json`, a September 2026 historical snapshot. Its old source paths are historical references, not the current documentation map. The export footer links this review so a standalone figure retains its status.

| Original stem | Reviewer stem | Scope or disposition |
|---|---|---|
| `jepa_pilot_gate` | `pilot_training_and_baseline` | Five-patient fitting diagnostic and persistence comparison; not a generalization result. |
| `hero_leg1_val` | `full_cohort_training_curve` | Reported validation checkpoints only; no complete epoch log. Earlier test patients are now development evidence. |
| `hero_leg2_val` | `continued_training_curve` | Continuation diagnostic; no new independent assessment. |
| `jepa_anatomy_forecast` | `existing_model_anatomy_forecast` | P1 anatomical comparison; ordinary 95% patient-bootstrap intervals. Its ratio already uses patient-uniform absolute errors. |
| `lesion_seed_relative_mae` | `lesion_model_training_repeats` | Withdrawn P2.4 forecasts after A40. |
| `lesion_seed_delta_ci` | `lesion_model_patient_intervals` | Withdrawn P2.4 intervals after A40. Valid arithmetic cannot repair leaked inputs. |
| `run6_tradeoff` | `response_training_comparison` | Historical auxiliary-response comparison; fold SD is not a confidence interval. |
| `sailor_transfer` | `second_cohort_transfer` | Historical same-space embedding scores and exploratory response readouts; unconfirmed SAILOR response mapping. |
| `sailor_gap_bins` | `second_cohort_visit_intervals` | Historical 243-pair embedding cache, distinct from the later 242-pair anatomy cohort. |
| `sailor_gaphead_vs_champ` | `interval_aware_prediction` | Legacy supplied-interval latent prediction; not prospective anatomical prediction. |
| `sailor_field_rescue` | `velocity_model_comparison` | Legacy latent-space velocity comparison; does not establish anatomical forecast benefit. |
| `sailor_field_ablation` | `treatment_context_comparison` | Legacy phase ablation; not a causal treatment effect. |
| `site_shift` | `cohort_feature_distributions` | PCA projection of historical feature arrays; visits are not independent patients. |
| `site_shift_aligned` | `aligned_cohort_feature_distributions` | A separately fitted PCA projection after alignment. Coordinates are not a shared before/after displacement frame. |
| `coral_alignment` | `feature_distribution_matching` | Historical feature-statistics diagnostic, including a transductive condition. No validated deployment correction follows. |
| `raw_vs_mni` | `image_processing_effect` | Descriptive image-change medians. Does not separate anatomical loss from acquisition nuisance. |
| `sailor_reprocessed` | `second_cohort_reprocessing` | Historical embedding and exploratory response scores; response semantics remain unconfirmed. |
| `sailor_reproc_vs_persist` | `reprocessed_image_forecasting` | Historical embedding errors and ratios to same-cache persistence. |
| `three_way_shift` | `processing_comparison_three_groups` | Historical cache projections with corrected skull labels. |
| `four_clouds` | `processing_comparison_four_groups` | Historical cache projections with corrected skull labels. |
| `betfix_decider` | `skull_tissue_sensitivity` | Historical dynamics comparison only; A21's readout-recovery interpretation was withdrawn. |
| `gauntlet_probe` | `response_feature_comparison` | Legacy response probes, fold SD, and embedding-error ratios. Observed follow-up has no prediction-error ratio. |
| `cross_site_adapt` | `second_cohort_adaptation` | Superseded by corrected-cache results. Weak scratch models do not establish JEPA superiority. |

The historical feature clouds retain their recorded coordinates. Different explained-variance labels reveal different PCA fits, so their panel coordinates cannot establish physical or representation-space movement. The current gallery does not use them as current transfer evidence.

## Coverage of the later work

| Question | Figure coverage |
|---|---|
| Current anatomy and observed change | R35 state and corrected independent-state contrasts; R36 matched frozen-readout contrasts. |
| Frozen spatial history and treatment status | Existing [R37 and R39 report figures and captions](../../report/README.md#figure-captions). |
| Joint training, horizon, and richer controls | R38, R40, R41/R42 crop and initialization, R43 rollout, R44 postoperative context, R45 response, and R46 perfusion/scanner comparisons. |
| Current-state forecast gain and transfer | R47 source and locked developmental-transfer contrasts. |
| Whether models fit and pretraining helps | R48 duration diagnostic, R49 absolute scores and paired intervals, R50/R51 versions, and R52 token-interface contrasts. |
| Image prediction and decoder limitations | R53 completion and forecast contrasts, R54 individual-example fitting, and R55 code/decoder continuation curves. |
| Regional state and latent objectives | R56 assimilation, R57 context/adjacency/history controls, and R58 future-feature contrasts. |
| Measurement correction and measured history | R59 old-target comparison, R60 erased-cell and direction-disagreement rates, R61/R62 six-arm scores, paired intervals, and error strata. |
| New-cohort support | R62 assigned versus history-eligible development patients. |

These are selected primary comparisons and explanatory diagnostics, not every secondary metric. The JSON aggregates retain the remaining seed, calibration, compartment, and perturbation results. R1 through R34 also include unrun proposals, repairs, and withdrawn claims; their existing record is not a missing series of successful trials.

## Remaining source limitations

- The earlier P1 ratio discrepancy is resolved in the [dated correction](../EXPERIMENTS.md#current-rano-free-anatomy-evidence) and [reconciled aggregate](../p1_anatomy_reconciled.json). Historical archives retain their original mixed-weighting presentation.
- [Report case figures](../../report/CASE_FIGURES.md) now include three longitudinal LUMIERE visits and two patient-level R39 forecast examples, generated from retained private artifacts. Case images and predictions remain under ignored `outputs/report-cases/`. These are separate from the aggregate gallery.
- R49's retained montage shows two fixed anatomical examples, but the operator confirmed deletion of the checkpoints and per-case arrays. The requested typical and failure cases with compartment Dice and volume errors remain unavailable. The labeled archival figure explicitly records this limitation.
- Public aggregates do not contain patient-wise trajectories or full prediction distributions. R39's private saved distributions support its case intervals; the aggregate figures' bootstrap intervals remain fitted-comparison uncertainty, not biological uncertainty or independence from prior cohort reuse.
- Old research pages and previously compiled documents do not update when their external figure files change. Their owners must rebuild any document that embeds the older images before distribution.

## Reproduction

Run these lightweight rendering commands from the repository root:

```bash
.venv/bin/python -B info/plots/make_plots.py
.venv/bin/python -B info/plots/make_plots_review.py
.venv/bin/python -B report/make_figures.py
```

The first command builds the original historical set and the current gallery. To rebuild only the additions, run `.venv/bin/python -B info/plots/make_current_plots.py`. Current figures use IBM Plex Sans labels, IBM Plex Serif titles, embedded PDF fonts, and 300-dpi PNGs. Their nominal width is 7.2 inches. Captions specify units, sample sizes, interval levels, and separate panel scales.
