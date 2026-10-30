# Research TODO

This checklist records current priorities and open decisions after the completed R35, R36 Phase 1, and R37 pilots. Historical launch checklists remain in the [October 11 snapshot](documentation_sources/versions/e3c486ef9cc77afcc1978e53ce4495ef1051a5b8d22cf31b9d627c72d4db029e.md) and the [dated archive](archive/2026-10-13/README.md).

Use [Notes and insights](NOTES_AND_INSIGHTS.md) for interpretation, [Experiments](EXPERIMENTS.md) for results, [Methods](METHODS.md) for protocols, and [research guardrails](research_guardrails.md) for experimental standards.

## Active threads

### R39 treatment-status feasibility completed, October 30, 2026

- [x] Test current SAILOR treatment phase against matched MRI-state and history controls under the [fixed protocol](METHODS.md#r39-cutoff-treatment-phase-forecasting-feasibility).
- [x] Complete the patient-held-out comparison and saved-artifact replay. The primary treatment contrast and the history-conditioned follow-up criterion were unmet. See [findings](EXPERIMENTS.md#r39-treatment-status-forecasting-october-30-2026), [interpretation](NOTES_AND_INSIGHTS.md#r39-did-not-establish-added-treatment-information) and [aggregate](r39_treatment_metrics.json).
- [ ] Decide whether to pursue timestamped planned-treatment inputs or return to the unchanged R38 continuation. Do not present per-session status as planned intervention or start a larger model as a routine repair of this inconclusive result.

### 1. R38 joint state and dynamics pilot (`exp/r38-joint-state-dynamics`)
R37 did not establish that multi-visit MRI history beats latest-scan-only prediction with an encoder trained on single-scan anatomy (`R35`). R38 tests whether training the spatial state encoder and future-dynamics predictor **jointly** (`JH` vs `JL` vs observation-only `OH` and `OL`) makes MRI history useful for next-visit anatomical prediction.

- [x] Implement the 4-arm joint state-and-dynamics model, source-only controls, and evaluation scorer on `exp/r38-joint-state-dynamics`.
- [x] Pass the 84-update real-data end-to-end canary on `mosaic-gpu` across all 599 reconstructed visits.
- [x] Complete seed-42 epoch-20 training on `mosaic-gpu` (`OH`, `OL`, `JH`, `JL` — 5,240 updates each).
- [ ] **Next action:** Resume and finish seeds 43 and 44 on `mosaic-gpu` directly (without the fragile process-lookup budget wrapper that interrupted seed 43), then run the CPU evaluation scorer to produce the R38 comparison table.

### 2. Workshop / archival paper track
The [workshop manuscript feasibility check](NOTES_AND_INSIGHTS.md#workshop-manuscript-feasibility) (`CONDITIONAL_GO_FOR_SHORT_WORKSHOP_DRAFT`) and [literature check](documentation_sources/updates/2026-10-24/literature-check.md) reviewed primary overlap ([PaIRNet/LILAC](https://proceedings.mlr.press/v227/kim24a.html), BrainDiff, Rudie et al.):

- [x] Check closest primary work and record the [main result figure specification](documentation_sources/updates/2026-10-24/figure-spec.json) and [R35–R37 Typst explanation](r35-r37-explanation.typ).
- [ ] Decide manuscript target after R38 completes (either a focused empirical paper combining R35–R38 or a workshop submission).

## Completed & closed experiments

- **R37 frozen-state dynamics pilot (Closed Oct 26, 2026):** All nine neural fits (`H`, `L`, `G` across seeds 42/43/44) and `P`/`E`/`R` controls completed on `mosaic-gpu`. Outcome: `INCONCLUSIVE_FOR_FULL_CAPABILITY` (`H` vs `L` patient-uniform CRPS contrast `+0.000559`, 99% CI `[-0.002975, +0.004431]`). See [findings](EXPERIMENTS.md#r37-full-pilot), [protocol](METHODS.md#r37), [noise ablation](EXPERIMENTS.md#r37-noise-ablation), [execution record](documentation_sources/updates/2026-10-26/r37-continuation.md), [result review](documentation_sources/updates/2026-10-26/reviews/result-v1.md), and [closure receipt](documentation_sources/updates/2026-10-26/closure.json).
- **R36 Phase 1 delta-supervision comparison (Closed Oct 13, 2026):** Twelve training arms and 30 source-only probes completed. Trained nonlinear change decoders (`C`) improved over `A` and retained direct volume accuracy, while frozen-encoder linear probing (`C` vs `B`) and SAILOR transfer remained inconclusive. See [completed findings](EXPERIMENTS.md#r36-completed), [D58 interpretation](NOTES_AND_INSIGHTS.md#d58), [result review](reviews/r36_phase1_result_review.md), [artifact audit](reviews/r36_phase1_artifact_audit.md), [closure receipt](reviews/r36_closure_receipt.json), [research page](research_run.html), and [metrics](r36_phase1_metrics.json). Historical runtime excerpts ([Oct 11 snapshot](documentation_sources/supplementary/r36-runtime-snapshot.json), [Oct 13 record](archive/2026-10-13/info/r36_delta_representation_results.md)) are preserved for provenance.
- **R35 spatial representation comparison (Closed Oct 13, 2026):** In-domain automated compartment volume estimation improved over pooled baselines, while paired heads and SAILOR transfer did not establish a longitudinal gain. See [R35 findings](EXPERIMENTS.md#r35), [result review](reviews/r35_result_review.md), and [closure receipt](reviews/r35_closure_receipt.json).
- **Earlier JEPA / RANO / P0–P2 baselines:** Recorded in [Experiments](EXPERIMENTS.md) (including A14, A24, A27, A40, A41, D16, D22, D33, D37, D53, D55, K3-16). Do not re-run closed sweeps.

## Standing interpretation limits & future options

<a id="priority-1-resolve-the-limits-on-interpretation"></a>
<a id="priority-2-decide-the-next-scientific-question-before-implementation"></a>

1. **Target reliability:** Automated DeepBraTumIA / ONCO masks are proxy targets, not expert-adjudicated biological truth (one native enhancing change disappeared at `96³` in R36; one suspect extracranial contour remains in LUMIERE). Any future biological-truth claim requires bounded expert adjudication or repeat-scan reliability data.
2. **External validation:** Reused `final13`, `unseen`, and SAILOR cohorts are developmental evaluation sets. See the [external-supervision audit](documentation_sources/versions/636f0274c36efa3cee145c530acebff689d5961eff995269ecc3bc91d8c6d89e.md) for candidate cohorts.
3. **Next experiments:** Before launching a new experiment beyond R38, write the short [experiment decision brief](research_guardrails.md#experiment-decision-brief) and follow the [documentation procedure](../AGENTS.md#maintain-the-research-documentation) and [source coverage](SOURCE_COVERAGE.md). Unrun proposals ([R36 Phase 2](METHODS.md#r36-phase2), [synthetic actions](METHODS.md#synthetic-actions)) remain optional candidates.
