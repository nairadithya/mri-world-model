# Research TODO

This checklist records remaining work after the October 26, 2026, R37 scoped result-review closure and the earlier R35/R36 closures. It uses saved reports, not live job status. Earlier launch checklists remain in the [October 11 snapshot](documentation_sources/versions/e3c486ef9cc77afcc1978e53ce4495ef1051a5b8d22cf31b9d627c72d4db029e.md) and the [dated archive](archive/2026-10-13/README.md).

Use [Notes and insights](NOTES_AND_INSIGHTS.md) for interpretation, [Experiments](EXPERIMENTS.md) for results, and [Methods](METHODS.md) for protocols. The [research guardrails](research_guardrails.md) govern new implementation. This checklist does not authorize experiments, data acquisition, or a change of method family.

## Current priority, October 26, 2026

The operator requests full R37 completion and approves one fixed continuation allowance: 48 hours from October 26 at 10:00:04 +05:30, at most eight additional GPU-hours and sixteen additional CPU-hours. Use SSH on `mosaic-gpu` for tests and research execution, including workloads otherwise sent to Kaggle. Keep prior expenditure separate, including unknown initial-preflight cost. The [continuation record](documentation_sources/updates/2026-10-26/r37-continuation.md) and [Methods amendment](METHODS.md#r37) own the bounded scope. The continuation completed once. All nine neural fits, P/E/R controls, calibration, nuisance evaluation and reporting are complete. The [independent result review](documentation_sources/updates/2026-10-26/reviews/result-v1.md) approves scoped reporting of **INCONCLUSIVE_FOR_FULL_CAPABILITY**, not scientific success. The [findings](EXPERIMENTS.md#r37-full-pilot) and [closure receipt](documentation_sources/updates/2026-10-26/closure.json) close this formulation. Do not use remaining allowance for more training or condition selection.

Archival acceptance is preferred. Non-archival workshop acceptance is a fallback requiring agreement. The next unresolved publication tasks are a confirmed archival route, human review of the contribution and a narrowly scoped manuscript using the completed evidence. No matching archival call, submission approval or acceptance is established. The earlier literature assessment remains conditional; no new experiment follows from R37's inconclusive result.

## Workshop manuscript feasibility, October 24, 2026

The operator requires an accepted paper at ICWM, ICLR, or ICML and permits a workshop paper. The current authorization covers the bounded feasibility check and figure sketch below, not new fitting, data acquisition, submission, or a new method family. [Notes and insights](NOTES_AND_INSIGHTS.md#workshop-manuscript-feasibility) owns the judgment; the [source record](documentation_sources/updates/2026-10-24/literature-check.md) owns primary citations and search limits.

- [x] Check the closest primary work before claiming novelty. PaIRNet/LILAC already compare pairs against separate-state differences, BrainDiff already probes frozen state/change information, and Rudie's primary abstract provides direct post-treatment glioma precedent. No first-of-kind claim survives for those broad ideas.
- [x] Identify the remaining contribution candidate. Center the matched R36 change-supervision objective comparison and retain decoder, frozen-probe, independent-state, harm, and transfer findings together. The assistant's verdict is `CONDITIONAL_GO_FOR_SHORT_WORKSHOP_DRAFT`, not established novelty or submission approval.
- [x] Sketch one main result figure from existing aggregate inputs. The [figure specification](documentation_sources/updates/2026-10-24/figure-spec.json) binds the existing public R36 metrics. No plot was rendered or new measurement computed.
- [ ] Proposed next action: draft the narrowly scoped workshop manuscript around that comparison. A six-section outline is recorded in Notes; no manuscript draft was started in this feasibility task. Do not make a future R37 success a condition for reporting completed R35/R36 evidence.
- [ ] Before submission, identify a matching confirmed archival call, check its authorship/disclosure and registration requirements, and obtain human scientific review of the contribution. A non-archival fallback requires further agreement. Complete the bounded Rudie full-text comparison; do not present unavailable methods as checked or start a reproduction without separate authorization.

No independent literature or manuscript review occurred in this task. Prior scoped result approvals do not establish novelty, venue fit, or acceptance.

## R37 protocol-first dynamics pilot, October 19, 2026

The operator selects one bounded frozen-spatial-state next-observation pilot. The [R37 protocol](METHODS.md#r37) owns the comparisons, budget, thresholds, and stop rule. This supersedes the earlier no-family-selected planning status only for R37. It does not authorize the historical R36 Phase 2 recipe, a backbone sweep, data acquisition, or new annotation.

- [x] Write the protocol-first decision brief and fix one probabilistic primary endpoint.
- [x] Obtain independent design review of stable protocol bytes and resolve blocking findings. The [fresh review](documentation_sources/updates/2026-10-19/reviews/final-design.md) returns `APPROVED_FOR_PILOT_DESIGN` for its original revised comparisons. The later E amendment and runner have separate scoped reviews.
- [x] Complete the full reviewed implementation and pre-fitting gates. October 26 closure supersedes the partial October 19 status. The preserved [first review](documentation_sources/updates/2026-10-26/reviews/implementation-v1.md) required geometry, ledger and assessment-provenance corrections; the [fresh re-review](documentation_sources/updates/2026-10-26/reviews/implementation-v2.md) approved the corrected exact code. Sixty-two spatial tests passed remotely. Actual geometry/content checks covered 2,396 files and all 599 original visits passed fixed reconstruction tolerances. Historical gaps remain approximate week-bin differences, not verified exact dates.
- [x] Execute the operator-authorized E with-noise versus without-noise ablation once. The [saved findings](EXPERIMENTS.md#r37-noise-ablation) cover 75 history-eligible pairs from 18 development patients, with 26 single-visit pairs separate. Both conditions selected multiplier 1 on validation. The [execution receipt](documentation_sources/updates/2026-10-19/r37-noise-execution.md) separates numerical replay, review scope, and remaining limits.
- [x] Lock primary E to `gaussian_observation` for the common-floor rationale before full fitting. Retain five H contrasts and two-sided 99% intervals. The prior noise ablation was known; this was not blinded prespecification or assessment-based condition selection.
- [x] Execute the fixed full pilot once under the explicit October 26 continuation, preserving original expenditure separately. Nine neural fits completed twenty epochs each, 47,160 updates total. All controls, selections, assessment and nuisance companions completed without a failed/retried scientific stage. Weights, optimizer states, orders, OOF predictions and individual predictions remain private.
- [x] Review saved results independently. The reviewer authored and executed a separate CPU checker once, passing 12,861 checks and eighteen bounded neural forwards. The verdict is `APPROVED_FOR_SCOPED_R37_REPORTING`; no history, earlier-spatial-detail or secondary latest-state benefit was established. Stop after this one analysis. No rescue sweep, R36 reopening or automatic follow-up is authorized.

October 24, 2026, execution status: the retained E-ablation ledger started its elapsed clock on October 19, so the original two-day window has passed. Further full-pilot execution is blocked pending an explicit operator amendment, without erasing prior expenditure or treating the missing v1 accounting as zero. This publication-feasibility task does not renew that window. The later October 26 operator approval above separately renews a fixed continuation window; retain this expired-window receipt as history.

Preliminary data checks are limited to cutoff correctness, target availability, and alignment within the protocol's four-CPU-hour allowance. Target-reliability and independent-validation work below limits interpretation but is not an open-ended prerequisite for this scoped proxy experiment. Known leakage, invalid targets, broken geometry, and privacy limits remain hard stops.

## R35 computation and scoped result review are closed

The [R35 findings](EXPERIMENTS.md#r35), [independent result review](reviews/r35_result_review.md), and [closure receipt](reviews/r35_closure_receipt.json) close the earlier review task.

- [x] Complete the corrected training, transfer, nuisance, and control comparisons.
- [x] Resolve the exact-membership, declared reference provenance, and summary-digest audit findings through executed supplemental checks.
- [x] Correct interval wording and retain unfavorable shortcut controls in the aggregate presentation.
- [x] Obtain independent approval for scoped reporting and preserve the audit, review history, and numerical receipts.

No retraining or broad scientific rerun is needed to close this thread. Target reliability, historical MRI-byte provenance, unreplayed shuffle values, and untouched validation remain limits. No paired-model, transfer-success, forecasting, or general representation-quality claim follows. New experiments still need authorization.

## R36 Phase 1 is closed for scoped reporting

The [completed findings](EXPERIMENTS.md#r36-completed) and [closure receipt](reviews/r36_closure_receipt.json) replace the old seed-43 and seed-44 launch tasks. Do not rerun completed work because an archived checkbox is open.

- [x] Complete twelve final-epoch training arms, thirty source-only probes, and both development-cohort assessments.
- [x] Preserve the bounded CPU recovery receipts without repeating GPU work.
- [x] Correct nuisance-artifact identity and coverage checks, with rejection fixtures.
- [x] Classify demonstrated adverse tradeoffs separately from overlapping intervals and missing evidence.
- [x] Obtain the [independent scoped result review](reviews/r36_phase1_result_review.md).
- [x] Complete the [independent artifact audit](reviews/r36_phase1_artifact_audit.md), including saved ridge replay and patient-level contrast recomputation.
- [x] Present all arms and required companions in the [research page](research_run.html) and [aggregate measurements](r36_phase1_metrics.json).

The primary C-versus-B encoder gain, D's nuisance tradeoff, and superiority over independent state differences remain INCONCLUSIVE. C passes the specified direct-volume retention margin against A. That is not a general state-improvement claim. No follow-up family was selected at closure, and the historical Phase 2 recipe remains unauthorized. The October 19 R37 decision above selects a separate bounded forecasting pilot.

The earlier implementation review was not retrospective pre-training approval. Computational completion and reporting approval do not resolve target reliability, biological attribution, or untouched external validation.

## Priority 1. Resolve the limits on interpretation

These tasks require approved study designs. They are not permission to acquire data or start training.

### Establish target reliability

Before calling automated mask differences biological change, establish how reliable the targets are:

- [ ] Propose bounded expert adjudication or independent repeat annotations.
- [ ] Separate segmentation repeatability, anatomical correspondence, registration error, and actual disease change.
- [ ] Quantify partial boundary-detail loss at 96 cubed before proposing a native-grid experiment.
- [ ] Keep empty-compartment errors and postoperative transitions visible.

R36 found one native enhancing disagreement that disappeared completely after resizing. It did not establish widespread complete disappearance of change targets. Correct affines and native volumes do not prove accurate segmentation or biological correspondence. The suspect extracranial contour remains in the primary analysis.

Get approval for the sample, annotations, cost, privacy handling, and endpoint. Local SAILOR expert masks may support a bounded check. Their availability does not establish equivalent biological targets across cohorts.

### Reserve independent validation

Before claiming independent validation, reserve a compatible test that has not influenced development:

- [ ] Specify a fixed task and a new external or hidden test.
- [ ] Check participant overlap, preprocessing, label definitions, access, and data-use terms.
- [ ] Verify the SAILOR clinical codebook independently before making a clinical RANO claim.
- [ ] Continue to label the reused historical final13 and encoder-unseen LUMIERE assessments as development data.

SAILOR remains controlled-access development transfer. Its tentative codes 3 and 5 are not needed for the anatomical proxy task. No download, public upload, or data-use agreement is authorized here. The [external-supervision audit](documentation_sources/versions/636f0274c36efa3cee145c530acebff689d5961eff995269ecc3bc91d8c6d89e.md) lists historical candidate cohorts and access limits.

## Priority 2. Decide the next scientific question before implementation

The [R36 interpretation](NOTES_AND_INSIGHTS.md#d58) separates the descriptive nonlinear-decoder benefit from the unresolved encoder hypothesis. The completed comparison itself does not select an architecture or authorize another experiment. The October 19 operator request separately selects [R37](METHODS.md#r37).

Before substantial research implementation:

1. State the capability under test and the proposed information cutoff.
2. Write the [experiment decision brief](research_guardrails.md#experiment-decision-brief).
3. Include a materially different approach, a simple baseline, matched comparisons, costs, independent checks, falsifiers, and unknowns.
4. Obtain the operator's decision on changes to scientific targets, method-family priorities, and what counts as success.

### R36 Phase 2 remains unauthorized

The historical [conditional dynamics proposal](METHODS.md#r36-phase2) remains unselected. The operator's October 19 decision authorizes only the separate [R37 pilot](METHODS.md#r37), with frozen state-only features and its own controls, budget, and review gates. It does not authorize the historical Phase 2 encoder-objective comparison.

Any future predictor must exclude future scans, masks, volumes, embeddings, gaps ending at future visits, and future treatment from inference inputs. Teacher observations can be training targets only. Selection cannot use SAILOR or the reused development assessment outcomes.

### Other ideas remain proposals

The [synthetic-action proposal](METHODS.md#synthetic-actions) is unrun and unselected. Physical-coordinate and target-consistency checks must precede a training proposal.

Native-grid sampling, stronger encoders, different sequence fusion, and revised acquisition objectives are separate hypotheses. Do not bundle them into a repair of the completed C-versus-B comparison.

A faithful pretrained comparator, a genuine reproduction of paired-scan assessment, and better clinical labels remain evidence gaps. Their historical RANO task does not make them the next R36 step. Treatment-conditioned prediction is also distinct from identifying causal treatment effects.

## Priority 3. Maintain the consolidated research record

Follow the [documentation procedure](../AGENTS.md#maintain-the-research-documentation):

- [ ] Add measurements, corrections, and negative findings to Experiments with their historical IDs and evidence.
- [ ] Add definitions and protocol changes to Methods. Add reasoning to Notes and insights. Keep this page limited to current actions and their authorization.
- [ ] Reconcile new source records with existing coverage before archiving duplicates.
- [ ] Preserve exact dirty or untracked bytes and record original paths and hashes.
- [ ] Keep dates explicit when a new status supersedes an old runtime snapshot.
- [ ] Check active links and code-facing references after relocation.
- [ ] Keep operational guides, review evidence, aggregate metrics, and current presentation generators discoverable.

[Source coverage](SOURCE_COVERAGE.md) describes the original 72-version capture and links the later update record. Existing archives, frozen sources, and accepted review evidence retain their exact bytes and scope. Documentation approval does not accept scientific results or authorize research execution.

## Closed work is not an active checklist

[Experiments](EXPERIMENTS.md) records completed pilots, readouts, preprocessing and geometry audits, RANO comparators, P2 attempts, clean reruns, R35, and R36 Phase 1.

Do not restart unchanged JEPA continuations, larger horizon heads, combined surgery and augmentation sweeps, transition-weighting sweeps, or frozen-feature combinations because an old checkbox is open. A40 withdrew learned P2.4 and adaptation forecasting claims. A41 and the completed clean reruns replace their old launch checklists.

The October 11, 18:30:51 UTC receipt reported seed 43 running and seed 44 unlaunched. That remains true as a historical statement, not as the present TODO. The [saved runtime excerpt](documentation_sources/supplementary/r36-runtime-snapshot.json) retains its origin and timestamp. The [October 13 execution record](archive/2026-10-13/info/r36_delta_representation_results.md) and closure receipt explain the later completion without rewriting those old receipts.
