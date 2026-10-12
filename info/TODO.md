# Research TODO

This checklist records the work left at the archived research snapshot of October 11, 2026, at 18:30:51 UTC. It is not a live progress report. The wording was revised on October 12, 2026.

Use this page to review the remaining work, not to authorize new experiments. The original [research rules](documentation_sources/versions/6e710be8c33b05d77bf8f13b415b64028fcc13dc03d71defb304e6a7631c78f4.md) and approved experiment protocols still apply.

Start with [Notes and insights](NOTES_AND_INSIGHTS.md) for the reasoning. Use [Experiments](EXPERIMENTS.md) to look up results and [Methods](METHODS.md) to look up the exact protocols.

## Priority 1. Finish the authorized R36 Phase 1

R36 compares two training objectives, B and C, to test whether giving changed regions more weight helps the encoder represent observed change. [Methods](METHODS.md#r36-phase1) defines the four experiment arms and their required comparisons. This experiment observes both scans. It does not predict a future scan.

At the archived snapshot, all twelve training checkpoints and ten seed-42 readout models had passed local checks. Seed 43 was running. Seed 44 had not started. No comparison of model quality on the evaluation patients had been scored. Do not treat these statements as current job status.

### Complete the remaining source-only readouts

A readout is a model that extracts a measurement from the encoder's features. Here, the encoder stays fixed, and the readout learns only from the 52 source-training patients.

Complete the readouts under the approved recipe:

- [ ] After the remote seed-43 job finishes, collect its outputs.
- [ ] Check each readout's epoch-20 checkpoint against its encoder, feature view, protocol, row order, and complete finite training history.
- [ ] Before starting seed 44, check the actual GPU quota and the remaining experiment allowance.
- [ ] If a launch intent or push receipt already exists, reconnect to that job instead of starting another copy.
- [ ] Collect and check the seed-44 outputs under the same rules.
- [ ] Require all thirty correctly linked readout models before the complete comparison.
- [ ] Record failed or incomplete work with its actual status.

Use three seeds and exactly 20 epochs. Score the final epoch only. Do not retry failed arms, shorten training, score partial arms, or repeat completed launches.

The total limit is 24 GPU-hours. Each session has a nine-hour maximum, and some follow-on jobs have tighter limits. This document does not authorize a new paid launch.

### Close the reporting review findings

The completed implementation review found risks in the reports. It did not establish a model failure or accept the scientific results. It also did not replace the required design review before training.

Resolve the reporting checks before accepting an aggregate report:

- [ ] Check each nuisance-test artifact's status, arm, seed, checkpoint, and readout identity.
- [ ] Check that every required patient and perturbation condition is present exactly once.
- [ ] Reject duplicate, missing, or mismatched reports.
- [ ] Report demonstrated harm to state estimation, change estimation, or nuisance robustness as NOT VERIFIED.
- [ ] Report overlapping confidence intervals or missing evidence as INCONCLUSIVE.
- [ ] Update stale review-status records without implying that the results have been accepted.

The [R36 protocol and recovery review](documentation_sources/supplementary/r36-protocol-recovery.md) gives the complete findings. It refers to `src/harness/analysis/delta_evaluation.py` in `world-model-r36`: lines 88-107 for nuisance aggregation, and lines 66-68 and 103-107 for negative verdicts. Those line numbers describe the reviewed revision, not a fresh code inspection.

No corrupted evaluation report or harmful model tradeoff had been observed in the archived record. Review of this code is separate from review of these documents and from the pending scientific result review.

### Produce the complete Phase 1 assessment

Use the fixed development rows and the matched readouts described in [Methods](METHODS.md#r36-phase1). Complete the following reports:

- [ ] Compare C with B using local linear readouts trained on the same 52 source patients, with each encoder held fixed.
- [ ] Include original R35, continued state-only arm A, zero change, later-scan-only readouts, and differences between separate state estimates.
- [ ] Do not score A's untrained change head.
- [ ] Report balanced occupancy error with equal weight per patient.
- [ ] Report changed and unchanged regions separately, including appearance, disappearance, sample counts, and groups defined from source-training data.
- [ ] Report native log-volume errors for state and change, including final measured visits and genuinely empty compartments.
- [ ] Keep the ridge-regression controls trained only on source patients, with validation grouped by patient.
- [ ] Compare C with D for sensitivity to acquisition changes.
- [ ] Report false change relative to the identity test, changes in state estimates, and the held-out resolution test separately.
- [ ] Apply the fixed three-seed comparison, patient bootstrap, and adjusted confidence intervals.
- [ ] Do not count seeds or voxels as independent patients.
- [ ] Evaluate every source-trained readout on local SAILOR data without fitting to those evaluation patients or choosing models from their results.
- [ ] Keep differences in label definitions, acquisition, and segmentation source visible in the transfer report.
- [ ] Check saved predictions, weights, normalizers, protocols, model source, tensors, visits, and pair identities before accepting a score table.

Keep every original eligible pair in the primary analysis. Keep the suspect contour outside the brain and the empty targets visible. Do not exclude cases after seeing which exclusions help a model.

### Review and close Phase 1

Complete the reporting and review steps in order:

1. Obtain independent scientific result review after all required companion evaluations are complete. The earlier recovery review is not a result verdict.
2. Publish the full results table for every arm and control, including negative, incomplete, and inconclusive outcomes.
3. Distinguish a better encoder representation from a better nonlinear decoder. Distinguish observed change from future prediction.
4. Append accepted findings and revised priorities to the execution and decision records.
5. If the conclusions change, update `info/research_run.html` and inspect the rendered page.
6. Present the Phase 1 report before choosing a follow-up method family.

## Priority 2. Resolve the limits on interpretation

These tasks remain open. They do not authorize data acquisition, new targets, or another model family.

### Establish target reliability

Before calling measured differences biological change, establish how reliable the masks and their alignment are:

- [ ] Propose a bounded expert review or independent repeat-annotation study.
- [ ] Separate segmentation repeatability, correspondence between anatomical locations, registration error, and actual disease change.
- [ ] Quantify partial boundary-detail loss at 96 cubed before proposing a native-grid experiment.
- [ ] Keep absent-compartment failures and postoperative transitions visible.

The R36 audit found one native enhancing disagreement that disappeared completely after resizing. It did not establish widespread complete loss of change targets. Correct grids and native volumes do not establish accurate segmentation or reliable biological change.

Get approval for the study's sample, annotations, cost, privacy handling, and scientific endpoint. SAILOR expert masks may help a local check, but their availability does not establish that both cohorts measure the same biological endpoint.

### Reserve genuinely independent validation

Choose and protect an independent test before making an independent-validation claim:

- [ ] Choose a new external or hidden test with a fixed task and no prior model selection.
- [ ] Check participant overlap, supplied preprocessing, label definitions, access, and data-use terms before assigning the cohort a role.
- [ ] Independently verify the SAILOR clinical codebook before making a clinical RANO claim.
- [ ] Label both the reused historical final13 and the repeatedly examined encoder-unseen LUMIERE patients as development data.

The tentative SAILOR mapping of codes 3 and 5 is not required for the current anatomical proxy task. No new acquisition, public upload, or acceptance of a data-use agreement is authorized here. The [external-supervision audit](documentation_sources/versions/636f0274c36efa3cee145c530acebff689d5961eff995269ecc3bc91d8c6d89e.md) lists candidate cohorts and access limits.

## Priority 3. Maintain the consolidated research record

Before any later consolidation, complete the document review:

- [ ] Review [Experiments](EXPERIMENTS.md) for preserved measurements, corrections, and negative outcomes.
- [ ] Review [Methods](METHODS.md) for separation of current, historical, and proposed protocols, and for the historical decision and audit indexes.
- [ ] Review [Notes and insights](NOTES_AND_INSIGHTS.md) for the reasoning and the limits of each conclusion.
- [ ] Keep the section-level source mapping and historical identifiers intact when adding new findings.
- [ ] Preserve exact copies of dirty and untracked sources. Git history alone does not preserve those files.
- [ ] Update existing Markdown links and references in code-facing documentation before a later consolidation.
- [ ] Keep `AGENTS.md`, the root `README.md`, CLI verification skills, and operating procedures discoverable.

These four documents are the consolidated research record, promoted with the operator's approval. The [archive](archive/2026-10-12/README.md) preserves the original pages without changing their bytes. Main's intentional deletions of `info/notes.md` and `info/explainer.html` remain unchanged in that checkout. Their historical copies are archived here, not restored as current guidance. The old notes repeat R29 to R32, now indexed in Methods.

## Separately gated work

### R36 Phase 2 remains unauthorized

Do not start Phase 2 without a separate operator decision. The [conditional dynamics proposal](METHODS.md#r36-phase2) requires source-validation selection rules, verified limits on information available at prediction time, and matched current-scan and history controls.

For future prediction, exclude future scans, masks, volumes, embeddings, gaps ending at future visits, and future treatment from model inputs. Do not choose encoders using SAILOR results or the development assessment results. State the additional capability that a new experiment tests.

### Other ideas remain candidates

Treat the following ideas as proposals, not approved next steps:

- The [action-conditioned representation brief](documentation_sources/versions/f3dad868d43b3aa2bf5f7913b3dfa2872cdf87a65fd9648b1eb9932e3f750855.md) proposes controlled synthetic changes to real MRI. It is unrun and unselected. Check physical coordinates and target consistency before proposing training.
- Native-grid sampling, stronger encoders, different MRI-sequence fusion, and revised acquisition objectives are separate hypotheses. Do not bundle them into repairs to the fixed C-versus-B comparison.
- A faithful pretrained comparator, a genuine reproduction of paired-scan assessment, and better clinical labels remain evidence gaps. Their historical RANO task does not make them the next R36 step.
- Treatment-conditioned prediction and causal treatment effects are different questions. Observational SAILOR treatment tokens do not support claims about what would happen under an alternative treatment.

Before substantial new research implementation, write the required brief. Include a materially different approach, a simple baseline, matched comparisons, cost, independent checks, and findings that would disprove the hypothesis. The operator decides changes to scientific targets and method-family priorities.

## Closed work is not an active checklist

Use [Experiments](EXPERIMENTS.md) to find completed pilots, readout tests, preprocessing and geometry audits, RANO comparators, P2 attempts, clean reruns, and the R35 comparison.

Do not restart unchanged JEPA continuations, larger horizon heads, combined surgery and augmentation sweeps, transition-weighting sweeps, or frozen-feature combinations because an old checkbox is still open. A40 withdrew the learned P2.4 and adaptation forecasting claims. A41 and the completed clean reruns replace their old launch checklists.

Completed computation does not resolve biological validity, independent result review, or external validity.

## Source and status provenance

The work priorities come from [D55 to D57 and the decision history](documentation_sources/versions/2b7293c884e0e0aa9f1db178d13f55497d584cd7060c68c022436d47330bb63c.md), [the latest R36 directions](documentation_sources/versions/eade325e60cbeeb92dc5f6570520d8f502a519e846309ef5803b2bf7e1a2d094.md), and [the approved R36 handoff addendum](documentation_sources/versions/8fc1116f80667885774a5ce6a8fb9ec40f1e360d1f87867fefb81baf01118ab6.md).

The newer [R36 execution record](documentation_sources/versions/51d6523f0da4b3d8745af10b79318fd684aba1fd9b47cc7d321e30df79984551.md) comes from `exp/r36-delta-objective` at `f4b96c0` and includes the 14:16 UTC resumption. The [older main-checkout record](documentation_sources/versions/4f80203aeaed182a3e106b44f5d066b213b65387a143d3979710282be6263077.md) ends at 04:04 UTC. Neither is a live monitor.

The later 18:30:51 UTC snapshot came from `outputs/r36-delta-objective/continuation.json`. The [recovered receipt](documentation_sources/supplementary/r36-runtime-snapshot.json) preserves its exact excerpt, timestamp, and source-line hash from tool-result line 701 of the saved October 10 parent session. The earlier temporary copy is unavailable. This rewrite does not inspect the moving receipt, claim a later job state, or report a fresh quota balance. The receipt's generic review field does not reliably distinguish implementation review from scientific result review.

Historical completion and withdrawal checks come from [the original TODO](documentation_sources/versions/c7c9fba4178609946f8fd4ce8b90af08b6a88392955d7f2e198aa2e02b14cc57.md), [clean rescues](documentation_sources/versions/eaca551880e0a9db5c4bdc7785e135675cd518ebfcd361d5aa482beca4f37dce.md), and [the R35 record](documentation_sources/versions/768fdfa0a9be4dc76e99fcef9ff971e26252d672e82b08267f5e9ea157ced0f3.md) at `2f6e1f5`.

[Source coverage](SOURCE_COVERAGE.md) records every captured version and the operating documents kept separate. Research-source links point to fixed copies inside the repository. The archived originals remain byte-identical. Documentation promotion does not authorize research execution or accept scientific results.
