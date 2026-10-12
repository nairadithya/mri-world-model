# Research TODO

Draft prepared 2026-10-12 from frozen research records. This is the proposed work list at the archived research snapshot. It does not replace the original records or authorize new experiments. R36 runtime status is a snapshot from 18:30 UTC, not a live monitor.

Read [the experiment record](EXPERIMENTS.md) for outcomes, [the methods](METHODS.md) for contracts, and [the notes and insights](NOTES_AND_INSIGHTS.md) for rationale. Existing [research guardrails](../documentation_sources/versions/6e710be8c33b05d77bf8f13b415b64028fcc13dc03d71defb304e6a7631c78f4.md) and locked experiment contracts still govern.

## Priority 1. Finish the authorized R36 Phase 1

The question is whether change-balanced supervision improves frozen access to directional observed MRI change. C versus B is the primary objective-only contrast. This is not a forecasting experiment or a universal representation-quality test.

All twelve final20 training checkpoints and ten seed42 source-fitted probes are locally validated. The continuation receipt at 2026-10-11 18:30:51 UTC reports seed43 RUNNING. Seed44 is not yet launched. No comparative query quality results have been scored. The newer [R36 execution record](../documentation_sources/versions/51d6523f0da4b3d8745af10b79318fd684aba1fd9b47cc7d321e30df79984551.md) was captured from the execution branch and includes the 14:16 UTC resumption. The older [main-checkout R36 record](../documentation_sources/versions/4f80203aeaed182a3e106b44f5d066b213b65387a143d3979710282be6263077.md) ends at 04:04 UTC. The later runtime snapshot is separately preserved below. None is a live job monitor.

### Complete the remaining source-only readouts

- [ ] Collect seed43 only after remote completion. Verify each final20 probe against its encoder, view, contract, row order, and complete finite history.
- [ ] Recheck actual quota and the remaining experiment allowance before the sequential seed44 launch. Existing launch intents or push receipts mean reattach, not push again.
- [ ] Collect and validate seed44 under the same rules.
- [ ] Require all thirty bound probes before the complete matched assessment. Preserve failed or incomplete work with its actual status.

The recipes remain three seeds, exactly 20 epochs, and final-only grading. Do not retry failed arms, shorten recipes, grade partial arms, or duplicate completed launches. The experiment ceiling remains 24 GPU-hours, with a nine-hour maximum per session and tighter existing follow-on job limits. No new paid launch is authorized by this document.

### Close the reporting review findings

The fresh post-implementation review completed. It did not replace pre-training design review or independent result review. Its findings concern reporting validity, not observed model failure.

- [ ] Before accepting the aggregate report, validate nuisance artifact status, arm, seed, checkpoint and probe binding, and patient-condition coverage. Add rejection checks for duplicates, omissions, and mismatched reports.
- [ ] Classify demonstrated adverse state/change tradeoffs or nuisance worsening as NOT VERIFIED. Keep overlapping intervals and missing evidence INCONCLUSIVE.
- [ ] Update stale review-status receipts without implying scientific acceptance.

The reviewed code is `src/harness/analysis/delta_evaluation.py` in `world-model-r36`. The review points to lines 88-107 for nuisance aggregation and lines 66-68 and 103-107 for incomplete negative verdicts. The complete [R36 protocol/recovery review](../documentation_sources/supplementary/r36-protocol-recovery.md) remains a local managed artifact. This is distinct from review of these documentation drafts and the still-pending independent experiment-result review. No corrupted query report or failed model tradeoff has been observed.

### Produce the complete Phase 1 assessment

- [ ] Score C versus B using matched, frozen, source52 local linear probes on the fixed encoder-unseen development rows.
- [ ] Retain original R35, continued state-only A, zero change, target-only readouts, and independent state differences. Do not grade A's untrained change head.
- [ ] Report patient-uniform balanced occupancy error, changed and unchanged partitions, signed appearance/disappearance, support counts, and source-defined strata.
- [ ] Report native log-volume state and change errors, including final measured visits and true empty compartments. Preserve source-only grouped-ridge controls.
- [ ] Report C versus D nuisance tradeoffs, identity-relative false change, state drift, and the held-out resolution stress separately.
- [ ] Apply the locked three-seed paired-patient bootstrap and adjusted contrast intervals. Do not treat seeds or voxels as independent patients.
- [ ] Run all source-fitted readouts on local SAILOR without query fitting or model selection. Keep ontology, acquisition, and developmental-transfer limits visible.
- [ ] Validate saved predictions, weights, normalizers, contracts, model source, tensors, visits, and pair identities before accepting a score table.

The primary analysis retains every original pair. The known suspect extracranial contour and empty targets remain visible. An unplanned model-dependent exclusion cannot repair an unfavorable result.

### Review and close Phase 1

- [ ] Obtain independent result review after the complete companion evaluations. The completed recovery review is not a result verdict.
- [ ] Publish the entire arm/control table, including negative, incomplete, and inconclusive outcomes.
- [ ] Distinguish an encoder feature-accessibility gain from a nonlinear decoder gain. Distinguish observed change from future predictability.
- [ ] Append the accepted findings and revised priorities to the execution and decision records. Update `info/research_run.html` and inspect its rendered pixels if conclusions change.
- [ ] Present the Phase 1 report before selecting a follow-up family.

## Priority 2. Resolve the limits on interpretation

These requirements remain open. They are not automatic permission to acquire data, change targets, or run another model family.

### Establish target reliability

- [ ] Define a bounded expert or independent repeat-annotation study before claiming biological change or a usable segmentation noise ceiling.
- [ ] Separate segmentation repeatability, anatomical correspondence, registration jitter, and actual pathology. Verified source grids and native volumes do not establish these properties.
- [ ] Quantify partial boundary-detail loss at 96 cubed before proposing a native-grid experiment. The R36 audit found one completely erased native enhancing disagreement, not widespread complete target erasure.
- [ ] Keep absent-compartment failures and postoperative transitions visible. A favorable aggregate does not resolve them.

The study's sample, annotations, cost, privacy handling, and scientific endpoint require a decision brief and operator approval. SAILOR expert masks may inform a local check, but their availability does not by itself establish a harmonized biological endpoint.

### Reserve genuinely independent validation

- [ ] Choose a new external or hidden test with a frozen task and no prior model selection.
- [ ] Audit participant overlap, supplied preprocessing, label definitions, access, and data-use terms before assigning that cohort a role.
- [ ] Verify the SAILOR clinical codebook independently before using it for a clinical RANO conclusion. The tentative 3-versus-5 mapping is not required for the current anatomy proxy.
- [ ] Keep the reused final13 and the repeatedly examined encoder-unseen LUMIERE patients labeled as development data.

No new acquisition, public upload, or data-use agreement acceptance is authorized here. The [external-supervision audit](../documentation_sources/versions/636f0274c36efa3cee145c530acebff689d5961eff995269ecc3bc91d8c6d89e.md) records candidate cohort roles and access blockers.

## Priority 3. Review the four-document drafts

- [ ] Review [EXPERIMENTS.md](EXPERIMENTS.md) for preserved measurements, corrections, and negative outcomes.
- [ ] Review [METHODS.md](METHODS.md) for current, historical, and proposed protocol separation.
- [ ] Review [NOTES_AND_INSIGHTS.md](NOTES_AND_INSIGHTS.md) for retained decision/audit history and scoped conclusions.
- [ ] Confirm the section-level source mapping and historical identifier coverage before removing any original research page.
- [ ] Preserve exact recoverable copies of dirty and untracked sources. Git history alone is insufficient.
- [ ] Update existing Markdown links and code-facing documentation references before any later consolidation.
- [ ] Keep `AGENTS.md`, root `README.md`, CLI verification skills, and operator procedures discoverable.

These files are drafts alongside the originals. No canonical document has been replaced. The main checkout's intentional deletions of `info/notes.md` and `info/explainer.html` are not changed. This documentation checkout inherited their committed copies from its base. They remain historical material, not restored canonical guidance. The old notes restate R29-R32, which already exist in the directions record. No original is deleted or promoted by this preparation.

## Separately gated work

### R36 Phase 2 remains unauthorized

A Phase 1 observed-change result does not authorize a future-state model. The [conditional dynamics plan](METHODS.md#r36-phase2) requires a separate operator decision, source-validation selection rules, a verified information cutoff, and matched current-only and history controls.

Do not feed future scans, masks, volumes, embeddings, ending gaps, or future treatment into forecast inference. Do not choose encoders using SAILOR or the development assessment outcomes. A new experiment must state what incremental capability it tests.

### Other ideas remain candidates

- The [action-conditioned representation brief](../documentation_sources/versions/f3dad868d43b3aa2bf5f7913b3dfa2872cdf87a65fd9648b1eb9932e3f750855.md) proposes controlled synthetic actions on real MRI. It is unrun and unselected. Its generator needs physical-coordinate and target-consistency checks before a training proposal.
- Native-grid sampling, stronger encoders, alternative modality fusion, and revised acquisition objectives are distinct hypotheses. They are not bundled repairs to the locked C-versus-B comparison.
- A faithful pretrained comparator, genuine paired-scan assessment reproduction, and improved clinical labels remain valid evidence gaps. Their historical RANO task does not make them the next active R36 step.
- Treatment-conditioned prediction and causal treatment effects are separate questions. Observational SAILOR treatment tokens do not authorize counterfactual-care claims.

Before substantial new research implementation, write the required brief with a materially different approach, a simple baseline, matched comparisons, cost, independent checks, and falsifiers. The operator decides scientific target and method-family changes.

## Closed work is not an active checklist

The experiment record retains the completed pilots, frozen-probe tests, preprocessing and geometry audits, RANO baseline/interface/comparator tests, P2 attempts, clean rescues, and R35 matrix.

Do not reopen the unchanged JEPA resumes, larger horizon heads, bundled surgery/augmentation/weighting sweeps, or frozen-feature concatenations merely because an old TODO box remains unchecked. A40 withdrew the learned P2.4 and adaptation forecasting claims. A41 and the completed clean-rescue records supersede their old launch checklists. Computational closure still leaves biological validation, independent review, and external validity unresolved.

## Source and status provenance

Current priorities come from [D55-D57 and the decision history](../documentation_sources/versions/2b7293c884e0e0aa9f1db178d13f55497d584cd7060c68c022436d47330bb63c.md), [the latest R36 directions](../documentation_sources/versions/eade325e60cbeeb92dc5f6570520d8f502a519e846309ef5803b2bf7e1a2d094.md), [the approved R36 handoff addendum](../documentation_sources/versions/8fc1116f80667885774a5ce6a8fb9ec40f1e360d1f87867fefb81baf01118ab6.md), and the newer execution record from `exp/r36-delta-objective` at `f4b96c0`.

Historical completion and withdrawal checks come from [the original TODO](../documentation_sources/versions/c7c9fba4178609946f8fd4ce8b90af08b6a88392955d7f2e198aa2e02b14cc57.md), [clean rescues](../documentation_sources/versions/eaca551880e0a9db5c4bdc7785e135675cd518ebfcd361d5aa482beca4f37dce.md), and the [R35 branch record](../documentation_sources/versions/768fdfa0a9be4dc76e99fcef9ff971e26252d672e82b08267f5e9ea157ced0f3.md) at `2f6e1f5`. These links resolve to frozen repository-local snapshots. [Source coverage](SOURCE_COVERAGE.md) records every captured version and retained operational exception. Canonical promotion still requires a later operator decision.

The runtime snapshot came from `outputs/r36-delta-objective/continuation.json` at 2026-10-11 18:30:51 UTC. The [recovered snapshot evidence](../documentation_sources/supplementary/r36-runtime-snapshot.json) preserves the exact receipt excerpt from tool-result line 701 of the saved October 10 parent session, plus its timestamp and source-line hash. The earlier temporary copy is unavailable. This draft does not inspect the moving receipt, claim a later job state, or report a fresh quota balance. The receipt's generic review field is not a reliable distinction between implementation review and result review. The original sources remain unchanged.
