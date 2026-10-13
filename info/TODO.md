# Research TODO

This checklist records remaining work after the October 13, 2026, R36 Phase 1 closure. It uses saved reports, not live job status. Earlier launch checklists remain in the [October 11 snapshot](documentation_sources/versions/e3c486ef9cc77afcc1978e53ce4495ef1051a5b8d22cf31b9d627c72d4db029e.md) and the [dated archive](archive/2026-10-13/README.md).

Use [Notes and insights](NOTES_AND_INSIGHTS.md) for interpretation, [Experiments](EXPERIMENTS.md) for results, and [Methods](METHODS.md) for protocols. The [research guardrails](research_guardrails.md) govern new implementation. This checklist does not authorize experiments, data acquisition, or a change of method family.

## R36 Phase 1 is closed for scoped reporting

The [completed findings](EXPERIMENTS.md#r36-completed) and [closure receipt](reviews/r36_closure_receipt.json) replace the old seed-43 and seed-44 launch tasks. Do not rerun completed work because an archived checkbox is open.

- [x] Complete twelve final-epoch training arms, thirty source-only probes, and both development-cohort assessments.
- [x] Preserve the bounded CPU recovery receipts without repeating GPU work.
- [x] Correct nuisance-artifact identity and coverage checks, with rejection fixtures.
- [x] Classify demonstrated adverse tradeoffs separately from overlapping intervals and missing evidence.
- [x] Obtain the [independent scoped result review](reviews/r36_phase1_result_review.md).
- [x] Complete the [independent artifact audit](reviews/r36_phase1_artifact_audit.md), including saved ridge replay and patient-level contrast recomputation.
- [x] Present all arms and required companions in the [research page](research_run.html) and [aggregate measurements](r36_phase1_metrics.json).

The primary C-versus-B encoder gain, D's nuisance tradeoff, and superiority over independent state differences remain INCONCLUSIVE. C passes the specified direct-volume retention margin against A. That is not a general state-improvement claim. No follow-up family is selected, and Phase 2 remains unauthorized.

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

The [R36 interpretation](NOTES_AND_INSIGHTS.md#d58) separates the descriptive nonlinear-decoder benefit from the unresolved encoder hypothesis. The completed comparison does not select an architecture or authorize another experiment.

Before substantial research implementation:

1. State the capability under test and the proposed information cutoff.
2. Write the [experiment decision brief](research_guardrails.md#experiment-decision-brief).
3. Include a materially different approach, a simple baseline, matched comparisons, costs, independent checks, falsifiers, and unknowns.
4. Obtain the operator's decision on changes to scientific targets, method-family priorities, and what counts as success.

### R36 Phase 2 remains unauthorized

The [conditional dynamics proposal](METHODS.md#r36-phase2) requires a separate operator decision. It is not an automatic continuation of Phase 1.

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
