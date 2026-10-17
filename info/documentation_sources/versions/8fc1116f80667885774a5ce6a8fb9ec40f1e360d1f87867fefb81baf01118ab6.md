# R36. Change-sensitive MRI representations and conditional dynamics handoff

Status. PLAN ONLY. No R36 implementation, training, or evaluation has run.

## Read this before starting

The project root is `/home/adithya/projects/world-model`. Preserve its existing tracked edits, untracked research documents, and intentional file deletions. Do not reset, stash, switch the dirty checkout, or restore deleted pages to install this experiment.

R35 implementation lives in the sibling worktree `/home/adithya/projects/world-model-representations`, branch `exp/representation-comparison`, at commit `2f6e1f5a65be4ccbceac6de706b537de8b695faa`. It is the same Git repository, not a replacement project. The current main checkout does not yet contain `src/model/representation.py`.

R35 artifacts physically live under `/home/adithya/projects/world-model/outputs/representation-comparison/`. The sibling worktree exposes that directory as `outputs`. All artifact paths below are relative to that physical directory unless specified otherwise.

Read `AGENTS.md`, `info/research_guardrails.md`, `info/methodology_validity.md`, and `info/eval_protocol.md` in the project root. Read the R35 contract and results from the sibling worktree. Resolve discrepancies before implementation. Do not assume the older sibling explainer is the current project page. The project currently uses `info/research_run.html`.

This is a finite research handoff, not a PR merge program. Execute it as a sequence of verified units with a persistent decision trail. No implementation or multi-hour run is authorized by this plan-writing request. Obtain the operator's go before execution. Phase 2 has its own authorization gate.

## State the question

Question and capability tested. Does a change-balanced objective make per-scan spatial MRI features retain localized, directional observed change without sacrificing state assessment or amplifying acquisition variation?

Current candidate. The R35 residual spatial encoder, shared between two observed scans, with a decoder driven by signed feature differences.

Materially different approach. Preserve independently estimated states and subtract their decoded masks and volumes. This is not an explicit temporal comparison network and was competitive in R35.

Simple baseline. Zero observed change, plus a frozen R35 encoder with the same post-training probes.

Rationale and assumptions. Most brain voxels are unchanged. A whole-grid objective may devote too little learning pressure to localized differences. This is a hypothesis, not an established explanation of R35. Automated mask changes are imperfect targets and include registration and annotation effects.

Matched comparisons. Hold backbone, initialization, patients, pair order, training exposures, input resolution, target availability, and evaluation readout capacity fixed. Change the objective between the key experimental arms.

Information cutoff. Phase 1 observes both scans. Phase 2 predicts a future state from cutoff-available history only. A future scan may be a training target, never an inference input to the future predictor.

Evaluation measures. Frozen-encoder local-change accessibility is primary. State and native log-volume change assessment, nuisance sensitivity, absent compartments, and developmental transfer are required companion outcomes.

Costs. Three seeds and four core objective arms. Reuse existing data and seed-specific initialization. No new cohort acquisition, native-resolution training, or public release. Check current GPU quota before scheduling any run.

Independent checks. Verify source dependence, identity behavior, signed reversal, spatial alignment, artifact identity, finite precision, and matched frozen probes. Independent design and result review remain required. Native subagent tooling was unavailable in the preceding session. Do not replace it silently or count parent self-review as independent review.

Timing limits and cohort status. Exact day horizons are not established across both cohorts. Both LUMIERE evaluation and SAILOR are repeatedly examined development data. SAILOR stays local and receives no model selection.

Falsifiers. Better training loss, greater latent variance, or a better flexible change head without better frozen features do not establish the representation hypothesis. A gain bought by worse state retention or greater false change fails the intended tradeoff.

Expected unknowns. Reliable anatomical correspondence, mask repeatability, biological attribution, available clinical context, and future predictability remain uncertain.

## Preserve the existing evidence

R35 establishes a scoped starting point, not a successful world model.

- Spatial-state direct MAE on LUMIERE was 1.1145 versus 1.4999 for the fresh frozen scalar-supervision control. The spatial model had additional dense-mask supervision.
- Independent decoded state differences gave observed-change MAE 1.2589 versus zero change 1.5446 on LUMIERE.
- The paired head did not establish an advantage over independent state differences. On SAILOR it was worse than that control.
- Resolution degradation raised spatial-state difference-probe false-change MAE from identity 0.0445 to 0.3275.
- Empty compartments had large errors. Do not hide these behind a favorable aggregate.

R35 already supervised volume deltas and lesion-upweighted mask differences. Its mask-change weight was `1 + 4 * max(source_mask, target_mask)`, followed by a whole-grid mean. R36 must test changed-region balancing and feature accessibility, not claim that simply adding delta supervision is new.

Read these artifacts before coding:

- `retry-complete/unpacked/results/comparison/run.json`.
- `retry-complete/unpacked/training_prepared.json`.
- `retry-complete/unpacked/results/comparison/seed{42,43,44}/state/best.pt`.
- `r35-final/summary.json` and `r35-final/closure.json`.
- `sailor-tensors/prepared.json`.

The state checkpoint SHA256 values are:

| Seed | SHA256 |
| --- | --- |
| 42 | `65a4c4db2ca1b44ebf29fa12659fbd09dea6ede606ce1d3279535144933a001a` |
| 43 | `395e75b47b8d4d65f5993b36af41ca8ce8c4823f303baca34f9706315ba4fd61` |
| 44 | `67747e19181fbd12758ab307ceda573143b355aa6e3420aea63f7b59ad993b6b` |

Check the bytes before loading. Record model source hashes, normalizers, and tensor-manifest hashes at creation. The earlier reference lacks a creation-time MRI-byte fingerprint. Do not retroactively relabel a completion-time audit as creation provenance.

## Unit 0. Freeze the contract and audit the targets

- [ ] Create an isolated `exp/r36-delta-objective` worktree after reviewing how to integrate the approved R35 modules with current project edits. Record its base SHA and file ownership. Do not merge the entire sibling branch blindly.
- [ ] Export an immutable `r36_contract.json` containing this matrix, coefficients, selection rules, readout budgets, primary contrast, seeds, costs, and the proposed decision margins below. Lock it before examining new evaluation results.
- [ ] Reuse the R35 patient lists. There are 52 supervised training patients, 13 internal validation patients, and 26 encoder-unseen LUMIERE patients. Read their identities from `run.json`, not a newly randomized split.
- [ ] Preserve the original usable pair rows. Do not bridge missing visits. Expected source inventory coverage is 599 visits and 496 pairs. Expected unseen assessment coverage is 149 visits from 26 patients and 117 pairs from 23 patients.
- [ ] Audit all target labels and native volumes. Preserve the three compartment definitions and the affine-aligned geometry contract. A matching affine is necessary but does not prove anatomical correspondence.
- [ ] Count changed and unchanged voxels per pair and compartment. Define change by binary compartment occupancy disagreement on the prepared grid, not by arbitrary MRI pixel differences.
- [ ] Save representative changed, unchanged, postoperative, and empty-compartment overlays locally. Record registration failures and suspect mask transitions without consulting model scores.
- [ ] Keep the full legacy-compatible pair set as the primary analysis. Any QA-filtered sensitivity analysis uses a fixed, model-blind list recorded before training. If target quality is too poor to support interpretation, stop and report that blocker rather than tune exclusions until a model wins.
- [ ] Check which native changes vanish after 96-cubed resizing. Report that count. Do not claim an objective can recover detail absent from its inputs. Native-grid detail recovery is a separate experiment.

Deliverables. Contract, patient lists, row and tensor hashes, target statistics, QA overlays, and an explicit coverage report. No GPU training before these exist.

## Unit 1. Implement one fixed architecture and its verifiers

Use the R35 `SpatialState` backbone. Four MRI sequences and four broadcast availability channels enter the encoder. Keep residual Conv3d blocks, GroupNorm, SiLU, average-pooling downsampling, and widths `(16, 32, 64, 128)`. Keep the per-scan segmentation decoder and native log-volume head.

```text
source MRI -> shared E -> Fs[0..3] -> existing state decoder and volume head
 target MRI -> same E -> Ft[0..3] -> existing state decoder and volume head
                            |
                   D[level] = Ft[level] - Fs[level]
                            |
             multiscale SpatialDecoder on signed D
                            |
             tanh output, three signed occupancy-change maps
```

Retain spatial maps at 96, 48, 24, and 12 cubed. Do not reduce the entire transition to the final pooled 128-vector. The final spatial latent is `[B,128,12,12,12]`.

The new change decoder accepts only signed feature differences. It does not accept concatenated states, images, patient IDs, scan dates, or masks. Use the existing `SpatialDecoder` shape so the new code is small. A pooled final-feature difference can feed a `Linear(128,3)` native log-volume delta readout. It is auxiliary and is not the primary proof of local information.

Independent-state mask and volume differences remain controls. A zero feature difference does not guarantee zero decoder output because of biases. Explicitly train and test identity pairs. Reversing feature differences also does not guarantee antisymmetric outputs for a nonlinear decoder. Train the signed target and report reversal error. Do not claim these properties hold by construction.

- [ ] Add the new model without changing `ObservedChange` or interpreting R35 checkpoints under a new graph. Proposed home is `src/model/delta_representation.py`.
- [ ] Keep target construction, losses, and evaluation ownership under `src/harness/analysis/`, with a separate R36 entry point. Proposed home is `src/harness/analysis/delta_representation.py`.
- [ ] Reuse `Visits`, `ObservedPairs`, geometry validation, and the unified CLI. Add `delta-representation` only after integrating the R35 dispatch. Proposed modes are `audit`, `smoke`, `train`, `encode`, `probe`, `nuisance`, `transfer`, and `report`. These commands do not exist yet.
- [ ] Preserve float32 BCE, Dice, region reductions, and normalizer calculations. Preserve AMP backoff before clipping, fatal full-precision nonfinite gradients, and bounded overflow failures.
- [ ] Add tests for same-image zero feature difference, source dependence, exact missing-sequence masking, positive and negative occupancy targets, empty changed sets, all-zero changes, pair reversal reporting, finite gradients, and source/query separation.
- [ ] Verify that masks enter only losses and reporting. Never pass target masks or target-defined crops to MRI inference.
- [ ] Test extraction against the real forward graph for the default four-level architecture. R35's final-map caching shortcut is insufficient if R36 needs every spatial level.
- [ ] Run a real aligned MRI pair through forward, backward, and one optimizer update before remote execution. Save numerical evidence, not just a compile log.

Deliverables. Tested model and losses, real CLI smoke log, source hashes, and a frozen pre-training baseline receipt. No representation-quality claim at this point.

## Unit 2. Train the locked objective matrix

Initialize every arm from the seed-matched R35 state checkpoint. Initialize auxiliary heads identically within a seed. Run seeds 42, 43, and 44. Continue for exactly 20 epochs and grade the final epoch, not an unseen-selected checkpoint. Save intermediate checkpoints for diagnosis, but do not select a different epoch for each objective using evaluation results.

Every arm processes the same ordered training pairs and computes both per-scan state branches. Use batch size 1, AdamW learning rate `3e-4`, weight decay `1e-4`, and gradient norm cap 1.0. Start a new optimizer for each arm. Do not resume an optimizer from the opt-stripped R35 checkpoint.

| Arm | Objective | Contrast it answers |
| --- | --- | --- |
| A | Continued state-only training on both visits | Extra training and state supervision control |
| B | A plus whole-grid signed-change and volume-delta losses | Ordinary delta supervision control |
| C | A plus change-balanced signed-change and the same volume-delta loss | Primary objective hypothesis, C versus B |
| D | C plus nuisance-only zero-change constraints | Nuisance tradeoff, D versus C |

Use the same architecture, normalizers, pair exposures, and auxiliary head capacity for B, C, and D. A has the same initialized module but does not update the unused change heads. Its independent state differences and post-training frozen probes are meaningful controls. Do not grade A's untrained change head.

Define the losses before execution:

- `L_state` is the average of the source and target R35 state losses. Each combines mask BCE, mask Dice, and L1 on source-training-normalized native log-volumes.
- `L_volume_delta` is L1 against native target-minus-source log-volume, using the source-training change scale. Keep this term identical in B, C, and D.
- B uses mean absolute signed occupancy error over the whole grid.
- C and D partition each compartment into changed voxels and unchanged voxels. Average error within each nonempty partition, then average the available partition means with equal weights. If there are no changed voxels, compute only the unchanged mean. Never divide by zero or invent a nonzero change target.
- Average the resulting losses over compartments and then pairs. Keep the natural pair distribution. Do not oversample large changes or normalize every observed delta to unit magnitude.
- Use `L_B = L_state + L_volume_delta + 0.5 * L_grid_change` and `L_C = L_state + L_volume_delta + 0.5 * L_balanced_change`.
- Use `L_D = L_C + 0.25 * L_nuisance`. The coefficients are proposed defaults to lock before running, not findings or clinical constants. No held-out coefficient sweep.

For D, use synthetic pairs `(x, T(x))` from training visits only. Give them zero occupancy and volume-change targets. Average the zero-change readout losses with an L1 consistency term between the two predicted state volumes. Cycle source and target visits and a deterministic perturbation schedule.

Use identity, foreground Gaussian noise with sigma 0.1, affine intensity scale 1.5 and offset 0.3, and smooth bias `exp(linspace(-0.4,0.4))`. Keep the R35 per-sequence standardization. Hold its half-resolution downsample-and-upsample transform out of nuisance training as a declared stress test. Do not train strong blur invariance that can erase lesion information without testing state retention.

For matched compute, all arms perform the same extra perturbed-scan forwards with the same schedule. Only D attaches their training loss. Log real-pair updates, nuisance exposures, successful updates, and AMP skips separately. Validation logs are diagnostic. They do not authorize new coefficients, architecture, or epoch selection.

Anti-collapse regularization is not part of the core matrix. With batch size 1, a batch-level VICReg penalty is not a valid shortcut. If it becomes a separate approved ablation, first specify independent-sample accumulation and apply the identical mechanism to both arms being compared. Do not enforce a variance floor on true stable deltas or predicted future deltas.

Deliverables. Twelve completed arms with final weights, optimizers where retained, normalizers, per-epoch logs, split and contract hashes, and input byte manifests. A time ceiling yields INCOMPLETE, not a scientific null. Do not quietly grade partial arms as a complete matrix.

## Unit 3. Test whether the encoder itself improved

Freeze each encoder. Do not let a stronger decoder alone establish the representation claim.

Primary evaluation. Train an identical local linear probe for every frozen feature pyramid. For each level, apply a shared voxelwise `Conv3d(width,3,1)` to signed feature differences, resize to 96 cubed, and average the level outputs. The probe receives no absolute states or target images. It is linear in the feature maps apart from the fixed resizing. Train it for 20 epochs on the same 52 source patients with the same balanced signed-change loss, initialization seed 2026, AdamW learning rate `1e-3`, and weight decay `1e-4`. Grade its final epoch. Save its actual weights.

The primary endpoint is patient-uniform balanced signed occupancy-change MAE for this frozen probe. Compute each pair's error by the same changed and unchanged partition rule, average pairs within patients, then average patients. Publish each partition separately so a background-only predictor cannot hide a failure on changed voxels.

Primary contrast. C versus B. This is the objective-only contrast. A versus C and C versus D are companion contrasts. A frozen original R35 encoder receives the same probe and is a useful initialization control.

Required companion evaluations:

- [ ] Direct native log-volume state MAE on all measured visits, including final visits. Report pooled and patient-uniform values and all compartments.
- [ ] Signed volume-delta decoding from pooled feature differences using the same grouped source-52 ridge protocol for every encoder. Use the existing source-only alpha grid and save scalers and coefficients.
- [ ] Direct change-decoder outputs and independently decoded state differences. Keep these separate from the frozen-probe result.
- [ ] Changed-voxel precision-recall and signed appearance and disappearance errors. Treat no-positive-target cases as explicitly empty, not perfect AUPRC. Record support counts.
- [ ] Zero change, independent-state differences, and an equally budgeted target-only local linear probe. The target-only probe uses target feature maps and the same change targets. Do not infer source contribution from gradient dependence alone.
- [ ] Identity, intensity, noise, bias, and held-out resolution perturbations with zero physical-change targets. Use the same frozen readouts as real changes. Report state drift and excess false change over identity.
- [ ] True empty and nonempty compartments, both-empty pairs, appearance, disappearance, and target-change-size strata. Reporting thresholds must be fixed from source-training targets, not evaluation outcomes.
- [ ] Within-patient permutation as a labeled post-hoc specificity diagnostic. Do not call it a biological-change test.
- [ ] All final encoders and source-fitted probes on all 269 SAILOR visits and 242 pairs, locally, without query fitting. Report the ontology and acquisition confounds and avoid a source26/source52 budget mismatch.

Use the fixed 26 encoder-unseen LUMIERE patients for evaluation. They are already development data, not a new held-out cohort. Do not use their outcomes to select a model or readout recipe. Source-26 nested probes, if retained for comparison with R35, must be separately labeled exploratory and must not replace the source-52 primary protocol.

Use three-seed mean patient errors, not predictions averaged into an unplanned ensemble. Bootstrap paired patients with 10,000 draws and seed 2026. Seeds and voxels are not independent bootstrap units. Predeclare the three companion contrasts and use family-three adjusted intervals for the primary representation comparisons. Report raw intervals too. Larger exploratory tables must remain explicitly descriptive.

Proposed progression criteria, to lock in Unit 0:

1. C has lower frozen-probe balanced change MAE than B with the family-adjusted paired patient interval below zero.
2. C preserves direct state MAE relative to A. Use an upper confidence bound below a margin of +0.05 log-volume MAE. This is a proposed research decision margin, not a clinical threshold.
3. D's nuisance loss reduces excess false change relative to C without exceeding +0.05 balanced change MAE or the same state margin. Keep the held-out resolution result separate from the trained perturbations.
4. C or D must outperform the matched target-only and independent-state controls before claiming added observed temporal information. A frozen-probe gain alone is a narrower feature-accessibility result.

If confidence intervals are wide, report INCONCLUSIVE. If only the direct nonlinear head improves, report a decoder benefit, not verified encoder improvement. If the model learns more nuisance, fails stable pairs, or loses state information, reject the intended tradeoff. These are task-specific decisions, not universal definitions of representation quality.

Deliverables. Per-row and per-patient predictions, all frozen probe weights, bootstrap draws or seeds, metric definitions, sensitivity analyses, and an aggregate report without raw SAILOR images or participant inventories.

## Unit 4. Close Phase 1 before any world-model claim

- [ ] Verify every artifact against checkpoint, source, contract, tensor, visit, and pair hashes. Persist trained probes, not just scores.
- [ ] Have an independent reviewer audit target quality, patient grouping, the objective-only contrast, nuisance information loss, and negative arms. If unavailable, retain the missing gate explicitly.
- [ ] Record VERIFIED, NOT VERIFIED, or INCONCLUSIVE for each hypothesis. Computational completion is not scientific success.
- [ ] Append results to this record and `info/decisions.md` without rewriting historical results. Update the current project research page and review its rendered pixels. Never restore a deleted explainer as a shortcut.
- [ ] Present all arms before choosing a follow-up. Keep SAILOR developmental status and empty-compartment failures visible.

Phase 1 is done when all runnable arms and evaluations have valid artifacts and all remaining unavailable prerequisites are named. No future dynamics claim follows from this phase.

## Conditional Phase 2. Test whether these changes are predictable

This phase is a separate scientific contract, not an automatic continuation of training. Obtain operator authorization after the Phase 1 report. Do not reopen the old forecast sweep unchanged.

Freeze the compared encoders. Compare the best source-validated objective family with the matched state-only control under identical dynamics capacity. Resolve any selection rule on the 13 internal validation patients before unseen evaluation, not from SAILOR or the 26 development patients.

Start with a small causal spatial-latent model, not full MRI diffusion. Project each observed scan's final `[128,12,12,12]` feature map to 16 input channels with a learned 1x1 convolution. A ConvGRU with hidden width 16 assimilates up to the four most recent observed visits in order. Pad only for batching and mask missing history explicitly. A residual output convolution maps the hidden state to a 128-channel feature-map delta. Its mean future state is the current frozen feature map plus that predicted delta.

The future teacher target is the frozen encoder's actual next feature map, not a jointly moving encoder target. Normalize teacher channels using source-training visits only. The latent loss averages errors in changed and unchanged target regions separately. Derive its target region mask by taking any compartment occupancy change at 96 cubed, then max-pooling with kernel and stride 8 to 12 cubed. This preserves small changed regions instead of letting nearest-neighbor downsampling erase them. These masks are training targets and loss weights only. They never enter the causal model. Keep unbalanced latent prediction as an objective control if this becomes the Phase 2 primary hypothesis rather than silently bundling two changes.

Decode the predicted future feature map with the same frozen global-pooling volume head and its saved normalizer. Do not substitute a separate mean-volume head that bypasses the predicted latent. An additional causal uncertainty head outputs three native-log-volume standard deviations using softplus plus a proposed floor of 0.05. The floor is a numerical research default, not a clinical uncertainty threshold. Train Gaussian native-volume NLL plus `0.1 * L_latent_delta`, with latent coefficients and the variance floor locked in the Phase 2 contract. Use batch size 1, AdamW learning rate `3e-4`, weight decay `1e-4`, and 20 epochs. Select a checkpoint by internal-validation patient-uniform native-volume NLL only. Record calibration and MRI-only persistence at that selected epoch.

Use the same dynamics graph, four-visit history cap, available context, targets, and optimizer for the compared encoders. Current-only controls receive the actual current map but have the earlier-history mask disabled. Predicting a future spatial map makes this a latent transition experiment rather than merely another scalar forecast head. It remains a scoped MRI dynamics model, not a complete biological patient simulator.

Do not claim fixed-day forecasts until a reliable prospective horizon exists. If that remains unavailable, use the existing next-observed-visit contract and explicitly model its unknown future gap. Never feed the actual next-visit gap, future treatment, future mask, future volume, or future embedding into inference. Teacher future observations are training targets only. Available treatment plans may be inputs if their cutoff availability is verified. Missing actions are not equivalent to no treatment.

Use these controls:

- MRI-only persistence of the frozen representation and its decoded volumes. Also report persistence of measured native volumes, but label it mask-assisted because its current ground-truth measurement is not an MRI-only encoder output.
- A source-fitted empirical change distribution, with the same legitimate horizon/context conditioning.
- The identical probabilistic forecast head given only the current scan features and available context.
- The identical history model on the state-only encoder.
- A cutoff-valid physical-history baseline using already available measurements, separately labeled mask-assisted if those measurements are not MRI-only outputs.

Assess patient-uniform forecast MAE, probabilistic log score or CRPS, coverage and sharpness, and incremental history information. Save each distribution and compare against a real probabilistic baseline, not a zero-variance persistence distribution. Do not choose favorable endpoints after testing. Multi-step rollout is a later capability test, not established by this one-step probe.

A better observed delta that is not forecastable supports state/change assessment, not useful dynamics. A calibrated distribution with similar point MAE can still be useful, but its gain must be established against matched probabilistic controls. Treatment-conditioned forecasts do not identify causal treatment effects or validate counterfactual care plans.

## Bound resources and stop safely

- No GPU exists on the dev host. Perform contract, geometry, numerical, and two-pair optimizer checks locally before a private T4 run.
- Recheck actual quota. Do not assume the preceding session's 30 hours remain available.
- Start with one seed's A and B capacity run. Estimate full-matrix cost from measured successful updates and evaluation overhead. Do not inspect encoder-unseen or SAILOR outcomes during this capacity check.
- Reserve a nine-hour ceiling per session. Proposed total Phase 1 reservation is at most 24 GPU-hours, subject to available quota. If the matrix does not fit, report the capacity blocker and revise the schedule before starting it. Do not drop unfavorable arms or shorten only some arms after seeing results.
- Keep training inputs limited to existing private LUMIERE data. No SAILOR upload, new data acquisition, public push, deployment, or acceptance of a data-use agreement is authorized.
- Log owned process IDs and runtime status. A bounded mechanical monitor may record progress but is not an independent reviewer or an automatic parent-chat wake mechanism.
- Preserve incomplete outputs with their status. Stop on persistent nonfinite gradients, missing native-volume agreement, incorrect row identity, suspect source/target mixing, or loss of target QA coverage. Never disable guards to finish a table.

## Give the implementer exact verification commands

The following commands exist. Run them from the R35-capable execution worktree after integration:

```bash
.venv/bin/python tests/test_representation.py
.venv/bin/python tests/test_representation_report.py
.venv/bin/python tests/test_harness.py
.venv/bin/python scripts/check_notebook_pairs.py
.venv/bin/python scripts/harness.py representation smoke \
  --prepared /home/adithya/projects/world-model/outputs/representation-comparison/smoke-tensors \
  --out /tmp/r36-existing-smoke.json
```

R36 tests and commands are deliverables to implement, not current interfaces. Add `tests/test_delta_representation.py` for loss semantics, source dependence, empty sets, and gradient behavior. Add `tests/test_delta_representation_protocol.py` for cutoff boundaries, matched budgets, fixed splits, and artifact identity. Exercise the new command through `scripts/harness.py`, not only imported functions. Follow the CLI verification skill under `/home/adithya/projects/world-model/.agents/skills/verify-world-model-cli/SKILL.md`.

Before commit, compile touched Python files, check the diff, inspect privacy-sensitive outputs, and regenerate any changed executable notebook from its `.py` source. Never compile Kaggle notebook sources containing live shell commands as ordinary Python.

## Deliver the handoff receipts

- [ ] Unit 0 contract and target QA.
- [ ] Unit 1 unit tests and real command-line optimizer evidence.
- [ ] Unit 2 complete, incomplete, or failed training receipts for every planned arm.
- [ ] Unit 3 matched frozen-probe, state, change, nuisance, and transfer reports.
- [ ] Unit 4 interpretation and independent-review status.
- [ ] Explicit operator decision before Phase 2.

Persist a decision trail at `.audit/r36-delta-objective.tsv`. Every row names the hypothesis, changed code or contract, artifact evidence, and whether the result moved the stated predicate. Keep private inventories and raw data under ignored outputs. Commit code, aggregate findings, and reproducible protocol definitions only.

## Alternatives and unresolved questions

Reject indiscriminate latent repulsion. It can reward patient identity and acquisition variation, and genuine stable states need not move.

Do not assume a future-latent loss becomes materially different merely by writing it as delta prediction. With fixed reference latents, adding the source state back gives the same squared prediction error. Local targets, balanced weighting, and nuisance constraints are the substantive changes here.

Do not change the backbone, native spatial sampling, decoder capacity, and objective in one comparison. Native-grid processing and transformers remain separate hypotheses.

Do not count a fixed three-volume target as a complete pathological state. This plan also examines local occupancy and acquisition responses, but expert biological labels and genuinely untouched external validation remain unavailable.

The proposed coefficient and noninferiority values are decisions to approve and lock, not values derived from a pilot. The target QA, training runtime, nuisance severity tradeoff, and ability of a linear probe to access local change remain empirical questions. No new prototypes have established them in this planning turn.

Related reading includes VICReg (`https://arxiv.org/abs/2105.04906`) and temporal-equivariant OCT representations (`https://papers.miccai.org/miccai-2024/452-Paper3246.html`). These motivate hypotheses, not evidence of success in glioma. Apply Model the Domain by keeping observed change, nuisance-only pairs, and forecast cutoffs explicit. Apply Prove It Works by checking real MRI pairs and persisted predictions before interpreting metrics.

## Execution addendum, 2026-10-10

The operator authorized Phase 1 execution after this planning record. Implementation is isolated in `/home/adithya/projects/world-model-r36` on `exp/r36-delta-objective`. The main checkout's existing edits and intentional deletions are preserved.

Unit 0 target and tensor QA passed for all 599 visits and 496 pairs. The immutable private contract and receipts are in `outputs/r36-delta-objective/qa/`. Unit 1 synthetic tests and two real command-line MRI optimizer updates passed. Unit 2 has begun a private seed42 A/B capacity run with source-only baseline probes and no query scoring. The full matrix, evaluations, and independent review are not complete.

See `info/r36_delta_representation_results.md` for measured findings and changed priors. Root documentation receives append-only findings. Code remains isolated. Phase 2 still requires a separate operator decision after the Phase 1 report.
