# R35. State, observed change, and nuisance comparison

Status: all runnable R35 experiments COMPLETE. All 12 trained arms, both fixed-source transfer readout budgets, direct heads, nuisance conditions, and state audits are verified. Biological validation, independent review, and untouched external validation remain blocked by missing prerequisites.

## Question and capability tested

Can an MRI-only spatial state encoder and an explicit observed-pair encoder preserve measured anatomical state and observed compartment changes better than frozen clean BRAINIAC features? Can they avoid false change under declared acquisition perturbations? These are proxy representation tests. They do not replace the P2 forecast contract or establish biological change.

## Candidates and simple baselines

The reference is the clean survival-free BRAINIAC backbone, without clinical inputs. The materially different candidate is a small spatial encoder with a dense three-compartment decoder and native log-volume supervision. An observed-pair model shares that encoder and compares multiscale features. A later-scan-only model and a frozen-state pair head isolate the need for the earlier scan and for encoder adaptation. Train-mean state, zero change, independent state-readout differences, and grouped-CV ridge on frozen features are controls.

## Matched comparisons

Keep the existing 65 encoder-training patients and 26 encoder-unseen patients. Partition the 65 internally with seed 2026 for training and validation. No unseen patient selects an epoch or architecture. Train state encoders for 20 epochs. Train each pair arm for 20 epochs, initialized from the selected state encoder. Execute seeds 42, 43, and 44 in that order. Pair arms are full paired comparison, later-scan-only comparison, and frozen state encoder. All pair arms receive the same target-visit state supervision and observed-change targets. Use the same pair rows, optimizer, batch size, and stopping rule. The frozen reference has a different pretraining history, so differences from it do not isolate architecture causally.

Architecture widths are 16, 32, 64, and 128. Inputs include four MRI sequences and explicit availability. Masks never enter learned MRI inference. Current-state decoding includes all measured visits, including final visits. Pair evaluation uses only original consecutive usable manifest rows. No bridge across a missing measured visit is created.

## Information cutoff and patient-clean split

State inputs contain one observed visit. Change inputs contain two observed visits. Both visits are legitimate for this assessment task. No claim about predicting an unobserved visit follows. Do not reuse these pair inputs in a forecasting command. Source and target visit IDs remain explicit in artifacts. Fit normalization and select checkpoints on the internal encoder-training partition only. Frozen-feature readouts use nested patient-grouped CV on the encoder-unseen cohort.

## Evaluation measures

State targets are the three native log1p compartment volumes. Change targets are their observed differences. Report pooled and patient-uniform MAE separately, per-compartment errors, and paired patient bootstrap intervals. Persistence means zero change for the observed-change diagnostic. It is not a future prediction result. Dense mask losses are supervision diagnostics, not independent confirmation.

Use paired feature probes, later-only probes, and differences of independent state readouts to test incremental pair information. Evaluate identity pairs and nuisance-only pairs with zero physical-change targets. Test noise, resolution degradation, smooth intensity bias, and affine intensity changes. Keep training and evaluation perturbation settings explicit. Report all nuisance arms rather than selecting a favorable one.

## Costs and scope

The first private T4 session has a nine-hour ceiling. The total program does not consume more than the observed 30-hour available quota without a new capacity check. SAILOR stays local. No cohort acquisition, public source push, or SAILOR upload is authorized by this run. Only existing private LUMIERE inputs and a private code-and-LUMIERE-metadata snapshot go to Kaggle. Full native-resolution training is separate from the matched 96-cubed comparison. Retained native images permit a later local spatial-sampling diagnostic, not an assumed GPU input.

## Independent checks

Test shape, availability, geometry, row identities, duplicate IDs, patient disjointness, source dependence, and exact absence of source dependence in the later-only arm. Test identity and nuisance transforms on synthetic fixtures. Run a real aligned MRI forward and backward before remote launch. Keep weights, normalizers, predictions, configs, and source hashes. The native subagent tool is unavailable in this session. Independent design and result review are missing gates, not replaced by self-review.

## Falsifiers and stop rules

A lower training loss alone is not a finding. Do not claim a development proxy gain unless a prespecified paired patient interval excludes zero against its relevant control. Do not claim longitudinal benefit if the model fails the later-only and independent-state-difference controls. Nuisance robustness must not come at the cost of state accuracy. Do not tune on SAILOR or unseen outcomes. Do not expand an unchanged failed sweep. A runtime ceiling yields an incomplete run, not a scientific null. Negative and inconclusive arms remain in the record.

## Timing limits and cohort status

Visit order is verified. Exact day horizons are not. LUMIERE and SAILOR have different automated measurement sources. Existing evaluation patients have been examined before. All results are development evidence. New expert change annotations and a genuinely unexamined cohort remain necessary for biological or independent-transfer conclusions.

## Expected unknowns

Segmentation repeatability, biological meaning of empty-compartment transitions, anatomical registration quality, whether the small training set supports pair learning, and whether apparent nuisance stability generalizes to real acquisition differences remain unknown.

## Implementation design

Two shapes were considered by the parent. Extending the legacy RANO trainers exposes assessment and forecast assumptions through several callers. A standalone representation package owns visit and observed-pair contracts and reuses the physical-grid boundary. The standalone shape is selected. This is not an independent arena result.

`src/data/representation.py` owns portable inventories and tensor preparation. `src/model/representation.py` owns the shared spatial state and observed-pair models. `src/harness/analysis/representation.py` owns fixed training, nested probes, nuisance tests, and artifact records. The unified CLI dispatches `representation` commands to that package.

The local decision trail is `.audit/representation-comparison.tsv`. Numerical artifacts are under the ignored `outputs/representation-comparison` directory in the original checkout, exposed as `outputs` in this isolated worktree.

## Control diagnostics, 2026-10-09

Artifact: `outputs/frozen_controls.json`. On 149 measured visits from 26 encoder-unseen patients, cached MRI-only ridge gave patient-uniform state MAE 1.6008 versus 1.8359 for the fold-trained mean. Learned minus mean was -0.2351, with paired patient 95% interval [-0.4280, -0.0423]. On 117 observed pairs from 23 patients, feature differences gave change MAE 1.5272 versus 1.5446 for zero change. The difference interval [-0.0893, 0.0519] crossed zero. These are native automated-compartment log-volume proxies, not expert-change or forecast outcomes.

A post-hoc within-patient permutation diagnostic increased state MAE to 1.7543. Original minus shuffled MAE was -0.1535, with patient bootstrap interval [-0.2230, -0.0879]. Artifact: `outputs/frozen-state-shuffle.json`. This suggests visit-specific information beyond a patient's average burden. It does not distinguish biological change from visit-specific acquisition effects. It is not a prespecified success gate.

A scalar-supervision control uses the same 52 training patients as the spatial model, rather than fitting readouts on unseen-patient folds. Artifact: `outputs/frozen-supervised-controls.json`. State MAE was 1.5000 versus 1.7051 for the source-training mean, with difference interval [-0.3792, -0.0361]. Pair concatenation MAE was 1.4882 versus 1.5446 for zero change. All change intervals and the incremental source controls must be inspected before interpreting that point estimate. This control has no dense-mask objective. Spatial-model comparisons cannot isolate architecture from additional dense supervision.

The cached reference records its creation checkpoint path, epoch, and validation loss, but not checkpoint bytes. Those metadata match the current clean checkpoint. Fresh GPU reference extraction is still required before attributing these cache-backed diagnostics definitively to that checkpoint.

The existing `p0_sailor_cache.pt` has no visit identifiers. It cannot support a safe new state join by guessing session order. The new cache join rejects it. Fresh local MRI encoding will be used instead. This is an artifact limitation, not a transfer score.

## Execution receipt

Private kernel: `nairadithya/r35-representation-comparison`, version 1. Server metadata confirms `is_private: true` and `machine_shape: NvidiaTeslaT4`. Source commit: `e28340827527237476b4713a9cd278da4ad313bd`. Snapshot SHA256: `971b1f1c6a649da36938e569d542da50b5bf0e1d767f30e56e24d4126470ff2c`. Only LUMIERE metadata and existing private LUMIERE inputs are in that session.

Nine representation contract tests, the existing 25 harness tests, geometry and temporal-validity regressions, Python compilation, notebook-pair synchronization, and the real CLI listing passed before launch. A real aligned 96-cubed visit passed forward, finite backward, and one optimizer step. Its loss changed from 11.6868 to 10.7578. That is a runtime check, not model quality evidence.

Local SAILOR preparation completed with 269 visits and 242 pairs. Native volume agreement and pair grid identity passed. Independent review is still unavailable. GPU completion, fresh-reference agreement, matched controls, and transfer remain pending.

## Numerical repair and fixed rerun

The first GPU candidate run is INVALID for model claims. A later 96-cubed mixed-precision fixture exposed overflow in the Dice probability sum. The first loss accumulated the sigmoid map in float16. The synthetic all-one-mask fixture produced loss 1.693359 in half precision versus 1.026480 in float32. No candidate test scores had been inspected when this defect was found. The repair converts dense logits and masks to float32 before loss and reduction. A regression now checks loss agreement and finite, nonzero half-input gradients.

The corrected checkpoint contract records `dense_loss_contract: float32-reductions-v1`. Transfer rejects pre-repair checkpoints. Corrected artifacts bind features to checkpoint digests and include split, source-data, normalization, and training details. Direct trained heads are saved separately from nested unseen-cohort readouts. The scalar-volume frozen control is their relevant simple comparator. Dense-supervision and pretraining differences remain explicit.

The installed Kaggle CLI supports `kernels logs --follow`; the live log confirmed Tesla T4. The batch cancellation API needs a session ID that this client does not expose for the launched batch. No ID was guessed and no unrelated session was touched. The first run remains capped at nine hours and will not be graded. Quota was rechecked at 0.94 used and 29.06 remaining hours. A distinct corrected private session also has a nine-hour ceiling, keeping worst-case total reservation below 30 hours.

Generated notebook pairs contain a private LUMIERE data inventory. They live under ignored `outputs/private-kernel*`, not in committed `kaggle/`. Commit only the packaging source, tests, and aggregate research record. Verify the generated pair with Jupytext before every push.

## Resume and optimizer repair, 2026-10-10

The laptop interruption affected local execution. After resume, the original local reference process completed and saved all 269 SAILOR visit features. It did not need restarting. The cloud run failed independently, before completing a candidate epoch. Its gradient clip raised on nonfinite scaled gradients before GradScaler could skip the update and reduce the scale.

A controlled half-intermediate fixture reproduced this ordering defect. Its forward loss was finite, its scaled gradient was infinite, and the old clip raised. Native GradScaler left the parameter unchanged and reduced the scale from 65536 to 32768. The shared optimizer-step boundary now lets AMP back off before clipping, clips only finite gradients, and records backoffs and successful update counts. Full-precision nonfinite gradients remain fatal. Sixteen consecutive AMP backoffs or an epoch with no successful update also stop execution. This is a numerical lifecycle repair, not a new scientific target or a relaxation of finite-weight requirements. The cloud mechanism still requires confirmation by the restarted run.

Fourteen representation tests passed, including unchanged weights on AMP overflow, a subsequent finite update, and rejection of full-precision nonfinite gradients. The real aligned-MRI state and paired CPU optimizer checks and the existing 25 harness tests passed. Independent review remains unavailable.

Restart inputs may reuse the failed cloud job's prepared LUMIERE tensors only when their inventory SHA256 matches the current private inventory exactly. The downloaded metadata passed that comparison. Reuse does not alter the cohort, masks, split, targets, or resolution. The rerun keeps seeds 42, 43, 44 and the same 20-epoch schedule, with a nine-hour session ceiling. Available quota was rechecked at 30 hours.

## Fresh-reference confirmation and developmental transfer

The cloud reference recomputed state MAE 1.600834 on the same 149 LUMIERE visits, confirming the earlier cache-backed point estimate. Reference artifacts were recovered from the failed job before any restart. The reference uses no dense training loss, so that candidate-training failure does not invalidate its feature extraction.

Artifact: `outputs/fresh-sailor-transfer.json`. Source-only grouped-ridge fitting on the 26 LUMIERE encoder-unseen patients gave SAILOR state MAE 4.7075 versus 2.2391 for the source-trained mean on 269 visits from 27 patients. Observed feature-difference change MAE was 0.9791 versus 0.9430 for zero change on 242 pairs. This is negative developmental-transfer evidence for the specified readout. Acquisition and automated-label-source differences remain confounded. It is not untouched external validation, a biological-change verdict, or a universal representation-quality gate. No SAILOR data were uploaded or used to select checkpoints.

## Restart receipt

Kernel: `nairadithya/r35-representation-retry`, version 1. Source commit: `3349acf`. Server metadata confirms private execution and the requested Tesla T4 machine shape. Status at restart verification was RUNNING; a completed candidate epoch has not yet been observed. Inputs may reuse the hash-matched prepared tensors from the failed fixed run. The original local encoding and failed-cloud reference artifacts are preserved. The decision trail records both the numerical repair and the negative transfer result. No unobserved training progress is claimed.

## Completed corrected comparison

The private retry completed all four arms for seeds 42, 43, and 44. Local artifact verification checked all 240 epochs, finite saved weights, matching feature/checkpoint digests, and training split identities. There were 74,592 successful optimizer updates and 48 recorded AMP backoffs. The numerical repair allowed recovery without accepting nonfinite parameter updates. Artifacts: `outputs/retry-complete/verification.json`, `outputs/retry-complete/summary.json`, and the saved weights and predictions in `outputs/retry-complete/unpacked/results/`.

The following are means of patient-level errors across the three seeds, not errors of an ensemble prediction. Summary intervals are descriptive paired patient bootstraps, not new prespecified gates.

- Direct spatial-state volume decoding gave MAE 1.1145 versus 1.5000 for the frozen scalar-supervision control. Difference -0.3855, 95% interval [-0.5185, -0.2481], 149 visits from 26 patients. The point reduction is about 26%. The spatial model also receives dense masks during training, so this does not isolate architecture from supervision.
- The direct paired head gave observed-change MAE 1.2696 versus 1.5446 for zero change. Difference -0.2750, interval [-0.4953, -0.0957], 117 pairs from 23 patients. Later-only head MAE was 1.4381; paired minus later-only was -0.1685, interval [-0.3438, -0.0155].
- Independent differences of the state model's decoded volumes gave MAE 1.2589. Paired minus independent difference was +0.0108, interval [-0.0791, 0.1066]. Differences from the paired model's own state decoder gave MAE 1.2348. Thus a dedicated paired head has not established incremental benefit beyond useful per-scan state decoding.
- The nested-probe results tell the same cautionary story. Paired learned-comparison feature MAE ranged 1.2151 to 1.3006; simple differences of that encoder's scan embeddings ranged 1.1554 to 1.2273. Report all views rather than selecting the favorable one.
- Synthetic resolution degradation still produced false-change scores. In the spatial-state difference probe, these ranged 0.2458 to 0.3827 versus identity scores 0.0343 to 0.0501. Nuisance separation is incomplete. These are declared synthetic perturbations, not measured biological changes.

Interpretation: supervised spatial state learning improved the scoped in-domain volume proxy. Useful state readouts also support observed-change estimates. A more elaborate paired network has not earned its place over independent state differences. This is not future forecasting, biological-change validation, or a claim of general architectural superiority. Trained-model SAILOR transfer has not yet been run. The negative frozen-reference transfer result above remains in the record. Independent review is still unavailable.

## Completed local transfer and remaining diagnostics

The fixed local matrix completed for all 12 trained arms. No checkpoint, architecture, regularizer, or target was selected using SAILOR. Each arm saved all 269 query visit embeddings and, where applicable, all 242 pair embeddings. It received three separate evaluations: direct source-52 trained heads, grouped-ridge readouts fitted on the 26 LUMIERE encoder-unseen patients, and grouped-ridge readouts fitted on the 52 source-training patients. The fresh frozen reference received matching source-only readout fits. SAILOR remained local throughout.

Artifacts are `outputs/sailor-matrix/` and `outputs/r35-final/summary.json`. Source scalers and fitted ridge weights are saved as `*.readouts.pt`. A cached replay persisted these readouts without changing any of the 36 prediction artifacts. Saved readouts reproduced the query predictions. Every arm passed visit and pair identity checks, finite-feature checks, checkpoint binding, and source/query patient disjointness checks. An original-forward comparison on a real SAILOR pair gave exactly identical embeddings for the cached extraction in paired, later-only, and frozen modes.

The tensor audit records completion-time hashes for every query tensor. It does not retroactively supply a creation-time MRI-byte fingerprint for the earlier reference cache. Model and prepared-metadata digests and the checked row identities remain explicit.

### Developmental transfer results

These values average patient errors across all three seeds. They are not an ensemble and do not treat seeds as independent patients.

- The direct spatial-state model gave SAILOR MAE 2.2401 versus 4.6876 for the matched frozen scalar-source52 readout. It did not establish a gain over the source-trained mean of 2.1105. Model minus mean was +0.1296, with descriptive patient interval [-0.2571, 0.5098]. Lower error than a weak frozen reference is not sufficient transfer evidence.
- Source-26 ridge on spatial-state embeddings gave MAE 1.9715 versus mean 2.2391. Its difference interval [-0.6190, 0.0760] crossed zero. Source-52 ridge gave 1.9525 versus mean 2.1105, with interval [-0.4966, 0.1759]. Neither readout budget resolves superiority over the simple mean.
- Direct independent state differences gave SAILOR observed-change MAE 0.9665 versus zero change 0.9430. Difference +0.0235, interval [-0.0368, 0.1052]. The direct paired head gave 1.0508. Its excess over independent state differences was +0.0843, with family-12 interval [0.0295, 0.1440]. The dedicated paired head is worse than that simple control on this transfer proxy.
- The paired head beat the later-only head at the point estimate, 1.0508 versus 1.1954. Its family-12 difference interval [-0.3001, 0.0224] crossed zero. Even a resolved later-only contrast would not override failure against zero change and independent state differences.
- All calibrated feature views are retained in the report. For example, source-26 difference probes in the paired arm gave 0.9422 versus zero 0.9430. This near tie is not a positive transfer verdict and does not justify choosing that view after inspection.

### Nuisance and state specificity

All five perturbation conditions are complete for all four candidate arms and the frozen reference. Noise uses seed 2026 and foreground standard deviation 0.1. Resolution halves each image dimension and restores it with trilinear interpolation. The smooth bias multiplies one axis by exp of coordinates from -0.4 to 0.4. Affine intensity uses scale 1.5 and offset 0.3. Nonidentity variants are standardized per present sequence on the original foreground. Identity is an exact clone. In the spatial-state difference probe, identity false-change MAE was 0.0445. Noise gave 0.0433, affine intensity 0.0445, smooth bias 0.0845, and resolution degradation 0.3275. Resolution exceeded identity by 0.2831, with descriptive patient interval [0.1986, 0.3747]. State-readout drift under resolution degradation was 0.6228. Acquisition separation is not established. Blur removes image detail, so this controlled physical-invariance test does not prove that the lost information was only nuisance. Affine intensity stability largely follows explicit standardization and is not a learned robustness claim.

Post-hoc within-patient shuffling raised direct spatial-state MAE from 1.1145 to 1.5162 on LUMIERE and from 2.2401 to 2.3361 on SAILOR. All three seed-specific original-minus-shuffle intervals were below zero in each cohort. This supports visit-specific information, not biological attribution or a new prespecified success gate.

Empty-compartment errors remain large. For the direct state model, source empty-target patient-uniform MAEs were 7.0523 for core on 7 visits, 5.7072 for enhancement on 22 visits, and 9.5619 for edema on 5 visits. SAILOR empty-target errors were also large. The headline mean must not hide absent-compartment failures. Compartment definitions and automated measurement sources differ across cohorts, so biological interpretation still needs expert annotation and repeatability analysis.

### Closure and limits

The finite R35 experiment matrix is complete. The exit condition covered all trained arms, matched transfer budgets, nuisance conditions, saved predictions and readouts, and permutation and empty-target audits. Code verification includes 16 representation tests, four report tests, the 25 existing harness tests, real CLI execution, real-pair extraction equivalence, and notebook-pair synchronization.

The current finding remains scoped. Dense-supervised MRI state learning lowered in-domain automated-volume error. Dedicated pair modeling did not add a resolved advantage over useful independent state differences and transferred worse than that control. Neither developmental transfer nor nuisance separation was established. No forecasting experiment was reopened.

Three prerequisites remain unavailable. Biological-change validation requires expert labels and segmentation repeatability. Independent review requires native delegation or another operator-approved review path. Untouched external validation requires a newly reserved cohort and its access approval. These are blocked evidence requirements, not unfinished CPU runs and not claims of completion.

The complete aggregate report is `info/r35_representation_results.html`. `info/explainer.html` links it and warns that older lesion forecasting and unaligned crop claims below are withdrawn. Both pages were rendered locally and reviewed as images. No legacy plot numbers were regenerated or promoted as new evidence.

Reproduce from the preserved private artifacts in this worktree:

```bash
.venv/bin/python scripts/finish_representation_run.py \
  --prepared outputs/sailor-tensors \
  --results outputs/retry-complete/unpacked/results/comparison \
  --training-metadata outputs/retry-complete/unpacked/training_prepared.json \
  --out outputs/sailor-matrix
.venv/bin/python scripts/report_representation_run.py \
  --results outputs/retry-complete/unpacked/results/comparison \
  --transfer outputs/sailor-matrix \
  --reference-source outputs/retry-complete/fresh-frozen-source52.predictions.npz \
  --prepared outputs/sailor-tensors --out outputs/r35-final
.venv/bin/python scripts/render_representation_report.py \
  --source outputs/r35-final/summary.json \
  --out info/r35_representation_results.html
```
