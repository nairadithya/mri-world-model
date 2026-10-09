# R35. State, observed change, and nuisance comparison

Status: control diagnostics complete. First GPU candidate run is invalid. Corrected run is being prepared. No valid trained candidate result yet.

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
