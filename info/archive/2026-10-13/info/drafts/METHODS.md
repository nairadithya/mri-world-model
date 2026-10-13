# Methods

Draft, 2026-10-11. Existing canonical records remain authoritative. This reference consolidates their methods without replacing the locked contracts. Read [TODO](TODO.md) for current work, [Experiments](EXPERIMENTS.md) for measurements, and [Notes and insights](NOTES_AND_INSIGHTS.md) for interpretation.

<a id="research-objective"></a>
## Research objective and scope

The objective is to build useful longitudinal MRI representations that preserve anatomical and pathological state and meaningful change over time while separating acquisition nuisance. Per-scan representation quality, observed-change assessment, and prediction of future observations are distinct questions and must be evaluated separately. Lesion awareness, JEPA, forecast heads, and whole-brain alternatives are hypotheses, not the objective.

The scoped RANO-free lesion-state experiment is not a universal definition of representation quality or a gate for every method. A failed endpoint does not establish that useful representations are impossible. Conversely, a lower training loss, a favorable readout, or a lesion forecast alone does not establish the broader objective. Claims about longitudinal benefit require checks that history adds information beyond the current scan, as well as nuisance and shortcut checks.

Operator approval controls changes to scientific targets, method-family priorities, and whether a proxy counts as success. Substantial research implementation requires a brief identifying the candidate, a materially different approach, simple baseline, assumptions, matched comparisons, information cutoff, costs, independent checks, falsifiers, and expected unknowns. Reports distinguish measurement, interpretation, and unknowns.

## Data, geometry, labels, and timing contracts

### Cohorts and patient identity

LUMIERE is the development cohort. The established patient partition contains 65 encoder-training patients, 13 development patients, and 13 historical final patients. The 26 patients in the latter two groups are encoder-unseen relative to the original champion; the fixed five-fold partition uses seed 2026. R35 and R36 use the 65-patient source pool, with an internal 52-patient training and 13-patient validation split, and assess on the same 26 encoder-unseen patients. These patients and the historical final13 have been reused; neither constitutes a new independent external holdout.

SAILOR is controlled-access developmental transfer, not untouched external validation. Keep it local. Do not upload SAILOR data to cloud services, use its outcomes for model selection, or describe its transfer results as independent confirmation.

All splits and joins are patient-clean and preserve visit and pair identities. Do not bridge missing visits or create a new "next" visit by dropping an ineligible intermediate visit. Immutable row IDs, patient lists, source and target visit IDs, and pair eligibility must travel with caches and predictions.

### Geometry and preprocessing

BRAINIAC inputs use the contract in [`preprocessing.md`](../preprocessing.md): per-modality N4 correction, 1 mm isotropic resampling, rigid MNI registration when a template is explicitly supplied, skull stripping, resize to 96³, and nonzero-voxel intensity normalization. The preprocessing stage resamples to 1 mm isotropic before the final resize to 96 cubed. The resized tensor is not a physical 1 mm ruler. Training inputs are preprocessed offline. Missing template registration is an explicit warning, not an implicit assurance of MNI alignment.

Physical lesion volumes are measured from native masks and affines before tensor resizing. Geometry-aware mask alignment uses verified MRI pre-resize grids and nearest-neighbor resizing. Identity-affine MRI tensors require content-bound geometry provenance, an exported geometry manifest, or a retained intermediate that reconstructs the final tensor. Reject missing or stale geometry and inconsistent modality grids. A valid affine or matching grid is necessary for alignment but does not prove template equivalence, anatomical correspondence, accurate registration, or accurate segmentation. Never infer template equivalence from grids.

LUMIERE lesion-state measurements use recovered DeepBraTumIA atlas masks and the empirically verified label mapping `{1: enhancing, 2: necrotic/nonenhancing, 3: edema}`. SAILOR's primary source is the complete ONCO mask triplet; CL is a required target-sensitivity arm, not interchangeable truth. SAILOR `-icor` files may contain background NaNs; finite-check inputs and prefer usable base variants. Acquisition and segmentation source differ between cohorts, so report source-stratified results.

### Labels and outcomes

The shared label contract separates clean response labels, treatment, operative events, and validity. Clean RANO response labels are {PD, SD, PR, CR}; operative or ambiguous ratings and missing values are not silently mapped to SD. Unknown or operative-ambiguous legacy action values use -1. Existing checkpoints may load numerically but are not thereby certified as trained under the corrected label contract. Prospective clinical vectors exclude eventual overall survival; survival can only be a separate censored target.

<a id="lesion-state-v1"></a>
The `lesion-state-v1` target contains three physical volumes in mm³. The compartments are `necrotic_non_enhancing`, `enhancing`, and `edema_flair`. Primary targets use future `log1p` volume and `delta log1p` volume. Any later growth/stable/shrinkage categories are imaging-change categories, not clinical progression or response.

The frozen eligibility rule joins consecutive curated visits when source and target both have all three measurements and the ordered gap is positive. The audited eligible set contains 496 LUMIERE pairs and 242 SAILOR pairs. The median usable gap is 84 days in LUMIERE and 77 days in SAILOR. Because visit timing does not support exact day-level horizons, the primary forecast is **next observed visit**. `gap_days` is auxiliary or a sensitivity variable; it does not support fixed-horizon, precise rate, or calibrated time-to-event claims.

### Finite, missing, and zero-volume behavior

Drop imageless visits and require usable pixels on both sides of any image-based loss pair. Check finite values at input and in losses, metrics, and saved artifacts. Do not let pandas NaN truthiness or present-but-empty scan files pass as valid observations.

A missing measurement is not a zero-volume measurement. A true empty compartment has measured volume zero, represented by `log1p(0) = 0`. Preserve absent-compartment and both-empty cases in the data and report support counts; do not discard them to improve scores. For a partitioned change loss, average over available partitions only: when no changed voxels exist, calculate the unchanged-region term alone. Never divide by zero or invent a nonzero change target. For precision-recall metrics, no-positive-target cases are explicitly empty, not perfect AUPRC.

## Evaluation protocols

### Per-scan state and observed change

Per-scan assessment observes one visit. Observed-change assessment observes both visits and measures change between them. R35's observed-pair comparison is not a future forecast and its two-visit inputs must not be passed to forecasting evaluation.

For the R35/R36 development assessment, preserve the 26 encoder-unseen LUMIERE patients, the 149 measured visits, and the 117 consecutive usable pairs from 23 patients. The 65-patient encoder-training pool is partitioned internally as specified by each frozen contract; unseen outcomes do not select epochs or architecture. Report pooled and patient-uniform errors separately, along with per-compartment values and paired patient-level uncertainty. In R35, all three seeds are reported as means of patient-level errors, not as an ensemble prediction.

Use task-appropriate controls. State assessment includes a source-training mean or scalar-supervision control. Observed change includes zero change, independent differences between per-scan state readouts, a later-scan-only control, and a frozen-feature readout where specified. A dedicated paired head must establish incremental value over independent state differences, not merely beat zero change. Dense-mask supervision is a training diagnostic, not independent confirmation.

Nuisance checks include identity, intensity, noise, smooth bias, and declared resolution perturbations. Report identity false change, state drift, and excess false change separately. A synthetic perturbation is not proof of real acquisition robustness. Within-patient permutation is a post-hoc visit-specificity diagnostic, not a biological-change test. Report empty compartments and changes in both directions rather than hiding them in an aggregate.

### Scoped next-observed-visit forecast contract

This remains the valid contract for the closed P1/P2 forecasting comparisons. It is not the currently authorized R36 observed-change experiment. The scoped RANO-free task predicts future `lesion-state-v1` at the next observed visit from information available at the source-visit cutoff. Primary metrics are log-volume MAE and delta-log-volume MAE, Spearman correlation, and improvement over volume persistence. Report pooled and patient-uniform estimates, paired patient-cluster intervals, and per-compartment results. Current-volume ridge is the learned scalar comparator. Compare persistence whenever it is appropriate to the outcome and horizon.

The input must not contain target-visit images, masks, volumes, or ending gaps. Forecast rows must be consecutive eligible manifest rows, with no cross-visit bridges. Training labels may be future targets; they are not inference inputs. Gap metadata remains auxiliary, not verified prospective horizon information. Developmental transfer fits without SAILOR outcomes and evaluates eligible SAILOR rows unchanged; it is not a pristine external test. SAILOR ONCO is primary and CL is a sensitivity analysis.

The earlier learned lesion forecast and adaptation claims are withdrawn under A40: target-visit images/masks and ending gaps entered inputs, and the old cache bridged missing-mask visits, yielding 123 rather than 117 legitimate unseen pairs. Those results are not evidence for past-only forecasting. A41's clean gap and velocity head reruns are separate latent-representation diagnostics; retrospective target-gap conditioning does not make them verified prospective-horizon forecasts.

<a id="historical-rano"></a>
## Historical frozen RANO protocol

The frozen downstream RANO classification protocol is version 1, dated 2026-09-11. It remains historical evidence distinct from active RANO-free lesion-state and R36 work. Its four clean classes are {PD, SD, PR, CR}; primary metric is macro-F1, with accuracy and per-class recall. Predictions are out-of-fold or single-net transfer; uncertainty is a 10,000-resample patient-cluster bootstrap, and comparisons use paired patient resamples. The fixed five folds over the 26 encoder-unseen patients use seed 2026. Never redraw folds or reshuffle per configuration.

`encoder_train` comprises 65 patients; `encoder_unseen` comprises 26. The 13-patient `dev` subset was used for original encoder checkpoint selection and is not selection-clean. The historical `final` 13 were repeatedly scored by downstream probes and are development evidence, not an untouched final holdout. The final13 must not be reused to select another representation, readout, threshold, or harmonization method. A new external or hidden cohort is required for an independent headline.

The legacy all-91 CV is not encoder-unseen CV: K3-16 established leakage of encoder-training patients into probe folds. Use the locked unseen protocol instead. Historical single-split gains are not robust evidence. A8, A24, and D28 correct earlier mixed-space and patient-count headlines; retain those corrections beside the historical claims rather than silently rewriting history.

The first-prefix fixes require current-schema caches and explicit task cutoffs. Older caches labeled `lumiere_legacy_cache` must be regenerated before current-schema claims. Task manifests use immutable row IDs and explicit assessment, forecast, anatomy, and surprise cutoffs. The corrected schema distinguishes clean `response_labels` and `response_valid` from operative/missing ratings and keeps separate treatment and operative-event fields. The 2026-09-21 P0 readout policy fixes a class-weighted MLP with hidden width 256, seed 42, and no feature standardization. The stored three-seed by three-width grid is a sensitivity report, not a fresh selection pool. Never infer that a legacy cache or checkpoint has current-schema semantics merely because it loads. Preserve cache provenance and validate row identity before scoring.

Historical RANO findings do not define the RANO-free target, forecast task, or general representation objective. SAILOR code mappings remain empirical, not independently codebook-verified clinical labels.

## R35 observed-state and change comparison

<a id="r35"></a>
### R35 protocol and status

R35, in sibling worktree `world-model-representations` at revision `2f6e1f5a65be4ccbceac6de706b537de8b695faa`, completed a corrected comparison of per-scan spatial state, observed pair change, and nuisance. Its source record is [the R35 comparison report](../../../world-model-representations/info/r35_representation_comparison.md). The R35 question is whether MRI-only spatial state and explicit observed-pair encoders preserve automated anatomical state and observed compartment change while avoiding false change under declared perturbations. It does not establish future prediction or biological change.

The reference was the clean survival-free BRAINIAC backbone. The candidate was a spatial encoder with dense three-compartment and native log-volume supervision; pair, later-scan-only, and frozen-state arms controlled for earlier-scan and adaptation contributions. The comparison kept the 65 training and 26 encoder-unseen patient identities, used seed 2026 for the internal training/validation partition, and trained state and pair arms for 20 epochs. Pair evaluation used original consecutive eligible rows. Masks did not enter learned MRI inference.

The reference and learned candidate differ in pretraining history and dense-mask supervision. Their comparison does not isolate architecture causally. Frozen readouts fitted by nested patient CV on the 26 unseen patients are a separate protocol from source52 trained direct heads. SAILOR transfer retains both source26 and source52 ridge budgets. Budgets are compared like for like, not selected after inspection. Completed measurements and limitations belong in [R35 evidence](EXPERIMENTS.md#r35).

## R36 locked Phase 1

<a id="r36-phase1"></a>
### Question and status

R36 tests whether change-balanced supervision improves frozen local access to signed occupancy change without sacrificing state assessment or increasing acquisition sensitivity. The primary contrast is C versus B under matched frozen source52 local linear probes. A is the continued state-only control; original R35 is the initialization reference; D tests the nuisance tradeoff. Observed-change utility does not establish future predictability.

Phase 1 is authorized but incomplete. At the 2026-10-11 18:30:51 UTC execution snapshot, seed43 source-only probes are RUNNING and seed44 is unlaunched. A fresh reviewer completed post-implementation review, not pre-training approval or final result acceptance. Nuisance-report identity and negative tradeoff classification remain open reporting findings. No comparative query-quality verdict exists. Phase 2 is unauthorized. [TODO](TODO.md) owns the remaining actions, and [R36 evidence](EXPERIMENTS.md#r36) owns the execution receipt.

The execution record is [R36 results](../../../world-model-r36/info/r36_delta_representation_results.md), from sibling worktree `world-model-r36` at revision `f4b96c00dd5d79c59ba02d186e024ff747f690ad`. The handoff and its execution addendum are in [`r36_delta_representation_handoff.md`](../r36_delta_representation_handoff.md). Do not treat early check-ins in that record as current status; use its latest dated status above and retain incomplete/unrun work explicitly.

### Locked cohort, identity, and target recipe

The immutable private contract fixes the four arms, three seeds, 20 final-graded epochs, coefficients, source52 readout budget, source-only normalizers, paired patient bootstrap, decision margins, and resource ceilings before comparative evaluation. It preserves 599 visits and 496 usable consecutive rows; the encoder-unseen assessment remains 149 visits from 26 patients and 117 pairs from 23 patients. There are no pair bridges or QA exclusions. Native labels are finite and declared; all 599 native compartment volumes match the inventory.

Artifacts bind checkpoint, source, contract, tensor, visit, and pair identities. The metadata canonicalization repair retains both prepared-metadata hashes and requires exact equality of every other field, including visits, rows, targets, geometry, and patient lists. It does not relax tensor-byte identity. Keep private inventories, raw data, and participant-level files under ignored local outputs.

Changed voxels are binary compartment occupancy disagreements on the prepared grid, not MRI intensity differences. The primary analysis keeps all original eligible rows, empty targets, and the suspect extracranial label. Any filtered sensitivity set must be fixed and model-blind. [R36 target QA](EXPERIMENTS.md#r36) records the occupancy statistics and reviewed overlays. Grid agreement and visual plausibility do not establish anatomical correspondence, segmentation accuracy, or repeatability.

### Initialization identity

The seed-specific R35 state checkpoints are content-bound before loading:

| Seed | Checkpoint SHA256 |
|---|---|
| 42 | `65a4c4db2ca1b44ebf29fa12659fbd09dea6ede606ce1d3279535144933a001a` |
| 43 | `395e75b47b8d4d65f5993b36af41ca8ce8c4823f303baca34f9706315ba4fd61` |
| 44 | `67747e19181fbd12758ab307ceda573143b355aa6e3420aea63f7b59ad993b6b` |

The contract is `r36-signed-feature-difference-v1`. The immutable remote model/data/loss snapshot is `1daf27d`. Later orchestration and documentation commits do not change that snapshot. Its archive SHA256 is `88bccf7a2a5bf8be21f39ec820a1bf52ccaf5a5a985b8de8f4a04ac79bb72de2`.

### Architecture and training

The shared encoder is the unchanged R35 `SpatialState` architecture. Four MRI sequences and availability channels feed the encoder. The model retains maps at 96, 48, 24, and 12 cubed, ending at `[B,128,12,12,12]`. The new change decoder receives only signed feature differences, not concatenated states, images, IDs, dates, or masks. Per-scan state decoder and native-volume head remain controls. A nonlinear change decoder is not guaranteed to output zero for identity or to be antisymmetric under reversal; test and report both behaviors.

Initialize every arm from the seed-matched R35 state checkpoint. Run seeds 42, 43, and 44 for exactly 20 epochs and grade the final epoch. Every arm processes the same ordered training pairs and computes both per-scan state branches. Use batch size 1, AdamW learning rate `3e-4`, weight decay `1e-4`, and gradient norm cap 1.0. Start a new optimizer for each arm; do not resume from the optimizer-stripped R35 checkpoint.

The four arms are:

| Arm | Training objective |
|---|---|
| A | Continued state-only training on both visits. Change heads are initialized but unused and are not graded. |
| B | A plus whole-grid signed-change and volume-delta losses. |
| C | A plus change-balanced signed-change and the same volume-delta loss as B. |
| D | C plus nuisance-only zero-change constraints. |

`L_state` averages the source and target R35 state losses, each combining mask BCE, mask Dice, and L1 on source-training-normalized native log-volumes. `L_volume_delta` is L1 on native target-minus-source log-volume using the source-training change scale and is identical in B, C, and D. B uses whole-grid mean absolute signed occupancy error. C and D partition each compartment into changed and unchanged voxels, average error within each nonempty partition, then average available partition means equally. If no changed voxels exist, use only the unchanged mean. Average over compartments and then pairs; retain the natural pair distribution.

The locked coefficients are `L_B = L_state + L_volume_delta + 0.5 * L_grid_change`, `L_C = L_state + L_volume_delta + 0.5 * L_balanced_change`, and `L_D = L_C + 0.25 * L_nuisance`. These are protocol coefficients, not clinical constants.

For D, synthetic `(x, T(x))` pairs come only from training visits and have zero occupancy and volume-change targets. Average zero-change readout losses with an L1 consistency term between predicted state volumes. Cycle source and target visits under a deterministic perturbation schedule. Use identity, foreground Gaussian noise with sigma 0.1, affine intensity scale 1.5 and offset 0.3, and smooth bias `exp(linspace(-0.4,0.4))`. Keep R35 per-sequence standardization. Hold out its half-resolution downsample-and-upsample transform as a stress test. All arms perform the same additional perturbed-scan forwards; only D attaches their training loss. Record real-pair updates, nuisance exposures, successful updates, and AMP skips separately. AMP backoff precedes clipping; persistent overflow and full-precision nonfinite gradients are fatal.

### Frozen probe, outcomes, and decision margins

Freeze each encoder. For each feature-pyramid level, the primary probe uses a shared voxelwise `Conv3d(width,3,1)` on signed feature differences, fixed resizing to 96 cubed, and averaging of level outputs. It receives no absolute states or target images. Train on the same 52 source patients for 20 epochs with balanced signed-change loss, initialization seed 2026, AdamW learning rate `1e-3`, and weight decay `1e-4`; grade the final epoch and retain actual probe weights.

The primary endpoint is patient-uniform balanced signed occupancy-change MAE. Use the same changed/unchanged partition rule, average pairs within patients, then patients, and publish each partition separately. C versus B is the primary contrast. A versus C and C versus D are companion contrasts. The frozen original R35 encoder is an initialization control. Grouped source52 ridge selects alpha with three patient-grouped folds over `[0.1, 1, 10, 100, 1000, 10000]`. Each fit saves its training-only scaler and coefficients.

Required companion evaluations include direct native log-volume state MAE on all measured visits, pooled and patient-uniform and per compartment; signed volume-delta decoding with the same grouped source-52 ridge protocol and source-only alpha grid; direct change-decoder outputs separately from independent state differences; changed-voxel precision-recall and signed appearance/disappearance errors with support counts; zero-change, independent-state-difference, and equally budgeted target-only local linear probes; identity and nuisance perturbations with state drift and excess false change; empty/nonempty, both-empty, appearance, disappearance, and source-training-defined change-size strata; and local SAILOR transfer on all 269 visits and 242 pairs without query fitting. The target-only probe uses target features and identical targets and capacity; it is not evidence of source contribution just because a gradient exists.

Use three-seed mean patient errors, not an ensemble. Bootstrap paired patients with 10,000 draws and seed 2026. Patients, not seeds or voxels, are the resampling units. Use family-three adjusted intervals for the primary representation comparisons and report raw intervals too. Larger exploratory tables are descriptive. SAILOR remains local and developmental.

The following criteria are locked in the Phase 1 contract:

1. C improves frozen-probe balanced change MAE over B. Its family-adjusted paired patient difference interval lies below zero.
2. C preserves direct state MAE relative to A. The upper confidence bound is below +0.05 log-volume MAE.
3. D reduces excess false change relative to C. It does not exceed +0.05 balanced change MAE or the same state margin.
4. C or D outperforms matched target-only and independent-state controls before a claim of added observed temporal information.

The +0.05 margins are research decisions, not clinical thresholds. Family-three adjustment uses a two-sided error level of `0.05 / 3`. Wide intervals or missing evidence are INCONCLUSIVE. Demonstrated adverse tradeoffs are NOT VERIFIED. A nonlinear-head gain alone is a decoder benefit, not verified encoder improvement. Report generation must verify expected nuisance artifact identity and complete patient-condition coverage before a D-versus-C verdict.

### Resource, privacy, and review guardrails

The total Phase 1 ceiling is 24 GPU-hours and the per-session ceiling is nine hours, subject to a fresh quota check. Training inputs are existing private LUMIERE data only. No SAILOR upload, new acquisition, public push, deployment, or data-use agreement is authorized. Incomplete arms remain incomplete; do not grade partial arms, retry a failed arm, change the recipe, duplicate a launch, or exceed the ceiling. Preserve incomplete outputs and stop on identity, geometry, target-QA, or nonfinite-gradient failures.

Independent design and result review are required. Parent self-review is not independent review. Computational completion does not imply scientific success. Verify byte identity and retain weights, normalizers, predictions, configs, split manifests, and provenance. Keep private inventories and raw participant data out of committed reports.

## Conditional Phase 2: future dynamics

<a id="r36-phase2"></a>
Phase 2 is a proposal only and is not authorized. It requires a separate operator decision after Phase 1 reporting. It must not be treated as an automatic continuation or a selected experiment.

If authorized, compare the source-validated objective family with matched state-only controls under identical dynamics capacity. Resolve selection using the 13 internal validation patients, not SAILOR or the 26 development patients. Freeze the compared encoders. Use a causal spatial-latent model: project each observed final `[128,12,12,12]` map to 16 channels with a learned 1x1 convolution; assimilate up to the four most recent observed visits with a hidden-width-16 ConvGRU and explicit history mask; predict a 128-channel residual map added to the current frozen feature map.

The future teacher is the frozen encoder's actual next feature map. Normalize channels using source-training visits only. The latent loss balances changed and unchanged target regions; derive the region mask from any 96-cubed compartment occupancy change and max-pool with kernel and stride 8 to 12 cubed. Masks are training targets/loss weights only, never causal inputs. Keep unbalanced latent prediction as an objective control if balancing is the primary hypothesis.

Decode future native-volume estimates through the same frozen global-pooling volume head and saved normalizer. A separate uncertainty head outputs three native-log-volume standard deviations with softplus plus a proposed floor of 0.05. The proposed training objective is Gaussian native-volume NLL plus `0.1 * L_latent_delta`; lock latent coefficients and floor in a Phase 2 contract before work. Use batch size 1, AdamW learning rate `3e-4`, weight decay `1e-4`, and 20 epochs. Select by internal-validation patient-uniform native-volume NLL only.

Never provide the inference model the actual next-visit gap, future treatment, future mask, future volume, or future embedding. Future teacher observations are training targets only. If exact horizons remain unavailable, retain the next-observed-visit target and model the unknown future gap explicitly. Treatment inputs are allowed only when documented available at the prediction cutoff; missing action does not mean no treatment.

Controls must include MRI-only persistence of representation and decoded volumes; separately labeled mask-assisted persistence of measured volumes; a source-fitted empirical change distribution; the identical probabilistic head with current scan only; the identical history model on state-only encoder; and cutoff-valid physical-history baseline, labeled mask-assisted where appropriate. Evaluate patient-uniform forecast MAE, log score or CRPS, coverage, sharpness, and incremental history information against real probabilistic controls. Multi-step rollout is a separate later capability. Forecast gains do not establish causal treatment effects or validate counterfactual plans.

## Proposed action-conditioned synthetic work

Action-conditioned synthetic representation work remains proposed, not selected or authorized. The research brief at [`action_conditioned_representation_brief.md`](../action_conditioned_representation_brief.md) describes a possible hypothesis. It does not replace the current objective, establish biological action labels, or authorize implementation. Any future proposal must specify operator approval, cutoff-valid actions, matched controls, nuisance checks, and falsifiers before execution. Do not imply that an action-conditioned experiment is the chosen next method.

## Provenance, reporting, and limitations

Every cache and trained artifact needs source and model identity, configuration/contract hash, git revision, creation date, patient and row identities, geometry and normalizer provenance, and byte-level content binding where available. Loaders fail closed on incompatible schema, time semantics, geometry, or checkpoint identity. Preserve historical artifacts without relabeling them as current-schema or corrected-geometry evidence.

Keep pooled and patient-uniform estimands distinct. Bootstrap patients, not visits, seeds, or voxels. Report paired intervals, all prespecified arms, empty compartments, negative and inconclusive findings, and target-source differences. SAILOR remains developmental transfer. RANO remains a historical separate task. No grids, overlays, proxy metrics, or training losses alone establish template equivalence, biological change, clinical validity, or external generalization.

Key method sources: [`research_guardrails.md`](../research_guardrails.md), [`anatomy_harmonization.md`](../anatomy_harmonization.md), [`eval_protocol.md`](../eval_protocol.md), [`preprocessing.md`](../preprocessing.md), [`methodology_validity.md`](../methodology_validity.md), [`p0_audit.md`](../p0_audit.md), [`p2_execution.md`](../p2_execution.md), [`p2_lesion_architecture.md`](../p2_lesion_architecture.md), and [`clean_rescues.md`](../clean_rescues.md). Historical corrections and project decision records remain append-only in the canonical files; this draft does not reproduce their result tables.