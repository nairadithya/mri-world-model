# Methods

This reference defines the methods, evaluation rules, and historical IDs used in the project. The October 13, 2026, update incorporates the completed R36 reporting rules and review scope. It does not change the locked experiment, refresh live job status, or authorize new research.

Use [Notes and insights](NOTES_AND_INSIGHTS.md) for the reasoning, [Experiments](EXPERIMENTS.md) for results, and [TODO](TODO.md) for remaining steps. Detailed frozen experiment contracts remain binding. The [archive](archive/2026-10-12/README.md) preserves the original records.

<a id="terms"></a>

## Terms used in this reference

The following terms describe models, measurements, and evaluation:

- An **encoder** turns an MRI into numerical features. Those features are its **representation**, also called latent features.
- A **decoder** turns features into an output, such as a mask or volume. A **head** is an output component of a model.
- A **readout** extracts a measurement from features. A **probe** is a readout used to test what information the features retain.
- A **frozen encoder** has fixed weights during readout training. A better nonlinear readout does not by itself establish a better encoder.
- **Persistence** predicts that the next measurement equals the current measurement. The change baseline predicts zero change.
- An experiment **arm** is one configuration in a comparison. A **seed** fixes a run's random choices, including initialization.
- **Source52** means the 52 source-training patients. **Source26** means a separate readout budget using 26 patients, where the stated protocol permits it.
- **MAE**, mean absolute error, is the average size of an error without its sign. Lower values are better.
- **Patient-uniform** scoring averages within each patient and then across patients. Each patient has equal weight. **Pooled** scoring combines observations and gives more weight to patients with more observations.
- **Cross-validation**, or CV, rotates which patient groups are used for fitting and evaluation. Visits from one patient stay in the same group.
- A **patient bootstrap** resamples patients to estimate uncertainty. A paired comparison uses the same resampled patients for both methods.
- A **confidence interval** describes uncertainty in an estimate. In this project, an error-difference interval that crosses zero does not establish which method is better.
- **Nuisance** is variation that is not the target disease signal, such as an acquisition-related intensity change.
- **Signed occupancy change** records whether a voxel enters or leaves a lesion compartment. Appearance and disappearance have opposite signs. It is not an MRI intensity difference.
- **Native volume** is measured on the mask's original physical grid. A resized model tensor is not the ruler for that measurement.
- **`log1p(v)`** means `log(1 + v)`. A log-volume difference is the later log-volume minus the earlier log-volume, not a change in raw cubic millimeters.
- A **normalizer** rescales values using saved statistics. Source-only normalizers use no evaluation-patient outcomes.
- A **checkpoint** stores model weights. An **artifact** is a saved output, such as a checkpoint, prediction file, or report. A **cache** stores computed features for reuse.
- An artifact's **provenance** records where it came from. SHA256 hashes identify file bytes. A protocol or **contract** fixes how the experiment is run and checked.
- **RANO** is a clinical response-label system. It is separate from the current automated lesion-volume task.
- **Macro-F1** averages F1 across classes with equal class weight. F1 combines precision and recall. It is not the same metric as balanced accuracy.
- **AUC** measures discrimination across thresholds. An AUC near 0.5 does not support useful ranking in these binary prediction tasks.
- **R²** compares squared error with a mean-prediction baseline. Negative R² means worse performance than that baseline.
- **Spearman correlation** measures whether predicted and observed values have similar rankings.
- **JEPA** trains a model to predict another observation's latent features rather than its pixels. The historical project uses BRAINIAC as its MRI backbone.
- An **EMA target** is a target network whose weights follow an exponential moving average of online weights.
- **AMP**, automatic mixed precision, uses lower-precision arithmetic for part of training. An AMP backoff reduces its loss scale after overflow.

The remaining sections give exact definitions and recipes. Proposed work is labeled separately from completed and authorized work.

<a id="research-objective"></a>

## Research objective and scope

The objective is to retain useful information about anatomy, disease, and change across MRI visits while separating acquisition effects. Per-scan state, change between two observed scans, and prediction of future observations are separate evaluation questions. Lesion-focused methods, JEPA, forecast heads, and whole-brain methods are possible approaches, not the objective.

The RANO-free lesion-state task uses automated anatomical measurements rather than clinical RANO labels. It is one experiment, not a universal test of representation quality. Failure on that task does not establish that useful representations are impossible.

A lower training loss, a favorable readout, or one lesion forecast does not establish the broader objective either. A claim that history helps requires a comparison against the current scan alone. It also requires checks for acquisition sensitivity and shortcuts unrelated to the intended capability.

The operator approves changes to scientific targets, method-family priorities, and the use of a proxy as evidence of success. A substantial research implementation requires a decision brief. The brief identifies the candidate, a materially different approach, a simple baseline, assumptions, matched comparisons, available inputs, costs, independent checks, falsifiers, and unknowns. A falsifier is a finding that would disprove the hypothesis. Reports separate measurements from interpretation and unresolved questions.

## Data, geometry, labels, and timing contracts

### Cohorts and patient identity

LUMIERE is the development cohort. Its original split has 65 encoder-training patients, 13 development patients, and 13 historical final patients. The latter two groups contain 26 patients not used to train the original champion encoder. Their fixed five-fold partition uses seed 2026.

R35 and R36 divide the 65 source patients into 52 training patients and 13 validation patients. They assess performance on the same 26 encoder-unseen patients. Both that cohort and the historical final13 have been reused. Neither is a new independent test set.

SAILOR tests transfer to a different cohort during development. It is controlled-access data, not an untouched external-validation cohort. The protocol requires local use, no cloud upload, and no model selection from SAILOR outcomes. Its transfer results do not count as independent confirmation.

All splits keep patients separate, and all joins preserve visit and pair identities. A pair cannot skip a missing or ineligible intermediate visit and call a later visit the new "next" visit. Caches and predictions retain fixed row IDs, patient lists, both visit IDs, and the eligibility rule.

### Geometry and preprocessing

The [preprocessing contract](documentation_sources/versions/e29b2b676b2a9a2db5406419ce3e1ee0868b4548c26460bcca0feacaee462cdf.md) defines BRAINIAC inputs. Each MRI sequence receives N4 correction for smooth intensity bias and resampling to 1 mm isotropic voxels. Rigid registration to MNI runs only when a template is explicitly supplied. Skull stripping removes nonbrain tissue. The volume is then resized to 96³, meaning 96 voxels on each axis, and normalized using nonzero-voxel intensities.

The final resize follows the physical 1 mm resampling. The resized tensor therefore cannot be used as a physical 1 mm ruler. Training inputs are prepared offline. Missing template registration produces a warning. It does not imply MNI alignment.

Physical lesion volumes use native masks and their affines before resizing. An affine maps voxel coordinates to physical coordinates. Mask alignment uses verified MRI grids from before the final resize, followed by nearest-neighbor resizing.

An MRI tensor with an identity affine no longer supplies the original physical mapping by itself. It requires geometry provenance tied to its content, an exported geometry manifest, or a retained intermediate that reconstructs the final tensor. Missing or stale geometry and inconsistent MRI-sequence grids are rejected.

A valid affine and matching grid are necessary for alignment. They do not prove that templates are equivalent, anatomical locations correspond, registration is accurate, or masks are correct.

LUMIERE lesion measurements use recovered DeepBraTumIA atlas masks. The empirically verified mapping is `{1: enhancing, 2: necrotic/nonenhancing, 3: edema}`.

SAILOR uses the complete ONCO mask triplet as its primary target source. CL is an alternative mask source used for the required target-sensitivity comparison. The two sources are not interchangeable ground truth.

SAILOR `-icor` files may contain nonfinite background values, called NaNs. Inputs require finite-value checks, and usable base variants are preferred. Reports separate cohorts and target sources because acquisition and segmentation differ.

The [preprocessing procedure](documentation_sources/versions/e29b2b676b2a9a2db5406419ce3e1ee0868b4548c26460bcca0feacaee462cdf.md) and [historical SAILOR reprocess recipe](documentation_sources/versions/c7aeeea2bff31ee76a1190a69a05ca48ec78fc476e89d3afc345d546074c3f0d.md) remain separate operating guides. The SAILOR recipe prepares only sessions with a recorded link between raw and derivative files. It translates MRI-sequence names, uses a new config, and skips completed outputs on resume.

Concurrent preprocessing jobs can write to the same output paths. Evaluation requires checks on skull-stripping fallbacks, finite outputs, metadata sidecar files, and pair counts. A23 replaces the recipe's historical A21 recovery claim. Keeping the recipe does not restore that claim.

### Chronology and projection semantics

R34 found that swapping historical tokens with equal gaps left the legacy temporal state unchanged. The optional elapsed-time coordinates changed that behavior while retaining checks that each prefix excludes future inputs.

`model.temporal.time_basis` defaults to `previous_gap`, which uses the gap from the previous visit. `elapsed` uses cumulative time coordinates. Both modes have the same saved weight keys, so loaders check the time meaning even when tensor shapes match.

Equal timestamps remain indistinguishable. The option does not recover exact visit dates. Any claim that elapsed mode improves results requires new training and a matched comparison.

The default JEPA loss updates the backbone's LoRA parameters and the predictor. It does not update the online projector. LoRA is the small set of trainable adaptation parameters added to the backbone. R34 documented this behavior without adding a new projector objective. A projector can load and run without having been trained by that loss. See [R34 repairs](documentation_sources/versions/7bc218e8ef5fd2fdb22bc0ed7f5bab72c4b1d099f1f7f2c3145f6c3bca6f3472.md).

### Labels and outcomes

The label contract separates response labels, treatment, surgery-related events, and label validity. The four clean RANO classes are PD for progressive disease, SD for stable disease, PR for partial response, and CR for complete response. Surgery-related, ambiguous, and missing ratings are not silently converted to SD.

Unknown or surgery-ambiguous legacy action values use -1. A checkpoint that loads successfully is not necessarily trained under the corrected label contract. Inputs for prospective prediction exclude eventual overall survival. Survival can be a separate censored target, where incomplete follow-up is handled explicitly.

<a id="lesion-state-v1"></a>

The `lesion-state-v1` target contains three physical volumes in mm³. The compartments are `necrotic_non_enhancing`, `enhancing`, and `edema_flair`. The primary forecast targets are the later `log1p` volume and its difference from the earlier `log1p` volume. Any later categories of growth, stability, or shrinkage describe imaging change. They are not clinical progression or response labels.

The fixed eligibility rule pairs consecutive curated visits when both have all three measurements and the later visit follows the earlier one. The audited set contains 496 LUMIERE pairs and 242 SAILOR pairs. Median usable gaps are 84 days in LUMIERE and 77 days in SAILOR.

The primary forecast targets the next observed visit because the timing records do not support exact day-level horizons. `gap_days` is an auxiliary variable or a sensitivity check. It does not establish a fixed forecast horizon, a precise rate of change, or a calibrated time-to-event estimate.

### Finite, missing, and zero-volume behavior

Image-based loss pairs require usable pixels at both visits. Visits without images are dropped. Inputs, losses, metrics, and saved artifacts require finite-value checks. A present but empty scan file is not a valid observation. Pandas NaN values also require explicit missing-value checks, because a NaN can otherwise pass a Boolean test.

A missing measurement is not a measured zero. A genuinely empty compartment has volume zero and `log1p(0) = 0`. The data retain absent compartments and pairs where both masks are empty, with sample counts reported. Those cases cannot be discarded to improve scores.

A partitioned change loss averages only the nonempty partitions. If no voxels change, it uses the unchanged-region error alone. It does not divide by zero or create a nonzero target. For precision-recall metrics, a case with no positive targets is recorded as empty. It is not assigned a perfect area under the precision-recall curve, or AUPRC.

## Evaluation protocols

### Per-scan state and observed change

Per-scan assessment uses one visit. Observed-change assessment uses two visits that have already been observed. R35 evaluates observed pairs, not future forecasts. Its later-scan inputs are therefore not valid inputs to a forecasting evaluation.

The R35 and R36 development assessment retains 26 encoder-unseen LUMIERE patients, 149 measured visits, and 117 consecutive usable pairs from 23 patients. Each fixed protocol defines the internal split of the 65 encoder-training patients. Evaluation outcomes do not select epochs or architecture.

Reports separate pooled and patient-uniform errors, per-compartment results, and paired patient-level uncertainty. R35 averages patient-level errors across the three seeds rather than averaging their predictions as an ensemble.

State assessment includes a mean fitted on source-training data or a scalar-volume-supervised control. Observed-change controls include zero change, subtraction of separate per-scan state estimates, a later-scan-only model, and a readout from fixed features where specified. A dedicated paired head needs an advantage over independent state subtraction, not only over zero change. Dense-mask supervision is part of training. It is not independent confirmation.

Nuisance tests use an unchanged scan pair, intensity changes, noise, smooth intensity bias, and declared resolution changes. Reports separate false change on the unchanged pair, changes in state estimates, and additional false change under each perturbation. Synthetic tests do not prove robustness to real acquisition differences.

Shuffling visits within a patient is a later diagnostic of visit specificity. It is not a test of biological change. Reports include empty compartments, appearance, and disappearance rather than only an aggregate error.

### Scoped next-observed-visit forecast contract

This protocol defines the closed P1 and P2 forecasting comparisons. It is separate from the authorized R36 observed-change experiment. The task predicts `lesion-state-v1` at the next observed visit using only information available at the earlier visit.

Primary metrics are log-volume MAE, log-volume-change MAE, Spearman correlation, and improvement over volume persistence. Reports include pooled and patient-uniform estimates, paired patient-bootstrap intervals, and per-compartment results. Current-volume ridge is the learned scalar comparator. Persistence remains a required comparator where it fits the outcome and horizon.

Forecast inputs exclude later-visit images, masks, volumes, and gaps ending at that visit. Forecast rows pair consecutive eligible visits without skipping an intermediate visit. Later observations can be training labels, but not inference inputs.

Gap metadata remains auxiliary. It does not establish a horizon known at prediction time. Transfer fitting excludes SAILOR outcomes and evaluates its eligible rows unchanged. This is development transfer, not an untouched external test. ONCO is the primary SAILOR mask source, and CL tests sensitivity to the target source.

A40 withdraws the earlier learned lesion forecasts and adaptation claims because later images, masks, and ending gaps entered the inputs. The old cache also skipped visits with missing masks, producing 123 rather than 117 legitimate unseen pairs. Those results are not evidence of past-only forecasting.

A41's clean gap and velocity reruns are separate latent-feature diagnostics. Using a target gap known retrospectively does not make them verified forecasts at a horizon known in advance.

<a id="historical-rano"></a>

## Historical frozen RANO protocol

The fixed RANO classification protocol is version 1, dated September 11, 2026. It is historical evidence, separate from the active RANO-free lesion task and R36. It uses PD, SD, PR, and CR, with macro-F1 as the primary metric. Reports also include accuracy and recall for each class.

Cross-validation predictions come from models not fitted on the patient's fold. Transfer predictions use a single network. Uncertainty uses 10,000 patient-bootstrap resamples, with paired resamples for comparisons. The five fixed folds over 26 encoder-unseen patients use seed 2026. Fold membership does not change across configurations.

`encoder_train` contains 65 patients, and `encoder_unseen` contains 26. The 13-patient `dev` subset helped select the original encoder checkpoint, so it is not independent of that selection.

The historical `final` 13 were repeatedly scored by downstream readouts. They are development evidence, not an untouched final test. They cannot select another representation, readout, threshold, or harmonization method. An independent-validation claim requires a new external or hidden cohort.

K3-16 found encoder-training patients in the legacy all-91 probe folds. That cross-validation is not encoder-unseen evaluation. The fixed unseen protocol addresses that overlap, but historical single-split gains remain weak evidence.

A8, A24, and D28 correct earlier claims based on different feature spaces and incorrect patient-win counts. The corrections remain beside the historical claims.

The corrected first-prefix rules require caches with the current schema and explicit limits on inputs available for each task. A schema defines the stored fields and their meaning. Caches labeled `lumiere_legacy_cache` require regeneration before a current-schema claim.

Task manifests retain fixed row IDs and separate input cutoffs for assessment, forecasting, anatomy, and surprise. The corrected fields `response_labels` and `response_valid` distinguish clean responses from surgery-related and missing ratings. Treatment and surgery-related events have separate fields.

The September 21, 2026, P0 readout policy fixes a class-weighted MLP with hidden width 256, seed 42, and no feature standardization. The saved grid of three seeds and three widths reports sensitivity. It is not a new set from which to choose a favorable model.

Successful loading does not prove that an older cache or checkpoint has current-schema meaning. Scoring requires recorded cache origins and validated row identity.

Historical RANO findings do not define the RANO-free target, forecast task, or general representation objective. SAILOR code mappings remain empirical, not independently codebook-verified clinical labels.

## R35 observed-state and change comparison

<a id="r35"></a>

### R35 protocol and status

R35, in sibling worktree `world-model-representations` at revision `2f6e1f5a65be4ccbceac6de706b537de8b695faa`, completed a corrected comparison of per-scan spatial state, observed pair change, and nuisance. Its source record is [the R35 comparison report](documentation_sources/versions/768fdfa0a9be4dc76e99fcef9ff971e26252d672e82b08267f5e9ea157ced0f3.md). The R35 question is whether MRI-only spatial state and explicit observed-pair encoders preserve automated anatomical state and observed compartment change while avoiding false change under declared perturbations. It does not establish future prediction or biological change.

The reference was the BRAINIAC backbone from the clean run without eventual survival as an input. The candidate spatial encoder received dense three-compartment mask supervision and native log-volume supervision.

Paired, later-scan-only, and fixed-state controls tested contributions from the earlier scan and encoder adaptation. The comparison retained the 65 training patients and 26 encoder-unseen patients. The internal training and validation split used seed 2026, with 52 fitting patients and 13 checkpoint-selection patients. State and pair arms trained for 20 epochs. R35 grades validation-selected best checkpoints, not R36's final-epoch-20 recipe.

Pair evaluation used the original consecutive eligible rows. Masks were training targets, not learned MRI inference inputs.

The reference and learned candidate differ in pretraining history and dense-mask supervision. Their comparison does not isolate architecture causally. Frozen readouts fitted by nested patient CV on the 26 unseen patients are a separate protocol from source52 trained direct heads. SAILOR transfer retains both source26 and source52 ridge budgets. Budgets are compared like for like, not selected after inspection. Completed measurements and limitations belong in [R35 evidence](EXPERIMENTS.md#r35).

R35 computation and scoped result review closed on October 13, 2026. The [result review](reviews/r35_result_review.md), [corrected audit](reviews/r35_artifact_audit.md), and [receipt](reviews/r35_closure_receipt.json) separate semantic approval from executable verification. The [saved evidence](reviews/evidence/r35/README.md) includes readout replay, exact nuisance membership, declared reference source/query provenance, summary prediction digests, and patient-level metric receipts. These checks do not supply original MRI-byte provenance, new dense behavioral or shuffle replay, or biological validation. The reference feature cache lacks an original-manifest stability hash; reference checkpoint equality remains metadata-level. Existing raw status fields remain historical.

## R36 locked Phase 1

<a id="r36-phase1"></a>

### Question and status

R36 tests whether giving changed regions more weight during training makes change information easier to extract from a fixed encoder. It also tests whether that training harms state estimates or increases sensitivity to acquisition effects.

The primary comparison is C versus B, using matched local linear readouts trained on the 52 source patients. A continues state-only training. Original R35 is the starting encoder. D tests the added nuisance constraint. Better observed-change estimates do not establish better future prediction.

Phase 1 computation and independently reviewed scoped reporting are complete as recorded on October 13, 2026. The [closure receipt](reviews/r36_closure_receipt.json) binds the completed result review and artifact audit. [Experiments](EXPERIMENTS.md#r36-completed) records the classifications. Reporting completion does not establish an encoder gain, biological validity, or future prediction. Phase 2 remains unauthorized.

### Historical execution snapshot

Phase 1 was authorized but incomplete at the October 11, 2026, 18:30:51 UTC snapshot. The seed-43 source-trained readouts were running. Seed 44 had not started. No comparison of model quality on evaluation patients existed.

The completed implementation review was neither design approval before training nor acceptance of scientific results. Checks on nuisance-report identity and classification of harmful tradeoffs remained open. Phase 2 was not authorized. These statements describe the archived snapshot, not current progress. [TODO](TODO.md) owns the remaining actions, and [R36 evidence](EXPERIMENTS.md#r36) owns the execution receipt.

The execution record is [R36 results](documentation_sources/versions/51d6523f0da4b3d8745af10b79318fd684aba1fd9b47cc7d321e30df79984551.md), from sibling worktree `world-model-r36` at revision `f4b96c00dd5d79c59ba02d186e024ff747f690ad`. The handoff and its execution addendum are in [`r36_delta_representation_handoff.md`](documentation_sources/versions/8fc1116f80667885774a5ce6a8fb9ec40f1e360d1f87867fefb81baf01118ab6.md). Its last branch-narrative receipt is 14:16 UTC. The later 18:30 UTC status above comes from the separately [recovered runtime snapshot](documentation_sources/supplementary/r36-runtime-snapshot.json). Keep these origins distinct and retain incomplete or unrun work explicitly.

The [complete R36 handoff](documentation_sources/versions/8fc1116f80667885774a5ce6a8fb9ec40f1e360d1f87867fefb81baf01118ab6.md#give-the-implementer-exact-verification-commands) preserves exact verifier commands and [unit receipts](documentation_sources/versions/8fc1116f80667885774a5ce6a8fb9ec40f1e360d1f87867fefb81baf01118ab6.md#deliver-the-handoff-receipts). Its initial PLAN ONLY and unavailable-delegation statements are dated history, superseded by its execution addendum and the later execution record. The conditional Phase 2 remains unauthorized. The [original execution checklist](documentation_sources/versions/5e5409445bd5d258f7b1bb090aaf316ff56241007f54f8173c1d180ba200bf3a.md) is retained as a dated operational snapshot, not an active launch checklist.

R35 already used volume changes and gave lesion mask differences extra weight. R36 tests a specific change in weighting: equal treatment of changed and unchanged regions. Its readouts test what change information the fixed encoder retains.

With fixed reference features, rewriting squared prediction error as a change plus the source state does not create a different objective. The fixed comparison excludes variance regularization at batch size one and losses that push all latent features apart indiscriminately.

### Locked cohort, identity, and target recipe

The private protocol fixes all choices before comparative evaluation. These include four arms, three seeds, 20 epochs with final-epoch scoring, loss coefficients, source52 readout budget, source-only normalizers, patient bootstrap, decision margins, and resource limits.

It retains 599 visits and 496 usable consecutive rows. The encoder-unseen assessment has 149 visits from 26 patients and 117 pairs from 23 patients. No pair skips an intermediate visit, and no quality-audit exclusion changes the primary set. Native labels are finite and declared. All 599 native compartment volumes match the inventory.

Saved artifacts identify the exact checkpoint, source, protocol, tensor, visit, and pair used to create them. The metadata-format repair retains both prepared-metadata hashes. Every other field must remain exactly equal, including visits, rows, targets, geometry, and patient lists. Tensor bytes must still match exactly. Private inventories, raw data, and participant-level files remain in ignored local outputs.

Changed voxels are binary compartment occupancy disagreements on the prepared grid, not MRI intensity differences. The primary analysis keeps all original eligible rows, empty targets, and the suspect extracranial label. Any filtered sensitivity set must be fixed and model-blind. [R36 target QA](EXPERIMENTS.md#r36) records the occupancy statistics and reviewed overlays. Grid agreement and visual plausibility do not establish anatomical correspondence, segmentation accuracy, or repeatability.

### Initialization identity

The seed-specific R35 state checkpoints are content-bound before loading:

| Seed | Checkpoint SHA256 |
|---|---|
| 42 | `65a4c4db2ca1b44ebf29fa12659fbd09dea6ede606ce1d3279535144933a001a` |
| 43 | `395e75b47b8d4d65f5993b36af41ca8ce8c4823f303baca34f9706315ba4fd61` |
| 44 | `67747e19181fbd12758ab307ceda573143b355aa6e3420aea63f7b59ad993b6b` |

The contract is `r36-signed-feature-difference-v1`. The fixed remote snapshot for the model, data, and loss code is `1daf27d`. Later orchestration and documentation commits do not change that snapshot. Its archive SHA256 is `88bccf7a2a5bf8be21f39ec820a1bf52ccaf5a5a985b8de8f4a04ac79bb72de2`.

### Architecture and training

The shared encoder is the unchanged R35 `SpatialState` architecture. It receives four MRI sequences plus channels that indicate which sequences are available. It keeps spatial feature maps at 96, 48, 24, and 12 voxels per axis. The final tensor is `[B,128,12,12,12]`, with batch size B and 128 feature channels.

The new change decoder receives only signed feature differences. It does not receive concatenated absolute states, images, IDs, dates, or masks. The per-scan state decoder and native-volume head remain controls.

The nonlinear decoder is not guaranteed to predict zero from an unchanged pair. Reversing the visit order is also not guaranteed to reverse the output sign. Both behaviors are tested and reported.

Each arm starts from the R35 state checkpoint for its seed. Seeds 42, 43, and 44 each run exactly 20 epochs, with final-epoch scoring. Every arm processes the same ordered training pairs and computes state outputs for both visits.

Training uses batch size 1, AdamW learning rate `3e-4`, weight decay `1e-4`, and gradient norm cap 1.0. Each arm starts a new optimizer because the R35 checkpoint has no saved optimizer state.

Each arm uses the same architecture, starting weights, training pairs, and training duration. Only the following loss terms differ:

| Arm | Training objective |
|---|---|
| A | Continued state-only training on both visits. Change heads are initialized but unused and are not graded. |
| B | A plus whole-grid signed-change and volume-delta losses. |
| C | A plus change-balanced signed-change and the same volume-delta loss as B. |
| D | C plus nuisance-only zero-change constraints. |

`L_state` averages the R35 state losses for the earlier and later scans. Each state loss combines mask binary cross-entropy, mask Dice loss, and L1 error on normalized native log-volumes. Binary cross-entropy scores voxel classification. Dice loss scores mask overlap, and L1 is absolute error. The normalizers use source-training data only.

`L_volume_delta` is L1 error on later-minus-earlier native log-volume, scaled using source-training changes. It is identical in B, C, and D.

B averages absolute signed occupancy error over the whole grid. C and D split each compartment into changed and unchanged voxels. They average within each nonempty group, then give the available group means equal weight. If no voxels change, only the unchanged mean is used. The loss then averages over compartments and pairs while retaining the natural pair distribution.

The fixed loss formulas are:

- `L_B = L_state + L_volume_delta + 0.5 * L_grid_change`.
- `L_C = L_state + L_volume_delta + 0.5 * L_balanced_change`.
- `L_D = L_C + 0.25 * L_nuisance`.

These coefficients define the experiment. They are not clinical constants.

D uses synthetic pairs `(x, T(x))`, where `T` applies a declared acquisition perturbation to a training scan. Their occupancy-change and volume-change targets are zero. The loss averages zero-change readout terms with L1 consistency between predicted state volumes.

A deterministic schedule alternates earlier and later training visits. Its perturbations are identity, foreground Gaussian noise with sigma 0.1, intensity scale 1.5 and offset 0.3, and smooth bias `exp(linspace(-0.4,0.4))`. R35's per-sequence standardization remains unchanged. The half-resolution downsample-and-upsample transform is held out as a stress test.

Every arm runs the same additional perturbed-scan forward passes. Only D uses their loss for training. Records separate real-pair updates, nuisance exposures, successful updates, and AMP skips. AMP backoff happens before gradient clipping. Persistent overflow or nonfinite full-precision gradients stop the run.

### Frozen probe, outcomes, and decision margins

The primary probe keeps the encoder fixed. At each spatial feature level, a shared voxelwise `Conv3d(width,3,1)` maps signed feature differences to three compartment outputs. Fixed resizing brings the outputs to 96 cubed, and the level outputs are averaged. The probe receives no absolute states or target images.

The probe trains on the same 52 source patients for 20 epochs with balanced signed-change loss. It uses initialization seed 2026, AdamW learning rate `1e-3`, and weight decay `1e-4`. Only the final epoch is scored, and the actual probe weights are retained.

The primary endpoint is balanced signed occupancy-change MAE with equal patient weight. The same rule gives equal weight to nonempty changed and unchanged groups. Pair errors are averaged within each patient, then across patients. Each group's error is also reported separately.

C versus B is the primary comparison. A versus C and C versus D are companion comparisons. The original fixed R35 encoder is the initialization control. Source52 ridge regression selects alpha with three patient-grouped folds over `[0.1, 1, 10, 100, 1000, 10000]`. Each fit saves its training-only scaler and coefficients.

The required companion evaluations are:

- Native log-volume state MAE on all measured visits, with pooled, patient-uniform, and per-compartment results.
- Signed volume-change decoding under the same source52 ridge protocol, patient-grouped validation, and source-only alpha grid. Alpha sets ridge regularization strength.
- Direct change-decoder estimates, reported separately from subtraction of independent state estimates.
- Changed-voxel precision and recall, appearance and disappearance errors, and sample counts.
- Zero-change, independent-state-difference, and later-scan-only local linear readouts with equal fitting budgets.
- Unchanged-pair and nuisance tests, with changes in state estimates and excess false change reported separately.
- Separate results for empty and nonempty compartments, both-empty pairs, appearance, disappearance, and change-size groups defined from source-training data.
- Local SAILOR transfer on all 269 visits and 242 pairs, without fitting to the evaluation patients.

The target-only readout receives later-scan features with the same targets and model capacity. A nonzero gradient through the earlier-scan branch does not establish that the earlier scan adds useful information.

The reported values average patient errors across the three seeds. They are not predictions from an ensemble. The paired patient bootstrap uses 10,000 draws and seed 2026. It resamples patients, not seeds or voxels.

The three primary representation comparisons use confidence intervals adjusted for testing three comparisons, called family-three intervals. Reports also include unadjusted intervals. Larger exploratory tables are descriptive. SAILOR remains local development transfer.

The following criteria are locked in the Phase 1 contract:

1. C improves frozen-probe balanced change MAE over B. Its family-adjusted paired patient difference interval lies below zero.
2. C preserves direct state MAE relative to A. The upper confidence bound is below +0.05 log-volume MAE.
3. D reduces excess false change relative to C. It does not exceed +0.05 balanced change MAE or the same state margin.
4. C or D outperforms matched target-only and independent-state controls before a claim of added observed temporal information.

The +0.05 margins are research decisions, not clinical thresholds. The family-three adjustment uses a two-sided error level of `0.05 / 3`.

Wide intervals or missing evidence produce INCONCLUSIVE results. Demonstrated harmful tradeoffs produce NOT VERIFIED results. A gain from a nonlinear head alone establishes a decoder benefit, not an encoder improvement. Before a D-versus-C verdict, report generation checks the identity of every required nuisance artifact and complete coverage of patients and conditions.

### Reporting validation and independent audit

The corrected nuisance validator checks complete status and exact arm and seed identity. It hashes actual final checkpoint and difference-probe bytes and binds the corresponding evaluation provenance. It also checks prepared metadata, the canonical tensor-manifest hash, and exact patient, first-visit, and condition tuples.

Every required tuple occurs exactly once. Errors and state drift are finite and nonnegative. Excess false change equals the perturbation error minus identity error, and condition summaries agree with their rows. Duplicate, missing, substituted, or mismatched evidence is rejected.

The D classifier returns NOT VERIFIED for demonstrated aggregate nuisance worsening or a demonstrated state or change margin violation. It returns VERIFIED only when nuisance benefit and both retention criteria hold. An interval overlapping a margin, or missing evidence, leaves the result INCONCLUSIVE. A positive error interval can establish worsening without establishing violation of the +0.05 allowance.

`tests/test_delta_reporting.py` covers successful, adverse, margin-violating, overlapping, and missing-evidence cases. It includes 18 identity, coverage, and metric tamper cases, plus replacement of actual checkpoint and probe bytes. `tests/test_delta_summary.py` checks aggregate shapes, privacy filtering, and presentation scope. The final reporting correction is `ac931da`.

The [result review](reviews/r36_phase1_result_review.md) approves the bounded interpretation and reporting patch. It did not independently recompute binary hashes, replay models, regenerate bootstraps, or execute tests. The separate [artifact audit](reviews/r36_phase1_artifact_audit.md) checked twelve checkpoints, thirty probes, fifteen assessment sets, 1,950 nuisance rows, and thirty ridge bundles. It replayed saved ridge coefficients and independently recomputed seed-mean paired-patient contrasts from raw rows. Floating-point reduction changed aggregates by about 1e-9 without changing conclusions.

The audit binds declared tensor provenance. It does not newly replay every dense occupancy prediction, precision-recall or direction measure, or original MRI tensor. Its [saved source](reviews/evidence/r36_artifact_audit.py), [aggregate results](reviews/evidence/r36_artifact_audit_results.json), and [console summary](reviews/evidence/r36_artifact_audit_console.json) retain the checked scope. No expert target adjudication or untouched external validation follows from those checks.

The failed initial native reviews were not accepted. The resumed reviewer and verifier completed under their original model and tool contracts. The verifier's full report was recovered from its exact child-recorded path after an output-routing error. The closure receipt binds its unchanged bytes. A completion summary alone was not accepted as the audit.

The raw aggregation CLI does not consume review artifacts and still labels reports review-pending. The separate closure receipt records completed review. Do not rewrite historical runtime receipts to hide that distinction.

### Reproduce the saved aggregation

The following commands require the existing private artifacts in the execution environment. They aggregate saved results, not new training or inference. They are reference commands, not work performed by this documentation update.

```bash
.venv/bin/python -B -m unittest discover -s tests -p 'test_delta*.py' -v
.venv/bin/python scripts/harness.py delta-representation report \
  --results outputs/r36-delta-objective/matched-final \
  --contract outputs/r36-delta-objective/qa/r36_contract.json \
  --out results/r36-report-reproduction.json
.venv/bin/python scripts/summarize_delta_representation.py \
  --results outputs/r36-delta-objective/matched-final \
  --report results/r36-report-reproduction.json \
  --out results/r36-public-metrics.json
```

[Aggregate metrics](r36_phase1_metrics.json) retain input and generator hashes. Original weights, predictions, tensors, and participant inventories remain under ignored outputs. The full [archived Phase 1 report](archive/2026-10-13/info/r36_phase1_report.md) preserves the original reproduction record.

### Resource, privacy, and review guardrails

The total Phase 1 ceiling is 24 GPU-hours and the per-session ceiling is nine hours, subject to a fresh quota check. Training inputs are existing private LUMIERE data only. No SAILOR upload, new acquisition, public push, deployment, or data-use agreement is authorized. Incomplete arms remain incomplete; do not grade partial arms, retry a failed arm, change the recipe, duplicate a launch, or exceed the ceiling. Preserve incomplete outputs and stop on identity, geometry, target-QA, or nonfinite-gradient failures.

Independent design and result review are required. Parent self-review is not independent review. Computational completion does not imply scientific success. Verify byte identity and retain weights, normalizers, predictions, configs, split manifests, and provenance. Keep private inventories and raw participant data out of committed reports.

## Conditional Phase 2: future dynamics

<a id="r36-phase2"></a>

Phase 2 is an unauthorized proposal for future prediction. It requires a separate operator decision after the Phase 1 report. It is not an automatic continuation or a selected next experiment.

The proposed comparison keeps dynamics-model capacity identical across the selected objective family and matched state-only controls. Selection uses the 13 internal validation patients, not SAILOR or the 26 development patients. The compared encoders remain fixed.

The proposed model uses only observed history. A learned 1x1 convolution reduces each observed `[128,12,12,12]` feature map to 16 channels. A hidden-width-16 ConvGRU combines up to the four most recent visits, with an explicit history mask. A ConvGRU is a recurrent model that operates on spatial maps. The model predicts a 128-channel residual map and adds it to the current fixed feature map.

The future teacher target is the fixed encoder's feature map from the actual next visit. Channel normalizers use source-training visits only. The latent loss balances changed and unchanged target regions.

The region mask marks any compartment occupancy change at 96 cubed. Max-pooling with kernel and stride 8 reduces it to 12 cubed. Masks act only as training targets or loss weights, never as inference inputs. If balancing is the primary hypothesis, the proposal retains unbalanced latent prediction as an objective control.

The proposal decodes future native volumes through the same fixed global-pooling volume head and saved normalizer. A separate uncertainty head outputs three native-log-volume standard deviations, using softplus with a proposed minimum of 0.05.

The proposed objective is Gaussian native-volume negative log-likelihood, or NLL, plus `0.1 * L_latent_delta`. NLL scores the probability assigned to an observed target. A separate Phase 2 protocol must fix the latent coefficients and uncertainty floor before work begins.

The proposed recipe uses batch size 1, AdamW learning rate `3e-4`, weight decay `1e-4`, and 20 epochs. Model selection uses only internal-validation patient-uniform native-volume NLL.

Never provide the inference model the actual next-visit gap, future treatment, future mask, future volume, or future embedding. Future teacher observations are training targets only. If exact horizons remain unavailable, retain the next-observed-visit target and model the unknown future gap explicitly. Treatment inputs are allowed only when documented available at the prediction cutoff; missing action does not mean no treatment.

The proposed controls include:

- MRI-only persistence of features and decoded volumes.
- Persistence of measured volumes, labeled separately as mask-assisted.
- An empirical change distribution fitted on source data.
- The identical probabilistic head using only the current scan.
- The identical history model using the state-only encoder.
- A physical-measurement history baseline using only information available at prediction time, labeled mask-assisted where appropriate.

The proposed evaluation includes patient-uniform forecast MAE and a probability score, either log score or CRPS. CRPS measures how well a predicted distribution matches an observed outcome. Coverage measures how often uncertainty intervals contain the outcome. Sharpness measures how narrow those intervals are. Matched probabilistic controls test whether history adds information.

Multi-step rollout is a separate later capability. Forecast gains do not establish causal treatment effects or validate plans for alternative treatment.

<a id="synthetic-actions"></a>

## Proposed action-conditioned synthetic work

Action-conditioned synthetic representation work remains proposed, not selected or authorized. The research brief at [`action_conditioned_representation_brief.md`](documentation_sources/versions/f3dad868d43b3aa2bf5f7913b3dfa2872cdf87a65fd9648b1eb9932e3f750855.md) describes a possible hypothesis. It does not replace the current objective, establish biological action labels, or authorize implementation. Any future proposal must specify operator approval, cutoff-valid actions, matched controls, nuisance checks, and falsifiers before execution. Do not imply that an action-conditioned experiment is the chosen next method.

## Operational references retained separately

The four documents consolidate research, not every operating procedure. Keep [AGENTS.md](../AGENTS.md), the [project README](../README.md), the [research index](README.md), the [acquisition toolkit](../scripts/README.md), and the [CLI verification skill](../.agents/skills/verify-world-model-cli/SKILL.md) discoverable. Its [feature map](../.agents/skills/verify-world-model-cli/features/README.md) retains command discovery, checkpoint inspection, and anatomy-manifest recipes. These operational documents can contain historical task descriptions; the scoped contracts and corrected evidence here govern interpretation. Frozen variants and their dispositions are listed in [Source coverage](SOURCE_COVERAGE.md).

## Provenance, reporting, and limitations

Each cache and trained artifact records its source, model, configuration or contract hash, git revision, and creation date. It also records patient and row identities, geometry and normalizer origins, and file-byte identity where available.

Loaders stop when schema, time meaning, geometry, or checkpoint identity is incompatible. Historical artifacts keep their original status. They cannot be relabeled as evidence from the current schema or corrected geometry.

Pooled and patient-uniform errors answer different averaging questions. Uncertainty estimates resample patients, not visits, seeds, or voxels. Reports include paired intervals, every arm specified in advance, empty compartments, negative and inconclusive findings, and differences between target sources.

SAILOR is development transfer. RANO is a separate historical task. Correct grids, plausible overlays, proxy scores, and falling training losses do not by themselves establish template equivalence, biological change, clinical validity, or generalization to new external data.

Key method sources: [`research_guardrails.md`](documentation_sources/versions/6e710be8c33b05d77bf8f13b415b64028fcc13dc03d71defb304e6a7631c78f4.md), [`anatomy_harmonization.md`](documentation_sources/versions/5fe9df82e326552d35667a3b92cdd7977504ade9f3ceebb9298a8f41d0dbdf19.md), [`eval_protocol.md`](documentation_sources/versions/7873b012053d04798d82c190a754e9708031bedff4ce9cf4d2abb168553f4386.md), [`preprocessing.md`](documentation_sources/versions/e29b2b676b2a9a2db5406419ce3e1ee0868b4548c26460bcca0feacaee462cdf.md), [`methodology_validity.md`](documentation_sources/versions/7bc218e8ef5fd2fdb22bc0ed7f5bab72c4b1d099f1f7f2c3145f6c3bca6f3472.md), [`p0_audit.md`](documentation_sources/versions/da674d536622373c6db154365a73c67d5fffcc9646a10713641169a0b5cb2130.md), [`p2_execution.md`](documentation_sources/versions/9ea08b90cf4971fb7b85d4b150ef1317994f42aa2fca2f1c9c0f5de5424e38e4.md), [`p2_lesion_architecture.md`](documentation_sources/versions/3e07b40bdd3849468d6b4c5f52a2d997fb891e3497ccd3d7b23b43aff4980007.md), and [`clean_rescues.md`](documentation_sources/versions/eaca551880e0a9db5c4bdc7785e135675cd518ebfcd361d5aa482beca4f37dce.md). Archived correction and decision records retain their original bytes. New findings belong in the consolidated documents with their historical IDs. Full result tables remain in the linked source records.

<a id="historical-indexes"></a>

## Historical decisions, proposals, and audits

These indexes define the historical labels used in the research records. A label records a decision, proposal, or audit finding. It does not by itself authorize work or prove a result. The full source records retain the original dates and reasoning.

## Decision ledger

This compact ledger records historical decisions and their current interpretive status. The archived [decision log](documentation_sources/versions/2b7293c884e0e0aa9f1db178d13f55497d584cd7060c68c022436d47330bb63c.md) is the authority for full rationale and chronology. "Current" means a governing scope or policy, not necessarily a current experiment; "superseded" means do not reuse the old claim or implementation as an unqualified present conclusion.

| ID | Decision or lesson | Status and qualification |
|---|---|---|
| D0 | JEPA latent prediction rather than pixel forecasting | Historical architecture choice; hypothesis, not established objective. |
| D1 | Pure PyTorch training loop | Historical implementation rationale. |
| D2 | LUMIERE RANO categories as action proxy | Provisional historical hypothesis; RANO is not a treatment action. |
| D3 | Full BRAINIAC preprocessing rather than a deferred pipeline | Historical execution choice. Resized tensors are not physical-volume rulers. |
| D4 | MNI152NLin2009cAsym template | Historical template choice. Equivalence with SAILOR is not established. |
| D5 | Multiprocessing with capped worker threads | Operator/engineering history, not research evidence. |
| D6 | LoRA rank 8 and alpha 16 on the frozen backbone | Historical JEPA design, not a requirement for every representation. |
| D7 | Official first-patch-token pooling | Retained in the historical BRAINIAC path after A3. |
| D8 | Official BrainIAC weights and behavioral verification | Governing integrity lesson. Matching keys are insufficient. |
| D9 | Mean over available single-sequence latents | Historical fusion choice. Missingness stays explicit. |
| D10 | Sinusoidal and learned time-gap encodings | Historical temporal design. R34 separately versions elapsed coordinates. |
| D11 | A target for each eligible sequence prefix | Historical training design. Cutoff validity still applies. |
| D12 | Cosine loss on normalized embeddings | Historical loss choice, not a representation-quality criterion. |
| D13 | Drop imageless visits and require pixels in loss pairs | Governing data validity in image-based paths. |
| D14 | Explicit NaN-safe clinical parsing | Governing finite-input safeguard. |
| D15 | Chunk backbone encoding | Historical memory safeguard. |
| D16 | Pin transformers below 5 and exclude torchvision | Current repository dependency constraint. It does not prohibit weights-only pretrained comparators. |
| D17 | Five-patient CPU pilot | Historical execution plan, with invalid and corrected runs retained. |
| D18 | Free Kaggle T4 for the hero run | Historical capacity choice, not current quota permission. |
| D19 | Additive clinical fusion | Parked proposal, not authorized active work. |
| D20 | Resume the best checkpoint at lower learning rate | Historical recipe. It does not supersede final20 grading in R36. |
| D21 | Temporal-prefix checkpointing and T4 memory fixes | Historical implementation and capacity record. |
| D22 | Champion continuation ejected from its optimum | Historical observation. Later accumulation and clean rescues qualify the mechanism. |
| D23 | Frozen downstream RANO probes | Historical first readout experiment. Later CV and schema corrections govern interpretation. |
| D24 | Literature pivot toward downstream representation tests | Historical prioritization, not proven JEPA superiority. |
| D25 | Joint JEPA and RANO auxiliary fine-tuning | Completed negative historical test. It does not authorize another sweep. |
| D26 | SAILOR adapter and cross-site evaluation | SAILOR remains controlled-access developmental transfer, not pristine external validation. |
| D27 | Accumulation, bucketing, target-eval and rating-strip fixes | Engineering record for the earlier training pipeline. |
| D28 | Same-space dynamics/persistence comparison | Governing correction: mixed-space ratio withdrawn. |
| D29 | Lead-time and atlas protocol | Historical analysis; findings and limits remain tied to its record. |
| D30 | Treatment-conditioned SAILOR dynamics | Historical test; no treatment benefit reported, not a general null for action conditioning. |
| D31 | R11 basin-hold launch and monitoring | Historical execution. Optimizer-stripped artifacts limited the first comparison. |
| D32 | R11 repair and transient short accumulated legs | Recipe-specific history, not durable optimization success. |
| D33 | Modular train/evaluation registry | Engineering architecture history. |
| D33b | Loaded optimizer-state CPU/CUDA crash and repair | Separate later engineering record. The fixed v2 basin matrix completed. |
| D34 | Native harness JEPA training and latent evaluation | Interface/provenance migration, not a new scientific result. |
| D35 | Kaggle call sites moved to the harness | Historical reproducibility wiring. |
| D36 | Kaggle pins and synchronized notebook pairs | Engineering integrity record. |
| D37 | uv and locked environment | Current dependency/reproducibility convention. |
| D38 | Verified harness migration and first script retirements | Historical compatibility checks. |
| D39 | Trainers moved under `src/harness/train` | Engineering ownership history. |
| D40 | Analyses and feature builders moved under the harness | Engineering ownership history. |
| D41 | Jupytext restored as a dev extra | Current notebook source/pair integrity rule. |
| D42 | Separate assessment, forecasting, anatomy forecasting, and surprise. Reset the validity checks. | Current task distinction and validity policy. |
| D43 | No broad JEPA sweep before matched benchmark | Historical gate; consistent with current evidence discipline. |
| D44 | Freeze refreshed P0 readout policy | Scoped RANO policy; historical `final` is development, not a new holdout. |
| D45 | Three-compartment RANO-free lesion endpoint | Scoped benchmark choice. Its initial missing-mask status was superseded by D46. |
| D46 | Recovered native masks and next-observed-visit contract | Governing scoped timing/eligibility contract. Exact horizon and rate claims remain unsupported. |
| D47 | Close P1 with negative JEPA lesion forecasting result | Negative for the tested model/task/cohorts; not universal. |
| D48 | P2 as gated lesion-aware repair | Historical P2 design decision; not permission to retrain. |
| D49 | Anatomy-first harness and legacy compatibility boundary | Current interface distinction; legacy RANO path does not define the active task. |
| D50 | Structured physical-history residual gate failed | Negative for the tested forecast. Pipeline completion does not authorize lesion-aware retraining. |
| D51 | Frozen vision/JEPA fusion did not rescue P2 | Scoped negative. It does not establish that every representation or supervision family fails. |
| D52 | Prefer UCSF supervision; storage/access gate | External cohort acquisition is gated; no implicit download or substitution. |
| D53 | Reopen bounded clean-data rescues; A40 cutoff repair | Historical authorized rescue context; invalid forecast claims remain withdrawn. |
| D54 | Verify observations/chronology; alignment and checkpoint boundary | Governing validity cautions; coordinate repair is not evidence of gains. |
| D55 | Objective is broader than current method | Governing research principle. |
| D56 | Whole-cohort source-grid QA | Integrity result only; not segmentation accuracy, repeatability, biological change, or forecasting benefit. |
| D57 | Execute matched R36 objective contrast | Phase 1 authorization only; no Phase 2 authorization and no result acceptance implied. |
| D58 | Separate decoder benefit from encoder progression | Completed R36 interpretation. Frozen accessibility and D's tradeoff remain inconclusive. C passes the specified direct-volume retention margin, not a general quality test. See [Notes](NOTES_AND_INSIGHTS.md#d58). |

## Direction and audit indexes

These indexes preserve useful lineage without turning every historical suggestion into a current TODO. Full descriptions and dates remain in [directions.md](documentation_sources/versions/eade325e60cbeeb92dc5f6570520d8f502a519e846309ef5803b2bf7e1a2d094.md). Direction labels are not evidence of execution or authorization.

### G directions

| IDs | Concise meaning and disposition |
|---|---|
| G1 | Eventual survival was a clinical input. Current prospective caches exclude it. |
| G2 | Recursive train mode affected the EMA target. The target-eval override was implemented. |
| G3 | Trailing whitespace misclassified operative ratings. Loader stripping was implemented. |
| G4 | SAILOR gaps could misalign after dropped visits. Preserve visit identities and audit timing instead of copying prefix gaps. |
| G5 | SAILOR sex was hardcoded in the old adapter. Historical metadata defect, not a biological finding. |
| G6 | `max_visits` was unused. Historical configuration hygiene. |
| G7 | Patient weighting differs from pair weighting. Neither is automatically the right estimand for every task. |
| G8 | Resume omitted saved optimizer state. Later R33 tests loaded versus fresh state explicitly. |
| G9 | Batch-1 training lacked accumulation. Accumulation was implemented and tested. |
| G10 | Row-random ridge tuning mixed a patient's visits. Current tuning is patient-grouped. |
| G11 | Probe CV included encoder-training patients. Locked unseen folds address that claim boundary. |
| G12 | Repeated hero-test use consumed independence. The historical final split is development data. |
| G13 | Test-only same-space persistence comparison was missing. A8/A24 filled and qualified it. |
| G14 | Trained GRU and last-visit baselines were absent. Old untrained losses are not trained-baseline evidence. |
| G15 | Surgical events can make short-gap transitions large. Intervention handling is a task-specific hypothesis. |
| G16 | Rationale codes were unused. Clinical meaning and support require audit before a new target. |
| G17 | Scanner parameters were unused. Acquisition metadata could test nuisance explanations. |
| G18 | SAILOR treatment files were unused. A15 tested conditioning without establishing a gain. |
| G19 | Perfusion and dose files were unused. They remain possible approved-data hypotheses, not automatic inputs. |
| G20 | Results contained a stale open-work list. Later completion records govern the current TODO. |

### ASTRA and K3 audits

ASTRA separates validity findings, V1 to V5, from proposed experiments, E0 to E4. V1 identifies an AUC implementation error. V2 identifies inconsistent label rules. V3 identifies survival leakage, and V4 identifies different prediction problems among the locked patients. V5 calls for a reset of the held-out evaluation and label version.

E0 establishes a prospective baseline. E1 proposes a fair assessment baseline using observed scans. E2 tests whether history adds information. E3 considers lesion-aware residual forecasting, and E4 adds information absent from the current corpus. These labels describe an ordered historical program, not permission to run every proposal. See [ASTRA_PROBE.md](documentation_sources/versions/b72ceb2da7c2ba695a4174a43643847c9fb27244c1785ff1a6acb5c898f2fb29.md).

K3-1 through K3-35 are preserved as an adversarial audit index:

| IDs | Meaning and disposition |
|---|---|
| K3-1 | Phantom SAILOR modalities. A23 added finite-nonzero content guards and rebuilt caches. |
| K3-2 | Arm tables mixed latent spaces. A24 requires each same-space comparison. |
| K3-3 | Transfer displacement magnitude and direction were poor. Oracle calibration is an ineligible diagnostic. |
| K3-4 | CORAL attribution omitted a mean-only control. Covariance shift is not an isolated cause. |
| K3-5 | Matching pipeline code did not match every preprocessing operation. Harmonization remains scoped. |
| K3-6 | Template-family similarity was overstated as equivalence. Rigid and affine transforms differ. |
| K3-7 | Thin staging correlation margin. It does not establish the transfer mechanism. |
| K3-8 | Missing drift-magnitude cell. The audit measured it as a diagnostic. |
| K3-9 | Quantization claim was appropriately narrow. It did not exonerate every preprocessing effect. |
| K3-10 | LUMIERE raw registration was not established. Raw-versus-MNI deltas include position effects. |
| K3-11 | Broad patient-win claims failed same-space checks. Retain aggregation and uncertainty. |
| K3-12 | Effective rank is weak at batch 1. Healthy monitors alone do not establish task quality. |
| K3-13 | LoRA contribution lacked a matched ablation. No causal attribution to LoRA follows. |
| K3-14 | Best-checkpoint tracking risks validation selection bias. R36 grades final20. |
| K3-15 | Patient weighting can protect patients or misweight pairs. Declare the chosen estimand. |
| K3-16 | All-91 probe CV leaks encoder exposure into evaluation. Use locked unseen folds. |
| K3-17 | SAILOR RANO mapping lacks independent confirmation. Anatomy proxy avoids a clinical primary claim. |
| K3-18 | Ceiling-cache anomaly undermined A21 readout claims. A23 withdrew the unreproducible columns. |
| K3-19 | Surprise is contemporaneous change detection, not established early warning. |
| K3-20 | Treatment-conditioned tie could reflect weak input use. It is not proof treatment is irrelevant. |
| K3-21 | Volume readout was vision-driven in that audit, not survival-driven. Prospective survival exclusion still applies. |
| K3-22 | Thin lead-time support limits the suggestive second-horizon result. Current-schema results supersede it. |
| K3-23 | Field fitting used a fixed budget without internal epoch selection. Record that scope. |
| K3-24 | MNI interval accuracy was not independently verified. Exact-horizon claims are unsupported. |
| K3-25 | LUMIERE uses nominal week bins. Suffixes preserve order, not exact acquisition time. |
| K3-26 | Weekly-MRI language contradicted actual gap distributions. Report measured gaps with their provenance. |
| K3-27 | Surgery pairs remained in training and evaluation. Later isolated R33 filtering was negative on pooled anatomy. |
| K3-28 | Eventual survival remained an input. Current-schema prospective caches exclude it. |
| K3-29 | Rating, gap, sex, and action status required source checks. Historical line references are not current verification. |
| K3-30 | Bookkeeping defects needed correction. Counts remain version-specific. |
| K3-31 | Filename-only caches lacked content provenance. Creation-time binding is now required. |
| K3-32 | Hero test reuse consumed independence. A reset name does not create a fresh test. |
| K3-33 | Trained temporal baselines were missing. Compare the specific prospective capability with appropriate trained controls. |
| K3-34 | Collation swallowed load exceptions. Missingness and failures must be visible. |
| K3-35 | Behavioral official-weight checks were supported. Key compatibility remains insufficient. |

### R directions

R1 to R28 are historical proposals and follow-ups catalogued in [directions.md](documentation_sources/versions/eade325e60cbeeb92dc5f6570520d8f502a519e846309ef5803b2bf7e1a2d094.md). The index below conveys topic and disposition, not a claim that each was run successfully.

| IDs | Meaning and disposition |
|---|---|
| R1 | Surprise lead-time. A13 and the current-schema refresh do not establish JEPA early warning. |
| R2 | Transition-type error atlas. A14 measured it; class-dependent patterns motivated later hypotheses. |
| R3 | Honest-number pass. A8/A24 corrected spaces, aggregation, and bootstrap claims. |
| R4 | Codebook, rating, gap, sex, and survival robustness bundle. Some fixes completed; clinical codebook independence remains open. |
| R5 | Mask-guided spatial pooling. A26's narrow frozen ROI interface was negative, not a universal lesion-method null. |
| R6 | Rationale-code targets. Candidate clinical-label audit, not an active approved run. |
| R7 | Surgery-reset or filtering ablation. R33 separately tested filtering and did not reach persistence on pooled anatomy. |
| R8 | Scanner-stratified errors. Candidate acquisition diagnostic, not established harmonization success. |
| R9 | Separate modality probes. Tested interfaces did not establish a locked gain; richer fusion remains a distinct hypothesis. |
| R10 | Survival from trajectory as a target. Candidate censored-outcome task, not permission to restore survival as an input. |
| R11 | Optimizer/accumulation basin hold. Historical transient holds and completed R33 reruns do not rescue anatomy forecasting. |
| R12 | Train GRU and current-visit MLP baselines. A task-specific capability comparison, not answered by old untrained losses. |
| R13 | Transition-balanced JEPA. Loss-side A34 and clean R33 weighting were negative. A sampler is a distinct proposal. |
| R14 | Real treatment conditioning. A15 found no gain for its small SAILOR field; causal treatment effects remain unidentified. |
| R15 | Velocity-field dynamics. A41 clean heads approach persistence without a resolved patient-uniform SAILOR win. |
| R16 | Site-adapter rescue. Support-only and transductive regimes must stay distinct from zero-shot transfer. |
| R17 | Acquisition conditioning or harmonization. Candidate method, not the inferred cause of every transfer gap. |
| R18 | Replace modality averaging. Candidate architecture change requiring a matched approved comparison. |
| R19 | Sharper or probabilistic predictor. Proposal. Lower latent loss alone does not validate useful change. |
| R20 | Hidden BraTS/Lighthouse validation. Candidate independent test requiring compatible task, access, and reservation. |
| R21 | Visit-consistent registration. Candidate alignment/noise study, not proof of biological correspondence. |
| R22 | Continuous-time models. Proposed dynamics family. Current timing does not support exact physical-rate claims. |
| R23 | Prospective walk-forward simulation. Requires strictly cutoff-available inputs and a new protocol. |
| R24 | Pseudo-progression discrimination. Candidate task requiring reliable treatment, imaging, and clinical labels. |
| R25 | Perfusion second tower. SAILOR-native hypothesis, not a current structural-MRI comparison. |
| R26 | Uncertainty and out-of-distribution checks. Required before clinical utility claims, not yet established. |
| R27 | Treatment counterfactuals. Hypothetical later capability, not identified by observational treatment labels. |
| R28 | Faithful pretrained supervised comparator. A29-A31 from-scratch failures do not answer it. |
| R29 | Train/validation generalization-gap monitor. Diagnostic proposal, not a scientific endpoint. |
| R30 | Imbalance handling beyond class-weighted CE. Historical proposal for classification evaluation. |
| R31 | Feature-alignment loss for cross-domain dynamics. Proposal, not proof that covariance/alignment explains transfer. |
| R32 | Pretrained comparator to test whether an apparent edge is BRAINIAC-specific. A useful controlled-comparison idea, not a current authorized run. |
| R33 | Bounded clean-data rescue matrix. Seven GPU arms completed without reaching pooled anatomy persistence. A40 withdrew leaked forecasts; A41 retained separate latent diagnostics. |
| R34 | Coordinate, chronology, and checkpoint repairs. Structural implementation review closed. Whole-cohort source-grid QA passed, but biological and repeatability limits remain. |
| R35 | State, observed-change, and nuisance representation comparison. Completed corrected sibling record; bounded as described above. |
| R36 | Delta-representation objective contrast. Phase 1 incomplete at the October 11 snapshot. Computation and scoped reporting closed October 13; primary encoder gain remains inconclusive. Phase 2 unauthorized. |
