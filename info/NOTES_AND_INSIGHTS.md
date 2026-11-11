# Notes and insights

## R68 has recorded follow-up, but not verified cutoff-known intent, November 9, 2026

The [bounded preparation](EXPERIMENTS.md#r68-preparation-completed-with-clinical-context-and-adjudication-blockers-november-9-2026) separates available imaging pairs from a trustworthy clinical comparison. The recommended post-initial-RT 12-week phase retains materially more history than its first-scan-only subset. That is a metadata-based scope decision, not evidence that treatment context or history will improve prediction. Event dates and delivered courses cannot answer what was planned and known at the cutoff. Likewise, expert refinement of each released mask is useful supervision but does not resolve whether a particular longitudinal difference is biological change, acquisition variation or annotation inconsistency.

The next work therefore needs specific source records and qualified pair review, not another forecast fit. This is a demonstrated limitation of the checked records under R68's contract, not an established explanation of all earlier forecasting failures. R64 can characterize measurement variation after access approval, but repeat scans do not supply the missing longitudinal-change truth or treatment intent. Preserve the closed model results and untouched reservation while these dependencies are resolved.

## Explicit cavity helps MU, but conditional retention is not native retention, November 6, 2026

The completed cavity build reduces current MU core/cavity confusion and core volume error on matched developmental visits. The original three compartments improve even without counting the newly supported cavity category. However, the larger core benefit includes fewer false positives when core is absent; improved nonempty-core Dice remains uncertain. Cavity itself still has moderate overlap and substantial false positives when absent. These are useful supervision gains with remaining measurement limitations, not reliable trajectories.

The LUMIERE result exposes a readout-contract problem: conditional four-category segmentation preserves its annotation task approximately, while the native five-category output agrees substantially less with those annotations. Unknown cavity truth prevents interpreting every disagreement as a biological error, but a conditional metric cannot validate a different readout shown in the timeline. LUMIERE missing-FLAIR lesion detail also worsens despite little whole-brain change. Preserve the old baseline for established applications and the new checkpoint as a developmental MU candidate. Resolve the annotation/readout distinction and retention before attributing improved future predictions to a better universal state; do not launch another temporal architecture or repeat this configuration across seeds as if those issues were solved.

## Anatomical supervision precedes further transition development, November 6, 2026

The operator deferred reserved assessment because its outcome would not alter the present supervision and longitudinal-target priorities, and the updated observation model may supersede the current forecaster. Preserve the post-hoc MU24 FLAIR-abnormality segmentation signal exactly as developmental evidence. It is not evidence of useful future-image synthesis. A later preplanned independent assessment can include the frozen candidate alongside the eventual integrated system.

Explicit cavity supervision must not turn missing annotations into known absence. MU can supervise an exclusive cavity category; LUMIERE's original categories are trained conditionally while cavity remains unknown. Conditional LUMIERE accuracy does not validate its native five-way cavity prediction, so both readouts and that limitation must be visible. Restoring the target also exposed a concrete distinction between image foreground and anatomical support: clipping anatomy to nonzero CT1 hides some released cavity labels. The new full-field anatomical recipe addresses that defect but must learn external background; successful execution does not establish that it has done so.

The six-case review reinforces the distinction between estimation error and reference-defined later change. Local core/cavity exchange can occur even when net core volume is nearly stable. Conversely, a large apparent core reduction may coincide with cavity expansion and cannot be called biological resolution from volume alone. Reference-transition tables and review flags make those ambiguities visible; they do not adjudicate them. The investment is justified by a concrete supervision deficiency, not proof that architecture is sufficient or data are the only limitation. Existing spatial and scalar forecasters remain the fixed references for assessing whether a better observation state eventually improves prediction.

Why have the experiments not yet established useful prediction of glioma change? This page explains what the results mean, why several earlier claims were withdrawn, and what the remaining experiments can test.

This page includes the R63 exploratory interpretation from November 3, 2026. Read the [detailed research summary](RESEARCH_SUMMARY.md) for the dated synthesis through R62. Earlier stage-specific interpretations retain their context; later corrections and closure statements govern current claims.

Use [Experiments](EXPERIMENTS.md) for measurements, [Methods](METHODS.md) for exact protocols and historical IDs, and [TODO](TODO.md) for remaining work. The [archive](archive/2026-10-12/README.md) preserves the original records and their correction history.

## Empty-core suppression and FLAIR forecasting are different findings, November 6, 2026

The completed bounded interpretation substantially narrows the MU result. At 24 weeks, future-empty core cases contribute +10.26 overall Dice percentage points, against total improvements of +9.44 (current-only) and +10.35 (history). Nonempty core becomes worse. Successful core resolutions start below 0.7 mL, and the meaningful core resolutions are not correctly predicted empty. Both current-error correction and locations unscored as cavity at the cutoff matter. It would be misleading to call the entire gain prediction of patient-specific evolution, or to treat the core as a reliable state measurement.

It would also be too strong to call every gain current-anatomy correction. The smaller MU 24-week FLAIR effect has no empty-target reward and improves reference-changing locations while sacrificing accuracy against current FLAIR. Current-only gains about 1.46 Dice points in seventeen meaningfully changing targets from twelve patients, including a positive paired interval against privileged current-reference persistence. That is an exploratory candidate signal rather than confirmation. Freezing its exact simpler checkpoint and endpoint is justified; widening the claim, changing the architecture, or reopening development scoring to manufacture robustness is not.

The practical decision has two distinct parts. The next substantial model-development investment should address explicit cavity/core supervision and longitudinal label reliability. Separately, the already-frozen current-only MU 24-week FLAIR point predictor can move toward independent patient confirmation without another training matrix. Neither its very broad volume uncertainty nor the core behavior belongs in that narrow claim. Do not insert the forecast unchanged into observation inference: it improves some current core errors at the expense of current FLAIR anatomy. The integrated observation model, readouts, timeline updates and matched controls remain useful assets regardless of whether the narrow candidate confirms.

## Structured prediction improved distributions without establishing temporal value, November 6, 2026

The completed anatomical-change formulation avoids the earlier extreme volume errors and learns materially more useful volume distributions than the failed feature-transition model. MU anatomy improves over persistence. Those gains do not establish the requested shared predictive capability: LUMIERE anatomy remains below persistence, history worsens LUMIERE Dice relative to current-only, and history has no resolved CRPS advantage over current-only in either dataset. The simpler volume control remains competitive. Nominal 90% intervals cover approximately 71–80%, with mean widths of 44–71 mL for the spatial history model.

The small positive MU 24-week history Dice interval is a limited development signal, not a substitute for the joint anatomical/distribution criterion. Added MU supervision and the changed formulation were introduced together, so improvements over the previous configuration do not isolate either cause. Full-volume selection favors intermediate checkpoints rather than the training boundary; repeating this recipe across seeds or simply extending it is not the supported next action. Preserve the working observation system and the evaluated predictive timeline while deciding which concrete remaining failure warrants further development.

## A structured transition is a new hypothesis, November 6, 2026

The failed feature transition already began at persistence. A residual connection therefore cannot explain or repair its failure by itself. The redesign instead limits changes to a small distribution of compartment boundary fields, derives volumes from each sampled map, and conditions the corresponding image on that same anatomical change. Direct volume CRPS and spatial distribution scoring now participate in training. Full-volume selection replaces patch-only selection.

This formulation introduces its own restrictions: eight scenarios approximate uncertainty; smooth boundary changes may poorly describe new distant tissue or disappearances; and straight-through gradients for discrete map volumes are a surrogate. A coherent sampled examination can still be inaccurate or miscalibrated. The scalar volume control tests whether the spatial model adds value beyond a simpler prediction of compartment quantities.

Restoring MU timing increases fitting support from 38/33 to 67/54 patients at 12/24 weeks, rather than merely increasing updates on the same trajectories. Those remain modest, selected retrospective populations with different annotation conventions. MU contributes recorded day differences, while LUMIERE contributes nominal week bins. Added patients and the changed formulation will be assessed together; a favorable result would not isolate which modification caused it. The previous failure likewise did not establish inherent unpredictability of future anatomy.

## Forecast image improvement did not preserve anatomy, November 6, 2026

The first integrated frozen-observation forecasters improved future image errors but substantially increased compartment-volume errors over persistence. History did not establish a Dice or volume advantage over the matched current-only model, and its volume-distribution score was worse. The very low coverage of nominal 90% intervals is a failure of the predictive distribution, not evidence that uncertainty was successfully quantified merely because the model emitted variances.

Replay of the saved timeline forecasts was exact, and zero forecast residual recovered both current-state readouts exactly. Subsequent observed states still decoded current anatomy. These bounded checks focus the immediate development question on the learned feature changes and uncertainty/loss behavior; they do not yet identify a specific causal mechanism. The bias affects stable as well as changed targets. Preserve this result while improving the same system. Do not replace useful current-state applications or claim a general absence of longitudinal signal from this failed configuration.

## Observation capability supports a forecasting attempt, November 5, 2026

The matched CT1/T1/T2 gain means available FLAIR alone cannot explain the new observation model's anatomy improvement. Joint training also improved missing-FLAIR lesion detail while slightly worsening whole-brain synthesis error. These establish two development applications sharing a declared spatial state. They do not isolate the contribution of pretraining, capacity, supervision, or the longer schedule, and they are not unseen-task transfer.

The bounded MU review exposed a cost of compatible-label training: ignoring the separately annotated cavity prevents a false background target, but also removes a constraint on cavity predictions. A good shared-compartment score can coexist with a large cavity mistakenly labelled as nonenhancing tissue. The preserved model should keep that caveat visible, especially for volumes; masking it with reference cavity labels during inference would conceal the problem. It is a concrete limitation to address in later observation work, not a reason to indefinitely postpone the authorized LUMIERE forecasting comparison.

For the predictive state, incremental value requires the history arm to improve on the trained current-only arm on common targets. Separate stability from change, and read interval coverage together with width and proper scores. Learned feature variance and decoded samples are an uncertainty mechanism, not evidence of calibrated or useful forecasts. All current fitting and selection cohorts remain developmental.

## Integrated development changes the question, November 5, 2026

The operator selected a useful shared observation state as the next foundation. Segmentation now shapes the encoder alongside reconstruction and local structure. Success would demonstrate jointly trained reuse, rather than transfer to an unseen segmentation application. The earlier R71 transfer result and retention failure remain unchanged. Complete spatial connections are part of the state; there is no 64 KiB target for this build.

A working timeline is an integration milestone, not proof of accurate forecasts. Persistence is visibly labeled, and comparison against a subsequent model-derived segmentation is a user-facing consistency check, not evaluation against an independent target. The stronger observation model and its future updater still need full-volume evaluation. Forecasting remains a distinct and uncertain capability; useful current segmentation and sequence completion alone do not establish the world-model objective.

## R67 did not establish added value from one previous visit, November 5, 2026

The [completed bounded comparison](EXPERIMENTS.md#r67-failed-the-joint-patient-history-benefit-claim) did not establish that one preceding eligible visit adds useful information beyond current panel C and common recipient timing under the tested predictor. The joint claim failed. Lesion MAE differences against current-only and predetermined donor history crossed zero. Genuine-history gradient error was worse against both controls, including in every seed. Nonlesion retention passed, with a modest point-estimate improvement that did not establish an aggregate reduction.

Genuine history outperformed registered previous-FLAIR persistence, but so did current-only. That result supports estimating the withheld present image from current observations rather than simply carrying forward the previous image. It does not show that the learned predictor used patient history beneficially. Nor does it establish learned temporal dynamics.

The predefined zero-week, 1–8-week, and longer-gap groups do not reveal a lesion/gradient improvement confined to short gaps. Genuine-minus-current point estimates were worse for both primary fidelity measures in every group. The stronger descriptive nonlesion reduction was in the longer-gap group. These summaries do not replace the failed aggregate claim.

The result remains conditional on one spatial predictor, its whole-brain objective and selection rule, registration quality, historical coverage, and developmental automated targets. A failure to extract useful history is not proof that no such information exists. This adds a negative historical-input comparison to the earlier evidence of useful but task-dependent representation learning. It does not reopen the closed representation studies or establish an MRI observability limit. No forecasting, treatment-response, or longer-history ordering claim follows, and no extension is selected.

## Current conclusion through R71, November 4, 2026

**R71 is closed as learned transfer demonstrated, with the required direct-image retention failed.** Both findings remain part of the result. The [completed comparison](EXPERIMENTS.md#r71-retained-learned-segmentation-signal-but-failed-the-direct-image-retention-margin) supports three conclusions.

1. **Some reusable learning exists.** Frozen learned features beat random features by 0.066786 Dice, with improvement in every seed. The representation learned something useful beyond its original FLAIR task. Learned-state Dice was 0.482944 versus 0.416158 for random features.
2. **Removing the storage constraint was insufficient.** The complete frozen spatial representation fell 0.079152 Dice behind adaptation, whose Dice was 0.562096. The paired interval [-0.093428, -0.065874] lies entirely beyond the allowed -0.05 margin. A large feature cache and good source-task predictions do not automatically produce a sufficiently reusable state.
3. **Transfer is uneven across anatomy.** Learned-over-random improvement was largest for edema, +0.109113 Dice. The largest deficit against adaptation was enhancing tissue, -0.143329 Dice. This motivates a target-specific learning explanation. It does not prove that enhancement information was absent. Accessibility to the readout, optimization, and target quality remain possible contributors.

The combined conclusion is that **our models learn useful but task-dependent representations. Spatial access improves prediction, yet neither compact export nor the complete frozen hierarchy has met the intended reuse standard.** This remains an architectural and learning limitation, not an established MRI observability limit. Every segmentation arm also produced supported false positives in reference-empty cases, so nonempty Dice alone would overstate practical capability.

The complete state costs 334.75 MiB per visit before serialization overhead, 4.05 times the Float32 input and 8.10 times the stored Float16 input. FLAIR lesion MAE remained 0.634692 from the same saved states, preserving R69's demonstrated capability and its unresolved absolute-fidelity limitations. These are developmental observations using automated anatomical targets, not efficient compression, biological-state recovery, or independent clinical validity.

The representation-study sequence is closed. R66, R69, R70, R71, and the privileged diagnostic will not be extended to make the compact representation pass. The next independent foundational question is [R67](METHODS.md#r67-panel-c-bounded-preparation-november-4-2026): does patient history add useful information beyond the current scan? Current panel C remains fixed and current FLAIR remains withheld. Genuine history, predetermined donor history, and previous-FLAIR persistence distinguish patient-specific historical information from added inputs and simple carry-forward. This asks whether history supplies information worth representing, before choosing how to represent it.

## Current conclusion through R70, November 4, 2026

R69 and R70 strengthen one working explanation: **providing a route around the compact state improved the predictor without making that state more useful to the tested fresh readout.** Spatial access substantially improved FLAIR prediction in R69, closing 74.46 percent of the lesion-error gap and 80.30 percent of the gradient-error gap to R65A by point estimate. Yet the enabled encoder's compact states supported worse fresh-readout predictions in R70 than the zeroed encoder's states.

The fresh zeroed-state readout recovered approximately the original zeroed model's lesion MAE, 0.748122 versus 0.756410. This descriptive comparison weakens a general explanation that fresh readouts cannot learn. It is not a formal equivalence test, and representation-specific decoding and optimization limitations remain open. Reduced pressure to preserve useful information in the compact route is a plausible mechanism, not an isolated cause.

R70 is closed as **failed to recover the spatial-access advantage from compact states under the fixed readout recipe**. The compact-readout diagnostic sequence stops here. R66, R69, and the privileged state-fitting diagnostic remain closed. R69's architectural pass does not resolve its absolute-fidelity limitations.

The next question is whether one complete frozen R69 enabled representation supports both the existing FLAIR decoder and a fresh segmentation readout. The representation includes the compact tensor and every spatial feature consumed by the decoder, before the decoder's learned projections. Native precision and the whole payload count matter. The 64 KiB limit was an experimental choice, not a requirement of an MRI world model. A complete state larger than its input is a reuse reference, not an efficient representation.

The scientific sequence is reuse first, then compression that preserves it. Neither a positive reuse result nor good within-visit FLAIR prediction would establish useful history, biological-state recovery, or acquisition transfer. Automated segmentation remains a developmental proxy. The operator approved the [bounded R71 plan](METHODS.md#complete-state-reuse-proposal-november-4-2026) after the footprint report, with readiness before cohort export and all nine selected models locked before scoring.

## R63 separates recoverable image information from useful predictive training

The [controlled-world pilot](EXPERIMENTS.md#r63-controlled-world-pilot-november-3-2026) did not establish the proposed distinction between future and contemporaneous prediction. Both predictive encoders retained broad ellipse geometry while their current-activity readouts stayed near chance. A supervised reference recovered activity substantially better, and even a random encoder supported a stronger activity readout. The current image therefore contained useful activity information that the tested prediction recipes did not make comparably accessible.

The important negative result is shared failure, not a future-only failure. Whole-image squared error, a small lesion, limited fitting, and the encoder-decoder combination are plausible contributors. This comparison did not isolate them. The missing pre-assessment small-example competence check further limits a strong mechanistic claim. Extending the same scored recipe until one arm wins would not resolve these issues.

The subsequent [generator-only calculation](EXPERIMENTS.md#generator-only-follow-up-how-much-loss-rewards-activity) quantifies one competing explanation without training. Given all other generating factors, knowing activity reduces expected whole-image error by about 0.000065 for either the contemporaneous or predictable-future target. That is about 1.7 percent of the error remaining when activity is averaged out and about 0.5 percent of a cohort-average-image reference error. The randomized future offers zero reward for current activity. The scalar objective therefore contains the intended distinction, but the positive reward for local activity is small on these scales. This does not establish a threshold below which learning must fail or show that the optimizer reached its best solution.

The working hypothesis is now a shared failure to make small observable features accessible under whole-image prediction training. Future unpredictability is not the leading explanation for the observed common failure. Privileged loss calculations, random-encoder readouts, supervised competence, prediction-model fitting, and hidden-state calibration answer different questions. None substitutes for the others.

The operator closed this pilot with a narrower conclusion. The images contain useful activity information, and the training score rewards it relatively little. A small reward remains a reward. The calculation explains how apparently good image scores can coexist with missed activity, but not why training reached those solutions. No additional fitting of the scored pilot is selected. A future causal test of the loss-weighting explanation would keep input images unchanged and deliberately vary the reward for lesion accuracy. That is a separate, unlaunched experiment.

R65 carries forward an evaluation requirement rather than a presumed solution. Added sequences must be assessed for lesion details as well as broad image accuracy. Lesion-region appearance is still an imaging endpoint. Improved reconstruction of a region labeled as FLAIR hyperintensity is not independent evidence about viable tumor.

The selected LUMIERE substitution separates access from the scientific question. Its within-visit task does not require new longitudinal timing information. Released atlas images avoid the project's 96-cubed bottleneck, but their nominal 1 mm grid cannot undo thick-slice acquisition blur. Shared preprocessing may also depend on the full sequence panel. The readiness success establishes that this direct predictor can fit two fixed examples under the selected objective; it does not establish a benefit from added sequences or a reusable patient-state representation. The cohort comparison remains necessary, and reused patients and automated masks limit any eventual result to developmental imaging evidence.

The completed R65A assessment supports the joint image-and-lesion criterion, with a much larger lesion benefit from T2 than from precontrast T1. This addresses the concern carried forward from R63: the positive finding is not only better broad anatomy. The added T2 lowers lesion-region error by 30.88 percent relative to the two-T1 panel, alongside a 7.85 percent whole-brain reduction. Lesion contrast and local gradients improve descriptively too. The precontrast-T1 lesion gain is smaller, its adjusted interval is close to zero, and its compartment-level improvements are not uniform.

This does not make a direct image-prediction network a world model. Spatial skip connections can pass detailed image information without a compact reusable state. R65A supplies a direct-image reference against which a later exported-state experiment can measure retained capability. A state that loses the added-T2 lesion benefit would lose an observed useful capability, even if its overall image score remains competitive. That is a separate test, not a conclusion from this one. External replication, acquisition robustness, stronger anatomical references, and trustworthy additional task labels remain unresolved.

The exact hidden counterfactuals provide an input-consistency result. Identical images produced identical probabilities for opposite hidden labels, as they must. The relevant uncertainty test compares those probabilities with the known fair-coin posterior. Their excess Brier loss shows imperfect uncertainty estimates. Equality alone does not establish calibrated uncertainty or rule out every possible leakage path. The continuous readouts did not estimate uncertainty at all.

R65 can test a separate real-image information question without requiring a positive R63 result. It should use competent direct-image predictors rather than assume that the current 64-number synthetic state is suitable for MRI. R64 remains necessary for real acquisition reliability and cannot be replaced by synthetic success or failure. The later frozen-state and temporal studies remain separate questions, with their own target and access conditions.

## R66 reference failure does not establish a compact-state limit

The [segmentation-readiness result](EXPERIMENTS.md#r66-segmentation-reference-readiness-failed-november-4-2026) stops the planned comparison at reference competence. The failing models had direct image access, so their missed necrotic/nonenhancing compartment cannot be attributed to the compact state. Successful seeds show that some fits can recover that compartment under this recipe, but do not explain why other seeds largely omit it.

A whole-visit score can conceal selective underfitting of a small compartment. Here the per-compartment scores expose that failure despite much better enhancing and edema overlap. Loss weighting, initial decoder behavior, encoder adaptation, and the fixed fitting duration remain possible explanations, not established causes. A next diagnosis should distinguish these explanations on the same fitting examples without using scoring outcomes. It must remain separate from the completed fixed-readiness result rather than quietly extending failed fits or retaining only successful seeds.

The 64 KiB preference remains a storage choice based solely on whole-brain FLAIR selection error. It is neither a segmentation result nor a biological-state claim. The two-application reuse question is still unanswered.

### The separate diagnostic supports a role for cross-entropy weighting

The [completed paired diagnostic](EXPERIMENTS.md#r66-separate-cross-entropy-weighting-diagnostic-completed-november-4-2026) narrows the earlier uncertainty. Changing only cross-entropy averaging improved the necrotic/nonenhancing Dice in all three seeds and both visits per seed. The images, architecture, initialization within each pair, example order, and update budget were matched. This supports cross-entropy weighting as a contributor to unstable fitting under this recipe. It does not establish absent MRI information, generalization, or biological validity of the automated compartments.

The result is not simply that Dice ignores a small class. Dice already averages present foreground compartments equally. Nor do the measurements show that cross-entropy and Dice always oppose each other. Inside the label-two reference region, both losses' recorded mean gradients favor increasing its logit. Averaging cross-entropy over classes changes the relative weight of these rare voxels and the much larger background. At initialization it reverses the cross-entropy head-bias gradient for two seeds and reduces it for the third, while leaving Dice gradients unchanged. Global gradient directions vary over the trajectory. Direction and scale both change, so this experiment does not identify a single universal gradient-conflict mechanism or explain every optimizer step.

Class balancing also changes the error tradeoff. Enhancing mean Dice improves in every seed, but edema mean Dice declines in every seed. Seed 31 recovers the small compartment with substantial false positives. Higher probability within the reference region therefore cannot stand in for faithful segmentation. The paired Dice and volume measurements are needed together.

These are new diagnostic fits of two repeatedly seen visits. They are not replacements for the original failed readiness fits, whose missing checkpoints prevent retrospective trajectory recovery. The unchanged R66 comparison remains paused. Any new segmentation recipe or readiness protocol requires a separate decision rather than treating this diagnostic as permission to resume.

The operator subsequently closed this fitting investigation and selected the [assessment-order amendment and segmentation revision](METHODS.md#r66-assessment-order-amendment-and-segmentation-revision-november-4-2026). The diagnostic tested supervised direct-image models, not frozen FLAIR-trained states. The immediate question is therefore whether the completed states retain R65A's lesion-appearance capability. Failed segmentation readiness need not block that independent measurement. Class-balanced CE is now a fixed revised segmentation recipe with known tradeoffs, not an established universally better objective or permission to substitute successful diagnostic seeds.

### The locked states miss the fixed FLAIR lesion-accuracy requirement

The [completed FLAIR scoring](EXPERIMENTS.md#r66-locked-state-flair-assessment-november-4-2026) separates two questions. Added T2 still improves lesion appearance relative to the two-T1 state at every size and seed. Yet the compact models' absolute lesion errors remain too far above the R65A references, especially in panels B and C. The retained-benefit margin is unresolved at every size. A positive T2 effect does not establish retention of enough of the direct-image capability.

More bytes improve whole-brain error, but do not monotonically improve lesion error across panels. The locked 64 KiB preference was chosen using whole-brain selection MAE, not lesion scoring results. It remains the correct record of that choice, even though it fails the lesion criterion. Replacing it after scoring would conceal the distinction between storage selection and scientific acceptance.

The visible blur is consistent with missing lesion detail in the current predictions. It does not locate the cause uniquely in storage size, the encoder, the no-skip decoder, image-loss weighting, or optimization. These states and selected decoders did not retain the required FLAIR prediction capability. That is narrower than proving that the states contain no recoverable compartment information.

The original ten-condition claim cannot pass when a necessary FLAIR condition fails, regardless of future segmentation results. Segmentation remains scientifically distinct. A successful frozen-state readout could demonstrate anatomical reuse without rescuing FLAIR retention. The revised segmentation recipe is already fixed, but its fresh readiness and cohort fitting have not started. Further segmentation work should be understood as that separate question, not an optimization route to changing the scored FLAIR outcome.

The operator subsequently closed R66 and deferred segmentation. The informative pattern is useful added-T2 information alongside inadequate absolute fidelity. All three storage budgets share the same sixteen-cubed spatial grid. Increasing channels tests one allocation of added capacity, not whether a different spatial grid would preserve lesion details. R65A and R66 also differ in architecture, spatial information paths, and training setup. Their gap does not isolate compression.

The next diagnostic asks whether a frozen selected decoder can express the missing detail when its state is fitted directly to the visit's FLAIR target. The ordinary encoded state is the baseline. Encoder and decoder weights stay frozen, and the optimized state must retain the original shape and Float16 payload. The operator subsequently fixed and authorized the [six-case protocol](METHODS.md#privileged-frozen-decoder-state-fitting-diagnostic-november-4-2026), including matched individual R65A predictions and three per-case recovery conditions. Fitting targets and lesion masks supply privileged information, so no optimized state is a deployable prediction or a replacement R66 result.

Success would demonstrate expressibility by that decoder and state format on those examples, rather than prove that the ordinary encoder can infer such a state from the permitted MRI. It would not distinguish encoder capacity, training objective, optimization, and input predictability as causes of the ordinary prediction gap. Failure would leave decoder limitations, state format, and unsuccessful state optimization as competing explanations. Neither outcome establishes an information-theoretic storage limit. A lower fitting objective alone must also be distinguished from recovery of the missing local detail.

### Privileged state fitting improves voxel errors without meeting detail recovery

The [six-case diagnostic](EXPERIMENTS.md#privileged-frozen-decoder-state-fitting-november-4-2026) is complete. Direct state optimization reduced lesion MAE below each matched R65A reference and improved nonlesion error in every case. The unchanged decoder and Float16 format can therefore produce substantially lower voxelwise error on these examples than the ordinary encoded states provide. The original prediction errors were not the best voxelwise fit demonstrated by that fixed decoder.

However, none of the six states met the full recovery definition. Every optimized prediction still had worse signed in-plane gradient error than its matched R65A reference. Gradient error improved relative to the ordinary state, but that improvement is not equivalent to the declared recovery. The target-fitted predictions and fixed crops retain visibly smoothed detail.

This is not a clean verdict against the decoder. The optimization objective was lesion and nonlesion MAE, not gradient error, and it ran for a fixed budget from one encoded initialization per case. Decoder expressibility, state format, objective priorities, and optimization remain competing explanations for the unresolved detail gap. Adding a gradient term or more updates now would be another experiment, not a completion of this one. No extension is selected.

The fitted states were also far from their encoder outputs, with displacement norms about 1.66 to 1.89 times the initial state norms. This does not invalidate the measured voxelwise expressibility. It does limit any informal assumption that these were small corrections an MRI encoder would necessarily learn. Target access is privileged, and distance in state coordinates does not itself prove a state is uninferable or outside a learned distribution. R66 remains closed, segmentation stays deferred, and this diagnostic does not establish a universal information limit.

### R69 isolates a useful spatial-access intervention, not a universal compression limit

The [completed R69 comparison](EXPERIMENTS.md#r69-spatial-access-passes-the-joint-architectural-comparison) passed the fixed joint architectural test on the scoring cohort. Spatial access reduced lesion MAE and the unchanged lesion-band gradient error while improving nonlesion MAE. Every seed favored the enabled arm for all three components, including their paired patient intervals. This result is stronger evidence for a cost of the bottleneck pathway under the tested recipe than the earlier readiness images or the unmatched R65A-versus-R66 architecture comparison.

The intervention was end-to-end. Enabling connections changed decoder access and encoder gradients, allowing the learned states to diverge. The result therefore supports spatial access as an intervention on the whole learning pathway. It does not identify how much error arose from encoder information loss, inadequate decoder access, the original representation-training objective, or their interaction. The spatial-access control is not a 64 KiB total representation. Point estimates show that spatial access closed about 74% of the lesion-error gap and 80% of the gradient-error gap to R65A. The bottleneck pathway is now a demonstrated source of degradation under this recipe, without identifying which part of that pathway caused the loss.

The relative success also falls short of reference-level fidelity. The enabled arm's lesion deficit against R65A averages +0.041755, but its paired interval extends to +0.074911, leaving the +0.05 tolerance unresolved. Its gradient deficit remains positive in every seed. Seed 31 passes the lesion tolerance, seed 23 is unresolved, and seed 17 fails it. A lower mean error than the zeroed arm is therefore not a sufficient restoration claim, and averaging seeds cannot erase this variation.

Together with the closed privileged-state diagnostic, R69 supports limitations of the current representation-training and spatial-decoding design as a productive explanation for part of the observed error. The privileged diagnostic showed unused voxelwise expressive capacity, while R69 showed that permitted MRI-derived spatial features improve ordinary end-to-end predictions. Neither study proves that all target detail is inferable from those inputs, and neither establishes a fundamental byte limit. R69 is closed as passed for the architectural comparison, with absolute fidelity unresolved. The operator subsequently authorized R70 as a separate frozen-state readout comparison, not an extension of R69. R66 and privileged state fitting remain closed.

### R70 does not preserve the spatial-access advantage through compact-only export

The [completed R70 comparison](EXPERIMENTS.md#r70-did-not-recover-the-spatial-access-advantage-from-compact-states) tested fresh, matched compact decoders on actual saved Float16 states from the six selected, completely frozen R69 encoders. It did not reuse trained R69 decoders or turn off their connections at inference. Enabled-state readouts were worse than zeroed-state readouts on lesion MAE and gradient error in every seed, and their seed-averaged nonlesion deficit exceeded the allowed +0.05 margin. All three joint components failed.

The enabled-state readout's lesion MAE was 1.017337, compared with 0.748122 for the zeroed-state readout and 0.634692 for the full R69 enabled predictor. Its paired deficit against the full predictor was +0.382645 for lesion MAE and +0.032326 for gradient error. These measured deficits do not support keeping the present spatial-access training unchanged while exporting only its compact state for this readout. They also show why the readiness checks could establish fitting capability but not resolve the cohort recoverability question.

The demonstrated R69 benefit remains dependent on the additional spatial features under the tested readout recipe. The fresh zeroed-state readout's lesion MAE of 0.748122 is close to the original zeroed model's 0.756410. This weakens a general fresh-readout learning failure explanation, without establishing formal equivalence or excluding representation-specific limitations. Reduced pressure to preserve information in the compact route is a plausible mechanism. Representation organization, decoder compatibility, and optimization remain alternatives. The experiment does not prove that the state contains no detail or that another objective or readout cannot recover it.

R69's positive architectural result and R70's negative compact-state result therefore coexist. The bottleneck pathway degraded predictions under R69's recipe, but better full-model prediction did not imply a better compact representation recoverable by the original fresh decoder. R70 is complete without extensions. R69 retains its closed architectural pass and unresolved absolute fidelity, while R66 and privileged state fitting remain closed.

## November 1 documentation closeout corrections

The documentation closeout changes no model, prediction, or aggregate. It corrects the old TODO summary of R37 from `+0.000559` with interval `[-0.002975, +0.004431]` to the raw aggregate's H-minus-L CRPS difference `+0.00334486` with adjusted 99% interval `[-0.04732935, +0.05234837]`. The full experiment section already reported the correct values. The outcome remains inconclusive.

The closeout also marks R48's pending follow-ups as historical, applies the existing R50/R51 causal-attribution correction to the TODO, and narrows R38's explanation. R38 tested joint training and recorded state-retention harm, but did not identify noisy-target overfitting as the unique cause or rule out every encoder limitation. The [summary](RESEARCH_SUMMARY.md) retains these distinctions.

## R62 adds a usable cohort but no useful measured-history gain

The [MU-Glioma-Post entry study](EXPERIMENTS.md#r62-did-not-establish-useful-history-forecasting-on-mu) establishes that this cohort supports a bounded longitudinal comparison with reviewed spatial masks and recorded MRI days. It also preserves forty unopened patients for later confirmation. Released timepoints remain selected follow-ups, and date/path mismatches require exclusions rather than inferred corrections.

The first prediction result resembles R61. Proper measured history barely changed current-only ridge performance and lost to persistence. This weakens an explanation based solely on LUMIERE's old coarse targets or its particular patient sample. It does not show that earlier MRI contains no information. R62 uses reviewed measurements directly, tests one strongly regularized linear recipe, and leaves future scheduling and intervention context unmodeled.

The error breakdown matters. Learned residuals reduced errors on changed and nonempty cells but added error to cells that remained empty. The fitted models minimize squared residual error while selection uses absolute error. All three ridges selected the strongest penalty, and their selected fits also lost to persistence on fitting MAE. These facts limit what a negative result says about temporal information. They do not justify changing the endpoint or tuning the model after scoring.

Keeping cavity separate was useful for interpretation. H lost to persistence on both tumor-only and cavity error, so the primary failure was not caused only by adding the cavity channel. The fixed images also showed that current-only and history predictions were nearly identical despite visibly changing anatomy. A small response to a 2-mm history shift is consistent with weak use of history under this recipe, not evidence of acquisition invariance.

MU remains available for a distinct spatial-representation question. R62 did not train BrainIAC, compare MRI-token interfaces, or test current-scan segmentation fidelity. Reviewed masks are stronger anatomical references than unreviewed automated proxies, but they remain retrospective annotations. Neither pretraining-patient separation nor independent biological validation has been established.

## R61 preserves the negative forecasting conclusion on corrected targets

The [corrected-target repeat](EXPERIMENTS.md#r61-corrected-targets-did-not-establish-useful-history-forecasting) found only a small history advantage over current-only regression, and history still lost to persistence. Removing coarse sampling was necessary for faithful source-volume targets, but it did not produce useful history forecasting in this fixed linear recipe.

The corrected comparison narrows the explanation for the earlier failures. The coarse-target defect cannot by itself explain why this recipe loses to persistence. R61 does not settle whether richer history, better spatial representations, known forecast intervals, or observed treatment plans would help. All three regressions selected strong regularization, and fitting squared error differs from selecting absolute error. Those limits remain even when target aggregation is correct.

The primary and conditional errors answer different questions. H slightly improved errors on future-positive and changed cells, while its extra corrections increased future-empty error enough to lose overall. The large transitions in the fixed examples remained poorly predicted. Neither a small average history effect nor corrected measurement precision establishes useful patient-specific anticipation of those transitions.

R59 and R61 share a qualitative result, but their absolute MAEs are not directly comparable. R61 changes both source-volume aggregation and the log transform's units. It is not a factorial estimate of the sampling correction alone. The same eight scoring patients are reused development evidence, and the source masks remain automated proxies. R57's earlier context gain still needs separate confirmation on corrected targets; R61 does not test current-state estimation or a spatial encoder.

## R60 removes coarse sampling from regional burden targets

The [completed correction](EXPERIMENTS.md#r60-quantified-and-corrected-coarse-regional-target-discrepancies) preserves original segmentation volume inside the same fixed physical regions. It separates spatial decoder resolution from burden measurement precision. The broader cohort check confirms that R59's missing-positive examples were not isolated: hundreds of coarse regional cells lost source foreground, and some changes reversed direction.

This resolves the demonstrated target sampling defect. It does not turn automated segmentations into expert biological truth, prove template equivalence, or establish that future disease changes are predictable. Voxel-center assignment preserves discrete source volume while retaining source-resolution uncertainty at regional boundaries. Out-of-field volume is explicit rather than silently dropped.

The corrected cache uses physical mm³ and `log1p` millilitres. Its contract and field names prevent accidental reuse of old log-count targets and their normalizer through the new loader. R61 subsequently refitted normalization on fitting patients and compared persistence, current-only, and history models on this endpoint. The completed R56 through R59 outcomes remain valid records of the old coarse-target experiments. R60 supplies corrected inputs for a new comparison, not revised scores for those studies.

## R59 identified a target-construction problem without explaining every forecast failure

The [completed R59 diagnostic](EXPERIMENTS.md#r59-found-target-resolution-artifacts-and-no-useful-measured-history-forecast-gain) answers part of the operator's data question. Earlier mask-derived measurements did not yield a useful forecasting gain in the fixed regression comparison, so R58's frozen MRI encoder is not the only plausible limitation. H improved on its matched current-only regression by a small amount but lost to persistence. Strong regularization and the squared-error versus absolute-error objective difference limit what this negative recipe can establish.

The image and source-mask comparison found a specific problem in the measurement pipeline. Regional targets count binary foreground after nearest-neighbor resampling to a 24-cubed whole-brain grid. In the inspected field, each target voxel represents about 617 mm³. A positive source region can therefore become empty, and small changes in shape or position can change which sample points hit foreground. One inspected region increased on the coarse grid while its source-mask volume decreased. This is target sampling error, even if the original automated segmentation is accepted as the proxy truth.

The next target definition should accumulate source-mask volume within fixed physical regions, or use a volume-preserving occupancy calculation, before taking logarithms. A low-resolution decoder output grid need not determine the precision of regional burden labels. Comparing only 24-cubed counts makes the learning task partly about sampling artifacts. Changing to physical volume also changes the endpoint and its normalization, so old results must remain historical rather than be silently replaced.

The inspection does not establish a corrupted dataset or explain all failed models. Some large changes were visible in both source masks and MRI. Cavity-like transitions suggest that intervention history may matter, but images alone do not establish a treatment event. We still lack expert biological targets, a known forecast interval for this task, and independent patients. Correct target aggregation would remove a demonstrated measurement problem; it would not guarantee predictable disease dynamics.

The positive R57 context result remains a gain on its stated coarse automated target. R56 through R59 remain bounded measurements under that target definition. R59 narrows their anatomical interpretation rather than reversing their numerical outcomes. The priority is now a small source-volume target correction and fidelity check, followed by the same simple information comparison under a separately fixed endpoint. Another architecture change would leave the demonstrated target problem in place.

## R58 future-feature alignment did not improve anatomical forecasting

The [spatial latent-forecast pilot](EXPERIMENTS.md#r58-did-not-establish-a-future-feature-or-history-forecasting-gain) carried regional context into a different task, next-observed-visit prediction. Every model received current automated measurements and spatial MRI tokens. The question was whether predicting frozen future features added useful anatomical information beyond direct forecasting, shuffled teachers, latest-only input, and measured-state persistence.

The proper future-feature/history model failed that test. Its anatomical MAE was worse than direct forecasting, with seed disagreement and a zero-crossing adjusted interval. It was worse than both shuffled teachers and the matched latest-only model in both seeds, with adjusted intervals above zero. No neural arm beat persistence on primary MAE. Lower nonempty-region error and CRPS are worth preserving as secondary results, but they do not establish the planned overall forecasting gain.

This is not evidence that training failed to update. All models reduced fitting error substantially by the final step, while selection often favored early checkpoints. One proper-teacher/history seed selected initialization, and eight of ten selected checkpoints failed fitting readiness. The contrast measures the whole fixed training-and-selection recipe. It cannot settle the attainable capability of a better-generalizing model, and final-checkpoint fitting success does not authorize substituting those checkpoints after scoring.

Latent alignment also needs a proper control. The proper teacher reduced latent error relative to direct anatomical training, but both proper-teacher arms still had higher mean latent error than zero feature change. Direct arms are free to repurpose predicted feature coordinates for their anatomical head, so their high latent error is not itself a failure. Neither reduced latent error nor retained feature variation supplied the missing anatomical advantage.

R57's positive current-state context result remains valid within its scope. R58 does not show that this benefit transfers to future change. The recurring absence of useful history gains points toward testing what additional predictive information the available prefixes contain before increasing model complexity. A future proposal should isolate that information question under matched current-measurement access and selection. It should also distinguish next-observed-visit uncertainty from a forecast at a known interval. No such follow-up is part of R58.

The same development patients, automated targets, and frozen supervised teacher remain limiting. Excluding targets absent from the state cache restricts the cohort further. These results support closure of one spatial predictive-objective recipe, not rejection of all spatial world models or validation of biological uncertainty. Conditional diffusion and flow matching remain lower priority because the required state and prediction evidence is still missing.

### R58 leaves data quality and task definition as competing explanations

The [post-hoc target check](EXPERIMENTS.md#r58-post-hoc-target-and-interval-diagnosis) found that most regional-compartment cells were empty at both visits. Current-measurement persistence is consequently a strong baseline. It is also given the current automated measurement directly, while earlier observations enter the learned models through frozen MRI features. The experiment asks for incremental future information beyond a strong measured current state, not whether a neural network can extract any useful anatomy.

There are only twenty-four fitting, six selection, and eight scoring patients in the history-eligible comparison. Selection contains more empty cells and a different mean direction of change. The selected models' poor fitting readiness and final models' much lower fitting errors are consistent with a generalization problem, but they do not identify its cause. Sparse targets, patient heterogeneity, coarse features, imperfect masks, and the training objective remain competing explanations.

Forecast intervals also vary. The model predicts the next observed visit without a known future interval or treatment plan, so it must average over unobserved future timing and interventions. Supplying the realized future gap after the fact would violate this task's information cutoff. A future study could instead define a prediction-time horizon explicitly, with appropriate data support. A bounded visual check of actual scan and mask changes would help distinguish real anatomical changes from segmentation or registration artifacts. Neither that adjudication nor a horizon-conditioned comparison was performed by this descriptive probe.

## R57 supports regional context more than prescribed adjacency or history

The [regional graph pilot](EXPERIMENTS.md#r57-regional-context-helped-without-established-adjacency-or-history-benefit) found a useful development improvement over a matched model that could read only its own region. That comparison held trainable parameter count, initialization, observed history, sample draws, and checkpoint selection fixed. Providing other regions' observations helped estimate nonempty compartments under this recipe.

The shuffled-correspondence control prevents attributing that gain to the chosen spatial relationships. A graph with the same topology and degree but different physical connections performed similarly. The evidence supports additional regional context; it does not establish the face-adjacency prior, anatomical interactions, or biological propagation. The frozen transformer already mixes spatial information, so this is added readout access rather than proof that earlier tokens contained no nonlocal information.

The latest-only graph had lower point nonempty-region error than the history graph. Removing history from the fitted history graph also helped that endpoint, but this post-hoc intervention cannot select a new model. More closely fitting the training trajectories did not yield the required history gain. R57 therefore does not justify future rollout of this graph-state recipe.

The changed primary endpoint and selection rule were motivated by R56's already observed failure on nonempty regions. All R57 arms shared the new rule, but comparisons with R56's fixed references are descriptive and the reused patients remain development evidence. Coarse grid blocks, imperfect masks, negative Gaussian estimates, and sensitivity to resolution limit interpretation. Preserve the context gain without turning it into an adjacency, temporal, or clinical claim.

## R56 tests observation assimilation before future transitions

The operator selected a regional probabilistic state-space model from the [four architecture alternatives](TODO.md#world-model-architecture-options-november-1-2026). The [first R56 stage](METHODS.md#r56-regional-probabilistic-observation-assimilation) estimates current regional automated-compartment burdens from frozen MRI tokens. It tests whether a prior based on earlier scans helps interpret the current observation. This capability can be assessed without reconstructing an MRI or predicting a future scan.

The state is eight regional Gaussian distributions over three compartment log-counts. Spatial location remains explicit, but the regions are coarse grid blocks rather than anatomical parcels. A fixed supervised encoder supplies observations. This isolates a small observation-assimilation recipe; it does not test an end-to-end learned latent world model or settle the broader probabilistic-state-space direction.

The current target remains the automated mask-derived measurement. Better agreement would establish proxy estimation, not recovery of a latent biological truth. Process and observation variances are architectural terms whose separate biological meanings are not identifiable from these targets. Repeated or adjudicated measurements would be needed for that interpretation. The current-state posterior of the existing R38 filter is a relevant simple control; its future-drift forecast belongs to a different endpoint.

The [completed first pilot](EXPERIMENTS.md#r56-did-not-establish-a-useful-learned-assimilation-gain) did not establish the required history gain. The learned filter improved overall MAE slightly but worsened nonempty-region error and lost to simple history averaging and the existing SSM posterior. Its better nominal coverage came with wider intervals. A probabilistic architecture therefore did not, by itself, resolve state quality or uncertainty calibration.

Unlike R53's forecasting recipe, every R56 neural cell passed its fitting-readiness criterion. Training error fell while selection scores largely flattened. Increasing training duration is not justified by an inability to fit. The result concerns one small supervised target-space filter with coarse regional observations. It does not rule out a learned latent-state model, better spatial observations, or a different anatomical endpoint. Any next study needs a distinct mechanism or missing capability rather than a tuning search on these scored patients.

## R55 weakens an encoder-only explanation for blurred reconstruction

The [direct-code diagnostic](EXPERIMENTS.md#r55-code-bypass-did-not-resolve-image-fidelity) bypassed MRI encoding and projection by optimizing one spatial-code array per training example. Joint code/decoder fitting ended almost tied with an equal-duration continuation of MRI-to-image training. Both remained below the fixed individual-fitting criterion and retained visible blur. Replacing MRI features with target-fitted codes did not solve the problem under this recipe.

The duration control matters. Comparing direct codes only with the earlier R54 checkpoint would credit code bypass for improvements that ordinary continued fitting also achieved. R55 therefore weakens an explanation based solely on inadequate MRI encoding. It does not demonstrate that the decoder architecture is incapable of faithful images. The frozen-decoder cell tests one learned decoder's local reachable outputs, and the trainable-decoder cell explores one warm-start optimization path with a fixed code learning rate.

All three curves still improved at the endpoint, and resetting optimizer moments temporarily worsened both trainable-decoder cells. Optimization and duration remain plausible contributors. The exact target-copy oracle validates the metric path, but supplies no evidence about decoder capacity. A future investigation should distinguish output parameterization from fitting dynamics before scaling temporal JEPA. R55 itself stops here, with neither a new decoder nor additional fitting selected from these outcomes.

## R54 puts the unresolved problem before temporal generalization

The [two-patient diagnostic](EXPERIMENTS.md#r54-learned-corrections-but-failed-substantial-individual-fitting) rules out a complete failure to update or learn. Forecasts learned corrections beyond persistence on their fitting images, and removing image tokens degraded the results. Yet current-FLAIR reconstruction remained blurred with both frozen and trainable encoders. Neither completion nor forecasting met the fixed individual-fitting criterion against the strongest controls.

The mean forecast improvement alone would overstate success. One example already had a strong persistence baseline; the other was better served by a shared mean of the two fitting targets. That target-fitted mean is deliberately a memorization control, not a legitimate future predictor for a new patient. The per-example criterion exposes weak individual fitting that an average persistence comparison hides.

R54 motivated direct per-example spatial-code optimization through the same decoder before a temporal objective. R54 did not run that control and could not attribute the blur to one component. The operator subsequently authorized [R55](#r55-weakens-an-encoder-only-explanation-for-blurred-reconstruction), which added equal-duration MRI continuation and found little extra benefit from code bypass. The failure still does not separate decoder parameterization from optimization or duration.

The curves were still improving at the fixed endpoint. Longer training remains an untested possibility, not a demonstrated fix, and the completed diagnostic is not extended from its results. Two examples and one seed provide neither generalization evidence nor proof that MRI history lacks information. A full temporal JEPA comparison remains premature while current-image fidelity is unresolved.

## R53 tests image prediction without lesion-mask supervision

The [R53 protocol](METHODS.md#r53-missing-flair-pretraining-and-frozen-forecasting) uses observed FLAIR images as targets. Completion asks whether current CT1, T1, and T2 can reconstruct withheld current FLAIR. The downstream task asks whether frozen features help predict the next observed FLAIR. The first task fixes representation training before temporal modeling; including history in completion would mix these questions.

Direct reconstruction is the first baseline. A future JEPA experiment must improve image predictions beyond this baseline, rather than claim success from lower latent loss. R53 starts from official BrainIAC weights instead of mask-supervised R49/R52 models. The frozen encoder exposes spatial tokens before any newly trained fusion. Every forecast head receives a fresh projection, so the unadapted comparator does not depend on frozen random fusion weights.

Crossing completion-pretrained versus original BrainIAC with history versus latest-only inputs separates two possible benefits. A completion gain without a history gain supports improved image features under this recipe, not useful longitudinal information. A history gain without a pretraining gain does not justify completion pretraining. Every neural forecast has the same latest-FLAIR persistence path. Earlier context consists of CT1, T1, and T2; R53 does not test the value of earlier FLAIR.

Image prediction has its own limits. Per-image intensity standardization removes absolute scanner intensity. Residual registration and brain-extraction differences can contribute to future image error. A fitting-only interior field, fixed regional scores, structural similarity, image gradients, and fixed examples expose some failures, but none supplies expert tumor truth. Five-percent useful-gain and 0.02 regional-retention margins are chosen development criteria, not clinically validated thresholds.

The study retains the historical patient groups and their anatomy-preparation eligibility. Removing lesion masks from supervision does not remove that cohort-selection history, or make reused scoring patients independent. The result can support a bounded standardized-image prediction capability. Tumor progression, future treatment response, clinical utility, and calibrated uncertainty remain separate questions.

### R53 reconstruction gains did not transfer to forecasting

The [completed assessment](EXPERIMENTS.md#r53-assessment-found-no-established-forecasting-benefit) found a small completion gain, below the fixed useful-gain requirement. That gain did not produce a forecasting advantage over original BrainIAC, latest-only input, or persistence. Learning a missing sequence from contemporaneous anatomy did not establish useful temporal features under this recipe.

All selected forecast models failed fitting readiness. The experiment therefore closes this frozen-encoder residual-prediction recipe without resolving whether a better-fitted model could use earlier MRI. It also leaves earlier FLAIR untested. Early checkpoint selection and worsening selection curves do not justify extending training after observing scoring results.

The image controls constrain what lower MAE would mean. A population-average fitting image beat persistence on future MAE, and resolution-degraded persistence beat its unperturbed version. Those changes need not represent individual disease evolution. Residual alignment, intensity differences, and smoothing can affect the endpoint; R53 does not identify their separate contributions. Structural similarity, image gradients, and the inspected panels help expose the mismatch between lower image error and faithful individual anatomy.

Completion outputs depend on their image tokens, but remain blurred. Nonzero token variation and successful prediction replay establish neither useful forecasting information nor biological validity. R53 supplies a direct-reconstruction baseline for a separately designed temporal JEPA comparison. It does not select completion pretraining as a better forecasting encoder or authorize joint-JEPA training.

## R52 compares the tokens available to a temporal model

The operator selected a [single-visit encoder comparison](METHODS.md#r52-matched-spatial-token-encoder-protocol) after discussing replacement of BrainIAC. Spatial tokens do not require a transformer. A convolutional feature grid can expose the same per-visit token interface. BrainIAC's R49 benefit over random initialization does not establish that it is the strongest available encoder.

The comparison must evaluate the representation that a later temporal model actually receives. A U-Net decoder can recover anatomy from high-resolution skip features even if its final tokens contain weak anatomical information. R52 therefore allows no skip access in either training heads or fresh frozen-token readouts. Both encoders export the same 216 tokens with 64 features. This holds the downstream interface fixed but does not establish that a coarse grid is optimal.

The CNN jointly processes MRI sequences while BrainIAC uses a pretrained per-sequence transformer and learned fusion. Initialization, architecture, capacity, and sequence processing differ together. A favorable result could justify a practical replacement under the fixed recipe, not an isolated causal claim about convolutions or fusion. Shared optimization settings may also favor one recipe; learning curves and fitting checks must bound that interpretation.

Current anatomical proxy quality is a prerequisite worth measuring separately from future prediction. Better mask or volume readouts would not establish that all temporally relevant information is preserved. R52 has no history input, forecasting endpoint, or temporal JEPA objective. Its reused patients cannot provide independent confirmation, and its automated masks do not supply expert biological truth. The [execution record](EXPERIMENTS.md#r52-single-visit-spatial-token-encoder-comparison-november-1-2026) distinguishes real smoke verification and launch from an eventual anatomical result.

### R52 did not establish better linearly accessible anatomical tokens

The [completed comparison](EXPERIMENTS.md#r52-completed-without-an-established-token-replacement) produced nearly identical primary fresh-readout Dice, about 0.0666 in each recipe. The difference changed direction between seeds. Both passed the weak fitting gate, but their fresh-head fitting Dice was only 0.064–0.077, and their empty-compartment false-foreground rates approached one. A successful relative-retention check cannot turn those poor absolute predictions into useful specificity.

The CNN's nonlinear training head had a larger favorable Dice point difference, but that secondary result cannot replace the declared frozen-linear-readout question. A token-only nonlinear decoder can extract anatomy differently from a tokenwise linear map. This experiment therefore does not prove that the tokens contain no anatomy, that CNNs cannot replace transformers, or that BrainIAC is optimal. The common coarse grid, different initialization and sequence processing, and fixed optimization schedule bound the conclusion. Late readout improvement also leaves convergence unresolved without authorizing an assessment-driven extension.

The prespecified decision is to keep BrainIAC as the reference and close this recipe comparison. The result does not select a temporal architecture. The operator's later joint-training discussion concerns a different question: whether temporal JEPA gradients improve an anatomical state beyond freezing or additional state-only adaptation. A matched detached-gradient state-only arm is needed to separate the latter. R38's earlier joint-forecast state degradation makes retained current anatomy a required endpoint, not an assumption. Future teacher MRI must remain a training target rather than predictor context. No temporal or joint-JEPA fit followed R52.

## R50 and R51 separate a clinical comparator from temporal-objective attribution

The [selected clinical studies](METHODS.md#r50-and-r51-observed-response-comparator-protocol) ask different questions on one matched task. DenseNet versus pretrained BrainIAC compares model recipes, so it cannot isolate pretraining or JEPA. Direct, proper JEPA, and shuffled JEPA instead share the same pretrained online architecture and initialization. Requiring proper JEPA to beat both controls tests whether an auxiliary benefit depends on the declared temporal target correspondence. Falling latent error alone cannot answer that question.

The classifier observes both MRI visits. The JEPA predictor sees only the earlier features, but that restriction does not turn the classifier's endpoint into a future forecast. Nor would a successful proper-versus-shuffled contrast prove added history beyond a latest-scan-only model, causal dynamics, or a clinically useful world model. Those remain separate tests.

Sampling and class weighting change the optimal classifier's class prior. Comparing raw log loss would partly measure that known training tilt. A common fitting-only analytic correction and selection-only temperature keep the primary comparison explicit. They cannot guarantee clinical calibration or repair sparse labels. The single scoring CR example and ten reused scoring patients bound any four-class claim. Readiness checks prevent a collapsed DenseNet or teacher from becoming evidence for the competing mechanism.

The [completed comparisons](EXPERIMENTS.md#completed-clinical-comparisons) did not support either proposed benefit. The direct pretrained classifier had worse point loss than DenseNet in both seeds, with an adjusted interval crossing zero. That is an unresolved advantage test, not proof that every DenseNet is better. Proper JEPA had worse mean primary loss than direct supervision, with the adjusted patient interval above zero, but the seed-specific signs differed. Proper JEPA also failed to beat shuffled targets. The result closes this clinical auxiliary-objective recipe without rejecting every possible JEPA formulation.

The teacher was not collapsed, but that was insufficient. Proper-JEPA teachers retained variation and rank while their selected clinical classifiers predicted only one raw fitting class. Their feature prediction error also exceeded same-teacher persistence. The classifier and target checks answer different questions; neither a healthy teacher nor falling auxiliary loss establishes clinically useful discrimination.

The operating-point tradeoff is visible. DenseNet's raw balanced accuracy was about 0.607, versus about 0.240 after prior correction and temperature calibration. Calibration improved proper loss but largely chose the dominant PD class. All calibrated BrainIAC predictions chose PD, and none of the neural arms recalled a minority scoring class. A fitting-prevalence constant had lower point log loss than every neural model. These results do not support a useful four-class clinical model even though DenseNet passed the declared fitting-readiness gate.

Learning curves did not justify an automatic duration extension. Selection minima occurred before the final epochs, while training loss continued to fall. Preserving the fixed endpoints avoids turning these ten scoring patients into another tuning set. R49's narrow anatomy-pretraining gain survives unchanged because it answers a different question under matched dense supervision. Independent clinical confirmation and pretraining-corpus separation remain open.

### What the November 1 methodology repair (`R50-v2` and `R51-v2`) resolved and exposed

Running the [four audited repairs](EXPERIMENTS.md#r50-v2-and-r51-v2-methodology-repaired-clinical-comparisons-november-1-2026) separated instrumentation bugs from the actual modeling bottleneck:

1. **Corner-token `[:, 0]` and uncentered cosine similarity were genuine JEPA bugs.** In `v1`, raw ViT tokens had `>99%` cosine similarity across visits (`0.009` persistence error), making the auxiliary loss nearly constant and noisy. Subtracting the per-scan spatial mean across all 216 patch tokens (`R51-v2`) raised same-pair teacher distance to `~0.27` (vs `~0.38` across patients) and allowed the predictor to learn a non-trivial mapping (`~0.11` training cosine error). Once that bug and the loss-scale mismatch were fixed, `jepa` beat both `direct` (`-0.0148`) and `shuffled` (`-0.0136`) in both seeds—reversing `v1`'s apparent harm (`+0.0421`)—though the ~1.7% gain remained too small and noisy to clear zero.
2. **End-to-end fine-tuning of a 93.9M-parameter ViT at `lr = 1e-4` without dense supervision collapses on 136 clinical pairs.** Adding the `classifier_fitting_readiness` gate to the BrainIAC arms revealed that all six selected ViT classifiers outputted near-constant logits across the 136 fitting pairs (`std < 0.012`, predicting 1 class), whereas DenseNet264 predicted 3–4 classes (`std 0.08–1.30`) and reached `0.311` calibrated balanced accuracy in seed 43. In `direct`, the unfrozen ViT + 3D fusion layers shrank cross-sample activation variance by `>98%` (`smap std = 0.0022` by epoch 20), whereas the JEPA auxiliary loss preserved representation variance (`smap std = 0.1518` at epoch 21) but still could not overcome the small sample size and `1e-4` backbone learning rate for 4-class classification.
3. **Selection-set class mix distorts scalar temperature when selection has 8 patients.** Because the stratified 8-patient selection split had 52% PD (`24/46`) while fitting had 71% PD (`96/136`) and scoring had 82% PD (`32/39`), fitting a temperature $T > 1$ on selection softened the prior-corrected logits toward uniform, worsening scoring log loss by `~0.04–0.06` across every arm compared with the unscaled exact effective-prior offset.

### November 1 corrections to the methodology audit

The v2 measurements above remain intact, but several earlier explanations were too strong and are superseded here.

The v1 prior offset was not mathematically wrong. With row prevalence `pi`, row sampling weight `s = 1 - pi`, and CE weight `w`, the sampled class mass is proportional to `pi * s`; weighting the risk gives effective mass proportional to `pi * s * w`. Recovering the fitting row prior therefore requires `-log(s * w)`, up to a constant. V2/v3's `log(pi) - log(effective_prior)` is the same offset after centering. A literal fixture checks equality and posterior recovery. Changes in CE normalization and selection are genuine estimator changes, not a repair of this formula.

The first ViT token is a spatial patch token rather than a classification token, but self-attention allows it to incorporate every patch. It is not limited to the input corner. First-token pooling also follows the repository's BrainIAC interface. Using all tokens is an architecture choice worth testing, not proof that v1 had a corner-only information bug. High raw cosine similarity likewise does not prove absent useful information; a shared direction can conceal smaller discriminative components.

Near-constant v2 fitting logits are observed. GroupNorm, the learning rate, and lack of dense supervision are competing explanations, not demonstrated causes. GroupNorm normalizes within each example, not across patients. Comparing raw-token SD with a different trained head's SD cannot quantify a causal percentage of lost information. The exploratory probes changed the head and normalization as well as adaptation. The selected [v3 recipe](METHODS.md#r50-v3-and-r51-v3-limited-adaptation-clinical-protocol) therefore tests a bundle. Even improved readiness would not isolate a unique collapse mechanism.

Positive scalar temperature divides all corrected logits equally and cannot change their argmax. Prior offsets can change class decisions; temperature changes confidence and proper loss. Temperature worsened scoring loss in v2, but different fitting/selection/scoring class mixes do not prove that class prevalence alone caused the deterioration. The primary score weights patients uniformly, so row percentages are not its full distribution. Similar CE and JEPA loss values also do not establish similar backbone gradient contributions, particularly with AdamW and clipping.

Several pre-v3 fitting/selection probes already guided the choice. The frozen standardized-token MLP rapidly fitted four classes but overfit selection; regularized logistic probes showed the same fitting-versus-selection tension, and some failed convergence. The six-epoch custom LoRA probe fitted multiple classes, but used a different rate, normalization population, CE reduction, and fixed teacher. No reloadable models or complete artifacts were saved for those exploratory runs. They motivated the fixed candidate but cannot serve as its verified outcome.

Repeated revisions on the same ten scoring patients are adaptive development. Per-revision model locks, patient disjointness, and bootstrap intervals prevent some direct leakage but cannot restore untouched confirmation. V3 retains the original failed studies, fixes its own protocol before cohort fitting, and stops at one bounded result. It must not turn into another sequence of hyperparameters chosen until scoring improves.

### V3 recovers fitting readiness without validating a clinical gain

The [completed v3 comparison](EXPERIMENTS.md#readiness-recovered-clinical-benefit-not-established) answered the immediate readiness question. All eight selected models predicted three fitting classes, and all BrainIAC base tensors remained unchanged. The combined LoRA, standardization, smaller-head, rate, and selection recipe produced a working nonconstant fitting interface. This does not show that freezing alone fixed the earlier failure. It also does not establish a broader capacity claim from three-class fitting at a permissive readiness threshold.

The clinical result remained unresolved or unfavorable. Direct had a large average loss advantage over this revision's DenseNet, but the signs disagreed between seeds. One poor DenseNet seed drove the mean. The adjusted patient interval crossed zero despite an ordinary 95-percent interval below zero, so the declared R50 criterion failed. Proper JEPA was slightly worse than direct in both seeds, and its point advantage over shuffled targets did not clear uncertainty. The working classifier did not reveal an accepted JEPA benefit.

The simpler prevalence constant had lower point primary loss than every neural arm. Offset-only neural probabilities looked better than the declared temperature-calibrated outputs, but choosing them now would be another scoring-driven revision. Temperature cannot explain the PD-only argmax of direct and shuffled after offsets, because positive scalar scaling preserves that argmax. Proper seed 42 recalled some PR and DenseNet seed 42 recalled minority cases, but behavior was not stable across seeds or classes. One CR example cannot establish CR capability.

Fitting CE improved while selection losses rose; selected checkpoints remained early. The next unresolved research need is reliable generalization evidence and adequate class support, not another automatic rate or head search on these same ten patients. Preserve v1, v2, and v3 outcomes and their adaptive history. No continuation or new clinical model is selected by this closure.

## R49 supports a narrow pretraining benefit, not a complete anatomy model

[R49](EXPERIMENTS.md#r49-patient-held-out-anatomy-pretraining-comparison-october-31-2026) moves beyond the two-example R48 fitting check. Under matched architecture, supervision, patient draws, and selection, pretrained initialization improved held-out-development nonempty Dice by about 0.040. Both fitted seeds favored pretraining, and the adjusted patient interval excluded zero. Native-volume retention met the declared allowance, but the volume-improvement interval crossed zero. This supports the selected recipe's pretraining benefit, not every claim about medical pretraining.

The absolute Dice of about 0.198 remains poor. The fixed renders show broad masks, and absent compartments still produce false foreground. This finding cannot validate biological anatomy or clinical deployment. It also cannot establish that the pretrained backbone or this decoder is the best architecture. R49 did not fit a faithful supervised clinical comparator; the separately selected R50 addresses an adapted response-assessment comparison, not anatomy segmentation.

The learning diagnosis differs from R48's short-budget failure. R49's training loss kept falling after selection performance peaked, and every selected checkpoint preceded the final epoch. More updates were not a demonstrated remedy for the held-out gap. The evidence is consistent with overfitting under the matched recipe. It does not prove a universal data ceiling or justify another scoring-driven hyperparameter search.

The new fitting-only mean mask is not the R48 union control. It never saw scoring targets and was empty at threshold 0.5. The substantive result comes from the matched pretrained-versus-random comparison, not merely from beating that empty predictor. Similarly, intensity normalization largely cancels the intensity stress, and the resolution companion removes real detail. Neither establishes learned nuisance separation.

R49 closes one development comparison. Source-patient reuse, small scoring support, two fitted seeds, unverified biological targets, and unaudited pretraining-corpus overlap limit its reach. Clinical response assessment, JEPA objective attribution, and fresh independent validation remain separate questions.

## R48 separates learning readiness from pretraining value

The [source-only learning probe](EXPERIMENTS.md#r48-source-only-comparator-readiness-october-31-2026) learned the two training volumes with both pretrained and random ViTs. Neither learned the sparse compartment masks well enough to pass the declared sanity criterion. Both produced oversized nonempty masks and fell below the same-example constant mean-mask Dice. This is narrower than R42's empty-mask result and still insufficient to launch a substantive spatial pretraining comparison under this recipe.

The target census matters. Each compartment contains 20 to 96 positive voxels on a 24-cubed target. Encoder updates and loss reduction alone therefore cannot certify useful local anatomy. The available counts locate the error but do not identify the cause. Coarse patch representation, decoder behavior, training duration, and loss weighting remain competing explanations. No further tuning followed the fixed failed criterion. The tiny source fit also cannot tell whether pretraining improves held-out learning.

A separate premise needed correction before implementing the supervised control. Matoso's reported best model is 3D DenseNet264 without pretraining, not a pretrained 2D ResNet. The historical proposal combined the legitimate need for a faithful comparator with a separate pretraining hypothesis. Keep those questions separate. Correct patient-level evaluation and inner checkpoint selection even when adapting a published recipe. Do not reproduce test-set selection merely to match the original implementation.

At the original R48 stage, priority 1 remained open. R49 through R51 subsequently completed the selected held-out-development, clinical-comparator, and objective-attribution studies. R48 remains evidence about the readiness of one instrument, not a negative verdict on medical pretraining or JEPA.

### R48 follow-up corrects the interpretation of the short learning test

The operator-requested [diagnostic follow-up](EXPERIMENTS.md#r48-diagnostic-follow-up-the-128-update-test-stopped-before-mask-fitting-was-adequate) isolates one factor: more updates from the unchanged step-128 model and optimizer. Both arms crossed the original fitting criterion by step 512. The fixed first run genuinely failed its declared criterion, but it stopped before spatial fitting was adequate. That result cannot justify saying that the recipe cannot learn anatomy.

The immediate error was broad foreground around correctly included lesions. Volume prediction had become accurate while background probabilities and false-positive masks still needed optimization. The Dice loss value alone hid its smaller shared-state gradient relative to BCE at the inspected checkpoint. Nonzero gradients, valid checkpoint replay, and local derivatives do not determine a unique cause, but unchanged continuation rules out an absolute spatial-capacity explanation for failure on these examples.

The fitted constant control also needs its proper interpretation. With only two binary training masks, thresholding their mean at an inclusive 0.5 returns their union. Its 0.678406 training Dice does not demonstrate useful anatomy estimation on a new image. Even at step 512, pretrained Dice remains below this control, while random Dice exceeds it. These are fitting diagnostics, not a claim that random initialization generalizes better.

Future comparisons need enough training to test the selected capability, with learning curves and source-only validation. An arbitrary short update count cannot establish a data ceiling, an architecture failure, or lack of pretraining value. Preserve the failed first result and separate the successful continuation diagnostic. Do not reinterpret the latter as completed priority 1 or extend this explanation to R42 without its own matched evidence.

## R47 narrows the anatomical measurement lead

The [R47 robustness study](EXPERIMENTS.md#r47-anatomical-signal-robustness-and-locked-transfer-october-31-2026) does not confirm an incremental longitudinal or observed-change capability. Current state with matched timing still beat persistence within LUMIERE. Adding measured history reduced point CRPS by about 2.2 percent, but its interval crossed zero and it missed the declared five-percent margin. Most of the persistence-relative gain therefore did not require previous anatomical values. A useful current-state forecast is a narrower result than learned dynamics.

The locked SAILOR result is an important boundary. Without fitting or recalibration on SAILOR, both learned current-state and history models were worse than persistence, and added history was worse than the matched current-state model. The failure also appeared without a realized query horizon. The evidence is consistent with cohort-specific forecasting or calibration rather than a portable dynamics claim, but it does not identify which acquisition, segmentation, clinical, or sampling differences caused the shift. SAILOR is reused developmental evidence, not untouched final validation.

For observed clinical response, paired volume features did not improve log loss over current volume, with or without the supplied previous rating. The strongest clinical comparator by point log loss was previous rating plus current volume, not the added-change model. The earlier 0.78 AUROC does not establish an incremental change biomarker. It came from a smaller, reused slice and did not resolve this stronger-control question.

Cross-fitted ranking metrics also need care. The source-prevalence null had AUROC 0.5 inside every fold but below 0.5 after pooling predictions from folds with different fitted prevalences. That is a cross-fold probability-ranking effect, not useful inverse clinical signal. Proper out-of-fold log loss stayed primary, and the diagnostic was recorded rather than used to choose a replacement endpoint. Synthetic measurement jitter did not reverse the main point contrasts, but it cannot validate mask truth or clinical deployment.

The appropriate conclusion is not that anatomy contains no information. It is that the supported lead is narrower and less portable than the initial exploratory scores suggested. Further positive claims need independent outcome evidence or a separately justified protocol, not another tuning pass on the same cohorts.

## What the pre-audit revisit does and does not reopen

The [completed R41 through R46 program](EXPERIMENTS.md#pre-audit-revisit-completed-pilots-october-31-2026) did not establish the selected incremental benefits. This closes the fixed pilots, not the research questions generally. Native resolution made image detail visible and the cutoff-fixed fields retained current labels, but neither the crop nor preceding MRI established added forecast value. The strongest anatomical control used measured current and preceding state. Comparing its score with earlier MRI-only experiments would conflate observation quality with dynamics.

The pretraining trial has a specific limitation. All four ViT current-mask readouts predicted empty compartments across the primary cohort after the six-epoch recipe. Its repeated nonzero Dice values came from empty targets, not learned anatomical contours. Encoder gradients, changed weights, and exact checkpoint replay prove execution but do not prove successful representation learning. The valid forecast-score comparison is therefore narrow. It cannot falsify pretraining value generally or establish a useful spatial state. No assessment-driven repair was launched.

Continuous-time integration also did not automatically produce better dynamics. On measured states, the tested vector field was worse than the matched discrete transition at one and two observed steps. Its superiority to unstable constant-velocity extrapolation is not the relevant capability claim. The postoperative comparator remained inconclusive. SAILOR regional perfusion had a worse point score with an interval crossing zero, and scanner fields worsened the fitted LUMIERE predictor. These are outcomes of the declared representations and fitting rules, not evidence that postoperative biology, physiology, or acquisition context is irrelevant.

The nominal-time restriction also matters. A continuous transition is the identity at a zero week-bin interval, while two ordered scans in the same bin can contain a real postoperative change. That formulation cannot recover the missing subweek timing. This bounds the physical interpretation; it does not establish that timing quantization caused the measured difference.

The observed-response pilot shows why a prevalence null matters. Paired MRI had lower point log loss than the poorly performing separate-state arm, but did not establish that contrast and had worse point log loss than a constant source-PD predictor. Its AUROC was below 0.5 for both seeds. A small loss alone cannot support useful clinical assessment in this imbalanced slice. The mask-assisted volume-change classifier had the highest point ranking score, but this reused, rare-class-poor cohort does not support a clinical generalization claim.

Prognosis and dose were not forced into the program with invented event flags or transforms. Their definition blocks remain real outcomes of the selected investigation. Further work requires a new justified protocol or the missing semantics, not an automatic rescue sweep on these assessment patients.

The operator selected all eight earlier threads for sequential investigation. The [first feasibility pass](EXPERIMENTS.md#pre-audit-revisit-feasibility-first-pass) supports bounded new comparisons, not automatic repetition of closed sweeps. R40's completed trainable spatial-horizon comparison still did not establish added history benefit. Its outcome does not identify a representation bottleneck or prove that anatomical history is useless.

The preceding conversational inventory overstated several historical claims. R34 demonstrated a serious coordinate defect in sampled LUMIERE masks, not that every legacy crop was in the wrong hemisphere. Its corrected record already contains an aligned crop runtime smoke. The old lesion encoder also extracted patch tokens separately by modality, so those operations were not wholly absent. R35 and R36 anatomical proxy findings did not establish clinical-response accuracy. Averaging modality features is a modeling choice to compare, not proof that useful contrast was erased or that it caused a negative result.

The important unresolved crop question is whether localization before whole-brain downsampling improves the matched anatomical task. A crop from an already downsampled `96³` tensor cannot recover discarded image detail. A mask-assisted crop also changes the information available to the predictor, so persistence and simple controls must receive the same cutoff masks. Future masks cannot select the predictor's field of view.

The prognosis and SAILOR branches need narrower definitions before model fitting. A duration without verified event status is not automatically an uncensored survival outcome. Concurrent rationale text can reveal the response label. A subject-level dose filename does not establish that a planned dose was known at every scan. These limits bound the selected tests. They are not an open-ended demand to audit the entire dataset again.

The October 31 continuation now has real-task smoke evidence for the selected forecasting, binary observed-response, and quantitative perfusion instruments. That answers whether the declared computations can run, not whether their hypotheses are supported. The inspected fixed-field source render also shows why improved image detail is not enough. Automated contours can extend beyond visible MRI foreground. A matched crop comparison can measure agreement with that proxy task, but it cannot certify biological anatomy. The interrupted full preparation is an input-transfer issue, not a negative model finding or permission to reduce the cohort.

## R38 joint training did not beat detached observation state

October 30, 2026. R37 froze a pre-trained spatial encoder and found no added benefit from multi-visit history over latest-scan-only prediction. R38 tested the most natural architectural objection to R37: perhaps the frozen encoder had discarded subtle temporal precursors, so training the 3D spatial state encoder *jointly* with the recurrent dynamics head (`JH`, `JL`) would learn predictive features that a detached observation-supervised encoder (`OH`, `OL`) misses.

The completed three-seed 2x2 factorial result ([R38 findings](EXPERIMENTS.md#r38-completed), [aggregate](r38_full_metrics.json)) did not establish the proposed joint-training benefit under this recipe (`INCONCLUSIVE_FOR_FULL_CAPABILITY`, with zero of fourteen capability flags met):

1. **Joint training did not establish a forecast gain.** On the 75-pair primary cohort, `JH` (CRPS 1.3690, MAE 1.8017) had slightly higher point error than detached `OH` (CRPS 1.3487, MAE 1.7818), and `JL` (CRPS 1.3672) had slightly higher point error than detached `OL` (CRPS 1.3295). The joint-versus-detached contrasts (`JH - OH = +0.0203`, `JL - OL = +0.0377`) and 2x2 interaction (`-0.0174`) had intervals crossing zero.
2. **Joint forecast gradients degraded current-visit state fidelity.** Because `OH` and `OL` detach the forecast head from the encoder, their current-scan decoder learns purely from cutoff-visit supervision (current MAE 1.4733, current Dice 0.3055). When `JH` and `JL` backpropagate noisy next-visit forecast errors into the shared 3D encoder, current-visit MAE worsens to 1.5397–1.5414 and current Dice drops to 0.2840–0.2872, failing the prespecified state-retention noninferiority gate (`current_state_retention.passes = false`).
3. **Multi-visit history still did not improve over latest-scan-only.** Both `JH - JL` (`+0.0018`) and `OH - OL` (`+0.0192`) were slightly positive (favoring latest-only at the point estimate) with wide intervals crossing zero. Simple non-neural baselines (`E` at 1.2226, `P` at 1.2691, `SSM` at 1.2737) continued to match or beat all four neural arms on next-visit volume CRPS.
4. **Resolution sensitivity remained.** All four arms were stable under intensity scaling and shifting because of nonzero z-score standardization, but 2x through-plane resolution degradation worsened CRPS by 12–17% in necrotic and enhancing compartments across both joint and detached arms. This establishes sensitivity to the specified perturbation, not a unique cause of forecast failure.

Together, R37 through R39 show that joint encoder training and coarse current treatment-phase inputs did not resolve the tested forecasting problem. R38 also failed the specified current-state retention requirement. These results weaken those proposed remedies under their fixed recipes. They do not identify a unique bottleneck or establish noisy-target overfitting as the cause. Better temporal representations and richer cutoff-valid intervention records remain different questions.

## R39 did not establish added treatment information

October 30, 2026. The operator selected treatment conditioning after a brainstorming recommendation. That recommendation overstated what was known. Missing treatment inputs were a plausible explanation for forecasting difficulty, not an established root cause. Action conditioning is not required for every predictive world-model formulation. LUMIERE's available metadata does not provide the systematic treatment course table the recommendation assumed, and R38 cannot establish a fundamental limit of action-blind forecasting.

The [R39 comparison](EXPERIMENTS.md#r39-treatment-status-forecasting-october-30-2026) instead tested the current CRT, TMZ, no-treatment and unknown labels documented in SAILOR's dataset descriptor. It used frozen MRI-derived anatomical states and small transition models. The primary comparison found no established improvement from adding treatment to history. A modest latest-state-only point improvement also remained uncertain. This closes the prespecified feasibility test without a positive treatment-conditioning claim.

Prediction error relative to measured-current-volume persistence was larger than the differences among treatment/history models. That makes accurate state estimation and cross-cohort calibration important unresolved issues. It does not prove that they caused R39's outcome. The reduction relative to MRI-state persistence can include calibration learned from the SAILOR training folds, rather than learned future dynamics.

For this direction to progress, the next scientific choice is whether current coarse phase is an adequate action description. Planned interventions with decision timestamps would answer a different question than retrospective per-scan status. The present data cannot support treatment counterfactuals, and the results do not justify launching a larger treatment-conditioned spatial model automatically. A new proposal should explain which information or modeling change makes the question distinguishable from this completed test. R38's three-seed factorial comparison subsequently completed and is interpreted above.

<a id="research-objective"></a>

## Research objective and interpretation

The goal is to represent what an MRI shows about anatomy and disease, and how those features change across visits. The representation must also distinguish disease-related differences from differences caused by acquisition, such as scanner settings or image intensity.

A representation is the set of numerical features that an encoder extracts from a scan. We want those features to retain useful information. We do not want to call a model useful merely because it produces a low training loss.

JEPA, lesion-focused features, forecast heads, clinical inputs, and whole-brain methods are possible ways to reach the goal. None is the goal itself. The [research rules](documentation_sources/versions/6e710be8c33b05d77bf8f13b415b64028fcc13dc03d71defb304e6a7631c78f4.md) and [decisions D55 and D57](documentation_sources/versions/2b7293c884e0e0aa9f1db178d13f55497d584cd7060c68c022436d47330bb63c.md) make that distinction explicit.

### R37 asks whether spatial history predicts an unseen observation

The October 19, 2026, operator request selects the bounded [R37 pilot](METHODS.md#r37). At that protocol-first status, this was a hypothesis, not a measured result. It freezes an existing state-only encoder and tests whether history improves the next observation's anatomical probability distribution beyond a matched latest-scan predictor and simple controls. The latest-image-only control retains identical past timing, history masks, and availability flags. The pooled-past control retains the full spatial current-scan pathway. These controls separate earlier MRI information from observation scheduling and earlier spatial detail from current-scan processing. Regularized linear spatial dynamics tests whether nonlinear recurrence earns its complexity. No contrast alone establishes anatomical correspondence or a physical change mechanism. Prior latent forecasts and physical-history failures motivate these controls, not a presumption of success.

A history gain would support this tested temporal-state capability. A latest-scan-only gain would support prediction from current state without establishing history benefit. A negative or inconclusive outcome ends this formulation within its budget. Automated volume agreement, reused development patients, and unknown future visit timing limit all three interpretations. The pilot does not reopen R36's encoder-objective findings or demonstrate biological, fixed-day, or treatment-effect prediction.

### R37 completed without an established history benefit

The [October 26 result](EXPERIMENTS.md#r37-full-pilot) is inconclusive on 75 pairs from 18 reused LUMIERE development patients. H-minus-L CRPS is +0.00334486, with adjusted 99% interval [-0.04732935, +0.05234837] in native `log1p(mm3)` coordinates. Scores, not predictions, were averaged across seeds 42/43/44 before patient-uniform aggregation and bootstrap. All five primary intervals cross zero. The matched current-image predictor, persistence, empirical-change distribution and linear dynamics prevent a forecast from earning credit merely for correcting encoder bias or using scheduling information. Neither the history hypothesis nor the secondary latest-state benefit was established.

This result does not say that MRI history is useless. It says that this fixed encoder/predictor/decoder formulation did not establish added benefit conditional on the supplied observation-history covariates. Nor does the unresolved H-minus-G contrast identify whether earlier anatomical correspondence matters. A frozen volume head can miss predictable spatial changes, and approximate visit gaps and unknown interventions limit what next-observed prediction can mean. These limits narrow the claim; they are not reasons to tune on the same assessment patients.

The other requirements matter even if a CRPS point difference looks favorable. The MAE bounds do not establish the required noninferiority, enhancing-compartment coverage misses the point target, and half-resolution perturbation worsens both H and L. The near-zero intensity response follows largely from standardization, not learned nuisance separation. The [result review](documentation_sources/updates/2026-10-26/reviews/result-v1.md) independently verified numerical reporting and bounded fixed-weight predictions, not biological truth or a full training replay.

The one-analysis stopping rule now closes this formulation. Remaining compute is not permission for a rescue sweep. R35's state/observed-change findings and R36's unresolved frozen-encoder objective comparison remain separate evidence. R37 is not a demonstrated predictive world model or a new publication contribution simply because all its computation completed.

### R37 observation noise changes the score only slightly

The October 19, 2026, [E-only ablation](EXPERIMENTS.md#r37-noise-ablation) compared the same source-fitted empirical changes with and without Gaussian observation noise. On 75 history-eligible pairs from 18 reused development patients, CRPS fell from 1.2230516797 to 1.2226323341, a 0.0342868% reduction. Both conditions selected the same multiplier on validation. The paired patient-bootstrap interval excludes zero, but the magnitude is tiny. That does not establish useful anatomical improvement or better uncertainty calibration.

The mean prediction and source changes are identical at this common multiplier. The score difference therefore tests the distribution's smoothing, not a better anatomical mean or learned dynamics. E does not consume earlier MRI history; restricting its assessment to history-eligible examples does not make it a history model. No conclusion about H versus L, spatial recurrence, or a world-model capability follows. Retain both reported conditions rather than promoting a winner from reused assessment patients. The full pilot was incomplete at the October 19 ablation receipt; the later full-pilot result above supersedes only that execution status.

### State, observed change, and future prediction are different questions

State means what is present at one visit, such as lesion burden or anatomical structure. A model that extracts lesion volume from a scan estimates state.

Observed change means the difference between two scans that we already have. A model can compare those scans without predicting anything about an unseen future.

Future prediction uses information available at the earlier visit to predict a later observation. A model cannot use the later image, mask, volume, or visit gap as an input and still count as a past-only forecast.

Acquisition nuisance is variation that is not the target disease signal. A different scanner setting may change an image even when the anatomy does not change. A useful change representation should not interpret every such difference as disease progression.

These distinctions matter because success on one task does not establish success on another. A model may estimate current anatomy well but fail to predict the next visit. A model may compare two observed scans well but add nothing beyond subtracting their separate state estimates.

The records sometimes use "causal" to mean that model inputs exclude future information. That is a restriction on information available at prediction time. It does not establish that treatment caused an outcome or reveal what would happen under a different treatment.

### A simple baseline sets a real requirement

Persistence predicts that the next measurement will equal the current measurement. For change, the equivalent baseline predicts zero change.

That baseline can be hard to beat when most measurements change little, visits are irregular, or measurement error is large. In the scoped lesion task, persistence tied or beat several learned forecasts. Those results are evidence against a gain for the tested methods and cohorts. They do not prove that useful MRI representations are impossible.

The original [proposal](documentation_sources/versions/e82ede5e41b034398b599f1f141ffa04335d47ad7d8e66761540a71ce2f72aaa.md) suggested predicting future latent features instead of pixels. A latent feature is an internal numerical feature of the model. That proposal was a hypothesis, not a demonstrated advantage over persistence.

### The strongest positive lead is state estimation

R35 found lower in-domain automated-volume error with its spatial state model. My interpretation is that separate state estimates deserve at least as much attention as a specialized network for paired scans.

The dedicated paired head did not establish an advantage over subtracting independent state estimates. That does not make the paired head useless for every task. It means that the tested head did not show the additional value needed for the stated claim.

The learned spatial model also received dense mask supervision, unlike the frozen scalar-volume control. The comparison therefore cannot identify architecture alone as the cause of the improvement. The [R35 results](EXPERIMENTS.md#r35) retain the measurements and the unsuccessful transfer controls.

### Sparse change motivates R36, but does not explain earlier failures

R36's target audit found that the median changed region occupied less than 0.5% of the grid in each compartment. One native enhancing disagreement disappeared completely after resizing to 96 cubed.

A whole-grid loss gives most of its weight to unchanged voxels because they are far more numerous. R36 tests whether giving changed and unchanged regions equal weight helps the encoder preserve change information.

That was a testable hypothesis. The completed comparison improves the descriptive nonlinear decoder but does not verify the primary frozen-encoder gain. The audit does not prove that sparse change caused earlier failures or that most change disappeared during resizing. The [R36 execution record](documentation_sources/versions/51d6523f0da4b3d8745af10b79318fd684aba1fd9b47cc7d321e30df79984551.md) supplies the audit evidence.

### A measurement pipeline is useful even when forecasting fails

The project recovered masks, checked native volumes, and repaired coordinate alignment. Those steps make later comparisons possible and expose invalid earlier measurements.

However, a mask on the correct grid can still be inaccurate. Correct volume arithmetic does not establish segmentation repeatability, correspondence between anatomical locations, or biological change. Those questions need different evidence, such as repeat annotations or expert review.

The [geometry and chronology record](documentation_sources/versions/7bc218e8ef5fd2fdb22bc0ed7f5bab72c4b1d099f1f7f2c3145f6c3bca6f3472.md), [P0 and P1 results](documentation_sources/versions/89605d7963ba3e0f942e5feb416846a32faea82a277f1dbe008b0cd0a6b86068.md), and [D56](documentation_sources/versions/2b7293c884e0e0aa9f1db178d13f55497d584cd7060c68c022436d47330bb63c.md) separate these claims.

### Plausible limits are not established causes

Sparse change, unreliable timing, segmentation error, scanner differences, unstable optimization, and weak supervision may limit the models. Some have direct audit evidence in particular runs. None explains every failure without a comparison that isolates that cause.

For example, uncertain visit timing limits claims about exact forecast horizons. It does not by itself explain a model's forecast error. Likewise, worse transfer performance does not tell us whether the cause is the representation, preprocessing, or differences in labels.

R35 includes explicit acquisition-perturbation tests. K3 and earlier cross-site work also identify intensity, MRI-sequence, preprocessing, and calibration concerns. The [K3 audit](documentation_sources/versions/7d63fd3f44c1ed975ea9d1b860b597592902c503fbb3124145a01db17ee996dc.md) records those concerns without establishing one universal transfer mechanism.

<a id="current-evidence"></a>

## Corrections, withdrawals, and current evidence

### Comparing errors requires the same target space

The early claim that JEPA beat persistence by 2.7-fold compared errors in different latent spaces. The two error values were therefore not directly comparable. D28 and A8 corrected the comparison by scoring both methods in the same space.

The early count of 82 wins among 91 patients also changed to 41 among 91 under the corrected comparison. A24 added paired confidence intervals. The pooled test difference remained unresolved, while the patient-uniform test difference was resolved on a reused development split. The broad statement "JEPA beats persistence" is not supported.

Pooled scoring gives more weight to patients with more observations. Patient-uniform scoring gives each patient equal weight. A result under one rule does not automatically hold under the other. The [historical result record](documentation_sources/versions/89605d7963ba3e0f942e5feb416846a32faea82a277f1dbe008b0cd0a6b86068.md) retains the measurements behind A8, A24, K3-2, and K3-11.

### A repeatedly examined test set becomes development data

Legacy cross-validation over all 91 patients included patients used to train the encoder. K3-16 identified that overlap. It cannot answer how the representation performs on patients unseen by the encoder.

The locked protocol fixes this particular overlap by using encoder-unseen patients. It does not restore an untouched test set. Both the historical final13 and the broader unseen cohort have been examined repeatedly. They now provide development evidence.

Readout seeds, model widths, and folds also change the reported scores. A favorable single split does not establish a robust RANO advantage. The task, cache version, patient set, and evaluation protocol must accompany a score.

### Future inputs invalidated the learned P2.4 forecasts

A40 found target-visit images, masks, and gaps ending at the target visit in the learned forecaster's inputs. The model had access to information unavailable at the earlier visit.

Its apparent forecast gains, seed stability, and adaptation results are withdrawn. They are invalid as evidence of past-only prediction, rather than merely weaker evidence. The old pair builder also skipped visits with missing masks, producing 123 rows instead of the legitimate 117.

The later history-only rerun was approximately tied with persistence. It did not replace the withdrawn result with a success. A41's clean gap and velocity head reruns are separate latent-feature diagnostics, not proof of prospective physical-volume forecasting.

The full history remains in [A40 and A41](documentation_sources/versions/eaca551880e0a9db5c4bdc7785e135675cd518ebfcd361d5aa482beca4f37dce.md), [P2 execution](documentation_sources/versions/9ea08b90cf4971fb7b85d4b150ef1317994f42aa2fca2f1c9c0f5de5424e38e4.md), and [D53](documentation_sources/versions/2b7293c884e0e0aa9f1db178d13f55497d584cd7060c68c022436d47330bb63c.md). These withdrawals apply to the tested methods and tasks, not to representation learning as a whole.

### Transfer to SAILOR is informative, but not independent confirmation

SAILOR differs from LUMIERE in acquisition and segmentation source. Performance on SAILOR tests transfer under those differences, but repeated use for development prevents an untouched external-validation claim.

Its RANO code mapping is empirical and lacks independent codebook confirmation. That limits clinical RANO conclusions. The anatomical proxy task avoids using those codes as its primary target, but still depends on mask definitions and measurement reliability.

SAILOR is controlled-access data. It remains local. The [Methods reference](METHODS.md) defines the source-trained transfer protocols and privacy limits.

### Completed observed-change experiments are not forecasts

R36 Phase 1 computation and scoped reporting review are now complete. Its [current interpretation](#d58) replaces the old launch and reporting tasks below. It does not establish a frozen-encoder gain or future predictability.

R35 completed its corrected state, observed-change, and nuisance comparison. Its [independent result review](reviews/r35_result_review.md) approved scoped reporting on October 13, 2026, after supplemental executable checks corrected the audit's overclaims. It did not show that a paired head beats independent state differences, and it did not test future prediction. The [closure receipt](reviews/r35_closure_receipt.json) preserves the provenance and target-reliability limits. Closing the review does not turn automated-volume proxy improvement into biological validation or architecture-only superiority.

At the October 11, 18:30:51 UTC snapshot, R36 had all twelve final-epoch training checkpoints and ten seed-42 readouts locally validated. Seed 43 was running, and seed 44 had not started. No model-quality comparisons on the evaluation patients had been scored.

A completed implementation review does not count as design approval before training or acceptance of scientific results. The remaining reporting concerns were correct identity checks for nuisance reports and correct classification of harmful tradeoffs. Phase 2 was not authorized.

The [R36 record](documentation_sources/versions/51d6523f0da4b3d8745af10b79318fd684aba1fd9b47cc7d321e30df79984551.md) and [handoff](documentation_sources/versions/8fc1116f80667885774a5ce6a8fb9ec40f1e360d1f87867fefb81baf01118ab6.md) preserve that history. Neither completed training nor a partial arm establishes a quality result. The old statement that delegation was unavailable is also historical, not current operating guidance.

## Audits, alternatives, and methodological lessons

### Published scores need matching tasks before comparison

The literature audits describe different prediction problems. Some methods assess scans already observed. Others generate future images or classify response. Those tasks cannot share a performance threshold without matching their inputs, patients, labels, and splits.

TaDiff-Net reports treatment-conditioned image generation and segmentation. Its LUMIERE evaluation uses external image metrics, not RANO classification. Tikhonov's reported assessment macro-F1 is about 0.50, and TRACE's is about 0.477.

Matoso's reported median 0.51 is balanced accuracy, not macro-F1. The later survey records Amato's 0.616 ± 0.048 as a secondary reported value without a fully checked primary protocol. It records Maurya's 0.6415 development and 0.5118 hidden-test balanced accuracies, not macro-F1.

There is also an attribution conflict. The P1 audit assigns Maurya's entry the same arXiv identifier that the frontier survey assigns to Matoso. That conflict needs primary-source checking before publication.

October 24, 2026, correction: the [primary version-2 record](https://arxiv.org/abs/2504.18268v2) and author list identify that paper as Matoso and colleagues. The P1 audit's Maurya association is incorrect. This corrects the identifier, not the separately reported Maurya numbers, which remain unverified. The [bounded literature check](documentation_sources/updates/2026-10-24/literature-check.md) records the checked scope without editing either frozen survey.

The [frontier survey](documentation_sources/versions/bd305198a22112d90d28cf638536c3638ff95806a6aeff087655acb8dea287d1.md) and [P1 comparison audit](documentation_sources/versions/77de3ea25aa45d894d9577a96b69195912e6036883dace704af07b89de07843b.md) are project literature records. Their claims were not independently checked for these documents. They do not establish a matched advantage for our representation or a threshold for past-only forecasting.

### New supervision and a better comparison are separate options

One possible next approach is to establish a task-specific simple baseline, then test whether history, paired scans, or learned representations add information. Another is to obtain suitable lesion and change supervision instead of repeatedly optimizing the same self-supervised signal.

The [external-supervision audit](documentation_sources/versions/636f0274c36efa3cee145c530acebff689d5961eff995269ecc3bc91d8c6d89e.md) considers UCSF-ALPTDG for lesion and change supervision, Burdenko for longer trajectories and treatment context, and UCSD-PTGBM for treatment-effect auxiliary labels. D52 prefers UCSF for supervision, but acquisition remains blocked by approval and storage limits.

The dated audit reports 298 paired UCSF patients and a noncommercial data-use agreement requirement. It reports 180 Burdenko patients in a restricted 131.23-GB imaging release. UCSD version 3 has 178 subjects and 243 timepoints in a 44.98-GB release, with corrected patient identifiers.

Those are audit-reported dataset details, not a current access or disk-space check. Patients used to train a teacher cannot later be described as an untouched external test. Neither research option is approved merely because the audit mentions it.

### Valid inputs and measurements come before interpretation

The [validity plan](documentation_sources/versions/e222fabaeb05f4d166a3d220a87398aa55b60e1697d479706b6e87d6bab71212.md) and [validity record](documentation_sources/versions/7bc218e8ef5fd2fdb22bc0ed7f5bab72c4b1d099f1f7f2c3145f6c3bca6f3472.md) require checks on coordinates, visit order, labels, and artifact origins before interpreting scores.

A file can load successfully and still have the wrong meaning. An older cache may contain outdated labels, future information, or incorrectly aligned masks. Numerical compatibility does not establish scientific validity.

The [anatomy-first CLI record](documentation_sources/versions/ad235fe7fd9dbc123dbc829a8242d3c11ce6427b241808ba661ac7ed0fb1db87.md) keeps legacy RANO commands separate from the active task. The [stale-code audit](documentation_sources/versions/eca8302b4c0da4df15f1a8918164254666650e647632c098c025513f6ebe5061.md) records paths with old or unsafe assumptions. Available code is not necessarily a valid current method.

### Synthetic actions would test a proposal, not treatment effects

The [action-conditioned representation brief](documentation_sources/versions/f3dad868d43b3aa2bf5f7913b3dfa2872cdf87a65fd9648b1eb9932e3f750855.md) proposes applying controlled synthetic changes to real MRI. The model would receive the source scan and the declared action, then predict the result.

That work is unrun and unselected. A proposal needs a materially different approach, simple baselines, valid input timing, patient-separated splits, cost, independent checks, and observations that would disprove its hypothesis.

Patient partitioning must happen before synthetic views are generated. Held-out transformation families would test whether the model generalizes beyond the actions seen in training. Geometric actions and physical targets need verified coordinates before resizing. Intensity changes need checks both before and after normalization.

No existing generator with the required action labels was identified. Even a correct synthetic generator would not establish glioma biology or treatment effects.

The earlier [proposal](documentation_sources/versions/e82ede5e41b034398b599f1f141ffa04335d47ad7d8e66761540a71ce2f72aaa.md) considered clinical actions. A RANO response label is not a treatment action. A valid action-conditioned comparison needs reliable action identity and timing, availability at prediction time, and matched no-action and simple temporal baselines.

## Historical identifiers and source records

The decision and audit tables now sit in [Methods](METHODS.md#historical-indexes), where you can look up their definitions without interrupting this explanation. They preserve D0 to D57, D33b, G1 to G20, R1 to R36, V1 to V5, E0 to E4, and K3-1 to K3-35. A proposal label is not evidence that the proposal was run or authorized.

References to I1 to I4 remain incomplete in the captured corpus. These documents do not invent missing standalone definitions:

- A1 associates I1 with the improvement a learned method needs over persistence. The full original lesson is unavailable.
- A5 and the pilot associate I2 with collapse monitors. A falling loss alone does not establish useful learning.
- D6 links the LoRA rationale to I3. That mention does not define a broader lesson.
- D8 links I4 to the corrupt community checkpoint. Matching parameter keys did not guarantee valid model behavior.

The [ASTRA audit](documentation_sources/versions/b72ceb2da7c2ba695a4174a43643847c9fb27244c1785ff1a6acb5c898f2fb29.md) separates measurement validity from proposed experiments. The [K3 audit](documentation_sources/versions/7d63fd3f44c1ed975ea9d1b860b597592902c503fbb3124145a01db17ee996dc.md) records challenges to earlier claims. The [decision log](documentation_sources/versions/2b7293c884e0e0aa9f1db178d13f55497d584cd7060c68c022436d47330bb63c.md) and [directions](documentation_sources/versions/eade325e60cbeeb92dc5f6570520d8f502a519e846309ef5803b2bf7e1a2d094.md) retain the full dates and reasoning.

R33's [clean reruns](documentation_sources/versions/eaca551880e0a9db5c4bdc7785e135675cd518ebfcd361d5aa482beca4f37dce.md), R34's [validity repairs](documentation_sources/versions/7bc218e8ef5fd2fdb22bc0ed7f5bab72c4b1d099f1f7f2c3145f6c3bca6f3472.md), [R35](documentation_sources/versions/768fdfa0a9be4dc76e99fcef9ff971e26252d672e82b08267f5e9ea157ced0f3.md), and [R36](documentation_sources/versions/51d6523f0da4b3d8745af10b79318fd684aba1fd9b47cc7d321e30df79984551.md) are different records. Their results and limits must not be combined into a single success claim.

[Source coverage](SOURCE_COVERAGE.md) lists all captured branch and worktree versions and their dispositions. Research-source links point to fixed repository copies. Those copies retain original wording and embedded links, including outdated claims and links that may no longer work. They are historical evidence, not newly checked operating instructions.

This rewrite does not restore the deleted `notes.md` or `explainer.html`, authorize old proposals, or independently verify the literature.

<a id="d58"></a>

## D58. Keep the R36 decoder finding separate from encoder progression, October 13, 2026

R36's completed assessment does not establish the primary frozen-encoder benefit of balanced change supervision. C retains the specified direct-volume state endpoint relative to A, but that narrow noninferiority result is not universal representation quality. The trained nonlinear heads improve descriptively while the frozen C-versus-B interval crosses zero. Both observations must remain in the record.

D reduces the mean trained-perturbation excess false-change endpoint, but worsens direct-state error and noise sensitivity. The state interval overlaps the allowed margin, so the intended tradeoff is inconclusive. A favorable aggregate cannot erase these negative companions. The affine-intensity stress is largely canceled by standardization and cannot establish learned scanner invariance.

C and D beat matched target-only occupancy probes. Neither establishes superiority over independent decoded-state differences. This limits the claim that a specialized paired representation adds useful observed temporal information. On developmental SAILOR, every frozen difference probe has higher descriptive error than its independent-state control, and every source52 ridge-change mean exceeds zero change.

Failure to establish C's primary gain is not proof that B and C are equivalent or that useful MRI representations are impossible. Passing C's direct-volume retention margin is also not proof that every anatomical capability is retained. The comparisons answer narrower questions than the research objective.

The [completed findings](EXPERIMENTS.md#r36-completed), [independent result review](reviews/r36_phase1_result_review.md), and [closure receipt](reviews/r36_closure_receipt.json) support this interpretation. The separate artifact audit reproduces the bounded classifications. It does not newly replay all dense predictions or establish biological correctness. Reporting approval cannot provide retrospective pre-training approval.

My reading is that the frozen-accessibility hypothesis remains unresolved, while the decoder responds to balanced supervision. Those are different findings. Neither selects a follow-up method family. Any new dynamics experiment needs a separately approved contract and operator decision. Expert target reliability and genuinely independent validation remain unresolved. Phase 2 remains unauthorized.

<a id="workshop-manuscript-feasibility"></a>

## Workshop manuscript feasibility, October 24, 2026

The operator requires an accepted paper at ICWM, ICLR, or ICML and permits a workshop paper. The authorized next action was a bounded literature comparison and main-figure sketch using completed evidence, not new training or submission. The [source check](documentation_sources/updates/2026-10-24/literature-check.md) records nine queries, primary-source read scopes, blocked access, and the [figure specification](documentation_sources/updates/2026-10-24/figure-spec.json). This is the assistant's publication judgment, not an independent novelty review or a new experiment.

The broad state-versus-change story has direct predecessors. [PaIRNet](https://proceedings.mlr.press/v227/kim24a.html) and [LILAC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11873959/) compare paired prediction with differences of separate state estimates. Rudie's post-treatment glioma study compares segmentation-based assessment with dedicated change networks, as verified in its primary abstract; its full text was unavailable to this check. [BrainDiff](https://arxiv.org/html/2609.00593v2#S5.SS2) explicitly probes a frozen brain-MRI encoder for interval change and single-study pathology. [LSOR](https://arxiv.org/html/2310.00213v1) evaluates frozen longitudinal representations. [STRIDE](https://arxiv.org/html/2610.08848v1) already compares single-timepoint and paired glioblastoma inputs. These papers use different endpoints and cohorts; they are context, not matched score targets or reproductions.

The remaining candidate is a narrower empirical question: do gains from change-balanced supervision remain detectable under a matched fixed-capacity readout of the frozen encoder? B averages signed-occupancy error over the whole grid. C averages changed and unchanged voxels separately, then gives the available group means equal weight. Architecture, starting weights, state supervision, volume-difference loss, training pairs and duration remain shared. A cautious working title is **Evaluating change-balanced supervision in longitudinal glioma MRI**. R36's matched C-versus-B objective comparison is the center. The descriptive nonlinear-head improvement, unresolved primary frozen-probe contrast, independent-state controls, state retention, measured nuisance harms, and development-transfer limits must appear together. R35 supplies separately labeled motivation and context, not another arm of one prospectively designed experiment.

The judgment is `CONDITIONAL_GO_FOR_SHORT_WORKSHOP_DRAFT`. The exact controlled comparison may support an empirical extension, but this bounded search does not establish priority or sufficient publication-level novelty. Do not claim the first MRI pair model, state-subtraction control, frozen change probe, or general distinction between state and change. Do not turn an unresolved probe interval into equivalence, absence of change information, or proof that only the decoder improved. No new world-model capability follows from this paper candidate.

No additional fitting is needed to draft from the approved scoped findings. A faithful published-system reproduction is absent; a matched superiority claim would require separate approved work. A full Rudie methods read, human scientific assessment of the contribution, the specific workshop call, and whether non-archival acceptance satisfies the obligation remain bounded pre-submission decisions. The official ICLR short-track guidance permits modest self-contained contributions but does not approve this manuscript. The sketch is not rendered or pixel-reviewed. Existing R35/R36 result approvals do not cover novelty, workshop fit, acceptance, or newly written text. At the October 24 feasibility check, R37 was a separate incomplete forecasting experiment, not a prerequisite that must succeed to make these observations reportable.

October 26, 2026, operator update: archival acceptance is preferred; a non-archival workshop is a fallback requiring agreement. The operator selected full completion of the already fixed R37 pilot as the immediate priority and approved a bounded continuation over SSH on `mosaic-gpu`. The [continuation record](documentation_sources/updates/2026-10-26/r37-continuation.md) preserves the additional allowance, prior expenditure, primary E choice, approximate historical timing and pre-fitting review gates. This changes execution priority, not the completed R35/R36 findings or the conditional nature of the manuscript assessment. R37 must retain its matched latest-scan, simple and linear controls and report a negative or inconclusive outcome without a rescue sweep. Archival preference does not make a positive forecasting result inevitable or establish publication novelty.

October 26, 2026, closure: R37 has now completed with independent approval for scoped reporting of its inconclusive outcome. This adds a bounded forecasting result, not the missing world-model success or a first-of-kind claim. The earlier manuscript candidate remains conditional on a clear contribution, human scientific review and a confirmed archival route. No manuscript has been drafted or submitted, and no acceptance has been obtained. A positive R37 result was never required to report the completed R35/R36 findings. Any new experiment needs its own capability, falsifiable hypothesis, fixed budget and operator decision.

November 1 closeout update: the [workshop manuscript source](workshop_manuscript.typ) and [compiled PDF](workshop_manuscript.pdf) now exist, centered on R35 through R39. The October 26 statement that no manuscript had been drafted is historical. This documentation closeout does not certify the draft's scientific completeness or record a submission or acceptance. The [R62 summary](RESEARCH_SUMMARY.md#10-closeout-and-publication-position) gives the broader current evidence boundary.

### Proposed short-paper outline

1. **Question.** Ask how the measured benefit of change-balanced supervision depends on the readout. Do not assume that a better trained output demonstrates a better frozen representation.
2. **Related work.** Position the study after PaIRNet/LILAC, Rudie, BrainDiff, LSOR, STRIDE and TRACE. Identify the matched objective comparison as the empirical extension, not a new principle or a reproduced published benchmark.
3. **Controlled design.** Define B and C, their shared architecture and fitting budget, automated signed-occupancy targets, frozen linear probe, patient-uniform aggregation, and locked uncertainty. Include A, D, original R35 and independent-state controls. Label the original R35 study separately.
4. **Results.** Lead with the inconclusive primary frozen-probe contrast. Then show the descriptive trained-head improvement, incremental-information controls, state retention and nuisance harms, empty-compartment errors, and all-arm development transfer. Use the sketched main figure and complete companion tables.
5. **Interpretation and limits.** State what this readout comparison establishes and what remains unresolved. Explain why observation-encoding evidence matters to MRI world models without calling it learned future dynamics. Retain target, provenance, cohort reuse, seed uncertainty and biological-validation limits.
6. **Reproduction and disclosure.** Bind the approved result artifacts and fitting recipes. Disclose absent published-model reproductions and AI assistance, with a human author responsible for the scientific claims. Adapt length and format only after selecting a confirmed workshop call.


## Better MU observation anatomy did not resolve forecasting, November 7, 2026

The [completed current-only adaptation](EXPERIMENTS.md#mu-cavity-state-current-only-forecasting-adaptation-november-7-2026) separates present-state improvement from future-state capability. It improves volume CRPS over new-state persistence and has a positive 24-week mean-Dice difference, but core overlap among nonempty targets collapses while correctly empty predictions improve. Narrow core intervals miss almost every nonempty outcome. Cavity overlap is also lower at the point estimate, and spatial volume-distribution accuracy does not clearly beat the simple scalar predictor.

These results do not establish improvement over the preserved current-only forecaster. They also do not isolate a cause: explicit cavity targets, full annotation support, MU-only fitting and an expanded transition output accompany the changed observation state. Better observed core estimation was a justified reason to attempt this adaptation, not a guarantee that its transition objective would preserve present-core trajectories. Keep the improved observation system and fixed references. No additional architecture change, extension, history fit or reserved assessment has been launched on the basis of these results.


## Why the closed spatial recipe warrants a different probability model, November 8, 2026

The matched nonempty-core subgroup sharpens the result: the scalar predictor can assign nonzero outcomes and materially better proper scores using the same available observations. That weakens a data-only explanation for the spatial model's confidently wrong predictions. It does not establish that existing data are adequate for general forecasting, that the scalar intervals are calibrated, or that the scalar model improves on persistence.

The observed failure is suppression of present core accompanied by improved empty-compartment and aggregate scores. The current training/selection criteria permit this tradeoff. Scenario parameterization, loss balance, surrogate gradients and optimization remain possible contributors; these results do not isolate their causal roles. More longitudinal data may help, but are not an established repair for this specific failure.

The next formulation should make presence and amount explicit random variables and make location conditional on them. An amount predictor that assigns positive mass to nonzero core must not be silently converted into empty maps by spatial decoding. Enforcing that contract addresses a demonstrable interface failure; it does not guarantee correct amounts, shapes, calibration or future prediction. Keep those claims separate when judging the redesigned system.


## Amount-first quantity gate, November 8, 2026

The [amount-first build](patient_amount_first_metrics.json) separates quantity failure from the spatial decoder: no learned-location training was needed to reject the terminal quantity candidate under the fixed rules. Its lower 24-week nonempty-core mean-volume error coexists with worse CRPS and wider intervals, and the paired MAE interval crosses zero. Presence predictions are close to fitting prevalence. Exact feasible sampling and conservation are working implementation properties; neither establishes useful forecasting. This result does not isolate the probability family, shared backbone, likelihood optimization or available supervision as the cause. The bounded run is complete without an extension or new redesign.


## Interpretation of saved amount-first quantities, November 8, 2026

The [saved-distribution analysis](patient_amount_first_interpretation.json) shows that reported core means are driven by a few large draws while presence probabilities barely distinguish future-present from future-empty cases. The favorable 24-week mean error is neither typical-sample accuracy nor a consistent patient-level benefit. The dominant positive-amount defect is extreme right skew and poor centering: conditional medians fall below average observed amounts, conditional means overshoot, and broad positive intervals still cover only 80%/75% of present outcomes. Nearly prevalence-level presence is an additional limitation, not proof that occurrence cannot be predicted. The existing scalar also has imperfect presence/amount behavior; comparisons against it alone do not establish calibration.

Recommend one quantity-head change, without launching it: replace the positive-count log-logistic family (a logistic distribution on log count) with a discretized log-normal truncated to the same feasible remaining native capacity. Retain the Bernoulli factor, autoregressive order, absence atom, spatial conservation and fixed gates. This specifically tests whether a lighter positive tail produces more representative means; it does not repair weak presence discrimination by itself. The recommendation is motivated by saved distribution shape, not an isolated causal diagnosis or an established fix. Do not trim samples, cap volumes retrospectively, select by favorable aggregate means, or fit another model without a new instruction.


## Gaussian family replacement and Monte Carlo error, November 8, 2026

The [variance-matched Gaussian replacement](patient_gaussian_count_metrics.json) reduced extreme tail draws and the discrepancy between 64-sample and fitted-CDF means. That engineering improvement did not recover useful quantities: deterministic nonempty-core CRPS is essentially unchanged, and neither family's deterministic means beat persistence at either horizon. Official 64-sample differences sometimes reverse under the numerical reference, including the earlier logistic 24-week apparent MAE win. Those official outcomes remain recorded, with the numerical limitation now explicit.

Presence has modest ranking signal (patient-balanced AUC 0.630/0.576), compressed near-prevalence probabilities and unchanged rankings across families. Replacing the positive family alone did not resolve it. This bounded comparison does not isolate representation, loss balance or available supervision as the remaining cause, and is not evidence for a further family sweep. Preserve the completed negative result and use deterministic core references to interpret Monte Carlo mean/CRPS claims. No additional fit is authorized by this outcome.


## Frozen MRI information and memory did not establish useful predictive quantities, November 8, 2026

The [fixed information program](EXPERIMENTS.md#frozen-state-information-program-no-eligible-predictive-configuration-november-8-2026) separates a modest data benefit, measurement correction, and unstable temporal signals. More fitting patients improved some likelihood comparisons without establishing the required core quantities. Released current-reference volumes improved overall persistence but did not rescue the learned transition; cancellation explains why reference persistence need not win every subgroup. The full-memory condition improved mean core CRPS, but patient intervals and leave-one-out sensitivity do not establish incremental temporal value, and the prespecified seed failed. Selecting seed 31 would be post hoc.

The clearest failure is the conditional amount distribution when these MRI projections are used: very poor joint likelihood accompanies narrow intervals and severe undercoverage across compartments. This is evidence against this fitted readout recipe, not proof that frozen MRI lacks predictive information or that anatomical evolution is inherently unpredictable. The experiment does not isolate feature normalization, readout capacity, regularization, optimization or limited trajectories as the dominant cause. Preserve the observed-state system, scalar controls and all terminal candidates; no spatial decoder can repair the failed quantity gate. Further research decisions belong to the originating research discussion, not an automatic extension of this build.

## Low-capacity current MRI forecasting remained unsuccessful, November 8, 2026

The [regularized quantity comparison](EXPERIMENTS.md#regularized-scalar-versus-pca-mri-quantity-study-failed-both-gates-november-8-2026) avoided the previous MRI readout's extreme selection NLL, but neither its scalar nor PCA-MRI model produced acceptable core quantities. MRI did not establish better core CRPS or mean MAE than the scalar arm, and had worse all-target NLL at 12 weeks. This result concerns eight unsupervised PCA components with this linear conditional model; it does not establish that the frozen MRI state has no predictive signal.

The fitted core log-count scales around 2.25 produce broad distributions with large exact means. High sample-based coverage coexists with substantial mean-volume error and failed CRPS. These scales came from excluded-patient errors rather than optimistic fitting residuals, but optimizing conditional likelihood did not make their volume distributions useful. This is a measured limitation of the completed recipe, not proof that a scale cap, further tuning or another architecture will fix it. Preserve all outcomes and return research decisions to the originating chat; no extension or spatial stage follows automatically.

## November 9: acquisition choice did not earn its added machinery

The acquisition-JEPA study demonstrated an executable observed-evidence state and better missing-FLAIR completion against the preserved observer, but both learned policies followed the fixed order on every A visit and native anatomy did not pass retention acceptance. Anatomy was statistically worse than reference, but its interval straddles the −0.05 margin; neither retention nor inferiority beyond that margin was established. The oracle advantage was only 0.00184 under this trained observer; that constrains the tested task's available headroom, not the value of MRI acquisition in general. These results do not isolate the causal effect of JEPA, and the terminal selected checkpoint leaves training adequacy uncertain. No additional fit or retuning is part of this closed study.
