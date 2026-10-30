# Notes and insights

Why have the experiments not yet established useful prediction of glioma change? This page explains what the results mean, why several earlier claims were withdrawn, and what the remaining experiments can test.

The October 26, 2026, update explains the completed R37 pilot and its scoped result review. The earlier October 13 update covered R36 Phase 1 without new computation. Earlier runtime snapshots remain dated history, not live job status.

Use [Experiments](EXPERIMENTS.md) for measurements, [Methods](METHODS.md) for exact protocols and historical IDs, and [TODO](TODO.md) for remaining work. The [archive](archive/2026-10-12/README.md) preserves the original records and their correction history.

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

### Proposed short-paper outline

1. **Question.** Ask how the measured benefit of change-balanced supervision depends on the readout. Do not assume that a better trained output demonstrates a better frozen representation.
2. **Related work.** Position the study after PaIRNet/LILAC, Rudie, BrainDiff, LSOR, STRIDE and TRACE. Identify the matched objective comparison as the empirical extension, not a new principle or a reproduced published benchmark.
3. **Controlled design.** Define B and C, their shared architecture and fitting budget, automated signed-occupancy targets, frozen linear probe, patient-uniform aggregation, and locked uncertainty. Include A, D, original R35 and independent-state controls. Label the original R35 study separately.
4. **Results.** Lead with the inconclusive primary frozen-probe contrast. Then show the descriptive trained-head improvement, incremental-information controls, state retention and nuisance harms, empty-compartment errors, and all-arm development transfer. Use the sketched main figure and complete companion tables.
5. **Interpretation and limits.** State what this readout comparison establishes and what remains unresolved. Explain why observation-encoding evidence matters to MRI world models without calling it learned future dynamics. Retain target, provenance, cohort reuse, seed uncertainty and biological-validation limits.
6. **Reproduction and disclosure.** Bind the approved result artifacts and fitting recipes. Disclose absent published-model reproductions and AI assistance, with a human author responsible for the scientific claims. Adapt length and format only after selecting a confirmed workshop call.
