# Notes and insights

Why have the experiments not yet established useful prediction of glioma change? This page explains what the results mean, why several earlier claims were withdrawn, and what the remaining experiments can test.

The October 13, 2026, update explains the completed R36 Phase 1 findings and their scoped review. Earlier runtime snapshots remain dated history, not live job status. No new experiment was run for this documentation update.

Use [Experiments](EXPERIMENTS.md) for measurements, [Methods](METHODS.md) for exact protocols and historical IDs, and [TODO](TODO.md) for remaining work. The [archive](archive/2026-10-12/README.md) preserves the original records and their correction history.

<a id="research-objective"></a>

## Research objective and interpretation

The goal is to represent what an MRI shows about anatomy and disease, and how those features change across visits. The representation must also distinguish disease-related differences from differences caused by acquisition, such as scanner settings or image intensity.

A representation is the set of numerical features that an encoder extracts from a scan. We want those features to retain useful information. We do not want to call a model useful merely because it produces a low training loss.

JEPA, lesion-focused features, forecast heads, clinical inputs, and whole-brain methods are possible ways to reach the goal. None is the goal itself. The [research rules](documentation_sources/versions/6e710be8c33b05d77bf8f13b415b64028fcc13dc03d71defb304e6a7631c78f4.md) and [decisions D55 and D57](documentation_sources/versions/2b7293c884e0e0aa9f1db178d13f55497d584cd7060c68c022436d47330bb63c.md) make that distinction explicit.

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

R35 completed its corrected state, observed-change, and nuisance comparison. It did not show that a paired head beats independent state differences, and it did not test future prediction.

At the October 11, 18:30:51 UTC snapshot, R36 had all twelve final-epoch training checkpoints and ten seed-42 readouts locally validated. Seed 43 was running, and seed 44 had not started. No model-quality comparisons on the evaluation patients had been scored.

A completed implementation review does not count as design approval before training or acceptance of scientific results. The remaining reporting concerns were correct identity checks for nuisance reports and correct classification of harmful tradeoffs. Phase 2 was not authorized.

The [R36 record](documentation_sources/versions/51d6523f0da4b3d8745af10b79318fd684aba1fd9b47cc7d321e30df79984551.md) and [handoff](documentation_sources/versions/8fc1116f80667885774a5ce6a8fb9ec40f1e360d1f87867fefb81baf01118ab6.md) preserve that history. Neither completed training nor a partial arm establishes a quality result. The old statement that delegation was unavailable is also historical, not current operating guidance.

## Audits, alternatives, and methodological lessons

### Published scores need matching tasks before comparison

The literature audits describe different prediction problems. Some methods assess scans already observed. Others generate future images or classify response. Those tasks cannot share a performance threshold without matching their inputs, patients, labels, and splits.

TaDiff-Net reports treatment-conditioned image generation and segmentation. Its LUMIERE evaluation uses external image metrics, not RANO classification. Tikhonov's reported assessment macro-F1 is about 0.50, and TRACE's is about 0.477.

Matoso's reported median 0.51 is balanced accuracy, not macro-F1. The later survey records Amato's 0.616 ± 0.048 as a secondary reported value without a fully checked primary protocol. It records Maurya's 0.6415 development and 0.5118 hidden-test balanced accuracies, not macro-F1.

There is also an attribution conflict. The P1 audit assigns Maurya's entry the same arXiv identifier that the frontier survey assigns to Matoso. That conflict needs primary-source checking before publication.

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
