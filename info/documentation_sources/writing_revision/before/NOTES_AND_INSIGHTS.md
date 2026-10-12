# Notes and insights

Draft prepared 2026-10-12 from frozen research records. This explanation interprets the research record. Existing canonical records remain authoritative and retain their original detail. Read [TODO](TODO.md) for current work, [Experiments](EXPERIMENTS.md) for measurements, and [Methods](METHODS.md) for binding contracts.

<a id="research-objective"></a>
## Research objective and interpretation

The research objective is useful longitudinal MRI representations that preserve anatomical and pathological state and meaningful change while separating acquisition nuisance. JEPA, lesion awareness, forecasting heads, clinical conditioning, and whole-brain methods are hypotheses for achieving that objective, not the objective itself. A scoped lesion experiment must not become a universal definition of representation quality. Changes to target, clinical interpretation, or method family require operator approval. See [research guardrails](../documentation_sources/versions/6e710be8c33b05d77bf8f13b415b64028fcc13dc03d71defb304e6a7631c78f4.md), [decisions D55 and D57](../documentation_sources/versions/2b7293c884e0e0aa9f1db178d13f55497d584cd7060c68c022436d47330bb63c.md), and the [newer R36 execution record](../documentation_sources/versions/51d6523f0da4b3d8745af10b79318fd684aba1fd9b47cc7d321e30df79984551.md).

A useful evidence vocabulary separates four things that are often conflated:

1. **State** is information present in an individual scan, such as anatomy or lesion burden.
2. **Observed change** is a difference between two observed scans. It can be characterized without claiming that the first scan predicts an unobserved future.
3. **Nuisance** includes acquisition and site variation that changes measurements without being the target biological signal.
4. **Prospective prediction** uses information available before the outcome and predicts a later observation. In the model records, "causal" means that inference excludes future inputs. It does not mean that observational treatment effects are causally identified.

Consequently, "the representation contains state," "paired scans reveal change," and "the representation predicts future change" require different inputs, comparisons, and evidence. Neither good reconstruction of observed differences nor a downstream association establishes causal forecasting.

### Useful leads that survive

R35's strongest positive lead is useful per-scan state decoding on the in-domain automated-volume proxy. My interpretation is that independent state estimates deserve at least as much attention as a specialized pair network. Dense-mask supervision differs from the frozen scalar control, so this does not identify architecture as the cause of the gain. The [R35 evidence](EXPERIMENTS.md#r35) records the actual measurements and failed transfer controls.

- **Representation comparisons should isolate the capability being tested.** R35 distinguishes independent per-scan state subtraction, paired observed change, and nuisance. Its corrected comparison is useful evidence about those candidate capabilities, but it does not establish future prediction or a matched paired-head advantage. See the sibling-worktree [R35 comparison](../documentation_sources/versions/768fdfa0a9be4dc76e99fcef9ff971e26252d672e82b08267f5e9ea157ced0f3.md), from `world-model-representations` at `2f6e1f5a65be4ccbceac6de706b537de8b695faa`.
- **Changed-region coverage is a plausible bottleneck, not a proven mechanism.** R36 target QA found median changed occupancy below 0.5% of the grid in each compartment; one native enhancing disagreement disappeared at 96³. That motivates testing changed-region balancing. It does not establish that sparse changed voxels caused earlier failures or that balancing will improve a representation. See [R36 results](../documentation_sources/versions/51d6523f0da4b3d8745af10b79318fd684aba1fd9b47cc7d321e30df79984551.md), from `world-model-r36` at `f4b96c00dd5d79c59ba02d186e024ff747f690ad`.
- **Acquisition effects deserve explicit measurement.** R35 includes nuisance comparisons; K3 and earlier cross-site investigations identify preprocessing, intensity, modality, and scalar-calibration concerns. These are plausible limits on transfer, not a single established explanation for every transfer failure.
- **The anatomical and lesion measurement pipeline is a reusable asset with bounded claims.** The records report geometry, masks, and volume integrity checks, but those checks do not establish segmentation accuracy, repeatability, biological change, template equivalence, or forecasting benefit. See [methodology validity](../documentation_sources/versions/7bc218e8ef5fd2fdb22bc0ed7f5bab72c4b1d099f1f7f2c3145f6c3bca6f3472.md), [P0/P1 record](../documentation_sources/versions/89605d7963ba3e0f942e5feb416846a32faea82a277f1dbe008b0cd0a6b86068.md), and [D56](../documentation_sources/versions/2b7293c884e0e0aa9f1db178d13f55497d584cd7060c68c022436d47330bb63c.md).
- **A simple baseline can be a strong result.** Persistence and measurement-based readouts have tied or beaten learned forecasting in the scoped lesion task. This is evidence against a gain for the tested methods and cohorts, not evidence that all representations or learning objectives are useless.

### Plausible limits versus established mechanisms

Sparse lesion change, imperfect timing, segmentation error, scanner/site shift, optimization instability, and weak supervision are plausible limits. Some have direct audit evidence in particular datasets or runs; none should be promoted into a universal causal explanation without an isolating experiment. For example, timing provenance constrains fixed-horizon claims, but does not itself explain a model's forecast error. A measured transfer gap does not by itself identify harmonization, representation quality, or label validity as its cause.

The proposal's original premise was JEPA: predict a future latent rather than pixels, using longitudinal structure and clinical context. That was a historical research hypothesis, not a demonstrated advantage. Likewise, RANO-as-action and treatment-conditioned dynamics were proposals whose semantics and evidential prerequisites are not met merely by encoding a clinical column. See [PROPOSAL.md](../documentation_sources/versions/e82ede5e41b034398b599f1f141ffa04335d47ad7d8e66761540a71ce2f72aaa.md), [action-conditioned experiment brief](../documentation_sources/versions/f3dad868d43b3aa2bf5f7913b3dfa2872cdf87a65fd9648b1eb9932e3f750855.md), [ASTRA_PROBE.md](../documentation_sources/versions/b72ceb2da7c2ba695a4174a43643847c9fb27244c1785ff1a6acb5c898f2fb29.md), and [K3_PROBE.md](../documentation_sources/versions/7d63fd3f44c1ed975ea9d1b860b597592902c503fbb3124145a01db17ee996dc.md).

<a id="current-evidence"></a>
## Corrections, withdrawals, and current evidence

### Corrected headlines

- The early 2.7-fold JEPA-versus-persistence headline compared different latent spaces. D28 and A8 require same-space scoring. A24 found an unresolved pooled test difference and a resolved patient-uniform test difference on a reused development split. The aggregate does not support a general gate win.
- The early 82/91 patient headline is superseded by 41/91 same-space wins. Keep the paired intervals and aggregation with that count. See A8, A24, D28, K3-2, K3-11, and [the historical results](../documentation_sources/versions/89605d7963ba3e0f942e5feb416846a32faea82a277f1dbe008b0cd0a6b86068.md).
- Legacy all-91 probe cross-validation is not encoder-unseen evaluation. K3-16 identified encoder-train patients entering probe-test folds. Locked unseen protocols and the correct cohort/cache version must accompany readout claims.
- Early single-split RANO gains are not robust evidence. Probe seeds and folds matter; historical RANO scores are development evidence and must be described with their protocol.
- Historical RANO SOTA numbers are reported in the project's literature audits, not independently verified here. The task definitions and input information differ, and the cited literature comparisons do not establish a like-for-like representation advantage. See [frontier survey](../documentation_sources/versions/bd305198a22112d90d28cf638536c3638ff95806a6aeff087655acb8dea287d1.md) and [P1 SOTA audit](../documentation_sources/versions/77de3ea25aa45d894d9577a96b69195912e6036883dace704af07b89de07843b.md).

### Withdrawn forecasting claims

A40 withdrew learned P2.4 and adapted-checkpoint forecasting claims after discovering target-visit images/masks and ending gaps in model inputs. The earlier pair builder also bridged visits with missing masks, yielding 123 rather than 117 legitimate unseen pairs. Those gains and adaptations are withdrawn, not merely downgraded. A41 clean reruns are a separate valid record; historical P2 seed-stability results do not rehabilitate the invalid causal claims. Closure applies to those methods and tasks, not to representation learning generally. See [A40/A41](../documentation_sources/versions/eaca551880e0a9db5c4bdc7785e135675cd518ebfcd361d5aa482beca4f37dce.md), [P2 execution](../documentation_sources/versions/9ea08b90cf4971fb7b85d4b150ef1317994f42aa2fca2f1c9c0f5de5424e38e4.md), and [D53](../documentation_sources/versions/2b7293c884e0e0aa9f1db178d13f55497d584cd7060c68c022436d47330bb63c.md).

### R35 and R36 boundaries

R35 is a corrected completed comparison of state subtraction, observed change, and nuisance. It is not a future-prediction result, nor does it prove a paired head beats a matched independent-state construction.

R36 Phase 1 remains incomplete as of the snapshot. The record says all twelve final20 training checkpoints and ten seed42 probes are locally validated. A continuation receipt at 2026-10-11 18:30:51 UTC reports seed43 RUNNING; seed44 remains unlaunched. There are no comparative query-quality scores. A reviewer completed post-implementation review, which is not retrospective pre-training approval or final result acceptance. Nuisance report identity and negative tradeoff classification remain reporting risks. Phase 2 is unauthorized. Do not infer an outcome from the capacity run or partial arms, and do not treat the old R36 status "delegation unavailable" as current. See [R36 execution record](../documentation_sources/versions/51d6523f0da4b3d8745af10b79318fd684aba1fd9b47cc7d321e30df79984551.md) and local [R36 handoff](../documentation_sources/versions/8fc1116f80667885774a5ce6a8fb9ec40f1e360d1f87867fefb81baf01118ab6.md).

## Decision ledger

This compact ledger records historical decisions and their current interpretive status. The canonical [decision log](../documentation_sources/versions/2b7293c884e0e0aa9f1db178d13f55497d584cd7060c68c022436d47330bb63c.md) is the authority for full rationale and chronology. "Current" means a governing scope or policy, not necessarily a current experiment; "superseded" means do not reuse the old claim or implementation as an unqualified present conclusion.

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
| D42 | Separate assessment, forecasting, anatomy forecasting, surprise; reset validity gate | Governing task distinction and validity policy. |
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

## Direction and audit indexes

These indexes preserve useful lineage without turning every historical suggestion into a current TODO. Full descriptions and dates remain in [directions.md](../documentation_sources/versions/eade325e60cbeeb92dc5f6570520d8f502a519e846309ef5803b2bf7e1a2d094.md). Direction labels are not evidence of execution or authorization.

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

ASTRA V1 to V5 and E0 to E4 divide measurement validity from proposed next experiments. **V1** identifies an AUC implementation error; **V2** inconsistent label contracts; **V3** survival leakage; **V4** mismatched prediction problems among locked patients; **V5** holdout/label-version reset. E0 establishes a prospective floor, E1 a fair observed-scan assessment baseline, E2 tests incremental temporal value, E3 considers lesion-aware residual forecasting, and E4 adds information unavailable in the current corpus. These are an ordered historical program, not blanket authorization. See [ASTRA_PROBE.md](../documentation_sources/versions/b72ceb2da7c2ba695a4174a43643847c9fb27244c1785ff1a6acb5c898f2fb29.md).

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

R1 to R28 are historical proposals and follow-ups catalogued in [directions.md](../documentation_sources/versions/eade325e60cbeeb92dc5f6570520d8f502a519e846309ef5803b2bf7e1a2d094.md). The index below conveys topic and disposition, not a claim that each was run successfully.

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
| R36 | Delta-representation objective contrast. Phase 1 incomplete at snapshot; Phase 2 unauthorized. |

## Audits, alternatives, and methodological lessons

The [external supervision audit](../documentation_sources/versions/636f0274c36efa3cee145c530acebff689d5961eff995269ecc3bc91d8c6d89e.md) compares UCSF-ALPTDG for lesion/change supervision, Burdenko for longer trajectories and treatment context, and UCSD-PTGBM for treatment-effect auxiliary labels. These are reported dataset characteristics and access/storage constraints, not independently verified literature claims in this draft. D52 prefers UCSF as a supervision fit but blocks acquisition pending authorization and storage capacity. The dated audit reports 298 paired UCSF patients and a required noncommercial DUA, 180 Burdenko patients with a restricted 131.23-GB imaging release, and 178 UCSD subjects with 243 timepoints in a 44.98-GB version-3 release. UCSD's version-3 metadata corrected patient identifiers. These are audit-reported capacities and constraints, not a current disk or access check. Teacher-training patients cannot later be called an untouched external test.

The literature snapshots preserve several different capabilities. TaDiff-Net reports treatment-conditioned image generation and segmentation, with LUMIERE used for external image metrics rather than RANO classification. Tikhonov's reported assessment macro-F1 is about 0.50, TRACE's about 0.477, and Matoso's median 0.51 is balanced accuracy, not macro-F1. The later survey records Amato's 0.616 ± 0.048 as a secondary reported value without a fully checked primary protocol. It records Maurya's 0.6415 development and 0.5118 hidden-test balanced accuracies, not macro-F1. The P1 audit attaches Maurya's entry to the same arXiv identifier that the frontier assigns to Matoso. That attribution needs primary-source verification before publication. None of these is a matched threshold for past-only forecasting.

The [frontier survey](../documentation_sources/versions/bd305198a22112d90d28cf638536c3638ff95806a6aeff087655acb8dea287d1.md) and [P1 SOTA audit](../documentation_sources/versions/77de3ea25aa45d894d9577a96b69195912e6036883dace704af07b89de07843b.md) report external literature comparisons. Their scores are not independently validated here, and comparison requires matching cohort, input information, endpoint, and split. An alternative research path is to first establish the task-specific simple floor, then test whether history, paired observations, or learned representations add information under matched patient-clean rows. Another is acquiring appropriately authorized lesion/change supervision rather than repeatedly optimizing the same self-supervised signal. Neither path is approved merely by appearing in an audit.

The [methodology validity plan](../documentation_sources/versions/e222fabaeb05f4d166a3d220a87398aa55b60e1697d479706b6e87d6bab71212.md) and [validity record](../documentation_sources/versions/7bc218e8ef5fd2fdb22bc0ed7f5bab72c4b1d099f1f7f2c3145f6c3bca6f3472.md) emphasize verification of geometry, chronology, labels, and provenance before interpreting results. The [anatomy-first harness](../documentation_sources/versions/ad235fe7fd9dbc123dbc829a8242d3c11ce6427b241808ba661ac7ed0fb1db87.md) retains legacy RANO workflows behind a compatibility boundary. The [stale-code audit](../documentation_sources/versions/eca8302b4c0da4df15f1a8918164254666650e647632c098c025513f6ebe5061.md) identifies legacy task paths and unsafe semantics; code availability is not evidence that a path remains scientifically current.

### Action-conditioned representations

The [action-conditioned representation brief](../documentation_sources/versions/f3dad868d43b3aa2bf5f7913b3dfa2872cdf87a65fd9648b1eb9932e3f750855.md) lays out a candidate comparison between a current approach, materially different approach, and simple baselines, with cutoff, patient-clean split, evaluation, cost, independent checks, and falsifiers. This is design material only. The proposed action-conditioned synthetic work was not run and is unselected. The source scan and declared action would be the only prediction inputs. Patient partitioning must precede view generation, and transformation families must be held out separately. Geometric actions and physical targets must be constructed in verified pre-resize coordinates, with intensity nuisance checked before and after normalization. No existing action-labelled generator was identified. Synthetic action correctness would not establish glioma biology. It must not be described as a result or as validated treatment-effect modeling. Earlier clinical-action hypotheses in [PROPOSAL.md](../documentation_sources/versions/e82ede5e41b034398b599f1f141ffa04335d47ad7d8e66761540a71ce2f72aaa.md) are historical; RANO response is not interchangeable with treatment action. An action-conditioned representation is scientifically meaningful only if action identity and timing are valid, available at the prediction cutoff, and compared against no-action and simple temporal baselines.

### Legacy I1 to I4 traceability

The decision log refers to lessons I1 to I4, but the complete captured corpus lacks authoritative standalone definitions of all four lessons. This draft therefore preserves references without reconstructing absent definitions as fact.

| Reference | Traceable source context |
|---|---|
| I1 | A1 uses I1 for the gap a learned method must close against persistence. The original full lesson entry is not reconstructed. |
| I2 | A5 and the pilot associate I2 with interpreting collapse monitors. Falling loss alone is insufficient evidence. The original full lesson entry is not reconstructed. |
| I3 | D6 links LoRA rationale to I3; do not infer a broader lesson definition from that mention. |
| I4 | D8 explicitly ties I4 to the corrupt community checkpoint lesson: matching keys did not guarantee valid behavior; verify weights behaviorally. |

## Source map and limitations

The assigned source mapping is explicit here: historical hypothesis and architecture context come from [PROPOSAL.md](../documentation_sources/versions/e82ede5e41b034398b599f1f141ffa04335d47ad7d8e66761540a71ce2f72aaa.md); decision lineage D0 to D57 and D33b from [decisions.md](../documentation_sources/versions/2b7293c884e0e0aa9f1db178d13f55497d584cd7060c68c022436d47330bb63c.md); V1 to V5/E0 to E4 from [ASTRA_PROBE.md](../documentation_sources/versions/b72ceb2da7c2ba695a4174a43643847c9fb27244c1785ff1a6acb5c898f2fb29.md); K3-1 to K3-35 from [K3_PROBE.md](../documentation_sources/versions/7d63fd3f44c1ed975ea9d1b860b597592902c503fbb3124145a01db17ee996dc.md); G1 to G20, R1 to R32, and the R36 priority addendum from [directions.md](../documentation_sources/versions/eade325e60cbeeb92dc5f6570520d8f502a519e846309ef5803b2bf7e1a2d094.md); R33 from [clean_rescues.md](../documentation_sources/versions/eaca551880e0a9db5c4bdc7785e135675cd518ebfcd361d5aa482beca4f37dce.md); R34 from [methodology_validity.md](../documentation_sources/versions/7bc218e8ef5fd2fdb22bc0ed7f5bab72c4b1d099f1f7f2c3145f6c3bca6f3472.md); dataset and literature framing from [frontier_lumiere.md](../documentation_sources/versions/bd305198a22112d90d28cf638536c3638ff95806a6aeff087655acb8dea287d1.md), [external_supervision_audit.md](../documentation_sources/versions/636f0274c36efa3cee145c530acebff689d5961eff995269ecc3bc91d8c6d89e.md), and [p1_sota_audit.md](../documentation_sources/versions/77de3ea25aa45d894d9577a96b69195912e6036883dace704af07b89de07843b.md); validity plan from [methodology_validity_plan.md](../documentation_sources/versions/e222fabaeb05f4d166a3d220a87398aa55b60e1697d479706b6e87d6bab71212.md); action-conditioned proposal from [action_conditioned_representation_brief.md](../documentation_sources/versions/f3dad868d43b3aa2bf5f7913b3dfa2872cdf87a65fd9648b1eb9932e3f750855.md); harness boundary from [harness_v2.md](../documentation_sources/versions/ad235fe7fd9dbc123dbc829a8242d3c11ce6427b241808ba661ac7ed0fb1db87.md); and code currency from [stale_code_audit.md](../documentation_sources/versions/eca8302b4c0da4df15f1a8918164254666650e647632c098c025513f6ebe5061.md). Current scientific boundaries use [R35](../documentation_sources/versions/768fdfa0a9be4dc76e99fcef9ff971e26252d672e82b08267f5e9ea157ced0f3.md), [R36](../documentation_sources/versions/51d6523f0da4b3d8745af10b79318fd684aba1fd9b47cc7d321e30df79984551.md), and the local [R36 handoff](../documentation_sources/versions/8fc1116f80667885774a5ce6a8fb9ec40f1e360d1f87867fefb81baf01118ab6.md).

Every research-source link in this draft points to an immutable captured version. [Source coverage](SOURCE_COVERAGE.md) records all branch and worktree variants, their read modes, and their dispositions. Archived sources retain their original wording and embedded links; they are evidence, not newly validated operating instructions. The literature claims remain reported rather than independently verified. This synthesis intentionally does not restore deleted `notes.md` or `explainer.html`, reproduce full result tables, or convert historical directions into current work authorization.