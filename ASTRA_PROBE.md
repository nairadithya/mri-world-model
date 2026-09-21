# ASTRA_PROBE — what to research next, and how to beat SOTA credibly

**Review anchor:** repository commit `7a018fe`; experiment record through A36.
**Scope:** research strategy informed by the proposal, results, K3 review, current model/data/evaluation code, and linked literature. This is not a new model evaluation. Existing scores below are attributed to their recorded experiments; the AUC counterexample is a new, read-only executable check. No implementation changes are included.

## Executive recommendation

**Do not spend the next GPU session trying to improve the champion's cosine loss.** First separate response *assessment* from future *forecasting*, repair the measurement instrument, and establish a competitive, information-matched baseline. Then invest in explicit lesion-change information and additional post-treatment patients.

The strongest program is:

> Learn a patient-specific, lesion-aware state that improves assessment of observed change and predicts clinically meaningful future change, with calibrated uncertainty across institutions.

That is more defensible than “a JEPA world model beats RANO SOTA.” The current repository has a potentially useful pretrained trajectory representation, but it has **not established generalizable dynamics superiority, a JEPA-specific early-warning advantage, or superiority to competitive published assessment systems**.

My priorities:

1. **Make the question and its labels identical across competing methods.** The current SOTA gap is not an apples-to-apples gap.
2. **Reproduce a strong radiomics + pretrained-image assessment model**, then test whether JEPA history adds complementary information.
3. **For actual forecasting, beat last-status, growth-trend, and small temporal baselines before adding architectural complexity.** Use physical lesion measurements and progression risk as primary endpoints; latent error is secondary.
4. **Add information rather than repeatedly optimizing the same 65 training patients:** UCSF-ALPTDG for supervised lesion/change representation learning; Burdenko for longer GBM trajectories; UCSD-PTGBM for treatment-effect discrimination.
5. **Reserve an actually new external evaluation.** Repeatedly calling the same 13 patients “final” cannot restore their independence from model development.

---

## 1. The core problem has split into different tasks

Let `H_t` contain only imaging and clinical information available at visit `t`.

| Task | Permitted information | Target | Appropriate comparison |
|---|---|---|---|
| Response assessment | Both scans `x_t, x_(t+1)`, their available masks, earlier history, metadata available by `t+1` | RANO at `t+1` | Tikhonov, TRACE, Amato, BraTS progression systems |
| Future response forecast | `H_t`; optionally a specified future horizon and a treatment plan already known at `t` | RANO at a future visit, or incident progression within a fixed interval | Last RANO, transition model, past growth/radiomics, supervised history models |
| Future anatomy forecast | `H_t` and specified horizon | Future lesion measurements, masks, or frozen image representation | Persistence, shrinkage/trend models, small dynamics models; TaDiff when outputs/protocol match |
| Surprise/change detection | Observed `x_(t+1)` plus its prediction from `H_t` | Abnormality/progression at `t+1` | Raw image/lesion change, persistence error, standardized innovations |

**The central mismatch is verified in code and the papers.**

- `src/model/temporal.py::forward_prefixes` constructs `state_t` from visits through `t`, not through `t+1`.
- `src/harness/data/tasks.py` maps `states_forecast[t]` to `labels[t+1]`.
- Tikhonov explicitly uses baseline **and follow-up** MRI and growth/shrinkage masks [S1]. TRACE does too [S2]. They assess an already observed response; “prediction” in their titles does not mean forecasting an unseen scan.
- `src/harness/encode/radiomics.py::_pair_feature` includes `v_(t+1)`, realized growth, and a nadir computed through `t+1`. The CNN comparator in `src/harness/train/cnn3d.py::_patient_pairs` also sees both images. Those are assessment comparators, not forecast comparators.

Therefore **0.309 forecast CV versus 0.50 assessment CV is not a measured 0.191 architectural deficit**. Nor does a weak assessment comparator prove that forecasting is solved. This mismatch changes the proposed objective, the baselines, and the interpretation of A26–A36.

### Choose the claim before the experiment

- **Fastest route toward beating published scores:** build an assessment model using the observed follow-up scan, explicit change features, and history. Describe it as assessment.
- **Most distinctive world-model contribution:** improve a genuinely prospective, clinically meaningful forecast over strong past-only baselines on a new institution. Published assessment F1 is not its target threshold.
- Run both as separate tracks only if resources allow. An assessment win does not establish forecasting, and a forecast model need not numerically beat a method given the answer-bearing scan.

For fixed-horizon forecasting, distinguish a user-specified 90-day horizon from the *realized* next appointment interval. The latter may depend on symptoms/progression and may not be known at prediction time. For next-visit prediction without a known schedule, explicitly state that the model marginalizes over the local follow-up process.

## 2. What the existing evidence actually supports

### Assets worth preserving

- A functioning longitudinal data pipeline, official BRAINIAC initialization, frozen caches, and a modular harness.
- Honest corrections of mixed-space persistence, encoder/probe overlap, and SAILOR empty modalities. A23/A24 supersede several K3 findings; do not simply repeat the old audit as if those fixes never happened.
- A25's recorded frozen-state forecast macro-F1: **0.309 [0.255, 0.358]** within the 26 encoder-unseen patients; **0.408 [0.299, 0.473]** for readout transfer from the 65 to those 26. These are different training regimes, not interchangeable estimates.
- A27 establishes substantial readout sensitivity: final-split scores span roughly **0.34–0.45** across seeds/widths. Small point gains are not a research result.
- A26–A36 give useful negative results for particular frozen-feature interfaces and particular training recipes. They argue against indiscriminate repeat sweeps.

### Claims that need narrowing

1. **General dynamics superiority is unproven.** A24 reports test-pooled JEPA-minus-persistence error `−0.0017 [−0.0072, +0.0012]`, not significant. Test patient-uniform improvement is significant, but train-pooled error is significantly worse. Same-space SAILOR dynamics lose approximately **7–10.5×** across preprocessing arms. State every claim with its aggregation and interval.
2. **“The representation is at its data-limited ceiling” is a hypothesis, not a demonstrated ceiling.** Multiple unsuccessful optimizers or objectives do not bound what different inputs, pretrained lesion features, corrected labels, or a smaller temporal model could achieve.
3. **“ROI does not help” is too broad.** A26 used approximately aligned, coarsely pooled frozen tokens; `interface.py` explicitly documents approximate mask alignment. That does not test a native-resolution lesion encoder pretrained with expert change masks. Replacing the input to an already-trained temporal stack also tests distribution compatibility, not only information content.
4. **“The remaining gap is entirely 4,800-feature radiomics” is not established.** A35 tests 22 scalar volume/growth features with a different classifier and cohort construction. Tikhonov's growth/shrinkage improvement involves *spatial masks and their radiomic texture*, not just scalar growth [S1].
5. **“JEPA transfers better than supervised learning” lacks a competitive control.** A29–A32 acknowledge underpowered/from-scratch CNN comparisons. Published methods use pretrained encoders and/or rich radiomics. Preserve the narrower result; do not generalize it to supervised learning.
6. **A36 is not a conclusive measurement-information ceiling.** Its `fused_next` “oracle” sees only the follow-up image, while the target vector includes baseline measurements, growth, and historical nadir. A single future snapshot is not an oracle for a pair/history-dependent target. Test `[v_t, v_(t+1), v_(t+1)−v_t]` and the required history before concluding growth is absent. Linear non-recoverability also does not prove information-theoretic absence.

## 3. Validity gates before interpreting another experiment

These are prioritized because they change research decisions, not because this should become a general code-cleanup project.

### V1 — AUC implementation is wrong, not merely imprecise about ties

**Verified executable counterexample** for `src/harness/eval/metrics.py::auc_mann_whitney`:

```text
scores = [0.2, 0.4, 0.1, 0.3]
labels = [0,   0,   1,   1]
current function: 0.75
correct pairwise AUC: 0.25

all four scores tied:
current function: 1.0
correct AUC: 0.5
```

The implementation treats `argsort` indices as per-observation ranks. Those are different permutations. It also gives ties distinct ranks. `tests/test_harness.py::test_auc` checks only perfectly separated scores, which does not expose the error.

**Required research action:** use a validated AUC implementation; test reversed, interleaved, tied, and single-class cases; rerun surprise, persistence-error, lead-time, and site-discrimination analyses that call this function. The current `surprise`, `persistence`, `leadtime`, and `freeze` paths use it. Trace historical implementations before deciding which published tables are affected. **Do not reuse the old AUC values or their fine-grained ranking as evidence until refreshed.** This does not invalidate macro-F1 or cosine-error arithmetic by itself.

The general point that contemporaneous surprise is not early warning remains true regardless of the corrected numbers. Whether JEPA has *any* lead advantage should be re-adjudicated, not presumed permanently closed by possibly affected AUCs.

Also, the generic `ReadoutEvaluator` currently discards logits before metric computation, and bootstraps macro-F1 even when the primary metric is accuracy. A future AUROC/AUPRC/calibration study must preserve scores and bootstrap the metric actually named; adding a CLI metric flag is not sufficient.

### V2 — There is no single clean label contract across methods

`LUMIEREDataset` maps missing/unrecognized RANO to action ID `2` (SD), and maps `Post-Op/PD` to PD. The frozen encoding path instead uses the raw clean-string map, excluding everything except `{PD, SD, PR, CR}`. CNN, auxiliary-head, and radiomics paths derive labels back from action IDs.

Thus “clean-labelled” means different things across methods: a missing rating can become an SD training/evaluation example in one path and be excluded in another. This is a code-confirmed risk; the number of affected cached rows was not measured in this review.

**Required contract:** separate `response_label`, `response_label_valid`, `treatment`, and `operative_event`. Never use response-derived “actions” as a substitute for treatment or as an implicitly cleaned label. Align comparisons by immutable `(patient, index visit, target visit, target definition)` IDs, then assert label equality and report exclusions.

For SAILOR, verify the numerical codebook with the data owner; a volume-inferred PR/CR mapping is not an independent reference standard. Report the code-swap sensitivity until verified. Do not infer labels from the same volume changes used as candidate predictors.

### V3 — Overall survival is still an input

`dataset.py::_clinical_vector` includes total eventual survival; the SAILOR adapter has the analogous field. This information is unavailable prospectively. **An image-only target does not make future clinical information in the predictor legitimate.** A weak clinical-only probe does not rule out interactions with imaging.

Zeroing survival and re-encoding is a useful sensitivity analysis, but a clean prospective claim needs training without it: the old temporal model was trained with that channel. Rebuild affected caches with a versioned feature schema. Use survival only as an outcome in an appropriately censored survival task, not as baseline context.

### V4 — Locked patients are not necessarily matched prediction problems

`tasks.py` changes the label offset according to the feature view. Consequently `states_forecast`, `states_current`, and `fused` can be evaluated on different visits or different labels despite sharing patient folds. `aggregate.py` intersects patients, not row identities. A patient-paired interval is not proof that two methods solved the same task.

For a valid history ablation, compare on **the same `RANO_(t+1)` rows**:

```text
current-image-only at t
current image + previous change, all observed by t
past-image pooling / small GRU
JEPA state through t
```

For assessment, all arms may see `t+1`, and all arms target the same label there. Keep missing-mask coverage fixed in the primary head-to-head comparison, with a separate full-coverage robustness result.

### V5 — The holdout and the label version need an honest reset

- `dev` was used for encoder checkpoint selection; the 26-patient CV pool is not wholly selection-clean.
- `final` has repeatedly been reported during architecture selection. Freeze the historical record, call it development evidence, and seek a new external cohort/hidden test. Merely redividing the same patients cannot make prior exposure disappear.
- For an all-LUMIERE, patient-wise SOTA comparison, refit every LUMIERE-trained representation inside each outer training fold, or use a genuinely external-only fixed encoder. A champion trained on 65 of the 91 cannot simply be reused for clean all-91 outer evaluation.
- All feature selection, scaling, imputation, radiomics filtering, harmonization, calibration, and hyperparameter tuning belong inside training folds. A36's ridge inner folds are row-random; make those patient-grouped too.
- Do not silently relabel historical RANO annotations as RANO 2.0. RANO 2.0's early pseudoprogression window is relative to **radiotherapy completion**, not simply three months after surgery [S6]. The current `surgery_window` is a dataset restriction, not a complete implementation of that guideline.

## 4. Update the SOTA map before choosing a target score

The local frontier survey is useful background but no longer a sufficient scoreboard.

| Reference | What is verified | How to use it |
|---|---|---|
| Tikhonov et al. [S1] | Paired observed scans; pretrained modality-specific ResNet-18 + 4,896 radiomic features + engineered change + CatBoost; reported patient-wise CV macro-F1 **0.50 ± 0.08** | First strong reproducible assessment target. Match pair inclusion, masks, pretraining, tuning, and aggregation. |
| TRACE [S2] | Paired MRI/masks; pretrained MedicalNet encoder, predicted measurements, deterministic derived concepts, metadata passthrough, learned task head; macro-F1 **0.4769 ± 0.1229**; binary macro-F1 **0.7085 ± 0.0935** | Test clinically explicit measurements and inspectable reasoning, not just a generic concept-concat MLP. Audit which concepts are supplied versus predicted. |
| Amato et al. [S3] | TRACE Table 3 reports macro-F1 **0.616 ± 0.048**; the DARE project report §2.29, p.92 independently reports multiclass F1 **0.616 ± 0.048** | **Higher reported reference requiring full-method verification.** The primary paper's complete methods were not accessible here; confirm averaging, folds, tuning, and inputs before treating it as an apples-to-apples benchmark. |
| Maurya et al. [S4] | Publisher abstract: Swin UNETR segmentation + shape/first-order radiomics + TabM; LUMIERE CV **balanced accuracy 0.6415**; hidden multicentric test **balanced accuracy 0.5118**, 1,010 cases/300 patients | A meaningful external-generalization target. These are **not macro-F1 scores**. Obtain challenge rules, code, and the same evaluator. |
| Gao et al., Siamese Vision Transfer [S5] | Institutional publication record reports top-two Task-11 test performance; paired-image transformer | Obtain full protocol/results. The abstract's “90% accuracy” is not a valid replacement for patient-wise macro-F1 or a hidden-test score. |
| TaDiff [S10] | Treatment-conditioned longitudinal image/mask generation | Relevant if pursuing future spatial outputs. Compare against mask persistence and the same observed history/horizon, not RANO F1 or whole-brain image similarity alone. |

**Consequences:**

- Do not target “0.51 macro-F1, therefore SOTA.” A higher F1 reference is reported, the hidden-test landscape has advanced, and metrics differ.
- Tikhonov's central edge is *where and how the tumor changes*: appearance of growing/shrinking regions, not merely a 22-dimensional volume vector.
- Maurya provides a counterpoint to “only thousands of textures can help”: segmentation plus a smaller shape/intensity vocabulary and a strong tabular learner is also worth reproducing.
- Verify access to an active/reopened BraTS progression evaluation with organizers. Do not assume last year's challenge remains open or upload any controlled-access SAILOR data to it.

## 5. Research program, ordered by expected information gain

### E0 — Correct the instrument and establish the prospective floor

**Hypothesis:** part of the apparent ceiling/ranking comes from unmatched labels, cutoffs, and baselines rather than missing network capacity.

**Minimum experiment:**

1. Create separate assessment and forecast manifests with explicit information cutoffs and clean label validity.
2. Remove future clinical inputs; run the sensitivity analysis before the full clean retrain.
3. Validate metric implementations and reissue affected AUC analyses.
4. On identical forecast rows, fit training-majority, last-observed-RANO, a smoothed RANO transition model, demographics/pathology-only, past lesion-volume/growth regression, and regularized current-image readouts.
5. For a first-progression endpoint, restrict to patients not already progressed at the index visit; do not obtain a strong “forecast” mostly by predicting persistent PD after it is already known.

**Gate:** a complete row-matched table with probabilities, class counts, patient counts, paired intervals, and missingness coverage. If last-RANO or simple growth matches the JEPA state, do not train a larger dynamics network to explain a gain that is not established.

**Cost:** CPU-scale analysis; clean representation retraining is a separate, budgeted step. No new GPU sweep before the label/metric contract passes.

### E1 — Beat a real assessment baseline with complementary history

**Highest-value near-term SOTA experiment.**

**Hypothesis:** explicit observed lesion change and pretrained global history contain complementary signals. Test this rather than forcing one representation to encode both perfectly.

Use observed baseline/follow-up images and masks, with earlier scans when available:

- Reproduce Tikhonov's feature families and CatBoost; obtain or document any unavailable details. Reproduce a Maurya-style compact shape/intensity arm before assuming 4,896 features are necessary.
- Add an ImageNet-pretrained ROI encoder, trained to a verified convergence regime. The ban on installing `torchvision` does not prohibit behaviorally verified weights-only loading into a compatible implementation.
- Build a **JEPA assessment** arm from paired embeddings/deltas and a state that actually includes the assessment scan. The existing `states_current` path omits the final visit because only forecasting prefixes are cached; do not lose those visits accidentally.
- Test late probability fusion between the radiomics/measurement expert and the image/history expert. Train the fusion on inner out-of-fold predictions, not in-sample logits.

**Small ablation ladder, same rows:**

| Arm | Purpose |
|---|---|
| Volumetry + past/current nadir + interval | Interpretable floor |
| Shape/intensity/growth-region radiomics + CatBoost | Strong non-deep baseline |
| Pretrained paired-image features + matched readout | Fair deep baseline |
| Radiomics + pretrained paired-image features | SOTA-family reproduction |
| Same + history / JEPA state | Incremental value of this project |

Use at most a small, prespecified classifier family set; give baselines comparable tuning budgets. Record train and validation curves for image models. Compare regularized linear/tabular models before a high-dimensional MLP.

**Go criterion:** the hybrid improves over the strongest *reproduced* comparator with a positive paired patient-level confidence interval and useful minority-class behavior. Require confirmation on new external data before a generalization claim.

**Stop criterion:** if a competent radiomics model wins and JEPA adds nothing, accept that outcome. A strong externally validated assessment pipeline is useful; there is no obligation to retain JEPA in the winning classifier.

### E2 — Determine whether learned temporal dynamics add anything

**Hypothesis:** the champion's useful state contains more than static pretrained anatomy, patient identity, and a bag of past scans.

Run a controlled temporal/objective ablation using the same backbone, training patients, clean inputs, dimensions where practical, and readout protocol:

1. External-pretrained BRAINIAC with no longitudinal training: current image and paired/past-change features.
2. Mean/history pooling and ridge or shallow MLP.
3. Small GRU / last-visit residual MLP trained on the same frozen targets.
4. JEPA temporal training.
5. Task-supervised temporal training under a matched optimization budget.

Use order-shuffled history, truncated history, and missingness/scan-count-only controls. Use training-time shuffling controls where needed; inference-only corruption can measure distribution shift rather than order utility. Never shuffle future scans into context.

**Architecture issue worth testing:** `TemporalTransformer` adds time-since-previous-visit encodings but no absolute/relative visit-position encoding. Within a prefix, attention is permutation-equivariant: permuting earlier `(token, incoming-gap)` tuples while fixing the final token leaves its output invariant in evaluation mode. Prefix truncation prevents future-image leakage, but does not by itself encode the order of earlier visits. Test cumulative time or time-relative-to-prediction as a cheap, isolated alternative—not a wholesale CDE/ODE rewrite.

A second backbone by itself cannot attribute a gain to JEPA versus BRAINIAC. The needed design crosses **backbone × longitudinal objective**, as far as compute permits.

**Gate:** show a stable gain from the correct history over last-visit and matched small models, and an incremental gain from JEPA over the corresponding non-JEPA representation. Otherwise frame the result as pretrained representation transfer, not a world-model advance.

### E3 — Replace undifferentiated latent prediction with lesion-aware residual forecasting

**Main modeling bet, after E0/E2.**

The present objective can reward predicting stable anatomy or shrinking toward a population mean while failing on the small lesion components that matter. A22 already reports a mismatch between cosine error and clinical readout quality; its historical numeric details have their own cache/protocol caveats.

**Candidate state:** keep a global pretrained branch, but add lesion-specific quantities/features:

- enhancing and nonenhancing disease, edema/FLAIR abnormality, resection cavity;
- burden, shape, spatial extent, prior growth rate, current-to-historical-nadir change;
- modality availability and segmentation uncertainty;
- known treatment phase, time since intervention, and acquisition metadata where reliable.

**Start with an intentionally small residual model:**

```text
u_t = clinically grounded lesion state observed by t
u_hat(t + h) = u_t + g(H_t, h) * delta_theta(H_t, h)
```

Initialize the correction near zero. Compare global gain calibration, gap-conditioned shrinkage, ridge trend, and a small residual MLP before a probabilistic network. K3's magnitude/direction decomposition motivates this, but needs a refreshed, non-oracle evaluation. Magnitude-oracle diagnostics are not deployable forecasts.

For uncertainty, move to a regularized distribution over lesion changes—e.g. a near-stable component plus a broader change component—only if the deterministic residual model earns it. Use proper scores and interval coverage/sharpness; predicting huge uncertainty is not success. A direction that is uninformative cannot be rescued just by scaling it.

**Targets and losses:**

- Future log-volume/measurement changes and, when independently annotated, new-lesion/change masks.
- Incident progression probability at a prespecified horizon.
- Optional fixed-teacher lesion-feature prediction as regularization, with the clinical targets retained as an external check.

Use masks/measurements from future visits as *training targets*, never as forecast inputs. Preserve units and registration transforms. `brainiac_contract.py` resizes whole brains to 96³ and writes an identity affine; those outputs are not a reliable physical-volume ruler. Extract radiomics/volumes in valid native/common physical geometry, not from identity-affine resized tensors as if their voxels were truly 1 mm³.

**Evaluation:**

- Same-pair lesion persistence, past linear/log-growth, population/site shrinkage, last-RANO/transition models, and small GRU.
- Fixed teacher space for all latent comparisons; do not compare raw cosine levels from different learned spaces as if the metric were invariant.
- Report MAE/proper scores, calibrated progression risk, patient-uniform and pair-pooled improvements, and stable versus changing strata. Stratification by future change is diagnostic only, not a mechanism for selecting “easy” test cases.
- Mask forecasting: include lesion/changed-region metrics and the no-change mask baseline. Whole-brain SSIM can be excellent while clinically relevant change is wrong.

**Go criterion:** meaningful gains over persistence/trend on clinical outputs and a new site, without sacrificing stable patients to improve a few dramatic transitions. If only cosine improves, reject the claimed clinical advance.

### E4 — Add the information the current corpus cannot supply

External data is not just a scaling slogan. Choose each source for a particular missing supervision signal and reserve evaluation roles before training.

| Dataset | Verified opportunity | Limitations and intended role |
|---|---|---|
| **UCSF-ALPTDG** [S7] | 298 patients, two post-treatment scans each; expert tissue and longitudinal change annotations; 80 GBM patients; treatment/outcome metadata | Best first source for lesion/change pretraining. Only two visits, mixed diffuse-glioma diagnoses, and pairs selected for interval changes: not an unbiased long-trajectory or natural-prevalence calibration cohort. |
| **Burdenko-GBM-Progression** [S8] | 180 primary GBM patients; planning imaging plus 1–8 follow-ups; variable scanners; treatment-response categories and RT information | Stronger candidate for actual temporal generalization and treatment-effect research. Audit visit-level label completeness/definitions and registration; the collection page gives differing summaries of response-label coverage. No assumption of equivalent four-class RANO. |
| **UCSD-PTGBM** [S9] | 178 patients / 243 timepoints; expert segmentations, structural plus diffusion/perfusion; 192 tumor-positive and 51 treatment-change-only timepoints | Useful auxiliary task for tumor versus treatment effect. Not 178 long sequences, not four-class RANO, and intentionally enriched/adjudicated rather than representative surveillance sampling. |
| **SAILOR** | Existing rich masks, treatment, perfusion and dose data | Useful developmental transfer/low-shot study, not a pristine external test after all prior probing. Respect controlled-access restrictions and verify codebook/times. |
| **BraTS progression hidden evaluation** [S4, S5] | Published external evidence now exists beyond local LUMIERE CV | Best assessment validation if organizers permit a new evaluation. Access and input/output rules must be confirmed. |

**Recommended data experiment:** pretrain a lesion/change branch on the permitted UCSF training subset; evaluate measurement and change sensitivity on held-out UCSF patients; then test incremental benefit on the corrected LUMIERE task. Keep global BRAINIAC fixed initially. For long-trajectory forecasting, inventory Burdenko and lock a test role before touching its outcomes. Do not consume every institution for pretraining and later call one of them external.

Important traps:

- UCSF's descriptor reports **62 patients overlapping an earlier preoperative dataset**. Deduplicate across pretraining and evaluation sources; audit segmentation-model training overlap too.
- UCSF images were coregistered to the second timepoint before atlas registration. For a strict baseline-only forecast, audit whether supplied baseline preprocessing depends on future information; use raw/prefix-causal processing where available, or disclose and restrict the claim.
- Resection cavity is not necrosis; FLAIR abnormality is not a pure tumor label. Harmonize concepts explicitly, retain site-specific mappings, and quantify segmentation disagreement.
- Derive treatment/progression targets from documented clinical adjudication, not a universal volume threshold. Do not convert pseudoprogression labels into a fabricated RANO-4 label set.
- Estimate *usable independent patients and transitions* before committing to downloads/compute. Thousands of slices and many correlated prefixes are not thousands of independent patients.
- Respect each source's current access/license terms. No external cohort acquisition or controlled-data transfer was performed for this review.

## 6. The clinically valuable extension: treatment effect versus true progression

This is a better long-term discriminator than another decimal on latent cosine loss, but it requires labels not currently supplied reliably by LUMIERE RANO alone.

RANO assessment depends on reference scans, confirmation rules, clinical status, treatment timing, and sometimes steroids—not only tumor volume [S6]. Enhancing disease can increase without true tumor progression. A whole-brain image-only forecast cannot identify an unobserved treatment effect by architectural cleverness alone.

Research a **measurement-and-treatment-effect model**, not an unconstrained “causal world model”:

1. Learn enhancing/nonenhancing burden, cavity, and lesion changes using external annotations.
2. Add documented RT/TMZ timing and, on cohorts that have it, diffusion/perfusion as auxiliary evidence.
3. Evaluate tumor versus treatment effect on independently adjudicated outcomes, keeping pathological and clinical confirmation strata separate.
4. Forecast risk under *observed clinical practice* first.

Do not claim counterfactual treatment benefits from simply swapping a treatment token. Observational treatment selection, near-uniform Stupp therapy, sparse action support, and treatment changes triggered by worsening violate the assumptions needed for causal identification. RANO class is an outcome, not an intervention. The existing phase-conditioning null is not proof that treatment is irrelevant, nor proof that an identifiable treatment-effect model exists.

## 7. What “beat SOTA well” should mean

### Evidence standard

- Same task, information cutoff, permitted clinical inputs, cohort eligibility, mask access, and patient split.
- A faithfully reproduced strong baseline, not only a literature number and not a collapsed from-scratch CNN.
- Fixed primary metric, class ordering, missing-class convention, and readout ensemble/seed policy before selection. Report both pooled out-of-fold macro-F1 and fold mean/spread when comparing against papers reporting the latter; they are not mathematically equivalent.
- Paired patient-cluster confidence intervals on score differences; multiple training seeds reported separately. Bootstrap intervals conditional on a trained model do not include full training/selection uncertainty.
- An untouched external confirmation, with explicit zero-shot versus adaptation regimes. Any site normalization must be fitted on allowed support data, not future query visits. CORAL using an entire target cohort is transductive adaptation, not pristine zero-shot transfer.
- For K-shot studies: fixed query patients, nested support-patient sets, matched rows/classes, and paired repeats. The current `cross_site.py` changes query composition with K and fits a fresh support-only readout; that is not automatically a monotonic adaptation curve from the zero-shot classifier.
- AUROC **and** AUPRC with prevalence, calibration/proper scores, per-class recall, and prespecified sensitivity/specificity operating points. Choose thresholds on development data only.

### Ambition, not fabricated guarantees

For assessment, an engineering target such as **at least +0.05 absolute macro-F1 over the strongest matched reproduction**, followed by a positive external difference, is more worthwhile than chasing +0.005 on the familiar 13 patients. This is a proposed practical margin, not a power calculation or an assurance it is attainable. Investigate the reported 0.616 F1 before defining an absolute “SOTA” threshold.

For forecasting, prioritize a prespecified useful reduction in lesion-change prediction error/proper score and improved incident-progression risk over last-status/trend, with calibration preserved. Derive the practical margin with clinical input. A defensible new-task benchmark can be valuable even without a directly comparable published forecasting SOTA; call it that, not an assessment leaderboard victory.

For fixed-horizon progression, handle interval-censored detection and censoring explicitly. A patient lost to follow-up before the horizon is not a negative; death before documented progression is not ordinary stable disease. Consider a landmark survival/discrete-hazard model before a complex generative model.

## 8. Concrete sequence and stopping rules

| Order | Work package | Deliverable / decision |
|---|---|---|
| 1 — CPU days | E0: labels, cutoff manifests, AUC validation, survival-input audit, matched rows | Trustworthy developmental baseline; list which historical conclusions survive |
| 2 — CPU days | Past-only label/growth/current-image models; gain/shrinkage calibration; source-method verification | Decide whether there is a demonstrated temporal advantage and establish the actual comparison target |
| 3 — cached CPU + one focused GPU budget | E1 assessment reproduction + E2 small temporal/objective controls | Either a competitive SOTA-family model or a precise explanation of remaining reproduction gaps |
| 4 — access/inventory in parallel | UCSF/Burdenko/UCSD schema, overlap, labels and data-use review | Counts of usable cases and a locked train/development/external-test allocation |
| 5 — gated training | E3 lesion-aware residual forecast; E4 external lesion/change supervision | Test added information, not just added capacity; evaluate on clinical endpoints |
| 6 — one final evaluation | Freeze all choices; new institutional/hidden test | External comparison and calibration, including negative results |

**Do not prioritize now:**

- More long champion resumes, larger horizon heads, or simultaneous bundles of augmentation/weighting changes. The record already says these are low-information bets.
- Full diffusion/voxel generation before a lesion/past-trend model works. Background reconstruction is not the bottleneck.
- Broad site-adversarial alignment to make embeddings indistinguishable. Disease/phase distributions differ too; removing site information can erase biology. First compare mean-only, scalar gain, covariance alignment, and small support-only calibration on fixed targets.
- Treating one failed 22-feature concat as a reason to abandon explicit measurements or high-resolution lesion information.
- Promoting raw surprise as early warning, or permanently abandoning it based on unrefreshed AUCs.
- Treatment counterfactuals or deployment claims from 27 observational SAILOR subjects.
- More infrastructure abstraction. The harness is sufficient to implement these experiments once the scientific row/label/score contracts are explicit.

**If only three things get done:** repair the task/metric/label contract; reproduce the paired-scan radiomics baseline and test history on top; obtain expert lesion-change supervision from an additional cohort. Those actions are much more likely to change the scientific answer than another JEPA optimization variant.

## Sources and reading assignments

These links support the externally sourced claims above. Numbers from abstracts/reports are identified as such; they do not imply a full reproduction or a complete leaderboard audit.

- **[S1] Tikhonov et al., hybrid response assessment.** Read Methods §§2.1–2.3 and Results §3, especially growth/shrinkage-mask construction and the pretrained encoders: https://arxiv.org/html/2509.06511v1
- **[S2] TRACE.** Read §§3–5, Table 3 and external-data appendix A.6; inspect predicted versus passthrough concepts and the low performance of threshold-only rules even with supplied concepts: https://arxiv.org/html/2606.30313v1
- **[S3] Amato et al., Integrating Deep Learning and Radiomic Features for Glioblastoma Treatment Response Classification.** Primary publication DOI: https://doi.org/10.1109/BIBM66473.2025.11356532 . Public supporting project report, §2.29/p.92: https://www.fondazionedare.it/wp-content/uploads/2026/01/SP1-D4.4-Report-on-developed-computational-models.pdf . Full primary experimental protocol remains a research task; TRACE Table 3 labels the reported F1 as macro-F1.
- **[S4] Maurya et al., Swin UNETR + radiomics/TabM assessment.** Publisher abstract verifies the balanced-accuracy and external-cohort figures: https://doi.org/10.1007/978-3-032-16370-7_22
- **[S5] Gao et al., Siamese Vision Transfer architecture.** Publication: https://doi.org/10.1007/978-3-032-16370-7_21 . Institutional abstract: https://scholar.nycu.edu.tw/zh/publications/a-siamese-vision-transfer-architecture-for-prediction-of-brain-tu/
- **[S6] Wen et al., RANO 2.0.** Baseline choice, post-RT confirmation, clinical factors, and limitations of an imaging-only label: https://pmc.ncbi.nlm.nih.gov/articles/PMC10860967/
- **[S7] UCSF Adult Longitudinal Post-Treatment Diffuse Glioma MRI dataset.** Read cohort selection, longitudinal annotations, preprocessing, overlap, and benchmark availability: https://pmc.ncbi.nlm.nih.gov/articles/PMC11294954/
- **[S8] Burdenko-GBM-Progression.** Official inventory, dates, registration limitations and access conditions: https://www.cancerimagingarchive.net/collection/burdenko-gbm-progression/
- **[S9] UCSD-PTGBM.** Official cohort inventory and tumor-versus-treatment-change definitions: https://www.cancerimagingarchive.net/collection/ucsd-ptgbm/ ; descriptor: https://www.nature.com/articles/s41597-025-06499-z
- **[S10] TaDiff.** Read the final paper's split and available-history definitions before proposing a matched future-mask comparison: https://arxiv.org/abs/2309.05406 ; https://doi.org/10.1109/TMI.2025.3533038

**Bottom line:** the likely route to a substantial advance is a correctly benchmarked combination of pretrained imaging, explicit spatial lesion change, and additional post-treatment supervision. Preserve JEPA only where controlled experiments show it contributes. The research goal is better assessment and forecasting—not defending the current architecture.
