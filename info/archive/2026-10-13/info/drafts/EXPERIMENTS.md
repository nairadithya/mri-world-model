# Experiments and evidence

Draft, 2026-10-11. Existing canonical records still govern. This reference consolidates evidence and corrections without replacing the originals. Read [TODO](TODO.md) for current work, [Notes and insights](NOTES_AND_INSIGHTS.md) for interpretation, and [Methods](METHODS.md) for contracts.

<a id="current-evidence"></a>
## Current evidence and withdrawals

### Current scoped findings

- **RANO-free anatomy forecasting remains negative for the existing frozen representations and deterministic lesion features.** On the locked encoder-unseen LUMIERE cohort, P1 persistence-relative patient-uniform MAE was 1.000, versus 1.393 for the JEPA state. On unchanged SAILOR transfer, it was 1.000 versus 2.873. In the matched P2 physical-feature experiment, the residual MLP was effectively tied with persistence in LUMIERE (relative MAE 0.997; paired 95% CI for its difference from persistence, −0.096 to +0.100) and worse in SAILOR (1.288; CI +0.176 to +0.369). Adding frozen vision or JEPA features did not resolve the problem. See [P1 anatomy baselines](../p1_anatomy_baselines.md), [P2 execution](../p2_execution.md), and [A38 to A39](../ablations.md).
- **R35 completed a separate observed-state and observed-change representation comparison.** Spatial state learning reduced in-domain automated-volume error, but a dedicated paired head did not establish incremental value over independent state differences. Nuisance separation and SAILOR developmental transfer were not established. This was not forecasting or biological-change validation. See [R35](#r35).
- **R36 Phase 1 is incomplete.** Its twelve training checkpoints and ten seed-42 source-only probes were locally validated, but comparative query-quality scores do not exist. At the latest receipt, seed 43 was running and seed 44 had not launched. Phase 2 is unauthorized. See [R36](#r36).

### Explicit withdrawals and limits

A40 invalidated learned P2.4 forecasting claims: the forecaster used the target visit's image/mask features and ending gap. Its reported in-domain gains, seed-stability claim, SAILOR transfer results, and support-adaptation forecasts are withdrawn. Its old learned cache also bridged missing-mask visits, producing 123 LUMIERE rows instead of the audited 117. The corrected history-only rerun is approximately tied with persistence, not a replacement success. Its historical relative MAE is 1.00595 with a paired interval spanning zero on 117 rows. That rerun predates the aligned-feature contract. No score from a new aligned causal training run is established. Frozen P1 and P2.2/P2.3 results are not invalidated by the target-input bug.

Historical 2.7-fold and 82/91 dynamics headlines are corrected by A8/A24: the original comparison mixed representation spaces, and the same-space in-domain gate is not a general win. The historical "SAILOR ~14-day regime" claim is also corrected: 14 days was approximately the lower tail; recorded gaps had median 76 days. Legacy all-91 probe CV is not encoder-unseen CV. Single-split RANO gains are not robust evidence.

The old 0.77/0.87 surprise AUCs are withdrawn pending current-schema reruns. Corrected survival-free results do not support an early-warning claim: LUMIERE JEPA-error AUC is 0.4971 versus persistence-error AUC 0.4611 for next-visit PD, and at horizons two and three persistence performs better. SAILOR remains controlled-access developmental transfer, not untouched external validation. Its empirical RANO mapping is not independently confirmed.

These results address distinct tasks and cohorts. A scoped lesion experiment is not a universal definition of representation quality. The broader objective remains MRI representations that preserve anatomical and pathological state and meaningful longitudinal change while separating acquisition nuisance.

## Compact scorecards

### P0 and P1 developmental RANO results

The current-schema P0 cache has 91 LUMIERE patients and 393 RANO-labelled visits. The primary locked comparison is on the 26 encoder-unseen patients. Historical final patients are now development evidence, not an untouched test.

| Evaluation | Result | Scope |
|---|---:|---|
| Locked `states_forecast` vs fused, pooled macro-F1 | 0.2718 vs 0.2095; paired difference +0.0623 [0.0003, 0.1273] | Current-schema development |
| Locked `states_forecast` vs fused, patient-uniform macro-F1 | 0.2247 vs 0.1428 | Current-schema development |
| P1 last-RANO baseline | 0.2566 CV / 0.3868 historical-final transfer | 79 patients with forecast rows |
| P1 current-image baseline | 0.2388 / 0.2619 | Same rows |
| P1 current-volume baseline | 0.2180 / 0.2509 | Same rows |
| P1 clinical baseline | 0.1625 / 0.1726 | Survival-free cache |
| P1 past-volume trend | 0.1597 / 0.2725 | Same rows |

The current-schema P0 comparison is developmental, not an external headline. The historical-final slice has been reused. Source and cohort restrictions, bootstrap results, and temporal controls are recorded in [P0 refresh](../p0_refresh.md), [P0 audit](../p0_audit.md), and [P1 baselines](../p1_baselines.md).

### Pilot and training gates

| Run | Outcome |
|---|---|
| Corrupt-weight pilot | Invalid. Loss fell, but target standard deviation was 0.0024 and rank about 1.1. The community weight port was behaviorally corrupt. |
| Official-weight 5-epoch pilot | Loss and collapse-monitor checks passed; persistence gate failed (JEPA 0.0576, persistence 0.0043). |
| Official-weight 25-epoch pilot | Healthy monitors; best validation loss 0.0093. Persistence remained better on average (JEPA 0.0110, persistence 0.0043); held-out Patient-029 was the worst case. |
| Full-cohort champion | Best validation loss 0.0081; held-out test loss 0.0074. Initial all-91 comparison was later corrected; see A8/A24. |
| Continued training and task fine-tuning | Basin-hold investigations found transient accumulation-dependent holds, not durable continuation. RANO auxiliary fine-tuning damaged dynamics without improving honest CV. |
| A32 objective changes | The short resume was inconclusive; the proper from-scratch run did not improve locked RANO and significantly worsened the overall dynamics gate. |

The CPU pilot trained with official BrainIAC weights; the full cohort was trained on a Kaggle T4. These are distinct execution facts, not evidence that training scale alone solves the persistence gate. Details: [pilot](../pilot.md), [results](../results.md), and A1 to A8, A17, A33 to A34 in [ablations](../ablations.md).

## Historical experiment index

The entries below retain the A identifiers and distinguish positive, negative, incomplete, invalid, and corrected evidence. Numeric results are tied to their original cohort, cache, split, or protocol; do not compare them as if they shared one evaluation contract.

### Initial checks, pilots, and early probes

- A1, random-init sanity (2026-09-03). Plumbing passed on Patient-001; JEPA loss 0.978, with a valid train step and EMA update. Persistence loss was 0.636.
- A2, checkpoint audit (2026-09-04). The community HF port had 137/137 matching keys but near-identical outputs for real inputs and noise (drift 0.0002). Official weights showed real-vs-noise drift 0.0709. Key matching alone is insufficient.
- A3, pooling (2026-09-04). First-token and mean-pooling longitudinal drift were nearly identical across sequences; retained the official first-token convention.
- A4, preprocessing (2026-09-04). Raw/minimally processed and full-contract drift were similar on the tested patient. This was a narrow preprocessing-fidelity check.
- A5, broken-weight era (2026-09-04). Fresh and trained backbones had identical drift; the collapse was downstream in heads fed by corrupt constant features. Monitors detected collapse.
- A6, official-weight pilot (2026-09-04). JEPA 0.0576 versus persistence 0.0043. Learning signal existed, but the gate failed.
- A7, extended pilot (2026-09-04/05). Validation plateaued around 0.0093 after roughly epoch 13. Mean JEPA 0.0110 versus persistence 0.0043; one train patient was a near-win and another a win, but the held-out patient lost substantially.
- A8, hero gate and correction (2026-09-05/06). The first report claimed 0.0081 versus 0.0218 and 82/91 wins. The persistence side was in online-projector space while JEPA was in EMA-target space. Correct same-space results over 547 pairs were: train pooled 0.0082/0.0065, validation 0.0078/0.0075, test 0.0070/0.0088, overall 0.0080/0.0069. Patient-uniform test was 0.0074/0.0160; overall patient wins were 41/91, not 82/91. See A24 for paired uncertainty.
- A9, frozen RANO probes, 2026-09-05/06. The initial hero-split forecast MLP reached F1 0.448. Legacy all-91 five-fold patient CV gave 0.334 ± 0.060, but mixed encoder-training patients into probe-test folds. Neither number is a current encoder-unseen result. A25 and the current-schema scorecard provide the corrected protocol. The original single-split comparison had 49 test visits and rare PR/CR support.
- A10, surprise (2026-09-05). Legacy next-visit PD AUC 0.7677 was later superseded by the corrected AUC implementation and current-schema P0 results. It is not current early-warning evidence.
- A11, volume probes (2026-09-06). Ridge with train-fold tuning yielded readout R² about 0.15; next-visit volume forecast lost to persistence (MAE 1.52 vs 1.07; R² about 0.04). Plain least squares in the high-dimensional, small-row regime catastrophically overfit.
- A12, SAILOR transfer (2026-09-06). On the original cache, frozen dynamics lost to persistence. Discriminative readouts were initially reported as F1 0.37 and surprise AUC 0.87; A23 rebuilt-cache corrections and P0 metric repairs supersede these readout/AUC values. The historical interval-regime explanation was also corrected: typical SAILOR gap was not 14 days.
- A13, lead-time (2026-09-06). Legacy JEPA surprise did not outperform persistence at predicting incident PD at horizons two or three. Current-schema P0 refresh further withdraws the old surprise numbers.
- A14, transition atlas (2026-09-06). Surgery-involved transitions had short median gaps and high error; SD→PD onset lost to persistence. CR→CR showed over-predicted change. Transition-level sample sizes and historical cache constrain interpretation.
- A15, treatment-conditioned SAILOR field (2026-09-07). Five-fold subject CV on 243 pairs gave conditioned and unconditioned field error 0.0036 ± 0.0011 versus persistence 0.0040 ± 0.0011. Treatment conditioning added no measurable gain. This pre-rebuild field result is not a current-schema result.
- A16, freezing battery (2026-09-07). Legacy diagnostics found strong site separability, partial CORAL recovery, and image-space change differences. These diagnose possible shift, not a validated correction or universal mechanism.
- A17, basin-hold (2026-09-07). Batch accumulation slowed ejection from the champion basin and short accumulated legs briefly improved validation loss, but longer runs ejected. Momentum-state comparison was initially untestable because optimizer state was absent. Later clean rescues did not establish an anatomy forecasting gain.
- A18, raw-vs-MNI changes (2026-09-08). On 939 modality-pair slots, median image change was 0.673 raw versus 0.173 MNI, about 3.9×. This is total-pipeline change, not PLHM-only; raw and processed masking/registration differ.
- A19, SAILOR reprocessing (2026-09-08). Reprocessing through the LUMIERE contract did not restore the dynamics gate. Skull-in transfer readout fell below majority. A later skull-out result was subsequently found unreproducible.
- A20, preprocessing QA (2026-09-08). Automated QA caught two known registration casualties. NCC was downgraded to warning after false alarms on healthy volumes.
- A21, skull-restored reprocessing (2026-09-09). An initial readout-recovery interpretation was withdrawn by A23 because the skull-out cache disagreed with a clean rebuild. The same-space dynamics failure persisted; do not cite A21 readout/ceiling columns.
- A22, predicted-latent probes (2026-09-09/10). Legacy caches suggested predicted states could carry readout signal despite cosine error rankings. The work also found probe-scale sensitivity and incomplete SAILOR modality-content checks. Treat as diagnostic, not current-schema evidence.
- A23, SAILOR content guard and rebuild (2026-09-10). A finite-nonzero content check rejected phantom modalities. Rebuilt same-space dynamics ratios were 7.11× (derivatives), 10.5× (skull-in reprocessed), and 8.95× (skull-out reprocessed); persistence won in all arms. Skull-out transfer-cache discrepancy made the old A21 readout unreproducible, with cause undetermined.
- A24, same-space and bootstrap gate (2026-09-10). On the LUMIERE champion cache, overall pooled paired difference was +0.0010 [−0.0000,+0.0022], not significant; test pooled difference −0.0017 [−0.0072,+0.0012], also not significant. Test patient-uniform difference was −0.0086 [−0.0182,−0.0009], significant on the selected aggregation; overall patient wins were 41/91. The broad claim "JEPA beats persistence" is unsupported.

### Locked evaluation and comparator studies

- A25, locked RANO protocol, 2026-09-11. The states-forecast MLP achieved within-unseen CV macro-F1 0.309 [0.255, 0.358]. Transfer trained the readout on 65 patients and evaluated the 26 encoder-unseen patients, giving 0.408 [0.299, 0.473]. Twenty-two unseen patients contributed forecast rows. This is not a final13-only result. It predates the survival-free current-schema reset and remains development evidence.
- A26, interface probes, 2026-09-11. ROI pooling, modality concatenation, and volumetry did not improve the recorded locked result. The legacy independently resized ROI geometry predates the R34 affine repair. This is not a valid comparison of verified aligned lesion features and does not close the lesion-aware hypothesis. The small cohort and readout capacity also limit interpretation.
- A27, task-trained temporal head, 2026-09-11. Temporal task training did not beat frozen readout. At fixed hidden width 256, readout seeds gave final13 macro-F1 0.382 to 0.448. Across seed and width configurations, the range was 0.335 to 0.448. The lower value used hidden width 64. This reused slice demonstrates readout-selection sensitivity, not a new holdout result.
- A28, supervised LoRA (2026-09-11). The properly optimized v2 did not improve locked results: within-unseen CV 0.289 versus frozen 0.309; historical final 0.400 versus seed-42 frozen 0.448. Single seed and readout noise limit the comparison.
- A29, from-scratch 3D CNN (2026-09-12). Under-trained comparator; final F1 0.222. Not a fair SOTA-family comparison.
- A30, from-scratch 2D CNN (2026-09-12). Majority-class collapse persisted; final F1 0.182.
- A31, ROI-cropped 2D CNN (2026-09-12). ROI crop did not rescue the from-scratch comparator; final F1 0.222. The three variants do not establish superiority over a faithful pretrained supervised baseline.
- A32, literature review (2026-09-12). Corrected the interpretation of A29 to A31: they were not faithful reproductions. Matoso's reported 0.51 was balanced accuracy, not macro-F1; reported median F1 was 0.1335. Suggested JEPA-side tests were hypotheses, not results.
- A33, short A32 resume (2026-09-12). Locked RANO results were effectively null, but the run was a weak test: fresh optimizer, epoch-1 selection, and altered loss scale.
- A34, A32 from scratch (2026-09-13). RANO representation tied within noise, while the same-space dynamics gate was significantly worse overall than the frozen champion. The bundled objective changes did not help.
- A35, reduced radiomics/growth comparator (2026-09-13). A 22-feature comparator reached locked CV F1 0.217 versus JEPA 0.309 and did not beat JEPA in SAILOR K-shot comparisons. It is not the full radiomics-plus-CatBoost SOTA.
- A36, concept probe (2026-09-13). Measurement/growth concepts were weakly recoverable from frozen latents; post-hoc concatenation did not improve the locked CV metric. Small transfer slices showed capacity-sensitive swings.
- A37, transfer mechanism audit (2026-09-09/14). Legacy cache diagnostics measured SAILOR prediction gain overshoot and weak direction, but are not current-schema or deployable calibration results.

### Anatomy, leakage audit, and clean rescues

- A38, physical lesion state (2026-09-22). On 496 LUMIERE and 242 SAILOR pairs, structured ridge and zero-initialized residual MLP did not resolve persistence-relative forecasting. LUMIERE relative MAEs were 1.008 and 0.997; SAILOR 1.898 and 1.288.
- A39, frozen feature fusion (2026-09-22). Adding frozen vision or JEPA state did not help. Relative MAE was 1.018/1.192 for vision and 1.040/1.426 for JEPA in LUMIERE/SAILOR, against structured-only 0.997/1.288. The small number of stable LUMIERE rows precludes a stable-stratum inference.
<a id="a40"></a>
- **A40, target-visit leakage, 2026-10-02.** Future-image and future-gap perturbations changed predictions in all 23 tested LUMIERE patients for each of seeds 42/43/44. Learned P2.4 forecasts and associated adaptation claims are withdrawn. Cache row construction also bridged missing visits. The later history-only rerun is tied with persistence within uncertainty, as recorded under R34 below.
<a id="a41"></a>
- **A41, clean gap and velocity heads, 2026-10-02.** On the clean teacher, gap-head historical LUMIERE one-step ratio was 0.358 versus persistence; the repeatedly used 13-patient split is development evidence. SAILOR zero-shot ratios were 7.409 champion, 6.056 gap, 1.051 velocity. Velocity approached persistence but did not demonstrate a patient-uniform win; pooled CI indicated harm. Recorded target gaps are retrospective metadata, so these are conditional latent diagnostics, not physical-volume forecasting.

## Clean rescues and observation-validity repairs

### R33 completed GPU rescue matrix

The [clean rescue record](../clean_rescues.md) prespecified independent reruns from the survival-free control `checkpoints/p0_clean_jepa/best.pt`. Its SHA256 is `7186a6267d7aa42720381e6c2780f9c86fe20e71067074f8eacdf63d48199d0a`. The four optimizer/accumulation arms, augmentation, surgery filtering, and transition weighting completed by 2026-10-03. The following anatomy readout values are reported as pooled persistence-relative MAE. They are not new patient-uniform bootstrap verdicts.

| Clean arm | JEPA-state relative MAE | Control relative MAE |
|---|---:|---:|
| Fresh optimizer, accumulation 1 | 1.3517 | 1.3932 |
| Fresh optimizer, accumulation 8 | 1.3791 | 1.3932 |
| Loaded optimizer, accumulation 1 | 1.3878 | 1.3932 |
| Loaded optimizer, accumulation 8 | 1.3925 | 1.3932 |
| Augmentation | 1.3864 | 1.3932 |
| Surgery filtering | 1.3963 | 1.3932 |
| Transition weighting | 1.3912 | 1.3932 |

Every reported ratio exceeds 1.0. None reaches persistence on that reported pooled endpoint. Loaded momentum does not rescue the result. The older pending launch boxes do not make these completed arms active work again. A41's completed gap/velocity head outcomes remain separate latent diagnostics. The source's earlier CPU alignment/field status is not a complete final outcome table, so no unrecorded CPU success is inferred here. Weights, predictions, configs, and logs remain under ignored `outputs/clean-rescues/`.

### R34 coordinate, chronology, and artifact repairs

The [methodology-validity record](../methodology_validity.md) showed a hemisphere mismatch from independently resizing LUMIERE masks as arrays. Affine-aligned masks changed the sampled LUMIERE placement. Three additional LUMIERE visits had legacy-versus-aligned union Dice 0.0000, 0.0181, and 0.0291. The three sampled SAILOR visits had Dice 1.0000. Native compartment volumes matched all six samples. These checks identify a coordinate defect, not a segmentation accuracy estimate.

The bounded implementation review closed at source `32310c166d79e892be629e977c6ab5e33a126df4`. Geometry, chronology, checkpoint-consumer, and real MRI numerical checks passed. The corrected historical causal-lesion rerun had relative MAE 1.00595, with a paired difference interval spanning zero on the legitimate 117 rows. Aligned features and elapsed-time semantics require new training before their forecasting contribution can be assessed. Historical leaked gains stay withdrawn, and old artifacts cannot be relabeled as aligned.

The 2026-10-08 whole-cohort QA covers 599 LUMIERE and 269 SAILOR measured visits. It resolved 3,451 MRI geometry records and checked all 2,604 measured compartment masks and native volumes. All LUMIERE compartment source grids differ from the MRI pre-grid; all SAILOR ONCO masks share it. Affine resampling is therefore required for the former. The audit did not execute voxel resampling for every visit.

Twenty-one reviewed overlays cover nineteen unique visits. The known extracranial label remains suspect. Across all LUMIERE compartment rows, one-empty-endpoint transitions contribute 24.9% of summed absolute log-volume change. In the locked unseen subset they contribute 32.9%, and in SAILOR 43.7%. These are automated target discontinuities, not confirmed biological events. ONCO-versus-CL median Dice is 0.650 for enhancement and 0.656 for edema, without independent repeat labels or expert adjudication. Source-grid and volume integrity are verified. Repeatability, biological validity, and a noise ceiling remain unknown.

## Current RANO-free anatomy evidence

P1's primary target is three `lesion-state-v1` log-volumes at the next observed visit. The locked encoder-unseen LUMIERE evaluation contains 117 rows from 23 patients; the full manifest has 496 LUMIERE rows. SAILOR transfer has 242 rows from 27 subjects. Persistence is the primary floor, with current-volume ridge as a comparator. SAILOR is unchanged developmental transfer, not an external holdout.

| Method | LUMIERE patient-uniform MAE | Relative | SAILOR MAE | Relative |
|---|---:|---:|---:|---:|
| Persistence | 1.545 | 1.000 | 0.943 | 1.000 |
| Current-volume ridge | 1.503 | 0.973 | 1.465 | 1.493 |
| JEPA state | 2.152 | 1.393 | 2.719 | 2.873 |

The current-volume ridge's LUMIERE improvement is unresolved (paired MAE difference −0.042; 95% CI −0.227 to +0.148); it is worse on SAILOR (+0.522; CI +0.384 to +0.658). JEPA is substantially worse than persistence, especially under transfer. This is next-observed-visit development evidence, not a fixed-horizon or clinical-response claim.

A38's physical-history residual MLP had LUMIERE MAE 1.5405 and SAILOR MAE 1.2143. Its apparent small LUMIERE improvement over persistence was unresolved; transfer was significantly worse. A39's frozen fusion branches did not add incremental signal. A40's future leakage means learned P2.4 metrics cannot be used as evidence for or against causal past-only forecasting.

## R35: observed state, observed change, and nuisance

<a id="r35"></a>
### R35 completed comparison

R35 is a completed comparison of MRI-only spatial state and observed-pair representations, originating in sibling worktree `world-model-representations` at revision `2f6e1f5a65be4ccbceac6de706b537de8b695faa`. Its report is [branch-only R35 record](../../../world-model-representations/info/r35_representation_comparison.md); that sibling path is local and not a durable canonical reference.

The study used 65 encoder-training and 26 encoder-unseen LUMIERE patients, with internal training/validation partitioning and three seeds, 42/43/44. The learned state encoder received dense native log-volume supervision. Pair models used two observed scans and observed-change targets, not future observations. Thus the results concern state decoding and observed change, not forecasting. The frozen reference used BRAINIAC without clinical inputs.

Across three seeds, direct spatial-state volume decoding had MAE 1.1145 versus 1.5000 for the frozen scalar-supervision control, difference −0.3855 [−0.5185, −0.2481], on 149 visits from 26 patients. The spatial model's dense supervision means this does not isolate architecture from supervision.

The direct paired head had observed-change MAE 1.2696 versus 1.5446 for zero change on 117 pairs from 23 patients, difference −0.2750 [−0.4953, −0.0957]. However, independent differences from state readouts had MAE 1.2589; paired minus independent difference was +0.0108 [−0.0791, +0.1066]. The paired architecture has not demonstrated incremental advantage. Results are means of patient-level errors over seeds, not an ensemble.

Nuisance separation was incomplete. In a synthetic resolution-degradation test, spatial-state difference false-change MAE was 0.3275 versus identity 0.0445, an excess of 0.2831 [0.1986, 0.3747]. This perturbation removes detail and is not proof that removed information was only nuisance. Noise and affine-intensity controls were near identity, but affine stability largely follows explicit standardization. Smooth bias increased false change. Empty-target errors were large, and automated segmentation differences across cohorts remain confounded.

On SAILOR, direct spatial-state MAE was 2.2401 versus 4.6876 for a matched frozen scalar-source52 readout, but it did not establish gain over the source-trained mean 2.1105: difference +0.1296 [−0.2571, +0.5098]. Source-26 and source-52 ridge results also had intervals crossing zero against their respective means. Independent state differences gave observed-change MAE 0.9665 versus zero-change 0.9430, difference +0.0235 [−0.0368, +0.1052]. The paired head scored 1.0508 and was worse than independent state differences by +0.0843, with family-12 interval [0.0295, 0.1440]. This is developmental transfer, not external validation.

R35 artifacts passed model, feature, row, and patient-disjointness checks. All twelve corrected arms completed 240 epochs, with 74,592 successful updates and 48 AMP backoffs. The first candidate run was invalidated by float16 dense-loss reduction overflow. A later AMP lifecycle failure occurred before a completed candidate epoch. Corrected finite-reduction and backoff-ordering artifacts are the graded runs. Earlier caches lack a creation-time MRI-byte fingerprint. Completion-time tensor hashes do not retroactively supply one. It did not establish biological validity, segmentation repeatability, nuisance invariance, or untouched external generalization. No future prediction claim follows.

## R36: delta-representation experiment status

<a id="r36"></a>
### R36 Phase 1 incomplete

The newer execution record is [R36 results](../../../world-model-r36/info/r36_delta_representation_results.md), from sibling worktree `world-model-r36` at revision `f4b96c00dd5d79c59ba02d186e024ff747f690ad`. The path is local and not a durable canonical reference.

R36 asks whether change-balanced supervision improves frozen local access to signed occupancy change. The primary contrast is C versus B under matched source52 local linear probes. Original R35 and continued state-only A are controls; D tests the nuisance tradeoff. The result, if completed, will concern observed change, not future predictability.

The immutable contract specifies four arms, three seeds (42/43/44), 20 final-graded epochs, source-only fitting and normalization, paired patient bootstrap, and a 24-GPU-hour total ceiling with nine-hour per-session ceiling. It preserves 599 visits and 496 usable consecutive rows. The encoder-unseen assessment includes 149 visits from 26 patients and 117 pairs from 23 patients. No pair bridges were introduced. SAILOR remains local.

Target QA found changed occupancy sparse: median 1,239.5 core, 524.5 enhancing, and 3,866 edema voxels, or 0.140%, 0.059%, and 0.437% of the 96³ grid. For core, enhancement, and edema, 7, 40, and 5 pairs respectively had both masks empty and zero signed change. No occupied pair had exactly unchanged targets. One native enhancing disagreement disappeared after resizing. These measurements motivate the contrast but do not establish biological change, segmentation accuracy, or that background dilution caused earlier results. A suspect extracranial label remained in the primary analysis; expert adjudication and a segmentation noise ceiling are unavailable.

The balanced loss, source-only probes, artifact binding, and real CLI numerical smoke were verified. Two finite optimizer updates ran on aligned MRI pairs; identity feature differences were exactly zero. Decoder reversal errors were 0.3865 and 0.3814 after those updates. These are smoke-test values, not quality scores.

The capacity session completed seed-42 A and B for exactly 20 epochs, with 5,940 pair exposures each. A had 5,935 successful updates and five AMP skips; B had 5,936 and four. Both R35 source-only probes completed 20 epochs. Ten remaining matrix arms were launched in two private T4 jobs. At the 2026-10-11 14:08 UTC receipt, all twelve final checkpoints had been downloaded and locally validated, and all ten seed-42 probes passed local validation. Seeds 43 and 44 probes remained.

At 18:30:51 UTC on 2026-10-11, continuation receipt reported seed 43 RUNNING and seed 44 not launched. No comparative query-quality scores had been calculated. Thus Phase 1 remains incomplete; no comparative verdict can be reported. A fresh reviewer had completed post-implementation review, which is not retrospective pre-training approval or final result acceptance. Independent result review and operator decision remain gates. Phase 2 is unauthorized. No computation was launched to update this draft.

## Earlier experiment details and source records

A1 to A41 are preserved above as an index; full setups, tables, and correction history remain in [ablations](../ablations.md). Additional primary records include [pilot](../pilot.md), [results](../results.md), [P0 refresh](../p0_refresh.md), [P1 baselines](../p1_baselines.md), [P1 anatomy baselines](../p1_anatomy_baselines.md), [P2 seed sensitivity](../p2_seed_sensitivity.md), [clean rescues](../clean_rescues.md), [methodology validity](../methodology_validity.md), and [P0 audit](../p0_audit.md).

The historical SAILOR reprocess procedure is preserved at [reprocess guide](../../src/preprocessing/reprocess_sailor.md). Its intended question was whether matching the LUMIERE preprocessing contract would restore dynamics transfer. A19's same-code reprocess did not flip the persistence gate; A21's skull-out readout-recovery interpretation was subsequently withdrawn because the rebuilt cache did not reproduce it. A23's rebuilt same-space ratios remain the relevant historical evidence. Reprocessing details and limitations remain in the guide and A18 to A24.

Numeric claims in this draft are scoped to the stated cache, task, cohort, and aggregation. Patient-level bootstrap intervals do not account for readout training-seed uncertainty unless stated. Developmental SAILOR transfer is not untouched external validation. Historical rows and scores are retained as history where corrected, invalid, or withdrawn; they must not be silently promoted to current evidence.