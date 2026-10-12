# Experiments and evidence

This reference records what the experiments measured and which claims remain valid. The wording was revised on October 12, 2026, using fixed copies of the original records. The latest R36 runtime snapshot remains October 11 at 18:30:51 UTC. This is not a live progress report.

Use [Notes and insights](NOTES_AND_INSIGHTS.md) for the reasoning, [Methods](METHODS.md) for protocols and term definitions, and [TODO](TODO.md) for remaining work. The [archive](archive/2026-10-12/README.md) preserves the original records and their correction history.

## How the results are reported

MAE is mean absolute error. Lower values are better. A persistence-relative error below 1.0 favors the model over persistence on that stated metric. A value above 1.0 favors persistence. Persistence predicts no change from the current measurement.

For an error difference, the convention is model minus comparator unless a row states otherwise. A negative difference favors the model. Brackets give a confidence interval. An interval that crosses zero leaves the comparison unresolved.

Patient-uniform results give each patient equal weight. Pooled results combine observations, so patients with more observations receive more weight. Relative scores retain the source's aggregation rule and cannot always be reconstructed by dividing two adjacent aggregate MAEs.

Macro-F1, AUC, R², and balanced accuracy measure different things. The [Methods glossary](METHODS.md#terms) defines them. Their values are not interchangeable.

An A identifier records a historical experiment or audit. R identifiers record research directions and later investigations. Completed execution, valid measurement, and scientific success are different statuses.

<a id="current-evidence"></a>

## Current evidence and withdrawals

### Current scoped findings

The tested RANO-free forecasts did not establish an advantage over persistence. In P1, JEPA-state patient-uniform relative MAE was 1.393 in LUMIERE and 2.873 in SAILOR, versus persistence at 1.000.

In P2, the physical-feature residual MLP had LUMIERE relative MAE 0.997. Its paired 95% confidence interval for the error difference was −0.096 to +0.100, so the apparent improvement remained unresolved. SAILOR relative MAE was 1.288, with interval +0.176 to +0.369, indicating worse transfer performance. Adding fixed vision or JEPA features did not fix the result.

The full records are [P1 anatomy baselines](documentation_sources/versions/4c077cc0f084f2d14b48b4cf366088ff7c0dc11a6ed3bf8b5678cbea901e141d.md), [P2 execution](documentation_sources/versions/9ea08b90cf4971fb7b85d4b150ef1317994f42aa2fca2f1c9c0f5de5424e38e4.md), and [A38 to A39](documentation_sources/versions/2ddad8d48f3ebb0e89c6855d2c8da8e2762d8291664d49d38faab77851d72549.md).

R35 completed a different task: state estimation and change between two observed scans. Its spatial model reduced in-domain automated-volume error. The dedicated paired head did not establish an advantage over subtracting separate state estimates. It did not establish nuisance separation or successful SAILOR transfer. It was neither forecasting nor validation of biological change. [The R35 section](#r35) gives the comparisons.

R36 Phase 1 was incomplete at the archived snapshot. Twelve training checkpoints and ten seed-42 source-trained readouts had passed local checks. No model-quality comparisons on evaluation patients had been scored. Seed 43 was running, seed 44 had not started, and Phase 2 was unauthorized. [The R36 section](#r36) records the dated status.

### Explicit withdrawals and limits

A40 withdrew the learned P2.4 forecasting claims. The forecaster used the later visit's image and mask features, plus the gap ending at that visit. Those inputs were not available at the earlier visit.

The withdrawal covers the reported in-domain gains, seed stability, SAILOR transfer, and forecasts after support-set adaptation. The old cache also skipped visits with missing masks, producing 123 LUMIERE rows rather than the audited 117.

The corrected history-only rerun had relative MAE 1.00595 on 117 rows, with a paired interval crossing zero. It was approximately tied with persistence, not a replacement success. That rerun preceded the corrected alignment protocol. No new forecast score from training with aligned features and valid past-only inputs is established. The target-input bug does not invalidate frozen P1, P2.2, or P2.3 results.

A8 and A24 correct the early 2.7-fold improvement and 82/91 patient-win claims. The original errors used different representation spaces. The corrected same-space comparison does not establish a general win.

The historical claim that SAILOR had typical gaps of about 14 days is also corrected. Fourteen days was near the lower tail. The recorded median was 76 days.

Legacy cross-validation over all 91 patients is not evaluation on encoder-unseen patients. A favorable single-split RANO score is not robust evidence.

The old surprise AUCs of 0.77 and 0.87 are withdrawn pending reruns with the corrected cache format and label rules. Surprise here means using prediction error as a signal of unexpected change.

The corrected results exclude eventual survival from the inputs. They do not support early warning: LUMIERE next-visit PD AUC was 0.4971 for JEPA error and 0.4611 for persistence error. Persistence performed better at horizons two and three.

SAILOR is controlled-access development transfer, not untouched external validation. Its empirical RANO code mapping lacks independent confirmation.

These results address distinct tasks and cohorts. A scoped lesion experiment is not a universal definition of representation quality. The broader objective remains MRI representations that preserve anatomical and pathological state and meaningful longitudinal change while separating acquisition nuisance.

## Compact scorecards

### P0 and P1 developmental RANO results

The current-schema P0 cache has 91 LUMIERE patients and 393 RANO-labelled visits. The primary locked comparison is on the 26 encoder-unseen patients. Historical final patients are now development evidence, not an untouched test.

The corrected P0 results use the fixed protocol:

| Evaluation | `states_forecast` macro-F1 | Fused macro-F1 |
|---|---:|---:|
| Pooled | 0.2718 | 0.2095 |
| Patient-uniform | 0.2247 | 0.1428 |

The pooled paired F1 difference is +0.0623 [0.0003, 0.1273]. For F1, higher values are better, so this positive difference favors `states_forecast`. These are development results under the corrected cache and label rules.

P1's baseline comparison uses 79 patients with forecast rows. Each column is a different evaluation:

| Baseline | CV macro-F1 | Historical-final transfer macro-F1 |
|---|---:|---:|
| Last RANO | 0.2566 | 0.3868 |
| Current image | 0.2388 | 0.2619 |
| Current volume | 0.2180 | 0.2509 |
| Clinical inputs without eventual survival | 0.1625 | 0.1726 |
| Past-volume trend | 0.1597 | 0.2725 |

The corrected SAILOR cache reversed the earlier apparent zero-shot advantage. Zero-shot means transfer without support-set adaptation. The recorded K-shot results use the support sizes defined in the source protocol.

| Support setting | JEPA macro-F1 | CNN control macro-F1 |
|---|---:|---:|
| Zero-shot | 0.123 | 0.261 |
| K=3 | 0.252 | 0.228 |
| K=5 | 0.241 | 0.225 |
| K=10 | 0.253 | 0.261 |
| K=15 | 0.264 | 0.264 |
| K=20 | 0.263 | 0.227 |

These are development RANO diagnostics with an unconfirmed SAILOR codebook. They are not anatomical transfer results. SAILOR persistence-error AUC was 0.6600 on 240 labelled pairs.

The matched temporal baselines completed on 393 forecast rows. The following macro-F1 values average pooled fold scores within the encoder-unseen cohort:

| Temporal baseline | Macro-F1 |
|---|---:|
| Scan-count | 0.2423 |
| Mean-history | 0.1322 |
| Last-visit MLP | 0.2236 |
| GRU | 0.2423 |
| Order-shuffled GRU | 0.2314 |
| Truncated-history GRU | 0.2522 |

An MLP is a feed-forward neural network. A GRU is a recurrent network for sequences. The matched JEPA comparison and temporal transfer to the historical final group were still pending in that source.

A separate 22-feature histogram-gradient-boosting proxy used observed scan pairs. It scored 0.3033 within-unseen and 0.5703 on historical-final transfer, with 43 final rows. That is assessment, not forecasting, and not a full CatBoost reproduction. See [P1 temporal controls](documentation_sources/versions/b8c596145365e19fbea3ed86218b7b7d7a3131e0be6e60915409d0eddcfddf31.md) and [P1 assessment audit](documentation_sources/versions/77de3ea25aa45d894d9577a96b69195912e6036883dace704af07b89de07843b.md).

The current-schema P0 comparison is developmental, not an external headline. The historical-final slice has been reused. Source and cohort restrictions, bootstrap results, and temporal controls are recorded in [P0 refresh](documentation_sources/versions/91f9eb3a5aa50c35bb9d8d79dbcca3e76e61bc3c93b814a08ae82b0a34ca4d8d.md), [P0 audit](documentation_sources/versions/da674d536622373c6db154365a73c67d5fffcc9646a10713641169a0b5cb2130.md), and [P1 baselines](documentation_sources/versions/b8c596145365e19fbea3ed86218b7b7d7a3131e0be6e60915409d0eddcfddf31.md). The [historical locked protocol](documentation_sources/versions/7873b012053d04798d82c190a754e9708031bedff4ce9cf4d2abb168553f4386.md) and its later readout-policy addendum are distinct from the current-schema audit; they are not one invariant cache or protocol.

### Pilot and training gates

Training checks can detect collapse, where the model outputs nearly constant features. Target standard deviation and rank measure feature variation. Healthy values do not by themselves establish better predictions than persistence.

| Run | Outcome |
|---|---|
| Corrupt-weight pilot | Invalid. Loss fell, but target standard deviation was 0.0024 and rank about 1.1. The community weight port was behaviorally corrupt. |
| Official-weight 5-epoch pilot | Loss and collapse-monitor checks passed; persistence gate failed (JEPA 0.0576, persistence 0.0043). |
| Official-weight 25-epoch pilot | Healthy monitors; best validation loss 0.0093. Persistence remained better on average (JEPA 0.0110, persistence 0.0043); held-out Patient-029 was the worst case. |
| Full-cohort champion | Best validation loss 0.0081; held-out test loss 0.0074. Initial all-91 comparison was later corrected; see A8/A24. |
| Continued training and task fine-tuning | Basin-hold investigations found transient accumulation-dependent holds, not durable continuation. RANO auxiliary fine-tuning damaged dynamics without improving honest CV. |
| A32 objective changes | The short resume was inconclusive; the proper from-scratch run did not improve locked RANO and significantly worsened the overall dynamics gate. |

The CPU pilot used official BrainIAC weights. The full-cohort run used a Kaggle T4. These facts describe where training ran. They do not show that more training data alone resolves the failure to beat persistence. Details: [pilot](documentation_sources/versions/d8e7bf20a34e6ee1186140c96bb090ed2df0a56f65f11ce140fe05acc4123436.md), [results](documentation_sources/versions/89605d7963ba3e0f942e5feb416846a32faea82a277f1dbe008b0cd0a6b86068.md), and A1 to A8, A17, A33 to A34 in [ablations](documentation_sources/versions/2ddad8d48f3ebb0e89c6855d2c8da8e2762d8291664d49d38faab77851d72549.md).

## Historical experiment index

The entries below retain the A identifiers and distinguish positive, negative, incomplete, invalid, and corrected evidence. Numeric results are tied to their original cohort, cache, split, or protocol; do not compare them as if they shared one evaluation contract.

### Initial checks, pilots, and early probes

- A1, random-initialization check, September 3, 2026. One training step and EMA update passed on Patient-001. JEPA loss was 0.978, and persistence loss was 0.636. This checked execution, not model quality.
- A2, checkpoint audit (2026-09-04). The community HF port had 137/137 matching keys but near-identical outputs for real inputs and noise (drift 0.0002). Official weights showed real-vs-noise drift 0.0709. Key matching alone is insufficient.
- A3, pooling (2026-09-04). First-token and mean-pooling longitudinal drift were nearly identical across sequences; retained the official first-token convention.
- A4, preprocessing (2026-09-04). Raw/minimally processed and full-contract drift were similar on the tested patient. This was a narrow preprocessing-fidelity check.
- A5, broken-weight era (2026-09-04). Fresh and trained backbones had identical drift; the collapse was downstream in heads fed by corrupt constant features. Monitors detected collapse.
- A6, official-weight pilot (2026-09-04). JEPA 0.0576 versus persistence 0.0043. Learning signal existed, but the gate failed.
- A7, extended pilot, September 4 and 5, 2026. Validation loss plateaued around 0.0093 after roughly epoch 13. Mean JEPA error was 0.0110 versus persistence at 0.0043. One training patient was close to a win and another was a win. The held-out patient was substantially worse.
- A8, full-run comparison and correction, September 5 and 6, 2026. The first report claimed error 0.0081 versus 0.0218 and 82 wins among 91 patients. Persistence used online-projector features, while JEPA used EMA-target features. Those errors were not directly comparable.

The corrected same-space comparison covers 547 pairs:

| Aggregation and split | JEPA error | Persistence error |
|---|---:|---:|
| Pooled training | 0.0082 | 0.0065 |
| Pooled validation | 0.0078 | 0.0075 |
| Pooled test | 0.0070 | 0.0088 |
| Pooled overall | 0.0080 | 0.0069 |
| Patient-uniform test | 0.0074 | 0.0160 |

The corrected overall count was 41 wins among 91 patients, not 82. A24 reports paired uncertainty.
- A9, frozen RANO probes, 2026-09-05/06. The initial hero-split forecast MLP reached F1 0.448. Legacy all-91 five-fold patient CV gave 0.334 ± 0.060, but mixed encoder-training patients into probe-test folds. Neither number is a current encoder-unseen result. A25 and the current-schema scorecard provide the corrected protocol. The original single-split comparison had 49 test visits and rare PR/CR support.
- A10, surprise (2026-09-05). Legacy next-visit PD AUC 0.7677 was later superseded by the corrected AUC implementation and current-schema P0 results. It is not current early-warning evidence.
- A11, volume probes (2026-09-06). Ridge with train-fold tuning yielded readout R² about 0.15; next-visit volume forecast lost to persistence (MAE 1.52 vs 1.07; R² about 0.04). Plain least squares in the high-dimensional, small-row regime catastrophically overfit.
- A12, SAILOR transfer (2026-09-06). On the original cache, frozen dynamics lost to persistence. Discriminative readouts were initially reported as F1 0.37 and surprise AUC 0.87; A23 rebuilt-cache corrections and P0 metric repairs supersede these readout/AUC values. The historical interval-regime explanation was also corrected: typical SAILOR gap was not 14 days.
- A13, lead-time (2026-09-06). Legacy JEPA surprise did not outperform persistence at predicting incident PD at horizons two or three. Current-schema P0 refresh further withdraws the old surprise numbers.
- A14, transition atlas, September 6, 2026. Transitions involving surgery had short median gaps and high error. The transition from SD to PD lost to persistence. The model overpredicted change between two CR visits. Small transition groups and the historical cache limit interpretation.
- A15, treatment-conditioned SAILOR field (2026-09-07). Five-fold subject CV on 243 pairs gave conditioned and unconditioned field error 0.0036 ± 0.0011 versus persistence 0.0040 ± 0.0011. Treatment conditioning added no measurable gain. This pre-rebuild field result is not a current-schema result.
- A16, freezing battery (2026-09-07). Legacy diagnostics found strong site separability, partial CORAL recovery, and image-space change differences. These diagnose possible shift, not a validated correction or universal mechanism.
- A17, continued training near the champion solution, September 7, 2026. Accumulating gradients delayed the rise in validation loss. Short runs briefly improved it, but longer runs did not retain the improvement. The first momentum-state comparison was impossible because the optimizer state was missing. Later clean reruns did not establish better anatomy forecasting.
- A18, raw-vs-MNI changes (2026-09-08). On 939 modality-pair slots, median image change was 0.673 raw versus 0.173 MNI, about 3.9×. This is total-pipeline change, not PLHM-only; raw and processed masking/registration differ.
- A19, SAILOR reprocessing (2026-09-08). Reprocessing through the LUMIERE contract did not restore the dynamics gate. Skull-in transfer readout fell below majority. A later skull-out result was subsequently found unreproducible.
- A20, preprocessing QA (2026-09-08). Automated QA caught two known registration casualties. NCC was downgraded to warning after false alarms on healthy volumes.
- A21, skull-restored reprocessing (2026-09-09). An initial readout-recovery interpretation was withdrawn by A23 because the skull-out cache disagreed with a clean rebuild. The same-space dynamics failure persisted; do not cite A21 readout/ceiling columns.
- A22, predicted-latent probes (2026-09-09/10). Legacy caches suggested predicted states could carry readout signal despite cosine error rankings. The work also found probe-scale sensitivity and incomplete SAILOR modality-content checks. Treat as diagnostic, not current-schema evidence.
- A23, SAILOR content guard and rebuild (2026-09-10). A finite-nonzero content check rejected phantom modalities. Rebuilt same-space dynamics ratios were 7.11× (derivatives), 10.5× (skull-in reprocessed), and 8.95× (skull-out reprocessed); persistence won in all arms. Skull-out transfer-cache discrepancy made the old A21 readout unreproducible, with cause undetermined.
- A24, same-space and bootstrap gate (2026-09-10). On the LUMIERE champion cache, overall pooled paired difference was +0.0010 [−0.0000,+0.0022], not significant; test pooled difference −0.0017 [−0.0072,+0.0012], also not significant. Test patient-uniform difference was −0.0086 [−0.0182,−0.0009], significant on the selected aggregation; overall patient wins were 41/91. The broad claim "JEPA beats persistence" is unsupported.

### Historical multi-horizon head study

The legacy gap-conditioned readout kept the encoder fixed. It used 2,793 pairs with inverse visit-count weighting, written as 1/n in the source. Its reported test errors were:

| Horizon | Frozen readout error | Persistence error |
|---|---:|---:|
| One | 0.0050 | 0.0088 |
| Five | 0.0065 | 0.0134 |
| Nine | 0.0116 | 0.0190 |

Few patients supported the farther horizons. The jointly trained encoder and head had best weighted validation loss 0.0170 and weighted test loss 0.0147. Its one-step error was 0.0137, versus persistence at 0.0088 and the frozen readout at 0.0050.

Legacy RANO cross-validation remained approximately unchanged, at 0.323 versus 0.334. Size readout remained near R² 0.15. Larger frozen heads and longer training did not improve the horizons with useful sample support.

The source table labels some horizon-three-to-five cells as ties or wins even though the listed errors exceed persistence. At horizon four, for example, model error is 0.0144 versus persistence at 0.0109. Those labels conflict with the displayed arithmetic and are not accepted here.

The table remains historical evidence, not an accepted broad claim of better far-horizon prediction. These are legacy latent-feature diagnostics conditioned on recorded gaps, not prospective physical-volume forecasts. The measured-gap correction also replaces this section's historical 14-day SAILOR explanation. See [historical multi-horizon record](documentation_sources/versions/89605d7963ba3e0f942e5feb416846a32faea82a277f1dbe008b0cd0a6b86068.md#follow-up-multi-horizon-forecasting-gate-passed-full-run-open).

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

- A38, physical lesion state, September 22, 2026. The manifest contains 496 LUMIERE pairs and 242 SAILOR pairs. Structured ridge had relative MAE 1.008 in LUMIERE and 1.898 in SAILOR. The zero-initialized residual MLP had 0.997 and 1.288. Neither established the intended forecasting advantage.
- A39, frozen feature combination, September 22, 2026. Adding fixed vision or JEPA features did not help. The comparison was:

| Input features | LUMIERE relative MAE | SAILOR relative MAE |
|---|---:|---:|
| Structured physical history only | 0.997 | 1.288 |
| Physical history plus vision | 1.018 | 1.192 |
| Physical history plus JEPA state | 1.040 | 1.426 |

Too few LUMIERE rows were stable to support a conclusion about the stable group.

<a id="a40"></a>

- A40, target-visit leakage, October 2, 2026. Changing future images or future gaps changed predictions in all 23 tested LUMIERE patients for each of seeds 42, 43, and 44. The learned P2.4 forecasts and related adaptation claims are withdrawn. Cache construction also skipped missing visits. The later history-only rerun was tied with persistence within uncertainty, as recorded under R34 below.

<a id="a41"></a>

- A41, clean gap and velocity heads, October 2, 2026. With the clean teacher, the gap head had historical LUMIERE one-step relative error 0.358. The reused 13-patient split is development evidence. SAILOR zero-shot relative errors were 7.409 for the champion, 6.056 for the gap head, and 1.051 for velocity. Velocity approached persistence but did not show a patient-uniform win. Its pooled confidence interval indicated harm. The target gaps were known retrospectively, so these are gap-conditioned latent diagnostics, not physical-volume forecasts.

### Historical frozen lesion-local diagnostics

P2.3 audited how much lesion information reached the encoder's spatial tokens. A token summarizes a region of the input. Fewer than two effective enhancing tokens were available in 40.4% of LUMIERE visits and 42.4% of SAILOR visits. For core, the corresponding values were 31.7% and 64.7%.

The crop rule used two tokens and 25% of visits as thresholds. The rule was chosen after those statistics were examined. It was not registered in advance. The resulting 64-voxel crop was resized to BRAINIAC's 96-cubed input.

The frozen-feature forecast results were:

| Feature set | LUMIERE relative MAE | SAILOR relative MAE |
|---|---:|---:|
| Compartment | 1.001 | 1.263 |
| Compartment minus surrounding ring | 0.971 | 1.867 |
| Combined residual features | 1.005 | 1.286 |
| Last transition | 0.994 | 1.368 |
| Transition history | 0.991 | 1.288 |
| Observation plus transition | 0.987 | 1.423 |

The apparent LUMIERE improvement at 0.971 had paired difference interval [-0.164, +0.092], so it remained unresolved. Every LUMIERE paired interval crossed zero, and every SAILOR forecast arm was worse than its comparator.

Combined tokens also decoded current anatomy. LUMIERE MAE was 1.141 and variance-weighted R² was 0.691. SAILOR MAE was 2.179 and R² was -0.155. Those state-readout scores are separate from the forecast scores.

A40's learned-forecaster bug does not invalidate these frozen arithmetic diagnostics. However, their legacy crop geometry predates R34's alignment repair. They are not evidence about verified aligned lesion representations. See [P2 lesion architecture](documentation_sources/versions/3e07b40bdd3849468d6b4c5f52a2d997fb891e3497ccd3d7b23b43aff4980007.md).

### Withdrawn learned-model seed and adaptation record

The P2.4 runs used trainer `18d211d`, eight anatomy epochs, stopping based on internal-development results, and seeds 42, 43, and 44. Their withdrawn historical scores were:

| Seed | LUMIERE MAE | LUMIERE relative MAE | SAILOR MAE | SAILOR relative MAE |
|---|---:|---:|---:|---:|
| 42 | 1.1833 | 0.759 | 0.9872 | 1.047 |
| 43 | 1.2125 | 0.778 | 1.1132 | 1.181 |
| 44 | 1.2375 | 0.794 | 0.9997 | 1.060 |

A40 withdraws every row as evidence of past-only forecasting, including the apparent three-seed stability. Completed execution does not make a forecast valid.

The recorded seed-43 execution took 41m18s after an unexplained earlier sequence of stalled and deleted kernels. Support-adaptation and transductive results from these checkpoints are also withdrawn. Transductive fitting uses information from the evaluation set and is distinct from zero-shot transfer. Full historical tables and recovery chronology remain in [P2 seed sensitivity](documentation_sources/versions/9392195ed8cf1ac4fea44e0347230844278803dbd3b21318ca48a6c7c0b2f841.md) and [P2 lesion architecture](documentation_sources/versions/3e07b40bdd3849468d6b4c5f52a2d997fb891e3497ccd3d7b23b43aff4980007.md).

## Clean rescues and observation-validity repairs

### R33 completed GPU rescue matrix

The [clean rescue record](documentation_sources/versions/eaca551880e0a9db5c4bdc7785e135675cd518ebfcd361d5aa482beca4f37dce.md) prespecified independent reruns from the survival-free control `checkpoints/p0_clean_jepa/best.pt`. Its SHA256 is `7186a6267d7aa42720381e6c2780f9c86fe20e71067074f8eacdf63d48199d0a`. The four optimizer/accumulation arms, augmentation, surgery filtering, and transition weighting completed by 2026-10-03. The following anatomy readout values are reported as pooled persistence-relative MAE. They are not new patient-uniform bootstrap verdicts.

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

A41 reports the following persistence-relative errors. Horizons count later observed visits, not fixed numbers of days.

| Method and cohort | Horizon one | Horizon two | Horizon three |
|---|---:|---:|---:|
| Champion, LUMIERE historical test | 0.449 | 0.780 | 1.075 |
| Gap head, LUMIERE historical test | 0.358 | 0.518 | 0.579 |
| Velocity head, LUMIERE historical test | 0.913 | 0.718 | 0.694 |
| Champion, SAILOR zero-shot | 7.409 | 5.062 | 3.939 |
| Gap head, SAILOR zero-shot | 6.056 | 4.147 | 3.213 |
| Velocity head, SAILOR zero-shot | 1.051 | 1.109 | 1.156 |
| Gap refit, held-out SAILOR | 1.127 | 0.838 | 0.696 |

Beyond horizon one, the champion reuses its one-step output. It does not roll the prediction forward. LUMIERE support shrinks from 13 patients to 10 and then 8. SAILOR support is 27, 27, and 26 patients. All paired patient-uniform intervals for the held-out SAILOR gap refit cross zero.

The recorded target gaps are known retrospectively. These results describe reconstruction conditioned on those gaps, not forecasts at horizons known in advance.

### R34 coordinate, chronology, and artifact repairs

The [methodology-validity record](documentation_sources/versions/7bc218e8ef5fd2fdb22bc0ed7f5bab72c4b1d099f1f7f2c3145f6c3bca6f3472.md) showed a hemisphere mismatch from independently resizing LUMIERE masks as arrays. Affine-aligned masks changed the sampled LUMIERE placement. Three additional LUMIERE visits had legacy-versus-aligned union Dice 0.0000, 0.0181, and 0.0291. The three sampled SAILOR visits had Dice 1.0000. Native compartment volumes matched all six samples. These checks identify a coordinate defect, not a segmentation accuracy estimate.

The bounded implementation review closed at source `32310c166d79e892be629e977c6ab5e33a126df4`. Geometry, chronology, checkpoint-consumer, and real MRI numerical checks passed. The corrected historical causal-lesion rerun had relative MAE 1.00595, with a paired difference interval spanning zero on the legitimate 117 rows. Aligned features and elapsed-time semantics require new training before their forecasting contribution can be assessed. Historical leaked gains stay withdrawn, and old artifacts cannot be relabeled as aligned.

The 2026-10-08 whole-cohort QA covers 599 LUMIERE and 269 SAILOR measured visits. It resolved 3,451 MRI geometry records and checked all 2,604 measured compartment masks and native volumes. All LUMIERE compartment source grids differ from the MRI pre-grid; all SAILOR ONCO masks share it. Affine resampling is therefore required for the former. The audit did not execute voxel resampling for every visit.

Twenty-one reviewed overlays cover nineteen unique visits. The known extracranial label remains suspect. Across all LUMIERE compartment rows, one-empty-endpoint transitions contribute 24.9% of summed absolute log-volume change. In the locked unseen subset they contribute 32.9%, and in SAILOR 43.7%. These are automated target discontinuities, not confirmed biological events. ONCO-versus-CL median Dice is 0.650 for enhancement and 0.656 for edema, without independent repeat labels or expert adjudication. Source-grid and volume integrity are verified. Repeatability, biological validity, and a noise ceiling remain unknown.

## Current RANO-free anatomy evidence

P1's primary target is three `lesion-state-v1` log-volumes at the next observed visit. The locked encoder-unseen LUMIERE evaluation contains 117 rows from 23 patients; the full manifest has 496 LUMIERE rows. SAILOR transfer has 242 rows from 27 subjects. Persistence is the primary floor, with current-volume ridge as a comparator. SAILOR is unchanged developmental transfer, not an external holdout.

| Method | LUMIERE patient-uniform MAE | Relative | SAILOR MAE | Relative |
|---|---:|---:|---:|---:|
| Persistence | 1.545 | 1.000 | 0.943 | 1.000 |
| Mean delta | 1.552 | 1.005 | 0.947 | 1.005 |
| Linear trend | 2.322 | 1.504 | 1.365 | 1.478 |
| Current-volume ridge | 1.503 | 0.973 | 1.465 | 1.493 |
| Volume-history ridge | 1.589 | 1.029 | 1.617 | 1.609 |
| Current image | 2.047 | 1.325 | 3.182 | 3.295 |
| Mean image history | 2.127 | 1.377 | 4.936 | 4.910 |
| JEPA state | 2.152 | 1.393 | 2.719 | 2.873 |

The current-volume ridge's apparent LUMIERE improvement is unresolved. Its paired MAE difference is −0.042, with 95% confidence interval −0.227 to +0.148. It is worse on SAILOR, with difference +0.522 and interval +0.384 to +0.658.

JEPA is substantially worse than persistence, especially in transfer. These are development results for the next observed visit. They do not establish fixed-horizon forecasting or clinical-response prediction.

A38's physical-history residual MLP had LUMIERE MAE 1.5405 and SAILOR MAE 1.2143. Its apparent small LUMIERE improvement over persistence was unresolved; transfer was significantly worse. A39's frozen fusion branches did not add incremental signal. A40's future leakage means learned P2.4 metrics cannot be used as evidence for or against causal past-only forecasting.

## R35: observed state, observed change, and nuisance

<a id="r35"></a>

### R35 completed comparison

R35 is a completed comparison of MRI-only spatial state and observed-pair representations, originating in sibling worktree `world-model-representations` at revision `2f6e1f5a65be4ccbceac6de706b537de8b695faa`. Its report is the frozen [branch-only R35 record](documentation_sources/versions/768fdfa0a9be4dc76e99fcef9ff971e26252d672e82b08267f5e9ea157ced0f3.md), captured as S062 with branch and live-worktree origins.

The study used 65 encoder-training patients and 26 encoder-unseen LUMIERE patients, with an internal training and validation split. The seeds were 42, 43, and 44. The learned state encoder received dense supervision and native log-volume targets.

Pair models used two observed scans and their measured change. The results therefore concern state estimation and observed change, not forecasting. The fixed reference used BRAINIAC without clinical inputs.

Across three seeds, direct spatial-state volume estimation had MAE 1.1145 on 149 visits from 26 patients. The fixed encoder with scalar-volume supervision scored 1.5000. The difference was −0.3855 [−0.5185, −0.2481].

The spatial model received dense mask supervision as well. This comparison therefore does not separate the effect of architecture from the effect of supervision.

The direct paired head had observed-change MAE 1.2696 on 117 pairs from 23 patients. Zero change scored 1.5446. Their difference was −0.2750 [−0.4953, −0.0957], favoring the paired head.

Subtracting independent state estimates scored 1.2589. Paired minus independent error was +0.0108 [−0.0791, +0.1066]. That interval crosses zero, so the paired architecture did not establish an additional advantage. Results average patient-level errors over seeds. They are not ensemble predictions.

Nuisance separation was incomplete. Reducing resolution in a synthetic test produced false-change MAE 0.3275 for spatial-state differences. The unchanged-pair test scored 0.0445. The excess was 0.2831 [0.1986, 0.3747].

Resolution reduction removes detail, which may include useful anatomy. The test does not prove that the lost information was only nuisance. Noise and affine-intensity tests were near the unchanged-pair result. Stability to intensity scaling and offset largely followed explicit standardization. Smooth intensity bias increased false change.

Empty-target errors were large. Differences in automated segmentation across cohorts also remain mixed with other transfer effects.

On SAILOR, the spatial-state model scored MAE 2.2401 versus 4.6876 for the matched fixed-encoder readout trained on source52. However, the source-trained mean scored 2.1105. Spatial minus mean error was +0.1296 [−0.2571, +0.5098], so the model did not establish an advantage over the mean.

The source26 and source52 ridge comparisons also had intervals crossing zero against their respective mean predictors.

Independent state differences had observed-change MAE 0.9665 versus zero change at 0.9430. Their difference was +0.0235 [−0.0368, +0.1052]. The paired head scored 1.0508 and was worse than independent state differences by +0.0843. Its interval adjusted for twelve comparisons was [0.0295, 0.1440]. These are development-transfer results, not independent external validation.

The frozen nested-CV state control scored 1.6008 versus a fold-trained mean of 1.8359, with paired difference -0.2351 [-0.4280, -0.0423]. Frozen feature differences scored 1.5272 versus zero change 1.5446, with the interval crossing zero. A source52 scalar-supervision control scored 1.5000 versus source mean 1.7051. These readout budgets differ and cannot be selected after inspecting query outcomes.

In LUMIERE, the paired head beat the later-scan-only head: MAE 1.2696 versus 1.4381, difference -0.1685 [-0.3438, -0.0155]. It did not beat independent state differences. On SAILOR, the later-scan-only comparison interval crossed zero.

Source26 and source52 state ridge MAEs were 1.9715 and 1.9525. Neither established an advantage over its matched mean predictor.

A later diagnostic shuffled visits within each patient. Spatial-state MAE rose from 1.1145 to 1.5162 in LUMIERE and from 2.2401 to 2.3361 in SAILOR. That supports sensitivity to the specific visit. It does not establish biological attribution.

Empty-target state errors were large:

| Compartment | Source MAE | Empty-target visits |
|---|---:|---:|
| Core | 7.0523 | 7 |
| Enhancing | 5.7072 | 22 |
| Edema | 9.5619 | 5 |

The aggregate gain does not resolve these absent-compartment failures.

R35 artifacts passed model, feature, row, and patient-disjointness checks. All twelve corrected arms completed 240 epochs, with 74,592 successful updates and 48 AMP backoffs. The first candidate run was invalidated by float16 dense-loss reduction overflow. A later AMP lifecycle failure occurred before a completed candidate epoch. Corrected finite-reduction and backoff-ordering artifacts are the graded runs. Earlier caches lack a creation-time MRI-byte fingerprint. Completion-time tensor hashes do not retroactively supply one. It did not establish biological validity, segmentation repeatability, nuisance invariance, or untouched external generalization. No future prediction claim follows.

## R36: delta-representation experiment status

<a id="r36"></a>

### R36 Phase 1 incomplete

The newer execution record is [R36 results](documentation_sources/versions/51d6523f0da4b3d8745af10b79318fd684aba1fd9b47cc7d321e30df79984551.md), from sibling worktree `world-model-r36` at revision `f4b96c00dd5d79c59ba02d186e024ff747f690ad`. The captured newer record is S065. S064 preserves the older main-checkout version, which ends before the later check-ins.

R36 tests whether giving changed regions more training weight makes signed occupancy change easier to extract from a fixed encoder. The primary comparison is C versus B under matched local linear readouts trained on the 52 source patients. Original R35 and continued state-only A are controls. D tests the added nuisance constraint.

The planned result concerns change between observed scans, not future prediction. [Methods](METHODS.md#r36-phase1) defines the arm objectives.

The immutable contract specifies four arms, three seeds (42/43/44), 20 final-graded epochs, source-only fitting and normalization, paired patient bootstrap, and a 24-GPU-hour total ceiling with nine-hour per-session ceiling. It preserves 599 visits and 496 usable consecutive rows. The encoder-unseen assessment includes 149 visits from 26 patients and 117 pairs from 23 patients. No pair bridges were introduced. SAILOR remains local.

The target quality audit found sparse occupancy change:

| Compartment | Median changed voxels | Share of the 96³ grid | Pairs with both masks empty |
|---|---:|---:|---:|
| Core | 1,239.5 | 0.140% | 7 |
| Enhancing | 524.5 | 0.059% | 40 |
| Edema | 3,866 | 0.437% | 5 |

Both-empty pairs had zero signed change. No occupied pair had exactly unchanged targets. One native enhancing disagreement disappeared after resizing.

These measurements motivate the comparison. They do not establish biological change, accurate segmentation, or that unchanged voxels caused the earlier failures. The primary analysis retained a suspect label outside the brain. Expert adjudication and a segmentation noise ceiling were unavailable. A noise ceiling describes the limit that measurement error places on attainable performance.

The balanced loss, source-only probes, artifact binding, and real CLI numerical smoke were verified. Two finite optimizer updates ran on aligned MRI pairs; identity feature differences were exactly zero. Decoder reversal errors were 0.3865 and 0.3814 after those updates. These are smoke-test values, not quality scores.

The capacity session completed seed-42 A and B for exactly 20 epochs, with 5,940 pair exposures each. A had 5,935 successful updates and five AMP skips; B had 5,936 and four. Both R35 source-only probes completed 20 epochs. Ten remaining matrix arms were launched in two private T4 jobs. At the 2026-10-11 14:08 UTC receipt, all twelve final checkpoints had been downloaded and locally validated, and all ten seed-42 probes passed local validation. Seeds 43 and 44 probes remained.

The branch-only check-ins preserve a change in the recorded budget. At 06:07 UTC, the provisional quota did not cover the remaining readout session limits. At 14:08 UTC, the account reported 13.23 hours remaining and 11.05 hours of R36 use. That left enough allowance for both remaining session limits. The source does not explain the accounting change.

The two training blocks took 13,342.81 and 13,876.09 seconds. Successful updates ranged from 5,928 to 5,936 per arm.

The seed-42 continuation stopped because a status command failed, not because training was reported to fail. Recovery collected the completed readouts without starting them again. The 14:16 repair recorded intent before launch, handled uncertain push outcomes, and prevented repetition of completed epoch-20 jobs. It recorded seed 43 as queued.

These dated receipts do not establish current quota or job state.

At 18:30:51 UTC on October 11, 2026, the continuation receipt reported seed 43 running and seed 44 not started. No comparison of model quality on the evaluation patients had been scored. Phase 1 was therefore incomplete, with no comparative verdict.

A reviewer had completed implementation review. That review did not provide retrospective design approval before training or accept the scientific results. Independent result review and an operator decision remained required. Phase 2 was unauthorized. The exact later receipt is retained in [recovered runtime evidence](documentation_sources/supplementary/r36-runtime-snapshot.json). The separate [protocol/recovery review](documentation_sources/supplementary/r36-protocol-recovery.md) preserves its two reporting findings. No computation was launched to update this document.

## Earlier experiment details and source records

A1 to A41 are preserved above as an index; full setups, tables, and correction history remain in [ablations](documentation_sources/versions/2ddad8d48f3ebb0e89c6855d2c8da8e2762d8291664d49d38faab77851d72549.md). Additional primary records include [pilot](documentation_sources/versions/d8e7bf20a34e6ee1186140c96bb090ed2df0a56f65f11ce140fe05acc4123436.md), [results](documentation_sources/versions/89605d7963ba3e0f942e5feb416846a32faea82a277f1dbe008b0cd0a6b86068.md), [P0 refresh](documentation_sources/versions/91f9eb3a5aa50c35bb9d8d79dbcca3e76e61bc3c93b814a08ae82b0a34ca4d8d.md), [P1 baselines](documentation_sources/versions/b8c596145365e19fbea3ed86218b7b7d7a3131e0be6e60915409d0eddcfddf31.md), [P1 anatomy baselines](documentation_sources/versions/4c077cc0f084f2d14b48b4cf366088ff7c0dc11a6ed3bf8b5678cbea901e141d.md), [P2 seed sensitivity](documentation_sources/versions/9392195ed8cf1ac4fea44e0347230844278803dbd3b21318ca48a6c7c0b2f841.md), [clean rescues](documentation_sources/versions/eaca551880e0a9db5c4bdc7785e135675cd518ebfcd361d5aa482beca4f37dce.md), [methodology validity](documentation_sources/versions/7bc218e8ef5fd2fdb22bc0ed7f5bab72c4b1d099f1f7f2c3145f6c3bca6f3472.md), and [P0 audit](documentation_sources/versions/da674d536622373c6db154365a73c67d5fffcc9646a10713641169a0b5cb2130.md).

The historical SAILOR reprocess procedure is preserved at [reprocess guide](documentation_sources/versions/c7aeeea2bff31ee76a1190a69a05ca48ec78fc476e89d3afc345d546074c3f0d.md). Its intended question was whether matching the LUMIERE preprocessing contract would restore dynamics transfer. A19's same-code reprocess did not flip the persistence gate; A21's skull-out readout-recovery interpretation was subsequently withdrawn because the rebuilt cache did not reproduce it. A23's rebuilt same-space ratios remain the relevant historical evidence. Reprocessing details and limitations remain in the guide and A18 to A24.

Numeric claims in this document are scoped to the stated cache, task, cohort, and aggregation. Patient-level bootstrap intervals do not account for readout training-seed uncertainty unless stated. Developmental SAILOR transfer is not untouched external validation. Historical rows and scores are retained as history where corrected, invalid, or withdrawn; they must not be silently promoted to current evidence.