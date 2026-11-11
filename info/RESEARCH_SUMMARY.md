# Research summary through R62

**Record closed for this work cycle on November 1, 2026.**

This summary explains the research trajectory, supported findings, corrections, and unresolved questions. It derives from the [experiment record](EXPERIMENTS.md), [protocols](METHODS.md), and [interpretations](NOTES_AND_INSIGHTS.md). Those pages own the underlying facts. [TODO](TODO.md#current-status-and-open-decisions) owns future decisions. The historical R labels include proposals, diagnostics, and follow-ups, so R1 through R62 do not mean 62 independent completed experiments.

## Executive conclusion

The project has demonstrated limited, task-specific anatomical learning. It has not demonstrated a useful, transferable longitudinal MRI world model, a dependable incremental benefit from earlier observations, or an advantage that makes JEPA the preferred temporal objective.

The strongest positive findings concern current state. A densely supervised spatial pipeline improved in-domain automated-volume estimation in R35. Matched pretrained initialization improved nonempty segmentation Dice in R49. Cross-region context helped a current-state estimator in R57, although that result used the subsequently questioned coarse regional targets. R47 also retained an in-domain probabilistic forecasting gain from current measurements and timing. None of these findings establishes useful history-based prediction or successful transfer.

The negative forecasting evidence now extends beyond one observation encoder and one cohort. R61 supplied corrected anatomical measurements directly on LUMIERE. R62 repeated a related measured-state comparison on MU-Glioma-Post using reviewed masks. Both history regressions lost to persistence. These findings narrow some explanations for previous failures, but they do not establish that glioma trajectories are intrinsically unpredictable.

The practical outcome is a more reliable measurement and evaluation system, a bounded set of positive anatomical findings, many closed negative or inconclusive comparisons, and forty unopened MU patients. Another architecture sweep on reused assessment patients is not the missing confirmation.

## 1. The capability under investigation

The objective is to maintain a spatial representation of a patient's condition from available MRI observations and predict meaningful aspects of later observations with uncertainty. Acquisition differences should not be mistaken for disease change. Spatial tokens are the preferred representation direction, but that preference is a research choice rather than a demonstrated superiority claim.

Four questions recur throughout the work:

| Question | Information available to the model | Evidence needed |
|---|---|---|
| Current state | A scan available now | Accurate anatomy, compartment burden, and absent-compartment behavior |
| Observed change | Both scans already observed | Improvement over subtracting independent state estimates |
| Future prediction | Observations available before the target scan | Improvement over matched current-only and persistence controls |
| Acquisition robustness | A specified change in imaging conditions | Reduced false anatomical change while preserving useful anatomy |

A successful state estimator is not automatically a forecaster. A successful paired-scan classifier is not predicting an unseen scan. Predicting a teacher's latent features is useful only if the representation supports the anatomical or clinical capability of interest.

Some studies supply measured masks or volumes at the cutoff. Others receive MRI alone and must estimate those quantities. Their persistence baselines therefore differ. Copying a measured volume and copying an MRI-decoded volume are different reference tasks. Likewise, retrospectively supplying the realized next-visit interval does not establish prospective forecasting at a known appointment time.

## 2. Cohorts and the strength of their evidence

### LUMIERE

LUMIERE supplied the main development program. Historical work includes 91 patients, with later experiment-specific source, selection, and scoring subsets. The commonly used R35 and R36 allocation had 65 source patients and 26 encoder-unseen patients. Subsequent studies often reused the original source52 and validation13 pool for new patient-disjoint fits.

Patient-disjoint fitting protects a particular comparison from direct patient overlap. It does not erase earlier exposure to those patients' outcomes during research decisions. The historical final and unseen groups are now development evidence.

Visit labels encode week bins. Suffixes indicate ordering within a week, not one-day offsets. These labels do not support exact daily rates. Lesion targets are automated proxies, and correct geometry does not make them expert biological truth.

### SAILOR

SAILOR is controlled-access developmental transfer data. Several anatomical studies use 269 visits and 242 eligible pairs from 27 patients. Its median recorded inter-visit interval is about 76 days, correcting an earlier description of roughly 14 days.

SAILOR has been reused and is not an untouched external test. Acquisition, clinical sampling, and segmentation differences coexist, so a failed transfer cannot identify a single cause. Its empirical response-code mapping also lacks independent confirmation. Planned-treatment timing, dose semantics, and prognosis definitions remain incomplete for their respective tasks.

### MU-Glioma-Post

R62 established access to [MU-Glioma-Post](https://doi.org/10.7937/7K9K-3C83), a public postoperative collection with 203 patients and 596 released visits. Its four structural MRI sequences and neuroradiologist-reviewed masks include a separate resection-cavity label. The released masks are stronger anatomical references than unreviewed automated outputs, but remain retrospective annotations on already-resampled images.

Before opening image outcomes, the fixed patient-hash rule assigned forty patients to an unopened reservation and 163 to development. Only the development images were downloaded. The development partitions contain 99 fitting, 32 selection, and 32 pilot-scoring patients. These are assigned patients, not counts of history-eligible trajectories.

Two released visits lack masks. Several clinical date sets disagree with image paths, and one fitting sequence has decreasing dates. R62 retained released labels and excluded affected comparisons rather than inventing corrections. It obtained 107 fitting triples from 52 patients, 47 selection triples from twenty patients, and 33 scoring triples from eighteen patients. Released successive scans are not necessarily consecutive clinical visits.

The forty reserved patients remain unopened. Their reservation preserves a future confirmation opportunity. It does not establish independence from foundation-model pretraining or other public releases.

Sources: [cohort contracts](METHODS.md#data-geometry-labels-and-timing-contracts), [R47 populations](EXPERIMENTS.md#r47-anatomical-signal-robustness-and-locked-transfer-october-31-2026), and [R62 feasibility](EXPERIMENTS.md#r62-acquisition-and-longitudinal-feasibility).

## 3. How the research progressed

| Stage | Main question | What the stage established |
|---|---|---|
| Early JEPA, D/A/I/G/V/E/K3 records, and R1–R32 directions | Can pretrained MRI features and temporal objectives support useful prediction? | Working instruments, several weak or negative controls, and corrections to weights, comparisons, labels, and leakage. Some R labels are unrun proposals. |
| R33–R34 | Do clean reruns and valid geometry rescue forecasting? | Completed rescues did not reach the reported persistence floor. Coordinate, chronology, and artifact repairs improved validity. |
| R35–R36 | Can spatial state and change supervision improve anatomical information? | In-domain state learning and a trained change-decoder benefit, with unresolved added paired information, encoder-objective benefit, and transfer. |
| R37–R40 | Does spatial history, joint training, treatment status, or a supplied horizon help? | No established useful history or treatment increment under the tested recipes. |
| R41–R47 | Do native crops, pretraining, continuous dynamics, clinical context, perfusion, or stronger measured-state controls help? | No primary incremental capability established. R47 preserved a narrower in-domain current-state forecasting lead that did not transfer. |
| R48–R52 | Are the learning controls working, and what does pretraining contribute? | Longer two-example fitting succeeded. R49 supported a matched anatomical pretraining benefit. Clinical JEPA and token replacement criteria remained unmet. |
| R53–R55 | Can image completion and direct code fitting establish image-prediction readiness? | No useful forecast gain. Individual-image fidelity remained unresolved, including when the encoder was bypassed. |
| R56–R58 | Do regional assimilation, graph context, and spatial latent forecasting improve the selected endpoints? | A narrow current-state context gain in R57, but no full graph, history, or future-feature benefit. |
| R59–R61 | Are regional targets valid, and does corrected measured history help? | Coarse sampling defects were quantified and corrected. The corrected history regression still lost to persistence. |
| R62 | Does the measured-history comparison improve on a new reviewed-mask cohort? | MU is usable for a bounded study. The tested history regression again lost to persistence. |

This is an adaptive research sequence. Later studies were motivated by earlier outcomes and often reused patients. The table is a history of questions, not a meta-analysis of independent replications. The [historical experiment index](EXPERIMENTS.md#historical-experiment-index) and [R-direction index](METHODS.md#r-directions) preserve individual identifiers and dispositions.

## 4. The positive findings worth retaining

### Current anatomical state can be learned

R35's spatial state model reduced LUMIERE automated-volume mean absolute error, or MAE, from 1.5000 for the fixed scalar-supervised encoder control to 1.1145. The comparison used 149 visits from 26 encoder-unseen development patients and native log-volume coordinates.

The spatial model also received dense mask supervision. The result supports the complete supervised pipeline, not architecture alone. On SAILOR, its error did not establish an advantage over a source-fitted mean control. Empty compartments and resolution-induced false change remained important failures.

### Pretraining helped under a matched anatomical recipe

R49 held architecture, supervision, patient draws, and selection opportunities fixed while comparing official BrainIAC initialization with random initialization. Nonempty-compartment Dice was 0.198176 versus 0.158143. The difference was +0.040033, with adjusted interval [+0.019563, +0.059459]. Both seeds favored pretraining, and the specified volume-retention allowance passed.

Dice measures mask overlap, with one indicating perfect overlap. This is a meaningful attribution result, but absolute segmentation quality remained poor. The volume-error improvement itself was unresolved. The scoring patients had been studied historically, and pretraining-patient separation was not verified. The study supports a narrow initialization benefit, not clinical utility or temporal understanding.

### Regional context can help current-state estimation

R57's graph with cross-region messages reduced nonempty-region error by 11.35% relative to parameter-matched self-only messages. The adjusted interval favored context, and both seeds improved.

Correct physical adjacency did not beat the fixed shuffled correspondence. History did not beat the latest-only graph. The study used the old coarse log-count targets. R60's later correction means this context finding needs separate confirmation on source-volume targets. R61 did not perform that current-state comparison.

### Current measurements can improve an in-domain forecast score

R47's query-aware current-state model achieved CRPS 1.006585 versus persistence at 1.120054, with an adjusted interval favoring the model. CRPS, the continuous ranked probability score, evaluates a predictive distribution. This result can therefore include calibration as well as its predicted center.

The corresponding history increment was only 2.21% with a zero-crossing interval, below the fixed useful-gain threshold. The current-state advantage did not transfer to SAILOR. This finding prevents the overly broad conclusion that no learned predictor ever beat persistence, while still leaving the longitudinal world-model claim unproven.

Sources: [R35](EXPERIMENTS.md#r35), [R49](EXPERIMENTS.md#matched-pretraining-result), [R57](EXPERIMENTS.md#r57-regional-context-helped-without-established-adjacency-or-history-benefit), and [R47](EXPERIMENTS.md#forecasting-a-current-state-gain-not-an-established-history-gain).

## 5. What the history comparisons say

History has been tested through frozen spatial features, trainable encoders, recurrent models, regional filters, graphs, latent forecasting, and directly supplied anatomical measurements. It has repeatedly failed to establish the selected useful increment over matched current-only controls and persistence.

R37 is the central frozen-spatial comparison. On 75 pairs from eighteen reused LUMIERE patients, H-minus-L CRPS was +0.00334486, with adjusted 99% interval [−0.04732935, +0.05234837]. This did not support the required history improvement. Calibration and resolution-sensitivity criteria also failed.

R38 allowed forecast gradients into the spatial encoder. It did not establish a benefit over detached observation training and failed current-state retention. This weakens the proposed remedy of joint training under that recipe. It does not prove that a frozen encoder can never be a limiting factor or identify noisy-target overfitting as the unique cause.

R40 supplied a retrospective horizon to trainable spatial predictors. Its history point advantage was about 0.05%, with a zero-crossing interval. R43's continuous measured-state dynamics were worse than matched discrete transitions at one and two steps. More explicit temporal machinery did not automatically improve the selected anatomical outputs.

The two latest measured-state comparisons remove the need to infer the cutoff burden from MRI:

| Study | Scoring cohort | Persistence MAE | Current-only MAE | History MAE | History versus current-only | History versus persistence |
|---|---|---:|---:|---:|---|---|
| R61, corrected LUMIERE targets | 30 pairs, 8 patients | 0.280948 | 0.289303 | 0.288491 | 0.28% lower error | 2.68% higher error |
| R62, MU reviewed masks | 33 triples, 18 patients | 0.165866 | 0.182648 | 0.182564 | 0.046% lower error | 10.07% higher error |

Both use patient-uniform regional log1p-mL MAE. The transform is `log(1 + volume in millilitres)`, so these errors are not raw millilitres. They remain different tasks: R61 uses three compartments on its fixed LUMIERE regions, while R62 uses four compartments including cavity on a different reference grid. Their absolute errors are not a cross-dataset leaderboard.

R61's small H-minus-L interval excluded zero but did not meet the practical gain criterion. R62's H-minus-L adjusted interval crossed zero. Both H-minus-P intervals favored persistence. In both studies, all three ridges selected penalty 100 and the trend coefficient was zero.

The appropriate conclusion is about these models and endpoints. Strong regularization, small scoring cohorts, heterogeneous follow-up, and the difference between squared-error fitting and absolute-error selection limit broader interpretation. In R62, learned corrections helped changed and nonempty cells while adding error to empty cells. Tumor-only and cavity errors both exceeded persistence, so cavity inclusion alone does not explain the failure.

Sources: [R37](EXPERIMENTS.md#r37-full-pilot), [R38](EXPERIMENTS.md#r38-completed), [R40](EXPERIMENTS.md#r40-completed-trainable-spatial-horizon-pilot), [R43](EXPERIMENTS.md#r43-nominal-time-anatomical-dynamics-and-propagated-rollout), [R61](EXPERIMENTS.md#r61-corrected-targets-did-not-establish-useful-history-forecasting), and [R62](EXPERIMENTS.md#r62-did-not-establish-useful-history-forecasting-on-mu).

## 6. BrainIAC, spatial tokens, and JEPA

BrainIAC initialization, the selected token interface, and the downstream decoder answer different questions. Pretraining may supply useful anatomical information that a constrained token interface fails to preserve or a particular readout fails to recover.

R49 supports useful pretrained information under matched supervision. R52 does not contradict that result. It tested BrainIAC and a compact CNN through a common 216-by-64 spatial-token interface with fresh token-only anatomical readouts. Both primary Dice scores were around 0.0666, and the replacement criterion failed. That does not establish whether the missing detail was absent in the backbone, lost in the interface, or inaccessible to the readout.

R54 and R55 further weaken an encoder-only explanation for blurred images. Directly optimizing spatial codes while bypassing the MRI encoder did not resolve the fixed image-fidelity criterion. A matched-duration continuation improved almost as much. The tested decoder and optimization path remained limiting candidates, but their failure was not proof of an architectural impossibility.

JEPA, short for Joint-Embedding Predictive Architecture, trains a predictor against another observation's learned features rather than its pixels. It has not earned a preferred role here. Early large improvements involved invalid comparisons or later-withdrawn forecasting inputs. The later R50 and R51 clinical comparisons separated direct supervision, proper temporal teachers, and shuffled teachers. Across the original and revised recipes, no accepted primary JEPA benefit emerged. R58 likewise failed to establish useful future-feature alignment for anatomical forecasting.

In R50-v3 and R51-v3, the limited-adaptation recipe recovered fitting readiness. Direct BrainIAC had lower mean calibrated log loss than DenseNet, but seed directions disagreed and the adjusted interval crossed zero. Proper JEPA was slightly worse than direct supervision. Several calibrated scoring classifiers still predicted only the majority class. A varying teacher representation and declining auxiliary loss did not establish useful clinical predictions.

These are bounds on tested recipes. They are not evidence that BrainIAC is generally incapable of anatomy, that spatial tokens are intrinsically unsuitable, or that every JEPA formulation must fail. A future spatial-state model would need to show what it preserves beyond the chosen interface and what useful capability it adds.

Sources: [R52](EXPERIMENTS.md#r52-completed-without-an-established-token-replacement), [R54](EXPERIMENTS.md#r54-learned-corrections-but-failed-substantial-individual-fitting), [R55](EXPERIMENTS.md#r55-code-bypass-did-not-resolve-image-fidelity), [R50-v3 and R51-v3](EXPERIMENTS.md#r50-v3-and-r51-v3-limited-adaptation-clinical-comparison), and [R58](EXPERIMENTS.md#r58-did-not-establish-a-future-feature-or-history-forecasting-gain).

## 7. Corrections that changed the interpretation

| Correction | Consequence for the research claim |
|---|---|
| Behaviorally corrupt community pretrained weights | Compatible parameter shapes and falling loss did not establish valid pretrained behavior. Official-weight runs replaced the invalid pilot. |
| A8 and A24 representation-space mismatch | The early 2.7-fold error improvement and 82-of-91 patient-win claim were not valid same-space comparisons. |
| A40 future-information leakage | Learned P2.4 forecasting, seed, transfer, and adaptation claims were withdrawn because later-image, mask, and interval features entered predictors. |
| R34 mask-coordinate and chronology repairs | Foreground overlap and filenames could not substitute for verified physical geometry and visit order. Repair did not establish biological truth. |
| R48 short fitting budget | Both models failed the original 128-update mask criterion but passed by 512 unchanged updates. The initial failure was not proof of inability to learn. |
| R50/R51 methodology audit | The v1 prior offset was valid, and a first-token readout was not corner-only input. Bundled v3 changes did not isolate the cause of v2's near-constant classifiers. |
| R59–R60 regional target construction | Coarse mask sampling erased positive regional burden and sometimes reversed measured change direction. Source-volume aggregation replaced coarse foreground counting. |

R60 reconstructed corrected targets for 322 visits from 64 patients. Across that cohort, the old target erased 605 positive region-compartment cells across 257 visits and produced 102 opposite change directions among 5,784 compared regional changes. The corrected contract assigns each labeled source voxel's physical volume by its center and verifies conservation.

R61 then repeated the measured-history comparison on corrected targets and still failed the useful-history criterion. Fixing measurement was necessary, but did not explain away the negative result. R59 and R61 also changed the logarithm's volume units, so their absolute MAEs cannot isolate the sampling correction as a single-factor effect.

The project found several real defects. There is no evidence for one universal defect explaining every failed experiment. The corrections and original outcomes remain part of the record rather than being replaced by a cleaner retrospective story.

Sources: [withdrawals](EXPERIMENTS.md#explicit-withdrawals-and-limits), [training checks](EXPERIMENTS.md#pilot-and-training-gates), [R48 diagnostic](EXPERIMENTS.md#r48-diagnostic-follow-up-the-128-update-test-stopped-before-mask-fitting-was-adequate), [methodology corrections](NOTES_AND_INSIGHTS.md#november-1-corrections-to-the-methodology-audit), and [R60](EXPERIMENTS.md#r60-quantified-and-corrected-coarse-regional-target-discrepancies).

## 8. What remains unresolved

- **Absolute anatomical fidelity.** Relative gains coexist with poor segmentation, missed small structures, and false burden in empty compartments. A useful current-scan reference has not yet been shown to be preserved by the proposed token interface.
- **Predictability of the selected outcome.** There is no validated ceiling separating genuinely unpredictable biology from timing, target error, missing interventions, and model limitations.
- **Generalization.** Source-cohort gains have repeatedly weakened or disappeared on SAILOR. Cohort reuse and unverified pretraining overlap limit independence claims.
- **Treatment information.** R39 tested coarse current status, not a timestamped future treatment plan. It neither validates treatment-conditioned prediction nor rules out a benefit from better intervention records.
- **Clinical endpoints.** Response assessment, incident progression, survival, pseudoprogression, and treatment effects require different labels and cutoffs. Neither missing event semantics nor an empirical response-code mapping can be repaired by a more complex model.
- **Uncertainty and nuisance.** Some experiments score distributions, but calibrated biological uncertainty and real scanner invariance remain unestablished. Intensity stability often comes from explicit normalization.
- **Independent scientific review.** Numerical reconstruction supports artifact consistency. It does not provide expert target adjudication, a new cohort, or independent review of every interpretation. R35, R36 Phase 1, and R37 have specific scoped reviews; later replay receipts should not be described as equivalent approvals.

## 9. Deliverables and verification

The durable research assets include corrected geometry and chronology handling, same-information baselines, a source-voxel regional-volume contract and loader, patient-level scoring, saved prediction replay, and explicit correction history. The code and records now distinguish successful execution from scientific benefit more carefully.

R61 verification reconstructed 582 measurement uses from 270 source visits and replayed thirty arrays. R62 checked all 2,403 downloaded development file sizes, independently reconstructed all 479 complete development masks, replayed nineteen prediction arrays, and independently solved the three selected ridge problems. Both studies reconstructed patient contrasts and inspected fixed image examples. These checks establish consistency with the recorded calculation, not expert biological correctness.

Public aggregates are linked from the owning experiment sections. Participant inventories, images, masks, individual predictions, and weights remain outside Git. R62's private bundle is `/home/aditya/experiments/world-model-r62-mu-20261101` on `mosaic-gpu`. Earlier run locations and reproduction commands remain in [Methods](METHODS.md).

The HTML handbook teaches prerequisites and the earlier program through R36. The generated research page covers R36 and R37. Their narrower coverage should not be mistaken for a current R62 synthesis. This document is the dated whole-program summary.

## 10. Closeout and publication position

The selected experimental work through R62 is complete. No further fitting is selected by this closeout, and R36 Phase 2 remains unlaunched. Forty MU patients remain unopened. The broad proposal inventory records possibilities, not a queue of authorized runs.

A defensible report can describe limited anatomical gains, unresolved added change information, repeated failure to establish useful history prediction, and the measurement corrections that altered earlier claims. It cannot present the sequence as a successful transferable world model, a clinically validated progression predictor, or a proof that longitudinal MRI has no predictive value.

The earlier publication assessment identified a conditional, narrower R36 paper question about change-balanced supervision under matched frozen readouts. Existing paired-scan and longitudinal-representation literature prevents a broad first-of-kind claim. A workshop manuscript draft now exists, but a draft is not a submission, acceptance, or independently confirmed contribution. The later R49 and R57 findings and R61–R62 diagnostics can be discussed as separately scoped evidence rather than combined into one prospectively designed experiment.

If work resumes, the next decision is a single capability with a distinct evidential basis. A candidate is whether spatial MRI tokens retain the anatomy recoverable by a competent current-scan reference on reviewed labels. That comparison would need matched inputs and supervision, meaningful absolute fidelity, empty-compartment checks, and a fixed evaluation plan. R62 did not perform it. Another possibility requires a concrete new source of cutoff-valid temporal information or a better-defined forecast outcome. Neither is automatically selected here.

**The overall conclusion is limited anatomical promise without demonstrated useful longitudinal world modeling.** The strongest next evidence would answer one of the unresolved capability questions under a fixed comparison, rather than add another architecture to the same reused endpoint.
