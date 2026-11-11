# Experiments and evidence

## R68 preparation completed with clinical-context and adjudication blockers, November 9, 2026

The [metadata-only preparation](r68_preparation_metrics.json) produced a private episode manifest for the existing 99 fitting and 32 selection MU patients: **391 recorded-or-released visits**, comprising 388 released structural panels and three clinical MRI records without released panels. Existing prepared arrays cover 387 visits. Forty reserved patients and the separate pilot-scoring group were excluded before interpreting patient values. No pixels, new segmentation measurements, model fits or reserved eligibility records were opened. This completes preparation, not R68 forecasting.

The provisional recommendation is **FLAIR-abnormality change at 84 ± 28 days during follow-up after recorded initial radiotherapy completion**. Include all such cutoffs; do not restrict enrollment to future survivors, treatment completers or changing disease. The phase remains retrospective because information-availability timestamps are absent. The following counts retain original split roles and count patients separately from episodes:

| Post-initial-RT inventory | Fitting | Selection |
|---|---:|---:|
| All event-defined cutoffs, patients / episodes | 62 / 189 | 23 / 84 |
| 12-week date-window matches, patients / pairs | 38 / 78 | 19 / 48 |
| 12-week usable released scan/label pairs, patients / pairs | 38 / 76 | 18 / 47 |
| Those pairs with earlier released MRI, patients / pairs | 26 / 47 | 17 / 36 |
| 24-week usable released scan/label pairs, patients / pairs | 23 / 37 | 14 / 30 |
| Verified cutoff-valid clinical context plus longitudinal adjudication | 0 / 0 | 0 / 0 |

At 12 weeks, 49 fitting and 13 selection cutoffs have only out-of-window later MRI; 62 and 23 have no later recorded MRI. Two fitting matches and one selection match are unusable. Documented deaths before the window ends occur for 26 fitting cutoffs from 19 patients and five selection cutoffs from three patients; these overlap other categories and are not an exclusion by themselves. The first-post-RT-only alternative has 34 fitting and 14 selection usable pairs, but only five and three with history, so it is not the recommended scope. Twelve weeks has more patient and pair support than 24 weeks; no forecast performance or observed-change values informed the choice.

The workbook has treatment event dates and delivered dose/fractions, but no plan, order or information-availability timestamps. Among the recommended 12-week usable pairs, dose/fractions exist for 75 fitting pairs from 37 patients and 42 selection pairs from 17 patients; these are **not validated planned-treatment inputs**. Private field-level examples show an ever-chemo flag before the actual start, a delivered dose without a plan timestamp, an earlier RT-completion event without known documentation time, and death code zero whose meaning includes lost follow-up/unknown. Across the full allowed inventory, 21 scan records from ten patients fall after recorded death, two released scans lack MRI dates, one lacks an annotation, and one trajectory has nonincreasing dates. They remain visible with unresolved flags; no dates or labels were silently repaired. All phase/horizon completeness and missingness counts are in the aggregate.

MU masks are not unreviewed automated proxies: the [release paper](https://www.nature.com/articles/s41597-025-06011-7) describes manual correction, independent review and final neuroradiologist consensus. That per-scan process does not establish independently adjudicated biological change for these pairs. The manifest reuses 145 prior transition-inspection records and contains a six-pair, six-patient review index (three per split), selected without model errors or outcome-change values. No expert longitudinal judgments were supplied or labels altered. Needed next are source clarification of timing conflicts, cutoff-dated treatment plans/orders and revisions, and qualified review of the bounded pairs. Zero fully qualified R68 patients were established in checked assets; this is not proof that source institutions lack those records.

The companion R64 access check found no approved project-access record in the checked local research records or bounded GPU-host filenames. This is not an account denial or a live entitlement check. [RIDER Neuro MRI](https://www.cancerimagingarchive.net/collection/rider-neuro-mri/) describes 19 repeat-imaged patients; [QIN GBM Treatment Response](https://www.cancerimagingarchive.net/collection/qin-gbm-treatment-response/) describes 54 participants with postoperative pretreatment repeats, typically 2–5 days apart. Both image routes remain controlled. The aggregate contains a draft research-use statement and the investigator/institutional fields needed under [TCIA's access route](https://www.cancerimagingarchive.net/nih-controlled-data-access-policy/) and [NIH request instructions](https://www.grants.nih.gov/policy-and-compliance/policy-topics/sharing-policies/accessing-data/dbgap). No access request, institutional attestation, third-party contact or image download was made. Actual accessible pair/sequence counts remain unverified; repeatability alone cannot establish biological-change truth.

## Cavity supervision completed: MU anatomy improves, joint retention unresolved, November 6, 2026

All 12,500 updates, both matched full-volume evaluations and the timeline exports completed. Training took 10,302.6 seconds (2 hours 52 minutes). Update 8,750 was selected; selection scores at updates 10,000/11,250/12,500 were 0.652623/0.652458/0.652772, worse than the selected 0.601289. There is no boundary-improvement evidence supporting an unchanged extension. The [completed comparison](patient_cavity_observation_metrics.json) supersedes the running status in the preserved [launch receipt](patient_cavity_build_metrics.json).

Evaluation uses the same **178 LUMIERE visits from 18 patients and 106 MU visits from 32 patients** for both checkpoints. Results average visits within patients, then patients. The new checkpoint is `d006424493442bd1bcd8fd5b15888b197aa3a95ef3f87aeed5df2f52e86ccacb`; the preserved baseline remains `87e2349c54fafd133a08246fd0d920956687faf640c8a080edc2f2cd1dc3ae59`. The 95% intervals below use 4,000 paired patient bootstrap draws, seed 17, without multiplicity adjustment. These are reused developmental selection patients, not independent confirmation.

| Measure | Preserved baseline | Cavity model | New minus baseline, paired 95% interval |
|---|---:|---:|---|
| MU original three-compartment Dice, full field | 0.62942 | 0.66392 | +0.03450 [0.01239, 0.05976] |
| MU core Dice, all targets | 0.29870 | 0.37922 | +0.08053 [0.02384, 0.14679] |
| MU core Dice, nonempty targets | 0.41481 | 0.44515 | +0.03034 [−0.00943, 0.07597] |
| MU core absolute volume error, mL | 8.04298 | 1.66595 | −6.37703 [−9.98008, −3.40841] |
| MU predicted core inside reference cavity, mL | 6.14277 | 0.78376 | −5.35901 [−8.77214, −2.69788] |
| MU false-positive core when absent, mL | 9.77065 | 1.25760 | −8.51305 [−15.04893, −3.58363] |
| LUMIERE conditional four-category readout, lesion Dice | 0.71045 | 0.71904 | +0.00860 [−0.00365, 0.02394] |
| LUMIERE native five-category output, available-label Dice | 0.71045 | 0.59256 | −0.11789 [−0.15097, −0.07964] |

MU now scores the released cavity as a real category and penalizes core predictions there. These baseline scores therefore differ from the earlier cavity-ignored MU score; do not compare the new 0.66392 directly with the old 0.66663 under different validity masks. Five-category MU scoring gives four-foreground-compartment mean Dice 0.49420 versus 0.62619, but that includes a newly supported category. The original-three-compartment improvement above is the more informative retention comparison.

New cavity Dice is **0.55783 on nonempty references** (97 visits, 30 patients; patient bootstrap interval [0.47428, 0.63803]). Cavity absolute volume error is 5.84969 mL. When cavity is absent, false-positive cavity averages **4.11243 mL** (nine visits, five patients; interval [2.03520, 6.30663]). Core false-positive reduction is substantial, but the nonempty-core Dice increment remains uncertain. These results support better current MU estimation, not reliable core trajectory prediction.

The LUMIERE distinction is essential: conditional scores discard the cavity logit before choosing a class, whereas the saved timeline uses the native five-way output. LUMIERE lacks cavity references, so the native disagreement does not prove every cavity prediction is biologically wrong; it does prevent claiming that the deployed five-way readout retained the original annotated task. The improved conditional point estimate cannot conceal the native decline.

Missing-FLAIR lesion MAE worsens from **0.57373 to 0.59040** on LUMIERE (difference +0.01667 [0.00458, 0.02658]) and **0.58854 to 0.59445** on MU (+0.00590 [−0.00306, 0.01525]). Whole-brain FLAIR MAE changes from 0.39070 to 0.39001 on LUMIERE and 0.39268 to 0.39809 on MU; both paired intervals include zero. Other missing-sequence measures remain in the aggregate. Near-unchanged pooled image error does not establish lesion-detail retention.

Three current LUMIERE visits, two current MU visits and three missing-FLAIR LUMIERE visits were exported. All eight saved segmentation maps reproduce their reported four-compartment volumes. Their forecasts are explicitly persistence only; no old learned forecast head was silently attached to the changed observation identity. The prior learned forecasts and scalar controls remain frozen. Forty reserved patients remain unopened. The result is a useful developmental MU supervision gain with unresolved cross-dataset readout and image-retention limitations; it is not a universal baseline replacement or a predictive-state milestone.

## Cavity supervision implemented and fine-tuning started, November 6, 2026

The operator prioritized anatomical supervision and deferred opening all forty reserved patients. The [build receipt](patient_cavity_build_metrics.json) records preparation, tests and launch status, **not improved model accuracy**. The [protocol](METHODS.md#cavity-supervised-observation-development-november-6-2026) fixes one seed, 12,500 updates from the preserved joint model, full-volume application-aware selection, and matched full selection evaluation. Earlier observation and forecasting outcomes remain unchanged.

All **281 MU fitting visits and 106 selection visits** reproduce their original prepared labels exactly outside the previously ignored cavity, after verifying source-mask orientation and affine geometry. Explicit cavity is nonempty in **238 fitting visits and 97 selection visits**. Images, elapsed-time joins and membership are unchanged. LUMIERE cavity remains unknown under conditional four-category supervision.

A bounded support check found that CT1 nonzero foreground excludes **30.973 of 2,381.545 mL** of fitting cavity annotations (1.30%) and **14.476 of 1,527.392 mL** of selection cavity annotations (0.95%). Individual visits have much larger exclusions. The new anatomy loss and scoring now cover the full annotated field, with balanced inside/outside-foreground background loss. Image losses retain image foreground. Old baseline inference remains unchanged, and common-foreground three-compartment results will be reported alongside full-field results. Initial smoke outputs have large false positives outside foreground; the smoke verifies execution only. Learning correct external background is part of the new model's required evaluation, not an assumed success.

Two real updates using both datasets passed encoder-update, checkpoint save/reload and serialized-state replay checks: maximum output differences were zero. Full-volume current anatomy and all four missing-sequence readouts ran on one LUMIERE and one MU visit after the support correction. A two-visit MU timeline saved and reloaded five-category states, appended observations and issued persistence forecasts. The existing viewer's cavity column and explicit untrained-smoke label were visually checked. Four cavity-supervision tests, 17 existing patient-state tests and six structured-forecast tests passed on the GPU host. These checks establish implementation integrity, not forecasting or anatomical usefulness.

The reference-only transition inventory covers **87 fitting targets (55 at 12 weeks, 32 at 24)** and **58 selection targets (36/22)**. On common support, substantial core reduction is flagged in 6/5 fitting targets and 1/2 selection targets at 12/24 weeks. Core/cavity exchange above 2 mL is flagged in 5/3 fitting targets and 0/1 selection targets; one fitting target has exchange above 1 mL in both directions. These are overlapping review flags, not independent patients or adjudicated biological outcomes.

Six distinct-patient diagnostic cases were visually inspected with current images, current reference, preserved current-model output, later image/reference, and exchange overlays. They show current cavity-to-core estimation errors separately from later reference changes. One selection case has only **+1.639 mL** net core change on common support while a selected region changes from core to cavity; other cases show substantial core reduction alongside cavity expansion. These selected examples cannot distinguish intervention, annotation convention and residual correspondence error by themselves. No target was automatically relabeled.

Training is running at approximately **0.62 seconds/update**. The current estimate is **6–8 hours** for fine-tuning, bounded selection, both matched full-volume evaluations and timeline export. No additional forecast fit, seed matrix, calibration claim or reserved assessment has started. The frozen narrow MU24 future-FLAIR-abnormality **segmentation** candidate remains preserved; future-image synthesis is not established.

This reference records completed outcomes and corrections, including the R63 exploratory pilot on November 3, 2026. The [research summary](RESEARCH_SUMMARY.md) retains the synthesis through R62. Dated execution snapshots describe their original stage; later closure sections govern current status. This is a record of saved evidence, not a live progress report.

Use [Notes and insights](NOTES_AND_INSIGHTS.md) for the reasoning, [Methods](METHODS.md) for protocols and term definitions, and [TODO](TODO.md) for remaining work. The [archive](archive/2026-10-12/README.md) preserves the original records and their correction history.

## MU forecast interpretation: empty core dominates, small FLAIR change signal remains, November 6, 2026

The operator-requested bounded interpretation is complete on all 58 existing MU selection targets, both selected anatomical models, and three compartments (348 case/compartment/method records). No model was trained or reselected. All replayed compartment Dice and scenario volumes matched the original evaluation. Original aggregate gains and the additive attribution partitions agreed within 5.6e−17. Four exact accounting tests passed, and three explicitly selected diagnostic slices were visually inspected. All four selected transition/control hashes and the preserved observation hash remain unchanged; forty reserved patients remain unopened. See the [full interpretation](patient_forecast_interpretation_metrics.json) and [protocol](METHODS.md#bounded-interpretation-of-mu-structured-forecasts-november-6-2026).

### Empty compartments explain the large headline gain

The table reports **Dice percentage-point contributions to the original overall gain**, retaining its original patient/cutoff/compartment weights. Contributions can exceed the total because other strata worsen.

| Horizon / method | Overall Dice gain over model persistence | Future-empty nonenhancing contribution | Nonempty nonenhancing contribution |
|---|---:|---:|---:|
| 12 weeks / current-only | +5.01 | +6.48 | −1.42 |
| 12 weeks / history | +5.81 | +6.48 | −1.37 |
| 24 weeks / current-only | +9.44 | +10.26 | −0.80 |
| 24 weeks / history | +10.35 | +10.26 | −0.83 |

At 24 weeks, ten targets from nine patients have empty nonenhancing reference compartments. Persistence correctly predicts none as empty; each forecaster correctly predicts five. Two of those successes are empty at both visits; three are reference resolutions from only **0.020, 0.045 and 0.662 mL**. The one resolution satisfying the existing meaningful-volume-change rule, from 13.494 mL, is not correctly predicted empty. At 12 weeks, successful resolutions likewise begin at 0.020–0.662 mL; the 15.561 mL meaningful resolution is not correctly predicted empty. These findings do not establish prediction of substantial core disappearance.

For the twelve 24-week targets from eight patients where nonenhancing tissue remains present in the future, current-only Dice is **0.0422 worse** than persistence [−0.0876, −0.0048]; history is **0.0409 worse** [−0.0942, −0.0012]. Excluding targets of at most 0.1 mL does not rescue the effect: differences are −0.0711 and −0.0688, respectively. Only one 24-week target and two 12-week targets have persistently nonempty core with meaningful reference-volume change; these cannot support a robust core-trajectory claim. One-patient strata receive no population confidence interval.

The empty-map successes are not solely an ensemble-vote cancellation artifact. A separate check of all 25 MU targets with empty future core found that every correctly empty vote had an empty scored-space scenario; the highest-weight scenario gives the same ten/five empty successes at 12/24 weeks for both models. Scored-space emptiness is distinct from whole-map volume when future cavity is ignored.

### Current errors, changing reference anatomy, and cavity are different contributions

The exact retrospective four-region Dice decomposition at 24 weeks assigns the following overall percentage-point contributions:

| Reference-defined region | Current-only | History |
|---|---:|---:|
| Valid current/future compartment occupancy unchanged | +2.42 | +3.10 |
| Valid current/future occupancy changed | +3.22 | +3.26 |
| Current cavity, unscored at cutoff but scored at the future visit | +3.80 | +4.00 |
| Other currently unscored locations | 0.00 | 0.00 |

This is exact accounting over annotation-defined regions, **not a causal fraction of biological evolution**. The changed-reference component includes tiny resolutions, boundary differences and imperfect correspondence. Current-cavity locations cannot be labeled current core correction or true core evolution because the shared current reference is ignored there. Locations ignored as cavity at both visits do not contribute to the future Dice at all.

On the common valid grid for future-nonempty core cases, forecasting improves Dice against the *current* reference by 0.261 (current-only) or 0.303 (history), while their future-core Dice worsens as described above. That is consistent with a correction/suppression mechanism, not reliable nonenhancing trajectories. It does not justify using the forecast as a general current-anatomy postprocessor: it reduces current-reference FLAIR Dice.

### A smaller FLAIR signal merits a narrowly frozen candidate

All 22 MU 24-week FLAIR-abnormality references are nonempty, so its gain has no empty-reference reward. Current-only improves label-3 Dice by **0.01264** [0.00414, 0.02144]; history improves it by 0.01755 [0.00759, 0.02776]. For current-only, the regional accounting is −0.00496 from unchanged-reference locations, +0.01681 from changed-reference locations, and +0.00079 from current cavity. This small effect differs from the dominant core result.

Among the 17 persistently nonempty targets from twelve patients with meaningful FLAIR volume change, current-only gains **0.01456** [0.00547, 0.02439] over model persistence and **0.01478** [0.00256, 0.02902] over privileged current-reference persistence on the common valid grid. The latter is an evaluation-only comparator, never a model input. Current-reference FLAIR Dice falls by 0.0574 in that subgroup, so the future gain is not simply an improvement of present FLAIR anatomy. This is a modest, post-hoc developmental signal from multiple explored strata; it does not establish independent generalization, biological evolution, clinical utility, or the original joint claim.

The simpler **current-only, 24-week MU label-3 point predictor** is now frozen as a candidate for independent patient confirmation, with checkpoint, observation, source archive and inference definition identified in the [candidate manifest](patient_forecast_confirmation_candidate.json). Confirmation has not started and no reserved metadata or outcomes were newly opened. The candidate's point segmentation is the weighted scenario vote used in evaluation; the viewer's default highest-weight scenario remains a separately labeled coherent sampled examination. No nonenhancing-trajectory or calibrated-volume claim is included.

### Uncertainty failures differ by compartment

At 24 weeks, history's nonenhancing interval has overall coverage 70.5% and width 6.15 mL. On nonempty references, coverage falls to **60.4%** with width 7.62 mL, and all misses are above the upper bound. On empty references, coverage is 72.2%; failures have positive lower bounds. History FLAIR coverage is 88.5%, but mean bounds span approximately **0.25–157.16 mL** (width 156.92 mL), compared with mean absolute reference volume change 45.11 mL; remaining misses are above the upper bound. Current-only FLAIR coverage is 74.4% with a 135.70 mL width. These are different bias/support/sharpness problems, not evidence for one shared scale correction. No calibration was fitted.

**Decision:** preserve the useful integrated system and this narrow FLAIR candidate, but put the next substantial training investment into explicit cavity/core supervision and reliable longitudinal compartment annotations. Do not promote the headline MU gain to general forecasting, reliable core evolution, or a current-observation correction module. Independent confirmation of the small FLAIR signal can be designed around the frozen candidate without another architecture search or training matrix.

## Structured transition completed: mixed anatomy gains, no consistent history benefit, November 6, 2026

All four 6,000-update configurations, 259 identical-target selection evaluations, and the trained predictive timeline completed. The two anatomical models both selected update 4,000. Scalar current/history controls selected 5,000/5,500. The four fits including bounded selection took about 64 minutes; final evaluation took 17.3 minutes. Observation preparation is additional. See the [complete aggregate](patient_structured_forecasting_metrics.json). These remain development-selection results.

| Dataset / horizon | Targets / patients | Persistence Dice | Current-only Dice | History Dice | History volume MAE, mL | History volume CRPS, mL | History 90% coverage / width, mL |
|---|---:|---:|---:|---:|---:|---:|---:|
| LUMIERE / 12 weeks | 112 / 16 | 0.276 | 0.251 | 0.242 | 11.86 | 9.01 | 78.7% / 44.13 |
| LUMIERE / 24 weeks | 89 / 15 | 0.216 | 0.204 | 0.192 | 17.94 | 14.20 | 71.5% / 54.92 |
| MU / 12 weeks | 36 / 18 | 0.350 | 0.400 | 0.408 | 12.90 | 10.05 | 79.6% / 62.74 |
| MU / 24 weeks | 22 / 13 | 0.254 | 0.349 | 0.358 | 19.15 | 14.51 | 80.3% / 71.04 |

Metrics average within patients before averaging patients. Dice uses the weighted scenario-vote map, whereas volume errors use distribution means. The viewer displays a single highest-weight coherent scenario and labels its distribution separately. MU targets use recorded day differences; LUMIERE targets use nominal week bins.

History-minus-current Dice was −0.00975 [−0.01522, −0.00531] at LUMIERE 12 weeks and −0.01188 [−0.02116, −0.00426] at 24 weeks. MU differences were +0.00800 [−0.00010, +0.01551] and +0.00914 [+0.00004, +0.01844]. The latter is a narrow unadjusted patient-bootstrap interval from thirteen patients, not a general temporal-success claim. History-minus-current volume CRPS intervals include zero in all four strata. History volume MAE point estimates are worse than current-only in every stratum.

History volume CRPS improves over point-mass persistence in all four strata with patient-bootstrap intervals excluding zero. Its intervals nevertheless under-cover the nominal 90% level and are broad. The simple history-conditioned volume control achieves CRPS 8.74/14.57 mL on LUMIERE and 10.30/15.67 mL on MU. None of the spatial-history versus scalar-history CRPS intervals excludes zero; at LUMIERE 12 weeks the spatial model has worse volume MAE by 1.56 [0.73, 2.34] mL.

Image detail does not establish a separate success: history lesion MAE is 0.805/0.880 versus persistence 0.799/0.888 on LUMIERE, and 0.731/0.835 versus 0.744/0.852 on MU. Whole-brain errors are very close to persistence, with small increases except LUMIERE 24 weeks. Changed-case history Dice is 0.224/0.172 on LUMIERE and 0.357/0.331 on MU, below the corresponding stable-case Dice. Stable-case volume MAEs remain 6.97/4.92 and 9.20/16.43 mL; MU 24 weeks has only two stable targets. No successful prediction of change follows merely from accurate stable cases.

Independent recalculation of patient aggregates from all 259 records differed by at most 1.5e−14. Sixteen timeline distributions were checked; four trained map/image pairs replayed exactly, and scenario-versus-map volumes differed by at most 0.0000073 mL. The bounded downstream-context perturbations changed Dice by at most 0.0033 for a 5% intensity increase and 0.0068 for 4 mm resampling; they do not test full observation re-encoding or clinical acquisition robustness. The observation baseline and forty reserved patients remain untouched.

The training-boundary review gives no basis for a routine extension: both anatomical checkpoints selected update 4,000, and update 6,000 was worse on the bounded full-volume criterion, especially for history. No new fit or seed repeat is running. The redesigned system substantially reduces the earlier extreme anatomical errors and improves distribution scores, but **the useful predictive-state milestone remains unachieved**.

## Structured transition readiness and restored MU timing, November 6, 2026

The operator authorized replacing the failed generic feature transition with direct anatomical-change distributions and a simple volume-change control. The observation baseline remains fixed. This section records completed readiness work, not a positive forecasting result. See the [protocol](METHODS.md#structured-anatomical-transition-development-november-6-2026) and [dated build receipt](patient_structured_build_metrics.json).

Exact joins to the official MU diagnosis-relative MRI-day fields recovered the following history-eligible targets at the existing ±4-week windows. One nonincreasing trajectory and four unusable visit joins were excluded; no dates were interpolated. Forty reserved patients' images and segmentation measurements remain unopened.

| Population | 12-week fitting targets / patients | 24-week fitting targets / patients | 12-week selection targets / patients | 24-week selection targets / patients |
|---|---:|---:|---:|---:|
| LUMIERE | 127 / 38 | 79 / 33 | 112 / 16 | 89 / 15 |
| MU | 55 / 29 | 32 / 21 | 36 / 18 | 22 / 13 |
| Combined | 182 / 67 | 111 / 54 | 148 / 34 | 111 / 28 |

The six metadata-chosen MU correspondence checks had matching geometry, brain-support Dice 0.923–0.985, T1 correlation 0.611–0.900, and zero coarse translation peaks at 4 mm sampling. Visual checkerboards showed residual local differences. This supports using the released atlas geometry for the bounded development build; it does not establish perfect registration throughout MU.

The small anatomical mixture and scalar volume control passed real train/save/reload/scoring checks. The final anatomical smoke averaged 0.302 seconds per update across two real batches; this is a short throughput measurement, not a cohort runtime guarantee. Six structured contract tests and the existing seventeen patient-state tests passed. An ignored-cavity regression verifies that the spatial energy loss does not silently treat unannotated cavity as background. The queued job also requires a real MU smoke before cohort fitting.

The integrated smoke timeline contains three observed visits, twelve learned distributions, and six comparisons after the latest observation. Four issued map/image pairs replayed exactly from saved context and parameters. Displayed map volumes and their selected scenario volumes differ by at most 0.0000062 mL (floating-point counting). These are engineering consistency checks; the two-update models are explicitly marked as untrained instrument checks. No useful predictive result has been established for the redesigned formulation.

## Integrated forecasting first configuration, November 6, 2026

Both frozen-observation temporal models completed 8,000 updates. Current-only took 10,831 seconds and selected update 4,500; history took 10,603 seconds and selected update 5,000. Later feature log scores continued improving, while decoded selection quality did not improve over these checkpoints. The unchanged observation baseline remains preserved. Full-volume assessment completed all 201 common development-selection targets: 112 at 12 nominal weeks from 16 patients, and 89 at 24 weeks from 15 patients. See the [aggregate results](patient_forecasting_metrics.json) and [fixed protocol](METHODS.md#frozen-observation-predictive-timeline-november-5-2026).

**This configuration did not achieve useful future anatomy or incremental history value.** All figures below are patient-averaged, and volume MAE averages the three compartments. Higher Dice and lower errors are better.

| Horizon | Method | Mean Dice | Volume MAE, mL | Volume CRPS, mL | Lesion image MAE |
|---|---|---:|---:|---:|---:|
| 12 weeks | Persistence | 0.276156 | 11.029351 | 11.029351 | 0.799497 |
| 12 weeks | Current-only | 0.198170 | 44.591854 | 36.629097 | 0.751404 |
| 12 weeks | History | 0.199894 | 44.178229 | 38.847435 | 0.752104 |
| 24 weeks | Persistence | 0.216449 | 19.880102 | 19.880102 | 0.888332 |
| 24 weeks | Current-only | 0.189897 | 46.165027 | 38.430583 | 0.803895 |
| 24 weeks | History | 0.190389 | 46.401923 | 41.205969 | 0.803359 |

History-minus-current Dice differences were +0.001724 (paired patient-bootstrap 95% interval −0.003542 to +0.007244) and +0.000492 (−0.003277 to +0.004056). Volume and volume-change contrasts were also unresolved. History's volume CRPS was worse by 2.218339 mL (1.052443 to 3.429448) and 2.775386 mL (1.042467 to 4.663692). Both learned methods improved image errors over copying the current image, but that did not translate into useful anatomy. History had a small gradient-error advantage over current-only; lesion-image differences were unresolved.

**Uncertainty failed calibration.** Nominal 90% volume intervals covered only 4.08% and 8.31% of outcomes for history, with mean widths 25.15 and 24.05 mL. Current-only coverage was 7.52% and 10.06%, with widths 25.97 and 25.80 mL. These are averages across compartments and patients. Wide intervals centered on biased predictions did not supply useful uncertainty. Persistence's distribution is explicitly a point mass, not a calibrated interval comparator.

The failure was not concealed by grouping stability with change. The descriptive rule identified 17 stable / 95 changed targets at 12 weeks and 7 / 82 at 24 weeks. History's volume MAE was 39.91 / 44.31 mL in the stable / changed 12-week subsets, versus persistence 3.40 / 12.09 mL. At 24 weeks the corresponding errors were 35.19 / 46.29 versus 1.89 / 20.41 mL. Large false-positive compartment volumes, particularly FLAIR abnormality, are a concrete development blocker.

**Working timeline and bounded verification.** The trained four-visit timeline is retained under `temporal-selection-v1/timeline` on `mosaic-gpu`; it contains both horizons, all three methods, decoded means, 16-draw volume distributions, subsequent scans and references, errors, and saved updated states. It replaces the smoke preview on viewer port 8776. A verification recomputed patient averages from all 201 individual records (maximum discrepancy 7.1e-15), replayed all four learned forecasts in the demonstration with zero segmentation-voxel mismatches, and verified that zeroing the forecast residual recovers current-state logits and images exactly. Decoding the two subsequent observed states also reproduced the saved segmentations exactly, with mean Dice 0.538375 and 0.701223 against their reference masks. This bounded check found no saved-state/decoder replay defect; it does not identify the cause of learned forecast bias or establish global correctness. The private receipt is `temporal-selection-v1/verification.json`.

The predictive system now executes and displays evaluated forecasts, but the useful-predictive-state milestone remains unachieved. The result does not show that longitudinal MRI lacks predictive information. Forty MU reserved patients remain unopened. Do not repeat seeds or extend this unchanged schedule merely because feature loss fell; the next system improvement must address anatomical bias and uncertainty behavior while retaining the observation applications.

## Integrated observation baseline and bounded MU inspection, November 5, 2026

The one-seed observation schedule completed all 37,500 updates in 26,505.65 seconds. The selected checkpoint is update 30,000, SHA256 `87e2349c54fafd133a08246fd0d920956687faf640c8a080edc2f2cd1dc3ae59`. The late segmentation curve plateaued while image errors still improved slightly. The working checkpoint/configuration and both three-visit timelines are preserved separately in `observation-preserved-v1` on the approved GPU host.

The [retained aggregate](patient_state_observation_metrics.json) records full-volume selection on 178 LUMIERE visits from 18 patients and 106 MU visits from 32 patients. Patient-averaged complete-panel mean Dice was 0.710482 and 0.666625 respectively. On the **same 178 LUMIERE visits**, CT1/T1/T2-only Dice was 0.642746 for the joint state, 0.559978 for the R71 adapted comparator, and 0.476703 for its learned readout. This is a matched-input development gain, not causal attribution or independent generalization. Missing-FLAIR lesion MAE decreased from R69's 0.717111 to 0.573728 (20.0%), while whole-brain MAE increased from 0.374885 to 0.390696 (4.2%). Both sides of that tradeoff belong with the result. Shared supervised applications demonstrate multitask capability, not transfer to an unseen segmentation task.

A bounded six-case MU inspection examined two worst nonempty nonenhancing Dice cases, two worst absent-compartment false positives, and two additional largest cavities. Source masks were reoriented and remapped independently and matched the prepared targets exactly in all six. No mapping/orientation defect was found. The two worst nonempty cases had only 0.009 and 0.052 mL reference nonenhancing compartments; 0.009 and 0.051 mL were missed. Absent-compartment cases had 9.529 and 8.518 mL false positives **outside cavity**, respectively. In one inspected large cavity, an additional 86.740 mL was labelled nonenhancing; another had 61.059 mL, while another large cavity had zero. Visual overlays confirmed substantial, inconsistent cavity/core confusion. These selected extremes are failure examples, not cohort averages.

MU cavity remains ignore -1 in shared supervision and scoring, rather than becoming background. This preserves annotation differences but leaves cavity behavior unconstrained; the shared score can hide substantial cavity-related output volume errors. No reference cavity mask was used to erase inference predictions. The frozen baseline retains this weakness. MU mean nonenhancing Dice is 0.360328, versus 0.509391 conditional on a nonempty reference. The limitation does not indefinitely block LUMIERE temporal development. Forty reserved patients remain unopened.

Both new temporal arms passed two-real-batch train/save/reload/decoded-score checks, with exact checkpoint replay and unchanged observation weights. The full-volume two-horizon smoke also passed decoded uncertainty scoring, fixed-intensity sensitivity measurement, and subsequent-state assimilation. The unified timeline CLI passed two chronological real visits. Seventeen contract tests passed, including cutoff independence, interval scoring, and sample agreement across overlapping patches. The documentation checks also passed. The [forecast build receipt](patient_forecasting_build_metrics.json) records the implementation and launch, not forecasting accuracy. An initial concurrent CLI check hit its deliberately restricted 25% GPU-memory allowance; the same check passed at 30%, without interrupting training or changing predictions. These are implementation checks, not learned forecasting evidence. The [forecast protocol](METHODS.md#frozen-observation-predictive-timeline-november-5-2026) owns the selected comparison and acceptance limits.

## Integrated patient-state first working build, November 5, 2026

This is an integration and training-readiness result inside the [selected system effort](METHODS.md#integrated-patient-state-system-november-5-2026), not a new isolated representation claim. The [aggregate build receipt](patient_state_build_metrics.json) contains runtime checks, cohort counts and a dated training snapshot. Private outputs are under `/home/aditya/experiments/world-model-patient-state-20261105` on `mosaic-gpu`.

The R69/R71 bridge processed three chronological fitting-patient visits through encode, state serialization, both readouts, current segmentation and volumes, and issued persistence forecasts at 12 and 24 nominal weeks. A later visit displays a comparison with an earlier forecast, including the target-time offset. Both readouts consume the saved complete state. The existing scan viewer was inspected with current anatomy and prior persistence overlaid in separate colors. End-to-end runtime was 2.09–3.93 seconds per visit. The largest serialized-state tensor payload was 358,297,392 bytes, including support and the prior coarse-memory tensor. This is working integration, not a new accuracy result or a forecasting benefit.

The MRI-pretrained ResEnc-L joint model passed two effective-batch-two updates spanning LUMIERE and MU, with changed encoder weights, exact checkpoint replay for both outputs, exact saved-state decoding, and finite anatomical scoring. The final combined smoke measured 0.953 seconds per update including startup and 11.69 GiB peak allocated memory. The original whole-volume decoder exceeded GPU memory. Overlapping complete patch states resolved that blocker: one full-visit missing-FLAIR inference took 9.41 seconds, with 1,436,233,512 bytes (1.338 GiB) of state. Its checkpoint had only two smoke updates; its outputs are not evidence of usable anatomy. A full-volume evaluation instrument exercised segmentation, four missing sequences and an intensity nuisance perturbation on a selection visit.

MU restoration completed for all 163 development patients. The fixed forty reserved patients were not requested or opened. Preparation opened only MU fitting/selection images, yielding 281 fitting visits from 99 patients and 106 selection visits from 32 patients; one fitting visit lacked a released segmentation mask and was excluded. The combined observation population is 586 fitting visits from 154 patients and 284 selection visits from 50 patients. MU cavity is ignored for shared segmentation, not mislabeled as background. MU scoring images have not been opened during this build's preparation.

Metadata-only LUMIERE coverage supports provisional 12/24-week queries with a four-week target tolerance: fitting has 152/107 eligible cutoffs from 39/35 patients, selection 124/98 from 16/15, and reused scoring 49/36 from 14/9. History-eligible subsets are smaller and recorded in the receipt. Exact nominal-horizon visits are sparse, particularly at 24 weeks; a tolerance-window result is not exact-time validation.

The frozen-observation forecasting wiring check passed two real pairs in each of the matched current-only and history arms. Both trained predictor checkpoints replayed exactly; decoded anatomical outputs and empirical volume CRPS ran, and the observation checkpoint remained unchanged. This establishes runnable components only. Final forecast fitting, calibration, comparison against persistence, and useful predictive-state claims remain pending the observation-model review.

Twelve state/cutoff/availability/label/uncertainty contract tests passed on the approved host. The first long observation attempt stopped before its first checkpoint because a randomly hidden region could fall outside brain support. The repair centers region masking on supported brain voxels; its regression test and repeated two-dataset smoke passed. The original failed log and preparation interruption remain preserved. The repaired seed-17 run is `observation-v2`, with 37,500 planned updates, followed by full-volume selection evaluation and complete-panel/missing-FLAIR timeline exports. Training status belongs in [TODO](TODO.md#integrated-patient-state-development-november-5-2026); no trained-model usefulness result is recorded yet.

## R63 controlled-world pilot, November 3, 2026

The first study in the operator's [six-question program](METHODS.md#six-question-observability-program-november-3-2026) is complete as an exploratory pilot. The [aggregate](r63_observable_state_metrics.json) and [checkpoint-replay receipt](r63_observable_state_verification.json) preserve the results. The primary hypothesis that future prediction loses current activity relative to contemporaneous prediction was **not established**. Neither predictive objective demonstrated adequate lesion-state recovery under this recipe. The stronger shared result is that the random encoder supported 70.1 percent activity accuracy, compared with 58.7 percent after reconstruction training and about 51 percent after either predictive objective. Training made activity less accessible to these readouts than it was at initialization. That is not proof of information erasure.

### Design and execution

Four synthetic worlds crossed visible or hidden current activity with predictable or independently randomized future activity. Each used 1,024 fitting, 256 selection, and 512 assessment subjects. The same subjects and acquisition draws were shared across objectives and model seeds. The comparison included reconstruction, contemporaneous alternate-contrast prediction, and future prediction of that same alternate contrast. Continuous factor labels and binary current activity were excluded from representation training.

All 36 representation fits completed 600 updates. Twelve directly supervised reference fits and twelve random encoders supplied controls. Each of the sixty encoder cells received a fitted linear readout and a 600-update nonlinear readout. Three training seeds were 17, 23, and 31. The 512 assessment subjects are the independent generated units. Reusing them across seeds, objectives, and worlds does not increase the sample size.

The all-world smoke passed before the full run. It included two training updates, checkpoint reload, finite checks, and scoring, plus a real LUMIERE FLAIR center-slice reconstruction check on `mosaic-gpu`. Smoke and full-run source and protocol hashes matched. The real image established engineering execution only. No reserved MU image or measurement was opened.

**Protocol deviation:** the separate fixed-small-example fitting-competence check described in the common protocol was not performed before full assessment. The two-update smoke checked execution rather than factor recovery. The static review identified this after launch. The run therefore remains exploratory. No post-assessment fitting threshold, added seed, longer training run, or tuned objective was used to convert it into an accepted recovery result.

### Anatomy recovery did not imply activity recovery

The following scores use the frozen encoder's nonlinear readout in the visible, randomized-intervention world. Values are means across three seeds. Lower Brier score and normalized mean-squared error are better. Anatomy here means two ellipse axes, not clinical neuroanatomy. Lesion error averages center coordinates, radius, and ellipticity, normalized by fitting variance.

| Representation training | Activity Brier | Activity accuracy | Anatomy normalized MSE | Lesion normalized MSE |
|---|---:|---:|---:|---:|
| Reconstruction | 0.238509 | 58.72% | 0.0255 | 0.7997 |
| Contemporaneous view | 0.251241 | 51.43% | 0.0194 | 1.0193 |
| Future view | 0.249871 | 51.56% | 0.0203 | 1.0077 |
| Random frozen encoder | 0.194963 | 70.05% | 0.0685 | 0.7748 |
| Direct factor supervision | 0.108899 | 85.35% | 0.0275 | 0.5347 |

A fair-coin activity predictor has population Brier score 0.25. The fitting-prevalence control and full per-seed results are in the aggregate. The direct reference's activity accuracy ranged from 84.18 to 86.91 percent across seeds. Thus the images contained recoverable activity information, although not every geometric factor was recoverable well. Even the direct reference had mean ellipticity R-squared about zero and radius R-squared only 0.177. This does not demonstrate full generating-state recovery.

The primary future-minus-contemporaneous activity Brier difference was **−0.001370**, with a paired-subject 95 percent interval **[−0.002938, +0.000176]**. The three seed differences were −0.004794, +0.003241, and −0.002556. Positive values were the predicted direction for greater activity loss under future training. The observed differences did not support that hypothesis. The interval averages each subject's loss across seeds before 10,000 subject resamples and conditions on this simulator and fitting draw.

Both predictive representations recovered stable ellipse axes well, but their lesion-factor errors were near the fitting-mean reference. The corresponding predictable-future world did not rescue current-activity recovery either. The pilot therefore did not demonstrate a regime where prediction preserves activity when predictive and discards it when unpredictable. The supervised and random-encoder results prevent attributing this failure solely to genuinely invisible activity. They do not identify the unique cause or prove that no other readout could recover it. The supervised reference establishes recoverable activity information, not adequate optimization of the image-prediction models.

### Hidden-state uncertainty and output inspection

Opposite hidden-activity labels produced bitwise-identical current images and identical predictions. This is input consistency, not an uncertainty result. The relevant uncertainty reference is the known posterior probability of 0.5. For the prediction-trained nonlinear readouts, paired-counterfactual Brier scores ranged from 0.251441 to 0.253371 across seeds and hidden worlds, above the analytic optimum of 0.25. Their probabilities were not perfectly calibrated fair coins. No hidden-state recovery was established. The paired copies count as one generating subject, not two independent samples. Continuous-factor readouts provide point predictions, not calibrated continuous-state uncertainty.

The first eight predetermined visible-intervention subjects were inspected for reconstruction, contemporaneous prediction, and future prediction at seed 17. Predicted images retained broad ellipse shape and intensity but visibly suppressed small lesion details. The source lesion radius is only about 1.24 to 2.64 pixels before ellipticity, with additional resolution degradation in half the inputs. Whole-image squared error gives those structures little weight. This is a plausible limitation of the tested setup, not an experimentally isolated explanation.

The appropriate conclusion is that these whole-image prediction recipes can fit broad appearance without making local state accessible to the tested readouts. The pilot does not distinguish a special failure of future prediction from the shared failures of both predictive objectives. It does not establish biological validity, an information-theoretic limit, or general JEPA behavior. No further tuning of this scored pilot is selected.

### Generator-only follow-up: how much loss rewards activity

The operator selected this calculation after the pilot, without additional training. The [protocol](METHODS.md#r63-generator-only-activity-value-calculation) compares two privileged references on the same 512 geometries. One knows current activity. The other averages over it. Both know the other generating factors and average over unknown target resolution, gain, clipped noise, and future intervention. The [aggregate](r63_activity_value_metrics.json) retains expected risks, per-geometry ranges, numerical checks, and provenance.

| Target | MSE with activity averaged out | MSE with activity known | Available MSE reduction | Reduction relative to activity-averaged risk |
|---|---:|---:|---:|---:|
| Contemporaneous alternate view | 0.003754497 | 0.003688763 | 0.0000657334 | 1.751% |
| Predictable future | 0.003751109 | 0.003686105 | 0.0000650038 | 1.733% |
| Randomized future | 0.003751109 | 0.003751109 | 0 | 0% |

The visible contemporaneous and predictable-future targets reward current activity by a similar amount. Each reduction is about 0.5 percent of the corresponding cohort-average-image reference risk. This denominator is a descriptive oracle over these geometries, not the pilot's fitted mean-image baseline. Every assessed geometry has positive activity value for those two targets. Per-geometry reductions range from 0.00000931 to 0.00017283 for contemporaneous prediction and from 0.00000879 to 0.00019843 for predictable-future prediction.

Randomized-future prediction gains exactly nothing from current activity because its future activity is an independent fair intervention. The hidden contemporaneous control also has exactly zero activity value. These zeros concern privileged conditional target distributions. They are separate from a learned model's calibration or input consistency.

The calculation quantifies a small shared reward in the whole-image loss, rather than showing that activity carries no predictive information. It supports weak local-detail incentives as a plausible contributor. It does not establish that this incentive caused the trained models to miss activity, determine what gradients they received, or exclude inadequate optimization and decoding. The original predictable-future control still failed to demonstrate useful learned activity recovery. No positive learning regime has been established by this calculation.

The eight-geometry check passed before the full calculation. Analytic clipped-noise moments and 64-point gain quadrature agreed with 32-point quadrature to at most `6.67e-16` in the conditional first and second moments. Direct risk subtraction agreed with one-quarter of the squared conditional-mean difference to at most `2.30e-18`. Independently sampled acquisition checks for four fixed geometries agreed within two Monte Carlo standard errors for all reported risks and gaps. They used 16,384 draws per target. This numerical agreement is not a biological validation or a population confidence interval. No model was trained, no checkpoint changed, and no clinical image or reserved measurement was opened.

### Numerical verification

The full-run verifier checked 1,176 artifact hashes. A separate replay script regenerated the synthetic data and reconstructed selected models, features, fitting-only normalizers, ridge solutions, logistic predictions, and nonlinear predictions from saved checkpoints. It replayed 18,432 image predictions and 92,160 readout rows across sixty encoder cells. All reported maximum absolute replay errors were zero. It also checked 15,360 model-specific hidden counterfactual pairs, repeatedly using the same 512 generated subjects. These counts describe verification work, not independent statistical observations. [The analysis lineage record](r63_analysis_lineage.json) identifies the analysis source and published aggregate bytes. Intermediate training checkpoints were hash-checked, not replayed.

The private working bundle is `/home/aditya/experiments/world-model-observability-20261103` on `mosaic-gpu`. `full-v1` contains immutable run outputs and the exact source and protocol copies. `analysis-v1` contains the public aggregates and replay receipt. Local instruments, fixed contact sheets, and the decision log remain in `scratch/observable-state/`. `src/` was not changed by R63. Numerical replay and static review are not independent scientific replication.

## R64 repeat-scan access feasibility, November 3, 2026

Primary-source review confirmed that [RIDER Neuro MRI](https://www.cancerimagingarchive.net/collection/rider-neuro-mri/) describes nineteen recurrent-glioblastoma patients scanned about one or two days apart. [QIN GBM Treatment Response](https://www.cancerimagingarchive.net/collection/qin-gbm-treatment-response/) describes 54 participants with postoperative, pretreatment double-baseline imaging, typically two to five days apart. Sequence availability varies. These are collection descriptions, not newly verified usable-pair counts.

Both current image-access tables point to the [NIH controlled-access procedure](https://www.cancerimagingarchive.net/nih-controlled-data-access-policy/). The procedure requires an eligible investigator's dbGaP request, followed by an approved account and authenticated retrieval. No image access approval was established in this run. No repeat-scan images were downloaded or measured. The collection pages list no related analysis results supplying the independent longitudinal-change judgments required for R64 sensitivity. That finding does not prove that no such annotations exist elsewhere.

R64 is **access-blocked, not completed**. The [protocol](METHODS.md#r64-repeat-acquisition-measurement-limits) separates repeat variability from independently adjudicated change sensitivity and records the small-sample limitation on 5 percent false-positive calibration. Healthy travelling-head data and synthetic degradation cannot replace the tumor-repeat comparison. R65 through R68 have staged protocols but no newly completed measurements in this run.

### R65 metadata restoration

The old R62 GPU bundle is absent at its documented location. The official MU collection still provides its skull-stripped images under CC BY 4.0. A metadata-only request enumerated 203 package directories and reconstructed the original 40 reserved, 99 fitting, 32 selection, and 32 pilot-scoring allocation. The resulting membership SHA256 exactly matched R62's public `4df898761ecdb299bc87f99749fbebc8faab294043d454c4055850fcc676e2d6`. [The restoration receipt](r65_metadata_restoration.json) records zero downloaded image files and zero opened segmentation measurements. Private membership is in `mu-restored-metadata` under the new GPU run root. At this metadata-only stage, image retrieval and R65 learning had not been attempted. The subsequent transfer attempt is recorded below.

## R65A preparation and transfer blocker, November 3, 2026

After the operator closed R63, R65A was selected as the next learning experiment. Its [execution brief](METHODS.md#r65a-execution-brief-and-readiness-stage) fixes postcontrast T1, added precontrast T1, and added T2 comparisons for held-out FLAIR. Whole-brain and tumor-region MAE are separate primary endpoints. Broad image improvement alone will not establish a lesion-detail benefit. The synthetic loss-weighting question remains separate and unlaunched.

The preparation, two-update smoke, and 512-update fitting-only readiness instruments are implemented in `scratch/within-visit/run65.py`. No real-MU smoke or readiness run has completed. Full-cohort training, controls, assessment, and bootstrap scoring are not implemented yet. The planned methods describe those future steps, not executed capabilities.

The official Aspera SDK was restored on `mosaic-gpu`, and a transfer requested only the first two fitting patients in the existing R62 hash order. Authentication and directory creation succeeded. Native client records repeatedly reported zero received data blocks and zero image bytes. The tool's initial SSH wait timed out while the remote transfer remained active. The transfer was later interrupted manually after the native zero-data evidence was inspected. This was not an automatic compute-budget stop. The final client exit code was one. The network or server cause is unestablished. The returned transfer specification did not advertise an HTTP fallback endpoint or port, and no firewall settings or access restrictions were changed.

The attempt left 25 zero-byte NIfTI placeholders and no usable image files. An executed preparation check rejected those placeholders before image loading and retained its exclusion manifest. No reserved directory was present. Bounded checks of the documented data locations and likely backup directories did not find another MU image copy. This does not prove that no backup exists elsewhere.

[The preparation receipt](r65_preparation_status.json) records requested patient counts, zero image bytes, the rejected-empty-file check, and hashes of the membership, instrument, and native transfer log. Logs and placeholder files remain under `/home/aditya/experiments/world-model-r65-20261103` on the approved host. R65A is **blocked at image restoration**, not a completed experiment or a negative learning result. No MRI arrays, segmentation measurements, or assessment outcomes were opened in this stage, and no model training was launched.

## R65A LUMIERE readiness and job setup, November 3, 2026

The operator explicitly selected existing LUMIERE data instead of MU and requested job setup. The [LUMIERE protocol](METHODS.md#r65a-lumiere-substitution-november-3-2026) retains the three input arms and separate whole-brain and lesion endpoints. The metadata inventory assigns 55 fitting, eighteen selection, and eighteen scoring patients. It finds 305, 178, and 116 complete visits respectively, plus 39 incomplete visit directories. These counts establish file eligibility, not completed cohort array validation.

The raw acquisition images do not share the atlas-mask grid. Two fitting examples instead verified the released DeepBraTumIA atlas images and masks on their common 182-by-218-by-182, 1 mm LAS grid. No 96-cubed cache or new resampling was used. Fixed overlay inspection showed lesion outlines at corresponding image locations in all four contrasts. FLAIR still has visible thick-slice acquisition blur despite the 1 mm output grid.

The real-data two-update smoke passed finite loss and gradient checks, target exclusion, matched draws, changed parameters, exact prediction reload, and optimizer-state value reload for all three arms. Its first attempt stopped on a verifier bug that compared CPU optimizer tensors to CUDA tensors. Comparing the same state values on CPU corrected that check; the model and fitting recipe were unchanged.

All three arms passed the predeclared 512-update two-example readiness test. The mean of the two fixed target planes had whole-brain MAE 0.518124 and lesion MAE 0.687394. Each arm had to halve both errors relative to that control and to its own initialization. Each model had 490,594 parameters, and both encoder and decoder changed.

| Input arm | Initial whole-brain MAE | Final whole-brain MAE | Initial lesion MAE | Final lesion MAE | Readiness |
|---|---:|---:|---:|---:|---|
| A, postcontrast T1 | 0.923081 | 0.051765 | 1.540219 | 0.044806 | Passed |
| B, added precontrast T1 | 0.910539 | 0.054543 | 1.511474 | 0.050326 | Passed |
| C, added T2 | 0.907389 | 0.052085 | 1.520101 | 0.043064 | Passed |

These are fitting-plane diagnostics, not generalization results or evidence that one sequence arm is better. Visual inspection of both fixed examples for arms A and C showed close target reproduction, with larger residuals near brain edges. A subsequent full-volume scorer exercise used the same two fitting patients. Its scores are instrument checks, not cohort assessment, and its poor uncertainty coverage is not a calibrated cohort result.

The scorer check initially needed the missing SciPy dependency. After installing SciPy 1.16.3, its identical-image test exposed float16 subtraction in the baseline gradient calculation versus float32 target subtraction. Converting baseline predictions to float32 before scoring corrected the mismatch. The rerun returned exactly zero for all applicable identity-image error metrics and exercised the mean-image, ridge, neural, compartment, contrast, gradient, and uncertainty scoring paths. No readiness training was repeated for these scorer corrections.

Three OpenCode restarts interrupted data transfer, not completed fitting or checks. The cancelled whole-archive transfer was replaced by 2,995 selected atlas image and mask files, about 11.9 GiB, copied from the existing local release. The second interrupted transfer resumed with `rsync --partial`. Local staging only copied released compressed files; image-array checks and fitting ran on `mosaic-gpu`. The third restart broke the detached SSH transfer too, superseding the earlier claim that detachment alone would protect it. No remote cohort job had started.

Dispatch now runs as the standard user service `r65-lumiere-transfer-20261103.service`, observed active and running at setup. Its failure-restart policy retries after thirty seconds, and rsync retains partial files. A pre-existing remote job marker prevents a transfer retry from dispatching a duplicate cohort job. This is a service-manager policy, not a custom training or compute-budget supervisor. Survival of a further server restart has not been tested. The laptop must remain awake and connected until transfer completes. Successful transfer chains to remote preparation, nine matched fits, scoring preparation after fit completion, and assessment. The remote job stops on command failure rather than bypassing a failed check. At setup handoff, cohort training was queued behind transfer and full array validation, not reported as running or completed.

[The aggregate setup receipt](r65_lumiere_setup.json) contains the fixed membership hash, readiness measurements, scorer receipt, source hashes, environment, and job phase at export, plus the later observed local service state. Private membership, images, masks, checkpoints, and detailed logs remain outside git. The remote root is `/home/aditya/experiments/world-model-r65-lumiere-20261103`; current local transfer output is `scratch/within-visit/service.log`. Earlier `dispatch.log` and `dispatch.pid` describe the interrupted detached attempt. No scientific R65A conclusion is established by job setup.

## R65A sequence additions improved held-out FLAIR prediction, November 3, 2026

The LUMIERE job completed all nine matched fits and its assessment with exit code zero. Scoring used eighteen patient-disjoint development patients and 116 complete visits. Four visits had no supported lesion label, leaving 112 lesion-scored visits and all eighteen patients. Visits were averaged within patients before patients received equal weight. These patients were reused historically, so this is developmental evidence rather than untouched validation.

The following MAEs are averages of the three fitted-seed scores, not errors from an ensemble prediction. Units are per-image foreground intensity standard deviations. Lower is better. The model always predicts contemporaneous FLAIR and never receives it as an input.

| Predictor | Whole-brain MAE | Lesion-region MAE |
|---|---:|---:|
| A, postcontrast T1 | 0.448905 | 0.906141 |
| B, added precontrast T1 | 0.437853 | 0.857814 |
| C, added T2 | 0.403502 | 0.592937 |
| Fitting-only mean image | 0.668191 | 1.541224 |
| Pointwise ridge A | 0.682560 | 1.694286 |
| Pointwise ridge B | 0.629371 | 1.614789 |
| Pointwise ridge C | 0.626004 | 1.528660 |

Every neural seed beat the mean-image control and its corresponding ridge on both primary endpoints. The neural models also passed the separate pre-assessment fitting-readiness check. This establishes a usable direct-image reference for this comparison, not clinical image quality or superiority over every possible synthesis baseline.

### Primary sequence comparisons

| Added sequence | Endpoint | Error reduction | Mean difference | Paired 98.75 percent patient interval |
|---|---|---:|---:|---|
| Precontrast T1, B minus A | Whole brain | 2.46% | -0.011052 | [-0.021184, -0.002897] |
| Precontrast T1, B minus A | Lesion | 5.33% | -0.048326 | [-0.084880, -0.003050] |
| T2, C minus B | Whole brain | 7.85% | -0.034350 | [-0.051478, -0.021648] |
| T2, C minus B | Lesion | 30.88% | -0.264877 | [-0.343851, -0.197277] |

All four multiplicity-adjusted primary intervals are below zero, and all three training seeds favor each addition on both endpoints. After seed averaging, T2 improves both endpoints in all eighteen patients. Precontrast T1 improves whole-brain error in fourteen patients and lesion error in fifteen. The precontrast-T1 lesion interval barely excludes zero and is less convincing than the larger T2 effect. Three matched fits establish observed seed-direction consistency, not independent replication. The patient bootstrap operates on seed-averaged patient differences and does not estimate uncertainty over arbitrary new training seeds. Eighteen patient clusters limit its tail precision.

The lesion MAE ranges across seeds are 0.876859 to 0.926997 for A, 0.824196 to 0.896947 for B, and 0.571430 to 0.608538 for C. Corresponding whole-brain ranges are 0.447960 to 0.449730, 0.434656 to 0.440110, and 0.400348 to 0.408934.

### Lesion details and uncertainty

T2 improved all reported compartment-error, lesion-contrast, and lesion-local-gradient point estimates relative to B in every seed. Seed-averaged lesion-contrast error fell from 0.684966 to 0.398669, and lesion-local gradient MAE fell from 0.219226 to 0.203734. These are secondary descriptive comparisons, not additional multiplicity-controlled success criteria. The gradient score includes lesion interiors and nearby edges, so it does not measure a pure anatomical boundary distance.

Precontrast T1 did not improve every secondary endpoint in every seed. Enhancing-region MAE worsened for seeds seventeen and twenty-three, and lesion-contrast error worsened for seed thirty-one. Its smaller union-lesion gain must not be described as uniform compartment recovery.

Whole-brain central 90 percent interval coverage averaged 88.28%, 88.75%, and 88.78% for A, B, and C respectively. Across individual fits it ranged from 86.88% to 90.17%. These marginal averages do not establish lesion-conditional calibration or reliable individual-patient uncertainty.

The two preselected scoring examples show that adding T2 can recover more of the lesion's bright structure than the T1-only arms. Predictions remain visibly smoothed and miss fine texture. The first example's T2-input predictions also show visible banding. Whole-image and lesion-crop montages were inspected across all three seeds, with identical target-mask outlines overlaid only for display. The fixed examples illustrate the measured benefit rather than proving clinical adequacy or representing all patients.

[The aggregate result and verification receipt](r65_lumiere_metrics.json) records a full assessment replay from all nine selected checkpoints, including refitting the fixed mean-image and ridge controls. All 14,249 numeric values matched exactly. Independent visit-to-patient aggregation matched exactly, and an independent calculation reproduced the four primary bootstrap intervals. Checks confirmed disjoint groups, identical initialization and training draws across arms within seed, changed model parameters, all 4,096 updates in each fit, eight selection opportunities, and selection of the minimum whole-brain-MAE checkpoint. No further neural fitting, changed seed choice, or tuned threshold was part of this evaluation.

The prepared data matched the planned 55 fitting, eighteen selection, and eighteen scoring patients, with 305, 178, and 116 complete visits respectively. Only 209 of 9,053,840 labeled scoring voxels fell outside the common postcontrast-T1 support, about 0.0023 percent. Lesion scoring consistently excludes those voxels in every arm.

The result concerns incremental information usable by this direct image predictor. It does not establish biological state, compact-state retention, temporal benefit, acquisition robustness, or prospective inference without full-panel preprocessing. R65B resolution learning and the advanced-imaging extensions remain unrun.

## R66 mask-free preflight, November 3, 2026

All three storage sizes passed the fixed two-example representation readiness check on `mosaic-gpu`. This is preflight, not a cohort result or evidence of segmentation reuse. The [fixed protocol](METHODS.md#r66-fixed-two-application-protocol-november-3-2026) and [generated receipt](r66_preflight_metrics.json) retain the full definitions and measurements. No R66 cohort or segmentation fits have been dispatched.

The first smoke stopped when the equal-shaped decoder projection bias differed across budgets because its initialization range depended on fan-in. Setting convolution biases to zero corrected that invariant without changing the architecture, storage, training budget, or readiness thresholds. The corrected two-update smoke passed at every size, with finite gradients, changed encoder and decoder parameters, exact checkpoint and optimizer reload, and exact Float16 state reload and decoded scores. Payloads contained exactly the required bytes without file headers or visit-specific metadata.

Readiness used panel C, seed seventeen, and exactly 512 updates on the first two fitting patients' earliest complete visits. Patient draws matched across sizes. Only four image or support arrays were opened, with no segmentation masks or scoring arrays. The two-image mean predictor had whole-brain MAEs of 0.440609 and 0.440071.

| Budget | Bytes per visit | Initial MAE, examples one and two | Final MAE, examples one and two | Fixed readiness | 512-update time | Peak allocated GPU memory |
|---|---:|---|---|---|---:|---:|
| 16 KiB | 16,384 | 1.020951, 1.110479 | 0.429019, 0.417799 | Both pass | 84.31 s | 5.48 GiB |
| 64 KiB | 65,536 | 1.055761, 1.120993 | 0.388247, 0.385778 | Both pass | 84.49 s | 5.48 GiB |
| 256 KiB | 262,144 | 1.030425, 1.072038 | 0.363692, 0.353191 | Both pass | 84.51 s | 5.49 GiB |

Every example more than halved its own initial error and beat its mean-image control. Visual inspection of all six fixed prediction panels found broad anatomy but strong blur and missing fine texture. The whole-brain readiness result does not establish lesion-detail retention. It does not choose the preferred size, which still requires cohort selection performance.

The encoder and FLAIR decoder together have 551,932, 552,706, and 555,802 parameters at the three respective sizes. Representation updates averaged about 0.165 seconds on an NVIDIA GeForce RTX 5090. Peak allocation is PyTorch allocated memory, not total device memory. Frozen-decoder timing surrogates took about 0.092 seconds per update with 3.30 GiB peak allocation. The skip-connected direct-image surrogate took 0.167 seconds with 5.77 GiB peak allocation.

The nominal projection for all ninety fits at 4,096 updates each, repeated selection, state export, and assessment is **15.31 GPU hours**. A hypothetical higher-cost scenario is **29.89 GPU hours**, obtained by doubling segmentation-stage surrogate work and adding 25 percent overhead. This scenario is not a measured confidence interval or a guaranteed upper bound.

Segmentation timing used real FLAIR targets and a two-output image loss on the corresponding decoder and direct-image architectures. Those probe weights are discarded. Four-class cross-entropy and soft-Dice costs remain unmeasured. The probes used two CPU-resident examples, so full-cohort random disk loading remains unmeasured too. The projection excludes implementation and scientific review time and does not shorten any fit. Cohort representation fitting, preferred-size locking, full state export, segmentation reference readiness, segmentation fitting, and joint acceptance evaluation remain unimplemented. Both-application reload verification also remains pending the segmentation stage.

### R66 cohort dispatch after the preflight report

The operator approved proceeding after the cost report. The separate cohort runner reproduced the fixed 305 fitting, 178 selection, and 116 scoring visit inventory without loading segmentation measurements. Two real fitting updates, checkpoint reload, optimizer-value reload, saved-state selection scoring, and patient-weighting checks passed on `mosaic-gpu`.

An extra exact-next-update check initially failed despite identical checkpoint values and decoded scores. A controlled probe reproduced small next-update differences between two identical CPU-mapped reloads. PyTorch strict deterministic mode identified `adaptive_avg_pool3d_backward_cuda` as lacking a deterministic implementation. The corrected smoke retained exact checkpoint, optimizer-value, and inference checks and recorded the next-update maximum parameter difference of 0.000000450760. It did not change the architecture or scientific criteria. Matched initialization and draws do not imply bitwise deterministic GPU training.

The persistent service `world-model-r66-representations.service` started on November 3, 2026 at 21:44:09 IST. It runs all 27 fixed representation fits, then exports fitting and selection states and locks the preferred size before exporting scoring and matched random states. The [dated execution receipt](r66_cohort_status.json) records launch status and source hashes. No segmentation job has been dispatched. Representation outcomes and all joint acceptance decisions remain pending. This dispatch supersedes the preflight entry's earlier unimplemented and undispatched status, not its measurements or limitations.

Synthetic checks on the approved host passed the segmentation empty-region rules, present-class Dice loss, support exclusion, paired patient bootstrap, and ten-condition decision logic. Both the segmentation and assessment command-line entry points rejected execution while the representation lock was absent. These checks opened no real masks. Segmentation training and assessment are implemented but still require post-lock real-data verification and execution.

### R66 representations locked, November 4, 2026

All 27 representation fits completed the fixed 4,096 updates and eight selection opportunities. The service completed fitting, preferred-size locking, and learned/random state export in 6 hours 34 minutes 47 seconds. Segmentation labels remained unopened until the representation lock. The [representation receipt](r66_representation_metrics.json) contains individual seeds, selected updates, panel means, and verification results.

| Whole-visit budget | Selection whole-brain FLAIR MAE | Deficit from best selection MAE | Within 0.01 of best |
|---|---:|---:|---|
| 16 KiB | 0.497441 | 0.019016 | No |
| 64 KiB | 0.487339 | 0.008914 | Yes |
| 256 KiB | 0.478425 | 0 | Yes |

The fixed rule therefore selects **64 KiB** as the preferred size. These averages weight selection patients, panels, and seeds equally. Preferred size is now frozen; later results cannot replace it. All three sizes still enter assessment.

Independent recalculation from every selection visit reproduced the patient-averaged means to within 0.000000000000000222. All selected checkpoints matched their best whole-brain selection scores. Draw hashes matched across conditions within seeds, and initial model hashes matched across panels within each size and seed. The export manifests contain 13,041 learned fitting/selection states and 19,305 additional random and scoring states. These checks verify the implementation, not scientific acceptance.

The post-lock segmentation smoke and nine direct-reference readiness checks are now running. An initial launch prerequisite check found that Matplotlib was absent, before any segmentation process started. Diagnostic rendering now uses the already available Pillow library, without changing model computation, loss, sampling, or thresholds. Synthetic segmentation checks passed again after this rendering-only change. No cohort segmentation fits have been dispatched, and lesion retention, segmentation reuse, and joint acceptance remain unassessed.

### R66 segmentation-reference readiness failed, November 4, 2026

The segmentation smoke passed exact model prediction and optimizer-value reload checks for a learned-state decoder and a direct-image reference. The subsequent readiness service completed all nine checks in 13 minutes 45 seconds, but **five references failed** the fixed mean nonempty-compartment Dice target of 0.80 after 512 updates. A successful process exit therefore does not mean scientific readiness passed. See the [readiness receipt](r66_segmentation_readiness_metrics.json).

| Input panel | Seed 17 Dice | Seed 23 Dice | Seed 31 Dice | Passing references |
|---|---:|---:|---:|---:|
| A, postcontrast T1 | 0.608563 | 0.904674 | 0.611784 | 1 of 3 |
| B, both T1 sequences | 0.861302 | 0.905720 | 0.663160 | 2 of 3 |
| C, both T1 sequences and T2 | 0.853184 | 0.592624 | 0.632262 | 1 of 3 |

The scores average the reference-nonempty compartments within each of the same two fitting visits, then average visits. Both visits contain all three compartments. Recalculation from saved reference, prediction, and intersection counts reproduced every Dice value and pass/fail decision.

All five failures showed severe underprediction of label two, the automated necrotic/nonenhancing compartment. Its two reference volumes contain 353 and 1,542 voxels within support. The failing references predicted at most 29 and 166 voxels respectively, with compartment Dice from zero to 0.188525. Enhancing and edema compartments had much higher overlap. This is an observed fitting failure, not an established explanation of its optimization cause.

All eighteen fixed diagnostic images were inspected. They support the missing-compartment finding. The raw panels also display foreground predictions outside the postcontrast-T1 support. Those voxels are outside both the specified loss and scoring domain; their appearance must not be interpreted as a contradiction of the support-restricted Dice values. No cleanup was applied to scored predictions.

The study is paused before the 63 cohort segmentation fits. Every panel has at least one failed seed, so its full planned reference comparison remains unestablished. No training was extended, no threshold was weakened, and no passing seeds were substituted for failed ones. The preferred 64 KiB choice and all representations remain locked. This readiness failure does not establish that a compact state loses segmentation information, that any byte budget is insufficient, or that the ten-condition joint claim fails. Those comparisons have not been run.

### R66 separate cross-entropy weighting diagnostic completed, November 4, 2026

All six panel-C fits completed exactly 512 updates on the same two complete fitting visits. Each seed pair shared initial weights and example draws. The only training change was cross-entropy averaging over present reference classes, including background, instead of supported voxels. Foreground Dice and the equal one-half loss weights stayed unchanged. See the [fixed diagnostic protocol](METHODS.md#r66-cross-entropy-weighting-diagnostic-november-4-2026) and [verified aggregate](r66_ce_weighting_metrics.json).

The primary endpoint is final necrotic/nonenhancing Dice, averaged equally over the two visits within each seed. Changes are class-balanced minus original, in absolute Dice units.

| Seed | Original label-two Dice | Class-balanced label-two Dice | Paired change | Enhancing Dice change | Edema Dice change |
|---|---:|---:|---:|---:|---:|
| 17 | 0.748841 | 0.905659 | +0.156818 | +0.017505 | -0.027645 |
| 23 | 0.000000 | 0.904583 | +0.904583 | +0.113085 | -0.014809 |
| 31 | 0.055835 | 0.687008 | +0.631173 | +0.036218 | -0.067275 |

Label-two Dice improved in all three seed means and all six visit-level pairs. Enhancing mean Dice also improved for every seed, while edema mean Dice declined for every seed. These are descriptive fitting results on two reused visits, not population estimates or generalization evidence.

| Seed | Example-one label-two Dice, original to balanced | Example-two label-two Dice, original to balanced | Original predicted voxels, examples one and two | Balanced predicted voxels, examples one and two |
|---|---|---|---|---|
| 17 | 0.692308 to 0.889169 | 0.805374 to 0.922148 | 245, 1,510 | 441, 1,772 |
| 23 | 0.000000 to 0.882500 | 0.000000 to 0.926667 | 0, 0 | 447, 1,758 |
| 31 | 0.065041 to 0.554160 | 0.046629 to 0.819856 | 16, 45 | 921, 2,205 |

The reference label-two volumes contain 353 and 1,542 supported voxels. Seed 31's balanced fit recovered all 353 reference voxels in example one but predicted 921 voxels. Its remaining low Dice therefore reflects substantial false positives rather than continued omission. All three final comparison images were visually inspected, covering both examples per seed. They show recovered label-two regions and some oversegmentation. These images mask the unscored exterior for display only. Scoring still uses unchanged raw argmax within the common support, without cleanup.

Final mean label-two probabilities inside its reference region increased from 0.64644 to 0.96042, 0.12748 to 0.96360, and 0.12827 to 0.97450 for seeds 17, 23, and 31. Outside-region means changed from 0.021602 to 0.004584, 0.004536 to 0.001942, and 0.085919 to 0.092206. Original seed 23 assigned every missed label-two voxel to enhancing tissue. Seed 31's misses were mainly enhancing tissue and background.

At identical initialization, the mean half-weighted cross-entropy gradient on the label-two head bias changed from +0.096230 to -0.024458, +0.124392 to -0.011709, and +0.169370 to +0.024394. Initial Dice gradients remained identical within pairs. Present-class averaging increases each label-two voxel's cross-entropy weight by factors of about 1,190.793 and 234.316 in the two examples. The recorded mean label-two-logit gradients inside its reference region were negative for both losses, favoring higher label-two logits. Full-model gradient magnitudes and relative directions varied across training and seeds. These observations support a role for cross-entropy weighting, not a universal opposition between cross-entropy and Dice or a complete explanation of optimizer behavior. The saved half-weighted loss gradients precede AdamW and weight decay and do not decompose the optimizer update.

Both two-update real-data smoke runs passed. The six-fit service finished successfully in 11 minutes 13 seconds. All thirty observation checkpoints replayed on both visits with exact saved predictions and label-two probabilities. Independent checks reproduced confusion counts, Dice, volumes, missed-class counts, parameter-gradient norms, and analytic cross-entropy logit gradients. Probability-mean discrepancies were at most 1.39e-16 and analytic cross-entropy logit-gradient mean discrepancies at most 5.39e-12. Initial predictions, initial weights, and example draws matched within every pair. Measurement left model, optimizer, and stored parameter gradients unchanged.

The original-condition runs are new fits, not recovered old readiness trajectories. The old runner saved no fitted checkpoints or learning curves, and CUDA training is nondeterministic. Original R66 records and protected checkpoints passed unchanged-file checks. The failed readiness outcome, locked representations, preferred 64 KiB size, and acceptance thresholds remain unchanged. No cohort segmentation fit was dispatched. This diagnostic does not replace readiness seeds or authorize resuming R66.

### R66 locked-state FLAIR assessment, November 4, 2026

Under the [assessment-order amendment](METHODS.md#r66-assessment-order-amendment-and-segmentation-revision-november-4-2026), all 27 locked FLAIR decoders were assessed before revised segmentation readiness. The [verified aggregate](r66_flair_metrics.json) contains every endpoint, ordinary 95 percent interval, adjusted bound, seed result, and verification receipt. No model was fitted or reselected. Each decoder received only its saved Float16 state. Scoring covered the original eighteen patients and 116 visits, with four empty-lesion visits excluded from lesion MAE as in R65A. The three sizes contain exactly 16,384, 65,536, and 262,144 bytes per visit. The preferred size remains 64 KiB.

All three sizes fail the fixed FLAIR branch under this encoder-and-decoder recipe. Panels B and C fail the absolute lesion-error deficit criterion at every size. Panel A fails at 16 and 64 KiB and is unresolved at 256 KiB. The paired retained-T2 criterion is unresolved at every size. These findings do not establish a universal storage limit or absence of lesion information from every possible readout of these states.

R65A's actual patient-level references give lesion MAEs of 0.906141, 0.857814, and 0.592937 for panels A, B, and C. A deficit is state MAE minus the matched direct-image MAE, in standardized intensity units. Passing requires its adjusted upper bound to be at most 0.05. A lower bound above 0.05 establishes failure of that condition. The paired bounds use the original 20,000 patient bootstrap draws with seed 6601 and seed averaging within patients. Each displayed lower or upper bound has one-sided 98 and one-third percent confidence, with the original three-size correction. They are not a 98 and one-third percent two-sided interval or simultaneous protection for every individual contrast.

| KiB | Panel | State lesion MAE | Paired deficit | Adjusted lower and upper bounds | Condition | Deficits for seeds 17, 23, 31 |
|---|---|---:|---:|---|---|---|
| 16 | A | 1.004226 | +0.098085 | [+0.057898, +0.138617] | fail | +0.106976, +0.052630, +0.134648 |
| 16 | B | 1.038204 | +0.180389 | [+0.127074, +0.230813] | fail | +0.210855, +0.135426, +0.194887 |
| 16 | C | 0.840178 | +0.247241 | [+0.214736, +0.285917] | fail | +0.276682, +0.215581, +0.249461 |
| 64 | A | 1.009566 | +0.103425 | [+0.061508, +0.141191] | fail | +0.065950, +0.120615, +0.123709 |
| 64 | B | 1.001706 | +0.143891 | [+0.092987, +0.194041] | fail | +0.114935, +0.199614, +0.117126 |
| 64 | C | 0.787663 | +0.194726 | [+0.167778, +0.228364] | fail | +0.224499, +0.142027, +0.217652 |
| 256 | A | 0.952469 | +0.046328 | [+0.006325, +0.084916] | unresolved | -0.023590, +0.077270, +0.085305 |
| 256 | B | 1.016649 | +0.158834 | [+0.108954, +0.209059] | fail | +0.116504, +0.170477, +0.189522 |
| 256 | C | 0.791756 | +0.198819 | [+0.172686, +0.228063] | fail | +0.257895, +0.143067, +0.195496 |

R65A's paired added-T2 lesion-MAE reduction is 0.264877. The state benefit is panel-B minus panel-C lesion error. The retention margin subtracts 0.75 times the matched direct benefit. Passing requires the margin's adjusted lower bound to be at least zero. Every state's per-seed T2 benefit is positive, but every size's margin bounds cross zero. The descriptive ratios below divide the aggregate benefits. They do not replace the paired margin test.

| KiB | State T2 benefit | Descriptive benefit retained | Retention margin | Adjusted lower and upper bounds | Margins for seeds 17, 23, 31 |
|---|---:|---:|---:|---|---|
| 16 | 0.198025 | 74.76% | -0.000633 | [-0.044419, +0.047057] | +0.015552, -0.026241, +0.008790 |
| 64 | 0.214043 | 80.81% | +0.015385 | [-0.033620, +0.068215] | -0.028185, +0.111501, -0.037161 |
| 256 | 0.224893 | 84.90% | +0.026235 | [-0.021893, +0.073181] | -0.060012, +0.081324, +0.057391 |

Whole-brain MAE is a companion endpoint, not a replacement for lesion acceptance:

| Predictor | Panel A | Panel B | Panel C |
|---|---:|---:|---:|
| R65A direct images | 0.448905 | 0.437853 | 0.403502 |
| 16 KiB state | 0.527324 | 0.521514 | 0.498999 |
| 64 KiB state | 0.521322 | 0.514506 | 0.488966 |
| 256 KiB state | 0.511355 | 0.507557 | 0.480227 |

All six fixed panel-C seed-17 prediction images were inspected across the three sizes and two visits. Predictions localize broad bright lesion regions but remain strongly blurred, with missing internal texture and boundary detail. These examples illustrate the error pattern, not an independent measurement of its prevalence or cause.

The initial verification service completed scoring and all 3,132 independent-score replay decodes, then stopped at exact aggregate dictionary equality. Every visit-level comparison had already passed its existing 1e-12 absolute tolerance. The original aggregate and scorer remain unchanged. A separate verifier repeated all decodes and saved the replayed aggregate and numerical differences. Independent NumPy visit MAEs differed by at most 2.78e-16. Exactly one aggregate field differed by 1.11e-16: the 16 KiB panel-A whole-brain MAE upper bound, 0.5466134173229668 versus 0.5466134173229669. This is numerical rounding, not a changed scientific result. All condition decisions matched exactly, and all aggregates met the 1e-12 absolute tolerance.

A separate implementation of patient aggregation and paired bootstrap compared 528 numeric values with maximum difference 2.22e-16. All FLAIR conditions and statuses matched. The real-data smoke verified exact prediction reload on both fixed visits at each size. Protected original checkpoints, manifests, representation lock, readiness record, and R65A reference metrics remained unchanged. The raw first service exit remains a failed verification record, not a successful service claim. The subsequent independent replay, statistics check, and aggregate export completed successfully.

Segmentation remains unassessed, and the joint ten-condition comparison has not been executed. Nevertheless, each size fails a necessary FLAIR condition, so none can satisfy the fixed full acceptance claim with these representations and selected decoders. Revised segmentation can test a separate anatomical reuse question, but cannot undo that failed necessary condition. Its class-balanced recipe remains fixed as authorized, with fresh readiness required. No revised segmentation fit has started.

November 4 closure. After reviewing the verified FLAIR result, the operator closed R66's fixed joint claim as failed and deferred the revised segmentation readiness and cohort matrix. The failure follows from measured necessary FLAIR conditions, not from running or failing the unexecuted segmentation comparisons. All original results, locked states, and the preferred 64 KiB choice remain unchanged. The proposed privileged state-fitting diagnostic is a separate, unlaunched question and cannot replace this result.

### Privileged frozen-decoder state fitting, November 4, 2026

All six authorized state fits completed exactly 512 Adam updates, with no extensions or checkpoint selection. The service finished successfully in 4 minutes 4 seconds. Each case used panel C, the selected 64 KiB model, and one of the same two fitting visits. Only a Float32 master state changed. Encoder and decoder parameters stayed frozen, each forward pass cast the state to Float16, and final scoring used the reloaded 65,536-byte payload after update 512. The [fixed protocol](METHODS.md#privileged-frozen-decoder-state-fitting-diagnostic-november-4-2026) and [verified six-case aggregate](r66_state_fitting_diagnostic_metrics.json) retain the exact definitions and measurements.

**Zero of six cases satisfied all three recovery conditions.** Every case passed lesion MAE and nonlesion MAE, but every case failed the unchanged R65A boundary-gradient criterion. The ordinary states also satisfied the full conjunction in zero cases. These are six state fits on two visits, not six independent patients. All R65A reference predictions came from the matching seed's selected panel-C model on that individual visit and were generated before state fitting.

The lesion-MAE condition permits at most the matched R65A error plus 0.05:

| Seed | Example | Ordinary | Matched R65A | Optimized | Maximum allowed | Condition |
|---|---|---:|---:|---:|---:|---|
| 17 | 1 | 0.597927 | 0.365096 | 0.157989 | 0.415096 | pass |
| 17 | 2 | 0.422627 | 0.353956 | 0.154475 | 0.403956 | pass |
| 23 | 1 | 0.522175 | 0.329273 | 0.160417 | 0.379273 | pass |
| 23 | 2 | 0.454706 | 0.376056 | 0.154426 | 0.426056 | pass |
| 31 | 1 | 0.565084 | 0.305631 | 0.169084 | 0.355631 | pass |
| 31 | 2 | 0.419165 | 0.311739 | 0.158412 | 0.361739 | pass |

The gradient condition permits no increase above the matched R65A boundary-gradient error. This remains the absolute error of signed in-plane adjacent differences within the three-dilation lesion band, not a new gradient definition:

| Seed | Example | Ordinary | Matched R65A and maximum allowed | Optimized | Condition |
|---|---|---:|---:|---:|---|
| 17 | 1 | 0.151700 | 0.114032 | 0.132437 | fail |
| 17 | 2 | 0.179893 | 0.155533 | 0.168711 | fail |
| 23 | 1 | 0.154424 | 0.117867 | 0.130766 | fail |
| 23 | 2 | 0.187640 | 0.155931 | 0.164978 | fail |
| 31 | 1 | 0.149477 | 0.120823 | 0.132537 | fail |
| 31 | 2 | 0.181660 | 0.156761 | 0.164454 | fail |

The nonlesion-MAE condition permits at most the ordinary encoded-state error plus 0.05. The R65A nonlesion error is descriptive here, not the threshold:

| Seed | Example | Ordinary | Matched R65A | Optimized | Maximum allowed | Condition |
|---|---|---:|---:|---:|---:|---|
| 17 | 1 | 0.459764 | 0.305389 | 0.366066 | 0.509764 | pass |
| 17 | 2 | 0.470921 | 0.311646 | 0.356045 | 0.520921 | pass |
| 23 | 1 | 0.483944 | 0.316105 | 0.368932 | 0.533944 | pass |
| 23 | 2 | 0.491301 | 0.320804 | 0.362256 | 0.541301 | pass |
| 31 | 1 | 0.481199 | 0.319800 | 0.374696 | 0.531199 | pass |
| 31 | 2 | 0.503342 | 0.329847 | 0.363814 | 0.553342 | pass |

Relative to ordinary states, privileged fitting reduced lesion MAE by 62.21% to 73.58% across the six cases. Optimized lesion MAE was lower than matched R65A in every case, and nonlesion error also improved in every case. Gradient error improved from ordinary states but remained 0.007693 to 0.018405 above the matched reference. The scalar fitting gains therefore do not satisfy the declared detail-recovery endpoint.

The optimized stored states moved substantially. Relative L2 is displacement norm divided by the initial state's norm, not the final norm or a population-standardized distance:

| Seed | Example | L2 displacement | Relative L2 | RMS displacement | Maximum absolute change | Initial/final cosine | Stored coordinates changed |
|---|---|---:|---:|---:|---:|---:|---:|
| 17 | 1 | 99.123910 | 1.658853 | 0.547587 | 3.207458 | 0.341685 | 99.9908% |
| 17 | 2 | 110.818547 | 1.885774 | 0.612192 | 3.606079 | 0.321652 | 99.9847% |
| 23 | 1 | 108.303481 | 1.681739 | 0.598298 | 3.679199 | 0.329497 | 99.9878% |
| 23 | 2 | 117.843788 | 1.840956 | 0.651001 | 3.321106 | 0.314759 | 99.9939% |
| 31 | 1 | 119.174412 | 1.846105 | 0.658352 | 3.586792 | 0.304540 | 99.9969% |
| 31 | 2 | 121.630094 | 1.873900 | 0.671918 | 3.952515 | 0.323554 | 99.9969% |

The aggregate also records initial and final norms and per-channel displacement. Final master-to-payload quantization RMS ranged from 0.000117 to 0.000142. Displacement had no recovery threshold and did not invalidate any case. Its magnitude does not measure how difficult a different MRI encoder would find these states to infer.

All twelve fixed full-view and lesion-crop images were inspected. Optimized states recover stronger lesion appearance and some spatial structure, but fine texture and local differences remain blurred relative to targets and the direct reference. Predictions outside brain support are displayed but remain unscored. The images do not override the six failed gradient conditions.

Both two-update smoke cases passed and their states were discarded. Verification replayed all eighteen full prediction arrays exactly: six matched R65A, six ordinary, and six optimized predictions. Metric checks against the unchanged R65A scorer agreed within 5.55e-17. Independent displacement calculations agreed within 1.42e-14. The verifier confirmed each 512-step Adam checkpoint, Float32 master, zero weight decay, exact Float16 payload, all six condition decisions, and unchanged protected original artifacts. Raw predictions, states, checkpoints, and participant identities remain private on `mosaic-gpu`.

The bounded diagnostic is complete with partial voxelwise-error improvement and no case meeting the full recovery definition. No unsuccessful case received extra updates. Decoder limitations, the state format, the MAE-only fitting objective, and unsuccessful state optimization remain competing explanations for the residual gradient deficit. This result does not establish an information-theoretic limit, deployable prediction, or a replacement R66 outcome. R66 remains closed and segmentation remains deferred.

### R69 spatial-access readiness and launch, November 4, 2026

Both settings passed the original two-visit representation-readiness check at seed 17 after exactly 512 updates each. No case received extra training or a changed recipe. This is fitting readiness, not evidence for the cohort-level architectural claim. The [fixed protocol](METHODS.md#r69-end-to-end-spatial-access-comparison) and [preflight measurements](r69_preflight_metrics.json) record the full conditions and receipts.

| Spatial gate | Example | Initial whole-brain MAE | Final MAE | Half initial MAE | Mean-FLAIR control MAE | Readiness |
|---|---|---:|---:|---:|---:|---|
| Zeroed | 1 | 1.055761 | 0.384434 | 0.527881 | 0.440609 | pass |
| Zeroed | 2 | 1.120993 | 0.379776 | 0.560497 | 0.440071 | pass |
| Enabled | 1 | 1.167712 | 0.232749 | 0.583856 | 0.440609 | pass |
| Enabled | 2 | 1.249425 | 0.220682 | 0.624712 | 0.440071 | pass |

These measurements came from one run per setting on the NVIDIA GeForce RTX 5090 on `mosaic-gpu`. Times below cover synchronized training updates and checks, with arm time also including initialization, scoring, and saving. They are planning measurements, not a repeatability or speedup claim.

| Spatial gate | 512-update time | Total arm time | Peak CUDA allocated | Peak CUDA reserved |
|---|---:|---:|---:|---:|
| Zeroed | 89.42 s | 94.01 s | 5.477 GiB | 6.473 GiB |
| Enabled | 89.86 s | 95.91 s | 5.477 GiB | 6.473 GiB |

The complete preflight took 194.49 seconds. Two real smoke batches per setting passed exact gate-zero agreement with the original compact forward pass, matched initialization, finite core gradients, exactly zero disabled-projection gradients, nonzero enabled-projection gradients, and exact checkpoint, optimizer, prediction, and score reload. Four fixed readiness images were inspected. The enabled predictions show more visible structure on these fitting examples, but readiness images cannot establish lesion recovery or replace the fixed cohort assessment. No segmentation masks or scoring arrays were opened during preflight. Four synthetic assessment checks passed, including averaging seeds within patients before resampling.

The six-fit cohort matrix was launched as `world-model-spatial-access.service` on `mosaic-gpu`. The service was confirmed active and running, with seed 17, gate zero started from the matched fresh initialization and an empty optimizer. The launch record fixes both gates, seeds 17, 23, and 31, all 4,096 updates per fit, original training draws, independently selected checkpoints, and assessment source hashes. Readiness models and optimizers are not reused. The remaining fits are queued in the same service. Cohort fitting and assessment are pending; no primary architectural or absolute-reference result is available yet.

R66 and privileged state fitting remain closed. Raw images, participant measurements, and checkpoints stay private. The new experiment root is `/home/aditya/experiments/world-model-spatial-access-20261104`.

### R69 spatial access passes the joint architectural comparison

The November 4, 2026 frozen cohort assessment **passed all three conditions of the joint architectural claim**. Higher-resolution spatial access reduced lesion MAE, the unchanged lesion-band gradient error, and nonlesion MAE relative to the matched zeroed arm. This result concerns the whole end-to-end learning pathway under the fixed recipe, not isolated encoder information loss or a universal storage limit. It supersedes the pending cohort status in the readiness and launch entry above.

All six fits completed exactly 4,096 updates without extensions or recipe changes. Training finished successfully in 1 hour 24 minutes 27.869 seconds. All three seed pairs had matching initial-parameter and training-draw hashes. The assessment used 116 visits from eighteen patients; lesion and gradient metrics used 112 nonempty-lesion visits. It averaged visits within patients, then seeds within patients, before 20,000 paired patient-bootstrap resamples with seed 69017. All intervals below are paired two-sided percentile 95% intervals. These reused LUMIERE patients remain development evidence.

The [frozen assessment aggregate](r69_spatial_access_metrics.json) contains all component and seed intervals. The [fit-completion record](r69_fit_metrics.json) contains selected checkpoints and training receipts; its assessment-pending flag records the earlier fitting-completion stage, not the current status.

#### Joint architectural outcome

Differences are enabled minus zeroed; negative values favor spatial access.

| Measurement | Paired difference | Paired 95% interval | Required upper bound | Result |
|---|---:|---|---|---|
| Lesion MAE | -0.121718 | [-0.148967, -0.088537] | strictly below 0 | pass |
| Lesion-band gradient error | -0.022380 | [-0.027293, -0.017585] | strictly below 0 | pass |
| Nonlesion MAE | -0.075795 | [-0.086286, -0.065731] | at most +0.05 | pass |

The ratios of the patient-uniform aggregate means correspond to reductions of 16.09% in lesion MAE, 9.66% in gradient error, and 16.12% in nonlesion MAE. These percentages describe aggregate changes, not the average of per-patient percentage changes. The paired intervals favor the enabled arm for every component in every seed.

At the operator's November 4 closeout, R69 is recorded as **passed for the architectural comparison, with absolute fidelity unresolved**. Using point estimates, spatial access closed 74.46% of the lesion-error gap and 80.30% of the gradient-error gap from the zeroed arm to R65A. Gap closure is `(zeroed error - enabled error) / (zeroed error - matched R65A error)`: 0.121718 / 0.163473 for lesion MAE and 0.022380 / 0.027872 for gradient error, with ratios calculated from unrounded values. These are descriptive point-estimate ratios, not confidence bounds or an equivalence claim. The unresolved absolute-fidelity label retains the unresolved lesion tolerance and the measured gradient deficit; it does not relabel the failed no-worse gradient component. The bottleneck pathway is a demonstrated source of degradation under this recipe.

| Seed | Lesion-MAE difference and 95% interval | Gradient-error difference and 95% interval | Nonlesion-MAE difference and 95% interval |
|---|---|---|---|
| 17 | -0.104799 [-0.152378, -0.041430] | -0.020854 [-0.026082, -0.015226] | -0.074057 [-0.085388, -0.062990] |
| 23 | -0.095414 [-0.124973, -0.068644] | -0.023020 [-0.027660, -0.018697] | -0.086209 [-0.096596, -0.076259] |
| 31 | -0.164940 [-0.194524, -0.134688] | -0.023267 [-0.028354, -0.018420] | -0.067119 [-0.077637, -0.056624] |

Each arm selected its own checkpoint using the fixed patient-uniform selection whole-brain MAE, with the earliest minimum retained. Selection did not use lesion or gradient error.

| Seed | Zeroed selected update | Enabled selected update | Zeroed selection MAE | Enabled selection MAE |
|---|---:|---:|---:|---:|
| 17 | 4096 | 4096 | 0.471326 | 0.374885 |
| 23 | 3584 | 3584 | 0.478168 | 0.369668 |
| 31 | 4096 | 3584 | 0.464945 | 0.372269 |

#### Absolute fidelity against matched R65A

The architectural improvement did **not** establish restored reference-level fidelity. Patient-uniform, seed-averaged errors were:

| Model | Lesion MAE | Gradient error | Nonlesion MAE | Whole-brain MAE |
|---|---:|---:|---:|---:|
| Zeroed compact arm | 0.756410 | 0.231606 | 0.470229 | 0.484252 |
| Enabled spatial-access arm | 0.634692 | 0.209226 | 0.394434 | 0.406595 |
| Matched R65A | 0.592937 | 0.203734 | 0.394342 | 0.403502 |

Deficits below are arm minus matched R65A, paired within seed and patient before aggregation. Positive values indicate worse errors.

| Arm | Lesion deficit and 95% interval | +0.05 lesion tolerance | Gradient deficit and 95% interval | No-worse gradient condition |
|---|---|---|---|---|
| Zeroed | +0.163473 [+0.140211, +0.192237] | failed | +0.027872 [+0.023279, +0.032837] | failed |
| Enabled | +0.041755 [+0.012718, +0.074911] | unresolved | +0.005492 [+0.004219, +0.007061] | failed |

The enabled lesion point deficit is less than +0.05, but its interval extends beyond the tolerance. It is not a passed tolerance check. Its gradient error remains worse than R65A in all three seeds. Both arms therefore remain below the full reference-level fidelity criterion, despite the passed relative architectural claim. For nonlesion MAE, the enabled-minus-R65A difference is +0.000092 [-0.006202, +0.006334]; this interval does not establish exact equivalence. The zeroed nonlesion deficit is +0.075888 [+0.063278, +0.089753].

Absolute errors remain seed dependent:

| Seed | Model | Lesion MAE | Gradient error | Nonlesion MAE |
|---|---|---:|---:|---:|
| 17 | Zeroed | 0.795604 | 0.231572 | 0.468599 |
| 17 | Enabled | 0.690805 | 0.210718 | 0.394541 |
| 17 | R65A | 0.571430 | 0.202325 | 0.392440 |
| 23 | Zeroed | 0.733767 | 0.231555 | 0.477047 |
| 23 | Enabled | 0.638353 | 0.208536 | 0.390838 |
| 23 | R65A | 0.608538 | 0.203252 | 0.390975 |
| 31 | Zeroed | 0.739858 | 0.231691 | 0.465042 |
| 31 | Enabled | 0.574919 | 0.208424 | 0.397923 |
| 31 | R65A | 0.598843 | 0.205625 | 0.399610 |

| Seed | Arm | Lesion deficit against R65A and 95% interval | Gradient deficit against R65A and 95% interval |
|---|---|---|---|
| 17 | Zeroed | +0.224174 [+0.197754, +0.254715] | +0.029247 [+0.024340, +0.034533] |
| 17 | Enabled | +0.119375 [+0.072464, +0.179398] | +0.008393 [+0.005940, +0.011397] |
| 23 | Zeroed | +0.125229 [+0.091891, +0.161129] | +0.028304 [+0.023954, +0.033034] |
| 23 | Enabled | +0.029815 [-0.005670, +0.060980] | +0.005284 [+0.003773, +0.006818] |
| 31 | Zeroed | +0.141016 [+0.113349, +0.172272] | +0.026066 [+0.021193, +0.031169] |
| 31 | Enabled | -0.023924 [-0.052092, +0.003878] | +0.002799 [+0.001893, +0.003651] |

The enabled lesion-tolerance result is failed for seed 17, unresolved for seed 23, and passed for seed 31. Seed 31's negative lesion point deficit does not establish superiority to R65A because its interval includes zero. These differences must not be hidden by the average. All three zeroed seeds fail the lesion tolerance, and both arms fail the no-worse gradient condition in every seed.

#### Fixed images and bounded verification

All twelve prespecified whole-view and lesion-crop images were inspected: two visits, three seeds, and two views. Each image includes the fixed middle and maximum-lesion-area planes. Columns show target, matched R65A, zeroed prediction, enabled prediction, and the three corresponding absolute-error panels. Common intensity and error ranges remained fixed. The zeroed predictions are visibly smoothed and lose lesion-boundary and internal structure. Spatial access restores substantial lesion appearance and surrounding anatomy, while residual edge, texture, and lesion-intensity errors remain. The images support the measured relative gain without proving reference-level recovery.

Images and the fixed prediction arrays are preserved privately under `assessment-v1/private/` in the R69 GPU root. Per-visit and per-patient measurements are in `records.json`; seed-averaged patient vectors are in `seed-averaged-patients.json` in that directory. The report used anonymous local image copies rather than committing images or participant identities.

Verification stayed within the agreed scope. The one selected seed-17 zeroed checkpoint reload reproduced its fixed prediction within the declared 1e-12 absolute tolerance and reproduced the score exactly. Its lesion, whole-brain, and unchanged gradient metrics matched the canonical R65A scorer with maximum difference zero. The assessment confirmed the frozen source hashes, six 4,096-update fits, selected-checkpoint hashes, matched reference aggregate, scoring membership, and identical missingness across arms and seeds. No exhaustive prediction replay, extra training, new objective, or new architecture was run.

R69 is complete. The result supports spatial access as a useful intervention on the whole learning pathway under this recipe. It does not separate encoder gradients and learned state contents from decoder access. The enabled control receives more than a 64 KiB representation; this comparison neither establishes a universal storage limit nor proves that all missing FLAIR detail is inferable from the permitted MRI. R66 and the privileged diagnostic remain closed.

### R70 frozen-state readout readiness, November 4, 2026

Both frozen encoder-state conditions passed the fixed two-visit readiness check after exactly 512 fresh-decoder updates at seed 17. These are fitting-readiness measurements, not a result about recoverable spatial-access benefit. The [fixed protocol](METHODS.md#r70-frozen-state-compact-readout-comparison) and [preflight aggregate](r70_preflight_metrics.json) record the information and optimization boundaries.

The preflight exported two visits from each selected seed-17 R69 encoder. All four saved states occupied exactly 65,536 bytes and round-tripped exactly as Float16 tensors. Both encoders remained frozen and unchanged. Two real decoder-only smoke updates per condition passed matching fresh initialization, finite gradients, model and optimizer reload, and exact prediction and score reload. The readout processes opened no raw MRI arrays or segmentation masks; FLAIR and support came from separate target-only caches. Both smoke readouts and optimizers were discarded before readiness.

| State source | Example | Initial whole-brain MAE | Final MAE | Mean-FLAIR control | Readiness |
|---|---|---:|---:|---:|---|
| R69 zeroed encoder | 1 | 0.911177 | 0.370794 | 0.440609 | pass |
| R69 zeroed encoder | 2 | 0.947081 | 0.357412 | 0.440071 | pass |
| R69 enabled encoder | 1 | 0.924663 | 0.399762 | 0.440609 | pass |
| R69 enabled encoder | 2 | 0.958472 | 0.390878 | 0.440071 | pass |

Every final MAE was below half its own initial value and strictly below the corresponding mean-FLAIR control. Readiness used matched initialization and original visit draws, with no checkpoint selection or extra updates. Four fixed readiness images were inspected. Their remaining blur does not determine the cohort outcome or establish absence of detail in the states.

On the RTX 5090, 512 updates took 51.35 seconds for zeroed-encoder states and 51.45 seconds for enabled-encoder states. Peak CUDA allocated memory was 3.283 and 3.275 GiB, respectively; peak reserved memory was 4.791 and 4.789 GiB. Total smoke and readiness time was 107.04 seconds, excluding the separate four-state export. These are single-run planning measurements. The paired assessment fixtures passed on `mosaic-gpu`.

The `world-model-frozen-readout.service` sequence was launched on `mosaic-gpu` after both conditions passed. It was confirmed active and running in the full-state-export phase, with six fresh 4,096-update readouts and the frozen assessment queued afterward. No readiness weights or optimizer state will be reused. Cohort outcomes and absolute comparisons with original R69 enabled predictions and R65A remain pending. R69, R66, and privileged state fitting remain closed.

### R70 did not recover the spatial-access advantage from compact states

The frozen assessment completed on November 4, 2026. **R70 is closed as failed to recover the spatial-access advantage from compact states under the fixed readout recipe. The joint comparison failed all three components.** Readouts from enabled-encoder states were worse than readouts from zeroed-encoder states on lesion MAE, the unchanged gradient error, and nonlesion MAE. This is evidence about what this fixed readout and training budget recovered, not proof that the enabled states contain no useful detail. The compact-readout diagnostic sequence stops here.

All six readouts completed exactly 4,096 updates, with matching fresh decoder initialization and original draw hashes within each seed. No R69 decoder weights were reused, no gate was switched off as an inference shortcut, and no encoder was trained. All 3,594 visit states across six collections occupied exactly 65,536 bytes each, totaling 235,536,384 payload bytes. All six encoders remained unchanged. The full export, fitting, and assessment service exited successfully after 59 minutes 34.198 seconds. The [export record](r70_export_metrics.json), [fit record](r70_fit_metrics.json), and [assessment aggregate](r70_frozen_readout_metrics.json) retain the receipts.

Assessment used 116 scoring visits from eighteen patients, with 112 nonempty-lesion visits for lesion and gradient metrics. Visits were averaged within patients and seeds within patients before the fixed 20,000 paired patient-bootstrap resamples. Intervals below are paired two-sided 95% percentile intervals. The readiness ordering did not enter assessment or selection.

#### Joint state-readout comparison

Differences are enabled-encoder-state readout minus zeroed-encoder-state readout. Positive values favor the zeroed-state control.

| Measurement | Paired difference | Paired 95% interval | Required upper bound | Result |
|---|---:|---|---|---|
| Lesion MAE | +0.269215 | [+0.204435, +0.334377] | strictly below 0 | failed |
| Lesion-band gradient error | +0.010567 | [+0.008312, +0.012970] | strictly below 0 | failed |
| Nonlesion MAE | +0.059163 | [+0.052855, +0.065733] | at most +0.05 | failed |

The nonlesion lower bound exceeds the +0.05 margin, so that component failed rather than remaining unresolved. Every seed had higher lesion, gradient, and nonlesion errors from the enabled states. Seed 17's nonlesion margin is unresolved by itself; seeds 23 and 31 fail it. Every seed fails the joint comparison because its lesion and gradient intervals favor the zeroed-state control.

| Seed | Lesion difference and 95% interval | Gradient difference and 95% interval | Nonlesion difference and 95% interval |
|---|---|---|---|
| 17 | +0.215685 [+0.162001, +0.271363] | +0.009540 [+0.007880, +0.011213] | +0.048384 [+0.042961, +0.054963] |
| 23 | +0.319591 [+0.229623, +0.408042] | +0.010335 [+0.007454, +0.013338] | +0.067378 [+0.058440, +0.076652] |
| 31 | +0.272370 [+0.184377, +0.361137] | +0.011826 [+0.008742, +0.015100] | +0.061728 [+0.055127, +0.068707] |

Each readout selected its checkpoint independently by patient-uniform whole-brain MAE on the selection group. All fits still ran the full 4,096 updates.

| Seed | Zeroed-state readout selected update | Enabled-state readout selected update | Zeroed-state selection MAE | Enabled-state selection MAE |
|---|---:|---:|---:|---:|
| 17 | 4096 | 3584 | 0.457931 | 0.515818 |
| 23 | 3584 | 4096 | 0.456932 | 0.537971 |
| 31 | 4096 | 4096 | 0.456125 | 0.528629 |

#### Capability retained relative to the full predictors

Patient-uniform, seed-averaged errors were:

| Predictor | Lesion MAE | Gradient error | Nonlesion MAE | Whole-brain MAE |
|---|---:|---:|---:|---:|
| Fresh readout from zeroed-encoder states | 0.748122 | 0.230984 | 0.458436 | 0.472237 |
| Fresh readout from enabled-encoder states | 1.017337 | 0.241551 | 0.517600 | 0.541226 |
| Original full R69 enabled predictor | 0.634692 | 0.209226 | 0.394434 | 0.406595 |
| Matched R65A | 0.592937 | 0.203734 | 0.394342 | 0.403502 |

The fresh zeroed-state readout's lesion MAE, 0.748122, is approximately the original R69 zeroed model's 0.756410. These are descriptive patient-uniform, seed-averaged point estimates, not a formal equivalence result. The [combined interpretation](NOTES_AND_INSIGHTS.md#current-conclusion-through-r70-november-4-2026) distinguishes this control from the much worse enabled-state readout.

Both compact readouts remained worse than both reference predictors on all three measured error components. In particular, the enabled-state readout did not approach its original full R69 predictor: its paired lesion deficit was +0.382645 and its gradient deficit +0.032326. Retaining the spatial-access training architecture while exporting only this compact state did not preserve the demonstrated fidelity under the tested fresh-readout recipe.

Deficits are readout minus the matching seed's reference, averaged within patients before resampling:

| State source | Reference | Lesion deficit and 95% interval | Gradient deficit and 95% interval | Nonlesion deficit and 95% interval |
|---|---|---|---|---|
| Zeroed encoder | Full R69 enabled | +0.113430 [+0.082243, +0.141390] | +0.021759 [+0.017150, +0.026610] | +0.064002 [+0.055098, +0.073474] |
| Enabled encoder | Full R69 enabled | +0.382645 [+0.300924, +0.462935] | +0.032326 [+0.025943, +0.038788] | +0.123166 [+0.110514, +0.135942] |
| Zeroed encoder | R65A | +0.155185 [+0.131790, +0.186618] | +0.027250 [+0.022826, +0.032173] | +0.064095 [+0.052260, +0.077165] |
| Enabled encoder | R65A | +0.424400 [+0.351499, +0.495864] | +0.037817 [+0.031704, +0.044265] | +0.123258 [+0.107891, +0.139329] |

Both readouts fail the +0.05 lesion tolerance and no-worse gradient condition against both references at the seed-averaged level. No percentage of retained information is inferred from these error differences.

| Seed | State source | Lesion MAE | Gradient error | Nonlesion MAE |
|---|---|---:|---:|---:|
| 17 | Zeroed encoder | 0.774382 | 0.230721 | 0.458385 |
| 17 | Enabled encoder | 0.990067 | 0.240261 | 0.506768 |
| 23 | Zeroed encoder | 0.742856 | 0.231398 | 0.459457 |
| 23 | Enabled encoder | 1.062447 | 0.241734 | 0.526835 |
| 31 | Zeroed encoder | 0.727128 | 0.230834 | 0.457468 |
| 31 | Enabled encoder | 0.999498 | 0.242660 | 0.519197 |

Per-seed deficits against the full R69 enabled predictor were:

| Seed | State source | Lesion deficit and 95% interval | Gradient deficit and 95% interval | Nonlesion deficit and 95% interval |
|---|---|---|---|---|
| 17 | Zeroed encoder | +0.083577 [+0.021649, +0.129690] | +0.020004 [+0.014385, +0.025350] | +0.063844 [+0.052747, +0.075443] |
| 17 | Enabled encoder | +0.299263 [+0.225013, +0.370280] | +0.029543 [+0.023455, +0.035415] | +0.112227 [+0.099086, +0.125790] |
| 23 | Zeroed encoder | +0.104503 [+0.075176, +0.138991] | +0.022863 [+0.018629, +0.027608] | +0.068619 [+0.060730, +0.076932] |
| 23 | Enabled encoder | +0.424094 [+0.326969, +0.525822] | +0.033198 [+0.026532, +0.040336] | +0.135997 [+0.123154, +0.148648] |
| 31 | Zeroed encoder | +0.152209 [+0.125352, +0.179698] | +0.022410 [+0.018090, +0.027065] | +0.059545 [+0.049992, +0.069281] |
| 31 | Enabled encoder | +0.424579 [+0.331101, +0.520889] | +0.034236 [+0.027642, +0.040988] | +0.121273 [+0.108106, +0.134561] |

Per-seed deficits against R65A were:

| Seed | State source | Lesion deficit and 95% interval | Gradient deficit and 95% interval | Nonlesion deficit and 95% interval |
|---|---|---|---|---|
| 17 | Zeroed encoder | +0.202952 [+0.172799, +0.238371] | +0.028396 [+0.023439, +0.033936] | +0.065945 [+0.052421, +0.080746] |
| 17 | Enabled encoder | +0.418637 [+0.364679, +0.468436] | +0.037936 [+0.032156, +0.044042] | +0.114328 [+0.099069, +0.130145] |
| 23 | Zeroed encoder | +0.134318 [+0.099252, +0.175323] | +0.028147 [+0.023898, +0.032954] | +0.068481 [+0.056963, +0.081202] |
| 23 | Enabled encoder | +0.453909 [+0.358584, +0.549260] | +0.038482 [+0.032198, +0.045389] | +0.135859 [+0.119983, +0.152488] |
| 31 | Zeroed encoder | +0.128285 [+0.108221, +0.153291] | +0.025209 [+0.020877, +0.029831] | +0.057858 [+0.046478, +0.070118] |
| 31 | Enabled encoder | +0.400655 [+0.306642, +0.496021] | +0.037035 [+0.030383, +0.043744] | +0.119587 [+0.103435, +0.136335] |

#### Fixed images, verification, and interpretation

All twelve fixed whole-view and lesion-crop images were inspected. Both fresh compact readouts remain heavily smoothed compared with the full R69 enabled predictor and R65A. The enabled-state readouts also show conspicuous errors in lesion appearance and internal brain structure across the fixed seeds and visits. Their residual-error panels remain substantially brighter around lesion structure than the full spatial-access predictor's panels. Images use the prespecified visits, planes, crops, and common ranges and did not select the result.

Private per-visit and per-patient measurements, seed-averaged patient vectors, comparison predictions, and images remain in `assessment-v1/private/` under `/home/aditya/experiments/world-model-frozen-readout-20261104`. The reference predictions were reused from R69's fixed saved arrays, not regenerated. Anonymous local image copies support the report; no raw images or participant identities were committed.

Verification stayed at the agreed single bounded selected-readout reload and score check. It passed, and the canonical R65A lesion, whole-brain, and gradient scoring comparison had maximum difference zero. The frozen assessment validated source hashes, export-manifest and scoring-payload hashes, the exact state size, selected checkpoints, matched reference records and image hashes, cohort membership, and matching metric exclusions. There was no exhaustive replay, new fitting, or hyperparameter adjustment.

The full spatial-access predictor's benefit remains dependent on its additional features under this compact-readout recipe. This result does not prove that the compact state is empty, that another decoder or optimization procedure could recover no detail, or that a universal storage limit has been reached. A stronger representation-learning objective or a different readout would be a new experiment, not an extension needed to complete R70. R70 is complete with a failed recoverable-benefit claim; R69 remains passed for the architectural comparison with absolute fidelity unresolved. R66 and privileged state fitting stay closed. No additional run is scheduled.

## Complete-state one-visit footprint, November 4, 2026

This sizing probe followed R70 closure. It is not a segmentation result or a reopened compact-readout diagnostic. It used the first established fitting-readiness visit and the selected seed-17 R69 enabled checkpoint at update 4,096. No segmentation mask, other visit image, or reserved MU image was opened. The [public measurement record](complete_state_footprint_metrics.json) retains exact bytes, timing samples, hashes, and scope.

The export contains every encoder tensor consumed by the enabled decoder, before its learned spatial projections. All tensors retain their native precision. Shapes below omit the singleton batch dimension.

| Tensor | Channels and spatial shape | Precision | Payload bytes | MiB |
|---|---|---|---:|---:|
| Compact state | 8 × 16 × 16 × 16 | Float16 | 65,536 | 0.0625 |
| Full-resolution feature | 8 × 192 × 224 × 192 | Float32 | 264,241,152 | 252 |
| Second feature | 16 × 96 × 112 × 96 | Float32 | 66,060,288 | 63 |
| Third feature | 32 × 48 × 56 × 48 | Float32 | 16,515,072 | 15.75 |
| Deepest feature | 64 × 24 × 28 × 24 | Float32 | 4,128,768 | 3.9375 |
| **Total** | | | **351,010,816** | **334.75** |

The serialized PyTorch file is 351,013,337 bytes, including 2,521 bytes of overhead. The three input sequences occupy 86,652,384 bytes as the model's Float32 tensor, or 82.638 MiB. Their stored Float16 payload is 43,326,192 bytes, or 41.319 MiB. The complete state is therefore **4.05 times the model input and 8.10 times the stored input payload**. The prepared four-channel source file also contains unused target FLAIR and is not the three-input denominator. These are uncompressed tensor-payload comparisons, not compressed NIfTI comparisons. Shared model weights are not visit payload.

On `mosaic-gpu`, an RTX 5090, eager CPU loading and blocking GPU transfer gave the following ballpark costs. Each row summarizes five warm page-cache repeats after the first postwrite load. All six loads and decodes completed successfully and validated exactly. CPU allocation and tensor materialization occur inside loading. CUDA work is synchronized. Equality checks are outside the timings.

| Stage | Median milliseconds | Range milliseconds |
|---|---:|---:|
| Load complete state to CPU | 66.48 | 61.42–72.10 |
| Transfer complete state to GPU | 18.76 | 18.17–19.05 |
| Existing FLAIR decoder | 38.61 | 38.56–38.66 |
| Sum of load, transfer, and decode | 124.14 | 118.69–129.80 |

The first postwrite load took 117.49 ms. One device-to-CPU copy took 163.73 ms, and one save with filesystem synchronization took 360.86 ms. No cache eviction, loader tuning, prefetch, or disk-throughput profiling was performed. Warm CPU loading was the largest measured stage, but its underlying limiter was not isolated. These one-visit figures do not predict cold cohort loading or segmentation training throughput. Before the probe, GPU utilization was zero and host load averages were 1.15, 1.24, and 1.54 across 32 logical CPUs.

Every reloaded tensor was bitwise identical. The existing FLAIR decoder reproduced both the mean and Laplace log-scale with maximum absolute difference zero. Model weights and the selected checkpoint remained unchanged. The raw receipt mistakenly names the second output `logvariance`; the public aggregate corrects that label to `logscale` without changing the measurement.

At this fixed shape, one encoder's 599 visits would require 195.816 GiB of tensor payload. Three learned and three random collections would require 1,174.894 GiB, about 1.15 TiB, before targets and checkpoints. Only 589.648 GiB was available before the probe. The [proposal](METHODS.md#complete-state-reuse-proposal-november-4-2026) therefore stages exports rather than retaining all six collections simultaneously. This is a reference for reuse, not an efficient representation.

The one-visit payload and raw receipts remain at `/home/aditya/experiments/world-model-complete-state-footprint-20261104` on `mosaic-gpu`. The probe source is `scratch/complete-state-reuse/footprint.py`. At this probe's close, no cohort export or segmentation fitting had launched. The operator subsequently approved the separate R71 study below.

## R71 complete-state reuse readiness, November 4, 2026

**The three fixed direct-reference readiness checks passed.** This permits the approved nine-fit cohort comparison; it is not yet a reuse result. The representation remains **334.75 MiB per visit plus serialization overhead**, 4.05 times the Float32 input and 8.10 times its stored Float16 payload. Neither readiness nor a later reuse pass establishes efficient compression or independent clinical validity.

The [preflight record](r71_preflight_metrics.json) retains smoke results, per-visit and compartment scores, timings, memory, and source hashes. Smoke ran two real updates in each of the learned-state, random-state, and direct-image arms. All three readouts had matching initial parameter hashes. Saved-state and checkpoint reloads passed, with zero prediction discrepancy and matching canonical scores. Native precision and the class-balanced segmentation objective were unchanged.

Each direct-reference check then started afresh and ran exactly 512 updates on the same two established fitting visits. The fixed criterion is the mean of the two per-visit nonempty-compartment Dice means, at least 0.80.

| Seed | First visit | Second visit | Two-visit mean | Result |
|---|---:|---:|---:|---|
| 17 | 0.870763 | 0.874171 | 0.872467 | passed |
| 23 | 0.898507 | 0.910978 | 0.904742 | passed |
| 31 | 0.781987 | 0.876446 | 0.829217 | passed |

Seed 31's first visit is below 0.80. That does not fail the prespecified two-visit mean criterion, but the pass must not be reported as every visit or every compartment exceeding 0.80. The necrotic/nonenhancing compartment on that visit has Dice 0.682785. All six fixed images were inspected. Visible overprediction remains, especially for the edema compartment in seeds 17 and 31. The full images also show outside-support foreground predictions, which the unchanged scoring policy excludes. No threshold, budget, objective, or seed was changed in response.

The three readiness fits took 93.16, 92.96, and 93.17 seconds, including the final evaluations and images. Peak allocated memory reached 6,066,526,208 bytes, or 5.650 GiB. The second smoke update took 0.1032 seconds for learned-state readout, 0.1036 seconds for random-state readout, and 0.1714 seconds for the direct-image model. The first learned-state update included initialization overhead and took 0.2419 seconds. These update timings exclude loading inputs and labels. They are not cohort throughput measurements.

Before cohort dispatch, the operator was given a rough four-to-six-hour planning estimate. Multiplying warmed smoke update costs gives approximately 77 minutes for the nine fits' updates alone. Applying the earlier warm one-visit state load and transfer costs to six frozen-readout training and selection schedules adds about 47 minutes. The one-visit forward, CPU copy, and synchronized-save costs give a separate rough 36-minute export allowance for six full collections. These surrogates omit or approximate label loading, hashing, cold I/O, selection decoding and scoring, prediction retention, and assessment. The service must retain the fixed update budget rather than adjust it to this estimate.

Eight instrument tests passed on `mosaic-gpu`, covering aggregation, paired intervals, acceptance boundaries, rendering, empty-region scoring, and explicit cache-retirement paths. The initial renderer import failed because this GPU environment lacks Matplotlib. Before source freezing or fitting, the renderer was replaced with Pillow, already used by the earlier studies. No dependency was installed. During readiness, the SSH waiter timed out, but systemd continued the original run successfully. The no-timeout waiter collected its completion; no fit was interrupted or extended.

The fixed matrix was launched after readiness and image inspection. It stages seed 17, then 23, then 31, finishing each seed's three fits before retiring its new fitting/selection exports. All nine selected models must be locked before scoring. Scoring retains predictions, measurements, fixed images, analysis inputs, and fixed example states before retiring that seed's scoring exports. The private run root is `/home/aditya/experiments/world-model-complete-reuse-20261104`; the active service is `world-model-complete-reuse.service`. There is no additional review gate. Cohort results remain pending.

### R71 retained learned segmentation signal but failed the direct-image retention margin

The fixed R71 cohort comparison completed on November 4, 2026. **R71 is closed as learned transfer demonstrated, with the required direct-image retention failed.** The joint claim failed. The complete frozen learned state improved segmentation Dice over matched frozen random features, but its deficit against the matched direct-image model exceeded the allowed 0.05 Dice. This is positive evidence of a separately decoded anatomical signal alongside retained FLAIR prediction, not a pass of the prespecified reference-level reuse claim.

The representation costs **334.75 MiB per visit plus serialization overhead**, 4.05 times the Float32 model input and 8.10 times the stored Float16 input payload. The study did not test efficient compression, biological-state recovery, or independent clinical validity. The [assessment aggregate](r71_complete_reuse_metrics.json), [nine-fit record](r71_fit_metrics.json), and [retention receipt](r71_retention_metrics.json) retain the measurements and execution evidence.

All nine cohort fits completed exactly 4,096 updates. Each seed's three arms shared fresh readout initialization and training draws. Selection remained patient-uniform primary Dice, evaluated every 512 updates, with the earliest maximum selected. No original FLAIR decoder weights initialized segmentation. The direct-image control adapted a separate copy of the selected learned encoder. All six exported learned/random encoders remained unchanged, and fitting/scoring export identities matched. The cohort export, fitting, and assessment service exited successfully after 4 hours 17 minutes 13.761 seconds. No fit extension, threshold change, seed replacement, or extra review round occurred.

#### Fixed joint assessment

All 116 scoring visits from eighteen patients entered segmentation assessment. Scores average compartments within visits, visits within patients, and seeds within patients. Both contrasts use the same 20,000 paired patient-bootstrap draws and two-sided 95 percent percentile intervals. Higher Dice is better.

| Arm | Patient-uniform, seed-averaged Dice |
|---|---:|
| Frozen learned complete state | 0.482944 |
| Frozen random complete state | 0.416158 |
| Adapted direct images | 0.562096 |

| Contrast | Difference | Paired 95% interval | Required lower bound | Result |
|---|---:|---|---|---|
| Learned minus random | +0.066786 | [+0.051279, +0.082710] | strictly above 0 | passed |
| Learned minus direct | -0.079152 | [-0.093428, -0.065874] | at least -0.05 | failed |

The entire direct-image contrast interval is below -0.05. This establishes failure under the fixed recipe, rather than an unresolved margin. The random-feature advantage is present in every seed.

| Seed | Learned Dice | Random Dice | Direct Dice | Learned minus random and 95% interval | Learned minus direct and 95% interval |
|---|---:|---:|---:|---|---|
| 17 | 0.476191 | 0.412190 | 0.580173 | +0.064000 [+0.043144, +0.087051] | -0.103982 [-0.124764, -0.085280] |
| 23 | 0.443445 | 0.380535 | 0.543092 | +0.062911 [+0.037165, +0.090356] | -0.099646 [-0.119187, -0.079387] |
| 31 | 0.529196 | 0.455748 | 0.563022 | +0.073448 [+0.054889, +0.093062] | -0.033826 [-0.050893, -0.017487] |

Seeds 17 and 23 individually fail the direct margin. Seed 31's direct margin is unresolved because its lower bound is just below -0.05. Its favorable point estimate does not change the failed seed-averaged joint result.

| Seed | Learned selected update | Random selected update | Direct selected update |
|---|---:|---:|---:|
| 17 | 4096 | 4096 | 2560 |
| 23 | 4096 | 4096 | 3584 |
| 31 | 3584 | 4096 | 4096 |

The frozen-state readouts each had 338,340 trainable parameters. Direct models had 559,684, including the adapted encoder. The readout architecture and its initialization were matched, not the number of trainable encoder parameters. That distinction is the intended frozen-versus-adapted control.

#### Compartments and false positives

These compartment results are descriptive companions, not additional acceptance tests. The largest learned-versus-direct Dice deficit is in the enhancing compartment. The edema difference remains within the numerical 0.05 margin at the aggregate level, but does not rescue the joint claim.

| Compartment | Learned Dice | Random Dice | Direct Dice | Learned minus random and 95% interval | Learned minus direct and 95% interval |
|---|---:|---:|---:|---|---|
| 1, enhancing | 0.408241 | 0.378830 | 0.551570 | +0.029411 [+0.011560, +0.045203] | -0.143329 [-0.160065, -0.126595] |
| 2, necrotic/nonenhancing | 0.422423 | 0.360587 | 0.483946 | +0.061835 [+0.041474, +0.082763] | -0.061524 [-0.088524, -0.036762] |
| 3, edema | 0.618168 | 0.509055 | 0.650770 | +0.109113 [+0.086501, +0.132986] | -0.032602 [-0.048820, -0.014118] |

Excluding reference-empty visits separately for each compartment gives:

| Compartment | Learned nonempty Dice | Random nonempty Dice | Direct nonempty Dice |
|---|---:|---:|---:|
| 1 | 0.453615 | 0.418002 | 0.615309 |
| 2 | 0.431753 | 0.367276 | 0.494036 |
| 3 | 0.625891 | 0.514955 | 0.658986 |

All-case supported false-positive volume, in mm³, averages visits within patients and then patients and seeds equally:

| Compartment | Learned | Random | Direct |
|---|---:|---:|---:|
| 1 | 6,308.14 | 8,583.39 | 3,738.83 |
| 2 | 14,857.88 | 16,027.90 | 13,719.47 |
| 3 | 35,161.16 | 43,101.23 | 29,801.71 |

Every arm and seed predicted some supported foreground in every reference-empty compartment case. Empty-reference false-positive frequency was therefore 1.0 throughout. The unchanged policy gives Dice zero when exactly one of reference and prediction is empty. Empty-reference volume means below include only patients with an eligible empty-reference visit.

| Compartment | Empty visits | Eligible patients | Learned false-positive mm³ | Random false-positive mm³ | Direct false-positive mm³ |
|---|---:|---:|---:|---:|---:|
| 1 | 17 | 8 | 4,659.43 | 7,759.47 | 2,295.33 |
| 2 | 6 | 2 | 3,819.23 | 7,729.23 | 8,775.13 |
| 3 | 4 | 1 | 39,765.08 | 58,041.42 | 33,417.00 |

Four visits had no foreground compartment. Patient-uniform empty-reference fractions were 0.109259, 0.029321, and 0.018519 for labels 1, 2, and 3. The patient-uniform all-foreground-empty fraction was 0.018519. These differ from raw visit fractions because patients receive equal weight. The label-three empty-reference companion contains only one patient, and label two only two. Their bootstrap intervals in the aggregate are descriptive resampling outputs, not reliable population uncertainty; in the one-patient case they necessarily have zero width. Per-seed compartment values and all companion contrasts remain in the aggregate.

#### Retained FLAIR, fixed images, and nuisance companion

Both applications consumed the same saved learned-state files. FLAIR performance matched the original R69 enabled aggregate: lesion MAE 0.634692, gradient error 0.209226, nonlesion MAE 0.394434, and whole-brain MAE 0.406595. The bounded original-model versus saved-state check in each seed found zero difference in both mean and Laplace log-scale. This preserves the established first application; it does not resolve R69's fidelity deficit against R65A.

All twelve fixed full-view and lesion-crop images were inspected. The learned-state segmentation tracks broad lesion regions but shows compartment boundary errors and overprediction. The direct models generally recover the enhancing-compartment pattern better in these fixed views. Random-feature readouts show less accurate lesion extent and internal compartment shape. Original FLAIR predictions remain smoother than targets, with residual edges visible. The full views include unscored outside-support predictions, particularly conspicuous in seed 23; those pixels must not be read as the supported false-positive volumes reported above. Planes, examples, crops, and ranges were fixed by the protocol, not selected by prediction quality.

The fixed 1.05 input-intensity multiplier on the two fitting-readiness visits produced small descriptive Dice changes. Across the six seed/visit pairs, learned-state mean change was -0.001299, range [-0.006197, +0.000446]; random-state mean change was -0.000033, range [-0.000163, +0.000118]; direct mean change was +0.000013, range [-0.001410, +0.001481]. These two previously fitted visits and one intensity perturbation do not establish acquisition robustness. No model or selection rule adapted to the nuisance results.

#### Staging, retained evidence, and closure

All nine selected models were locked before scoring. The three fitting/selection stages each exported 966 states, 315.790 GiB of payload, and the three scoring stages each exported 232 states, 75.842 GiB. Across stages, 3,594 cohort states carried 1,261,532,872,704 payload bytes and 1,261,543,412,076 serialized bytes. Thus the study generated about 1.15 TiB sequentially, not simultaneously. The complete per-visit payload stayed fixed; no feature truncation or precision reduction was used.

Each seed's three fits, selected checkpoints, selection curves, encoder identities, export manifests, and fixed example states were retained before retiring that seed's fitting/selection caches. Scoring retained all 348 seed/visit prediction files, twelve comparison images, per-visit measurements, patient vectors, and paired resampling inputs before retiring each scoring cache. Twenty-four fitting/scoring example states remain, in addition to readiness and nuisance artifacts. Only the explicit new export files were retired. Source images, previous experiments, checkpoints, and results were not deleted.

The agreed bounded selected-readout reload and independent saved-prediction score checks passed in every seed. There was no exhaustive GPU replay. A final metadata check confirmed that the nine-model lock still matches, retained prediction/image/fixed-state files exist at their recorded sizes, all six cache-retirement receipts match, and retired exports are absent. Large prediction/state files were not rehashed in that final check; the assessment runner verified retained result hashes before retirement. The [retention receipt](r71_retention_metrics.json) states this scope explicitly.

Private artifacts remain under `/home/aditya/experiments/world-model-complete-reuse-20261104`. Anonymous local figure copies support the report. No participant inventory, image, prediction array, or model weight was committed.

R71 is closed as **learned transfer demonstrated, with the required direct-image retention failed**. The representation-study sequence stops here. R66, R69, R70, and the privileged diagnostic remain closed. No extension or compression run is scheduled. The operator selected preparation of the independent R67 history-information comparison next, not another attempt to make the present compact representation pass.

## R67 metadata preparation for current-FLAIR history comparison, November 4, 2026

**Superseded preparation.** This initial unbounded donor map was not used for registration or fitting. The subsequent [eight-week caliper inventory](#r67-eight-week-donor-caliper-before-image-processing) replaces its eligibility counts and launch map. The original aggregate and private files remain unchanged.

After closing R71 and the representation-study sequence, the operator selected preparation of the independent history-information question. The [metadata inventory](r67_preparation_metrics.json) used existing cohort JSON and prepared-file existence checks only. It opened no MRI arrays, segmentation measurements, or model weights. Registration and fitting have not run.

The candidate comparison fixes current panel C and one preceding eligible visit. Genuine and donor history may contain previous FLAIR. All three trained conditions and both nontrained references share the same target-visit set. The proposed donor rule uses only fitting patients, requires an exact nominal week-gap match and complete prior sequence panel, then chooses the nearest donor-anchor week with deterministic hash tie-breaking. The [protocol](METHODS.md#r67-panel-c-bounded-preparation-november-4-2026) owns the rule and its limitations.

| Preparation stage | Fitting | Selection | Scoring |
|---|---:|---:|---:|
| Original eligible patients | 55 | 18 | 18 |
| Original eligible visits | 305 | 178 | 116 |
| First eligible visit excluded for lacking a predecessor | 55 | 18 | 18 |
| History-eligible patients | 52 | 17 | 18 |
| History-eligible target visits | 250 | 160 | 98 |
| Targets excluded for no exact-gap fitting donor | 13 | 2 | 4 |
| **Common patients after metadata rules** | **52** | **17** | **18** |
| **Common target visits after metadata rules** | **237** | **158** | **94** |

Three fitting patients and one selection patient had only one eligible visit. No scoring patient was lost through lack of history. Donor availability removed target visits but no additional patients. The original R65A inventory had already excluded 39 incomplete visit directories: 26 fitting, seven selection, and six scoring. These are upstream exclusions, not additional removals from the 599-visit starting catalogue above.

Retained same-week pairs number 27, six, and six by group. Suffixes establish their order but do not imply positive elapsed days. Nominal-gap ranges and medians are 0–42 weeks with median eleven for fitting, 0–25 with median eleven for selection, and 0–30 with median thirteen for scoring. Three, three, and four retained pairs respectively skip an incomplete raw visit, because the selected history is the immediately preceding eligible visit rather than the immediately preceding incomplete acquisition.

The four scoring targets without an exact-gap fitting donor had nominal gaps of 23, 47, 49, and 87 weeks. Selection excluded gaps of 23 and 50 weeks. The thirteen fitting exclusions and the complete retained gap distributions are in the public aggregate. These exclusions restrict what a later result can say about sparse or very long histories.

The fixed map uses 178 distinct donor prior visits in fitting, 92 in selection, and 72 in scoring. Maximum reuse of one donor prior visit within a group is four, eight, and four targets. Donor-anchor nominal-week differences have medians two, three, and 3.5 weeks, but maxima 141, 129, and 116 weeks. Thus exact matching of the previous-to-current nominal gap does not mean tight matching of the observation's nominal position in follow-up, clinical phase, or lesion burden. This limitation is retained rather than hidden by calling the donors fully matched.

The common 489 target visits imply 978 distinct true/donor-to-current registration jobs. **These are metadata-defined counts, not final post-registration eligibility.** A failed true or donor registration must exclude the target from every arm and reference before fitting. No donor rematching or target-dependent crop selection is allowed.

Historical R65A logs provide nine completed-fit timings totaling 2,916.36 seconds including selection. That 48.61-minute historical observation supports the proposed 1.5–2.5-hour planning allowance for the new nine fits, not a measured R67 throughput claim. Registration runtime and the new model's real-data smoke remain unmeasured. No training service was dispatched. The private map and exclusions remain in `/home/aditya/experiments/world-model-r67-preparation-20261104`; their source and map hashes are recorded in the public aggregate.

## R67 eight-week donor caliper before image processing

On November 4, 2026, the operator required donor anchors within inclusive ±8 nominal weeks while retaining identical gaps, historical sequence availability, different fitting-patient identity, and deterministic tie-breaking. The [revised inventory](r67_caliper8_preparation_metrics.json) read metadata and checked file existence only. It opened no MRI arrays or segmentation measurements. This fixed tolerance does not establish equivalent clinical phase.

| Metadata stage | Fitting | Selection | Scoring |
|---|---:|---:|---:|
| History-eligible targets | 250 | 160 | 98 |
| No exact-gap fitting donor | 13 | 2 | 4 |
| Additional eight-week caliper exclusions | 62 | 53 | 30 |
| Common target visits | 175 | 105 | 64 |
| Common patients | 51 | 17 | 18 |
| Zero-week targets | 25 | 6 | 6 |
| 1–8-week targets | 53 | 34 | 19 |
| Longer-than-eight-week targets | 97 | 65 | 39 |

The caliper removes one additional fitting patient and no additional selection or scoring patients. Retained donor-anchor differences have medians one, one, and 0.5 weeks, with maxima eight in every group. The maximum donor reuse counts are four, four, and three. Retained pairs skipping an incomplete visit number one, one, and two. All 344 targets require genuine and donor registrations, totaling 688 distinct jobs. These counts are provisional until observed-input preparation completes.

The private revised map is `/home/aditya/experiments/world-model-r67-caliper8-20261104/private-pairs.json`, SHA256 `3829fb605a73e923b66b46c742c581c332a6bc5295accdec0615441c003d2ccc`. The earlier preparation remains in its original directory. The [amended protocol](METHODS.md#r67-panel-c-bounded-preparation-november-4-2026) fixes equal recipient timing inputs, exactly five joint acceptance conditions, unchanged fitting and selection, and persistence on identical targets. The representation-study sequence remains closed.

## R67 fixed CT1 registration smoke and instrument correction

The [registration smoke aggregate](r67_registration_smoke_metrics.json) records four registrations for two fixed fitting targets, each with genuine and donor history. Registration estimated transforms from CT1 only. Current FLAIR and segmentation were not used. Historical FLAIR was resampled with the same transform as the other three prior sequences.

The first geometry assertion incorrectly required bitwise identity after physical-space interpolation. Array roundtrips were exact, but interpolation produced residuals around zero up to 8.14×10⁻¹⁴ in the diagnostic. No voxel differed by more than 0.000001. The assertion was corrected to that numerical tolerance, with the failed receipts retained.

The next smoke passed finite and geometry checks but failed visual inspection. One genuine-history alignment visibly displaced the ventricles and increased CT1 absolute error from 0.819261 to 0.901167. An optimizer trace on that same fixed pair showed large updates under the Once learning-rate estimate and deterioration after better mutual-information values. Before processing any other pairs, the recipe changed to EachIteration learning-rate estimation with a one-millimeter maximum physical-step parameter. An unsupported best-parameter API option was removed before the corrected run; no output from that failed invocation was used. The registration objective, rigid transform, sampling, scales, iterations, and common exclusion policy did not change.

| Fixed fitting target | History | CT1 MAE before | CT1 MAE after | Current support covered |
|---|---|---:|---:|---:|
| First | Genuine | 0.819261 | 0.551817 | 98.98% |
| First | Donor | 0.892650 | 0.766371 | 84.37% |
| Second | Genuine | 0.800243 | 0.474701 | 97.32% |
| Second | Donor | 0.905890 | 0.873102 | 75.25% |

All four corrected CT1 overlays were inspected. The genuine-history images showed improved ventricular and outer-brain alignment. Donor alignment improved gross pose but retained substantial cross-patient anatomical and support differences. These observations do not establish expert-verified registration or equivalent clinical phase. No CT1-error threshold was added to cohort exclusion after viewing the images.

Independent physical-coordinate trilinear calculations checked 257 supported points in every pair and all four prior sequences. Maximum absolute disagreement was 2.01×10⁻⁷, below the fixed 0.0001 tolerance. Maximum identity-interpolation error across the corrected smoke was 1.27×10⁻¹³. Four jobs including checks and rendering took 24.08 seconds. This is a smoke timing, not a measured full-cohort runtime.

Settings and the common failure policy were frozen at 17:18:39 UTC on November 4, before dispatching the remaining 684 jobs with four independent one-thread workers. Failed geometry, nonfinite inputs/transforms/outputs, exceptions, or no supported overlap exclude the target from every arm and reference. Finite iteration-limit termination alone does not fail. There is no identity fallback or donor rematching. The original unsuccessful smoke outputs are retained separately. Training remains gated on final common eligibility and real-data smoke/readiness.

## R67 cohort registration completed without common exclusions

The [registration aggregate](r67_registration_metrics.json) records all 688 genuine and donor registrations. The remaining 684 jobs completed in 999.66 seconds, or 16 minutes 40 seconds, with four workers. No registration met the frozen failure criteria, so no target was removed. The final common populations remain 51 patients/175 targets fitting, 17/105 selection, and 18/64 scoring. The final private manifest SHA256 is `66a1b5c5ed98d41e50242ad99edfd0209a730837abadb4f3587de606c0d572b3`.

Registration wall time per job had median 5.26 seconds and range 4.65–12.56 seconds under concurrent execution. Transformed CT1 support covered a median 96.55 percent of current support, with a range of 48.69–99.95 percent. A finite registration with partial coverage was not excluded under the frozen rule. Scoring will use full current CT1 support rather than hide uncovered regions. These finite/overlap checks do not establish that every alignment is anatomically optimal.

Manifest validation confirmed patient separation, fitting-only donor identities, exact nominal gaps, the inclusive eight-week anchor caliper, and the common registered paths. No donor was rematched. The real-data training smoke and unchanged readiness checks were dispatched after final eligibility was fixed; cohort fitting remains conditional on their pass.

## R67 real-data smoke and fixed readiness passed

The [training smoke](r67_training_smoke_metrics.json) passed two real batches for each seed-17 condition, followed by model and optimizer reload and scoring. All conditions had identical initial parameter hashes. Current FLAIR perturbation did not change inference inputs; current-only ignored historical paths; historical channels and common recipient timing matched their definitions. Genuine-history FLAIR channels matched the registered image used for persistence. All intended parameter tensors changed and gradients remained finite.

Three fresh seed-17 models then completed exactly 512 updates each on the same two CT1-defined fitting planes. The [readiness receipt](r67_readiness_metrics.json) records a pass for each plane in each condition. Every final error was below half its own initialization and below its own error against the two-plane mean. No extension or recipe change was used.

| Condition | Initial MAE, first plane | Final MAE, first plane | Initial MAE, second plane | Final MAE, second plane |
|---|---:|---:|---:|---:|
| Current-only | 0.656137 | 0.099303 | 0.737035 | 0.038818 |
| Genuine history | 0.619727 | 0.098313 | 0.693824 | 0.042885 |
| Donor history | 0.637772 | 0.101528 | 0.720176 | 0.042255 |

The mean reference errors were 0.379956 and 0.398242 for the two planes. Saved before/after predictions, targets, support, and errors were rendered and inspected for all three conditions. These are fitting-readiness examples, not evidence of generalization or historical benefit. Final readiness weights and optimizer states remain private and do not initialize the cohort fits.

The three readiness runs took 15.23, 17.13, and 17.13 seconds of fitting respectively. A full donor-arm selection pass over all 105 common selection visits took 34.00 seconds. Warm smoke batches took about 0.028–0.029 seconds including loading and update; first-use setup was slower. These measurements suggest about 1–1.5 hours for nine fits including selection, with the original 1.5–2.5-hour allowance retained for slower random cohort loading. This is a planning extrapolation, not a completed matrix runtime.

After readiness passed and its images were inspected, the nine fits were dispatched with unchanged objective, 4,096 updates each, matched initialization/draws, and selection every 512 updates. Assessment is chained after the nine-model lock. No scoring outcome has informed registration, eligibility, fitting, or selection.

## R67 failed the joint patient-history benefit claim

The fixed nine-fit panel-C comparison completed on November 5, 2026 at 00:09:50 on the host's UTC+05:30 clock. All fits completed 4,096 updates unchanged, and all selected models were locked before the frozen assessment. The [assessment aggregate](r67_history_metrics.json), [fit lock](r67_fit_metrics.json), and [bounded verification](r67_verification_metrics.json) preserve the result. **The joint history-benefit claim failed.** Genuine history did not establish lower lesion error than either trained control and had higher gradient error than both. Nonlesion retention passed.

### Final common population and exclusions

| Stage | Fitting | Selection | Scoring |
|---|---:|---:|---:|
| Original complete eligible visits | 305 | 178 | 116 |
| First visits excluded for lacking history | 55 | 18 | 18 |
| No exact-gap fitting donor | 13 | 2 | 4 |
| Additional eight-week anchor-caliper exclusions | 62 | 53 | 30 |
| Registration exclusions | 0 | 0 | 0 |
| **Final common targets** | **175** | **105** | **64** |
| **Final common patients** | **51** | **17** | **18** |

The original preparation had already excluded 26, seven, and six incomplete visit directories upstream. These 39 directories are not additional removals from the complete-visit counts above. Every trained arm, persistence, and the fitting-mean reference used the identical final targets. Donor identities remained fixed. The caliper matches nominal timing within a tolerance, not clinical phase. All 688 registrations passed the frozen finite/geometry/overlap criteria. Historical coverage was not used to reduce scoring support.

### Joint outcome and paired intervals

Errors are patient-uniform, with seeds averaged within patients before 20,000 shared paired patient-bootstrap resamples. Intervals are two-sided 95 percent percentile intervals. Negative genuine-minus-control differences favor genuine history.

| Required component | Difference | Paired 95% interval | Decision |
|---|---:|---|---|
| Lesion MAE, genuine minus current | +0.010022 | [-0.015112, +0.035332] | Unresolved |
| Lesion MAE, genuine minus donor | +0.001778 | [-0.031200, +0.035084] | Unresolved |
| Gradient error, genuine minus current | +0.003181 | [+0.000507, +0.005730] | Failed |
| Gradient error, genuine minus donor | +0.003982 | [+0.001200, +0.006621] | Failed |
| Nonlesion MAE, genuine minus current | -0.008483 | [-0.023667, +0.008767] | Passed +0.05 upper-bound margin |

Exactly these five conditions determine the joint decision. Nonlesion error against donor history remains descriptive. The lesion intervals permit small gains or losses; they do not establish equivalence. Both gradient intervals lie wholly on the worse side for genuine history.

| Condition | Lesion MAE | Gradient error | Nonlesion MAE | Whole-brain MAE |
|---|---:|---:|---:|---:|
| Current-only with common timing | 0.560184 | 0.207067 | 0.377864 | 0.385879 |
| Genuine history | 0.570206 | 0.210248 | 0.369382 | 0.378521 |
| Donor history | 0.568427 | 0.206266 | 0.378723 | 0.387232 |
| Registered previous-FLAIR persistence | 1.018946 | 0.283040 | 0.510130 | 0.534353 |
| Fitting-patient-uniform mean | 1.526178 | 0.253176 | 0.624504 | 0.669687 |

Genuine history beat registered persistence by -0.448740 lesion MAE [-0.518969, -0.376090], -0.072792 gradient error [-0.084874, -0.061037], and -0.140748 nonlesion MAE [-0.165498, -0.116578]. Persistence was not the strongest method. Current-only also beat persistence, so the genuine-versus-persistence advantage does not isolate a contribution from history or demonstrate learned temporal dynamics.

### Per-seed results

| Seed | Condition | Lesion MAE | Gradient error | Nonlesion MAE |
|---|---|---:|---:|---:|
| 17 | Current | 0.571777 | 0.208017 | 0.377792 |
| 17 | Genuine | 0.567795 | 0.210032 | 0.372924 |
| 17 | Donor | 0.566204 | 0.206140 | 0.379270 |
| 23 | Current | 0.541970 | 0.205670 | 0.379495 |
| 23 | Genuine | 0.554732 | 0.210936 | 0.369079 |
| 23 | Donor | 0.572223 | 0.206457 | 0.377236 |
| 31 | Current | 0.566805 | 0.207514 | 0.376306 |
| 31 | Genuine | 0.588091 | 0.209776 | 0.366143 |
| 31 | Donor | 0.566856 | 0.206200 | 0.379664 |

Genuine-minus-current lesion differences were -0.003982, +0.012762, and +0.021286 for seeds 17, 23, and 31. Genuine-minus-donor lesion differences were +0.001591, -0.017491, and +0.021235. Gradient error was worse for genuine history against both controls in every seed. Nonlesion error against current-only was lower in every seed, but its aggregate interval crossed zero. Per-seed intervals and descriptive compartment errors are retained in the aggregate.

### Predefined nominal-gap groups

Each group is descriptive. Patients can contribute to more than one group, so subgroup patient counts do not sum to eighteen. These summaries do not replace the registered aggregate claim.

| Nominal gap | Patients / targets | Genuine minus current lesion MAE, 95% interval | Genuine minus current gradient error, 95% interval | Genuine minus current nonlesion MAE, 95% interval |
|---|---|---|---|---|
| Zero weeks | 5 / 6 | +0.011154 [-0.080467, +0.102776] | +0.003583 [+0.000108, +0.007057] | -0.011593 [-0.030102, +0.007117] |
| 1–8 weeks | 14 / 19 | +0.012916 [-0.022602, +0.051351] | +0.003053 [-0.000268, +0.006662] | -0.002162 [-0.026572, +0.024157] |
| Longer than eight weeks | 16 / 39 | +0.014810 [-0.025841, +0.055663] | +0.003970 [+0.000085, +0.007960] | -0.015964 [-0.029709, -0.001181] |

There is no aggregate lesion/gradient gain hidden by a short-gap concentration. Genuine-minus-current lesion and gradient point estimates are worse in every predefined group. Against donor history, lesion differences are -0.022186, +0.006107, and +0.012076 across these groups, with all intervals crossing zero. Genuine-minus-donor gradient differences are +0.002833, +0.004184, and +0.005106. Registered persistence remains worse than genuine history on lesion and gradient error in every group. The descriptive nonlesion improvement is strongest in the longer-gap group, not a short-gap exception that rescues the joint claim. No subgroup was promoted to the headline.

### Bounded verification and limits

Fitting took 3,407 seconds, or 56 minutes 47 seconds. Assessment took 240 seconds. The source-bound pipeline verified one saved prediction reload against the canonical scorer. A bounded post-run check confirmed all nine complete update schedules, matched initializations and draws, earliest-minimum checkpoint selection, checkpoint hashes, and identical scoring targets. It reproduced all five intervals from the retained patient vectors and shared bootstrap matrix, and checked exact persistence equality to registered prior FLAIR for one scoring visit. All 20,000 resamples were valid for the primary metrics. No refit, scoring sweep, or exhaustive replay was performed.

The two predefined seed-17 comparison panels were inspected after model lock. They show the trained predictions reproducing broad FLAIR structure while missing some detail; persistence retains earlier lesion appearance that differs from the current target. These examples were fixed in advance, are both same-week pairs, and do not determine the aggregate conclusion. There are 704 retained private prediction volumes, plus selected checkpoints, registration transforms and receipts, fitting curves, patient vectors, bootstrap inputs, and images.

This result bounds the approved predictor, current panel C, one preceding eligible visit, matching rule, and processed developmental cohort. It does not prove that history contains no additional information or that MRI has reached an observability limit. Registration imperfections, partial historical coverage, automated lesion masks, reused patients, and released preprocessing remain limitations. Forecasting, treatment-response modeling, and longer-history ordering were not tested. The bounded run is complete; no extension or new representation rescue is selected.

## R62 MU-Glioma-Post entry study, November 1, 2026

The operator selected MU-Glioma-Post after R61. The [fixed protocol](METHODS.md#r62-mu-glioma-post-measured-history-entry-study) tests measured regional history on a new cohort while reserving patients before image outcomes. The [aggregate](r62_mu_history_metrics.json) reports `USEFUL_HISTORY_NOT_ESTABLISHED`. This is an annotation-conditioned development comparison, not a trained MRI-token model or independent clinical validation.

### R62 acquisition and longitudinal feasibility

The official package contains 203 patients, 596 released visits, and 2,978 files. Two visits lack masks. The fixed split reserves forty patients and assigns 99 to fitting, 32 to selection, and 32 to pilot scoring. All 2,403 development files downloaded to `mosaic-gpu` and matched server file sizes. Reserved images were not downloaded or measured.

Clinical dates and image paths do not agree everywhere. Across the full metadata inventory, six patients have mismatched sets of dated and released timepoints. One fitting patient has no MRI dates and another two-visit fitting patient has decreasing dates. No corrections or timepoint reassignments were inferred. The released directories include numbering gaps and do not establish complete clinical adjacency.

| Partition | Assigned patients | Released visits | Complete measured visits | History-eligible patients | Forecast triples |
|---|---:|---:|---:|---:|---:|
| Fitting | 99 | 282 | 281 | 52 | 107 |
| Selection | 32 | 106 | 106 | 20 | 47 |
| Pilot scoring | 32 | 93 | 92 | 18 | 33 |
| Reserved | 40 | 115 | Unopened | Not assessed | Not assessed |

All 479 complete development visits used a common 240-by-240-by-155 physical grid. Every regional volume was independently reconstructed by source-array counting, with full compartment-volume conservation. Checks covered 2,395 source file hashes, finite MRI values, common geometry, and mask labels. Four MRI files from each mask-missing visit were excluded from measurement. The four channels preserve resection cavity separately from the three tumor-related compartments.

### R62 did not establish useful history forecasting on MU

The six-model two-triple smoke passed fitting, finite scoring, exact JSON reload, and future-field exclusion checks. Models were locked before pilot-scoring masks were opened. L, H, and S selected penalty 100; T fitted a coefficient of zero and therefore equals persistence. Errors below are patient-uniform MAEs in regional `log1p` millilitres.

| Predictor | MAE | Tumor-only MAE | Cavity MAE | Nonempty-cell MAE | Changed-cell MAE | Empty-cell MAE |
|---|---:|---:|---:|---:|---:|---:|
| Persistence, P | 0.165866 | 0.191091 | 0.090191 | 0.510087 | 0.609398 | 0.011733 |
| Average residual, E | 0.182874 | 0.205016 | 0.116448 | 0.482667 | 0.569061 | 0.047362 |
| Previous trend, T | 0.165866 | 0.191091 | 0.090191 | 0.510087 | 0.609398 | 0.011733 |
| Current-only ridge, L | 0.182648 | 0.204920 | 0.115833 | 0.482393 | 0.568228 | 0.047120 |
| History ridge, H | 0.182564 | 0.204896 | 0.115569 | 0.481646 | 0.567501 | 0.047280 |
| Shuffled-history ridge, S | 0.182223 | 0.204409 | 0.115664 | 0.481705 | 0.567357 | 0.047049 |

H improved on L by 0.0462%, below the fixed 5% threshold. The H-minus-L difference was −0.00008436, with two-comparison-adjusted interval [−0.00059561, +0.00043833]. H had 10.07% higher error than P. Its H-minus-P difference was +0.01669771, with adjusted interval [+0.00784910, +0.02518548]. Ordinary 95% intervals and all companion scores remain in the aggregate. The useful-history criterion failed on both comparisons. Shuffled history also had slightly lower point error than proper history.

Shifting previous source centers by +2 mm along reference axis two changed H's MAE to 0.182581. This small response under the selected strongly regularized model does not establish scanner robustness. H reduced errors on changed and nonempty cells relative to P but increased empty-cell error. Its tumor-only and cavity errors also exceeded P, so inclusion of cavity alone does not explain the primary failure.

Verification replayed nineteen saved prediction arrays, independently solved all three selected ridge problems by augmented least squares, and reconstructed patient aggregation and bootstrap intervals. All 2,403 downloaded file sizes matched the official path inventory. The four preselected sequences were visually inspected with FLAIR overlays and compartment trajectories. The examples showed both large changes and persistent empty compartments. L and H predictions were nearly coincident; some empty compartments received positive predictions, and several larger changes remained poorly predicted. These examples do not estimate failure prevalence or establish expert biological correctness. One inspected sequence includes a negative diagnosis-relative day, retained as released without reinterpreting it.

The raw aggregate SHA256 is `d1ea5dd40e1a4721cb6e59b6a9e235265d85c9ad590b2d4a2990e1f09ee473cd`. Private source files, locked models, prediction arrays, verification receipt, and inspection figures remain under `/home/aditya/experiments/world-model-r62-mu-20261101` on `mosaic-gpu`. Forty reserved patients remain unopened. No neural sweep, rescue fit, or reserved-cohort assessment followed.

## R61 corrected-source measured-history repeat, November 1, 2026

The operator authorized the same simple comparison on R60's corrected regional source volumes. R61 retained R59's six estimators, patients, ordered pairs, penalty grid, and selection rule, while refitting all predictors on `log1p` millilitres. The [protocol](METHODS.md#r61-corrected-source-measured-history-repeat) fixed the criteria and physical-displacement diagnostic before scoring. The [verified aggregate](r61_corrected_history_metrics.json) records `MEASURED_HISTORY_CRITERION_NOT_MET`.

### R61 corrected targets did not establish useful history forecasting

The two-pair real-data smoke passed. All six models were fitted and locked before corrected-target assessment. L, H, and S again selected penalty 100, and T's fitted coefficient was zero. Scoring uses thirty pairs from eight reused development patients. Errors are patient-uniform MAEs in `log1p` millilitres, not the earlier log-count units.

| Predictor | MAE | Nonempty-region MAE | Changed-cell MAE | Empty-cell MAE |
|---|---:|---:|---:|---:|
| P, current-measurement persistence | 0.280948 | 0.885680 | 1.021446 | 0.059882 |
| E, fitting mean change | 0.289938 | 0.884330 | 1.013473 | 0.075326 |
| T, constrained previous-trend extrapolation | 0.280948 | 0.885680 | 1.021446 | 0.059882 |
| L, current measurements and observed past gap | 0.289303 | 0.879942 | 1.009423 | 0.075958 |
| H, current measurements plus previous change | 0.288491 | 0.877989 | 1.007027 | 0.075503 |
| S, fitting-shuffled previous change | 0.289393 | 0.880556 | 1.009491 | 0.075975 |

H improved on L by 0.28%, below the required 5%. The H-minus-L difference was -0.000812, with adjusted interval [-0.001313, -0.000386]. H had 2.68% higher error than persistence, with difference 0.007543 and adjusted interval [0.001380, 0.011603]. H retained nonempty-region error relative to L and beat S by a small point margin. The full criterion failed. H improved on P for future-positive and changed cells, but produced greater error in future-empty cells and lost on the fixed primary endpoint.

Selected fitting MAEs were P 0.264868, L 0.270357, and H 0.267712. Selection MAEs were P 0.246588, L 0.258219, and H 0.257912. These models fit squared change error but select absolute future error. Strong selected regularization produced small corrections, so this comparison does not exhaust the information available in history. Removing previous change from H increased scoring MAE to 0.289210. The fixed 8 mm previous-mask displacement gave 0.288574, with no foreground displaced outside the reference field. P, E, and L were exactly invariant to both interventions. T was invariant because its coefficient was zero.

Verification reconstructed 582 previous, current, or future measurement uses from 270 distinct source visits. All thirty saved prediction arrays replayed exactly. Three ridge coefficient matrices matched an independent augmented least-squares solve, and all three selected penalties were reproduced. Patient groups were disjoint, the metadata matched the original adjacency, and future-field perturbations did not change inference. Separate calculations reproduced patient errors and both bootstrap interval families.

The fixed six-example figure was inspected. All three displayed predictors missed the large changes in the two growth and two disappearance examples. In the corrected targets, one old disappearance retains positive residual source volume. L and H remained close to one another; their small adjustments tracked the two stable examples more closely. These selected examples illustrate forecast behavior, not population accuracy or expert biological truth.

The initial smoke attempt exposed a Python package-path conflict with the old experiment imports, before any fitting. R61 then used its own deployed source package. The first plotting attempt found no plotting dependency, so the completed figure reused the existing dependency-free PNG renderer. Neither issue changed model definitions or the locked comparison. R61 is complete, with no further tuning launched. The endpoint correction did not reverse this recipe's negative result. Absolute MAEs cannot establish improvement over R59 because the target construction and units changed together.

## R60 source-volume target correction, November 1, 2026

The operator authorized fixing the regional target construction after R59. The [new implementation](METHODS.md#r60-source-volume-regional-targets) assigns original labeled voxel volumes to fixed physical regions by voxel center, without nearest-neighbor foreground sampling. It writes physical mm³ and explicitly transformed log-millilitre targets under a new cache contract. The [verification aggregate](r60_regional_source_target_metrics.json) records `SOURCE_VOLUME_TARGETS_VERIFIED`.

The real CLI smoke passed on two fitting visits from distinct patients. The full cache then completed for all 322 existing state visits from 64 patients. Every regional volume agreed with an independent extraction from the original source segmentation. Regional plus outside volume conserved source totals in every visit; outside foreground was zero throughout this cohort. Visit ordering, source hashes, loader contract checks, and the eighteen fixed R59 source-region measurements passed. The previously erased positive regions and the inspected reversed change now reproduce the source volumes.

### R60 quantified and corrected coarse regional target discrepancies

| Group | Visits | Patients | Regional cells | Positive source cells erased by coarse grid | Visits with erased positive cells | Adjacent pairs | Opposite change directions | Compared change cells |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fitting | 195 | 40 | 4,680 | 357 | 159 | 148 | 60 | 3,552 |
| Selection | 67 | 11 | 1,608 | 120 | 48 | 53 | 22 | 1,272 |
| Scoring | 60 | 13 | 1,440 | 128 | 50 | 40 | 20 | 960 |

Across the complete existing state cohort, 605 of 7,728 region-compartment measurements were zero on the old coarse grid despite positive source volume. They occurred in 257 visits. No coarse-positive cell had zero source volume. Among 5,784 region-compartment changes across 241 retained adjacent pairs, 102 had opposite nonzero change directions under coarse counting and source-volume aggregation. These counts include small changes and use all retained pairs, not only history-eligible pairs. They are endpoint-pooled target discrepancies, not patient-weighted model scores or a general population prevalence estimate.

All twelve unit and CLI tests passed on `mosaic-gpu`, including a synthetic small source region that nearest-neighbor downsampling erases but the corrected method retains. The first test run required a `1e-12` absolute tolerance for an anisotropic voxel-volume determinant differing from the literal expected value by about `1.8e-15`. No algorithm or real-data target repair was needed after that test adjustment. The source-volume cache is complete at the private R60 `targets-v1` path. The new loader rejects legacy units, changed array bytes, and mismatched visit order.

R60 is complete. Source masks, historical coarse arrays, and previous results remain intact. The correction changes the endpoint to `log1p` millilitres, so historical MAEs and normalization cannot be carried over as corrected-target scores. No model was retrained, and no forecasting improvement is claimed. The original automated masks and registration assumptions still limit biological interpretation.

## R59 measured history and target inspection, November 1, 2026

The operator authorized the [measured-history and target-quality diagnostic](METHODS.md#r59-measured-history-and-target-quality-diagnostic) after asking whether repeated forecasting failures indicated a data problem. R59 uses R58's fixed history-eligible cohort and targets, but supplies earlier automated regional measurements directly. It compares persistence, mean change, constrained trend extrapolation, and three small regressions. A separate image inspection checks six target-selected examples against their original segmentations and pre-resize MRI. These are reused development patients.

### R59 found target-resolution artifacts and no useful measured-history forecast gain

The [regression aggregate](r59_measured_history_metrics.json) records `MEASURED_HISTORY_CRITERION_NOT_MET`. The two-pair real-data smoke passed all six models, exact coefficient reload, direct mask-count reconstruction, and future exclusion. Deterministic ridge fitting then completed and all models were locked before R59 scoring. L, H, and S each selected penalty 100, the strongest allowed value. T's fitted nonnegative extrapolation coefficient was zero, so T equals persistence. Primary scoring contains thirty pairs from eight patients. Errors below are patient-uniform regional log-count MAEs.

| Predictor | MAE | Nonempty-region MAE | Changed-cell MAE | Empty-cell MAE |
|---|---:|---:|---:|---:|
| P, current-measurement persistence | 0.339672 | 0.939263 | 1.287528 | 0.179829 |
| E, fitting mean change | 0.350063 | 0.946243 | 1.275036 | 0.192965 |
| T, constrained previous-trend extrapolation | 0.339672 | 0.939263 | 1.287528 | 0.179829 |
| L, current measurements and observed past gap | 0.349692 | 0.943672 | 1.271247 | 0.193053 |
| H, current measurements plus previous change | 0.349027 | 0.941137 | 1.269013 | 0.192748 |
| S, fitting-shuffled previous change | 0.349843 | 0.944322 | 1.271514 | 0.193219 |

H improved MAE over L by only 0.19%, below the fixed five-percent useful-gain requirement. The H-minus-L difference was -0.000665, with family-two adjusted interval [-0.001032, -0.000323]. This supports a small directional advantage on this development cohort, not the required useful benefit. H had 2.75% higher MAE than P, with difference 0.009355 and adjusted interval [0.003370, 0.015063]. It retained nonempty error relative to L and had slightly lower point error than S. The full criterion nevertheless failed.

Selected fitting MAEs were L 0.319503 and H 0.316633 versus P 0.312145. Selection MAEs were L 0.302710 and H 0.302549 versus P 0.286401. These regressions fit squared error but select absolute error, and the strongest selected regularization produces small corrections. The comparison does not demonstrate that this model class exhausts available history information. Removing earlier change from H increased scoring MAE to 0.349671. Shifting the previous coarse mask one voxel gave 0.349134. P, E, and L were exactly unchanged by both interventions. T was unchanged because its coefficient was zero.

Verification reconstructed all consumed previous, current, and future regional measurements from masks across the three groups. Thirty prediction arrays replayed exactly. All selected penalties were reproduced, and three ridge solutions agreed with an independent augmented least-squares solve. Patient splits were disjoint and changing future target fields did not affect inference. A separate calculation reconstructed patient errors and both bootstrap interval families for the two planned contrasts. No neural training or hyperparameter expansion followed.

### R59 source-mask inspection confirmed coarse-target information loss

All six fixed previous/current/future image panels were inspected in both initial and refined renderings. The refined panels show the selected region boundary, original segmentation contours on pre-resize CT1 and FLAIR, and the actual 24-cubed target outlines. The two growth examples show visible expansion of FLAIR abnormalities. The disappearance examples show substantial structural and signal changes, including a cavity-like appearance in one sequence. These observations do not identify treatment or establish expert tumor labels. Source contours broadly follow the visible abnormal regions in these selected views; no gross coordinate mismatch was apparent. Acquisition resolution and contrast vary visibly between visits.

The [source-volume aggregate](r59_target_inspection_metrics.json) confirms a concrete target-construction limitation. One voxel of the inspected 24-cubed field represents 617.044343 mm³. Three of the eighteen selected region-visit measurements were zero on that coarse grid despite positive original segmentation volumes ranging from 77 to 982 mm³ within the same physical region. In one of the six current-to-future comparisons, the coarse measurement increased by 16.67% while the original segmentation volume decreased by 9.50%. These are resampling discrepancies within the same automated source labels, not disagreements between expert raters.

The source comparison counts original labeled voxel centers within the physical regional bounds. An independent axis-based extraction reproduced all eighteen source regional volumes. Original MRI content hashes and mask hashes matched, reconstructed model MRI and coarse target arrays were exact, and the pre-resize physical grids matched across the inspected visits. Thus the observed coarse-grid zeroing and direction reversal are not explained by a stale cache or a mistaken rendering input. Matching physical grids still does not prove anatomical registration accuracy or mask validity.

The six examples were deliberately selected for growth, disappearance, and stability. Three discrepancies among their eighteen measurements are not a population prevalence estimate. Large changes also remained in the source segmentations and MRI, so coarse sampling does not explain every difficult forecast. R59 establishes that regional burden and change can be distorted by nearest-neighbor sampling to 24 cubed before counting. It does not quantify how much this contributed to R56 through R59 model outcomes.

R59 is complete. Numerical outcomes on the historical coarse targets remain reproducible. Biological interpretations and further model comparisons need a volume-preserving regional target definition and a bounded source-mask check before another architecture study. No source masks, cached targets, or historical results were overwritten.

## R58 spatial latent forecasting, November 1, 2026

After R57, the operator authorized the next experiment, a [spatial latent-prediction transformer](METHODS.md#r58-spatial-latent-prediction-for-anatomical-forecasts). R58 tests whether a frozen future-feature objective improves anatomical forecasts beyond direct forecasting and shuffled future teachers. It preserves eight regional tokens and supplies current automated compartment measurements to every predictor. Future features and measurements are targets only. Actual adjacent pairs must have both visits in the verified state cache, excluding terminal targets absent from that cache. There are 148 fitting, 53 selection, and 40 scoring pairs, with 119, 45, and 30 history-eligible pairs respectively.

All five two-update real-data smokes passed. Prefix, target, and donor MRI tokens and regional measurements reconstructed exactly. Trainable components changed, checkpoint and optimizer reloads were exact, future-target changes left predictions unchanged, and latest-only arms ignored earlier scans. Ten fixed fits completed 1,024 updates each, totaling 10,240. Within each seed, all five arms shared initialization, 81,030 parameters, and patient-uniform sample draws. Checkpoints and calibration were locked before R58 scoring.

Selection chose DH-42 at step 64, DL-42 at 192, JH-42 at 384, JL-42 and SH-42 at 256, DH-43, DL-43, and SH-43 at 64, JL-43 at 192, and JH-43 at initialization. Only JH-42 and JL-42 passed the selected-checkpoint fitting-readiness criterion. Final training MAEs ranged from 0.082429 to 0.134018, substantially below their initial values. All final checkpoints exceeded the fitting criterion descriptively, but those checkpoints were not selected for scoring. The result therefore reflects a failed training-and-selection recipe, not a demonstrated inability to fit the training data.

### R58 did not establish a future-feature or history forecasting gain

The [verified aggregate](r58_spatial_latent_forecast_metrics.json) records `FUTURE_FEATURE_CRITERION_NOT_MET` and `HISTORY_CRITERION_NOT_MET`. Primary scores use thirty history-eligible pairs from eight reused development patients. The table averages patient-uniform scores across seeds 42 and 43. Errors and widths use regional `log1p` standardized-grid voxel counts.

| Predictor | MAE | CRPS | Nonempty-region MAE | 90% coverage | 90% width |
|---|---:|---:|---:|---:|---:|
| DH, direct with history | 0.372471 | 0.324579 | 0.887246 | 0.813833 | 0.786741 |
| DL, direct with latest only | 0.366280 | 0.319003 | 0.882300 | 0.782955 | 0.624356 |
| JH, proper future teacher with history | 0.397308 | 0.335170 | 0.888195 | 0.864180 | 1.427842 |
| JL, proper future teacher with latest only | 0.367460 | 0.303126 | 0.891611 | 0.845269 | 0.929495 |
| SH, shuffled future teacher with history | 0.361112 | 0.308869 | 0.862401 | 0.820778 | 0.893916 |
| P, current-measurement persistence | 0.339672 | 0.356008 | 0.939263 | 0.836992 | 1.208343 |
| E, fitting mean change | 0.376658 | 0.358910 | 0.944858 | 0.829923 | 1.200212 |

JH had 6.67% higher MAE than DH. Its mean difference was 0.024837, with family-three adjusted interval [-0.001541, 0.047472]. Seed differences were -0.003889 and 0.053563. The ordinary 95% interval was [0.003522, 0.043948], but the prespecified adjusted interval crossed zero. The objective did not establish a useful gain, and the adjusted comparison does not establish harm relative to direct forecasting. JH retained nonempty-region error within two percent of DH but failed CRPS retention and lost to both measured-state controls on primary MAE.

JH had 10.02% higher MAE than SH, with difference 0.036197 and adjusted interval [0.021274, 0.056053]. It had 8.12% higher MAE than JL, with difference 0.029848 and adjusted interval [0.018310, 0.041065]. Both seeds were worse in these two comparisons. The proper temporal pairing and history criteria therefore failed. All five neural arms had higher primary MAE than persistence, although their nonempty-region MAE and CRPS were lower. Those secondary improvements do not satisfy the fixed primary criterion.

Proper-teacher latent-delta MSE averaged 0.714928 for JH and 0.565167 for JL, versus 0.511645 for zero feature change. JH's predicted delta RMS was 0.525072 versus target RMS 0.715294. Its future-feature variance ratio was 1.117789, which does not indicate constant predictions but also does not establish correct dynamics. DH latent MSE was 2.140117, but direct arms can change feature coordinates to serve the anatomical head. Their latent scores are not a fair representation-quality criterion. Anatomical error on changed cells averaged 1.159885 for JH and 1.177875 for DH. This descriptive subset result does not replace the primary comparison.

JH primary MAE was 0.397229 under latest-image intensity perturbation and 0.404596 under resolution perturbation. Removing history gave 0.392954, and reversing earlier MRI gave 0.396710. Latest-only outputs were exactly invariant to both history interventions. Synthetic nuisance tests retain current automated measurements, so they do not establish whole-pipeline scanner robustness. All-pair MAEs were DH 0.348910, DL 0.379557, JH 0.386116, JL 0.368034, and SH 0.367669 across forty pairs from ten patients.

All four figures were inspected. In the first fixed example, both models missed disappearance in region five for compartments one and three, and missed new compartment-three burden in region eight. In the second, JH improved compartment-three burden in region five but produced wide intervals across several regions. Both underestimated compartment-two burden in region six. Negative Gaussian means or interval bounds remained visible. Fitting curves fell in both seeds while selection curves often worsened, especially for direct history. These observations support the selection limitation without selecting a replacement checkpoint from scoring results.

Verification re-encoded five full MRI token archives, reconstructed regional targets and actual adjacent-pair membership in all three groups, and replayed seventy neural archives plus two controls. All 214 mean, scale, and delta arrays matched exactly. It verified 10,240 final optimizer updates, shared initialization and draws, cross-patient donor mapping, selected steps, twelve calibrations, source hashes, future exclusion, and independently calculated patient MAE contrasts and both interval families. Numerical verification passed without model or scoring repairs. No independent scientific review has occurred.

R58 is complete. The failed selected-checkpoint readiness, restricted cohort, coarse automated targets, and reused patients limit capability claims. Neither new encoder pretraining nor independent clinical validation was tested. The fixed recipe does not justify a larger transformer or a generative expansion.

### R58 post-hoc target and interval diagnosis

After closure, the operator asked whether repeated forecasting failures indicated a data problem. A [descriptive cache probe](r58_target_diagnostics.json) checked the fixed history-eligible cohort without refitting or changing selection. `scratch/spatial-latent-forecast/diagnose58.py` reproduces these aggregate checks on `mosaic-gpu`. Fractions and signed changes give equal weight to patients, then pairs, then region-compartment cells. Interval ranges and medians are pair-weighted nominal week bins, not exact scan-day intervals.

| Group | Patients | Future cells empty | Cells empty at both visits | Absolute log-count change above 0.1 | Mean signed log-count change | Nominal gap min / median / max, weeks |
|---|---:|---:|---:|---:|---:|---|
| Fitting | 24 | 74.75% | 68.82% | 28.19% | -0.023768 | 1 / 14 / 87 |
| Selection | 6 | 82.10% | 78.36% | 19.18% | 0.064710 | 1 / 13 / 30 |
| Scoring | 8 | 79.20% | 72.52% | 25.23% | -0.096204 | 1 / 11 / 18 |

No retained history-eligible pair had a zero nominal week gap. Persistence MAEs were 0.312145, 0.286401, and 0.339672 in fitting, selection, and scoring respectively. Empty cells explain many exact persistence predictions, but about one quarter of scoring cells changed by more than 0.1 log-count units. Mean signed change differed across splits, with positive selection change and negative fitting and scoring change. These are descriptive differences from small patient groups, not a diagnosis of scanner shift, label corruption, or biological change. Existing target reconstruction verifies agreement with cached automated masks, not the masks' biological accuracy.

## R57 regional graph state estimation, November 1, 2026

After R56, the operator requested the next step and work moved to the [regional graph option](TODO.md#r57-regional-graph-state-estimation). The [fixed protocol](METHODS.md#r57-regional-graph-state-estimation) compares correct grid adjacency with self-only messages, shuffled correspondence, and a latest-only graph. All arms use the same frozen spatial encoder, regional targets, patient allocation, loss, sample draws, and nonempty-region checkpoint selection. This tests current-state context, not future progression. The new endpoint was motivated by the already observed R56 nonempty-region failure and remains exploratory development evidence.

All four two-update real-data smokes passed. Fresh MRI tokens and current targets reconstructed exactly; expected weights changed; checkpoint and optimizer reloads were exact; target values did not affect inference. Changing only region zero of the latest observation affected exactly nodes 0, 1, 2, and 4 for GH/GL, only node 0 for SH, and nodes 0, 1, 2, and 3 for RH. These are zero-based grid indices, not participant identifiers. GL was exactly invariant to earlier-token changes. GH, SH, and RH had identical 10,508 trainable parameters and initialization; GL had 9,030 parameters.

Eight fixed fits completed 1,024 updates each, totaling 8,192. Selected steps were GH-42 and RH-42 at 832, GL-42 at 896, and the other five cells at 1,024. Every cell passed both overall and nonempty fitting-readiness criteria. Several selected final checkpoints and continued improving near the endpoint, so the schedule does not establish convergence. All eight models and calibrations were locked before R57 scoring arrays were opened. Their outcomes had already been studied under earlier protocols; this lock does not make the patients independent validation.

### R57 regional context helped without established adjacency or history benefit

The [verified aggregate](r57_regional_graph_metrics.json) records `GRAPH_STATE_CRITERION_NOT_MET` and `HISTORY_CRITERION_NOT_MET`, while preserving a positive primary context comparison. The forty history-eligible scoring endpoints come from ten reused patients. Scores below average patient-uniform metrics over seeds 42 and 43. Targets and interval widths use regional `log1p` standardized-grid voxel counts.

| Estimator | Nonempty-region MAE | Overall MAE | CRPS | 90% coverage | 90% width |
|---|---:|---:|---:|---:|---:|
| GH, correct graph with history | 1.285667 | 0.503679 | 0.377594 | 0.853509 | 1.773536 |
| SH, self-only messages with history | 1.450226 | 0.512232 | 0.396216 | 0.825521 | 1.582148 |
| RH, shuffled correspondence with history | 1.275725 | 0.492305 | 0.370926 | 0.874516 | 1.827338 |
| GL, correct graph with latest only | 1.218356 | 0.513937 | 0.382147 | 0.888212 | 2.282271 |
| Fixed R56 FP, simple SSM posterior | 1.255398 | 0.481357 | 0.391363 | 0.808869 | 1.445700 |
| Fixed R56 C, fitting-patient mean | 1.520279 | 0.835144 | 0.566769 | 0.872436 | 2.814034 |

GH reduced nonempty-region MAE by 11.35% relative to SH and retained overall MAE and CRPS. Both seeds improved, with differences of -0.140561 and -0.188558. The mean difference was -0.164559, with a family-three adjusted interval of [-0.274863, -0.056162]. This passes the fixed primary context criterion under matched parameter counts, initialization, sampling, and selection. It supports a bounded benefit from access to other regions in this recipe.

Correct correspondence did not earn a separate claim. GH was 0.78% worse than RH on mean nonempty-region MAE. Its GH-minus-RH difference was 0.009942 with adjusted interval [-0.061701, 0.072871], and the seed directions disagreed. RH changes which physical regions are linked while preserving graph topology and degree. The result does not show that face-adjacent relationships are preferable to the fixed alternative correspondence.

History also failed its separate criterion. GH was 5.52% worse than GL, with both seeds worse. The GH-minus-GL difference was 0.067311 with adjusted interval [-0.053810, 0.183771]. GL had the lowest point nonempty-region MAE among the new arms, despite fitting less closely than the history models. GH also failed to beat the fixed simple-filter FP reference. Those reference controls used R56's different selection rule and are descriptive; the matched SH, RH, and GL comparisons carry the attribution tests. None of the zero-crossing intervals establishes that a useful effect is impossible.

Removing history from the fitted GH reduced nonempty-region MAE from 1.285667 to 1.179589. Reversing earlier observations gave 1.277938. Replacing its adjacency with identity worsened error to 1.378898. GL remained exactly invariant to history removal and reversal; replacing its edges with identity worsened error from 1.218356 to 1.327225. SH was exactly invariant to the self-edge intervention. These checks establish model dependence, but post-hoc interventions do not replace separately fitted controls or select a new model.

Latest-image intensity perturbation left GH nonempty-region MAE near identity at 1.285367. Resolution perturbation lowered it to 1.174528; GL's resolution value was 1.048014 versus identity 1.218356. Lower error under synthetic degradation is not evidence of biological fidelity or real scanner robustness. GH and GL had wider nominal intervals than SH, and Gaussian intervals still extended below feasible counts. The aggregate reports 50% coverage and all-region diagnostics as well.

All four figures were inspected: two fixed regional prediction panels and the fitting and selection curves. Correct graph messages improved some nonempty regions in the first example, while both methods missed substantial compartment-one and compartment-two burden in region seven of the second example. Several zero-target regions still had negative predicted means or interval bounds. The curves confirmed sustained fitting and showed latest-only selection error below the history graph for much of the schedule. These fixed panels are examples, not a second model-selection procedure.

Verification re-encoded five complete MRI token archives, reconstructed current regional targets and cutoff sequences in all three groups, and replayed 64 new prediction archives plus twelve fixed reference archives. All 152 mean/scale arrays matched exactly. It also verified 8,192 final optimizer updates, identical history-arm initial parameters, shared sample draws, all selected steps, eight calibrations, and restoration of adjacency after diagnostic interventions. A separate direct calculation reconstructed nonempty patient errors and both interval families for all three contrasts. The numerical verification passed without model or scoring repairs. No independent scientific review has occurred.

R57 is complete. Preserve the positive cross-region-context result and the unestablished correspondence and history benefits separately. The full graph-state criterion failed, and no future-transition run, larger graph, loss search, or additional fit followed. The broader graph architecture remains unresolved on coarse automated targets and reused patients.

## R56 regional probabilistic observation assimilation, November 1, 2026

The operator selected option 1 in the [architecture inventory](TODO.md#world-model-architecture-options-november-1-2026). The [fixed first-stage protocol](METHODS.md#r56-regional-probabilistic-observation-assimilation) tests current regional proxy estimation with a common frozen R52 spatial encoder. H learns Gaussian observation assimilation across up to four observed visits; L uses the latest observation with the same available timing metadata. Eight fixed grid regions each carry three compartment log-count targets. This is current-state estimation, not future MRI prediction or native-volume forecasting.

The real-data smoke passed two updates per model, exact fresh-MRI versus cached-token reconstruction, target reconstruction, intended parameter changes, frozen encoder checks, exact checkpoint and optimizer reload, target exclusion, and history dependence. All four fixed fits then completed 1,024 updates each, totaling 4,096. Selected steps were L-42 at 896, H-42 at 448, L-43 at 896, and H-43 at 1,024. All models, controls, and scale calibrations were locked before scoring images. Every neural cell passed fitting readiness, improving fitting history-endpoint MAE over the fitting-mean control by 54.78% to 64.41%.

### R56 did not establish a useful learned-assimilation gain

The [verified aggregate](r56_regional_state_metrics.json) records `ASSIMILATION_CRITERION_NOT_MET`. The primary comparison includes forty history-eligible endpoints from ten reused scoring patients. Numbers below average patient-uniform scores across seeds 42 and 43. MAE and CRPS use regional `log1p` standardized-grid voxel counts. Coverage is a fraction, and width is in the same log-count units.

| Current-state estimator | MAE | CRPS | Nonempty-region MAE | 90% coverage | 90% width |
|---|---:|---:|---:|---:|---:|
| L, latest observation | 0.510983 | 0.385782 | 1.290387 | 0.808810 | 1.531403 |
| H, learned regional assimilation | 0.502479 | 0.384873 | 1.386604 | 0.871865 | 1.892188 |
| P, previous-observation persistence | 0.595635 | 0.490250 | 1.563316 | 0.780603 | 1.724621 |
| M, mean observation history | 0.490629 | 0.399273 | 1.290662 | 0.809859 | 1.504759 |
| FP, R38-style SSM current posterior | 0.481357 | 0.391363 | 1.255398 | 0.808869 | 1.445700 |
| C, fitting-patient mean | 0.835144 | 0.566769 | 1.520279 | 0.872436 | 2.814034 |

H reduced primary MAE by only 1.66%, below the fixed five-percent requirement. Its H-minus-L difference was -0.008503, with a paired patient-bootstrap 95% interval of [-0.040886, 0.026523]. Both seeds improved, with differences of -0.013746 and -0.003260, but the interval crossed zero. The interval also includes a useful improvement, so this is inconclusive about the broader capability rather than evidence that history cannot help.

H also failed to beat mean-history estimation and the simple SSM posterior. Nonempty-region error worsened by 7.46%, exceeding the two-percent retention allowance. Of the region-compartment cells across the forty scoring endpoints, 76.25% had zero targets. This fraction is endpoint-pooled, not patient-weighted. The nonempty score therefore matters when interpreting the small aggregate gain. H retained CRPS, but its improved nominal 90% coverage came with wider intervals. Nominal 50% coverage was 0.616895 for H and 0.432178 for L. Neither these marginal scores nor the two learned variance terms establish calibrated biological uncertainty.

The history-dependence diagnostics were consistent with use of earlier observations. H's mean MAE rose from 0.502479 to 0.518100 when history was removed and to 0.505132 when earlier observations were reversed. L remained exactly unchanged. These interventions on a fitted H do not replace the separately trained L comparison. Latest-image intensity and resolution perturbations gave H MAEs of 0.502252 and 0.499284, versus L values of 0.511324 and 0.530006. Lower error under a synthetic perturbation is not evidence of better biological state estimation.

The two fixed regional prediction panels and the curve figure were inspected. The panels showed missed compartment burden in nonempty regions, intervals extending below the feasible zero count, and varying interval widths between examples. In the second example, both models missed the large compartment-two burden in region seven. H continued reducing fitting error after selection scores had mostly flattened. This supports actual fitting and a development generalization limitation, not a failure to update. The panels were selected by the fixed lexical rule, not for favorable appearance.

Verification re-encoded all five full MRI token archives: fitting identity, selection identity, and scoring identity, intensity, and resolution. Every token archive matched exactly. It independently reconstructed the regional targets in all three groups, replayed 36 complete prediction archives containing 72 mean/scale arrays, refitted eight controls and all calibration choices exactly, checked 4,096 final optimizer updates, and independently recalculated patient MAE differences and the bootstrap interval. The verifier required two reporting repairs: a `1e-12` tolerance for independently reduced float64 normalization moments, whose maximum observed discrepancy was about `6.7e-16`, and conversion of a source filename to `Path` for hashing. Model weights, predictions, selection, and scores did not change.

The first R56 current-state recipe is complete. It has not earned a forecasting expansion. The broader regional probabilistic architecture remains unresolved because this test used coarse grid regions, one frozen supervised encoder, zero-heavy automated targets, and ten reused scoring patients. No future-transition study or further tuning followed. No independent scientific review has occurred.

## R55 direct spatial-code fitting, November 1, 2026

The operator authorized the [decoder-isolation diagnostic](METHODS.md#r55-direct-spatial-code-fitting-diagnostic) after R54. All three cells started from the same R54 trainable-completion checkpoint and exactly reproduced its initial predictions. Z optimized per-example spatial codes with the decoder frozen. ZD optimized codes and decoder jointly. MRI continued full MRI-to-image fitting for the same 512 additional updates. Optimizer moments were reset in every cell. Only the same two current fitting visits were read.

The three real two-update smokes passed finite-loss and code-gradient checks, intended parameter changes, frozen decoder invariance, exact checkpoint prediction replay, and optimizer reload. Peak allocated memory was 2.430 GiB. A conservative projection from the slower smoke updates plus a five-minute assessment/replay allowance was 7.005 minutes. All three fixed fits completed, totaling 1,536 additional updates. Their fitting, scoring, saving, and companion timers summed to 0.753 minutes, excluding initial construction. Full numerical verification took 7.857 seconds before rendering. These are scoped timers, not a performance benchmark.

### R55 code bypass did not resolve image fidelity

The [verified aggregate](r55_decoder_capacity_metrics.json) records `DIRECT_CODE_FITTING_CRITERION_NOT_MET`. No cell met the required 20% MAE reduction over the strongest fixed control on both examples.

| Recipe or reference | Example 1 MAE | Example 2 MAE | Mean MAE | Mean SSIM | Mean gradient MAE |
|---|---:|---:|---:|---:|---:|
| R54 common starting prediction | 0.383038 | 0.313523 | 0.348281 | 0.493755 | 0.141468 |
| Z, codes fitted through frozen decoder | 0.354825 | 0.294349 | 0.324587 | 0.538023 | 0.139079 |
| ZD, codes and decoder fitted jointly | 0.332335 | 0.286810 | 0.309572 | 0.584117 | 0.136431 |
| MRI, equal-duration continuation | 0.337559 | 0.284215 | 0.310887 | 0.564222 | 0.135899 |
| Two-example current-FLAIR mean | 0.337465 | 0.337465 | 0.337465 | 0.582974 | 0.109845 |

The two-example mean remained the minimum-MAE fixed control for each case. Z was 5.14% worse on example one and 12.78% better on example two. ZD improved by only 1.52% and 15.01%. MRI was 0.03% worse and 15.78% better. ZD's mean MAE advantage over equal-duration MRI was only 0.42%, with opposite directions on the two examples. That descriptive difference does not establish an encoder-bypass advantage. ZD passed mean SSIM retention by a small margin; no cell passed gradient retention.

All four figures, three prediction panels and one curve figure, were inspected. Direct-code fitting sharpened some coarse structures but left blurred detail and substantial residual errors. The jointly fitted code/decoder and continued MRI predictions looked similar. Displayed artifacts outside the fixed scoring field were neither trained nor scored. Both trainable-decoder cells initially worsened after the optimizer restart and then recovered. Every cell still improved between updates 256 and 512, so the endpoint does not establish convergence or a decoder capacity ceiling.

Zero-code mean MAE was 0.514215 for Z, 0.529465 for ZD, and 0.511959 for MRI. Exchanging codes between examples increased MAE to 0.586976, 0.603868, and 0.586111. Exchanged-code predictions exactly exchanged the identity outputs, and zero codes produced identical images across the two examples. These checks establish dependence on example-specific codes, not generalizable anatomical representations. MRI intensity and resolution companions scored MAE 0.350019 and 0.331763, versus identity 0.310887. Code-only cells have no MRI input after initialization; invariance to image perturbation would be automatic rather than evidence of robustness.

The target-copy oracle returned exact zero MAE and unit SSIM. This checks the metric/target path and shows that exact images are valid outputs in unconstrained voxel space. It does not prove that this decoder can represent or learn those images. Three fixed image controls were reconstructed and matched R54 exactly. All eleven final prediction arrays replayed exactly. Verification reconstructed the initial codes from the original R54 MRI model, checked common decoder initialization and starting predictions, frozen decoder hashes, changed trainable weights, all 1,536 optimizer updates, source hashes, and independent per-example MAE decisions. Images, learned codes, weights, and case membership remain private.

R55 is closed. Bypassing the MRI encoder did not resolve poor individual-image fitting under the fixed recipe, and much of the improvement over R54 was also obtained by further MRI fitting. Decoder parameterization, warm-start basin, optimizer settings, and finite duration remain unresolved alternatives. Two fitting examples and one seed cannot establish generalization, clinical utility, or an intrinsic architectural limit. No new decoder, longer fit, held-out evaluation, or JEPA training followed. No independent scientific review has occurred.

## R54 two-patient fitting diagnostic, November 1, 2026

After R53 closed, the operator authorized a [separate fitting-capacity diagnostic](METHODS.md#r54-two-patient-fitting-capacity-diagnostic). Two fitting patients and one history-eligible positive-week-gap pair per patient were fixed by identifier and numeric visit order before target inspection. Selection and scoring images remain closed. The private run is `/home/aditya/experiments/world-model-r54-20261101/` on `mosaic-gpu`.

The real two-update smoke passed all four stage and encoder-trainability combinations. Intended encoder and head weights changed, frozen encoders stayed unchanged, and saved predictions replayed exactly. Forecasts initialized exactly to persistence. Completion excluded FLAIR inputs; forecast construction excluded future images and the future interval. Earlier-image replacement changed history outputs while latest-only outputs remained identical. All measured losses and gradients were finite.

Peak allocated GPU memory was 2.633 GiB. Projecting 512 updates per cell from the slower of each smoke's two measurements gave 5.097 minutes for six fits. Adding ten minutes for assessment and replay gave a 15.097-minute planning estimate. This includes cold-update overhead but is not an observed full-run duration or speed comparison. All six fits completed under `fit-01`, totaling 3,072 updates. Their timers summed to 2.548 minutes, including intermediate scoring, final checkpoint writes, and nuisance assessment, but excluding initial model construction. Numerical replay took 14.389 seconds before rendering.

### R54 learned corrections but failed substantial individual fitting

The [verified aggregate](r54_flair_learnability_metrics.json) records all six cells below the declared substantial-fitting criterion. F freezes the encoder; T trains it. H uses non-FLAIR history; L uses latest-only non-FLAIR input with matched timing. All four forecast cells start from the same final T completion encoder. These are final-checkpoint fitting errors on two examples, not held-out estimates.

| Recipe or control | Example 1 MAE | Example 2 MAE | Mean MAE |
|---|---:|---:|---:|
| Completion F | 0.403018 | 0.314564 | 0.358791 |
| Completion T | 0.383038 | 0.313523 | 0.348281 |
| Completion two-example mean | 0.337465 | 0.337465 | 0.337465 |
| Completion R53 fitting mean | 0.501323 | 0.475645 | 0.488484 |
| Forecast F-H | 0.233411 | 0.356829 | 0.295120 |
| Forecast F-L | 0.234704 | 0.356308 | 0.295506 |
| Forecast T-H | 0.236918 | 0.366571 | 0.301745 |
| Forecast T-L | 0.236272 | 0.361170 | 0.298721 |
| Forecast persistence | 0.267839 | 0.546724 | 0.407282 |
| Forecast smoothed persistence | 0.347267 | 0.534809 | 0.441038 |
| Forecast two-example mean | 0.331676 | 0.331676 | 0.331676 |
| Forecast R53 fitting mean | 0.505099 | 0.481484 | 0.493292 |

The two-example means deliberately use both fitting targets. They are shared-template fitting controls, not deployable future predictors. They were fixed before fitting as part of a test of individual-image memorization beyond a shared template. Current T2, the remaining completion control, had mean MAE 0.970126.

Both completion models improved over the R53 fitting mean but failed to beat the two-example mean on the first example. Completion T was 13.50% worse on example one and 7.09% better on example two than that stronger control. Completion F was 19.43% worse and 6.79% better. Neither achieved the required 20% improvement on both cases. Mean completion SSIM was 0.460257 for F and 0.493755 for T, versus 0.582974 for the two-example mean. Gradient MAE was 0.141984 and 0.141468 versus 0.109845. Encoder adaptation slightly improved completion but did not resolve the fidelity failure.

All forecast cells improved mean fitting MAE over persistence. F-H had the lowest descriptive mean, 0.295120 versus 0.407282, a 27.54% reduction. It improved example one by only 12.85% over its strongest control, persistence. On example two it was 7.58% worse than the strongest control, the two-example future mean. The other forecast cells also failed the per-example criterion. History versus latest-only differences were small, and trainable-encoder forecasts had worse mean error than their frozen counterparts in this one-seed run. These observations do not establish a general history effect or a causal optimization explanation.

Forecast SSIM retention passed for all cells against the per-example minimum-MAE controls, but gradient retention failed for all. F-H mean SSIM was 0.722924 versus persistence's 0.528457; gradient MAE was 0.154665 versus persistence's 0.160118. F-H's mean predicted-change cosine was 0.568675, and its mean change-norm ratio was 0.543713. Error in the largest-actual-change decile fell from persistence's 1.609509 to 1.005868, while the target-fitted two-example mean scored 0.581535. These retrospective diagnostics measure image change, not tumor evolution.

All six prediction panels and both learning-curve figures were inspected. Completion recovered coarse example-specific structures but blurred boundaries and fine detail. Forecasts learned broad corrections, including some reduction of conspicuous appearance differences, while leaving substantial structural residuals. Displayed predictions outside the fixed scoring field were not trained or scored. Both completion and forecast curves still improved between steps 256 and 512. The fixed endpoint therefore does not establish convergence or an intrinsic architectural capacity ceiling.

Zero-token completion mean MAE rose to 0.501602 for F and 0.514215 for T, from 0.358791 and 0.348281. F-H zero-token MAE rose to 0.405643 from 0.295120. These models depend on image tokens. F-H intensity-companion MAE was 0.409879 versus matched perturbed persistence's 0.417122; resolution-companion MAE was 0.347101 versus matched persistence's 0.441038. Compared with identity inputs, perturbations weakened fitting performance. Two memorized examples do not establish acquisition robustness.

After all fits and per-cell artifacts were saved, summary export failed on NumPy boolean comparison flags. A separate `report54.py` converted those flags to JSON-native values and reconstructed the aggregate from saved metrics. Training sources and models remained unchanged. Numerical verification then passed before a curve-label comma unsupported by the bitmap font interrupted rendering. Removing that punctuation and rerendering fixed the presentation without changing scores. The decision log preserves both failures and repairs.

Verification replayed all 24 complete neural prediction arrays exactly and reconstructed seven control arrays. It independently recomputed per-example MAE and fitting decisions, reconstructed the common forecast normalizer, and checked source hashes, matched initial heads and forecast encoders, frozen weights, all 3,072 optimizer updates, and fitting-only input access. Aggregate export passed participant-identifier checks. Individual images, predictions, weights, and membership remain private.

R54 is closed. It establishes that the pipeline can learn corrections on two fitting examples, but fails the specified individual-fidelity criterion at 512 updates. Updating the encoder did not solve that failure. It does not identify the responsible component, establish optimal fitting duration, support generalization, or justify a temporal JEPA run. No additional fitting followed these results. No independent scientific review has occurred.

## R53 protocol preparation, November 1, 2026

The operator selected completion pretraining followed by frozen-encoder future-FLAIR prediction and requested fixed patient allocation, alignment, scoring, and compute conditions. The [protocol](METHODS.md#r53-missing-flair-pretraining-and-frozen-forecasting) fixed these conditions before execution. Its [public receipt](r53_flair_protocol_inventory.json) records preparation only. At that point no R53 model, training smoke, cohort fit, or predictive assessment had run. The later execution and results follow below.

On `mosaic-gpu`, `scratch/flair-forecast-protocol/inventory.py` reproduced exact R49/R52 patient and completion membership. It checked all selected forecast pairs against the full historical LUMIERE chronology and verified contiguous history, four-sequence availability, file existence, and matching recorded pre-resize grids. Fitting, selection, and scoring contain forty, eleven, and thirteen patients, with 195, 67, and sixty forecast pairs. The history-eligible subsets contain 153, 55, and 45 pairs from thirty, eight, and ten patients. No scoring voxel arrays were opened. Original unseen26 and SAILOR remain excluded.

Ten MRI files from two predetermined fitting examples passed file-hash, finite-value, shape, and nonzero mean/standard-deviation checks. Their displayed current sequences, future FLAIR, and absolute differences were inspected at axial indices forty and 56. Broad anatomical structures are visible in corresponding slices, with residual boundary displacement and substantial appearance differences. This inspection does not establish voxel-perfect anatomical correspondence or classify the changes as biological.

The scoring-field script verified and read the 195 fitting T1 images only. Patient-uniform foreground occupancy at threshold 0.5 followed by one-voxel erosion produced a fixed 133,171-voxel field. There are 79,275 valid seven-cubed SSIM window centers. Six nonempty physical-coordinate regions partition the field exactly. The rendered field was inspected. Its saved hash and actual tensor spacing, approximately 2.010417 by 2.385417 by 2.010417 mm, appear in the receipt. The field uses existing brain extraction, not lesion masks. Historical cohort eligibility still inherits mask-availability restrictions.

The host reported one RTX 5090 with 32,607 MiB, 163 MiB used, and zero-percent GPU utilization during preparation. The twelve-fit plan allows 47,040 successful updates and an estimated 2–4 host/GPU hours. These are workload and budget specifications, not completed training or measured R53 throughput. Runtime and memory must be checked by the real-data smoke before launch. The private membership, image checks, and scoring field are under `/home/aditya/experiments/world-model-r53-20261101/protocol-01/`.

### R53 execution after operator authorization

The operator subsequently requested execution of the fixed study. The first cache-preparation attempt stopped before image work because the R38 environment does not directly install MONAI. Adding the existing vendored MONAI and einops path used by earlier BrainIAC studies resolved the import. The failed `prepared-01` record remains preserved. `prepared-02` cached official tokens for 283 observed fitting and selection visits, fitted completion normalization on 195 fitting visits, and fitted forecast normalization on 210 unique fitting-history visits. Target-only visits supplied no encoder normalization statistics. Scoring images remained closed.

The real smoke passed two updates for adapted and frozen completion, plus the history and latest-only forecast heads. Literal MAE, RMSE, gradient, SSIM, patient-weighting, and positive/negative decision checks passed. Intended trainable weights changed, frozen encoder weights remained unchanged, and checkpoint predictions replayed exactly with optimizer step counts of two. Completion excluded current FLAIR. Forecasting excluded future sequences and the future interval. Earlier-image replacement changed history predictions while latest-only predictions remained identical.

The smoke's peak allocated GPU memory was 3.152 GiB. Projecting every fixed fit from the slower of its two update measurements, including first-update overhead, and actual selection passes gave 90.194 minutes for fitting and selection. Adding a sixty-minute setup/export/assessment/replay allowance gave 2.503 hours. This is a conservative workload projection, not an observed full-run duration or a tuned speed benchmark. The real smoke panels show blurred two-update completion and forecasts close to persistence. Their inspection establishes runnable outputs, not useful prediction.

All four fixed completion fits finished under `completion-01`, followed by eight frozen-encoder forecast fits under `forecast-01`. Every cell completed eighty epochs and 3,920 updates. Completion used 15,680 updates and forecasting used 31,360, totaling 47,040. Selected completion epochs were 56 and 66 for C, and 64 and 57 for R. Selected forecast epochs were six/five for C-H, one/three for C-L, ten/25 for R-H, and six/nineteen for R-L at seeds 42/43. No duration extension followed early selection.

All twelve selected models were locked before scoring. The completed fit timers, including per-epoch selection and final artifact handling, summed to 18.201 minutes for completion and 11.396 minutes for forecasting. These exclude initial model construction, feature export, assessment, and verification. Core fitting sources and checkpoint hashes matched their launch records. Locked assessment completed under `assessment-01` in 805.963 seconds before rendering. All nine assessment figures were inspected. Full numerical verification passed in 520.874 seconds. These scoped timers do not measure total session wall time.

### R53 assessment found no established forecasting benefit

The [verified aggregate](r53_flair_prediction_metrics.json) records `FULL_CAPABILITY_NOT_ESTABLISHED`. C denotes completion-adapted BrainIAC and R denotes original BrainIAC. H uses observed non-FLAIR MRI history; L uses the latest non-FLAIR MRI. Every forecast receives the same latest-FLAIR residual path. P copies the latest FLAIR, and D adds the fitting-only mean change to that image.

| Recipe | Completion MAE | Future-FLAIR MAE |
|---|---:|---:|
| C | 0.505298 | |
| R | 0.519802 | |
| C-H | | 0.656915 |
| C-L | | 0.656850 |
| R-H | | 0.656754 |
| R-L | | 0.657223 |
| Persistence P | | 0.656362 |
| Mean-change D | | 0.662454 |
| Fitting-mean FLAIR, descriptive control | 0.521707 | 0.578055 |

Completion scores use sixty visits from thirteen scoring patients. Primary forecast scores use 45 history-eligible pairs from ten patients. Each patient has equal weight, and neural scores average the two seed scores rather than ensemble images. MAE is in clipped, standardized image-intensity units within the fixed fitting-T1 field.

| Prespecified contrast | MAE difference | Family-five 99% patient interval | Relative MAE reduction |
|---|---:|---|---:|
| Completion C minus R | -0.014504 | [-0.021487, -0.009192] | 2.7903% |
| Forecast C-H minus R-H | +0.000161 | [-0.001704, +0.001122] | -0.0245% |
| Forecast C-H minus C-L | +0.000065 | [-0.001422, +0.001069] | -0.0099% |
| Forecast C-H minus P | +0.000553 | [-0.000806, +0.001442] | -0.0842% |
| Forecast C-H minus D | -0.005538 | [-0.006432, -0.004512] | 0.8361% |

All five superiority criteria failed. Completion improved in both seeds, with an adjusted interval below zero, but missed the fixed five-percent useful-gain margin. Forecast pretraining and history differences changed direction across seeds. C-H was worse than persistence in both seeds. Its advantage over mean-change D was below one percent and failed the useful-gain margin.

All four completion fitting-readiness checks passed. All eight selected forecast models failed the required five-percent fitting MAE reduction over the better of P and D. Their fitting MAEs ranged from 0.614233 to 0.617607 against a control MAE of 0.617319. The regional guard passed, with maximum C-H worsening versus persistence about 0.001118 against the 0.02 allowance. Passing that guard does not establish forecasting accuracy.

Completion C had SSIM 0.261707 versus R's 0.232632, but the fitting-mean image had SSIM 0.286356 and RMSE 0.745719, better than either neural completion recipe on those measures. The inspected completion panels show coarse blurred anatomy and substantial residual errors. C-H forecast SSIM was 0.280527 versus persistence's 0.282073; gradient MAEs were 0.244764 and 0.244755. Forecast panels look nearly identical to persistence, including conspicuous future appearance changes that the models failed to predict. The selection curves generally failed to improve with continued forecast fitting.

The descriptive fitting-mean future image beat persistence on MAE, 0.578055 versus 0.656362, while having worse SSIM, 0.235781 versus 0.282073. Resolution degradation reduced C-H MAE to 0.569831, but matched degraded persistence was still better at 0.569011. Intensity perturbation gave C-H MAE 0.637109 versus matched persistence's 0.635412. These controls do not support a learned temporal or robustness advantage. Smoothing and population-average appearance can improve this image-error endpoint without predicting individual change.

Zero-token completion MAE worsened to 0.614688 for C and 0.636452 for R, supporting image dependence despite blurred outputs. C-H zero-token MAE was 0.658607, close to its identity score. All recorded encoder channels had fitting variation above the standard-deviation floor, which does not establish useful predictive content. The aggregate retains all-pair, single-observation, same-week, positive-week, regional, and nuisance companions as descriptive endpoints.

Numerical verification replayed ninety complete prediction archives exactly, including 72 neural archives and eighteen controls. It reconstructed actual encoder tokens, fitting-only normalizers, mean-image and mean-change controls, and the brain and SSIM fields. All frozen encoder hashes, matched initial heads, sampling orders, selected epochs, and optimizer counts passed. An independent calculation path reproduced the five primary patient MAE contrasts and bootstrap intervals. The attached verification receipt confirms all 47,040 updates. Individual images, predictions, weights, and participant inventories remain private.

R53 is closed without an established forecasting benefit. The failed forecast fitting checks limit this result to the tested recipe. They do not establish that earlier MRI is intrinsically uninformative. These are reused development patients with standardized appearance targets, two seeds, and unverified separation from foundation-model pretraining patients. No independent scientific review has occurred. The study establishes neither biological progression prediction nor clinical utility or calibrated uncertainty. No rescue fitting, additional seeds, duration extension, or joint-JEPA training followed the assessment.

## R52: single-visit spatial-token encoder comparison, November 1, 2026

The operator approved one matched BrainIAC-versus-CNN comparison after evaluating the proposed common token interface. The [fixed protocol](METHODS.md#r52-matched-spatial-token-encoder-protocol) evaluates anatomical information in the exact tokens available to a later temporal model. It does not reopen R49 or the clinical comparisons and trains no temporal JEPA model. The new artifact root is `/home/aditya/experiments/world-model-r52-20261101/` on `mosaic-gpu`.

Preparation reproduced R49's patient allocation and reconstructed fitting-only statistics without opening scoring arrays. All 195 fitting visits and 67 selection visits have four sequences present. Natural missing-sequence handling is therefore untested. Both architectures passed two real effective-batch-four updates, encoder-weight changes, finite gradients from exported tokens to MRI, token-only-head equivalence, and exact token, prediction, and optimizer reload. Fresh readouts passed two real updates and exact reload while encoder tensors stayed fixed without gradients. Literal scoring, empty-target, token-order, normalization, decision, and privacy fixtures passed. Real smoke images were inspected; two updates do not establish useful anatomical predictions.

The shared interface is 216 spatial tokens of width 64 per visit. BrainIAC has 88,537,792 encoder parameters including sequence fusion and 88,607,302 total trainable state-model parameters. The compact CNN has 446,400 encoder parameters and 515,910 total trainable state-model parameters. This difference is part of a practical recipe comparison, not an equal-pretraining or architecture-only effect. Both training heads have 69,510 parameters and identical per-seed initialization.

Smoke-01 mistakenly counted trainable parameters after freezing the encoder for its readout check, reporting only the remaining training heads. Smoke-02 corrected the count placement and repeated the actual checks; the first receipt remains preserved. Warm-update and selection timing from Smoke-02 projected about 26 minutes for all four eighty-epoch state fits. That estimate excludes cold-cache I/O, checkpoint saving, fresh readouts, assessment, and verification. It is not an end-to-end runtime or a measured comparative speedup. The original planning allowance was 2–4 host/GPU hours on one 32-GB RTX 5090.

All four state fits, four fresh frozen-token readout fits, locked assessment, and forty-eight-archive replay completed. The [aggregate](r52_spatial_token_encoder_metrics.json) records `TOKEN_REPLACEMENT_NOT_ESTABLISHED`. All eight fitting-readiness gates passed, but the CNN did not establish the required fresh-readout Dice gain. No independent scientific review has occurred.

### R52 completed without an established token replacement

Both state models trained for eighty epochs per seed, with 3,920 successful updates per cell and 15,680 total. Selected BrainIAC state epochs were 48 and 61; selected CNN epochs were twelve and 27. Fresh readouts also trained for eighty epochs each, with 1,040 updates per cell and 4,160 total. Their selected epochs were 76 and four for BrainIAC and 76 and eighty for CNN. All selected state models and readouts exceeded the declared fitting Dice `0.05` gate. State fitting Dice ranged from 0.319218 to 0.496937; fresh-readout fitting Dice ranged from 0.063593 to 0.077457. The latter is weak absolute fitting despite passing a permissive sanity threshold.

| Encoder recipe | Scoring Dice, trained nonlinear head | Scoring Dice, fresh linear readout, primary | Fresh-readout native-volume MAE | Fresh-readout empty false-foreground rate |
|---|---:|---:|---:|---:|
| Pretrained BrainIAC | 0.191584 | 0.066598 | 1.206147 | 0.987500 |
| Compact CNN | 0.227884 | 0.066748 | 1.107062 | 1.000000 |

Scores average patient scores across the two fitted seeds, not ensemble predictions. Dice and volume MAE use all thirteen scoring patients and sixty visits. Empty-compartment false-foreground rate uses eight eligible patients. Volume error is in native `log1p(mm3)` units. The common token grid and feature width do not equate parameter count, pretraining, receptive fields, or optimization adequacy.

| Primary fresh-readout contrast, CNN minus BrainIAC | Mean difference | Family-three 98.333333% patient interval | Seed 42 / 43 differences |
|---|---:|---|---|
| Nonempty-compartment Dice, higher favors CNN | +0.000150 | [-0.016150, +0.018829] | -0.010064 / +0.010365 |
| Native-volume MAE, lower favors CNN | -0.099085 | [-0.222332, +0.023650] | -0.020410 / -0.177759 |
| Empty false-foreground rate, lower favors CNN | +0.012500 | [0.000000, +0.050000] | 0.000000 / +0.025000 |

The primary Dice difference was effectively flat, changed direction across seeds, and did not reach the declared `0.03` useful-gain threshold. Its adjusted interval crossed zero. Volume and empty-compartment relative-retention gates passed, but neither makes the readouts accurate: the fresh heads predicted foreground in nearly every empty compartment. The rounded empty-rate upper bound lies exactly on the `+0.05` allowance. Passing that relative margin is not evidence of good specificity. Volume improvement remained uncertain because its interval crossed zero.

The nonlinear training-head Dice point gain was larger, at +0.036300, with positive differences in both seeds. Its family-three companion interval was [-0.000834, +0.084453]. This is a secondary endpoint with a different decoder, not an accepted replacement for the failed fresh-readout primary. Both heads consumed only exported tokens, so the difference is not a skip-feature leakage artifact. It does show that the conclusion depends on readout capacity; it does not prove the encoders contain no useful spatial information.

Fitting-only median-volume and mean-volume constants scored MAE 1.318868 and 1.295181. The fitting-mean mask was empty at threshold 0.5 and matched the empty-mask control's near-zero nonempty Dice. Both neural recipes improved these point controls, but that does not validate their absolute local anatomy or establish the declared CNN advantage. All-empty targets remained excluded from primary Dice rather than receiving perfect scores.

The fixed real scoring overlays show broad foreground from the fresh linear heads compared with the targets. The nonlinear heads localize more closely in those examples, with substantial compartment errors remaining. All three assessment figures were inspected. The initial curve render placed adjacent row labels too close together; only layout was changed and figures were regenerated from the unchanged aggregate. State training losses fell while selection scores plateaued or worsened. Several fresh-readout curves were still improving late, and CNN seed 43 selected the final epoch. The fixed schedule therefore does not establish optimal or converged readout fitting. No duration extension or new head was selected from these outcomes.

Identity versus intensity fresh-readout Dice was nearly unchanged. Mean resolution-companion Dice was 0.049305 for BrainIAC and 0.068767 for CNN. Mean zero-image Dice fell to 0.004000 and 0.003507, respectively, versus approximately 0.0666 with actual images. This supports image dependence in the tested models, not general scanner invariance or proof against all shortcuts. Resolution perturbation removes genuine image detail, and synthetic companions remain descriptive rather than model-selection endpoints.

State fitting and per-epoch selection timers summed to 25.893 minutes across the four cells. Fresh-readout fitting and selection timers summed to 11.087 seconds, excluding encoder token export, initial construction, and final artifact writes. Locked assessment took 46.569 seconds before figure rendering. These are scoped completed-work timings, not full-session wall time or a tuned architectural speed benchmark. Numerical verification checked all 3,920 state and 1,040 readout optimizer steps per cell. It replayed forty-eight complete prediction archives exactly, including actual encoder token arrays and both heads, reconstructed fitting-only target and token statistics and constants, reproduced independent patient scores and intervals, and verified selected epochs, identical heads, frozen encoder hashes, sampling orders, and unchanged training and assessment code. Individual arrays, weights, images, and participant inventories remain private.

R52 is closed without a supported replacement. BrainIAC remains the reference under the prespecified stop rule, not a demonstrated best encoder. No architecture, seed, threshold, head, normalization, or duration rescue follows. The operator separately discussed frozen, state-only-adapted, and jointly JEPA-adapted temporal training; none of those follow-on experiments has been launched or authorized by this closure. Reused patients, automated targets, weak absolute fresh-readout accuracy, two seeds, and unaudited foundation-pretraining overlap limit the result.

## R50 and R51: clinical comparator and JEPA attribution, October 31, 2026

The operator requested both remaining priority-1 studies after R49. All eight fits, locked clinical assessment, and numerical replay completed on `mosaic-gpu`. The [aggregate](r50_r51_clinical_metrics.json) records both declared benefit criteria as unmet. R50 did not establish a pretrained direct-classifier advantage over the adapted DenseNet264. R51 found no temporal-specific clinical benefit; proper JEPA had worse mean primary loss than direct supervision. These are outcomes under the [fixed observed-response protocol](METHODS.md#r50-and-r51-observed-response-comparator-protocol), not independent clinical validation or prospective forecasts. The run is `/home/aditya/experiments/world-model-r50-r51-20261031/fit-01`, with assessment in sibling `assessment-01`.

Metadata eligibility retained 221 pairs from 47 patients. It excluded fifteen ambiguous or missing annotations, 74 non-four-class annotations, and twelve marked under-three-month targets. No required sequence was missing in the retained prepared-pair pool. The original anatomy selection split had no PR example on this task. The new source-only label-stratified allocation accepted its fourth deterministic candidate; original scoring patients never moved.

| Role | Pairs | Patients | PD / SD / PR / CR pairs |
|---|---:|---:|---|
| Fitting | 136 | 29 | 96 / 24 / 7 / 9 |
| Checkpoint selection and temperature | 46 | 8 | 24 / 12 / 3 / 7 |
| Locked scoring | 39 | 10 | 32 / 3 / 3 / 1 |

Each arm passed two real physical-batch-four updates, finite-gradient checks, changed encoder tensors, exact model prediction replay, and optimizer reload. Direct, proper, and shuffled BrainIAC had identical initial online hashes. EMA teachers changed without gradients. Predictor gradients were nonzero for earlier MRI and exactly zero for later MRI. Clinical logits depended on both observed images in all four arms. No scoring MRI array was opened by preparation or smoke. DenseNet has 40,366,404 trainable parameters; each BrainIAC arm has 92,492,036.

The smoke projected approximately 100 minutes of training for eight maximum-length fits, excluding selection, saving, assessment, and verification. This was a warm-update workload projection, not measured end-to-end runtime. All cells have the same declared 100-epoch maximum and early-stopping opportunities. Fitting readiness and the known-prior logit correction were fixed after smoke and before cohort fitting. They were not selected from scoring errors.

Assessment fixtures passed for four-class and binary proper scores, family-three intervals, calibrated probabilities, and export privacy. The host lacked Matplotlib and Pillow, so a dependency-free PNG renderer replaced plotting imports. Literal PNG checksum, pixel, and dimension fixtures passed, and both synthetic figures were visually inspected. This repair changed neither training nor scoring formulas. Trial instruments, receipts, figures, and the decision trail remain in `scratch/clinical-comparators/` and the approved-host root.

Only one scoring CR example is available. Even favorable comparisons could support only this reused developmental recipe, not comprehensive four-class clinical validation, a prospective forecast, or patient separation from the foundation pretraining corpus. No independent scientific review has occurred. Preserve earlier negative and withdrawn results.

### Completed clinical comparisons

Every cell finished its fixed early-stopping rule. DenseNet selected epochs eight and sixteen and stopped at twenty and 26 for seeds 42 and 43. Direct BrainIAC selected four and fifteen and stopped at twenty and 25. Proper JEPA selected four and six; shuffled JEPA selected five and ten. All four JEPA fits stopped at twenty. Across the matrix, 171 epochs produced 5,814 successful optimizer updates and 23,256 sampled pair exposures. No cell reached the 100-epoch maximum, and no duration extension followed.

The primary score is calibrated four-class log loss, averaged within patient and then across the ten patients. The table averages scores over seeds, not ensemble probabilities. Raw balanced accuracy uses uncorrected argmax logits; calibrated balanced accuracy uses the fixed prior-corrected and temperature-scaled probabilities. Their different class behavior must not be hidden.

| MRI-only arm | Primary log loss | Raw balanced accuracy | Calibrated balanced accuracy | Calibrated macro-F1 |
|---|---:|---:|---:|---:|
| Adapted DenseNet264 | 0.734232 | 0.607479 | 0.239744 | 0.218871 |
| Direct pretrained BrainIAC | 0.785354 | 0.363462 | 0.250000 | 0.224138 |
| Proper temporal JEPA | 0.827426 | 0.250000 | 0.250000 | 0.224138 |
| Cross-patient shuffled JEPA | 0.816729 | 0.358696 | 0.250000 | 0.224138 |

| Primary contrast, lower favors candidate | Mean loss difference | Family-three 98.3333% patient interval | Seed 42 / 43 differences |
|---|---:|---|---|
| Direct minus DenseNet, R50 | +0.051122 | [-0.037958, +0.159750] | +0.093867 / +0.008377 |
| Proper JEPA minus direct, R51 | +0.042072 | [+0.008912, +0.071766] | -0.012284 / +0.096428 |
| Proper JEPA minus shuffled, R51 | +0.010697 | [-0.016573, +0.034524] | +0.009947 / +0.011448 |

R50's candidate was worse by point loss in both seeds, and its adjusted interval crossed zero. This fails the declared advantage criterion but does not establish equivalence or a definitive DenseNet advantage. Proper JEPA increased mean loss by 5.36 percent relative to direct supervision. Its adjusted patient interval excluded zero on the unfavorable side, conditional on these fitted seeds. One seed favored JEPA against direct and the other did not, so the harm is not a seed-consistent general claim. Proper JEPA was also worse than shuffled in both seeds by point loss, with a zero-crossing adjusted interval. It therefore failed both required attribution comparisons.

The fitting-only prevalence control had log loss 0.708800, lower by point score than every neural arm. The supplied-previous-rating transition control scored 0.748203. The explicitly privileged observed-volume/change/previous-rating logistic control scored 0.733300. These companions share scoring rows but not the neural MRI-only information budget. Their descriptive intervals do not create additional accepted primary tests. No neural clinical gain over the simple prevalence control was established.

### Readiness, clinical limitations, and closure

DenseNet passed both readiness requirements in both seeds. Its selected raw fitting models predicted all four classes, with balanced accuracies 0.445754 and 0.421228. This was not another collapsed CNN comparator. Proper JEPA teachers also passed. Their effective ranks were 7.885597 and 8.590916, with normalized feature standard deviations 0.002942 and 0.002791. Noncollapsed targets did not produce a useful clinical objective effect.

The selected proper-JEPA classifiers themselves predicted only one raw fitting class in each seed, with fitting balanced accuracy 0.25. Direct seed 42 had the same fitting limitation; direct seed 43 predicted four classes with fitting balanced accuracy 0.340427. Teacher readiness therefore must not be mistaken for classifier readiness or useful temporal prediction. On the fixed fitting diagnostic, proper-JEPA feature prediction errors were 0.016062 and 0.012942, versus lower same-teacher persistence errors of 0.009791 and 0.009051. Falling auxiliary error did not beat this feature-space persistence control.

The real probability/confusion and learning-curve figures were visually inspected. After prior correction and temperature calibration, all BrainIAC scoring predictions chose PD. DenseNet seed 42 also chose PD throughout, while seed 43 made some PR errors. None of the calibrated neural arms recalled a minority scoring class. The raw DenseNet balanced accuracy of 0.607479 shows a real operating-point tradeoff, not comprehensive clinical performance from the primary proper score. Calibration improved natural-prevalence loss but did not establish useful minority-class decisions. Temperatures ranged from 1.603957 to 1.962758 and were all inside the declared bounds.

Every selected checkpoint preceded its final epoch. Training CE fell while final selection CE exceeded the selected minimum in every cell. These curves support stopping at the fixed selection rule rather than extending the same scored recipe. Training and selection CE use different sampling and augmentation, so their absolute levels are not directly interchangeable. The intensity companion left primary losses nearly unchanged. Resolution changed mean losses to 0.739903, 0.794500, 0.817920, and 0.787567 in table order. These are descriptive synthetic stresses, not scanner-invariance evidence or another basis for selection.

Verification replayed all forty complete fitting, selection, identity-scoring, intensity, and resolution neural arrays exactly. It reconstructed all three fitting-only controls, selection temperatures, source eligibility and label joins, patient splits, shuffled-target ownership, seeded draws, checkpoint minima, early stopping, and optimizer counts. Separate patient-score and bootstrap implementations reproduced log loss, confusion matrices, and primary intervals. Export privacy checks passed. This is numerical verification by the implementing workflow, not independent scientific review.

R50 and R51 are closed at their fixed negative or inconclusive benefit outcomes. No additional seed, target, architecture, threshold, coefficient, or rescue training was selected. R49's scoped anatomy-pretraining gain remains valid and separate. Fresh patients, clinical target support, pretraining-corpus separation, and independent scientific review remain unresolved. Only the aggregate is public; participant inventories, predictions, figures with individual rows, and weights remain private.

### R50-v2 and R51-v2: methodology-repaired clinical comparisons, November 1, 2026

After auditing R50 and R51, the operator requested carrying out all four methodology repairs: preserving BrainIAC `zscore_nonzero` with independent per-visit/per-sequence foreground noise, retaining all 216 (`6×6×6`) spatial patch tokens instead of slicing corner token `[:, 0]`, applying the exact effective-training-prior offset with normalized class weights and prior-corrected selection, and predicting spatially centered 216-patch tokens in JEPA. All four arms passed the two-update real-data smoke (`smoke-01`), literal assessment fixtures (`fixtures-01`), eight-cell training (`fit-01`), locked assessment (`assessment-01`), and exact 40-array verification (`verify.py`) on `mosaic-gpu`. See the [repaired protocol](METHODS.md#r50-v2-and-r51-v2-methodology-repaired-clinical-comparator-and-jepa-attribution) and [aggregate](r50_r51_v2_clinical_metrics.json).

| MRI-only arm (`R50-v2` / `R51-v2`) | Selected / final epochs (seed 42 / 43) | Prior-corrected unscaled loss | Primary calibrated log loss | Raw / calibrated balanced accuracy | Classifier fitting ready |
|---|---|---:|---:|---:|---|
| Adapted DenseNet264 | 1/20, 6/20 | 0.750108 | 0.800122 | 0.251923 / 0.280662 | True (3 & 4 classes) |
| Direct spatial-token BrainIAC | 2/20, 5/20 | 0.800512 | 0.852571 | 0.250000 / 0.250000 | False (1 class) |
| Proper spatially centered JEPA | 11/21, 10/20 | 0.779454 | 0.837773 | 0.250000 / 0.250000 | False (1 class) |
| Cross-patient shuffled JEPA | 6/20, 14/24 | 0.798660 | 0.851413 | 0.250000 / 0.250000 | False (1 class) |

| Primary contrast (`R50-v2` / `R51-v2`), lower favors candidate | Mean loss difference | Family-three 98.3333% patient interval | Seed 42 / 43 differences | Relative loss reduction |
|---|---:|---|---|---:|
| Direct minus DenseNet (`R50-v2`) | +0.052449 | [-0.027291, +0.145539] | +0.072184 / +0.032713 | -6.56% |
| Proper JEPA minus direct (`R51-v2`) | -0.014798 | [-0.042185, +0.004355] | -0.018819 / -0.010777 | +1.74% |
| Proper JEPA minus shuffled (`R51-v2`) | -0.013640 | [-0.039835, +0.005202] | -0.022702 / -0.004578 | +1.60% |

The four repairs resolved the technical confounds identified in the audit:
- **Spatially centered 216-token JEPA (`Fix 4`):** Teacher persistence cosine distance rose from `0.009` in `v1` to `0.271` initially (`0.1287` and `0.1389` at the selected `jepa` checkpoints), and training JEPA loss dropped from `~0.51` to `~0.11` with effective teacher ranks `5.20` and `5.29`. Unlike `v1`, `jepa` now had lower (better) point log loss than both `direct` (`-0.014798`) and `shuffled` (`-0.013640`) in **both seeds**. However, the reduction was only `1.74%` (`< 5%` target), the family-three intervals crossed zero (`[-0.042185, +0.004355]` and `[-0.039835, +0.005202]`), and the classifiers failed the fitting-readiness gate.
- **Unfrozen ViT representation collapse under `1e-4` AdamW (`Fix 2` + `Fix 3` diagnostic):** Requiring `classifier_fitting_readiness` across all arms exposed the true limiter. While `densenet` passed (`3` and `4` fitting classes; `densenet-43` achieved `0.3113` calibrated scoring balanced accuracy and `0.2934` macro-F1), **all six selected BrainIAC checkpoints failed classifier fitting readiness** (`predicted_fitting_classes_raw = 1`, `fitting_BA_raw = 0.25`, with cross-sample logit standard deviation `0.0005–0.012`). Unlike `R49` (which had dense `24³` voxel mask + continuous volume supervision), fine-tuning all 93.9M ViT parameters end-to-end with AdamW `lr = 1e-4` and `weight_decay = 0.01` on only 136 noisy 4-class pairs drives the two-scan difference `late_map - early_map` toward zero (`smap` cross-sample standard deviation `0.0022` in `direct-42` vs `0.1518` in `jepa-42` at epoch 20), so the linear head outputs a near-constant prior vector.
- **Prior offset vs scalar temperature (`Fix 3`):** The exact effective-prior offset alone (`prior_corrected_unscaled`) achieved lower scoring log loss (`0.7501` DenseNet, `0.8005` Direct, `0.7795` JEPA, `0.7987` Shuffled) than adding selection-fitted scalar temperature (`T = 1.35–2.12`), because the 8-patient selection set has a flatter minority ratio (`24/12/3/7`, 52% PD) than the 10-patient scoring set (`32/3/3/1`, 82% PD).

Both `R50-v2` and `R51-v2` therefore remain **`CRITERION_NOT_MET`** (`40/40` prediction arrays replayed exactly).

### R50-v3 and R51-v3: limited-adaptation clinical comparison

November 1, 2026. The operator selected continuation after the source-only exploratory probes. V3 tests frozen-base rank-eight LoRA with fitting-only token standardization and a smaller pooled head under the [fixed protocol](METHODS.md#r50-v3-and-r51-v3-limited-adaptation-clinical-protocol). It also changes learning rate and imposes fitting readiness during checkpoint selection. These bundled changes cannot identify the unique cause of v2's near-constant classifiers. The [dated audit correction](NOTES_AND_INSIGHTS.md#november-1-corrections-to-the-methodology-audit) supersedes earlier causal and prior-offset claims without altering v1/v2 measurements.

Preparation reproduced the original clinical rows exactly and fitted token statistics from 175 unique fitting visits, opening no selection or scoring images. All four arms passed two real physical-batch-four updates, finite checks, changed trainable tensors, exact prediction replay, and optimizer reload. BrainIAC base tensors stayed bitwise unchanged while LoRA tensors changed. Teacher tensors updated without gradients. BrainIAC arms had identical initial online hashes. Predictor gradients reached earlier MRI only; classifier gradients reached both visits and an interior patch.

Each BrainIAC arm has 1,725,514 trainable parameters, versus DenseNet's 40,366,404. Literal zero-output-LoRA, prior-offset-equivalence, readiness-selection, scorer, privacy, and PNG fixtures passed. The noise test used identical images and verified distinct channel residuals and preserved background, rather than relying on differences between real early and late scans. The smoke projected about 86 minutes of training for eight 100-epoch fits, excluding per-epoch fitting/selection scoring, saving, and final assessment. This is a workload projection, not measured end-to-end runtime.

All eight fits, locked assessment, and numerical replay completed under `/home/aditya/experiments/world-model-r50-r51-v3-20261101/` on `mosaic-gpu`. The [v3 aggregate](r50_r51_v3_clinical_metrics.json) records both primary benefit criteria as unmet, while every classifier and the proper teachers passed readiness. Original v1/v2 artifacts remain unchanged; no independent scientific review has occurred.

#### Readiness recovered, clinical benefit not established

Every cell stopped at twenty epochs under the fixed rule, for 160 epochs, 5,440 successful updates, and 21,760 sampled pair exposures. Selected DenseNet epochs were six and seven. Direct and proper JEPA selected five and one; shuffled JEPA selected five and two for seeds 42 and 43. All eight selected classifiers predicted three raw fitting classes. Fitting balanced accuracy ranged from 0.352653 to 0.608635 in the BrainIAC arms and from 0.415196 to 0.435058 in DenseNet. These are nonconstant fitting models under the declared gate, not demonstrated clinical generalization or convergence.

| MRI-only arm | Offset-only scoring log loss | Primary calibrated log loss | Calibrated balanced accuracy | Calibrated macro-F1 |
|---|---:|---:|---:|---:|
| Adapted DenseNet264 | 0.934643 | 0.969898 | 0.396042 | 0.325185 |
| Direct BrainIAC with LoRA | 0.697521 | 0.808499 | 0.250000 | 0.224138 |
| Proper token JEPA with LoRA | 0.701598 | 0.811174 | 0.291555 | 0.284100 |
| Shuffled-target JEPA with LoRA | 0.707216 | 0.817777 | 0.250000 | 0.224138 |

Scores average each patient's scores across the two fitted seeds, not ensemble probabilities. The ten scoring patients and one scoring CR example are unchanged. The primary remains calibrated log loss even though offset-only losses are lower.

| Primary contrast, lower favors candidate | Mean loss difference | Family-three 98.3333% patient interval | Seed 42 / 43 differences |
|---|---:|---|---|
| Direct minus DenseNet, R50-v3 | -0.161400 | [-0.305401, +0.008607] | -0.335594 / +0.012795 |
| Proper JEPA minus direct, R51-v3 | +0.002676 | [-0.002237, +0.008289] | +0.004869 / +0.000483 |
| Proper JEPA minus shuffled, R51-v3 | -0.006602 | [-0.020553, +0.006327] | -0.000599 / -0.012605 |

Direct had a 16.64-percent lower point loss than DenseNet, but failed the same-direction seed requirement and its adjusted interval crossed zero. The ordinary 95-percent interval excluded zero, but it is not the declared primary multiplicity rule. The point gain was driven by DenseNet seed 42 scoring 1.161681 versus direct at 0.826086. At seed 43, DenseNet scored 0.778116 versus direct at 0.790911. This does not establish robust superiority of the pretrained recipe.

Proper JEPA was slightly worse than direct in both seeds by primary loss. Its mean relative loss reduction was negative, at -0.33 percent, and its adjusted interval crossed zero. Proper beat shuffled by point loss in both seeds, but that interval also crossed zero. R51-v3 therefore did not establish the required direct or temporal-correspondence benefits. The offset-only companions did not change the conclusion: proper-minus-direct was +0.004077 with adjusted interval [-0.004342, +0.012954]. Do not select a different calibration or endpoint from these scoring outcomes.

The fitting-prevalence constant scored 0.708800, the supplied-previous-rating transition control 0.748203, and the privileged volume/change/previous-rating logistic control 0.733300. Every neural primary loss was worse by point score than these controls. These descriptive comparisons do not supply extra accepted hypothesis tests. Direct's offset-only point loss of 0.697521 is not an accepted replacement for its calibrated primary result.

#### Predictions, curves, and closure

The fixed probability/confusion and learning-curve figures were visually inspected. Direct and shuffled predicted PD for every calibrated scoring row in both seeds. Proper seed 42 recalled part of PR, while proper seed 43 remained PD-only; neither proper model recalled SD or CR. DenseNet seed 42 recalled minority classes, including the sole CR example, but made substantial PD errors. DenseNet seed 43 recalled no minority class. The apparent clinical operating points remain unstable and cannot support comprehensive four-class validation.

Proper-teacher effective ranks were 11.605197 and 11.626280, with normalized feature SD about 0.03545. Selected proper feature prediction errors were 0.572054 and 0.774861, worse than same-teacher persistence errors 0.470040 and 0.473507 on the fixed fitting diagnostic. Falling training auxiliary loss therefore did not establish a superior temporal predictor. The v3 loss uses fitting-population standardization rather than v2's per-scan spatial centering, so the latent error magnitudes are not directly comparable across revisions.

All selected checkpoints preceded their final epochs. Training CE fell while selection loss worsened in every cell. Readiness improved, but the learning curves do not justify an automatic extension or establish why v2 failed. The bundle changes head, normalization, adaptation, rate, and selection together. The official frozen base contains 88,340,736 parameters; earlier dialogue's 91.8M base count was inaccurate. Frozen base invariance does not prevent LoRA-adapted outputs or the head from changing, and does not isolate a causal freezing effect.

Selection-fitted temperatures ranged from 1.289077 to 1.863357. The calibrated mean losses exceeded offset-only losses in every arm. Positive temperature leaves offset-corrected argmax decisions unchanged; this difference concerns probability confidence. Mean intensity-companion losses were 0.979268, 0.806001, 0.809733, and 0.815569 in table order. Resolution-companion losses were 1.065236, 0.809146, 0.811961, and 0.814028. These remain descriptive synthetic sensitivities, not real scanner robustness.

Verification replayed all forty complete neural fitting, selection, identity-scoring, intensity, and resolution arrays exactly. It reconstructed token statistics from all 175 fitting visits, checked online and teacher frozen bases against official weights, reproduced temperatures and controls, and checked readiness-qualified selection, first-eligible stopping, seeded orders, and optimizer counts. Separate patient-score and bootstrap implementations reproduced raw, offset-only, and calibrated losses/confusions and both primary and offset-only contrast intervals. Privacy checks passed. This was numerical verification within the implementing workflow, not independent scientific review.

V3 is closed at improved fitting readiness with neither primary clinical benefit criterion met. No additional seed, duration, calibration choice, rate, or architecture search follows this result. The earlier versions and R49's separate anatomy-pretraining finding remain intact. All scoring patients were already studied and influenced the research direction, so even a favorable interval would remain adaptive development evidence. Individual predictions, participant inventories, weights, and figures with individual rows stay private.

## R49: patient-held-out anatomy pretraining comparison, October 31, 2026

The operator selected a held-out comparison after the R48 learning diagnosis. All four matched fits completed eighty epochs each, followed by the locked assessment and numerical replay. The [aggregate](r49_pretraining_state_metrics.json) reports **`SUPPORTED_WITHIN_FIXED_DEVELOPMENT_RECIPE`**. Official BrainIAC initialization improved nonempty-compartment mask Dice over the identical random ViT under this training recipe, while meeting the declared native-volume retention margin. This is a narrow development finding, not independent validation or evidence for JEPA dynamics. The [protocol](METHODS.md#r49-patient-held-out-anatomy-pretraining-protocol) was fixed before fitting.

Fitting used 195 visits from forty patients, checkpoint selection used 67 visits from eleven other patients, and scoring used sixty visits from thirteen other patients. All came from the original source52 and validation13 pool. Original unseen26 and SAILOR were excluded. The scoring patients have been studied historically, so they are not untouched. Volume normalization, foreground weights, and constant controls used fitting patients only. All four selected checkpoints were locked before scoring images were opened.

### Matched pretraining result

Both arms used identical architecture, dense state supervision, patient draws within each seed, non-backbone initialization, and selection opportunities. Every cell completed 3,920 successful optimizer updates, giving 15,680 updates and 62,400 sampled visit exposures across the four fits. The two seeds were 42 and 43. Primary Dice excludes empty target compartments before patient-uniform averaging. Native MAE is measured in `log1p(mm3)`. Seed means below average scores, not predictions.

| Initialization | Selected epoch, seed 42 / 43 | Held-out nonempty Dice | Held-out native log-volume MAE |
|---|---|---:|---:|
| Official pretrained BrainIAC | 44 / 22 | 0.198176 | 1.114082 |
| Identical random ViT | 17 / 12 | 0.158143 | 1.250673 |
| Fitting-only median volume and empty mask | Not applicable | effectively zero | 1.318868 |
| Fitting-only mean volume and empty mask | Not applicable | effectively zero | 1.295181 |

The fitting-only mean-occupancy mask at threshold 0.5 was also empty on this cohort, so it had the same Dice as the empty-mask control. Unlike R48's two-example union, it was fitted without scoring targets.

Pretrained minus random Dice was `+0.040033`, with two-test-family 97.5% paired-patient interval `[+0.019563, +0.059459]`. Seed-specific gains were `+0.044058` and `+0.036008`. The point gain exceeded the declared +0.03 target, both seeds favored pretraining, and the adjusted interval excluded zero. The interval still includes gains smaller than +0.03, so the minimum effect itself is not established as a confidence bound.

Pretrained minus random volume MAE was `-0.136590`, with adjusted interval `[-0.295532, +0.036304]`. The upper bound was below the +0.05 retention allowance, satisfying that criterion. The interval crossed zero, so a native-volume improvement was not established. All four prespecified support conditions were met. Point advantages over constant volume controls remain descriptive rather than additional accepted hypothesis tests.

### Absolute quality and generalization limits

The fixed held-out renders were inspected. Masks remained broad and did not establish faithful compartment segmentation. Across seeds, scoring Dice was approximately 0.198 for pretrained and 0.158 for random initialization, despite selected-checkpoint fitting Dice of 0.353 and 0.273. The scoring set had seven empty core targets, eight empty enhancing targets, and one empty edema target. Pretrained fits predicted false foreground in 3/3/1 and 3/5/0 of those compartments at seeds 42 and 43. Empty-target failures therefore remain present and are not hidden by the nonempty primary endpoint.

Longer training did not steadily improve selection performance. Selected epochs were 44 and 22 for pretrained, versus 17 and 12 for random initialization. Across all four cells, training objective fell substantially while selection objective was worse at epoch eighty than at the selected checkpoint. This is consistent with overfitting rather than evidence that the final epoch simply needed another extension. It does not establish an optimal training schedule or isolate every source of the generalization gap. No continuation followed.

The synthetic resolution companion reduced pretrained scoring Dice from 0.198176 to 0.176406. Random Dice changed from 0.158143 to 0.165247. Intensity-companion Dice was 0.198111 and 0.158372, respectively. The transform restandardizes intensities, largely cancelling the affine intensity change. These descriptive scores do not establish acquisition invariance or a useful robustness advantage.

### Artifact verification and closure

Both real effective-batch-four smokes passed before fitting. Verification replayed twenty complete neural prediction arrays across fitting, selection, scoring, and nuisance conditions exactly. Independent patient-score reconstruction reproduced Dice, volume MAE, and the reported intervals. Fitting-only normalization, foreground weights, median/mean controls, and mean mask were reconstructed. Seeded sample draws, first-maximum checkpoint selection, patient disjointness, and final optimizer counts were checked.

The first public-export attempt was blocked by an overbroad privacy check that matched the prose `Patient-held-out` as an identifier. Numerical verification had completed. The original pre-export receipt was preserved, and the guard was corrected to reject actual identifier patterns and private identifier keys. Literal positive and negative fixtures passed before full replay was repeated successfully. No model, endpoint, patient membership, threshold, or assessment score changed.

Scripts, private models, individual predictions, renders, and decision logs remain under `scratch/pretraining-state-generalization/` and the approved-host R49 directory. Only aggregate evidence is public. No independent scientific review has occurred. The study does not establish pretraining-corpus patient disjointness, biological mask accuracy, clinical utility, future forecasting, or a JEPA-objective advantage. The clinical comparator and objective-attribution stages were pending at R49 closure and were later selected separately as R50 and R51 above. No further training or new comparison was selected by R49's closure itself.

## R48: source-only comparator readiness, October 31, 2026

The operator selected priority 1 from the hypothesis checklist: working pretraining controls, a faithful supervised comparator, and objective attribution. The first bounded unit tested whether pretrained BrainIAC and an identical random ViT could learn direct anatomical state on two source training examples. The [aggregate](r48_control_readiness_metrics.json) records **`LEARNING_SANITY_FAILED`**. This is an instrument result, not a pretraining-superiority or generalization test. The [protocol](METHODS.md#r48-source-only-learning-sanity-protocol) was fixed before fitting.

Both arms passed a two-update real-data smoke with model and optimizer save/reload and exact prediction replay. Each then restarted from its original initialization and completed exactly 128 effective-batch-two updates. Each model had 88,607,302 trainable parameters. Every trainable parameter tensor had a recorded optimizer step count of 128. The model received whole-brain current MRI and sequence-availability flags only. Measured volumes and masks were training targets, not predictor inputs. No validation13, unseen26, or SAILOR image was opened for this test.

| Fitted predictor | Mean training mask Dice | Training native log-volume MAE |
|---|---:|---:|
| Pretrained ViT with direct-state heads | 0.310863 | 0.058051 |
| Identical random ViT with direct-state heads | 0.305852 | 0.057272 |
| Constant mean mask and mean volume fitted to the same two examples | 0.678406 | 0.291395 |

Both arms halved total training loss and met the volume-MAE threshold. Neither met mean hard Dice of at least 0.50 or Dice of at least 0.25 for all six positive example-compartment targets. The images had been selected for positive-compartment learning sanity, not representative prevalence. Each 24-cubed target contained only 20 to 96 positive voxels per compartment. The fixed final masks were nonempty but oversized. Their mean Dice also fell below the fitted constant mean-mask control. The rendered target/pretrained/random panels were visually inspected. No checkpoint or probability threshold was selected after fitting.

Swapping the two image/availability inputs increased pretrained volume MAE from 0.058051 to 0.580954. This is consistent with example-dependent volume fitting, not held-out anatomical accuracy. Both model predictions replayed exactly from their final checkpoints. A separate verification independently recomputed hard Dice, native-volume MAE, and foreground-weighted BCE from the saved arrays. The aggregate preserves per-compartment true positives, false positives, false negatives, foreground weights, and loss contributions. These counts locate the spatial fitting failure but do not identify whether the cause is representation resolution, decoder design, optimization duration, or weighting. There was no extra fit, seed search, or larger comparison after failure.

The accompanying primary-source comparator audit found a premise error in the historical R28/R32 execution proposal. [Matoso et al.](https://arxiv.org/html/2504.18268v2) report their best model as a **3D DenseNet264 without pretraining**, using paired T1, T2, and FLAIR. Their [primary implementation](https://github.com/anamatoso/RANO-classification) constructs that architecture with `pretrained=False`. A pretrained 2D ResNet is therefore not a faithful reproduction of this comparator. The need for a fair supervised comparator remains, but pretrained initialization is a separate hypothesis. Five downloaded source files and their hashes remain in `scratch/priority1-controls/literature/`.

At the original R48 closure, priority 1 was incomplete. This unit did not run a clinical DenseNet comparator, a held-out pretraining comparison, or a same-backbone JEPA attribution experiment. R49 through R51 subsequently completed those bounded comparisons. R48 does not establish biological truth, independent validation, or a cause for R42's empty-mask outputs. All array work and fitting ran on `mosaic-gpu`. Production source and earlier experiment outputs were unchanged. No independent scientific review has occurred.

### R48 diagnostic follow-up: the 128-update test stopped before mask fitting was adequate

October 31, 2026. The operator requested investigation of the failed readiness test. A separate fixed diagnostic preserved the original 128-update checkpoints, optimizer states, arrays, and `LEARNING_SANITY_FAILED` outcome. See the [diagnostic aggregate](r48_learning_investigation_metrics.json) and [diagnostic protocol](METHODS.md#r48-failure-investigation-protocol). This follow-up is not a replacement result or an assessment-cohort rescue.

Replaying the original checkpoints reproduced every saved prediction. Their poor hard Dice was mainly excess foreground. The pretrained arm included all six target compartments' positive voxels but predicted 3.99 to 10.52 times their voxel counts. The random arm missed only one target voxel across the six compartments. The constant mean-mask control equals the union of these two training masks at the inclusive 0.5 threshold. That identity explains its strong two-example fitting score. It is not a held-out or image-conditioned control.

The no-fit diagnostic found nonzero shared-state gradients for all three losses. The soft-Dice loss value was large, but its gradient norm was smaller than BCE's in all four example-arm cases. Local volume-versus-mask gradient cosines ranged from approximately -0.109 to -0.023. These derivatives do not demonstrate a strong antiparallel conflict at that checkpoint, and do not identify a trajectory-wide cause. Foreground-weight correction alone did not recover accurate masks, so no new threshold was adopted. Average-pooled target occupancy located the small lesions in two to thirteen coarse cells per compartment, but does not prove a representational ceiling.

The duration intervention changed only the update count. A two-update continuation smoke per arm first passed finite-update, save/reload, and exact-prediction checks. The fixed diagnostic then restarted from each original step-128 model and optimizer and added 384 updates to reach step 512. Learning rates, objective, input examples, decoder, precision, clipping, and threshold remained unchanged. No checkpoint was selected.

| Arm | Mean training Dice at 128 | Mean training Dice at 512 | Native log-volume MAE at 512 |
|---|---:|---:|---:|
| Pretrained | 0.310863 | 0.593454 | 0.010874 |
| Random | 0.305852 | 0.722342 | 0.014050 |

Both arms met every original learning threshold at step 512. Pretrained mean Dice progressed through 0.379923 at step 256 and 0.502828 at step 384. Random mean Dice reached 0.434138 and 0.623923. This establishes that the fixed 128-update fitting failure was not a hard inability to fit these two source masks. Additional unchanged optimization was sufficient to cross the declared criterion. The original test stopped too early to conclude that the recipe could not learn the masks.

Median background probability fell from 0.105211 to 0.001367 in the pretrained arm and from 0.120853 to 0.000667 in the random arm. False positives fell while target recall remained high. At step 512, swapping the source images reduced Dice to 0.236072 and 0.230520, respectively. Zero-image masks had effectively zero Dice. These observations support image-dependent fitting of the two examples, not generalization.

The final masks were rendered and inspected. The pretrained arm still scored below the fitted constant mean-mask Dice of 0.678406, while the random arm exceeded it. Random training Dice being higher does not establish superior held-out initialization. Neither checkpoint demonstrates convergence, biological accuracy, or a valid medical-pretraining comparison. The result does not explain R42 by itself.

A separate verifier replayed all four original and continued checkpoint prediction arrays exactly, independently recomputed Dice and native-volume MAE, and checked original/continued optimizer step counts of 128/512. Both original checkpoint hashes remained unchanged. No validation13, unseen26, SAILOR, or clinical outcome was used. At this diagnostic's closure, a held-out protocol and the clinical comparisons were still required. R49 subsequently supplied the matched anatomical comparison, and R50/R51 supplied the clinical and JEPA studies. Those later outcomes do not change this two-example diagnostic.

## R47: anatomical-signal robustness and locked transfer, October 31, 2026

The operator selected validation of the simple anatomical forecasting and measured-volume response leads. The fixed [R47 protocol](METHODS.md#r47-anatomical-signal-robustness-protocol) tested incremental information against stronger controls, not a rescue of R41 through R46. The [aggregate](r47_anatomical_signal_metrics.json) reports **`INCREMENTAL_HISTORY_AND_CHANGE_NOT_VALIDATED`**. A current-state forecasting advantage survived within LUMIERE, but neither added history nor observed volume change met the primary criterion, and the anatomical advantage did not transfer to SAILOR without adaptation.

All new LUMIERE fitting, selection, calibration, and evaluation excluded the reused unseen26 assessment patients. The original source52 plus validation13 patients supplied 322 forecast pairs from 64 eligible patients. Primary history assessment used 253 pairs from 48 patients. Binary observed-response assessment used 173 clean pairs from 40 patients, including 120 PD, 24 SD, 13 PR, and sixteen CR targets. Two repeated outer five-fold patient partitions each used three patient-disjoint inner folds for penalty selection and calibration. This is internal development robustness on previously studied cohorts, not untouched validation.

### Forecasting: a current-state gain, not an established history gain

Each model received measured cutoff anatomy. L and H share known past timing and history-availability flags. H adds preceding measured-state velocity. The q arms additionally share the retrospective realized-interval query and its declared interactions. P predicts persistence, and M adds a source mean residual correction. Lower patient-uniform native-log-volume CRPS is better. Scores average the two partition repetitions, not predictions.

| Predictor | Primary source CRPS |
|---|---:|
| Persistence, P | 1.120054 |
| Mean residual correction, M | 1.167618 |
| Current state and known schedule, L | 1.003208 |
| Added history, H | 0.980585 |
| Query-aware current state, Lq | 1.006585 |
| Query-aware history, Hq | 0.984348 |

Hq minus Lq was `-0.022237`, with two-test-family 97.5% patient-bootstrap interval `[-0.059149, +0.009702]`. The point reduction was 2.21 percent, below the five-percent criterion. Both repetition contrasts were negative, but the interval crossed zero. H minus L without the realized query was similarly unresolved: `-0.022623`, interval `[-0.056630, +0.005712]`.

Lq minus persistence was `-0.113469`, interval `[-0.195667, -0.031457]`, a 10.13-percent point reduction. Most of the persistence-relative improvement therefore did not require preceding anatomical measurements. This secondary result supports an in-domain current-state forecasting/calibration lead, not a demonstrated longitudinal world model. No-primary-history cases remain a separate 69-row, 64-patient companion in the aggregate.

### Observed response: change does not establish added value over current state

Both endpoint scans and their automated native measurements were already observed. The earlier expert response rating was a supplied retrospective context variable, not a certified prospectively available report. The later response label was a target only. No rationale text or survival information entered a predictor.

| Predictor | Patient-uniform binary log loss |
|---|---:|
| Cross-fitted source prevalence, N | 0.536043 |
| Later volume only, C | 0.488014 |
| Paired volumes and change, V | 0.506927 |
| Previous response rating only, R | 0.513637 |
| Previous rating plus current volume, RC | 0.466832 |
| Previous rating plus paired volumes/change, RV | 0.473989 |

The primary RV-minus-RC difference was `+0.007156`, with 97.5% interval `[-0.051815, +0.073679]`. Its repetition contrasts changed sign, from `-0.016608` to `+0.030921`. Without a previous-rating input, V minus C was `+0.018913`, interval `[-0.035716, +0.082064]`. Added change did not meet the prespecified incremental criterion. The earlier 0.78 AUROC remains a small reused-assessment finding, not a validated change biomarker.

Pooled out-of-fold AUROC was 0.676 for V and 0.670 for C. These secondary ranking summaries need care because independently fitted folds can have different probability scales. The prevalence model had pooled AUROC 0.307 even though its within-fold AUROC was exactly 0.5 in every fold. The saved-prediction diagnostic preserves that cross-fold ranking effect. It did not replace the primary proper log-loss endpoint, select a threshold, or trigger new fitting.

### Locked anatomical transfer: the gain did not transport

The six final forecast models selected penalties and uncertainty on the 64 source patients, then saved their parameters before opening SAILOR target scores. No SAILOR fitting, standardization, penalty selection, or uncertainty recalibration followed. Native mask volumes and original session adjacency were checked for all 242 eligible pairs from 27 patients, independently of perfusion availability. Primary transfer used 215 history-eligible pairs from all 27 patients. The 27 no-history pairs are separate companions.

| Source-fitted predictor | SAILOR primary CRPS |
|---|---:|
| Persistence, P | 0.866600 |
| Mean residual correction, M | 0.894149 |
| Current state and known schedule, L | 0.995019 |
| Added history, H | 1.029152 |
| Query-aware current state, Lq | 1.020366 |
| Query-aware history, Hq | 1.052593 |

Hq minus Lq was `+0.032227`, with descriptive 95% interval `[+0.014689, +0.052121]`. H minus L without the realized query was also worse: `+0.034133`, interval `[+0.017954, +0.052062]`. Lq minus persistence was `+0.153766`, interval `[+0.041663, +0.274628]`. Neither the source history increment nor the learned current-state advantage transported under this fixed recipe. Different automated targets and cohort distributions limit causal explanations. SAILOR has been reused, so this is developmental cross-cohort transfer, not an independent final validation. No clinical transfer was attempted without a verified response-label mapping.

### Measurement sensitivity and verification

Fixed visit-consistent multiplicative volume perturbations at log-SD 0.05 and 0.10 did not reverse the main point contrasts. At 0.10, source Hq/Lq CRPS was 0.985576/1.007805, clinical RV/RC log loss was 0.473088/0.465110, and SAILOR Hq/Lq CRPS was 1.054673/1.021905. This narrow synthetic stress does not certify segmentation quality, clinical utility, or scanner robustness. Zero compartments were retained and outcome measurements were not perturbed.

Two-real-row numerical fitting smokes passed before the source study and locked transfer. A separate replay reconstructed all 90 complete prediction arrays across both repetitions, tasks, transfer, and stress conditions exactly from 126 saved numerical models. Source metrics and transfer primary CRPS were rescored. Outer patient membership was checked. The [aggregate](r47_anatomical_signal_metrics.json) binds code, protocol, input hashes, model-lock receipts, all contrasts, and the post-assessment AUROC diagnostic. Raw rows, fitted coefficients, predictions, and mask arrays remain private. No image network, new frozen-feature experiment, rescue split, or post-result threshold search was run. There has been no independent scientific review.

## Pre-audit revisit completed pilots, October 31, 2026

All selected runnable comparisons completed on `mosaic-gpu`. The [aggregate](preaudit_revisit_metrics.json) binds R41 through R46, preparation receipts, seed scores, calibration, nuisance companions, source hashes, and saved-artifact replay. The [fixed protocols](METHODS.md#pre-audit-revisit-fixed-execution-protocols) preceded assessment. These are short exploratory development pilots, not untouched validation or evidence that the methods cannot work. No primary added-information hypothesis established a benefit. No rescue fitting followed. R35 through R40 remain unchanged.

The image models completed 3,024 optimizer updates across eighteen fits: 1,584 CNN forecast updates, 792 ViT forecast updates, and 648 observed-response updates. Four anatomical-state transition fits completed another 800 updates. Each trainable family passed real-data smoke updates and serialized-model replay first. Earlier partial preparation failed only because transfer was incomplete; the final preparation retained all 423 declared rows.

### R41: native crops and aligned past-only forecasting

Source52 supplied 262 fitting pairs, validation13 supplied 60 selection and calibration pairs, and primary assessment used 75 history-eligible pairs from eighteen reused development patients. Scores below average seeds 42 and 43, then average within patient and across patients. CRPS uses full native `log1p(mm3)` targets. Every predictor receives measured cutoff-mask volumes. Ridge additionally receives previous measured state, making it a conservative extra-information control rather than an identical MRI-only observation budget.

| Predictor | Primary CRPS |
|---|---:|
| Whole-brain history | 1.135076 |
| Whole-brain latest-only | 1.137080 |
| Native-crop history | 1.133347 |
| Native-crop latest-only | 1.133663 |
| Measured-cutoff persistence | 1.150014 |
| Empirical source residuals | 1.222697 |
| Horizon-aware measured-state ridge | 1.006442 |

Native-history minus whole-history was `-0.001729`, with unadjusted paired 95% interval `[-0.004980, +0.001632]`. Its 0.15-percent point reduction did not meet the five-percent criterion. Native-history minus native-latest was `-0.000316`, interval `[-0.003968, +0.003902]`. Neither localization nor added MRI history established the primary forecast benefit. This does not establish equivalence. Native-history was worse than measured-state ridge by `+0.126905`, interval `[+0.049522, +0.210683]`.

The cutoff-only crop rule retained all cutoff compartment voxels in every prepared row. Widths ranged from 128 to 185 mm in source, 128 to 181 mm in validation, and 128 to 165 mm in assessment. Future retention fell below 99 percent in at least one compartment for 26 of 262 source rows, four of 60 validation rows, and four of 101 total assessment rows. No future mask recentered an input, and no low-retention row was removed. Full native volumes remained the target. Mask Dice is view-specific at `24³`, not a same-grid whole-brain segmentation comparison.

The fixed assessment render was inspected. CNN forecast masks spread across compartments and did not demonstrate faithful spatial forecasts. The source render also retains a contour beyond visible MRI foreground. More visible local detail is not expert-verified anatomy. MRI-only current-volume heads remained inaccurate, while forecast means also received the measured cutoff state. Small input-perturbation effects therefore do not establish learned acquisition robustness.

### R42: trainable pretrained spatial encoding

On the same mask-assisted task, pretrained BrainIAC history scored `1.142207`, random-initialized ViT history `1.156104`, and native CNN history `1.133347`. Pretrained minus random was `-0.013897`, interval `[-0.031416, +0.004509]`, a 1.20-percent point reduction below the five-percent criterion. Pretrained minus CNN was `+0.008859`, interval `[-0.006671, +0.025664]`. Neither pretraining value nor model-family benefit was established under this fixed six-epoch recipe.

A post-assessment diagnostic checked repeated Dice values without refitting. All four ViT fits predicted empty current masks for all three compartments on all 75 primary rows. Their nonzero average Dice came from genuinely empty target cases. Dice on nonempty current targets was effectively zero. Future necrotic and enhancing masks were also empty or nearly always empty. The primary forecast scores replay correctly, but the spatial mask readout did not become usable. This is a training/readout limitation of the short pilot, not evidence that pretraining cannot improve anatomical representation. All active encoders were trainable; this was not a frozen-feature experiment.

### R43: nominal-time anatomical dynamics and propagated rollout

Both neural transitions propagated their own predicted state across later observed visits. Intermediate future measurements never updated history. The task used measured anatomical state, not frozen MRI features. Nominal week-bin intervals do not support exact growth-rate claims.

| Later observed target | Assessment pairs / patients | Continuous CRPS | Discrete CRPS | Ridge CRPS | Persistence CRPS |
|---|---|---:|---:|---:|---:|
| One step | 75 / 18 | 1.138617 | 1.029067 | 1.006442 | 1.150014 |
| Two steps | 55 / 17 | 1.280795 | 1.103521 | 1.125298 | 1.290983 |
| Three steps | 37 / 13 | 1.644485 | 1.390603 | 1.353836 | 1.703721 |

Continuous minus discrete was `+0.109551`, interval `[+0.038689, +0.197760]`, at one step, and `+0.177274`, interval `[+0.032511, +0.335841]`, at two steps. The three-step difference was `+0.253882`, interval `[-0.052821, +0.587713]`. The continuous formulation did not improve the matched predictor and was worse at the first two horizons. All three contrasts against persistence crossed zero. Constant-velocity extrapolation was unstable, with CRPS 2.967219, 6.714766, and 6.022563. Beating that control does not establish a world-model capability. The aggregate retains the Gaussian-moment empirical-change companion and process-noise coverage limits.

### R44: known-at-cutoff postoperative context

Adding prior postoperative status and nominal time since the latest observed postoperative week to measured-state ridge gave CRPS `1.007163`, versus `1.006442` without it. The difference was `+0.000721`, interval `[-0.010310, +0.014198]`, on the same 75 pairs and eighteen patients. Added benefit was not established. This tested known history, not future operative labels or R33's completed filtering question. No larger regime-conditioned network followed.

### R45: observed binary response assessment

The fixed endpoint was PD versus SD, PR, or CR after both endpoint scans were observed. The reused assessment contains 48 pairs from sixteen patients, with 36 PD, ten SD, two PR, and no CR targets. Concurrent rationale text was not an input. All three image comparisons trained pretrained encoders rather than treating undertrained random CNNs as strong controls. The short recipe still limits conclusions.

| Predictor | Patient-uniform log loss | AUROC | Balanced accuracy at 0.5 |
|---|---:|---:|---:|
| Later MRI only | 0.473646 | 0.291961 | 0.500000 |
| Separate estimated states and difference | 0.689709 | 0.634914 | 0.537463 |
| Paired spatial representation | 0.506916 | 0.346912 | 0.516498 |
| Constant source PD prevalence | 0.440473 | 0.500000 | 0.500000 |
| Mask-assisted measured-volume change | 0.391340 | 0.784474 | 0.533766 |

Image rows average seed scores, not predictions. Paired minus separate-state log loss was `-0.182792`, interval `[-0.357865, +0.034696]`. Paired minus later-only was `+0.033270`, interval `[-0.032550, +0.097602]`. Paired minus the source-prevalence null was `+0.066443`, interval `[-0.019416, +0.165743]`. The primary paired benefit was not established. Lower log loss than the separate-state arm did not establish useful discrimination or superiority to a majority-class predictor. These clinical losses are not anatomical forecast CRPS. No four-class, rationale-generation, prospective-progression, or SAILOR clinical claim follows.

### R46: native perfusion and descriptive scanner context

The SAILOR-native comparison retained 238 adjacent pairs from 27 patients under identical availability and patient folds. Two pairs lacked cutoff perfusion and two failed the positive-reference requirement. Every retained pair used cutoff normal-appearing white matter for the declared within-patient ratio normalization. Future maps and unverified dose maps were not inputs.

Structural measured-state history scored CRPS `0.788263`, its region-missingness companion `0.800766`, and added regional perfusion `0.977529`. Perfusion minus structural was `+0.189265`, interval `[-0.002511, +0.559999]`. Perfusion minus missingness was `+0.176763`, interval `[-0.013475, +0.534328]`. This fixed quantitative comparator did not establish added value. Its worse point estimate does not certify harm or rule out a different physiological representation. Absolute perfusion units remain uncertified.

The separate LUMIERE scanner-context comparator scored `1.071522`, versus `1.006442` without scanner fields. The difference was `+0.065080`, interval `[+0.006604, +0.126176]`. Added scanner fields worsened this fitted predictor. Patient-scanner confounding and descriptive metadata prevent a causal nuisance conclusion.

### Blocked branches and verification scope

Overall-survival training remains blocked because released duration semantics do not establish event and censoring status. The primary-paper check did not resolve that gap. First-observed progression separately needs a landmark, observation endpoint, and competing-death handling. Subject-level dose conditioning remains blocked on physical transform, dose units, and known-at-cutoff planned-versus-delivered timing. No labels, transforms, or availability timestamps were invented to force these branches to fit.

Saved-artifact replay matched two real predictions exactly in each of eighteen image cells and rescored their complete saved assessment predictions. It also checked twelve transition/horizon predictions to tolerance `1e-5`, both postoperative and scanner numerical models exactly, and all fifteen perfusion fold models exactly. This verifies bounded saved-artifact behavior, not a full fitting replay or independent scientific review. Private rows, predictions, masks, source images, and weights remain outside git. The [aggregate](preaudit_revisit_metrics.json) preserves unfavorable results, single-visit companions, coverage, seed variation, nuisance checks, and the empty-mask diagnostic. Unadjusted intervals across several exploratory contrasts are not familywise-confirmatory evidence.

## Pre-audit revisit feasibility, first pass

The operator selected eight earlier threads for sequential investigation: native-resolution lesion crops, aligned causal lesion forecasting, trainable pretrained spatial encoding, anatomical continuous-time rollout, known postoperative context, observed-pair response assessment, trajectory-based prognosis, and perfusion or dose context. These are distinct questions. This first pass does not report eight trained comparisons.

The rerunnable metadata census, source probes, private file mappings, and working plan are in `scratch/preaudit-revisit/`. MRI-array work ran on the approved `mosaic-gpu` host. No MRI arrays or model weights were loaded on the laptop. Native MRI intermediates were available locally for all 353 prepared source visits. Two source examples were transferred for a bounded crop and pretrained-backbone smoke.

| Check | Observation | Limit |
|---|---|---|
| Native crop | A 128-mm crop retained all labeled compartment voxels in both examples. One tiny non-enhancing component was present in a native crop but absent from its whole-brain `96³` mask. Both figures were inspected. | Automated labels, two source examples, and supplied cutoff-mask localization. No forecast benefit or segmentation truth follows. |
| Trainable BrainIAC | Two real source updates completed with four separate modalities and 216 spatial tokens per modality. All 88,340,736 encoder parameters remained trainable. Backbone weights changed and saved predictions replayed exactly. | A current-state runtime smoke, not a trained forecasting result. |
| Multi-step support | Source chains have 262, 201, and 164 eligible targets at one, two, and three later visits. Validation has 60, 40, and 30. | Existing eligible adjacency only. Intervals remain nominal week bins. No anatomical ODE was fitted. |
| Postoperative context | All earlier rating rows identify prior postoperative context at 210 source cutoffs, including 53 within thirteen nominal weeks. | Observed postoperative scan weeks are not exact surgery dates. Future operative labels are not predictor inputs. |
| Response assessment | Valid response labels on both observed endpoints leave 144 source pairs and 29 validation pairs. The corresponding reused assessment slice has 48 pairs, with no CR targets. | Insufficient support for a strong four-class generalization claim. Rationale is free text, not a ready categorical codebook. |
| Prognosis | LUMIERE demographics provide 86 positive survival durations among 91 patients, without an event or censoring column. A first-non-PD landmark before any observed PD is available for 31 source patients, of whom 27 later have observed PD. | Outcome semantics remain unresolved. No-later-PD is not a confirmed negative or a death endpoint. |
| SAILOR perfusion | Filename census finds rCBV and rCBF in 267 subject-sessions across 27 subjects. Two rCBV examples are finite and share their structural reference grid. | File presence and two content samples do not establish all usable forecast pairs or clinical utility. |
| SAILOR dose | Twenty-six subject-level dose files exist. Two samples are finite but have different array shapes from the structural MNI grid. | Physical alignment, units, and planned-versus-delivered timing need verification before conditioning. |
| Scanner context | The LUMIERE metadata table contains 2,487 sequence rows with acquisition fields. | Metadata presence is not a validated numeric encoding or proof that scanner effects explain forecast error. |

The pretrained smoke required a version-specific loader correction. MONAI 1.6 declares cross-attention normalization parameters even when that branch is disabled. The scratch adapter verifies the disabled branch, removes the inactive norms, excludes the checkpoint's contrastive projection head, and strictly loads the complete encoder. It does not accept missing active encoder weights. The configured initialization SHA256 is `f22bdbcae26823a9d9e8aee883c6f24386ba4617339c12269848b6666cc62693`. This binds the configured bytes, not independently certified download origin.

Earlier crop smoke attempts failed an overly strict cache-reconstruction check. The recorded diagnosis found small FP32 discrepancies across all eight sequence examples, with maximum absolute error about `1.07e-5` and matching nonzero support. FP16 rounding amplified some elementwise differences. The final probe checks unchanged processed MRI bytes and FP32 reconstruction at the existing R38 `1e-4` tolerance. It records the errors and does not claim byte-identical historical tensors. An intermediate run also exposed a missing optional `einops` dependency. These were instrument failures, not negative model results.

### Continuation: real-task smokes, October 31, 2026

The operator requested continued execution of all eight threads. The [fixed continuation protocols](METHODS.md#pre-audit-revisit-fixed-execution-protocols) now separate matched anatomical forecasting, observed binary response assessment, and quantitative SAILOR perfusion. These are new bounded development trials, not repeats that replace R35 through R40.

All four CNN crop/history arms and both fully trainable ViT forecast arms passed two source-row optimizer updates on `mosaic-gpu`. Encoder weights changed, native-volume scoring remained finite, and saved predictions replayed exactly. Forecast gradients depended on preceding MRI only in the history arms. Unlike the earlier current-state-only foundation smoke, these smokes include future anatomical losses. Three trainable pretrained observed-response arms also passed two real updates and exact replay. Their source smoke includes one PD and one other-response target. Earlier-image gradient dependence matched the declared arm.

The quantitative perfusion comparator passed a two-pair native SAILOR content, normalization, numerical fitting, scoring, and serialized-model replay check. Multiplying each map by 1.5 left the declared reference-normalized regional features unchanged. Its in-sample two-row fit is a runtime check, not evidence of clinical or forecasting benefit. No future perfusion map entered the inputs.

The continuous and discrete anatomical-state models passed two real source-row updates and checkpoint replay for both declared seeds. Postoperative and scanner ridge comparators passed two-real-row fitting and JSON replay. These standalone smokes did not run the fixed cohort assessment or select a winning model.

Two new fixed-field source renders were inspected in `scratch/preaudit-revisit/receipts/fixed-field-source-smoke.png`. The local views expose more image detail, but a sampled automated contour extends beyond visible MRI foreground. These remain automated proxy targets. No expert correction or outcome-driven exclusion was made.

Full preparation processed 120 of the fixed 423 rows before encountering a native file that had not yet arrived over SSH. No comparison fit or assessment started in that attempt. The partial preparation is not a smaller substitute cohort. Both native transfers subsequently completed. The fixed SAILOR content check retained 238 pairs from 27 patients, excluding two pairs without cutoff maps and two with nonpositive reference medians. All retained pairs used normal-appearing white matter reference regions. This preparation took 237.16 seconds and did not fit or assess models. The full sequential comparison runner then started with the unchanged cohort and recipes. The scratch receipts retain the passed smokes, timing estimates, instrument checks, and partial-preparation traceback. This entry reports dated preparation and launch evidence, not completed model outcomes.

The bounded LUMIERE primary-paper check also leaves prognosis fitting blocked. The paper reports overall survival and Kaplan-Meier curves but does not supply patient-level event-versus-censoring semantics for the released duration column. First-observed-PD fitting still needs follow-up and competing-death definitions. Dose conditioning remains blocked on verified physical transform, dose units, and cutoff-availability semantics.

## R40 completed trainable spatial-horizon pilot

All six fits and the prewritten assessment completed on `mosaic-gpu`, with 3,960 optimizer updates, 31,440 sampled pairs, and 3,624.49 wall seconds including assessment. The [aggregate](r40_full_metrics.json) reports `HISTORY_BENEFIT_NOT_ESTABLISHED` on the 75-pair, eighteen-patient reused development cohort.

| Predictor | Patient-uniform CRPS |
|---|---:|
| Trainable spatial history, Hq | 1.221023 |
| Trainable latest-only, Lq | 1.221642 |
| MRI-state persistence, P | 1.229328 |
| Empirical forecast residuals, E | 1.204891 |
| Horizon-aware ridge, Rq | 1.196132 |

Hq minus Lq was `-0.000619`, with paired 95% patient-bootstrap interval `[-0.017474, +0.015834]`. The point reduction was about 0.05 percent, below the specified five-percent threshold. The comparisons against P, E, and Rq also had intervals crossing zero. This does not establish equivalence or identify the cause of earlier inconclusive outcomes. R40 supplies the realized week-bin interval as a retrospective query, not a prospectively known next-visit time. Computation is complete. No independent result review has been performed.

## How the results are reported

MAE is mean absolute error. Lower values are better. A persistence-relative error below 1.0 favors the model over persistence on that stated metric. A value above 1.0 favors persistence. Persistence predicts no change from the current measurement.

For an error difference, the convention is model minus comparator unless a row states otherwise. A negative difference favors the model. Brackets give a confidence interval. An interval that crosses zero leaves the comparison unresolved.

Patient-uniform results give each patient equal weight. Pooled results combine observations, so patients with more observations receive more weight. Relative scores retain the source's aggregation rule and cannot always be reconstructed by dividing two adjacent aggregate MAEs.

Macro-F1, AUC, R², and balanced accuracy measure different things. The [Methods glossary](METHODS.md#terms) defines them. Their values are not interchangeable.

An A identifier records a historical experiment or audit. R identifiers record research directions and later investigations. Completed execution, valid measurement, and scientific success are different statuses.

## R38 continuation status, October 30, 2026

R38 has no complete scientific assessment yet. The October 27 run completed all four seed-42 cells at epoch 20, then a process-supervisor exception interrupted OH43. The original private interruption receipt records 686 persisted OH43 update entries but no OH43 checkpoint; its unflushed tail is unknown. No seed-43 or seed-44 cell completed in that attempt. The original failed stage and expenditure remain preserved.

The October 30 operator instruction authorizes the unchanged [continuation](METHODS.md#october-30-continuation). Four new adoption/rejection tests passed on `mosaic-gpu`, and the real CPU input/checkpoint check passed without GPU updates. It verified all four completed checkpoints against preserved backup hashes, all 20,960 completed update records, optimizer counts, cohort membership, normalizers, original seeded initializations and original sampling schedules. It did not inspect assessment scores. The real check took 15.752 wall seconds and 12.559 process CPU seconds.

The continuation is queued behind an unrelated GPU job. It will reuse seed 42, train the eight remaining cells and run full inference, controls, calibration, nuisance checks and the original scorer. No new training or R38 outcome is claimed by this queued status. Private queue logs, status, pins and eventual outputs are under `outputs/r38-joint-dynamics/continuation-20261030/`. A completion notification is arranged. This entry records a dated launch status, not live process evidence or reporting approval.

Later October 30 status supersedes that queue snapshot: training started and saved OH43 and OL43, while JH43 was in progress. The operator then requested stopping after seed 43 for the time being. A reversible gate prevents the seed-44 initialization from being opened after JL43 finishes; its exact bytes are held separately and will be restored after the process exits. A detached finalizer checks the saved seed-43 cells and records the planned pause. The runner's expected missing-initialization stop receipt remains preserved and must not be mistaken for a model failure. At request verification, no seed-44 cell had started. Pause completion and full assessment are not yet claimed.

The later pause receipt confirms **PAUSED_AFTER_SEED43**. All four seed-43 epoch-20 checkpoints are saved, seed 44 never started, and its initialization was restored with the same byte hash. Eight of twelve R38 cells are complete across seeds 42 and 43.

While `mosaic-gpu` was paused, a temporary Kaggle T4 offload was probed (`scratch/kaggle-seed44/`): all four arms passed a 3-step BF16 smoke test under pinned PyTorch 2.9.1+cu128, but lacking native BF16 Tensor Cores on Turing (`sm_75`) made 3D convolutions ~6–8x slower than on `mosaic-gpu` (2.03–3.83 s/step, projecting ~22.4 single-GPU hours vs. ~49 minutes per seed on RTX 5090). Once `mosaic-gpu` became free, the Kaggle offload was cancelled and deleted, and the native `mosaic-gpu` continuation resumed from the eight verified seed-42/43 checkpoints to finish seed 44 and the full three-seed assessment on identical hardware.

<a id="r38-completed"></a>

### R38 full three-seed factorial assessment, October 30, 2026

The native `mosaic-gpu` continuation completed the four seed-44 arms (`OH_44`, `OL_44`, `JH_44`, `JL_44` — 20,960 new updates, reusing the 41,920 verified updates from seeds 42 and 43 for 62,880 total updates) plus full three-seed inference, calibration, common controls, nuisance checks, and assessment in 3,182.28 wall seconds (3,181.59 process CPU seconds). The [aggregate](r38_full_metrics.json) records **`INCONCLUSIVE_FOR_FULL_CAPABILITY`** (`sha256: 0bdc7fd9b802f7c3b2d256314b7b86ee83ddb69622692bd30d88497a55a54d83`).

Primary cohort scores on 75 history-eligible pairs from 18 reused LUMIERE development patients (`log1p(mm3)` scale, averaging seed scores, then rows within patient, then patients equally):

| Arm / Control | Patient-uniform CRPS | Forecast MAE | Current-state MAE | Current Dice | Future Dice | 80% coverage |
|---|---:|---:|---:|---:|---:|---:|
| JH, joint spatial history | 1.369021 | 1.801673 | 1.539730 | 0.287190 | 0.218269 | 0.660935 |
| JL, joint latest-only | 1.367230 | 1.805719 | 1.541398 | 0.284038 | 0.211165 | 0.707892 |
| OH, detached-observation history | 1.348686 | 1.781765 | 1.473347 | 0.305464 | 0.214302 | 0.736552 |
| OL, detached-observation latest-only | 1.329503 | 1.763897 | 1.473347 | 0.305464 | 0.213003 | 0.706815 |
| P_JH, own-state persistence (`JH`) | 1.340510 | 1.797596 | — | — | — | 0.781966 |
| P_JL, own-state persistence (`JL`) | 1.341909 | 1.779317 | — | — | — | 0.803547 |
| P_OH / P_OL, own-state persistence (`OH`/`OL`) | 1.303277 | 1.729657 | — | — | — | 0.822457 |
| P, common frozen-encoder persistence | 1.269140 | 1.657288 | — | — | — | 0.827601 |
| E, empirical change + observation noise | 1.222632 | 1.697116 | — | — | — | 0.637860 |
| SSM, damped linear state-space filter | 1.273682 | 1.652073 | — | — | — | 0.777998 |
| FP, filtered persistence | 1.325807 | 1.650384 | — | — | — | 0.548648 |

Primary factorial contrasts (10,000-draw patient bootstrap, family-adjusted 99.5% intervals across 10 contrasts; negative CRPS differences favor the first arm):

| Contrast | CRPS difference | Relative reduction | 95% CI | 99.5% adjusted CI | Result |
|---|---:|---:|---|---|---|
| `JH - OH` (joint vs. detached with history) | +0.020334 | -1.51% | [-0.040349, +0.078353] | [-0.069038, +0.103084] | Unresolved |
| `JH - JL` (history vs. latest under joint training) | +0.001791 | -0.13% | [-0.068582, +0.066840] | [-0.100218, +0.094264] | Unresolved |
| `JL - OL` (joint vs. detached without history) | +0.037726 | -2.84% | [-0.035925, +0.115080] | [-0.071775, +0.148155] | Unresolved |
| `OH - OL` (history vs. latest under detached encoder) | +0.019183 | -1.44% | [-0.013552, +0.052189] | [-0.027233, +0.067681] | Unresolved |
| `(JH - JL) - (OH - OL)` (2x2 interaction) | -0.017392 | — | [-0.067013, +0.026913] | [-0.090307, +0.043624] | Unresolved |
| `JH - P` | +0.099881 | -7.87% | [-0.000678, +0.211663] | [-0.035770, +0.265588] | Unresolved |
| `JH - E` | +0.146388 | -11.97% | [+0.051216, +0.241087] | [+0.012658, +0.281944] | Harmful |
| `JH - FP` | +0.043214 | -3.26% | [-0.110706, +0.201908] | [-0.165397, +0.272537] | Unresolved |
| `JH - SSM` | +0.095338 | -7.49% | [-0.019194, +0.222591] | [-0.058556, +0.282945] | Unresolved |
| `(P_JH - JH) - (P_OH - OH)` | +0.016899 | — | [-0.005622, +0.040277] | [-0.013496, +0.050371] | Unresolved |

Companion gates and robustness checks also failed to clear their prespecified thresholds:
- **Current-state retention (`current_state_retention.passes = false`):** Allowing forecast gradients into the spatial encoder slightly degraded current-visit estimation (`JH` current MAE 1.5397 vs. `OH` 1.4733; `JH` current Dice 0.2872 vs. `OH` 0.3055), failing the noninferiority margins (+0.05 MAE / +0.02 Dice) in necrotic and enhancing compartments.
- **Calibration coverage (`JH_coverage_80`):** Only necrotic/non-enhancing (0.7482) fell inside `[0.70, 0.90]`; enhancing (0.6264) and edema/FLAIR (0.6082) were under-covered.
- **Acquisition nuisance (`nuisance.passes = false`):** Intensity scaling/shift passed on all arms (`<0.02%` relative CRPS change due to nonzero standardization), but synthetic 2x slice-thickness resolution degradation worsened CRPS across all four arms beyond the 10% relative / 0.05 MAE margins (for `JH`, identity CRPS 1.8745 rose to 2.1089 under resolution perturbation, with +12.7% necrotic and +17.4% enhancing relative CRPS worsening).
- **Single-visit companion (26 rows, 23 patients):** `OL` (CRPS 0.7451, MAE 0.9631) and `OH` (CRPS 0.7511, MAE 0.9663) slightly outperformed `JH` (CRPS 0.8045, MAE 1.0607) and `JL` (CRPS 0.8227, MAE 1.0771).

This completes the fixed R38 2x2 study without establishing a joint-training or history benefit.

## R39 treatment-status forecasting, October 30, 2026

The fixed [R39 protocol](METHODS.md#r39-cutoff-treatment-phase-forecasting-feasibility) tested whether current treatment status improves next-visit anatomical prediction. Five nested patient-held-out folds covered 242 consecutive SAILOR pairs from 27 reused development patients. All pairs passed the identity and original-session adjacency checks. Of these pairs, 215 had an earlier MRI state and 212 had known cutoff treatment status. Source labels were CRT for 101 pairs, TMZ for 95, no treatment for 16, and unknown for 30.

Inputs were saved R35 seed-42 MRI-derived volume estimates, current treatment phase and past timing. No future label, future MRI, future measured volume or actual next interval entered a predictor. Targets were three native automated compartment log-volumes. L uses the latest state; H adds the immediately preceding state difference; T adds current treatment and state-by-treatment interactions. All four share timing covariates. This is a small linear transition test on a compressed state, not a new spatial encoder experiment.

| Predictor | Patient-uniform Gaussian CRPS | Patient-uniform MAE |
|---|---:|---:|
| Latest MRI state, L | 1.281134 | 1.775553 |
| Latest state plus treatment, LT | 1.236070 | 1.697539 |
| MRI state history, H | 1.264075 | 1.740442 |
| History plus treatment, HT | 1.265126 | 1.722333 |
| Time only | 1.453798 | 2.033726 |
| Treatment and time only | 1.468382 | 2.064741 |
| MRI-derived state persistence | 1.885925 | 2.391879 |
| Measured-current-volume persistence, privileged input | 0.840362 | 0.943004 |

Scores average compartments, then visits within each patient, then patients. Units are `log1p(mm3)` for MAE and the corresponding Gaussian CRPS scale; lower is better. Sigma comes from inner held-out residuals for fitted models and outer-training residuals for fixed persistence. The measured-volume reference receives current automated mask volumes, unlike the MRI-only models.

The primary HT-minus-H CRPS difference was **+0.001051**, with paired 99% patient-bootstrap interval **[-0.080809, +0.081545]**. Adding treatment was 0.083% worse at the point estimate, not the required 5% improvement. HT-minus-LT was +0.029056, interval [-0.024579, +0.093310]. The secondary LT-minus-L difference was -0.045064, interval [-0.117247, +0.015843], a 3.52% point reduction that did not establish superiority. Both treatment feasibility and the history-conditioned follow-up criteria were unmet. This is inconclusive for incremental treatment or history benefit, not evidence of equivalence or treatment irrelevance.

HT beat MRI-state persistence by -0.620799, interval [-1.044117, -0.246458], and treatment/time-only by -0.203256, interval [-0.420882, -0.017003]. Those gains do not isolate learned dynamics. Measured-current-volume persistence was substantially better than every MRI-derived predictor at the point estimate. Differences in state-estimation quality and target calibration remain plausible contributors. Known-status and history-available companions are retained in the [aggregate](r39_treatment_metrics.json); they were not used to select a method.

Execution used CPU on `mosaic-gpu`, without GPU training or SAILOR image transfer. Seven behavioral tests and a real-row canary passed before the fixed run. Saved-weight replay reconstructed all 40 outer-fold model files, eight aggregate score pairs and five bootstrap contrasts. The full-run process used 0.49 user plus 0.03 system CPU seconds; that excludes imports and checks in other processes and is not a total-workflow cost. The one-CPU-hour ceiling was not approached. Fitted models, rows and predictions remain private. No independent scientific review occurred.

Limits include retrospective per-session status with unverified prospective availability, approximate intervals, coarse treatment categories, observational confounding, automated targets, reused development patients, constant Gaussian uncertainty and no acquisition-robustness test. Bootstrap intervals condition on fitted folds rather than refitting them. These results neither identify treatment effects nor test planned intervention schedules. The [aggregate and provenance](r39_treatment_metrics.json) and runnable `scripts/check_treatment_forecast.py` preserve the checked scope. R38 subsequently completed its full three-seed assessment above.

<a id="r37-full-pilot"></a>

## R37 full frozen-state dynamics pilot, October 26, 2026

The fixed [R37 pilot](METHODS.md#r37) completed on `mosaic-gpu` using the frozen original R35 seed-42 encoder, volume head and target normalizer. Source52 supplied 262 fitting pairs; internal13 supplied 60 selection/calibration pairs. H, L and G each ran seeds 42, 43 and 44 for twenty epochs, with identical per-seed initialization, patient-uniform sampled orders and 5,240 updates per fit. P/E/R controls also completed. No encoder fitting, extra configuration, assessment retry or rescue sweep ran. The [aggregate](r37_full_metrics.json), [execution receipt](documentation_sources/updates/2026-10-26/r37-execution.md), [closure metadata](documentation_sources/updates/2026-10-26/closure.json) and [independent result review](documentation_sources/updates/2026-10-26/reviews/result-v1.md) preserve the evidence and exact hashes.

The primary cohort is 75 history-eligible pairs from 18 reused LUMIERE development patients. Another 26 single-visit pairs from 23 patients are separate companions; patients may overlap the primary cohort. Nuisance evaluation uses one earliest eligible primary cutoff per patient, 18 examples. These are automated native-volume proxies, not expert biological outcomes or independent validation. Historical gaps are approximate week-bin differences, not verified dates. No future gap, target image, target mask or treatment enters a predictor.

CRPS, the continuous ranked probability score, scores the predictive distribution; lower is better. Both scores below use native `log1p(mm3)` coordinates. Average fitted-seed scores, then compartments/examples within patient, then patients equally. Do not ensemble predictions. P copies the latest frozen MRI-decoded volume, not a measured cutoff mask. L retains H's past gaps, masks and sequence-availability flags but removes earlier MRI maps. G pools only earlier maps, retaining the spatial current-scan pathway.

| Arm | Patient-uniform CRPS | Mean-prediction MAE |
|---|---:|---:|
| H, spatial history | 1.30031836 | 1.69376433 |
| L, latest MRI with matched observation history | 1.29697350 | 1.67683953 |
| P, calibrated decoded-state persistence | 1.26913951 | 1.65728793 |
| E, source joint changes with Gaussian observation noise | 1.22263233 | 1.69711601 |
| R, voxelwise ridge dynamics | 1.28416460 | 1.68391809 |
| G, pooled earlier MRI maps | 1.31326489 | 1.69254350 |

E has the lowest point CRPS, not established superiority over every arm. Its Gaussian observation condition and fixed 0.05 observation standard deviation were locked for the common-floor rationale before full fitting, after the earlier noise ablation was already known. This was not untouched or blinded prespecification.

Negative paired CRPS differences favor H. All five two-sided 99% Bonferroni-adjusted patient-bootstrap intervals span zero. The 10,000 paired patient resamples use seed 2026 after averaging fitted-seed scores. The unadjusted 95% companions also span zero and remain in the aggregate.

| Contrast | CRPS difference | Adjusted 99% interval |
|---|---:|---|
| H minus L | +0.00334486 | [-0.04732935, +0.05234837] |
| H minus P | +0.03117885 | [-0.03592705, +0.10415815] |
| H minus E | +0.07768603 | [-0.03430240, +0.20740137] |
| H minus R | +0.01615376 | [-0.05803496, +0.08182145] |
| H minus G | -0.01294653 | [-0.05938712, +0.02905705] |

H's point CRPS is 0.2579% worse than L, not the required 5% improvement. Added-history benefit and earlier-spatial-detail benefit are not established. This does not prove equivalence or that MRI history has no useful information. Secondary L-minus-P and L-minus-E CRPS differences are +0.02783399 and +0.07434116, with paired 95% intervals [-0.01396873, +0.06935944] and [-0.03036085, +0.18425660]. Latest-state prediction benefit is not established either.

All four capability requirements failed:

- Primary adjusted comparisons do not establish the required improvements.
- H-minus-L MAE is +0.01692480, 95% interval [-0.02721033, +0.06738806]; H-minus-P is +0.03647640, [-0.05329185, +0.14005208]. Both upper bounds exceed +0.05. Failure of this noninferiority requirement is not proof of significant inferiority.
- H's 80% coverage is 84.36%, 66.16% and 87.82% for necrotic/non-enhancing, enhancing and edema/FLAIR compartments. Enhancing coverage misses the required 70% to 90% range. This is a point calibration-gate failure, not a population-calibration significance claim.
- The half-resolution perturbation exceeds H's +10% CRPS and +0.05 MAE margins, as detailed below.

On the fixed 18-patient nuisance set, half-resolution downsample/upsample changes versus identity were:

| Arm | Relative CRPS worsening | CRPS increase, paired 95% interval | MAE increase, paired 95% interval |
|---|---:|---|---|
| H | +19.3641% | +0.37950101 [0.10504015, 0.72258824] | +0.37930948 [0.06658814, 0.75110040] |
| L | +18.6841% | +0.37233767 [0.11416493, 0.69375683] | +0.35931552 [0.07253778, 0.70238198] |

Mean absolute log-volume drift across compartments is 0.79093224 for H, 95% interval [0.53066429, 1.09873528], and 0.75966924 for L, [0.50350286, 1.06064980]. The raw key `mean_absolute_volume_drift` uses log-volume coordinates, not mm3. These are scoped synthetic-resolution sensitivity findings. Their 95% companion intervals are separate from the five adjusted primary contrasts; they do not establish clinical scanner effects. No adjusted primary interval establishes H degradation, so the primary `demonstrated_harms` list is empty. That is not proof of harmlessness. Intensity-perturbation changes are approximately numerical noise because standardization largely cancels the affine transform, not evidence of learned intensity invariance.

The aggregate retains every seed, single-visit, compartment, coverage/width and empty/appearance/disappearance companion. Primary empty-target counts are 6/16/4 pairs by compartment, appearance 2/5/2 and disappearance 3/4/3. These subsets are sparse, sometimes concentrated in few patients; do not treat them as biological transition validation or new success endpoints.

The independent reviewer authored, compiled and executed a standalone CPU checker once. It passed 12,861 checks without discrepancies, including independent chronology reconstruction, scoring/bootstrap arithmetic and eighteen bounded fixed-weight neural forwards. It did not refit training trajectories, repeat image encoding or run a full validation/assessment sweep. The [exact checker](documentation_sources/updates/2026-10-26/reviews/result-replay-v1.py) and [bound receipt](documentation_sources/updates/2026-10-26/reviews/result-replay-v1.json) preserve that scope. Before fitting, actual geometry/content checks covered 2,396 MRI files, and every one of 599 original visits reproduced pooled features and decoded volumes at fixed `atol=rtol=1e-4`.

The verdict is `APPROVED_FOR_SCOPED_R37_REPORTING`; the outcome is **INCONCLUSIVE_FOR_FULL_CAPABILITY**, not scientific success. This formulation is closed after its one analysis. Calibration, uncertainty, frozen-decoder bias, approximate timing, unknown interventions and reused development patients limit interpretation. No biological validation, independent generalization, fixed-day/multistep prediction, novelty or publication acceptance follows. No rescue training is authorized.

<a id="r37-noise-ablation"></a>

## R37 E observation-noise ablation, October 19, 2026

The operator-authorized [noise comparison](METHODS.md#r37) completed once using the frozen R35 seed-42 cached embeddings and volume head. No encoder forward pass or neural training ran. This is an E-only distribution comparison, not the full R37 history-versus-current-scan pilot. Source evidence is the saved [aggregate metrics](r37_noise_ablation_metrics.json), private v6 run bundle under `outputs/r37-noise-ablation/v6/`, and [independent-reference replay receipt](documentation_sources/updates/2026-10-19/reviews/noise-result-replay.json). The reviewer wrote the replay; the parent executed it unchanged because the reviewer lacked an execution tool. That distinction is not independent execution by the reviewer.

The source52 fit used 262 adjacent eligible pairs; internal13 calibration used 60 pairs. The history-eligible development assessment used 75 next-visit pairs from 18 patients. Another 26 single-visit assessment pairs were reported separately. Both conditions retained identical patient-weighted joint changes and decoded cutoffs. The discrete condition added no observation noise. The Gaussian condition added fixed observation standard deviation 0.05 per native log-volume coordinate. Both independently selected multiplier 1 from `[0.5,1,2]` using validation CRPS only.

| Common multiplier 1 result | Patient-uniform CRPS in `log1p(mm3)` |
|---|---:|
| Without added observation noise | 1.2230516797 |
| With Gaussian observation noise | 1.2226323341 |
| Gaussian minus discrete | -0.0004193456 |
| Paired patient-bootstrap 95% interval | [-0.0005599483, -0.0002635300] |

CRPS averages compartments within example, examples within patient, then patients. The interval uses 10,000 paired patient resamples with seed 2026. The relative reduction is 0.0342868%. The validation-selected comparison has the same point difference because both selected multiplier 1; its independently drawn interval is [-0.0005597608, -0.0002647989]. All-grid and single-visit scores remain in the aggregate file. There was no assessment-based condition selection.

The independently authored reference reproduced every saved CRPS array to a maximum absolute difference of 3.11e-14 and bootstrap draws to 3.57e-15. It reconstructed eligible rows, checked split separation and content bindings, and confirmed the saved calibration choices. This establishes numerical replay, not biological validity. Scoped result-review disposition is recorded in the [execution receipt](documentation_sources/updates/2026-10-19/r37-noise-execution.md).

The measured difference favors smoothing on this development score, but is tiny. It does not establish a meaningful anatomical benefit, history benefit, uncertainty calibration, or a successful world model. Targets are automated mask volumes, development patients were reused, and historical gaps use approximate week bins with suffixes used only for ordering. Same-week target visits can remain eligible; invalid retained historical gaps exclude a row without bridging visits. No full-pilot control or multiplicity choice is selected by this result. At this October 19 receipt the full H/L/P/E/R/G pilot remained incomplete. The separate [October 26 completion](#r37-full-pilot) supersedes that execution status without changing the ablation result.

The v1 preflight had no resource ledger, so its expenditure is unavailable. Superseded preflight receipts were preserved rather than replaced. Later measured preflight expenditure and the completed CPU-only execution were recorded cumulatively; no assessment retry occurred.

<a id="current-evidence"></a>

## Current evidence and withdrawals

### Current scoped findings

The early P1 and P2 RANO-free forecasts did not establish an advantage over persistence. Later R47 retained a narrower in-domain current-state CRPS gain, described above, without a validated history increment or transfer benefit. In P1, JEPA-state patient-uniform relative MAE was 1.393 in LUMIERE and 2.883 in SAILOR, versus persistence at 1.000. The November 2, 2026 reconciliation below corrects the earlier SAILOR value of 2.873, which used pair-weighted errors.

In P2, the physical-feature residual MLP had LUMIERE relative MAE 0.997. Its paired 95% confidence interval for the error difference was −0.096 to +0.100, so the apparent improvement remained unresolved. SAILOR relative MAE was 1.288, with interval +0.176 to +0.369, indicating worse transfer performance. Adding fixed vision or JEPA features did not fix the result.

The full records are [P1 anatomy baselines](documentation_sources/versions/4c077cc0f084f2d14b48b4cf366088ff7c0dc11a6ed3bf8b5678cbea901e141d.md), [P2 execution](documentation_sources/versions/9ea08b90cf4971fb7b85d4b150ef1317994f42aa2fca2f1c9c0f5de5424e38e4.md), and [A38 to A39](documentation_sources/versions/2ddad8d48f3ebb0e89c6855d2c8da8e2762d8291664d49d38faab77851d72549.md).

R35 completed a different task: state estimation and change between two observed scans. Its spatial model reduced in-domain automated-volume error. The dedicated paired head did not establish an advantage over subtracting separate state estimates. It did not establish nuisance separation or successful SAILOR transfer. It was neither forecasting nor validation of biological change. [The R35 section](#r35) gives the comparisons.

R36 Phase 1 computation and scoped reporting review are complete. The primary C-versus-B encoder-accessibility result remains INCONCLUSIVE. C passes the specified direct-volume retention margin against A. D's nuisance tradeoff and added temporal information remain INCONCLUSIVE. Descriptive nonlinear-head gains do not replace the encoder comparison. [The completed R36 findings](#r36-completed) retain the controls, measured harms, and audit limits. Phase 2 remains unauthorized.

### Explicit withdrawals and limits

A40 withdrew the learned P2.4 forecasting claims. The forecaster used the later visit's image and mask features, plus the gap ending at that visit. Those inputs were not available at the earlier visit.

The withdrawal covers the reported in-domain gains, seed stability, SAILOR transfer, and forecasts after support-set adaptation. The old cache also skipped visits with missing masks, producing 123 LUMIERE rows rather than the audited 117.

The corrected history-only rerun had relative MAE 1.00595 on 117 rows, with a paired interval crossing zero. It was approximately tied with persistence, not a replacement success. That rerun preceded the corrected alignment protocol. At that audit stage, no aligned-feature, past-only retraining result was established. Later R33 and R37 onward are separate completed studies, not retroactive validation of P2.4. The target-input bug does not invalidate frozen P1, P2.2, or P2.3 results.

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

| Method | LUMIERE patient-uniform MAE | Relative | SAILOR patient-uniform MAE | Relative |
|---|---:|---:|---:|---:|
| Persistence | 1.545 | 1.000 | 0.943 | 1.000 |
| Mean delta | 1.552 | 1.005 | 0.947 | 1.004 |
| Linear trend | 2.322 | 1.504 | 1.365 | 1.447 |
| Current-volume ridge | 1.503 | 0.973 | 1.465 | 1.553 |
| Volume-history ridge | 1.589 | 1.029 | 1.617 | 1.715 |
| Current image | 2.047 | 1.325 | 3.182 | 3.374 |
| Mean image history | 2.127 | 1.377 | 4.936 | 5.235 |
| JEPA state | 2.152 | 1.393 | 2.719 | 2.883 |

**November 2, 2026 aggregation correction.** The original `outputs/p1_anatomy_baselines.json` contains two valid MAEs with different weighting. SAILOR JEPA's pair-weighted MAE is 3.0850704308634858 and pair-weighted persistence MAE is 1.0736978502204089, giving 2.8733134095687927. Their patient-uniform MAEs are 2.7186150919700562 and 0.943003840037004, giving **2.8829310937518304**. This is an aggregation mismatch, not rounding. `_evaluate_fold` saved a pair-weighted transfer ratio, while `_merge_folds` saved a patient-uniform LUMIERE ratio. The earlier table combined SAILOR's patient-uniform absolute errors with its pair-weighted ratios for every method. The table above now uses patient-uniform errors throughout. Historical values remain in the immutable records and in the [reconciled aggregate](p1_anatomy_reconciled.json). `report/reconcile_p1.py` checks the per-patient means and both ratio definitions against the original artifact without altering it. The correction does not change any method's direction against persistence.

The current-volume ridge's apparent LUMIERE improvement is unresolved. Its paired MAE difference is −0.042, with 95% confidence interval −0.227 to +0.148. It is worse on SAILOR, with difference +0.522 and interval +0.384 to +0.658.

JEPA is substantially worse than persistence, especially in transfer. These are development results for the next observed visit. They do not establish fixed-horizon forecasting or clinical-response prediction.

A38's physical-history residual MLP had LUMIERE MAE 1.5405 and SAILOR MAE 1.2143. Its apparent small LUMIERE improvement over persistence was unresolved; transfer was significantly worse. A39's frozen fusion branches did not add incremental signal. A40's future leakage means learned P2.4 metrics cannot be used as evidence for or against causal past-only forecasting.

## R35: observed state, observed change, and nuisance

<a id="r35"></a>

### R35 completed comparison

R35 is a completed comparison of MRI-only spatial state and observed-pair representations, originating in sibling worktree `world-model-representations` at revision `2f6e1f5a65be4ccbceac6de706b537de8b695faa`. Its report is the frozen [branch-only R35 record](documentation_sources/versions/768fdfa0a9be4dc76e99fcef9ff971e26252d672e82b08267f5e9ea157ced0f3.md), captured as S062 with branch and live-worktree origins.

The [archived R35 presentation](archive/2026-10-13/info/r35_representation_results.html) retains the full historical direct-head, transfer-budget, and nuisance tables. Its old review-status statements remain historical. The dated R35 closure below records the later accepted review.

The study used 65 encoder-training patients and 26 encoder-unseen LUMIERE patients, with an internal training and validation split. The seeds were 42, 43, and 44. The learned state encoder received dense supervision and native log-volume targets.

The seed-average confidence intervals below are descriptive patient bootstraps, not additional prespecified success criteria. Intervals are ordinary 95% intervals unless explicitly labeled family-12 adjusted. Neither seeds nor visits are independent bootstrap units.

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

In LUMIERE, the paired head had lower error than the later-scan-only head: MAE 1.2696 versus 1.4381, difference -0.1685. Its ordinary 95% interval [-0.3438, -0.0155] excludes zero, but its family-12 adjusted interval [-0.4258, +0.0495] crosses zero. On SAILOR, paired minus later-only error was -0.1446, with ordinary interval [-0.2546, -0.0331] and family-12 adjusted interval [-0.3001, +0.0224]. Thus neither cohort resolves this contrast after that adjustment. The paired head did not establish an advantage over independent state differences.

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

### R35 scoped result review closed on October 13, 2026

The [independent scientific result review](reviews/r35_result_review.md) returned `APPROVED_FOR_SCOPED_REPORT`. The [corrected artifact audit](reviews/r35_artifact_audit.md) returned `PASS_FOR_SCOPED_REPORT`. The [closure receipt](reviews/r35_closure_receipt.json) binds the exact reports, native runs, and [executed evidence](reviews/evidence/r35/README.md). R35 computation and scoped result review are closed. Lower in-domain automated-volume error is reportable; paired-model benefit, successful transfer, biological validity, and future forecasting are not established.

The reviewer found that the earlier audit overstated exact nuisance membership, reference provenance/budget checks, and source prediction-to-summary binding. Executed supplemental assertions resolved those defects with zero failures. The corrected report also retains separate numerical stdout, stderr, and exit receipts. Initial blocked reviews and the challenged audit remain unchanged historical evidence.

All 228 private manifest artifacts remained unchanged. The reference feature cache was absent from that original manifest; its current digest does not establish historical stability. Reference checkpoint checks compare recorded metadata, not newly hashed checkpoint bytes. Earlier MRI-byte provenance, new dense behavioral replay, independently replayed shuffle values, target reliability, and untouched validation remain unavailable. Raw saved review-status fields and archived reports retain their original dates; this separate receipt records accepted closure. No new experiment or Phase 2 is authorized.

## R36 observed-change evidence

<a id="r36"></a>

<a id="r36-completed"></a>

### Completed Phase 1 findings, October 13, 2026

All twelve training arms and thirty source-only probes completed the locked final-epoch recipe. Both development assessments completed, including bounded CPU recovery of interrupted seed-44 work. No GPU work was repeated for reporting closure. Each arm recorded 5,940 real-pair exposures, with 5,928 to 5,936 successful updates after disclosed AMP skips.

Both scans are observed inputs. This experiment tests access to observed change, not future prediction. C versus B is the primary objective-only comparison. A controls continued state-only training, original R35 is the initialization reference, and D adds nuisance constraints. All readouts use the matched source52 budget.

LUMIERE includes 149 visits from 26 encoder-unseen development patients and 117 pairs from 23 patients. SAILOR includes 269 visits and 242 pairs from 27 development subjects. No SAILOR fitting or checkpoint selection occurred. Neither cohort is untouched independent validation.

#### Fixed hypothesis classifications

Differences are candidate minus comparator. Intervals are the locked family-three adjusted paired-patient intervals. The report averages patient errors over seeds 42, 43, and 44, not predictions from an ensemble. It uses 10,000 patient-bootstrap draws and seed 2026. The intervals describe these fitted seeds, not unrestricted training-seed uncertainty.

| Endpoint | Difference | Adjusted interval | Classification |
|---|---:|---|---|
| C minus B frozen local occupancy MAE | -0.003067 | [-0.007995, +0.002644] | INCONCLUSIVE |
| C minus A direct native log-volume state MAE | -0.029774 | [-0.073244, +0.017074] | VERIFIED against the +0.05 retention allowance |
| D nuisance tradeoff | Trained-perturbation excess falls, but state retention is unresolved | State interval overlaps +0.05 | INCONCLUSIVE |
| Added observed temporal information | C and D beat target-only probes, but not independent-state differences conclusively | See the controls below | INCONCLUSIVE |

C minus B's raw 95% interval also crosses zero, at [-0.007162, +0.001430]. Failure to establish a gain is not proof of equivalence or universal representation failure. C's retention result is noninferiority on the specified direct-volume endpoint. It is not a verified improvement or retention of every anatomical capability.

#### All encoder arms and local controls

State columns are native `log1p`-volume MAE. Other columns are balanced signed-occupancy MAE. Values are three-seed mean patient-uniform errors. Companion tables are descriptive, not new selected hypothesis tests. A and original R35 have no graded nonlinear change head.

LUMIERE has zero-change occupancy MAE 0.484840.

| Arm | Direct state | Frozen difference | Independent-state difference | Target-only | Nonlinear change head |
|---|---:|---:|---:|---:|---:|
| A | 0.929272 | 0.376955 | 0.376293 | 0.437612 | not graded |
| B | 0.994212 | 0.381971 | 0.376508 | 0.440397 | 0.483508 |
| C | 0.899498 | 0.378904 | 0.375365 | 0.437862 | 0.270405 |
| D | 0.947002 | 0.381402 | 0.375873 | 0.435008 | 0.274940 |
| R35 | 1.114490 | 0.373396 | 0.375064 | 0.440490 | not graded |

SAILOR has zero-change occupancy MAE 0.469092.

| Arm | Direct state | Frozen difference | Independent-state difference | Target-only | Nonlinear change head |
|---|---:|---:|---:|---:|---:|
| A | 2.209396 | 0.423821 | 0.396249 | 0.480684 | not graded |
| B | 2.279716 | 0.436557 | 0.395125 | 0.483279 | 0.468779 |
| C | 2.117310 | 0.431323 | 0.391896 | 0.483676 | 0.342619 |
| D | 2.154923 | 0.436907 | 0.394481 | 0.490904 | 0.340042 |
| R35 | 2.240098 | 0.438596 | 0.397776 | 0.470256 | not graded |

C and D improve the descriptive nonlinear change-head error relative to B. That is a decoder result, not the primary frozen-encoder gain. On SAILOR, every frozen difference probe has higher descriptive error than its independent-state control.

The LUMIERE temporal controls use the same occupancy endpoint and 23 pair-bearing patients:

| Candidate minus control | Difference | Adjusted interval |
|---|---:|---|
| C minus target-only | -0.058958 | [-0.074268, -0.043641] |
| C minus independent-state difference | +0.003539 | [-0.008029, +0.013840] |
| D minus target-only | -0.053606 | [-0.068688, -0.038017] |
| D minus independent-state difference | +0.005528 | [-0.005650, +0.016473] |

Beating target-only is insufficient when the independent-state control is not beaten conclusively.

#### Source52 ridge companions

The following values are native log-volume state and change MAEs from source-only ridge fits. These are separate readouts from the direct state heads and occupancy probes above.

| Arm | LUMIERE state | LUMIERE change | SAILOR state | SAILOR change |
|---|---:|---:|---:|---:|
| A | 0.887015 | 1.189453 | 2.180392 | 0.983491 |
| B | 0.937617 | 1.230057 | 2.207182 | 1.006660 |
| C | 0.897726 | 1.165711 | 2.051489 | 0.969616 |
| D | 0.915186 | 1.171241 | 2.065202 | 0.963816 |
| R35 | 1.031088 | 1.273334 | 1.952590 | 0.980395 |

Zero native-volume change scores 1.544612 in LUMIERE and 0.943004 in SAILOR. Every SAILOR ridge-change mean exceeds zero change. These controls do not establish a transfer gain.

#### D's unresolved tradeoff includes measured harm

D minus C mean excess false change over identity, averaged across noise, intensity, and bias, is -0.000220 with adjusted interval [-0.000416, -0.000083]. Bias excess falls by -0.000691, with interval [-0.001278, -0.000277]. Noise excess rises by +0.0000301, with interval [+0.0000182, +0.0000436]. Held-out resolution excess is -0.000575, with interval [-0.001358, +0.000106].

D minus C direct-state error rises by +0.047505, with interval [+0.004175, +0.098686]. This establishes worsening. The interval overlaps the locked +0.05 allowance, so it does not demonstrate violation of that specific margin. The intended state-preserving tradeoff remains INCONCLUSIVE, not VERIFIED.

Per-sequence foreground standardization largely cancels the affine-intensity transform. Near-zero intensity drift is not evidence of learned intensity invariance. The locked stress is retained with this limitation rather than replaced after observing results.

#### Target and empty-compartment limits

The target audit retains all 599 visits and 496 usable pairs without model-selected exclusions. Sparse changed occupancy motivated balancing. It did not establish background dilution as the cause of earlier failures. One native enhancing disagreement disappeared completely after resizing. Partial boundary-detail loss remains unquantified.

Empty compartments still have large state errors. The reviewer reports C42 LUMIERE empty-compartment MAEs of 3.8969, 3.0033, and 7.9117 for core, enhancement, and edema. Empty-target coverage is 7, 22, and 5 LUMIERE visits, and 58, 28, and 1 SAILOR visits. Sparse support limits interpretation. No-positive AUPRC cases remain empty, not perfect scores.

The [aggregate companions](r36_phase1_metrics.json) retain changed and unchanged partitions, appearance and disappearance, both-empty pairs, source-fixed change-size groups, native scalar change, ridge readouts, reversal, and nuisance responses. Partition summaries are supported-pair pooled means, not the primary patient-uniform endpoint. A post-hoc within-patient permutation tests visit specificity, not biological change.

Correct affines and native volumes do not establish accurate segmentation or anatomical correspondence. The suspect extracranial contour remains in the primary analysis. Automated differences are not expert-verified biological change. No repeatability estimate or noise ceiling is available.

#### Independent review and scoped closure

The [result reviewer](reviews/r36_phase1_result_review.md) returned APPROVED_FOR_SCOPED_REPORT and reporting-patch OK. The [artifact audit](reviews/r36_phase1_artifact_audit.md) passed with zero failed assertions. The [closure receipt](reviews/r36_closure_receipt.json) binds both completed native runs and the unchanged recovered audit report.

The audit checked twelve checkpoints, thirty probes, all fifteen assessment sets, 1,950 nuisance rows, and thirty saved ridge bundles. Saved-coefficient replay and independent raw-row seed-mean bootstraps reproduced the classifications. Floating-point reduction differences of about 1e-9 did not change conclusions.

The audit binds declared tensor provenance. It does not newly replay every dense occupancy prediction, precision-recall or direction measure, or original MRI tensor. The [reporting rules and reproduction commands](METHODS.md#r36-phase1) retain that scope. The raw aggregation CLI still labels reports review-pending because it does not consume review evidence. The separate closure receipt records completed review.

Failed initial review runs were not accepted. The resumed native runs completed under their original contracts. Independent review cannot provide retrospective pre-training approval, expert biological validation, or untouched external validation. No forecast or general representation-quality claim follows, and Phase 2 remains unauthorized. The [original report](archive/2026-10-13/info/r36_phase1_report.md) and [execution chronology](archive/2026-10-13/info/r36_delta_representation_results.md) retain their exact bytes.

### Historical execution through October 11

Phase 1 was incomplete in the following dated receipts. The [completed findings](#r36-completed) supersede their open launch and review tasks, not their historical facts.

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


## MU cavity-state current-only forecasting adaptation, November 7, 2026

The single authorized adaptation completed: 167 observation contexts, 6,000 updates each for the current-only spatial model and simple volume control, all 58 matched MU selection targets, and a predictive timeline with both horizons. This remains reused development selection: 36 targets from 18 patients at 12 weeks and 22 targets from 13 patients at 24 weeks. No history model was trained and no reserved patients were opened. See [full aggregate measurements](patient_cavity_forecasting_metrics.json), [checkpoint/runtime/verification receipt](patient_cavity_forecast_build_metrics.json), and the [protocol](METHODS.md#mu-cavity-state-current-only-forecasting-adaptation-november-7-2026).

The resulting transition improves some scores over persistence of the improved observation state, but does not establish the useful predictive milestone. It has no established distribution advantage over the scalar control or matched improvement over the preserved current-only forecaster, and performs poorly when core is genuinely present.

| Native evaluation | 12-week persistence | 12-week current-only | 24-week persistence | 24-week current-only |
|---|---:|---:|---:|---:|
| Original three-compartment mean Dice | 0.3454 | 0.3892 | 0.2488 | 0.3760 |
| Four-compartment volume CRPS, mL (lower is better) | 10.2423 | 8.8003 | 16.4198 | 13.5597 |
| Enhancing Dice | 0.3454 | 0.3602 | 0.2218 | 0.2298 |
| Core Dice, all targets | 0.2117 | 0.3190 | 0.1102 | 0.4666 |
| FLAIR-abnormality Dice | 0.4791 | 0.4883 | 0.4143 | 0.4317 |
| Cavity Dice | 0.3462 | 0.3361 | 0.2822 | 0.2672 |
| Core Dice, nonempty reference only | 0.1828 | 0.0184 | 0.1361 | 0.0123 |
| Core Dice, empty reference only | 0.1389 | 0.8611 | 0.0556 | 0.8333 |

Native evaluation uses the full five-category reference, including cavity outside image support. The original-three Dice is computed from the native five-way map, not the conditional legacy map. All means weight patients equally after averaging their repeated targets; empty/nonempty subsets reweight their own contributing patients. At 12 weeks there are 21 nonempty-core targets from 15 patients and 15 empty-core targets from nine patients; at 24 weeks, 12 nonempty-core targets from eight patients and ten empty-core targets from nine patients.

Paired patient bootstrap intervals for current-only minus new-state persistence:

- Native original-three Dice: +0.0438 [−0.0237, +0.1119] at 12 weeks; +0.1273 [+0.0337, +0.2182] at 24 weeks.
- Four-compartment volume CRPS: −1.4420 [−2.2309, −0.5510] mL and −2.8602 [−4.4482, −1.0870] mL.
- Original-three volume-change MAE: +0.3397 [−0.4733, +1.2363] mL and +0.1806 [−1.0620, +1.4335] mL. Actual change-error improvement is not established.
- Decoded future-image lesion MAE: −0.00884 [−0.01422, −0.00318] and −0.01831 [−0.03638, −0.00647]. These are measured secondary image-error gains, not demonstrated useful future-image synthesis.

The simple volume control's four-compartment CRPS is 8.8460/13.6599 mL at 12/24 weeks. Spatial-minus-scalar CRPS is −0.0457 [−0.6067, +0.4672] and −0.1002 [−0.9251, +0.8178] mL: added spatial value for volume distributions is unresolved.

Matched historical comparisons separately retain the original ignored-cavity/image-support scoring convention. Current-only minus preserved current-only Dice is −0.00022 [−0.04516, +0.04534] at 12 weeks and +0.03976 [−0.01618, +0.10859] at 24 weeks. Matched three-compartment CRPS differences are +0.0619 [−0.5941, +0.6657] and +0.3714 [−0.6915, +1.3716] mL. Thus improved current anatomy has not established better forecasting than that frozen reference. Do not substitute native four-compartment aggregate errors for these matched historical comparisons.

Uncertainty remains inadequate. For nonempty core, nominal 90% interval coverage is 6.7%/0% at 12/24 weeks, with mean widths 0.354/0.347 mL. The scalar control covers 93.3%/100% with widths 5.13/7.60 mL on those same nonempty-core subsets. Spatial FLAIR-abnormality intervals average 98.95/100.02 mL wide with 68.1%/64.1% coverage. Cavity intervals average 12.14/13.10 mL with 76.4%/52.6% coverage. A single overall calibration mean obscures these different failures.

Both selected checkpoints are at update 500, well before the 6,000-update boundary; late bounded selection scores did not improve on them. No extension or seed repeat was launched. Scalar fitting took 92.6 seconds and spatial fitting 769.4 seconds; the successful restart through preparation, training, evaluation and timeline verification took 42.8 minutes. Earlier capacity interruptions occurred before cohort fitting and remain in private logs.

The final three timelines contain five visits. Verification checked 24 displayed maps and their volumes, six scenario distributions, four reference comparisons, and exact anatomy/image replays from all five saved observation states. The viewer distinguishes native and conditional legacy anatomy, model-assimilation comparisons and released-label comparisons, and representative displayed-map errors from distribution means. These are engineering consistency checks, not predictive acceptance. The observation checkpoint remains frozen; its approximately 2.9% LUMIERE missing-FLAIR lesion-MAE tradeoff remains visible without a retrospective acceptability threshold.


## Closure of the cavity-state spatial forecasting recipe, November 8, 2026

The operator closed `mu-cavity-current-structured-v1` without an unchanged extension, light retuning or seed repetition. Its November 7 measurements and selected checkpoints remain unchanged. Both selected updates were 500; no training-boundary evidence supports extension. See the [closure and preservation receipt](patient_cavity_forecast_closure.json).

On the same 12 nonempty-core targets from eight patients at 24 weeks, patient-weighted volume MAE/CRPS are 3.597/3.528 mL for the spatial model, 1.493/1.370 mL for the scalar model, and 1.281/1.281 mL for persistence. Spatial nominal 90% coverage is 0%, with 0.347 mL mean width; the scalar model covers 100%, with 7.599 mL width. These are descriptive subgroup results. They do not establish calibration or scalar superiority over persistence.

The completed recipe's selected spatial and scalar checkpoint copies were SHA256-verified on the laptop against their recorded identities. All 58 case records, aggregate metrics, run configurations/curves, source snapshots and logs are preserved in the private `outputs/mu-cavity-forecast-20261107/frozen-recipe` directory; GPU originals remain. The observation models, scalar references and frozen FLAIR candidate are retained. No reserved data were opened and no replacement fit was launched as part of closure.


## Forecast mixture-likelihood repair, November 8, 2026

The operator requested the cheapest concrete correction rather than more pre-fit documentation. Inspection found that `log(clamp_min(p, 1e-6))` gave zero cross-entropy gradient when the true tissue probability fell below the floor. The loss now uses stable log-softmax/log-sum-exp inside the cutoff-derived domain; inference and model parameters are unchanged. On one real fitting pair, 6,879 of 8,144 future-core coarse cells were below the old floor. Their true-core likelihood gradient changed from exactly zero to finite and nonzero. Ten focused tests passed on `mosaic-gpu`; the preserved checkpoint's real-pair inference remained exact.

One seed-17 candidate completed the original 6,000-update schedule, retaining the same data, architecture, other loss terms, and draws. Its initial 500-update subset pilot was diagnostic only. The fixed update-6,000 candidate was assessed once on all 58 existing MU selection targets; there was no checkpoint competition. All three nonempty-core conditions failed at both horizons. Expected core Dice was 0.0206 versus persistence 0.1828 at 12 weeks and 0.0000 versus 0.1361 at 24 weeks. Mean core-volume MAE was 2.863 versus 1.692 mL and 3.772 versus 1.281 mL, respectively; core CRPS also failed the frozen scalar comparison. No checkpoint was promoted. Saved-weight reload passed.

The numerical correction is retained, but useful forecasting was not recovered. This repair is not the proposed presence/amount/location redesign, does not isolate the remaining mechanism, and does not establish an information limit. The earlier recipe closure remains unchanged. No further run is queued; reserved patients remain unopened. Aggregate measurements and artifact locations: [repair results](patient_forecast_logspace_fix_metrics.json).


## Amount-first MU forecaster stopped at the quantity gate, November 8, 2026

`mu-amount-first-v1` completed the requested one-seed, 6,000-update terminal quantity fit on 87 fitting pairs and evaluated all 58 fixed selection targets. [Results and protocol](patient_amount_first_metrics.json). The selected cavity-aware observation system and scalar control stayed fixed; the new quantity backbone was initialized from the scalar and adapted. Presence and feasible native counts were trained by joint likelihood, with no spatial loss. The prior cavity forecasting closure is unchanged.

| Nonempty core, patient-averaged | 12 weeks | 24 weeks |
|---|---:|---:|
| Targets / patients | 21 / 15 | 12 / 8 |
| New predictive-mean volume MAE, mL | 2.088 | 0.819 |
| Persistence MAE, mL | 1.692 | 1.281 |
| New volume CRPS, mL | 2.007 | 2.026 |
| Frozen scalar CRPS, mL | 1.693 | 1.370 |
| New 90% interval coverage / width | 93.3% / 9.073 mL | 100% / 15.949 mL |
| Frozen scalar coverage / width | 93.3% / 5.128 mL | 100% / 7.599 mL |

The fixed quantity gate failed both conditions at 12 weeks and CRPS at 24 weeks. Outcome: **no eligible checkpoint**. Spatial fitting and spatial checkpoint selection did not start; only the explicitly discarded one-step location smoke ran. There is no claim about learned localization or full-model Dice. The 24-week MAE difference from persistence was −0.461mL (paired patient 95% interval −1.971 to 0.408); seven of eight patients had higher errors, so the favorable average does not establish consistent improvement. Corresponding CRPS difference from scalar was +0.656mL (−0.010 to1.601).

Core presence Brier/log scores at 12 weeks were 0.233/0.659, compared with fitting-only prevalence 0.233/0.659 and current-state presence Brier 0.319; at 24 weeks they were 0.254/0.701 versus prevalence 0.257/0.708 and current-state Brier 0.487. Deterministic current-presence log loss is explicitly infinite because of impossible observed events. These presence results and the interval coverage do not establish calibration. The JSON retains all four compartments, conditional-positive versus unconditional distributions, empty-compartment false positives, separate cavity results, patient-paired intervals and fixed-domain misses. Private per-target records remain with the checkpoint.

The real-data smoke verified 64 exact-count native maps, likelihood normalization/extreme tails, training and save/reload. Five numerical/conservation regression tests passed. A two-horizon diagnostic timeline verified 128 further maps using the exact evaluation samples, displaying predetermined sample 0 with its own measured volumes separately from means/intervals and presence probabilities. The real inference CLI reproduced both anatomical readouts and all four observation image outputs exactly, and matched evaluation counts. Quantity training took 28.34seconds after context preparation; one complete inference visit took 26.52seconds on mosaic-gpu. These runtimes cover this small frozen-context build, not observation training. Terminal weights and a compact viewer bundle are copied to `outputs/mu-amount-first-20261108`; the full state path remains on the GPU. The prior approximately 2.9% LUMIERE missing-FLAIR lesion-error tradeoff remains visible. Forty reserved patients remain unopened. No further fitting is queued.


## Saved amount-first core distributions, November 8, 2026

One bounded analysis used the unchanged 64 samples and exact saved presence probabilities for all 58 targets; no model, scan or new prediction was loaded. [Aggregate analysis](patient_amount_first_interpretation.json); private target and patient-horizon records are in `outputs/mu-amount-first-20261108/analysis/`. There are 36 targets/18 patients at 12 weeks and 22 targets/13 patients at 24 weeks. Every statistic below first averages targets within each patient and horizon/subset, then patients equally. Presence subsets overlap in patient membership.

| Core result | 12 weeks | 24 weeks |
|---|---:|---:|
| All-target mean-volume MAE: new / scalar / persistence, mL | 2.151 / 1.864 / 1.816 | 1.346 / 1.752 / 1.889 |
| All-target CRPS: new / scalar / persistence, mL | 1.635 / 1.520 / 1.816 | 1.200 / 1.180 / 1.889 |
| Future-present mean-volume MAE: new / persistence, mL | 2.088 / 1.692 | 0.819 / 1.281 |
| Future-present CRPS: new / scalar, mL | 2.007 / 1.693 | 2.026 / 1.370 |
| Future-empty expected false-positive volume, mL | 2.232 | 2.441 |

Core-presence probabilities averaged 0.553/0.552 for future-present/empty targets at 12 weeks and 0.562/0.561 at 24 weeks. The respective target-wide ranges were 0.524–0.598 and 0.533–0.601. All 58 cross the descriptive 0.5 threshold, including all 25 future-empty targets; sampling itself does not use that threshold. New presence Brier scores for all/present/empty targets were 0.233/0.200/0.306 at 12 weeks and 0.254/0.192/0.315 at 24 weeks. Fitting-only prevalence Brier was 0.233/0.257 across the respective full horizon populations.

On future-present core, average conditional positive medians were 1.464/1.910 mL, conditional means 5.122/7.146 mL, and observed amounts 3.099/3.766 mL at 12/24 weeks. Conditional median MAE was 2.047/2.312 mL and conditional mean MAE 3.761/3.380 mL. Conditional 90% coverage was 80%/75%, with width 36.860/32.238 mL. These are distributions conditional on predicted presence; future presence only defines evaluation membership.

Across all targets, the reported unconditional means averaged 2.646/3.100 mL, whereas unconditional medians averaged 0.220/0.324 mL. The largest four of 64 draws supplied 55.0%/62.6% of sampled volume on average; the single largest supplied 23.2%/31.9%. Largest individual core draws were 869.178/677.986 mL. At 24 weeks, removing the largest draw solely as a sensitivity calculation changes nonempty-core mean-MAE from 0.819 to 1.800 mL; official predictions and gates retain every draw. The original 24-week MAE advantage over persistence comes from one of eight patients; the other seven worsen, and their average difference is +0.262 mL.

**Numerical clarification:** one 24-week positive conditional interval had 40 positive draws. GPU float32 cumulative weights placed rank 38 just below 0.95, selecting rank 39 (24.276 mL) instead of rank 38 (8.071 mL). This analysis uses exact equal-weight ranks. The patient-averaged positive interval width is therefore 32.238 mL, superseding the earlier 34.263 mL for that conditional statistic only. Coverage, unconditional intervals, CRPS, mean errors and the failed quantity gate are unchanged. Original saved artifacts remain unmodified.


## Variance-matched Gaussian quantity fit failed the gate, November 8, 2026

`mu-amount-first-gaussian-v1` changed only the positive-count family from logistic to Gaussian in log-count space. It started from the original scalar initialization, not the fitted logistic model. Initial raw weights were identical; Gaussian standard deviation was logistic scale times pi/sqrt(3), matching log-space variance to 1.2e-7 relative error. One seed 17 fit completed 6,000 updates on the same 87 fitting pairs, using the same optimizer, sampling order and 64 uniform streams. All 58 selection targets were assessed. [Full official results, deterministic references and verification](patient_gaussian_count_metrics.json).

| Nonempty core, patient-averaged mL | 12 weeks | 24 weeks |
|---|---:|---:|
| Gaussian MAE: official 64 / deterministic CDF | 1.688 / 1.922 | 1.475 / 1.691 |
| Preserved logistic MAE: official 64 / deterministic CDF | 2.088 / 1.845 | 0.819 / 1.769 |
| Persistence MAE | 1.692 | 1.281 |
| Gaussian CRPS: official 64 / deterministic CDF | 1.996 / 1.961 | 2.014 / 1.957 |
| Preserved logistic CRPS: official 64 / deterministic CDF | 2.007 / 1.953 | 2.026 / 1.954 |
| Frozen scalar CRPS | 1.693 | 1.370 |

**Outcome: no eligible checkpoint.** The unchanged official gate failed CRPS at 12 weeks and both conditions at 24 weeks. No spatial fitting started. Deterministic references are diagnostic, not substituted selection rules. They also fail both quantity conditions at both horizons. The logistic model's apparent 24-week persistence MAE win and the Gaussian model's narrow 12-week MAE pass do not persist under their fitted CDF means. The Gaussian-minus-logistic deterministic nonempty-core MAE differences were +0.077 mL (paired 95% interval −0.346 to 0.620) and −0.078 mL (−0.549 to 0.725); CRPS differences were +0.008 mL (−0.004 to 0.027) and +0.002 mL (−0.005 to 0.012). No improved nonempty-core distribution is established.

Across all targets, deterministic core MAE changed from 2.005 to 1.916 mL at 12 weeks and 2.282 to 1.903 mL at 24 weeks; deterministic CRPS changed from 1.603 to 1.606 and 1.225 to 1.217 mL. The mean absolute discrepancy between official 64-sample and fitted-CDF means fell from 1.361 to 0.557 mL and 1.034 to 0.543 mL. The largest four samples' average volume share fell from 55.0% to 50.8% and 62.6% to 56.2%; maximum draws fell from 869/678 to 311/241 mL. Tails remain consequential. Gaussian positive-core conditional 90% coverage/width was 73.3%/22.360 mL and 75%/23.855 mL, using exact equal-weight ranks within saved positive samples. Narrower intervals do not establish calibration.

Presence ranking was unchanged: patient-balanced AUC 0.630 at 12 weeks and 0.576 at 24 weeks for both families. Gaussian all-target Brier was 0.233/0.254, and every future-empty target still exceeded the descriptive 0.5 presence threshold. On official nonempty-core scores,10/15 and 5/8 patients had worse mean-volume error than persistence;5/15 and 5/8 had worse CRPS than scalar. Deterministic counts were 8/15 and 5/8 for MAE, and 4/15 and 3/8 for CRPS. Private records retain all 58 targets and 31 patient-horizon rows, including empty/nonempty subsets, under `outputs/mu-gaussian-count-20261108/comparison/`.

Nine numerical/conservation tests passed, covering matched initialization, tail-safe Gaussian sampling, finite nonzero extreme-tail likelihood gradients, and deterministic CDF scores against exhaustive small-support probability masses. All 116 family/target sample sets replayed exactly. The Gaussian real-data smoke verified 64 native maps; its diagnostic timeline verified 128 more maps with unchanged observation/completion artifacts. Quantity training took 31.40 seconds; deterministic evaluation took 5.53 seconds after loading, using all feasible integer counts and all 16 presence patterns per target. There was no probability floor, sample clipping, additional evaluation sampling, history fit or reserved-patient access. The selected observation model, scalar control and previous closures remain unchanged. No extension, family sweep or further fitting is queued.


## Frozen-state information program: no eligible predictive configuration, November 8, 2026

Completed the authorized 24 terminal quantity fits: eight conditions, seeds 17/23/31, 6,000 updates each. Restoring 104 development visits recovered the full 156-pair/56-patient fitting population (97 pairs/48 patients at 12 weeks; 59/34 at 24). The original 87 pairs remain an unchanged subset; all 58 selection targets remain unchanged. Forty reserved patients remain unopened. These are reused development-selection results.

Values below are patient-balanced means of three seeds; slash-separated values denote 12/24 weeks. Core MAE and CRPS use the fitted discrete CDF on future-present core (21 targets/15 patients and 12 targets/8 patients). Joint NLL covers all targets and compartments.

| Condition | Joint NLL ↓ | Core mean-volume MAE, mL ↓ | Core CRPS, mL ↓ |
|---|---:|---:|---:|
| Original pool, scalar | 40.799 / 39.028 | 1.924 / 1.692 | 1.961 / 1.957 |
| Original pool, current MRI | 1205.930 / 705.066 | 2.124 / 3.372 | 2.064 / 1.918 |
| Expanded pool, scalar | 38.796 / 37.899 | 1.866 / 1.787 | 2.036 / 2.112 |
| Expanded pool, current MRI | 515.419 / 539.858 | 1.955 / 3.166 | 1.727 / 1.709 |
| Current-reference volumes (diagnostic) | 38.466 / 37.606 | 1.698 / 1.716 | 2.035 / 2.276 |
| Latest MRI + history metadata | 428.526 / 525.034 | 1.963 / 2.512 | 1.747 / 1.564 |
| + Previous volumes and rates | 513.867 / 588.870 | 1.898 / 2.614 | 1.677 / 1.446 |
| + Previous MRI | 1428.747 / 1411.352 | 1.819 / 1.385 | 1.646 / 1.201 |

**No deployable configuration passed the required seed-17 and three-seed-average gates. No spatial fit or new predictive timeline was started.** Persistence core MAE was 1.692/1.281 mL; the frozen scalar CRPS was 1.693/1.370 mL. Full-memory seed 31 alone passed both horizons, but choosing it would violate the fixed integration rule.

**Data and MRI contrasts.** Expanded scalar fitting reduced NLL by 2.003 at 12 weeks (paired patient 95% interval −4.787 to −0.014) and 1.129 at 24 (−2.719 to 0.112), without passing quantity gates. Adding current MRI worsened NLL in both pools at both horizons; all four paired intervals excluded zero. More fitting data reduced this MRI-associated deterioration more at 12 weeks (interaction −688.508; interval −1348.160 to −144.141); the 24-week interaction interval included zero.

**Measurement diagnostic.** Overall core MAE at 12/24 weeks was 1.816/1.889 mL for estimated-current persistence, 1.960/2.177 for its learned scalar head, 1.459/1.060 for reference-current persistence, and 1.689/1.544 for the privileged learned head. Reference-current learning remained worse than reference persistence on overall core MAE: paired differences +0.230 [0.028, 0.406] and +0.484 [0.210, 0.661] mL. Reference persistence is not an upper bound: on future-present core at 24 weeks its MAE was 1.416 versus 1.281 for estimated persistence. Signed measurement/change terms and their cancellation are retained in the comparison.

**Memory and uncertainty.** Full memory lowered mean core CRPS relative to metadata-matched latest state by 0.101/0.363 mL, but both patient intervals crossed zero; leave-one-patient-out differences could reverse sign. Adding past MRI beyond past volumes improved only 2/8 patients on 24-week nonempty-core CRPS. Full-memory joint NLL instead worsened by 1000.221/886.318 versus latest state, with positive paired intervals. Conditional amount NLL contributed 1427.254/1409.186 of total 1428.747/1411.352. Its nominal 90% present-core coverage was 30.4%/25.0%, with widths 2.179/2.807 mL; FLAIR-abnormality coverage was 5.1%/11.1%. These are severely overconfident distributions, not calibrated forecasts.

**Execution and artifacts.** Four focused checks and the real-data train/save-reload/score smoke passed. The original scalar seed 17 reproduced the earlier deterministic core scores. Array restoration took 214.3 seconds, descriptor extraction about 458.0 seconds including its initial smoke, and all 24 fits/evaluations took 792.3/76.2 seconds; these stage timings exclude archive transfer and can overlap. Only 2.097 GB of selected archive members were sent to the GPU. The central equal-weight quantile repair changed 14 logistic and 12 Gaussian conditional interval widths without changing coverage or historical decisions. Corrected reports remain separate from originals.

The consolidated private artifact is `outputs/mu-information-program-20261108/comparison.json`, with per-seed scores, patient effects, paired intervals, leave-one-patient-out sensitivity, all compartments, empty/nonempty strata, coverage/width, tail summaries, signed decomposition, source identities and checkpoint hashes. Reports, 24 terminal checkpoints, the scalar control and manifests were copied to the laptop through an explicit allowlist; prepared tensors and MRI descriptors remain on mosaic-gpu. The frozen observation system and its approximately 2.9% missing-FLAIR lesion-error tradeoff remain unchanged. No additional fit is queued.

## Regularized scalar versus PCA-MRI quantity study failed both gates, November 8, 2026

The approved two-arm study completed on 156 fitting pairs from 56 patients and all 58 unchanged selection targets. S uses current estimated quantities, horizon and six ROI-empty flags; M adds eight fold-local PCA components of the same frozen MRI state. Nested patient CV selected penalty 0.01 for S and 0.1 for M. Both final models were locked before selection, with no location refitting after attaching excluded-patient scales. [Full aggregate](patient_linear_quantity_metrics.json) and [protocol](METHODS.md#regularized-scalar-versus-pca-mri-quantity-study-november-8-2026).

| Nonempty-core measurement, mL | 12 weeks: 21 targets / 15 patients | 24 weeks: 12 targets / 8 patients |
|---|---:|---:|
| S exact mean MAE | 5.177 | 11.065 |
| M exact mean MAE | 5.210 | 13.662 |
| Persistence MAE | 1.692 | 1.281 |
| S exact CRPS | 1.924 | 2.571 |
| M exact CRPS | 2.117 | 2.686 |
| Frozen full scalar CRPS | 1.693 | 1.370 |

**Neither arm is eligible.** Both conditions failed at both horizons; no spatial training followed. For either arm, 13/15 and 8/8 patients had worse nonempty-core mean MAE than persistence, and 13/15 and 7/8 had worse CRPS than the scalar. M-minus-S nonempty-core CRPS was +0.193 mL (paired 95% interval −0.109 to +0.659) and +0.115 (−0.391 to +0.614). Its mean-MAE contrasts were +0.033 (−3.023 to +3.039) and +2.597 (−2.553 to +8.278). Patient effects and leave-one-patient-out ranges are retained; MRI has not established added value in this fixed representation/readout.

All-target joint NLL was S 38.125/37.537 versus M 38.474/37.498 at 12/24 weeks. M-minus-S was +0.349 (0.073 to 0.651) and −0.039 (−0.493 to 0.365); the corresponding leave-one-patient-out ranges were 0.254–0.412 and −0.127–0.128. Core Brier was S 0.193/0.254 versus M 0.193/0.236; AUC was S 0.739/0.665 versus M 0.746/0.654. The Brier contrasts did not exclude zero. All/empty/nonempty core, original compartments, separate cavity, presence controls, full weighted scalar controls and outside-domain accounting are in the aggregate. Other-compartment interval and volume reports use 64 shared coherent samples; core gates use the fitted CDF.

Outer excluded-patient joint NLL at training sigma 1 was 36.805 for S and 37.409 for M. Final five-fold penalty-selection NLL was 36.689/37.409. Attached scales in core/enhancing/cavity/FLAIR order were S [2.242, 1.310, 1.942, 0.973] and M [2.262, 1.449, 2.131, 0.984]. In-sample scale-calibration NLL on those outer-excluded locations was 35.519/35.799; that is not independent uncertainty validation. Broad core intervals accompanied large exact means: sample-based nonempty-core 90% coverage/width was S 100%/34.56 mL and 100%/52.11 mL, versus M 86.7%/25.89 mL and 100%/87.30 mL. Coverage alone does not rescue the failed quantity forecasts.

Three focused numerical/preprocessing checks and the real-data complete-fit, reload, feasible-sampling and exact-CDF smoke passed. All 1,704 final head/scale solves met the fixed criterion, with maximum gradient infinity norm 7.26e-7. Early numerical stopping failures and the fitting-only arithmetic/solver corrections remain documented; no selection result informed them. The completed nested fit took 32.55 seconds and locked assessment 6.85 seconds, excluding implementation, earlier failed attempts and transfer. Final checkpoints, transforms, CV/fold assignments, excluded-patient predictions, paired patient results and code are preserved under `outputs/mu-linear-quantity-20261108/`; the GPU root has the same basename under `/home/aditya/experiments/`. `scripts/predict_linear_quantity.py` provides cutoff-only inference. The observer, completion tradeoff, prior closures and forty reserved patients are unchanged.

## Acquisition JEPA implementation prepared; GPU smoke pending, November 8, 2026

The approved acquisition study has frozen R/P/V/A roles and implemented the shared single-sequence encoder, observed-set fusion, separate hypothetical feature predictor, EMA target, joint losses, policy descriptors/selection, locked assessment and saved-state viewer episode paths. [Preparation status](patient_acquisition_jepa_metrics.json) and [fixed protocol](METHODS.md#acquisition-conditioned-jepa-development-study-november-8-2026). R has134 patients/538 visits; P20/48; V16/58; A16/48. Eligible patients were not reduced to the retained longitudinal cache.

Three focused checks passed on mosaic-gpu: unavailable-array/support isolation, order-independent set fusion, and all-observed/unknown-cavity/native-cavity supervision. The external OpenMind initialization was restored and its original SHA256 verified. Most source images had been archived during cleanup; exact R/P/V restoration is in progress. A sources have not been restored or opened for this study. The real-data GPU smoke, throughput estimate, representation fit, selector fit, assessment and viewer examples remain pending. During preparation other users' jobs left 940–8827MiB free GPU memory (8827MiB at the latest check), below the prior observation smoke's12.6GB allocation. No training or viewer was launched, and other jobs were left untouched. This is an execution status, not a JEPA capability result.

### Acquisition JEPA execution resumed, November 8, 2026

All 3,221 R/P/V source files (13,744,950,501 bytes) were restored and prepared; the frozen role membership and all prepared paths were checked. The two-update real-data smoke passed checkpoint reload, saved-state decoding, unavailable-input isolation, reveal-order invariance and hypothetical-feature separation. Its initial assertion incorrectly required raw decoder output to equal window-blended output bit for bit; the corrected check preserves exact save/reload equality and measures blending error (maximum 2.38e-7). The successful smoke averaged 2.377 seconds/update with 11,882,268,672 bytes peak CUDA allocation, projecting 8.25 hours for 12,500 updates before selection/evaluation overhead. A fresh seed-17 fit from the external MRI initialization has started, with P/V policy fitting queued afterward. A and reserved patients remain unopened. These are execution measurements, not evidence of acquisition-policy capability.

## Acquisition JEPA completed: no adaptive acquisition gain, November 9, 2026

The fixed seed-17 study completed all 12,500 representation updates, P/V policy selection, and full-volume assessment on all 48 A visits from 16 patients. Preprocessing used P only; checkpoint, selectors, fixed order and population were locked before A restoration. A is reused development data, not independent confirmation. Reserved and separate MU scoring patients remained unopened. [Full aggregate measurements](patient_acquisition_jepa_metrics.json); private patient-level records remain under outputs/acq-jepa-v1 and on mosaic-gpu.

Fixed, direct and JEPA policies took the **same actual path on every target: T1 → CT1 → FLAIR**. All scored 0.6253, averaging patient-level anatomy utility at two and three observed sequences (0.5900 and 0.6605). Both JEPA contrasts were exactly 0 with 95% paired-patient intervals [0, 0]. Oracle utility was 0.6271; oracle-minus-fixed was 0.0018 [0.0001, 0.0041]. This is headroom under this trained observer and released offline reveal task, not a conclusion that MRI acquisition has no value. The fixed acquisition criterion failed.

Native all-four anatomy Dice was 0.6718 versus 0.7177 for the preserved cavity-aware reference; difference -0.0459 [-0.0702, -0.0222]. Anatomy was statistically worse than reference, but the interval straddles the predefined −0.05 noninferiority margin. Retention did not pass acceptance and remains not established; inferiority beyond that margin was not established either. Present-compartment Dice (new/reference) was enhancing 0.6837/0.7695, core 0.3927/0.4179, FLAIR abnormality 0.8310/0.8714 and cavity 0.5781/0.5833. Empty-core false-positive volume was 1.976/1.179 mL (21 visits, 11 patients); empty-cavity false positives were 8.908/4.516 mL (7 visits, 3 patients). The sole empty-enhancing target had 6.834/2.521 mL. All new-model empty-compartment targets had some false-positive tissue; no FLAIR-empty targets existed. The small cavity/enhancing subgroups limit interpretation.

Missing-FLAIR lesion MAE improved and passed the predefined +0.05 retention margin for every fixed panel. Negative differences favor the new model; intervals are paired by patient. Whole-brain MAE also improved on all four panels (see JSON).

| Available panel | New lesion MAE | Reference lesion MAE | Difference [95% interval] |
|---|---:|---:|---|
| T1 | 0.7645 | 1.0447 | -0.2803 [-0.3674, -0.1935] |
| T1 + CT1 | 0.7451 | 0.9129 | -0.1678 [-0.2303, -0.1063] |
| T1 + T2 | 0.5227 | 0.6565 | -0.1338 [-0.1686, -0.0997] |
| T1 + CT1 + T2 | 0.5270 | 0.5879 | -0.0609 [-0.0926, -0.0345] |

The preserved observer is a capability reference trained on a broader population, not a causal JEPA control. One seed and a terminal selected checkpoint (step 12,500) limit conclusions; no extension or retuning followed A. Training took 6.36 hours, P/V enumeration 44.6/51.6 minutes and A evaluation 61.7 minutes. Peak training allocation was 11.96 GB. Four-sequence encoding averaged 12.84 seconds and the shared anatomy/image readout 1.58 seconds. State payloads were 6.44/9.65/12.87/16.09 GB for one/two/three/four observed sequences. They are computed on demand, not cached for the cohort.

Two predetermined V episodes are complete and visually checked in the existing viewer; actual and hypothetical evidence are explicitly separate. Six written maps matched displayed native volumes, and anatomy/completion identify the same saved state. The first episode passed saved-state reload and unavailable-sequence perturbation checks. A host-memory OOM interrupted the second episode after two steps; only its missing final step was resumed from mapped saved storage, without repeating assessment. Initial/update timings lost in that interruption are explicitly unavailable; the resumed update took 3.54 seconds. Both examples occupy 58.71 GB. A reference-input CPU/CUDA mismatch was corrected before any completed A target was saved. Temporary viewer processes were stopped. The full code, selected checkpoint, locks, scores and saved examples are preserved; the laptop has the report, private measurements, code snapshot and checksums.
