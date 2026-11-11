#set text(lang: "en", font: "IBM Plex Serif", size: 10.5pt, fill: rgb("#1a1a1a"))
#set page(paper: "a4", margin: (top: 2.2cm, bottom: 2.2cm, left: 2.2cm, right: 2.2cm), numbering: "1")
#set par(justify: true, leading: 0.72em, spacing: 1.12em)
#show heading.where(level: 1): it => block(above: 1.7em, below: 0.75em)[#set text(size: 1.32em, weight: 700, fill: rgb("#111111")); #it.body]
#show heading.where(level: 2): it => block(above: 1.25em, below: 0.55em)[#set text(size: 1.1em, weight: 600, fill: rgb("#222222")); #it.body]
#show link: it => text(fill: rgb("#1d4ed8"), font: "IBM Plex Sans")[#underline(it)]
#show raw: it => text(font: "IBM Plex Mono", size: 0.88em)[#it]
#set heading(numbering: "1.")
#set figure.caption(position: bottom)
#set table(stroke: (x, y) => if y == 0 { (top: 1pt + rgb("#222222"), bottom: 0.6pt + rgb("#222222")) } else { (bottom: 0.4pt + rgb("#dddddd")) }, inset: 5pt, align: left)
#show table.cell.where(y: 0): set text(weight: "bold")

#let blue = rgb("#0f4d92")
#let red = rgb("#991b1b")
#let muted = rgb("#555555")
#let r36 = json("r36_phase1_metrics.json")
#let r37 = json("r37_full_metrics.json")
#let r38 = json("r38_full_metrics.json")
#let r39 = json("r39_treatment_metrics.json")
#let fmt(x, d: 4) = {
  let neg = x < 0
  let v = calc.abs(calc.round(x, digits: d))
  let s = str(v)
  let parts = s.split(".")
  let frac = if parts.len() > 1 { parts.at(1) } else { "" }
  while frac.len() < d { frac = frac + "0" }
  (if neg { "−" } else { "" }) + parts.at(0) + "." + frac
}
#let sfmt(x, d: 4) = (if x >= 0 { "+" } else { "" }) + fmt(x, d: d)

#align(center)[
  #text(size: 16.5pt, weight: "bold")[Separating state estimation, observed change, and future forecasting in longitudinal glioblastoma MRI]
  #v(0.45em)
  #text(size: 10pt, fill: muted)[Workshop manuscript draft · October 30, 2026]
]

#v(0.35em)
#block(inset: (left: 1.1cm, right: 1.1cm), stroke: (left: 2pt + rgb("#cccccc")))[
  #text(weight: "bold")[Abstract.] Longitudinal MRI studies of post-operative glioblastoma often mix three tasks: estimating anatomical state at one visit, measuring change between two observed scans, and forecasting a future scan from past visits alone. Across five controlled studies on the LUMIERE and SAILOR development cohorts (R35 through R39), we test where 3D spatial representations help and why next-visit forecasting fails to beat persistence controls. In R35, a 3D spatial state model lowers in-domain automated compartment volume MAE compared with a scalar-supervised frozen reference (1.1145 versus 1.5000 in `log1p(mm3)`). In R36, balancing changed against unchanged voxels improves a trained nonlinear change decoder on observed scan pairs (balanced occupancy MAE #fmt(r36.cohorts.lumiere.C.local.direct_change.patient_uniform_balanced_mae) versus #fmt(r36.cohorts.lumiere.B.local.direct_change.patient_uniform_balanced_mae)). However, the matched frozen-encoder linear probe difference remains inconclusive (#sfmt(r36.comparisons.at("C-B").difference), adjusted 95% interval #fmt(r36.comparisons.at("C-B").family_adjusted_ci.at(0)) to #sfmt(r36.comparisons.at("C-B").family_adjusted_ci.at(1))). When forecasting an unseen future visit, multi-visit spatial history fails to beat a latest-scan-only predictor or calibrated persistence under both a frozen encoder (R37, CRPS #fmt(r37.arm_scores.H.crps) for history versus #fmt(r37.arm_scores.L.crps) for latest-only and #fmt(r37.arm_scores.P.crps) for persistence) and end-to-end joint training (R38, CRPS #fmt(r38.cohorts.primary.scores.JH.seed_score_average.crps.mean) for joint history versus #fmt(r38.cohorts.primary.scores.OH.seed_score_average.crps.mean) for detached history). Backpropagating forecast gradients into the 3D encoder degrades current-visit state accuracy, and conditioning on recorded cutoff treatment status in SAILOR does not improve forecasts (R39, CRPS difference #sfmt(r39.comparisons.HT_vs_H.difference)). Single-scan state error and through-plane resolution sensitivity dominate predictable multi-visit dynamics in these cohorts.
]

= Three distinct longitudinal questions

Longitudinal medical imaging studies often treat temporal representation learning as one problem. In post-operative glioblastoma surveillance, the inputs available at evaluation define three distinct questions:

1. *Single-visit state estimation.* The model maps one multimodal MRI study $x_t$ to anatomical quantities at visit $t$, such as necrotic, enhancing, and edema volumes or segmentation masks.
2. *Observed-pair change estimation.* The model takes two acquired studies $(x_t, x_(t+1))$ and estimates the anatomical difference between visit $t$ and visit $t+1$. Because $x_(t+1)$ is an input, this task measures pair comparison rather than future prediction.
3. *Future forecasting.* The model takes only information available up to cutoff visit $t$, written $x_(<=t)$, and predicts a probability distribution over targets at the next observed visit $t+1$. The predictor never sees $x_(t+1)$, future masks, future treatment, or the unelapsed inter-visit interval.

Prior work establishes strong controls for the first two questions. PaIRNet and LILAC compare pair-trained networks against subtracting independent single-visit predictions on longitudinal brain MRI @kim2024pairnet @kim2025lilac. BrainDiff uses frozen-feature probes to show that interval change is harder to decode than single-study pathology @patel2026braindiff. LSOR evaluates longitudinal objectives by freezing encoders and fitting matched downstream heads @ouyang2023lsor. In glioblastoma MRI, Rudie et al., Matoso et al., STRIDE, and TRACE study scan subtraction, interval-conditioned fusion, and concept bottlenecks on observed scan pairs @rudie2022glioma @matoso2025gbm @stride2026 @trace2026.

#figure(
  table(columns: (0.65fr, 1.75fr, 2.2fr, 1.6fr),
    [Study], [Inputs at assessment], [Primary question], [Outcome],
    [R35], [$x_t$ (state) or $(x_t, x_(t+1))$ (pair)], [Does dense 3D state or pair training beat scalar or subtraction controls?], [State MAE improves in-domain. Pair model matches state subtraction.],
    [R36], [Observed pair $(x_t, x_(t+1))$], [Does change-balanced supervision improve a frozen-encoder local linear probe?], [Nonlinear head improves descriptively. Frozen probe is inconclusive.],
    [R37], [Past scans $x_(<=t)$ (frozen encoder)], [Does multi-visit spatial history beat latest-scan-only and persistence?], [Inconclusive. Persistence and empirical change lead.],
    [R38], [Past scans $x_(<=t)$ (joint $2 times 2$)], [Does joint encoder-dynamics training unlock history or beat detached training?], [Inconclusive. Joint gradients degrade current-visit state.],
    [R39], [Decoded state $hat(y)_(<=t)$ + treatment], [Does cutoff treatment status improve next-visit volume forecasts on SAILOR?], [Inconclusive. Measured-mask persistence dominates.],
  ),
  caption: [Summary of the five studies. R35 and R36 observe both scans of a pair. R37, R38, and R39 forecast an unseen future visit under a strict information cutoff.],
)

This paper connects all three questions in one controlled sequence on post-operative glioblastoma MRI (Table 1). Rather than proposing a clinical forecaster, we report a diagnostic result. Even when 3D spatial state estimation (R35) and change-balanced decoder training (R36) succeed on observed scans, neither frozen-state recurrence (R37), joint end-to-end training (R38), nor cutoff treatment conditioning (R39) beats calibrated persistence for forecasting the next visit.

= Data, targets, and evaluation rules

We use two longitudinal glioblastoma MRI cohorts for development and transfer testing:
- *LUMIERE.* This cohort provides four sequences (`CT1`, `T1`, `T2`, `FLAIR`) resampled to a $96^3$ grid with nonzero z-score normalization and affine-aligned DeepBraTumIA automated masks for necrotic core, enhancing tumor, and edema. Fixed patient splits assign 52 patients to source training, 13 patients to validation and calibration, and 26 encoder-unseen patients to assessment.
- *SAILOR.* This cohort provides multi-visit post-operative MRI from 27 reused development-transfer patients (242 consecutive session pairs) with automated masks and recorded per-visit treatment status (`CRT`, `TMZ`, `no`, `unknown`).

Volume targets and errors use native mask coordinates transformed as `log1p(mm3)`. Every neural experiment trains three random seeds (`42`, `43`, `44`) with identical per-seed initialization and patient-uniform sampling orders. Scoring averages compartments first, averages visit pairs within each patient second, and weights patients equally. We average metric scores across seeds rather than ensembling predictions. Confidence intervals use 10,000 paired patient-level bootstrap draws with seed 2026.

Two scope limits apply to every measurement in this paper. First, automated segmentation masks are development proxies rather than expert-adjudicated biological truth. Second, the LUMIERE assessment patients and SAILOR transfer cohort are reused across R35 through R39. They measure controlled algorithmic contrasts, not untouched external generalization.

= State estimation and observed change (R35 and R36)

== R35: single-visit state versus paired-scan change

R35 trained a 3D convolutional encoder-decoder (`widths = (16, 32, 64, 128)`) on LUMIERE source visits using dense compartment mask supervision and native log-volume regression. On 149 visits from 26 unseen LUMIERE patients, the 3D spatial state model reached a patient-uniform volume MAE of 1.1145 `log1p(mm3)`, compared with 1.5000 for a frozen BRAINIAC reference probe trained with scalar volumes alone. Because R35 changed both the spatial representation and dense mask supervision, this gain reflects the combined training recipe rather than architecture alone.

On 117 consecutive observed scan pairs from 23 LUMIERE patients, a dedicated two-scan paired model (MAE 1.2696) did not beat subtracting two independent single-visit state predictions (MAE 1.2589; paired-minus-independent difference +0.0108, 95% bootstrap interval −0.0791 to +0.1066), although both beat predicting zero change (MAE 1.5446). Consistent with LILAC @kim2025lilac, independent state subtraction is a strong baseline that paired models must clear.

== R36: change-balanced supervision on observed pairs

Changed voxels occupy less than 0.5% of the $96^3$ grid in median LUMIERE pairs. R36 tested whether equal weighting of changed and unchanged voxel regions makes observed signed-occupancy change easier to extract. Starting from the R35 state checkpoints, four arms trained for 20 epochs across three seeds: `A` continued state-only supervision, `B` applied an unweighted whole-grid signed-occupancy change loss, `C` weighted changed and unchanged voxel groups equally, and `D` added acquisition-nuisance consistency penalties to `C`.

#figure(
  table(columns: (2.4fr, 1fr, 1fr, 1fr, 1fr),
    [Evaluation readout and cohort], [Arm A], [Arm B], [Arm C], [Arm D],
    [LUMIERE trained nonlinear change head], [none], [#fmt(r36.cohorts.lumiere.B.local.direct_change.patient_uniform_balanced_mae)], [#fmt(r36.cohorts.lumiere.C.local.direct_change.patient_uniform_balanced_mae)], [#fmt(r36.cohorts.lumiere.D.local.direct_change.patient_uniform_balanced_mae)],
    [LUMIERE frozen-encoder local linear probe], [#fmt(r36.cohorts.lumiere.A.local.frozen_probe.patient_uniform_balanced_mae)], [#fmt(r36.cohorts.lumiere.B.local.frozen_probe.patient_uniform_balanced_mae)], [#fmt(r36.cohorts.lumiere.C.local.frozen_probe.patient_uniform_balanced_mae)], [#fmt(r36.cohorts.lumiere.D.local.frozen_probe.patient_uniform_balanced_mae)],
    [LUMIERE independent decoded-state difference], [#fmt(r36.cohorts.lumiere.A.local.independent_state.patient_uniform_balanced_mae)], [#fmt(r36.cohorts.lumiere.B.local.independent_state.patient_uniform_balanced_mae)], [#fmt(r36.cohorts.lumiere.C.local.independent_state.patient_uniform_balanced_mae)], [#fmt(r36.cohorts.lumiere.D.local.independent_state.patient_uniform_balanced_mae)],
    [SAILOR frozen-encoder local linear probe], [#fmt(r36.cohorts.sailor.A.local.frozen_probe.patient_uniform_balanced_mae)], [#fmt(r36.cohorts.sailor.B.local.frozen_probe.patient_uniform_balanced_mae)], [#fmt(r36.cohorts.sailor.C.local.frozen_probe.patient_uniform_balanced_mae)], [#fmt(r36.cohorts.sailor.D.local.frozen_probe.patient_uniform_balanced_mae)],
    [SAILOR independent decoded-state difference], [#fmt(r36.cohorts.sailor.A.local.independent_state.patient_uniform_balanced_mae)], [#fmt(r36.cohorts.sailor.B.local.independent_state.patient_uniform_balanced_mae)], [#fmt(r36.cohorts.sailor.C.local.independent_state.patient_uniform_balanced_mae)], [#fmt(r36.cohorts.sailor.D.local.independent_state.patient_uniform_balanced_mae)],
  ),
  caption: [R36 patient-uniform balanced signed-occupancy MAE (3-seed mean; lower is better; bound to `r36_phase1_metrics.json`).],
)

Table 2 separates the co-adapted nonlinear decoder from the frozen encoder representation. Through each arm's trained nonlinear change head, equal-group weighting in `C` lowered LUMIERE balanced occupancy MAE from #fmt(r36.cohorts.lumiere.B.local.direct_change.patient_uniform_balanced_mae) (`B`) to #fmt(r36.cohorts.lumiere.C.local.direct_change.patient_uniform_balanced_mae) (`C`), and lowered SAILOR MAE from #fmt(r36.cohorts.sailor.B.local.direct_change.patient_uniform_balanced_mae) to #fmt(r36.cohorts.sailor.C.local.direct_change.patient_uniform_balanced_mae). Arm `C` also passed the direct state-volume retention margin relative to `A` (difference #sfmt(r36.at("state_C-A").difference) `log1p(mm3)`). When all encoders were frozen and re-probed with an identical local linear readout fitted on the 52 source patients, the primary `C - B` difference was #sfmt(r36.comparisons.at("C-B").difference) (family-adjusted 95% interval #fmt(r36.comparisons.at("C-B").family_adjusted_ci.at(0)) to #sfmt(r36.comparisons.at("C-B").family_adjusted_ci.at(1))). Most of `C`'s gain resided in the co-adapted nonlinear change decoder rather than linearly accessible encoder features, and neither `C` nor `D` beat independent decoded-state subtraction on LUMIERE (`C - independent_state` difference #sfmt(r36.control_comparisons.C.independent_state.difference)) or SAILOR.

= Forecasting unseen future visits (R37, R38, and R39)

== R37: frozen spatial state with recurrent history dynamics

R37 moved from observed pairs to past-only forecasting on 75 history-eligible pairs from 18 unseen LUMIERE patients (262 source training pairs, 60 validation calibration pairs). Freezing the R35 seed-42 spatial encoder, a gap-conditioned recurrent dynamics model predicted a Gaussian distribution over next-visit compartment log-volumes. We scored predictions with the continuous ranked probability score (CRPS) using validation-selected scale multipliers from `{0.5, 1.0, 2.0}` and a 0.05 observation noise floor.

To prevent observation scheduling or current-scan bias from appearing as a history benefit, R37 matched `H` (up to four spatial MRI visits) against five controls: `L` (latest MRI scan only, given the exact same past inter-visit gaps, visit-count mask, and modality-availability flags as `H`), `G` (spatially pooled earlier maps with spatial current scan), `R` (voxelwise ridge dynamics), `P` (calibrated persistence of the latest decoded volume), and `E` (latest decoded volume plus the source-fitted empirical change distribution).

On the primary cohort, `H` reached CRPS #fmt(r37.arm_scores.H.crps) (MAE #fmt(r37.arm_scores.H.mae)) versus #fmt(r37.arm_scores.L.crps) (MAE #fmt(r37.arm_scores.L.mae)) for `L`. The `H - L` CRPS difference was #sfmt(r37.primary_comparisons.at("H-L").difference) (99% Bonferroni interval #fmt(r37.primary_comparisons.at("H-L").ci.at(0)) to #sfmt(r37.primary_comparisons.at("H-L").ci.at(1))). Calibrated persistence `P` (#fmt(r37.arm_scores.P.crps)) and empirical change `E` (#fmt(r37.arm_scores.E.crps)) both achieved lower point CRPS than every learned neural arm (`H`, `L`, `G`).

== R38: 2x2 factorial test of joint encoder and dynamics training

A natural objection to R37 is that freezing a single-visit encoder discards subtle anatomical precursors of growth before the dynamics head sees them. R38 tested that hypothesis across three seeds and 20 epochs (5,240 updates per cell; 62,880 total updates) in a $2 times 2$ factorial design. The first factor varied *gradient coupling*: joint end-to-end training (`J`), where forecast losses backpropagate into the 3D spatial encoder, versus detached observation training (`O`), where forecast inputs are detached so the encoder learns only from cutoff-visit state supervision. The second factor varied *temporal input*: multi-visit MRI history (`H`) versus latest MRI with matched observation-history metadata (`L`).

#figure(
  table(columns: (2.35fr, 0.9fr, 0.9fr, 1.0fr, 0.9fr, 0.9fr),
    [Arm or control], [CRPS], [Forecast MAE], [Current MAE], [Current Dice], [80% Cov],
    [`JH` (joint + history)], [#fmt(r38.cohorts.primary.scores.JH.seed_score_average.crps.mean)], [#fmt(r38.cohorts.primary.scores.JH.seed_score_average.mae.mean)], [#fmt(r38.cohorts.primary.scores.JH.seed_score_average.current_mae.mean)], [#fmt(r38.cohorts.primary.scores.JH.seed_score_average.current_dice.mean)], [#fmt(r38.cohorts.primary.scores.JH.seed_score_average.at("coverage_0.8").mean)],
    [`JL` (joint + latest-only)], [#fmt(r38.cohorts.primary.scores.JL.seed_score_average.crps.mean)], [#fmt(r38.cohorts.primary.scores.JL.seed_score_average.mae.mean)], [#fmt(r38.cohorts.primary.scores.JL.seed_score_average.current_mae.mean)], [#fmt(r38.cohorts.primary.scores.JL.seed_score_average.current_dice.mean)], [#fmt(r38.cohorts.primary.scores.JL.seed_score_average.at("coverage_0.8").mean)],
    [`OH` (detached + history)], [#fmt(r38.cohorts.primary.scores.OH.seed_score_average.crps.mean)], [#fmt(r38.cohorts.primary.scores.OH.seed_score_average.mae.mean)], [#fmt(r38.cohorts.primary.scores.OH.seed_score_average.current_mae.mean)], [#fmt(r38.cohorts.primary.scores.OH.seed_score_average.current_dice.mean)], [#fmt(r38.cohorts.primary.scores.OH.seed_score_average.at("coverage_0.8").mean)],
    [`OL` (detached + latest-only)], [#fmt(r38.cohorts.primary.scores.OL.seed_score_average.crps.mean)], [#fmt(r38.cohorts.primary.scores.OL.seed_score_average.mae.mean)], [#fmt(r38.cohorts.primary.scores.OL.seed_score_average.current_mae.mean)], [#fmt(r38.cohorts.primary.scores.OL.seed_score_average.current_dice.mean)], [#fmt(r38.cohorts.primary.scores.OL.seed_score_average.at("coverage_0.8").mean)],
    [`P_JH` (own-state persistence, `JH`)], [#fmt(r38.cohorts.primary.scores.P_JH.seed_score_average.crps.mean)], [#fmt(r38.cohorts.primary.scores.P_JH.seed_score_average.mae.mean)], [none], [none], [#fmt(r38.cohorts.primary.scores.P_JH.seed_score_average.at("coverage_0.8").mean)],
    [`P_OH` (own-state persistence, `OH` and `OL`)], [#fmt(r38.cohorts.primary.scores.P_OH.seed_score_average.crps.mean)], [#fmt(r38.cohorts.primary.scores.P_OH.seed_score_average.mae.mean)], [none], [none], [#fmt(r38.cohorts.primary.scores.P_OH.seed_score_average.at("coverage_0.8").mean)],
    [`P` (common frozen persistence)], [#fmt(r38.cohorts.primary.scores.P.crps.mean)], [#fmt(r38.cohorts.primary.scores.P.mae.mean)], [none], [none], [#fmt(r38.cohorts.primary.scores.P.at("coverage_0.8").mean)],
    [`SSM` (damped linear filter)], [#fmt(r38.cohorts.primary.scores.SSM.crps.mean)], [#fmt(r38.cohorts.primary.scores.SSM.mae.mean)], [none], [none], [#fmt(r38.cohorts.primary.scores.SSM.at("coverage_0.8").mean)],
    [`E` (empirical change + noise)], [#fmt(r38.cohorts.primary.scores.E.crps.mean)], [#fmt(r38.cohorts.primary.scores.E.mae.mean)], [none], [none], [#fmt(r38.cohorts.primary.scores.E.at("coverage_0.8").mean)],
  ),
  caption: [R38 primary cohort scores across 75 pairs from 18 unseen LUMIERE patients (`log1p(mm3)` scale, averaged across seeds 42, 43, and 44; bound to `r38_full_metrics.json`).],
)

Table 3 and Figure 1 show three findings from the R38 factorial comparison:
1. *No joint-training forecast gain.* `JH` had higher error than `OH` (`JH - OH` CRPS difference #sfmt(r38.primary_contrasts.at("JH-OH").difference), 99.5% family-adjusted interval #fmt(r38.primary_contrasts.at("JH-OH").ci_99_5.at(0)) to #sfmt(r38.primary_contrasts.at("JH-OH").ci_99_5.at(1))), and `JL` had higher error than `OL` (#sfmt(r38.primary_contrasts.at("JL-OL").difference)).
2. *No multi-visit history gain.* `JH - JL` was #sfmt(r38.primary_contrasts.at("JH-JL").difference) and `OH - OL` was #sfmt(r38.primary_contrasts.at("OH-OL").difference). The $2 times 2$ interaction `(JH - JL) - (OH - OL)` was #sfmt(r38.factorial_interaction.difference) (99.5% interval #fmt(r38.factorial_interaction.ci_99_5.at(0)) to #sfmt(r38.factorial_interaction.ci_99_5.at(1))).
3. *Forecast gradients degraded current-visit state estimation.* Because `OH` and `OL` detach the forecast head, their shared encoder learned purely from cutoff-visit supervision (current MAE #fmt(r38.cohorts.primary.scores.OH.seed_score_average.current_mae.mean), current Dice #fmt(r38.cohorts.primary.scores.OH.seed_score_average.current_dice.mean)). Allowing noisy next-visit forecast gradients into the encoder (`JH`, `JL`) worsened current-visit MAE to #fmt(r38.cohorts.primary.scores.JH.seed_score_average.current_mae.mean) for `JH` and #fmt(r38.cohorts.primary.scores.JL.seed_score_average.current_mae.mean) for `JL`, and lowered current Dice to #fmt(r38.cohorts.primary.scores.JH.seed_score_average.current_dice.mean) for `JH` and #fmt(r38.cohorts.primary.scores.JL.seed_score_average.current_dice.mean) for `JL`, failing the noninferiority margins (+0.05 MAE, +0.02 Dice) on necrotic and enhancing compartments.

#let r38-keys = ("JH-OH", "JH-JL", "JL-OL", "OH-OL", "JH-JL-OH+OL", "JH-P", "JH-SSM", "JH-E")
#let dmin = -0.18
#let dmax = 0.32
#let pwidth = 10.5cm
#let px(x) = (x - dmin) / (dmax - dmin) * pwidth
#let r38-row(key) = {
  let item = r38.primary_contrasts.at(key)
  let lo = px(item.ci_99_5.at(0))
  let hi = px(item.ci_99_5.at(1))
  let mid = px(item.difference)
  let col = if item.ci_99_5.at(0) > 0 { red } else { blue }
  box(width: pwidth, height: 0.62cm, {
    place(dx: px(0), dy: 0pt, line(length: 0.62cm, angle: 90deg, stroke: 0.5pt + muted))
    place(dx: lo, dy: 0.31cm, line(length: hi - lo, stroke: 1.5pt + col))
    place(dx: mid - 2.8pt, dy: 0.31cm - 2.8pt, circle(radius: 2.8pt, fill: col))
  })
}

#figure(
  kind: image,
  block(width: 100%, inset: 6pt, {
    text(size: 8.8pt, fill: muted)[First arm better (negative ΔCRPS) ← #h(3.8cm) → Comparator better (positive ΔCRPS)]
    v(4pt)
    grid(columns: (2.8cm, pwidth), row-gutter: 0pt,
      ..r38-keys.map(k => (text(size: 9pt, font: "IBM Plex Mono", k), r38-row(k))).flatten(),
    )
    h(2.8cm)
    box(width: pwidth, {
      place(dx: px(-0.10) - 10pt, [#text(size: 8.5pt)[−0.10]])
      place(dx: px(0) - 3pt, [#text(size: 8.5pt)[0]])
      place(dx: px(0.10) - 10pt, [#text(size: 8.5pt)[+0.10]])
      place(dx: px(0.20) - 10pt, [#text(size: 8.5pt)[+0.20]])
      place(dx: px(0.30) - 10pt, [#text(size: 8.5pt)[+0.30]])
    })
    v(12pt)
  }),
  caption: [R38 primary CRPS contrasts and 99.5% family-adjusted patient-bootstrap intervals across 10,000 draws (bound to `r38_full_metrics.json`). None of the factorial contrasts is negative below zero; `JH - E` lies entirely above zero.],
)

== R39: conditioning on recorded cutoff treatment status in SAILOR

A second objection to action-blind forecasting is that post-operative glioblastoma trajectories depend on therapy between scans. While LUMIERE lacks a structured per-visit therapy table, SAILOR records timepoint treatment status (`CRT` in 101 pairs, `TMZ` in 95, `no` in 16, `unknown` in 30 across 242 consecutive pairs from 27 patients). Using five nested patient-held-out folds on R35 MRI-derived state estimates, R39 compared ridge transitions using latest state (`L`), latest state plus treatment (`LT`), state history (`H`), and history plus treatment (`HT`).

Adding recorded treatment status to state history produced an `HT - H` CRPS difference of #sfmt(r39.comparisons.HT_vs_H.difference) (99% bootstrap interval #fmt(r39.comparisons.HT_vs_H.ci99.at(0)) to #sfmt(r39.comparisons.HT_vs_H.ci99.at(1)); `HT` CRPS #fmt(r39.scores.HT.crps) versus `H` #fmt(r39.scores.H.crps)). Privileged persistence using the *measured* cutoff mask volume reached CRPS #fmt(r39.scores.measured_persistence.crps) (MAE #fmt(r39.scores.measured_persistence.mae)), outperforming every MRI-decoded predictor (`L` #fmt(r39.scores.L.crps), `LT` #fmt(r39.scores.LT.crps), `H` #fmt(r39.scores.H.crps), `HT` #fmt(r39.scores.HT.crps)) by more than 0.39 CRPS units.

= Discussion: why single-scan error and resolution dominate forecasting

The results across R35 through R39 explain why next-visit forecasting fails to clear persistence on these cohorts:

1. *Single-visit estimation error exceeds typical inter-visit change.* In R38, current-visit volume estimation from MRI has a patient-uniform MAE of #fmt(r38.cohorts.primary.scores.OH.seed_score_average.current_mae.mean) `log1p(mm3)` (`OH` and `OL`). In R39, replacing MRI-decoded state persistence (CRPS #fmt(r39.scores.persistence.crps)) with measured cutoff volume persistence (CRPS #fmt(r39.scores.measured_persistence.crps)) cuts forecast error by more than half. When single-visit state error exceeds typical inter-visit biological change, taking differences across historical MRI estimates amplifies estimation noise rather than extracting a growth velocity.
2. *Through-plane resolution sensitivity corrupts longitudinal comparisons.* Across all four R38 arms, nonzero z-score standardization makes predictions invariant to global intensity scaling and shifting (`<0.02%` relative CRPS change). However, synthetic $2 times$ slice-thickness downsampling worsens `JH` CRPS from #fmt(r38.nuisance.JH.conditions.identity.seed_score_average.crps.mean) to #fmt(r38.nuisance.JH.conditions.resolution.seed_score_average.crps.mean) (+12.7% necrotic and +17.4% enhancing relative CRPS worsening) and shifts current-visit predictions by 0.39 to 0.90 `log1p(mm3)`. Longitudinal differences across scans with varying slice thickness alias acquisition blur into apparent anatomical change.
3. *Required controls for longitudinal MRI benchmarks.* Future longitudinal glioma studies should require three controls before claiming a learned dynamics capability: (a) a latest-scan-only predictor that receives identical observation-history timing and availability metadata (`L`), (b) calibrated persistence and empirical-change baselines (`P`, `E`), and (c) explicit reporting of single-scan state retention and resolution-perturbation drift. Until single-visit state estimation is accurate and resolution-invariant across scanners, adding recurrent history heads or end-to-end forecast gradients offers no measurable forecasting advantage.

#bibliography("references.bib", style: "ieee", title: "References")
