#set text(lang: "en", font: "IBM Plex Serif", size: 11pt, fill: rgb("#1a1a1a"))
#set page(paper: "a4", margin: (top: 2.5cm, bottom: 2.5cm, left: 2.5cm, right: 1cm), numbering: "1")
#set par(justify: true, leading: 0.75em, spacing: 1.2em)
#show heading.where(level: 1): it => block(above: 2em, below: 1em)[#set text(size: 1.6em, weight: 700, fill: rgb("#111111")); #it.body]
#show heading.where(level: 2): it => block(above: 1.5em, below: 0.75em)[#set text(size: 1.2em, weight: 600, fill: rgb("#222222")); #it.body]
#show heading.where(level: 3): it => block(above: 1.2em, below: 0.5em)[#set text(size: 1em, weight: 600, fill: rgb("#444444")); #it.body]
#show link: it => text(fill: rgb("#2563a8"), font: "IBM Plex Sans")[#underline(it)]
#show raw: it => text(font: "IBM Plex Mono", size: 0.9em)[#it]
#show quote.where(block: true): it => pad(left: 1.5em)[#block(stroke: (left: 3pt + rgb("#cccccc")), inset: (left: 1em))[#text(style: "italic", fill: rgb("#555555"))[#it.body]]]
#set heading(numbering: "1.")
#set figure.caption(position: bottom)
#set table(stroke: none, inset: 5pt, align: left)
#show table.cell.where(y: 0): set text(weight: "bold")
#let blue = rgb("#0f4d92")
#let muted = rgb("#666666")
#let r37 = json("r37_full_metrics.json")
#let r36 = json("r36_phase1_metrics.json")

#align(center)[
  #text(size: 21pt, weight: "bold")[R35 to R37: Learning from longitudinal MRI]
  #v(0.6em)
  #text(size: 11pt, fill: muted)[Research explanation · Results recorded through October 26, 2026]
]

The three studies ask progressively different questions. R35 estimates anatomy in scans already observed. R36 tests whether a different training loss makes observed change easier to extract. R37 predicts a later observation from scans available at an earlier visit. Only R37 is a forecasting experiment.

R35 found a useful in-domain state estimation result. R36 found a descriptive improvement in a trained change decoder, but its primary test of the frozen encoder was inconclusive. R37 did not establish that MRI history improved prediction. These findings do not add up to a demonstrated predictive world model. They do identify the controls that future work must take seriously.

= Three questions, not three versions of one test

A *state* is what the model estimates at a visit. Here, the main anatomical targets are volumes derived from automated masks of three lesion compartments. *Observed change* compares two scans that are already available. *Forecasting* uses only information available at an earlier visit to predict a later observation.

#figure(
  table(columns: (1fr, 2fr, 2fr),
    [Study], [Inputs at assessment], [Question],
    [R35], [One scan, or both observed scans], [Can a spatial model estimate state and observed change?],
    [R36], [Both observed scans], [Does balanced change supervision improve access to change in a frozen encoder?],
    [R37], [Current and earlier scans only], [Does MRI history improve prediction of the next observed visit?],
  ),
  caption: [The studies test different capabilities. The later image is an input in R35 and R36, but a target in R37.],
)

A model can succeed at one question and fail at another. In particular, comparing two available scans does not show that a model can predict a scan it has not seen. The distinction also rules out using the future visit gap as a forecasting input when that gap is not known at the cutoff.

= R35: state estimation and observed change

R35 compared an MRI-only spatial state model with a fixed BRAINIAC reference and tested a separate model that received a pair of observed scans. The spatial state model used dense mask supervision and native-volume targets. The fixed reference had scalar-volume supervision. The comparison therefore combines changes in model and supervision. It cannot attribute the result to architecture alone.

The LUMIERE assessment included 149 visits from 26 encoder-unseen development patients and 117 consecutive pairs from 23 patients. The table reports patient-uniform mean absolute error (MAE), averaged across three training seeds. Volume scores use `log1p(volume in mm³)`, not mm³. Lower scores are better.

#figure(
  table(columns: (2.4fr, 2.3fr, 0.8fr),
    [Task], [Method], [MAE],
    [State], [Spatial state model], [1.1145],
    [State], [Fixed encoder, scalar-volume control], [1.5000],
    [Observed change], [Paired model], [1.2696],
    [Observed change], [Difference of separate state estimates], [1.2589],
    [Observed change], [Zero change], [1.5446],
  ),
  caption: [R35 LUMIERE development results. State and change scores answer different questions and should not be compared across tasks. Source: `EXPERIMENTS.md`, R35.],
)

The spatial state model had lower state error than the fixed reference. The paired model also had lower error than zero change. But the paired model did not establish an advantage over subtracting two separate state estimates. The paired-minus-independent difference was +0.0108, with a 95% patient-bootstrap interval of −0.0791 to +0.1066. A dedicated pair model therefore did not demonstrate the additional benefit that its complexity was meant to provide.

R35 also tested changes in acquisition. A synthetic reduction in resolution increased false change relative to an unchanged-pair test. Resolution reduction can remove anatomical detail as well as acquisition information, so the test does not isolate nuisance alone. Intensity scaling and offset produced little change, largely because preprocessing standardized each sequence.

SAILOR was a development-transfer assessment, not untouched external validation. The spatial state model did not establish an advantage over a source-mean predictor. For observed change, the paired head performed worse than differences of separate state estimates. Automated mask differences and other site differences remain mixed together.

= R36: whether weighting change helps

R35 left a practical question. Changed voxels occupy a small fraction of the prepared grid. In the R36 audit, the median changed region occupied less than 0.5% of the grid in each compartment. A loss averaged over the whole grid can give unchanged voxels much more influence. R36 tested whether balancing changed and unchanged regions improves access to observed change in the encoder.

The study started each arm from the corresponding R35 state checkpoint and used the same training pairs and duration. Arm A continued state-only training. Arm B used a whole-grid change loss. Arm C weighted changed and unchanged regions equally. Arm D added nuisance constraints to C. The primary comparison was C against B, using matched local linear readouts on frozen encoder features. The primary target was signed occupancy change: whether a voxel entered or left a mask compartment.

#figure(
  table(columns: (2.5fr, 1fr, 1fr),
    [LUMIERE endpoint], [B], [C],
    [Frozen local change readout], [0.381971], [0.378904],
    [Trained nonlinear change head], [0.483508], [0.270405],
  ),
  caption: [R36 patient-uniform balanced signed-occupancy MAE, averaged across three seeds. Lower is better. The trained-head comparison is descriptive; the frozen readout is the primary test. Source: `EXPERIMENTS.md`, R36.],
)

The primary C-minus-B frozen-readout difference was −0.0031. Its family-adjusted patient-bootstrap interval ran from −0.0080 to +0.0026. The result is inconclusive: the study did not establish a frozen-encoder gain, but it did not prove that the objectives are equivalent.

#figure(
  block(width: 100%, inset: 8pt, {
    let result = r36.comparisons.at("C-B")
    let width = 11cm
    let pos(x) = (x + 0.01) / 0.015 * width
    text(size: 9pt, fill: muted)[C better ← #h(5.7cm) → B better]
    v(6pt)
    box(width: width, height: 0.8cm, {
      place(dx: pos(0), dy: 0pt, line(length: 0.8cm, angle: 90deg, stroke: 0.5pt + muted))
      place(dx: pos(result.family_adjusted_ci.at(0)), dy: 0.4cm,
        line(length: pos(result.family_adjusted_ci.at(1)) - pos(result.family_adjusted_ci.at(0)), stroke: 1.5pt + blue))
      place(dx: pos(result.difference) - 3pt, dy: 0.4cm - 3pt, circle(radius: 3pt, fill: blue))
    })
    v(4pt)
    box(width: width, {
      place(dx: pos(-0.01), text(size: 9pt)[−0.010])
      place(dx: pos(-0.005) - 9pt, text(size: 9pt)[−0.005])
      place(dx: pos(0) - 3pt, text(size: 9pt)[0])
      place(dx: pos(0.005) - 15pt, text(size: 9pt)[+0.005])
    })
    v(14pt)
  }),
  caption: [R36 primary C-minus-B difference in frozen-readout MAE. The dot is the difference and the line is the family-adjusted patient-bootstrap interval. The plot reads `r36_phase1_metrics.json`.],
)

The nonlinear change head improved descriptively under C. That observation concerns the trained encoder and decoder together. It does not show that a fixed-capacity readout can extract more change information from the encoder. C also passed the specified state-volume retention allowance against A. That result applies to the stated direct-volume endpoint, not to every possible state representation.

Arm D reduced some perturbation-related false change, but its direct state error increased relative to C. The interval did not establish whether the increase exceeded the prespecified allowance. The overall nuisance tradeoff remains inconclusive. C and D beat a control that used the later scan alone, but neither conclusively beat differences between independent state estimates.

The target audit also found one enhancing-compartment disagreement that disappeared when the grid was resized. It did not establish that most changes disappear during resizing or that sparse change caused earlier failures. The automated masks remain measurement targets, not expert-verified biological truth.

= R37: prediction from MRI history

R37 asked whether earlier MRI scans improve prediction beyond the latest scan. It froze the original R35 seed-42 spatial encoder and volume head. A small gap-aware recurrent model predicted a distribution for the next observed visit. The model received up to four available visits and past timing information. It received no future image, future mask, future gap, or treatment.

The target was the next *observed* visit, not a scan at a specified future day. Historical gaps came from approximate week bins. The model therefore cannot answer what happens at an exact future date or under a particular treatment.

== Controls that distinguish competing explanations

#figure(
  table(columns: (0.5fr, 3.8fr),
    [Arm], [What it tests],
    [H], [Spatial MRI history, using the recurrent predictor.],
    [L], [Latest MRI only, with the same past gaps, visit masks, and sequence-availability flags as H.],
    [P], [Persistence of the latest volume decoded by the frozen state model.],
    [E], [Latest decoded volume plus a source-fitted distribution of observed volume changes.],
    [R], [Regularized linear dynamics on spatial features.],
    [G], [The H architecture with earlier MRI maps spatially averaged; the current map stays spatial.],
  ),
  caption: [R37 controls. L tests whether earlier MRI adds information beyond the latest image and observation history. G tests whether retaining earlier spatial detail helps.],
)

The distinction between H and L matters because the number and spacing of prior visits can carry information without the earlier MRI images. P and E test whether a learned predictor improves on simple forecasts. R tests whether nonlinear recurrence helps against a regularized linear alternative. G retains the current scan's spatial pathway, so its comparison with H concerns earlier spatial detail rather than richer processing of the current scan.

== What the forecasting results show

The primary assessment used 75 history-eligible pairs from 18 reused LUMIERE development patients. The score was the continuous ranked probability score (CRPS). CRPS rewards predictive distributions that place probability near the observed target without becoming needlessly broad. Lower is better. Scores use `log1p(mm³)` coordinates. The analysis averages examples within each patient, patients equally, and fitted-seed scores without ensembling predictions.

#figure(
  table(columns: (0.7fr, 2.8fr, 1fr, 1fr),
    [Arm], [Predictor], [CRPS], [Mean MAE],
    [H], [Spatial history], [1.3003], [1.6938],
    [L], [Latest MRI], [1.2970], [1.6768],
    [P], [Decoded-state persistence], [1.2691], [1.6573],
    [E], [Empirical change distribution], [1.2226], [1.6971],
    [R], [Linear spatial dynamics], [1.2842], [1.6839],
    [G], [Pooled earlier maps], [1.3133], [1.6925],
  ),
  caption: [R37 primary-cohort point scores. E has the lowest point CRPS; this table does not establish superiority over every other arm. Source: `r37_full_metrics.json`.],
)

The study required at least a 5% CRPS improvement for H over L, along with several other conditions. H was instead 0.26% worse than L on the point estimate. All five adjusted intervals for H versus its controls included zero. The secondary latest-scan comparison did not establish that L beat persistence or E. These results do not prove that MRI history contains no useful information. They show that this fixed encoder, predictor, decoder, and assessment did not establish a benefit.

// The interval plot reads the reviewed aggregate instead of duplicating its numbers.
#let contrasts = ("H-L", "H-P", "H-E", "H-R", "H-G")
#let domain-min = -0.08
#let domain-max = 0.22
#let plot-width = 11cm
#let xpos(x) = (x - domain-min) / (domain-max - domain-min) * plot-width
#let interval-row(key) = {
  let result = r37.primary_comparisons.at(key)
  let lo = xpos(result.ci.at(0))
  let hi = xpos(result.ci.at(1))
  let mid = xpos(result.difference)
  box(width: plot-width, height: 0.68cm, {
    place(dx: xpos(0), dy: 0pt, line(length: 0.68cm, angle: 90deg, stroke: 0.5pt + muted))
    place(dx: lo, dy: 0.34cm, line(length: hi - lo, stroke: 1.5pt + blue))
    place(dx: mid - 3pt, dy: 0.34cm - 3pt, circle(radius: 3pt, fill: blue))
  })
}
#figure(
  block(width: 100%, inset: 8pt, {
    text(size: 9pt, fill: muted)[H better ← #h(5.8cm) → Control better]
    v(5pt)
    grid(columns: (1.1cm, plot-width), row-gutter: 0pt,
      ..contrasts.map(key => (text(size: 10pt, key), interval-row(key))).flatten(),
    )
    h(1.1cm)
    box(width: plot-width, {
      place(dx: xpos(-0.05) - 9pt, [#text(size: 9pt)[−0.05]])
      place(dx: xpos(0) - 3pt, [#text(size: 9pt)[0]])
      place(dx: xpos(0.10) - 9pt, [#text(size: 9pt)[0.10]])
      place(dx: xpos(0.20) - 9pt, [#text(size: 9pt)[0.20]])
    })
    v(14pt)
  }),
  caption: [R37 paired CRPS differences, H minus each control. Dots are differences and lines are Bonferroni-adjusted 99% patient-bootstrap intervals. All five intervals include zero. The plot reads `r37_full_metrics.json`.],
)

The full capability criteria also included calibration, mean-error retention, and sensitivity to acquisition changes. For the enhancing compartment, the nominal 80% prediction interval covered 66.16% of targets, below the required 70% to 90% range. On a separate 18-patient nuisance set, synthetic half-resolution input increased H's CRPS by 19.36%. The corresponding mean absolute error increased by 0.3793 in log-volume coordinates. These are scoped synthetic tests, not measurements of clinical scanner effects.

The earlier E-only observation-noise comparison found that adding a fixed Gaussian noise component changed CRPS only slightly. The mean predictions did not change at the shared selected multiplier. That result concerns the shape of E's predictive distribution, not learned dynamics. The Gaussian E condition was fixed for the full pilot after the earlier ablation was known, so the choice was not untouched, blinded prespecification.

= What remains uncertain

The strongest positive finding is R35's lower in-domain error for automated state volumes. That result does not isolate architecture from supervision, and the transfer assessment did not establish a gain over a simple source-mean predictor. R36's balanced loss improved the trained nonlinear decoder descriptively, while its primary frozen-encoder comparison remained inconclusive. R37 completed its fixed forecasting comparison without establishing an advantage for history or for the latest-scan predictor over the simple controls.

The shared limitations matter. LUMIERE assessment patients and SAILOR subjects have been reused in development. Automated masks are imperfect proxies. Acquisition changes, preprocessing, segmentation differences, approximate visit times, and unknown interventions complicate interpretation. A frozen volume decoder may also miss predictable spatial changes. None of the studies establishes expert-verified biological prediction, fixed-day forecasting, or generalization to an untouched population.

The results support a clear methodological lesson. Any future history model needs a matched latest-scan control, appropriate persistence or other simple controls, and outcomes that matter beyond its own latent error. Those requirements do not decide which model to build next. R37's fixed formulation is closed, and the historical R36 Phase 2 remains unauthorized.

= Sources and reading guide

This explanation summarizes the recorded results through October 26, 2026. The source pages preserve the full cohorts, protocols, confidence intervals, corrections, and review limits.

- `info/EXPERIMENTS.md` contains the R35, R36, and R37 measurements and controls.
- `info/METHODS.md` defines the study protocols, target construction, splits, and decision rules.
- `info/NOTES_AND_INSIGHTS.md` explains the reasoning and limits. `info/TODO.md` records current status and authorization.
- `info/reviews/` contains the R35 and R36 audit and review records. `info/documentation_sources/updates/2026-10-26/` contains the R37 execution and result review.
- `info/r36_phase1_metrics.json` and `info/r37_full_metrics.json` preserve aggregate results. The R37 review is separate from the raw aggregate's original pending-review status.
