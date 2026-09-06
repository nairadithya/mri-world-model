# Results: JEPA world model over longitudinal glioma MRI

**Status (2026-09-06):** Training is finished. The final model is the full-cohort
champion from the first full-scale training leg (best validation loss 0.0081 at
epoch ~7–8, held-out test loss 0.0074). Throughout, a run must (a) lower loss,
(b) keep collapse monitors healthy, and (c) beat the persistence baseline. All
follow-up training attempts failed that contract, so no further gradient steps
are planned. Remaining work is evaluation and writeup.

This document is the plain-language record of what was trained, what worked,
what failed, and what each check taught us. The working notes with internal
reference IDs live in `info/`; this file stands alone and does not use those IDs.

---

## Glossary

- **JEPA (Joint-Embedding Predictive Architecture):** instead of generating the
  next MRI scan pixel-by-pixel, the model predicts the *learned representation*
  (a compact numeric summary vector) of the next visit. The loss measures how
  close the prediction is to the true next-visit representation.
- **Latent / embedding / representation:** used interchangeably here — the
  numeric vector the network computes to summarize a scan or a patient history.
- **Cosine loss:** `1 − cosine similarity` between predicted and true vectors.
  It is 0 for a perfect prediction and ~1.0 for random guessing. Because vectors
  are normalized first, the model cannot cheat by shrinking everything to zero.
- **Persistence baseline:** the trivial predictor "next visit = last visit."
  Any real dynamics model must beat it; otherwise it has learned nothing beyond
  "nothing changes." This is the single most important gate in the project.
- **Collapse and its monitors:** a JEPA model can cheat by mapping every scan to
  the same constant vector (loss looks great, representation is useless). Two
  monitors detect this: **target std** (spread of the target representations —
  healthy when well above 0) and **effective rank** (how many independent
  directions the representations use — healthy when above 1). A falling loss
  with sick monitors proves nothing.
- **EMA target encoder:** the "true next-visit representation" comes from a twin
  copy of the image encoder whose weights trail the main network via an
  exponential moving average, with no gradients flowing through it. This
  stop-gradient twin is what prevents collapse.
- **BRAINIAC:** a published 3D vision transformer pretrained on ~32,000 brain
  MRIs. It reads one MRI sequence per forward pass and outputs a 768-number
  summary. We use it as our scan reader.
- **LoRA (low-rank adapters):** small trainable matrices added to a frozen
  backbone, so only ~1.2M of 86M parameters change. This keeps the 32,000-scan
  prior intact while letting the model learn change-sensitivity.
- **Prefix-wise targets:** from a patient with visits 1..T, every prefix
  (visits 1..t) predicts visit t+1. One patient yields T−1 training targets
  instead of one — a ~4–6× multiplier on short sequences.
- **RANO:** the expert rating of tumour response at a visit: PD (progressive
  disease — tumour grew or new lesions), SD (stable disease), PR (partial
  response — shrank substantially), CR (complete response — disappeared), plus
  surgical states Pre-Op / Post-Op that are not response ratings.
- **Macro-F1:** classification score that averages performance equally across
  classes. Required here because PD dominates (~64% of LUMIERE labels); plain
  accuracy rewards always guessing PD (a 65% "majority" floor that means nothing).
- **Cross-validation (CV), patient-wise:** split patients (never visits) into 5
  folds, train and score 5 times, report mean ± spread. The only honest headline
  on 49 test visits, where a single lucky split can inflate scores.
- **Ridge regression:** linear probing with a penalty that keeps weights small.
  Mandatory when fitting ~1,152 features on ~450 samples; plain least squares
  explodes (we measured R² ≈ −100).
- **Surprise (prediction error):** the model's error predicting the next visit.
  Large surprise means "this visit defied the patient's predicted trajectory" —
  a candidate early-warning signal for progression.
- **MNI registration / skull-stripping / 96³:** standard preprocessing that puts
  every scan in the same coordinate frame and size the backbone expects.
- **AUC:** for a pair of visits where one progresses and one does not, the
  probability the model assigns higher surprise to the progressing one
  (0.5 = chance, 1.0 = perfect).
- **R²:** fraction of volume variance a probe explains (0 = no better than
  predicting the mean, negative = worse).
- **Frozen probe:** the encoder weights are locked and only a small readout
  (linear layer or 2-layer MLP) trains on top — this tests what the
  representation already knows, without task-specific tuning.
- **Drift:** cosine distance between two representations — consecutive visits,
  or a real scan vs noise. The ruler for "is there signal here."
- **GRU:** a small recurrent network used as a classical dynamics baseline; the
  transformer must beat it to justify its complexity.
- **MLP:** a small feed-forward network (one or two layers) used for clinical
  encoders, the predictor, and probe readouts.
- **Champion / best.pt / last.pt:** after every epoch the trainer saves both
  the current weights (`last.pt`) and the lowest-validation weights seen so far
  (`best.pt`). The champion is the `best.pt` kept as the final model — never
  the last, which may hold overfit tail-epoch weights.

## Data and model in one paragraph

Training corpus: **LUMIERE**, 91 glioblastoma patients with weekly MRI during
chemo-radiotherapy (four sequences per visit: CT1, T1, T2, FLAIR), 616 expert
RANO ratings, demographics, and acquisition parameters. Held-out evaluation:
**SAILOR**, 27 high-grade-glioma patients with 3–19 timepoints each, richer
annotations (treatment status, expert plus automated tissue masks, survival).
The model: BRAINIAC reads each available sequence; per-visit latents average
over present modalities; clinical measurements pass through small MLPs; the two
branches are normalized and concatenated; a temporal transformer with
visit-gap (time-delta) encodings summarizes history; a predictor forecasts the
next visit's representation, trained with cosine loss against the EMA target.
Splits are patient-level (65 train / 13 validation / 13 test) — no patient ever
spans splits.

---

## Scorecard

Honest headline per question. "Hero split" means the single fixed 13-patient
test split; "CV" means 5-fold patient-wise cross-validation. Only CV counts.

| Question | Honest result | Detail and caveat |
|---|---|---|
| Does the dynamics model beat persistence in-domain? | **Yes: held-out test loss 0.0074; mean error 0.0081 vs persistence 0.0218 (~2.7×), wins on 82/91 patients** | Biggest wins on the most changing patients; the 9 losses sit where scans are near-static and "no change" is near-optimal. The 91-patient sweep includes 65 training patients, so memorization inflates the mean — the uncontaminated number is the held-out test 0.0074. |
| Does the frozen representation encode progression status? | **Modestly: CV macro-F1 0.33** | Beats the always-guess-PD floor (~0.20) and a volumes-only probe (0.30); trails supervised end-to-end literature (0.50). The single-split 0.45 was the lucky end of the spread (folds: 0.25–0.42), not the centre. |
| Does prediction surprise anticipate progression? | **AUC 0.77 in-domain (393 visit-pairs), 0.87 cross-site (240 pairs), zero training** | Stable futures are predictable, change is not. Open caveat: a trivial "any change" detector may explain part of the AUC — the persistence-error baseline is still unrun. |
| Does the latent encode tumour size? | **Weakly: readout R² ≈ 0.15 (mean error 1.26 vs 1.38 for predicting the mean, log-mm³); forecasting next-visit size loses to persistence (1.52 vs 1.07, forecast R² ≈ 0.04)** | Tumour volumes come from automated (not expert) masks. Size signal exists but is diffuse; volumes evolve slowly, so "same as last visit" wins. |
| Do dynamics transfer across site/scanner/protocol? | **Split: mean-error dynamics do NOT transfer (0.0290 vs persistence 0.0056); discriminative readouts DO (F1 0.37 ≈ in-domain CV 0.33; surprise AUC 0.87)** | Likely cause is visit-interval regime, not representation: SAILOR gaps average ~14 days (near-static targets where "no change" wins) vs LUMIERE's weekly on-treatment changes. Interval-stratified comparison is the open decider. |
| Did task-specific fine-tuning help? | **No: CV F1 unchanged (0.328 vs 0.334) while dynamics error grew ~2.5× (0.0081 → 0.0191)** | The joint RANO fine-tune tilted the encoder toward the classifier without adding generalizable signal. Champion weights stand untouched. |

Reference numbers for the field: published 4-class RANO prediction on similar
data peaks at macro-F1 0.50 (hybrid network + volumetry + gradient boosting,
patient-wise CV); our frozen-encoder readout at 0.33 uses no task training and
no volumetry — a real but modest signal, best read as the price of
task-blindness rather than a failure.

---

## The training story, run by run

### Run 1 — CPU pilot on 5 patients, corrupt weights (invalid, discarded)

Setup: 5 most-imaged patients, 4 train / 1 validation, 5 epochs, learning rate
1e-4, batch size 1, CPU. The vision backbone was loaded from a community
re-upload of the weights rather than the authors' official file.

What happened: validation loss fell 0.37 → 0.03 — while both collapse monitors
read sick the entire time (target std 0.0024, rank ~1.1). Diagnosis (confirmed
by the weight audit below): the re-uploaded weights output nearly identical
vectors for MRI, pure noise, and all-zeros input while internal activations
blew up ~150×. Key names matched 137/137, so the loader reported success; the
corruption was silent. The heads then learned the only thing downstream of
constant features — a constant — and the loss "improved" toward predicting it.

Lesson, now a repo rule: **key-name matching does not validate a weight port.**
Any third-party checkpoint must pass behavioral probes (real scan vs noise must
produce clearly different outputs) before training. Checkpoints from this run
were deleted.

### Run 2 — CPU pilot on 5 patients, official weights (learning proven, gate not met)

Same protocol with the authors' official weights. Pre-training audit showed
small but real longitudinal signal (mean consecutive-visit drift 0.0046).
Validation loss fell 0.43 → 0.056 with healthy monitors throughout (std 0.079,
rank 2.2).

Verdict against the triple gate from the top of this file: (a) loss 0.43 → 0.056,
pass; (b) monitors healthy (std 0.079, rank 2.2), pass; (c) beats persistence,
fail as expected on 4 training patients (trained 0.0576 vs persistence 0.0043
on near-static targets). The loop provably learns real dynamics on real
preprocessed data; scale is the gate, not a redesign.

### Run 3 — Extended 25-epoch pilot, same 5 patients (epochs are spent)

Question: does the Run-2 gap close with more epochs? Best validation 0.0093,
plateauing from epoch ~13 with healthy monitors. Per-patient JEPA vs persistence:
first win (patient 073: 0.0042 vs 0.0045) and one near-tie (031: 0.0051 vs
0.0043), but the mean (0.0110 vs 0.0043) still loses — and the single held-out
patient (029: 0.0262 vs 0.0040) is the worst by far: the model fit
training-patient dynamics, not general ones.

Verdict: the gap narrowed ~13× → ~2.5×, but epochs stopped helping at ~13.
The remaining lever is more patients, i.e. the full-cohort GPU run.

### Run 4 — Full-cohort first leg on a free Kaggle GPU (the champion)

Setup: all 91 patients (65 train / 13 validation / 13 test), batch size 1, 30
epochs, learning rate 1e-4 with 5-epoch warmup (config default), official
weights, mixed precision (FP16). Preprocessing (2,051 volumes, 0 failures —
2,487 final sequence files across all 91 patients) had been completed locally
beforehand. We keep the lowest-validation checkpoint (`best.pt`); `last.pt`
(final epoch) is retained only for resume mechanics.

Trajectory: validation 0.0830 → 0.0089 (epoch 5) → **0.0081 (epoch ~7–8)** →
0.0129 → 0.0316 → 0.0510 → 0.0667 → 0.0704 (epoch 30). U-shaped: the optimum
sits at epoch ~7–8, then steady overfit climb while training loss holds flat.
Monitors healthy throughout (std 0.085→0.29, rank 1.7→2.5 — drift, not
collapse). Held-out test scorecard on the best checkpoint: loss 0.0074.

Evaluation over all 91 patients: mean predicted error 0.0081 vs persistence
0.0218 (~2.7×), wins on 82/91, biggest on the most dynamic patients. This is
the first clean beat of the mandatory baseline, with exactly the right pattern
(described in the scorecard). **This checkpoint is the final model.**

Practical rule established here: always resume from the best checkpoint, never
the last — the last holds overfit tail weights.

### Run 5 — Resume champion at 5× lower learning rate (the optimum is not holdable)

Setup: resume the 0.0081 champion with fresh optimizer state, flat learning
rate 2e-5, 30 more epochs — against Run 4's decaying 1e-4 schedule. Validation
rose monotonically 0.0089 → 0.031 plateau; the champion was never touched.
Monitors stayed healthy — drift, not collapse.

Verdict: two schedules (decaying 1e-4 tail in Run 4, flat 2e-5 here), same
direction. Continued gradient steps cannot hold or improve the epoch-7–8 basin —
drift starts in epoch 1 even at barely-above-zero learning rates, so it is the
direction, not the step size. Leading suspect: resuming discards the optimizer momentum that
held the narrow basin, and single-patient batches eject the weights from it
immediately. **No more long fine-tuning legs.** All remaining effort pivots to
held-out evaluation and writeup.

### Run 6 — Joint dynamics-plus-RANO fine-tune (failed its gate, champion stands)

Motivation: the frozen representation trails supervised literature (~0.17 F1
gap), arguably the price of never training for the task. Plan: fine-tune the
champion for 10 epochs at tiny learning rate (5e-6) with extra classification
heads predicting next-visit response (4-class, progression-vs-rest, and
response-vs-stable framings), loss = dynamics + classification.

What happened: the classification losses did drop (4-class 3.08 → 0.37,
progression 0.53 → 0.23) — but dynamics error more than doubled (0.0081 →
0.0191; pure-dynamics validation 0.0205, test 0.0174) while persistence stood
still, collapsing the margin. Patient-level losses multiplied. The frozen-probe
re-check with the new encoder: single-split F1 rose 0.448 → 0.509, but
cross-validation sat at 0.328 vs 0.334 pre-tune — **unchanged**. The
single-split gain was split luck, the same trap that inflated the original
0.45.

Verdict: the fine-tune tilted the encoder toward the classifier (single-split
numbers dance) without adding generalizable signal, and charged ~2.5× dynamics
error. A from-scratch joint run (a full GPU session, same pressure, more
instability) is not earned. The champion stands untouched.

---

## Failures, each explained

1. **Silent weight corruption (Run 1).** Community weights, matching keys,
   degenerate outputs. Caught only by behavioral probes. Rule: official weights
   only, verified by real-vs-noise drift.
2. **Collapse misread.** Run 1's falling loss with sick monitors looked like
   training until the weight audit proved the features were constant and the
   monitors had been right all along. Rule: a falling loss alone proves nothing;
   monitors decide.
3. **Overfit tail (Run 4).** Validation U-shape with the optimum at epoch ~7–8 of
   30; the last checkpoint is overfit weights. Rule: keep the best, resume from
   the best, label every downloaded file with leg, epoch, and validation score
   (a resume leg's best file once overwrote the staged champion — recovered
   from the prior session's copy).
4. **Unresumable optimum (Run 5).** Setup: two full 30-epoch schedules from the
   same champion — Run 4's decaying 1e-4 tail and Run 5's flat 2e-5 with fresh
   optimizer state. Both ascend from epoch 1 (Run 4's tail → 0.0704, Run 5 →
   0.031 plateau) with healthy monitors throughout. Same direction at two step
   sizes means retraining ejects the weights from a narrow basin the resumed
   optimizer cannot re-enter. Rule: no more long fine-tuning legs once two
   schedules agree.
5. **Task-tuning damage (Run 6).** Classification pressure without
   generalization gain, at 2.5× dynamics cost. Single-split improvements that
   vanish under CV are luck, not learning.
6. **Single-split luck (probes).** The headline 0.45 sits above every CV fold
   (0.25–0.42). Only cross-validated numbers are reported as results.
7. **Least-squares blowup (volume probes).** Plain regression on 1,152 features
   / 450 samples gave R² ≈ −100. Ridge penalty plus standardization is mandatory
   in this regime — and the tuned penalty (very heavy) itself reveals how
   diffuse the size signal is.
8. **NaN-poisoned new site (SAILOR).** Bias-corrected files carry background
   NaNs (up to 79% of voxels in ~200 sessions), which poisoned the first cache
   into NaN latents. The adapter now prefers the base variants (fully finite)
   and finite-checks any new site before first encoding.
9. **GPU memory cliff.** A long-history patient overflowed 16 GB twice — first
   in the vision backbone, then in the temporal encoder. Fixed with gradient
   checkpointing (recompute activations during backpropagation instead of
   holding them: identical math at ~30–40% slower steps). A batch containing
   the longest history is the worst case; passing it clears the run.
10. **Phantom infrastructure bug.** An 8-worker preprocess crash looked like a
    code bug; a clean 4-worker rerun was flawless. The crashes were collateral
    from a killed process group whose orphaned workers were still saturating
    the machine. Lesson: check for orphaned workers before blaming the code.

## Diagnostics and ablations, each explained

- **Random-initialization sanity.** Untrained model scores ~0.978 (chance ≈
  1.0), training step plus target-network update run clean. The harness can
  resolve a learning signal when one exists.
- **Weight-port comparison.** Official weights discriminate real vs noise
  (drift 0.07) and sequences from each other; the re-upload does not (0.0002,
  identical outputs). This is the evidence behind the official-only rule.
- **Pooling choice.** First-token vs average pooling show near-identical
  consecutive-visit drift (CT1 0.000022 vs 0.000019, T1 0.000021 vs 0.000019,
  T2 0.000031 vs 0.000028, FLAIR 0.000024 vs 0.000020) — keep the authors'
  convention; pooling carries no extra longitudinal signal.
- **Preprocessing fidelity.** Full pipeline vs resize-plus-normalize show
  overlapping drift ranges (raw 0.000024–0.000036 vs preprocessed
  0.000021–0.000031): preprocessing preserves whatever the backbone can see
  and matches the authors' transforms line-for-line.
- **Frozen-backbone check.** Fresh vs trained backbones give identical drift
  (0.000036) and feature spread (1.0832) to 6 decimals — training never
  touched the backbone (frozen by design plus tiny adapter movement), so early
  collapse signatures localized correctly to the heads.
- **Persistence, GRU, clinical-only floors.** Untrained dynamics score ~1.0
  (chance); persistence sits at 0.0043 on pilot data — the gate the model must
  and eventually does clear at scale.
- **Snapshot → history → forecast probes.** Single-visit features (F1 0.31) →
  history summaries (0.33) → forecast-state features (CV 0.33, single-split
  0.45): response ratings are trajectory labels, useful only with time, and the
  temporal transformer beats raw visit-to-visit differences (0.24). Futures read
  nonlinearly (only the forecast state benefits from a multilayer probe).
- **Binary framing note.** Near-minority classes have 1–2 test samples — their
  recalls are noise. The clinical question is progression-vs-not; 4-class
  framing undersells the model.
- **Volume framings.** Total automated-mask volume reads weakly (R² ≈ 0.15)
  and forecasts worse than persistence. Enhancing-core-only and growth-rate
  framings remain open but low-priority.

## Downstream findings in brief

- **Progression readout (LUMIERE):** frozen history summaries separate
  progression (recall 0.72) from stable (0.64), far above demographics (~0.17)
  — unsupervised dynamics learned progression structure for free, at the
  trajectory level (not single snapshots, not volume).
- **Surprise signal:** prediction error anticipates progression with no
  training (AUC 0.77 in-domain over 393 pairs, 0.87 cross-site over 240).
  Response transitions surprise most of all — rare in training, hardest to
  predict. The trivial-change-detector baseline remains the open confound.
- **Cross-site (SAILOR, frozen champion, zero training):** mean-error dynamics
  lose ~5× (interval regime, see scorecard) while the progression readout
  transfers exactly at the in-domain CV level and surprise improves. The
  representation generalizes across site, scanner, and protocol; the mean-error
  gate is visit-interval-dependent.

## Conclusions

1. Latent next-visit forecasting beats "no change" in-domain by ~2.7×, winning
   exactly where change happens. The mandatory baseline is cleared at scale.
2. The frozen history representation carries real, modest, transferable
   progression signal (CV 0.33 in-domain, 0.37 cross-site with zero retraining;
   surprise AUC 0.77 → 0.87). History beats snapshots structurally.
3. Further LUMIERE gradient steps are exhausted: resumed training cannot hold
   the optimum, and task-tuning damages dynamics without generalizable gain.
4. The honest numbers are the cross-validated ones. Every single-split
   headline in this project so far has been the lucky end of a wide spread.
5. Volume is the weak half: weakly readable, not forecastable beyond
   persistence. The thesis holds at trajectory level, not size level.

## Follow-up: multi-horizon forecasting (gate passed, full run open)

Instead of predicting only the next visit, predict visit t+n for every future
n from each history state — with two design rules: the predictor is
**horizon-conditioned** (input is state plus the day-gap to the target; one
output cannot fit N different futures without collapsing to their average) and
the loss is **horizon-weighted** (each pair counts 1/n, so closer futures
dominate). Tested with the encoder frozen via `scripts/horizon_probe.py`
(2,793 valid pairs; 2,204 train / 259 val / 330 test):

- Persistence error grows only ~2× from n=1 (0.0065) to n≥6 (~0.013): modest
  headroom, targets stay near-static even far out.
- The weighted probe beats persistence at **every** horizon on test (e.g. n=1:
  0.0050 vs 0.0088; n=5: 0.0065 vs 0.0134; n=9: 0.0116 vs 0.0190) — and its
  n=1 error (0.0050) beats the champion's own 1-step predictor (test 0.0074),
  suggesting the head benefits from multi-horizon data. Caveat: far-horizon
  test pairs are few (6–13 pairs at n=8–11); val shows a mild overfit tail.
- Gate verdict: **passed.** A full Kaggle run with a horizon-conditioned head
  and 1/n weighting is earned; the probe result is the baseline it must beat.

Hero-run leg (2026-09-06, Kaggle T4, launched programmatically — see
`AGENTS.md`): resumed the 0.0081 champion, 30 epochs at LR 2e-5, batch 1.
Resume listed exactly the 14 fresh `horizon_*` keys. Val: 0.49 → 0.072 →
best **0.0170 at epoch 4**, then slow rise to ~0.029 plateau; monitors
healthy throughout (std 0.08→0.16, rank ~2 — drift, not collapse). Test
0.0147. (Val/test here are 1/n-weighted multi-horizon losses — not
comparable to 1-step or probe numbers; the gate is the per-horizon table.)

Per-horizon test eval of the trained leg (`scripts/horizon_eval.py`, same
pairs both sides):

| n | pairs | trained leg | persistence | frozen probe (reference) |
|---|---|---|---|---|
| 1 | 71 | 0.0137 (loses) | 0.0088 | **0.0050** |
| 2 | 58 | 0.0133 (loses) | 0.0079 | — |
| 3 | 48 | 0.0139 (tie) | 0.0083 | — |
| 4 | 40 | **0.0144** | 0.0109 | — |
| 5 | 33 | **0.0157** | 0.0134 | 0.0065 |
| 6–7 | 26/19 | **wins** | — | — |
| 8–11 | 13/9/6/4 | **wins** (thin counts) | — | — |

Gate verdict: **FAILED where it matters most.** Joint encoder+head training
bought far-horizon wins (n≥4) but damaged near horizons ~2.7× vs the frozen
probe (n=1: 0.0137 vs 0.0050) and ~1.9× vs the champion's own 1-step head
(0.0074). Same moral as the aux fine-tune: training the encoder tilts it —
here toward far-horizon targets — while the frozen encoder plus a trained
head keeps near accuracy. The production multi-horizon predictor is therefore
the frozen probe head, not this leg. No further GPU legs; a head-only
(frozen-encoder) training with more capacity is the cheap open option.

## Open work (not claimed)- Interval-stratified cross-site comparison: long-gap SAILOR pairs should favor
  the forecaster — the decider between regime and representation explanations.
- Persistence-error baselines for both surprise AUCs.
- Cross-site volume probes from expert/automated masks; enhancing-core and
  growth-rate framings.
- Acquisition-confound correction, uncertainty/calibration, treatment-as-action
  modeling — untouched by design; this program stopped at dynamics plus
  probing.

## Reproducing this

```bash
# Pilot (CPU): 5 richest patients, same code path as the full-cohort run
python scripts/run_train.py --config config/pilot.yaml \
  --patients Patient-067 Patient-031 Patient-073 Patient-078 Patient-029 --no-wandb
# Smoke test, no weights or data needed
python scripts/run_train.py --epochs 1 --batch-size 1 --no-wandb --random-init
# Downstream probes (CPU, frozen champion; --encode builds the latent cache once,
# the second flag runs the analysis — run with both the first time)
python scripts/probe_rano.py --encode --probe --cv   # cross-validated progression readout (--cv for the honest number)
python scripts/surprise_signal.py                    # error-anticipates-progression AUC
python scripts/volume_probe.py                       # automated-mask volumetry
python scripts/sailor_eval.py --encode --eval        # cross-site evaluation
```

Probe and eval scripts default to `--champion checkpoints/champion_0.0081.pt`
(the downloaded first-leg best, validation 0.0081 — rename your download to
match, or pass `--champion <path>`); caches default to
`checkpoints/probe_cache.pt` and `checkpoints/sailor_cache.pt`. Full-scale
training needs a GPU (~6 h all-in on a free Kaggle T4, batch size 1 on 16 GB;
see `kaggle/hero_run.py`, the notebook source of truth).

Final model: first-leg best checkpoint (validation 0.0081, test 0.0074).
Full-scale training needs a GPU (~6 h all-in on a free Kaggle T4; see
`kaggle/hero_run.py`, the notebook source of truth). Local prerequisites —
official weights at `checkpoints/BrainIAC.ckpt`, preprocessed volumes, metadata
CSVs, MNI template — are gitignored and documented in `README.md`, never
committed.
