# Ablations and diagnostics

Append-only. Each entry: setup → numbers → inference. IDs referenced from
`decisions.md` and `pilot.md`.

## A1 — Random-init sanity (2026-09-03)

- Setup: full JEPA model, `BrainiacEncoder(checkpoint=None)`, Patient-001,
  real NIfTIs, CPU eval.
- Numbers: JEPA loss 0.978 (chance ≈ 1.0 for cosine), z shapes (1,3,768)
  = 3 prefix targets from 4 visits, train step + EMA update OK.
- Inference: plumbing correct; loss scale calibrated. Baselines run:
  persistence 0.636 < random JEPA — the gap the model must close (I1: the
  evaluation harness can resolve a learning signal when one exists).

## A2 — HF-port vs official weights (2026-09-04)

- Setup: `BrainiacEncoder` with `eugenehp/brainiac:backbone.safetensors`
  (137/137 keys matched) vs official `BrainIAC.ckpt` from the authors'
  Dropbox; probe = real CT1 vs pure noise vs all-zeros, cosine drift.
- Numbers:

  | weights     | real-vs-noise | CT1-vs-FLAIR | output norm                      |
  |-------------|---------------|--------------|----------------------------------|
  | HF port     | 0.0002        | 0.0000       | 30.02 (identical for all inputs) |
  | official    | 0.0709        | 0.0858       | 27.75                            |
  | random-init | 0.8707        | —            | —                                |

  HF-port block activations explode 150× (std 2.6 → 385); tensors
  finite, shapes matching — silent corruption, invisible to key-count
  checks (I4).
- Inference: key-name matching does not validate a weight port. Real
  behavioral probes (noise/contrast discrimination) are mandatory for any
  third-party checkpoint. → D8.

## A3 — First-token vs mean pooling (2026-09-04)

- Setup: official weights, all 5 pilot patients, per-sequence
  consecutive-visit drift under both poolings.
- Numbers: CT1 0.000022/0.000019, T1 0.000021/0.000019,
  T2 0.000031/0.000028, FLAIR 0.000024/0.000020 (first/mean).
- Inference: pooling carries no additional signal → keep the official
  first-token convention (D7).

## A4 — Raw+minimal vs full preprocessing (2026-09-04)

- Setup: same drift metric on raw NIfTIs (resize + z-score only) vs
  full-contract preprocessed volumes, Patient-067, all sequences.
- Numbers: raw drift 0.000024–0.000036 vs preprocessed 0.000021–0.000031
  per sequence — indistinguishable.
- Inference: our preprocessing preserves whatever signal the backbone
  can see; it neither destroys nor creates longitudinal contrast. The
  pipeline is faithful (matches official transforms line-for-line).

## A5 — Backbone drift: fresh vs trained (2026-09-04, broken-weights era)

- Setup: raw backbone features (no projector), fresh pretrained net vs
  online/EMA backbones after the 5-epoch pilot on corrupt weights.
- Numbers: all three give drift 0.000036, feat-std 1.0832 — identical to
  6 decimals.
- Inference: training did not touch the backbone (frozen + negligible
  LoRA movement). The first pilot's collapse signature (target std
  0.0024, rank 1.1, val loss → 0.03) was constant-target collapse in the
  *heads*, downstream of constant features from corrupt weights — not a
  flaw in the training loop. The monitors (std/rank) detected it
  correctly; the misreading was human (I2).

## A6 — Persistence baseline, official weights (2026-09-04)

- Setup: 5-epoch pilot model (`checkpoints/pilot/best.pt`), all 5 pilot
  patients, per-patient loss averaged. GRU/clinical baselines untrained
  (reference floors only).
- Numbers: JEPA 0.0576, persistence 0.0043, GRU 0.9993, clinical-only
  1.0027. (Persistence 0.0043 ≈ pre-train drift mean 0.0046 — consistent.)
- Inference: **learning signal proven** (JEPA far below chance and below
  untrained dynamics), but **persistence not beaten** — expected after 5
  epochs on 4 patients against near-static targets. Beating persistence
  is the explicit gate for the hero run (proposal §6); it needs scale
  (more patients, more epochs, LoRA fully warmed), not a redesign.

## A7 — Extended 25-epoch pilot (2026-09-04/05)

- Setup: same 5 patients + `config/pilot.yaml` as Run 2, `--epochs 25`,
  official weights, CPU. Best val 0.0093 (plateau ~0.0093–0.0100 from
  epoch 13 on; monitors healthy throughout: std 0.0791, rank 2.2).
  Per-patient JEPA vs persistence on `checkpoints/pilot/best.pt`:
- Numbers:

  | patient     | JEPA   | persistence | split            |
  |-------------|--------|-------------|------------------|
  | Patient-067 | 0.0100 | 0.0047      | train            |
  | Patient-031 | 0.0051 | 0.0043      | train            |
  | Patient-073 | 0.0042 | 0.0045      | train, JEPA wins |
  | Patient-078 | 0.0093 | 0.0041      | train            |
  | Patient-029 | 0.0262 | 0.0040      | val (held-out)   |
  | mean        | 0.0110 | 0.0043      | —                |

- Inference: gap narrowed 13× → ~2.5×, with the first patient-level win
  (073) and a near-tie (031). But the only held-out patient (029) is the
  worst by far — the model fit train dynamics, not general ones. Mean
  persistence (0.0043, identical to A6) still unbeaten. Scale (more
  patients, not more epochs — epochs stopped helping at ~13) is the
  remaining lever; the train/val generalization gap is what the hero run
  must close.

## A8 — Merit verdict, hero leg 1 (2026-09-05)

- Setup: leg-1 `best.pt` (epoch ~8, best val 0.0081), per-patient JEPA
  vs persistence over all 91 patients (train+val+test) in target space.
- Numbers: mean JEPA 0.0081 vs persistence 0.0218 (~2.7×); 82/91
  patient-level wins. Biggest wins on the most dynamic patients (084:
  0.0256/0.1380; 076: 0.0022/0.0680; 002: 0.0102/0.0612; 012:
  0.0102/0.0630). The 9 losses (007/008/011/020/029/035/036/040/058)
  sit where targets are near-static and "no change" is near-optimal.
- Inference: FIRST clean beat of the mandatory baseline (proposal §6) —
  with exactly the right pattern: JEPA wins where change happens, loses
  only where predicting no change is near-optimal. Caveat: 65/91 are
  train patients, so memorization inflates the mean; the uncontaminated
  figure is the test scorecard (0.0074 on 13 held-out, `best.pt`) —
  test-only persistence comparison still open. Even so, the A6/A7 gate
  is cleared: at scale, the architecture beats persistence.
- Next: leg 2 resumes from `best.pt` (epoch-8 weights), not `last.pt`.

## A9 — Frozen RANO probe, champion representation (2026-09-05, local CPU)

- Setup (D23): freeze leg-1 champion (val 0.0081), linear/MLP probes on
  per-visit latents → 4-class response {PD, SD, PR, CR} (operative
  states + missing excluded; n=393 labelled visits: train 298 / val 46
  / test 49). Reuses hero 65/13/13 splits. Script `scripts/probe_rano.py`;
  encode ~6 min CPU total (96³ volumes → short ViT sequences), probe
  seconds. Class-weighted CE, macro-F1 primary (PD 64% of labels).
- Numbers (test, n=49; majority-PD acc 0.6531, macro-F1 ~0.20):
  fused-linear acc 0.43 F1 0.31; fused-mlp 0.37/0.17; vision-linear
  0.35/0.21; vision-mlp 0.43/0.30; clinical-linear 0.20/0.17;
  clinical-mlp 0.39/0.19. Follow-up delta features [fused_t,
  fused_t−fused_{t−1}]: linear 0.47/0.24, mlp 0.51/0.23. Test PR n=1,
  CR n=2 — minority recalls are noise.
- Inference: WEAK-POSITIVE at best. Representation probes (F1 ~0.3)
  beat clinical-only (~0.17) and majority on macro-F1 (~0.20), and
  fused-linear recovers SD at 0.57 — but everything loses to majority
  on accuracy, CIs are huge at n=49, and the delta-feature follow-up
  (change SHOULD be the signal for response classes) did not improve
  F1. D23's "clearly beats" bar is NOT met. No D19 retrain on this
  evidence. Next cheap steps before any retraining: (a) temporal-state
  features (true history summary; needs re-encode caching states),
  (b) cross-validation over splits to shrink CIs, (c) only then
  revisit D19/multi-task RANO loss.

### A9 addendum — temporal-state readout (same night)

- Same task/labels/splits; inputs now temporal states s_t (history ≤ t):
  `states_current` (RANO_t) and `states_forecast` (RANO_{t+1}, no
  visit-t+1 pixels in input — no leakage). Test n=49.
- Numbers: states_current-linear acc 0.48 F1 0.33; states_forecast-linear
  0.57/0.38; **states_forecast-mlp acc 0.6735 (beats majority 0.6531)
  macro-F1 0.4483** — PD recall 0.72 (23/32), SD 0.64 (9/14); PR 1/1,
  CR 0/2 (minority still noise). No leakage: s_t sees visits ≤ t only.
- Inference: UPGRADED to positive. Frozen temporal readout lands in
  SOTA territory (Tikhonov hybrid 0.50, TRACE 0.477) with a 2-layer MLP
  and zero task training of the encoder — the history summary DOES
  encode progression dynamics (thesis 1, temporal level). D24's D19
  gate ("approaching 0.50") is now in play: 0.45 frozen vs 0.50
  end-to-end-hybrid. Remaining gaps to a writeup-grade claim: n=49 CIs
  (cross-val), CR/PR minority (volume/auto-mask probe as second
  result), LUMIERE→SAILOR generalization.

## A10 — Surprise-as-signal: JEPA error anticipates PD (2026-09-05, local CPU)

- Setup (D24.c): frozen champion, per valid (t → t+1) pair err =
  1 − cos(predictor(state_t), target_{t+1}); label = clean RANO of visit
  t+1. Script `scripts/surprise_signal.py`. n=393 pairs, PD-rate 0.639.
- Numbers: mean err PD 0.0074 (n=251) / SD 0.0061 (95) / PR 0.0088 (20) /
  CR 0.0115 (27). **AUC(err → next-visit PD) 0.7677** — zero training.
- Inference: prediction surprise anticipates progression: flat/stable
  futures are predictable, change is not, and response transitions
  (PR/CR, rare in training) surprise most of all. Caveat: partly
  expected (PD = big change = hard to predict); the quantification and
  the PR/CR pattern are the new bits. A persistence-error baseline is
  still open — change-detection alone may explain part of the AUC.

### A9 second addendum — CV corrects the headline (same night)

- 5-fold patient-wise CV of states_forecast-mlp (`--cv`): F1 =
  0.25/0.38/0.30/0.42/0.32 → **mean 0.334±0.060**, acc 0.536±0.083.
  The hero-split 0.45 was the lucky end, not the centre.
- Inference: honest headline is **0.33, not 0.45** — above volumes-only
  (0.30) and majority (~0.20), below SOTA 0.50. D24's D19 gate
  ("approaching 0.50") is NOT met; no fusion retrain. The representation
  carries real but modest progression signal; volume probes (auto-masks
  downloading) and SAILOR generalization are now the load-bearing next
  results.

## A11 — Volume probes from frozen latents (2026-09-06, local CPU)

- Setup (D24.b): labels = DeepBraTumIA `measured_volumes_in_mm3.json`
  (auto-masks, 599 studies; total = necrotic + enhancing + edema).
  Log-mm3 least squares, hero splits. Script `scripts/volume_probe.py`.
  Debugging note: plain LSQ on 1152-d/450-row gave R2 ≈ −100 (p≫n
  overfit) — ridge (λ=10) + standardization is mandatory at this
  sample regime; any future probe above ~100-d features must regularize.
- Numbers: readout log-vol from fused_t — test n=73, MAE 1.34 (mean
  baseline 1.38), R2 0.27. Forecast log-vol_{t+1} from state_t — test
  n=56, MAE 1.53 vs persistence 1.07, R2 0.21.
- Inference: latent holds weak size signal (readout beats mean), but
  next-visit volume is better predicted by current volume than by state
  dynamics — expected, volumes evolve slowly and the dynamics add noise.
  Volume half of the thesis: unblocked and measurable, not yet
  competitive. Enhancing-core-only + growth-rate framings still open.

### A11 addendum — CV-tuned ridge lowers the headline (same night)

- λ picked by 5-fold CV on train rows (grid 0.1–1000): λ=1000 both
  tasks (heavy shrinkage — size signal is weak/diffuse). Readout:
  MAE 1.26 (base 1.38), R2 0.15 (was 0.27 at fixed λ=10 — partly
  train-overfit). Forecast: MAE 1.52 vs persistence 1.07, R2 0.04.
- Honest headline: R2 ≈ 0.15 readout, forecast loses to persistence.
  Volume half stays uncompetitive; enhancing-core/growth-rate framings
  open but low-priority behind SAILOR generalization.

## A12 — SAILOR cross-site eval, frozen champion (2026-09-06, local CPU)

- Setup (D26): 27 subjects / 270 sessions via `src/data/sailor.py`
  (base variants after the NaN recon; intervals-days deltas; empirical
  codebook {1:PD, 2:SD, 3:PR, 5:CR}); 243 valid pairs, 240 RANO-labelled
  (PD-rate 0.31; SAILOR majority is SD 0.48 — different regime phase
  than LUMIERE's PD 0.64). Script `scripts/sailor_eval.py`.
- Numbers: (a) JEPA 0.0290 vs persistence 0.0056 — persistence WINS ~5×.
  (b) LUMIERE-trained forecast-mlp transferred: acc 0.52 (maj 0.48),
  macro-F1 0.37, PD recall 0.63 / SD 0.60. (b2) SAILOR-fit ceiling
  0.85/0.85 (train=test, memorization — ceiling only). (d) surprise-AUC
  0.87 (n=240).
- Inference: SPLIT. Mean-loss dynamics do NOT transfer — but the likely
  cause is regime, not representation: SAILOR intervals run ~14 days,
  targets near-static, "no change" near-optimal (cf. A8's 9 losses,
  same pattern). Discriminative readouts DO transfer: F1 0.37 ≈
  LUMIERE-CV 0.33 with zero SAILOR training, and surprise-AUC 0.87
  beats LUMIERE's 0.77. Representation generalizes across
  site/scanner/protocol; the dynamics gate (A8-style mean win) is
  interval-regime-dependent. Follow-ups: interval-stratified JEPA-vs-
  persistence (long-gap SAILOR pairs should favor JEPA), persistence-
  error baseline for both AUCs, SAILOR volume probes (ONCO masks).

### A8 addendum — same-space correction: the 2.7× mixed spaces (2026-09-06)

- Setup (G13/R3): the champion's 1-step predictor run over cached EMA states
  (`scripts/split_gate.py`, CPU seconds, no backbone forward) vs persistence
  computed on the identical cached EMA endpoints — both sides in the SAME
  space. A8's headline instead compared JEPA error in EMA-target space
  (`model(b)['loss']`) against `PersistenceBaseline(model.projector)` over
  `model.encode_visits` (kaggle/hero_run.py eval cell) — i.e. persistence in
  **online-projector space**. The online projector (trained) spreads
  consecutive visits ~3× wider than the lagging EMA projector, so the 0.0218
  half of the headline never measured the task the model was trained on.
- Numbers (same EMA space; pooled pairs | patient-uniform):

  | split      | pairs | JEPA              | persist | JEPA_pu                 | persist_pu |
  |------------|-------|-------------------|---------|-------------------------|------------|
  | train (65) | 407   | 0.0082 (loses)    | 0.0065  | 0.0083 (loses)          | 0.0071     |
  | val (13)   | 69    | 0.0078 (loses)    | 0.0075  | 0.0081 (wins)           | 0.0083     |
  | test (13)  | 71    | **0.0070** (wins) | 0.0088  | **0.0074** (wins ~2.2×) | 0.0160     |
  | pool (91)  | 547   | 0.0080 (loses)    | 0.0069  | 0.0081 (wins narrow)    | 0.0086     |

- Inference: headline margin REVISED DOWN, gate VERDICT STANDS where it
  counts. The selection-metric-consistent comparison (patient-uniform, the
  val metric best.pt was chosen on) is a narrow 0.0081-vs-0.0086 win, and
  the uncontaminated test wins clearly under both aggregations (pooled
  0.0070/0.0088; patient-uniform 0.0074/0.0160 — a few dynamic test
  patients with high persistence error and few pairs). Pooled-all loses
  because static pair-rich patients dominate pair counts. SAILOR transfer
  failure is UNAFFECTED (interval-stratified eval was same-space both
  sides); horizon probe tables likewise (cache endpoints both sides).
  Open: how much of the online-vs-EMA spread gap is LoRA
  change-amplification (D6 working) vs EMA smoothing — distinguishes what a
  retrain (R13) should preserve.

## A13 — Lead-time: surprise does not precede PD (2026-09-06, local CPU)

- Setup (R1): frozen champion, per-pair JEPA + persistence errors; k-step
  label = incident PD newly appearing at visit t+k (clean window, no earlier
  PD). Script `scripts/leadtime.py`. Harness check first.
- Numbers:

  | k | pairs | PD-rate | JEPA-AUC                                   | pers-AUC   |
  |---|-------|---------|--------------------------------------------|------------|
  | 1 | 393   | 0.639   | **0.7677** (= A10 exactly — harness valid) | 0.7521     |
  | 2 | 124   | 0.484   | 0.6990                                     | **0.8102** |
  | 3 | 60    | 0.400   | 0.7396                                     | 0.7731     |

- Inference: NO JEPA lead advantage. Contemporaneous surprise is
  change-detection (as baselined); at k=2 raw visit-to-visit change predicts
  incident PD clearly better than model surprise (gap ~0.11, SE ~0.06 —
  suggestive, n=124), k=3 inconclusive (n=60). Volatile trajectories precede
  progression, and the model's surprise is the worse volatility meter. RQ1
  as stated ("deviations PRECEDE RANO calls") answers NO for JEPA error:
  surprise is contemporaneous change-detection, nothing more. No further
  surprise-as-early-warning work is justified.

## A14 — Transition error atlas (2026-09-06, local CPU)

- Setup (R2): same per-pair errors, broken by (RANO_t → RANO_{t+1});
  `scripts/leadtime.py`. Designs R13 weights; quantifies G15.
- Numbers (n, JEPA, persistence, median gap-days):

  | trans                                                               | n   | jepa   | persist | gap |
  |---------------------------------------------------------------------|-----|--------|---------|-----|
  | PD>PD                                                               | 131 | 0.0074 | 0.0050  | 84  |
  | X>X (surgery-involved)                                              | 106 | 0.0100 | 0.0111  | 7   |
  | X>PD                                                                | 60  | 0.0078 | 0.0076  | 94  |
  | X>SD                                                                | 55  | 0.0068 | 0.0102  | 98  |
  | SD>PD (onset)                                                       | 43  | 0.0062 | 0.0036  | 91  |
  | PD>X                                                                | 41  | 0.0080 | 0.0087  | 21  |
  | SD>SD                                                               | 37  | 0.0053 | 0.0041  | 91  |
  | CR>CR                                                               | 18  | 0.0138 | 0.0024  | 98  |
  | (rare response cells, n ≤ 12: JEPA 0.007–0.010, elevated as in A10) |     |        |         |     |


- Inference: (a) G15 CONFIRMED — X>X at ~7d gaps is the highest-error cell
  for both sides; resections are intervention discontinuities, and JEPA is
  the only regime that beats persistence there (0.0100 < 0.0111: predicting
  change beats no-change only where change is certain). Model surgery as
  intervention/reset (R7/R13), not as 1-day natural evolution. (b) Onset
  (SD>PD, the clinically critical cell) loses ~2× to persistence — onset is
  small in absolute terms; both predict fine, neither detects. (c) CR>CR has
  the highest JEPA error (0.0138) against near-zero persistence (0.0024):
  the over-predicts-change pathology behind the SAILOR failure, visible
  in-domain. (d) R13 weighting: down-weight X>X (unlearnable discontinuity),
  up-weight SD>PD (critical, persistence-dominated). (e) Gap reframe: cell
  medians cluster 85–100d — LUMIERE pairs are NOT predominantly weekly
  (dense only peri-operatively); SAILOR's 76d median is a similar regime.
  The cross-site difference is phase composition (PD 64% vs SD 48%) + PLHM,
  not sampling density — the regime exoneration stands, mechanism updated.

## A15 — Treatment-conditioned velocity field on SAILOR, frozen champion (2026-09-07, local CPU)

- Setup (option 2 / R14 core): champion encoder frozen; `treatment.txt` wired
  into the adapter as a phase channel (CRT/TMZ/no/unknown; G4 gaps
  index-aligned, provably identical while nothing is dropped);
  `VelocityField`+`PatientTempo` (small: hidden 256, last layer x0.01-init so
  training starts AT persistence) fit on cached all-(t,u) pairs, 1/n-weighted
  Euler integration across true gaps — same math as `_dynamics_loss`.
  5-fold subject CV; variant A = true treatment phase, variant B = constant
  phase (same capacity/init/protocol — the RQ2 ablation). Held-out scoring
  on 1-step pairs, same target space both sides. Script
  `scripts/train_field.py` (encode once ~15 min, CV ~40 min, all CPU).
- Numbers (held-out 1-step err; surprise-AUC PD):

  | fold | held_n | cond              | uncond            | persist                                           |
  |------|--------|-------------------|-------------------|---------------------------------------------------|
  | 0    | 54     | 0.0038            | 0.0039            | 0.0047                                            |
  | 1    | 54     | 0.0022            | 0.0022            | 0.0025                                            |
  | 2    | 38     | 0.0037            | 0.0038            | 0.0042                                            |
  | 3    | 56     | 0.0030            | 0.0030            | 0.0033                                            |
  | 4    | 41     | 0.0054            | 0.0054            | 0.0055                                            |
  | CV   | 243    | **0.0036±0.0011** | **0.0036±0.0011** | 0.0040±0.0011                                     |
  | AUC  | —      | 0.8348            | 0.8397            | 0.8482 (persist highest — change-detection again) |

  Velocity norms 6.5–10.3 (moving, not v≈0-collapsed).
- Inference: (a) TREATMENT ADDS NOTHING — cond ≡ uncond to 4 decimals on
  error and AUC, every fold. The shuffle control is moot (no gain exists to
  be spurious). Reading: given the 1152-d history state (which already
  encodes disease stage), the phase label is redundant — NOT proof treatment
  is useless in principle, but no operational gain at N=27. (b) SITE REFIT
  FLIPS THE TRANSFER GATE: a SAILOR-fit field (either variant) beats
  persistence ~10% on held-out subjects where the LUMIERE head lost 5–14×.
  The cross-site failure localizes finally to dynamics SCALE (fixable on-site
  with N=27 + a small field — R16 realized), not representation, not regime,
  not treatment-blindness.   (c) Caveats: fixed 400-epoch budget; N=27 power; treatment↔stage
  confounding cuts both ways here (it should have made phase MORE
  predictive, yet tie). Robustness: hidden-128 rerun (halved capacity)
  reproduces everything — cond 0.0035±0.0011 / uncond 0.0036±0.0011 /
  persist 0.0040±0.0011, AUCs 0.8431/0.8431/0.8482. The tie is not a
  capacity artifact.

## A16 — Freezing battery: is the frozen encoder the problem? (2026-09-07, CPU)

- Setup: five zero/low-training diagnostics on existing caches, asking
  whether transfer failure needs representation change. Script
  `scripts/freeze_battery.py` (+ per-phase slice of A15's saved
  `checkpoints/field_scores.pt`).
- (1) Site separability: logistic probe frozen-latent → cohort. z-space:
  acc 0.956, AUC 0.9998 (638+270 visits); states: acc 0.77, AUC 0.986.
  SAILOR per-subject logits all strongly positive (3.7–15.7): a UNIFORM
  site shift, not outliers. The "different neighborhood" is measured, not
  gestured at.
- (2) CORAL alignment (fold-wise honest: fit on train folds, apply to
  held-out; z and states spaces separately; frozen LUMIERE head scored):

  |                                    | head   | persist |
  |------------------------------------|--------|---------|
  | unaligned                          | 0.0290 | 0.0039  |
  | CORAL-aligned                      | 0.0119 | 0.0042  |
  | transductive (fit-all upper bound) | 0.0080 | —       |

  Second-order matching alone cuts head error 2.4× with zero training and
  zero unfreezing; persistence is preserved (geometry not distorted).
  ~2/3 of the excess error is covariance shift. Fold 4 flat (0.0284) —
  4/5 improve, reported honestly. Residual (0.012 vs 0.004) is
  higher-order/conditional.
- (5) Image-space damping: consecutive-visit mean|Δ| (96³, brain-masked):
  SAILOR median 0.24 (p10 0.17 / p90 0.41, n=20) vs LUMIERE 0.76 (0.56 /
  0.86, n=10, two earliest visits skipped against peri-op inflation) — NO
  overlap even after the correction (LUMIERE min > SAILOR max), and SAILOR's
  sample includes peri-operative pairs too. Damping is IN THE IMAGES,
  pre-encoder: total-pipeline difference (PLHM prime suspect, uint8 +
  registration contributors not excluded — raw comparison blocked on
  unextracted tarballs).
- (7) Within-phase scoring (A15 pairs sliced by pair-t phase): cond ≡
  uncond inside every phase (CRT 0.0040/0.0041 n=102; TMZ 0.0026/0.0027
  n=95; no/unknown ties; both beat persistence per phase). The A15 tie is
  NOT confounding masking signal — treatment is redundant given history,
  homogeneously. TMZ < CRT errors match clinical volatility ordering.
- (8) Interval-matched gaps: SAILOR latent drift ~HALF LUMIERE's at matched
  gaps every bin (e.g. 30-90d: 0.0040 vs 0.0056; 90-180d: 0.0030 vs 0.0058)
  while the frozen head sits flat ~0.03 on SAILOR in all bins (LUMIERE head
  0.007-0.012). Same gaps, different drift, uniform head failure:
  scale/shift, not interval.
- Inference: freezing EXONERATED three ways — (i) statistics matching
  recovers 2/3 of head error with the encoder untouched; (ii) damping is
  measurable pre-encoder in image space; (iii) site-refit heads already beat
  persistence (A15). What remains is higher-order shift (CORAL residual) +
  damped inputs (PLHM) — neither is fixed by unfreezing per se. Projector-
  only tuning (the A8-correction site) and raw-vs-PLHM deltas are the two
  still-open targeted tests; full LoRA unfreeze is NOT earned.

## A17 — R11 basin-hold, partial verdict: noise isolated, momentum untestable (2026-09-07, Kaggle T4)

- Setup: 4 legs × 5 epochs from the 0.0081 champion, flat LR 2e-5, identical
  schedules; {fresh, loaded-opt} × {accum 1, accum 8}. Kernel
  `nairadithya/r11-basin` errored at the verdict cell, but legs A and C
  completed and their trajectories survive in kernel stdout.
- Numbers (val loss by epoch):

  | leg | setup | ep1 | ep2 | ep3 | ep4 | ep5 |
  |---|---|---|---|---|---|---|
  | A | accum1/fresh | 0.0086 | 0.0092 | 0.0108 | 0.0121 | 0.0139 |
  | C | accum8/fresh | 0.0078 | 0.0078 | 0.0078 | 0.0079 | 0.0080 |
  | B/D | loaded-opt | — | — | — | — | crashed (no opt state in checkpoint) |
- Monitors: leg C std 0.091→0.095, rank 1.7 throughout (healthy drift, no
  collapse); leg A same signature while ascending. Staged champion
  byte-verified post-hoc (975,963,368 B = local file exactly) — the
  by-definition selection picked the right weights.

- Inference: (a) A replicates Run 5's ejection step-for-step (0.0089→0.0167
  by ep5 there) — batch-1 drift is real and reproducible. (b) C HOLDS flat
  5 epochs at/below champion level with the ONLY change being pair-weighted
  accumulation ×8. Batch-1 noise is (at least part of) the ejector — and
  continued training is NOT impossible: with accumulation the basin holds.
  (c) The momentum half (B/D) is untestable from any existing file: all
  ferried champions are opt-stripped by design (D20), leg-1's Adam moments
  died with that session, and moments cannot be reconstructed. D22's
  specific suspect (discarded momentum) stays open but is now secondary —
  the operational question (can training continue?) answers YES via C.
  (d) Caveat: 5 epochs is short; C's durability beyond needs the repair leg
  (C-extended 20 epochs + accum-4 dose cell, proposed).

### A17 addendum — R11-repair verdict: transient hold, dose-response, exploit protocol (2026-09-07, Kaggle T4)

- Setup: v2 kernel (parser fixed + fixture-tested, `python -u`, no assert-fail
  verdict cell). C2 = accum-8/fresh, 20 epochs; E = accum-4/fresh, 10 epochs;
  B/D dropped (no opt state exists anywhere). Monitors healthy throughout
  (std 0.091→0.112, rank 1.7–1.8 — drift, never collapse).
- Numbers: C2 0.0078 (ep1–6 flat) → 0.0080 (ep7) → 0.0103 (ep20),
  best 0.0077. E 0.0079 → 0.0084 (ep5) → 0.0102 (ep10), best 0.0077.
  Dose at ep10: accum-8 0.0084 < accum-4 0.0102 < accum-1 ~0.012+
  (A-ref) — monotonic in accumulation, i.e. in gradient-noise scale.
- Inference: (a) NOISE MECHANISM CONFIRMED — ejection speed scales
  monotonically with batch-1-ness; D22's suspect graduates to cause
  (momentum's share stays untestable). (b) HOLD IS TRANSIENT — even
  accum-8 ejects after ep6; v1's 5-epoch hold did not reproduce longer.
  The basin is genuinely narrow; noise sets ejection speed, not fate.
  Same U-shape as Run 4 on a compressed scale. (c) EXPLOIT PROTOCOL, new:
  both legs found best 0.0077 BELOW champion 0.0081 within 5 epochs — short
  accum-8 legs from the champion can IMPROVE, not just hold. D22's "no
  exploit path" holds for batch-1 only. Rule: accum-8, ≤6 epochs,
  best-tracking (trainer default), then stop. (d) R13 UNBLOCKED under that
  protocol (transition-weighted short legs); long legs and batch-1 remain
  banned. Artifacts: legC2/legE best.pt (val 0.0077) fetched to
  `outputs/r11-basin-v2/` — new best 1-step models, re-gate probes pending.

## Synthesis — what the RANO + cross-site results mean (2026-09-06)

Logged inferences (evidence-backed; see A9/A10/A12 for numbers):

1. History beats snapshot structurally: snapshot F1 0.31 →
   states_current 0.33 → states_forecast 0.38–0.45. RANO is a
   trajectory label; the representation is useful only with time.
   Learned summarization beats raw deltas (0.24): the temporal
   transformer does real work (supports T2).
2. Unsupervised dynamics learned progression structure for free:
   frozen states separate PD (recall 0.72) from SD (0.64), far above
   demographics (F1 ~0.17). T1 holds at trajectory level — not
   snapshot, not volume.
3. The binary framing is where signal lives: PR (n=1)/CR (n=2) are
   noise; the clinical question is progression-vs-not. 4-class framing
   undersells the model.
4. The 0.51 explained: PD-F1 ~0.77 + SD-F1 ~0.62 + one free PR point +
   CR 0, on a favorable repeatedly-selected 13-patient split (above
   every CV fold). CV 0.33 is the contact-with-SAILOR number. Only CV
   counts from here on.
5. Gap to SOTA 0.50 ≈ price of task-blindness (frozen generic latents
   + tiny MLP vs supervised end-to-end + volumetry) — an achievement
   framing, not just a deficit.
6. Nonlinear readout for futures only (MLP>linear on forecast alone):
   futures stored folded; current status reads linearly.
7. Cross-site split (A12): mean-loss dynamics fail on 14-day-interval
   SAILOR (persistence wins 5× — regime, cf. A8's 9 losses), while
   discriminative readouts transfer (F1 0.37 ≈ LUMIERE-CV;
   surprise-AUC 0.87 > LUMIERE 0.77). Representation generalizes;
   the mean-loss gate is interval-regime-dependent.

## A18 — Raw-vs-MNI deltas: the pipeline damps ~4× in image space (2026-09-08, local CPU)

- Setup: every consecutive MNI pair with both sessions linked in
  `raw-mni-link.tsv` (243 pairs = the cross-site eval set); mean|Δ| on
  96³ for the two RAW sessions (`t1wc/t1w/t2w/t2wflair`, canonical
  reorientation, crude foreground mask, z-score within mask) vs the two
  MNI base sessions (`T1c/T1/T2/Flair`, `BrainExtractionMask`, z-score
  within mask). Same pairs, same metric shape — only the pipeline
  differs. Script `scripts/raw_mni_deltas.py`; raw inputs selectively
  extracted from `rawdata.tar.bz2` (1311 files, 7.5 GB). Debugging notes:
  canonical reorientation returns negative-stride views on t2w slabs
  (512×512×28 needs axis flips) which torch rejects — `ascontiguousarray`
  guard; numpy `None in (arrays…)` tuple-membership does elementwise ==
  and raises — use `any(v is None …)`. T2 came back n=0 on the first full
  run from the stride bug, not missing data.
- Numbers (n=939 pair-slots, 33 dropped):

  | slot  | n   | raw med (p10/p90)  | MNI med (p10/p90)  | ratio |
  |-------|-----|--------------------|--------------------|-------|
  | all   | 939 | 0.673 (0.50/0.82)  | 0.173 (0.10/0.50)  | ~3.9× |
  | T1c   | 233 | 0.676 (0.52/0.80)  | 0.153 (0.10/0.33)  | ~4.4× |
  | T1    | 242 | 0.737 (0.59/0.87)  | 0.120 (0.08/0.29)  | ~6.1× |
  | T2    | 225 | 0.637 (0.44/0.77)  | 0.246 (0.15/1.03)  | ~2.6× |
  | Flair | 239 | 0.649 (0.50/0.79)  | 0.195 (0.13/0.44)  | ~3.3× |

- Inference: the MNI pipeline (denoise + intra-patient registration +
  PLHM + affine + uint8) removes ~3/4 of visit-to-visit image change on
  identical pairs — raw SAILOR change sits at LUMIERE scale (0.67 ≈
  0.76), processed at ~0.17. The damping is IN THE PIPELINE, pre-encoder,
  measured not suspected. Scope, honestly held: raw slabs are
  unregistered/unmasked, so position/skull signal that registration
  legitimately removes is inside the 3.9× — total pipeline effect (upper
  bound on over-damping), NOT PLHM-alone (needs intermediates we don't
  have). Two nuances: T2 damps least (2.6×) with a heavy MNI tail (p90
  1.03) — T2 change survives processing best, consistent with T2-Progr.
  rationale codes marking non-enhancing progression; and this metric is
  z-scored, so its MNI median (0.17) is not directly comparable to A16's
  unstandardized 0.24 — the within-script raw-vs-MNI ratio is the
  apples-to-apples number.

## A19 — SAILOR reprocess through the executed LUMIERE contract: gate does not flip, readouts drop (2026-09-08, local CPU)

- Setup: `src/preprocessing/reprocess_sailor.md` executed end to end.
  270 linked sessions staged as symlinks (`data/sailor_staging/`, MNI
  ids, `t1wc→CT1/t1w→T1/t2w→T2/t2wflair→FLAIR`); new
  `config/sailor_reprocess.yaml` (only root/raw_root changed);
  `scripts/preprocess.py --workers 2` → 1065/1066 vols (~2 h).
  `--root` flag added to `scripts/sailor_eval.py` /
  `scripts/sailor_interval_eval.py`; sidecar `.txt` (intervals/RANO/
  treatment/age) ferried from derivatives (metadata, same sessions —
  without them all gaps read 0 and labels vanish); `T1c/Flair` alias
  links per session. Numbers → `info/plots/metrics.json`
  (`sailor_reprocessed`) + `sailor_reprocessed.png`.
- Numbers (frozen champion_0.0081, zero training; 27 subs / 270 ses /
  243 pairs — identical pair/bin counts to derivatives: 88/20/129/6):

  | arm | JEPA | persist | transfer acc/F1 | ceiling | surprise AUC |
  |-----|------|---------|-----------------|---------|--------------|
  | derivatives (A12) | 0.0290 | 0.0056 | 0.52 / 0.37 | 0.85 | 0.87 |
  | reprocessed | 0.0245 | 0.0037 | 0.45 / 0.25 (< maj 0.48) | 0.77 | 0.80 |

  Per-bin JEPA-vs-persist reprocessed: 0.0248/0.0023, 0.0260/0.0025,
  0.0242/0.0022, 0.0231/0.0057 — persistence wins every bin.
- Audit (§6 row 3: staging bug vs genuine shift): mapping vindicated —
  pair/gap/label accounting exact both sides (gaps med 76 identical,
  PD-rate 0.312 = A12's 0.31, T2 matched-session corr 0.58 > crossed
  0.46, T1 center-box repro↔deriv 0.57). No id-shift bug. Two genuine
  shifts found: (1) hd-bet 2.x is unrunnable in this venv (needs
  torchvision, banned by policy; wrapper flags `-mode/-tta` exit 2) so
  the EXECUTED contract used the percentile fallback on full-head
  SAILOR raws — skull retained (finals nzfrac med 0.49 vs LUMIERE
  0.17; LUMIERE inputs were source-stripped so the same fallback was
  near-identity there — venv predates the LUMIERE run, same code
  path). Same-code harmonization, asymmetric effect. (2) one casualty:
  sub-07/ses-03 CT1 rigid-reg hard-failed (Mattes MI no-overlap) and
  T1/FLAIR registered near-empty (nz 0.003/0.03 vs cohort p1 0.32) —
  2 of 243 pairs affected, negligible pooled. Plus a correction: the
  "SAILOR ~14-day regime" was p10, not typical — recorded gaps were
  med 76 / mean 61 all along (metrics.json `sailor_gap_bins`), so
  persistence winning the 61–180d bin (129 pairs) was never regime.
- Inference: same-code reprocessing does NOT restore the gate (JEPA
  0.0245 vs persist 0.0037, ~6.6×) — pipeline mismatch as far as we
  could harmonize it is not the whole story. But transfer readouts
  DROPPED (F1 0.37→0.25, below majority; ceiling 0.85→0.77) while
  label-free accounting held exact and surprise-AUC stayed 0.80 — the
  LUMIERE-calibrated readout does not survive skull-in inputs, even
  though in-domain-decodable signal persists (ceiling 0.77). Residual
  is therefore skull-confound + site (scanner/physiology), NOT proven
  site alone. Decider proposed: re-strip with derivatives
  `BrainExtractionMask` in our reg space (restores brain-only state,
  no model-calibrated stage touched) and re-run; transfer-F1 recovery
  with dynamics still lost would convict site physiology cleanly.

## A20 — Post-preprocess QA gate: what automated checks catch (2026-09-08, local CPU)

- Setup: new `scripts/preprocess_qa.py` (shape / finite / empty /
  nzfrac<0.05 / foreground-bbox<5% FAIL; NCC-to-template WARN-only;
  nzfrac drift vs `--ref` root reported). Run against both cohorts.
- Numbers: SAILOR reprocessed → exit 1 with exactly the 2 known-bad
  volumes (sub-07/ses-03 T1 nz 0.003, FLAIR nz 0.032), 13 partial
  sessions listed, drift 2.80× SKULL-SUSPECT (med 0.49 vs LUMIERE
  0.17). LUMIERE → exit 0.
- Calibration lesson (recorded so the next run doesn't re-learn it):
  NCC-to-raw-template was demoted from FAIL to WARN after it fired on
  healthy volumes — two normal SAILOR T1s (nz 0.42/0.45) and a run of
  in-domain LUMIERE T2s (contrast-driven). Against z-scored finals it
  only separates total collapse (≈0) from everything else (0.03–0.53),
  so it lists but never gates.
- Inference: the audit findings of A19 are now one command to
  reproduce. Any future reprocess (e.g. the mask-controlled decider)
  should clear this gate — exit 0 + drift ≈1× — before evals run.

## A21 — Skull-restored decider: readouts recover, dynamics still lost (2026-09-09, local CPU)

- Setup: the A19 decider via real HD-BET instead of mask transport
  (`scripts/bet_repair.py`): hd-bet 2.x in an isolated venv (torchvision
  rebuilt from the torch CPU index — PyPI's build repeats the D16 `nms`
  failure; 2.x dropped the v1 `-mode/-tta` flags), applied per-session
  (one model load per session via folder mode) to the existing
  `_reg.nii.gz` intermediates, finalized 96³ + nonzero z-score into new
  root `data/sailor_reprocessed_bet/` (old tree untouched). Toolchain
  lessons: canonical reorientation returns negative-stride views on
  flipped slabs (torch rejects — `ascontiguousarray`); never symlink with
  relative targets into a foreign dir (dangling links scanned empty and
  failed all 268 sessions identically — absolute paths only);
  cross-device `os.replace` raises EXDEV (`shutil.move`). QA gate on the
  new tree: drift 1.03× (0.179 vs LUMIERE 0.175), zero warnings; only the
  2 known sub-07/ses-03 registration casualties fail (pre-existing).
  Validation: finished strips nzfrac med 0.175, band 0.14–0.20, zero
  outliers. Evals: frozen champion, zero training, `--root` override;
  fresh `sailor_betfix_cache.pt` (derivatives cache untouched). Sidecar
  bug caught mid-run: subject-level `intervals-days.txt` was never ferried
  (all gaps read 0, all pairs one bin) — copied for all subjects,
  interval eval relaunched; pooled errors were unaffected.
- Numbers (27 subs / 270 ses / 243 pairs, 88/20/129/6 bins):

  | arm | JEPA | persist | transfer acc/F1 | surprise AUC |
  |-----|------|---------|-----------------|--------------|
  | derivatives (A12) | 0.0290 | 0.0056 | 0.52 / 0.37 | 0.87 |
  | repro skull-in (A19) | 0.0245 | 0.0037 | 0.45 / 0.25 | 0.80 |
  | repro skull-out (this) | 0.0339 | 0.0059 | 0.44 / 0.32 | 0.89 |

  Per-bin skull-out JEPA/persist: 0.0353/0.0032, 0.0372/0.0030,
  0.0317/0.0039, 0.0352/0.0066 — persistence wins every bin (~5–12×),
  champion flat ~0.032–0.037 (same over-prediction signature).
  Four-cloud PCA: skull-out rejoins the derivatives neighborhood (not
  LUMIERE) — stripping restored derivatives-like features without
  approaching the training site.
- Inference (A19's pre-registered rule fires): readouts recovered
  0.25 → 0.32 with AUC best-yet 0.89 while dynamics still lose
  everywhere — the residual is site (scanner/physiology + the
  un-harmonized denoise/PLHM/affine/uint8 stages), NOT skull and NOT the
  encoder. Recovery is partial (0.32 vs 0.37), consistent with those
  remaining stages rather than a broken representation. Anomaly flagged,
  not concluded: SAILOR-fit ceiling 0.30 sits BELOW transfer 0.32
  (derivatives: 0.85) — memorization failing on this cache smells like
  the fit path, not the features; needs a look before anyone cites it.
  The transfer story is closed: representation transfers, scale needs a
  site, statistics need alignment, inputs need their skull (and ideally
  their pipeline).

  Addendum (2026-09-09) — uint8 exonerated; preprocessing loop closed.
  The skull-out tree was built from full-precision raw (int16 via dcm2niix,
  float32 throughout our contract) and never touched uint8, yet the
  dynamics failure persists at the same magnitude with the same flat
  signature. Whatever uint8 shaves in the derivatives arm, it is not
  load-bearing — ruled out as necessary, not proven zero-effect. Same
  elimination pattern as skull-on-dynamics (restored, failure identical)
  versus skull-on-readouts (restored, F1 came home). Standing assessment:
  no fundamental preprocessing lever remains for the transfer question.
  Further harmonization (undoing denoise/PLHM/affine) has the wrong sign —
  it degrades their pipeline toward ours, discarding their one better
  step (intra-patient registration), for less than the skull bought
  (which bought zero dynamics). The open preprocessing item is orthogonal:
  visit-consistent registration for LUMIERE (in-domain noise floor,
  proposal stage A, never tried) — sharpens future runs, cannot fix
  transfer. Preprocessing preserved and aligned what was acquired; the
  residual (phase, scanner physics, sampling regime) is a calibration
  problem, and the field refit already priced that calibration as cheap.

## A22 — Predicted-vs-actual latent probes across the dynamics gauntlet (2026-09-09/10, local CPU)

- Setup: `scripts/pred_latent_probe.py`. `--refit` reruns the A15 5-fold
  field protocol (hidden 256, 400 epochs, cond + uncond, same seeds) but
  persists models + held-out 1-step prediction vectors
  (`checkpoints/field_models.pt`, 37 MB — `train_field.py` saved scores
  only, so predictions were unrecoverable without a refit). `--probe`
  builds, per valid 1-step pair with clean RANO_{t+1} (393 LUMIERE / 240
  SAILOR), the ACTUAL EMA latent plus the PREDICTED latent per method
  (champion 1-step head; LUMIERE-trained gap head; SAILOR-fit field —
  held-out fold models on SAILOR, 5-model uncond ensemble on LUMIERE, since
  cond phase ids are SAILOR-treatment semantics). 4-class linear/MLP probes
  (probe_rano protocol) with one deviation: train-stat standardization (see
  debugging note). LUMIERE hero-split + 5-fold patient CV; SAILOR transfer
  (train on LUMIERE-train rows) + SAILOR-fit ceiling + 5-fold subject CV.
  Plot: `info/plots/gauntlet_probe.png` (via make_plots.py from
  metrics.json `gauntlet_probe`).
- Numbers (macro-F1, MLP; CV rows are the honest numbers):
  error context, cosine err on the same labelled pairs — LUM: champ 0.0075
  / gap 0.0039 / field 0.0177 / persist 0.0057; SAI: champ 0.0290 / gap
  0.0236 / field 0.0035–0.0036 / persist 0.0040. (LUMIERE gap head beats
  persistence pooled — never previously reported; SAILOR-fit field on
  LUMIERE fails symmetrically to the champion on SAILOR.)
  LUM CV: true 0.25±0.03 / champ 0.30±0.04 / gap 0.28±0.06 /
  field 0.27±0.04. SAI transfer: true 0.12–0.14 / champ 0.18–0.20 /
  gap 0.25–0.27 / field 0.19–0.20. SAI subject-CV: true 0.33±0.02 /
  champ 0.39±0.08 / gap 0.35±0.05 / field 0.31–0.32±0.06. Ceiling
  (train=test, linear): true 0.99 / champ 0.91 / field 0.92 / gap 0.72
  (MLP all 1.00).
  Debugging note: feature norms differ wildly (EMA ~28, champ ~33, gap
  ~134 — cosine training is scale-blind), and the unstandardized
  fixed-lr probe fit degenerates on large-norm spaces (gap ceiling F1
  read 0.05 before standardizing, 0.72 after). Train-stat standardization
  is mandatory for cross-space probe comparison; A9 is unaffected (it
  compared similarly-scaled LayerNorm'd spaces).
- Inference: (a) predicted ≥ actual everywhere — the predictor is a
  history-conditioned denoiser, and dynamics ADD discriminative value on
  top of the snapshot (history-enrichment, cf. Synthesis-1). (b) Cosine
  error dissociates from signal in BOTH directions: the LUMIERE-applied
  field has the worst error (0.0177, 3× persistence) yet gap-level signal
  (0.27); the SAILOR field has the best error (0.0035) yet less signal
  than the champion (0.32 vs 0.39 — it wins cosine by shrinking toward
  persistence). JEPA error is not a representation-quality metric —
  measured, not asserted. (c) Transfer lives in states, not snapshots:
  snapshot-latent transfer is dead (0.12–0.16) against states at 0.37
  (A12-b); gap-head predictions transfer best of the predicted latents
  (0.27). (d) cond ≈ uncond on signal too (0.31 vs 0.32) — the phase
  channel is invisible in error AND in preserved signal (the dead-channel
  vs redundancy question stays open). (e) The refit replicates A15 to the
  decimal (fold errors 0.0038/0.0039 … 0.0054/0.0054, tie intact) —
  independent replication of the headline tie, now with saved models.
  Caveats: CV spreads ±0.03–0.08; tentative SAILOR codebook; ~10% of
  SAILOR pairs touch empty-T2/T1 inputs counted as present (unaudited —
  content-check + cache rebuild still open). Rankings within noise should
  not be over-read; the predicted≥actual pattern and the two
  dissociations are the robust findings.

## A23 — K3-1 phantom-modality guard + SAILOR cache rebuild (2026-09-10, local CPU)

- Setup (K3-1 closed): `src/data/sailor.py::_central_nzfrac` scores the
  central 42–58% box of each volume and a modality file counts as present
  only if ≥5% of that box is FINITE nonzero (NaN counts as absent, so
  NaN-background `-icor` files no longer pass). It reproduces the K3
  probe's central-slab flags on the derivatives tree exactly (21/21, zero
  mismatches). `_image_path` applies it to every candidate; results
  persist in `<root>/_volume_content.json` (size+mtime+metric-version
  keyed, so repeated loads are free and a metric bump invalidates).
  `scripts/preprocess_qa.py --sailor-root <trees>` exposes the same check
  as a gate (delegates to the new `scan_empty_modalities`). Rebuilt all
  five SAILOR caches (`sailor_cache`, `sailor_z_cache`, `field_cache`,
  `sailor_reprocessed_cache`, `sailor_betfix_cache`) and re-ran the three
  arm tables (`sailor_interval_eval.py`, `sailor_eval.py --eval`,
  `sailor_gap_probe.py`). Logs `logs_rebuild_*`.
- Detection: derivatives = 21 present-but-empty modalities (19 T2 + 2 T1)
  across sub-01/02/20/23/24/27 (K3-1 said 19 T2 + 2 T1); repro skull-in
  and skull-out = only the known sub-07/ses-03 T1+FLAIR registration
  casualties. 27/243 derivatives pairs touched a phantom visit (K3:
  25/243), same 6 subjects. `-icor` audit: 3 empty-base T2s had `-icor`
  variants, but all are ~90% NaN background and are correctly rejected
  (K3-1's "NaN-filled or also empty" confirmed).
- Numbers (same-space pooled from the per-bin EMA table in parentheses;
  (a) mixed-space; transfer = LUMIERE→SAILOR states_forecast MLP):

  | arm | JEPA/persist same-space (ratio) | (a) mixed | transfer F1 | ceiling F1 | surprise AUC |
  |-----|--------------------------------|-----------|-------------|------------|--------------|
  | derivatives | 0.0294/0.0041 (7.11×) | 0.0294/0.0060 | 0.357 | 0.843 | 0.844 |
  | repro skull-in | 0.0244/0.0023 (10.5×) | 0.0244/0.0036 | 0.251 | 0.776 | 0.798 |
  | repro skull-out | 0.0335/0.0037 (8.95×) | 0.0335/0.0060 | 0.234 | 0.767 | 0.883 |

  derivatives per-bin JEPA/persist: 0.0283/0.0052, 0.0362/0.0027,
  0.0290/0.0036, 0.0314/0.0047 (243 pairs, 88/20/129/6). Pre-rebuild
  same-space ratio was 7.32×; (a) mixed ~5× before and after.
- Guard impact: small and arm-specific. Derivatives (the only arm with
  phantoms) moved 0.0290/0.0040→0.0294/0.0041, transfer F1 0.37→0.357,
  surprise 0.87→0.844; both reprocessed arms are essentially unchanged
  (only sub-07/ses-03's two modalities). The absolute magnitudes are far
  inside the CV noise K3 flagged, so no arm verdict flips — the K3-1
  artifact inflated persistence errors only slightly, and the "~5×" head-
  line is a mixed-space number (K3-2) whether or not the guard is applied.
- New finding — the pre-rebuild betfix cache is INCONSISTENT with the tree
  (K3-31 made concrete). Rebuilding from `data/sailor_reprocessed_bet` gives
  transfer 0.4417/0.3177 → 0.2917/0.2343 and ceiling 0.3042/0.3118 →
  0.7875/0.7670: the A21 ceiling<transfer anomaly (K3-18) is RESOLVED, but
  skull-out transfer falls BELOW the 0.479 majority, so A21's "readouts
  recovered 0.25→0.32" does not survive a clean rebuild. The other two arms
  reproduce their A19/A12 readouts to ~0.01 F1, so the code path, champion,
  and LUMIERE probe net are all fine; only the skull-out cache disagrees.
  Ruled out the K3-1 guard as cause: re-inserting sub-07's pre-guard states
  changes nothing (0.2917/0.2343 either way).
  Cause is UNDETERMINED: the old cache was overwritten mid-session and
  carried no provenance (K3-31). Leading clue: `scripts/bet_repair.py` was
  edited 2026-09-09 11:19, AFTER the cache (10:33), and A21 records several
  toolchain fixes to that script (stride handling, absolute symlinks, EXDEV
  move), so the cache may come from an earlier/intermediate betfix tree
  whose replacement preserved `_reg` mtimes. Not proven.
  **Correction (same day):** an earlier draft of this note called the cache
  "stale" because it "predated the final tree (root mtime 15:05)". That is
  withdrawn — the root mtime was 2026-09-08 15:05, i.e. BEFORE the cache,
  so the mtime argument does not hold. The reproducible facts are only:
  A21's skull-out readouts were produced by a real run (logged in
  `betfix_decider`, committed 7a01e9e, not invented), and they are not
  reproducible from the current tree. Treat the rebuilt numbers as
  authoritative; do not cite A21's transfer/ceiling columns.
  The dynamics verdict (JEPA ~9× persistence, same-space) is unchanged.
  Recommendation: add `{root, champion, git sha, date}` provenance to the
  SAILOR caches and assert on load (K3-31) before any further cross-site
  readout is cited.
- Not refreshed here (needs a refit, not just a re-encode): `train_field
  --train` (field_scores.pt) and the A22 `pred_latent_probe --refit`
  field models, both of which now have a rebuilt `field_cache.pt` input.
  The A15/A22 field numbers are therefore pre-rebuild until that rerun.

## A24 — K3-2 same-space arm rows + K3-11 in-domain gate with bootstrap CIs (2026-09-10, local CPU)

- Setup. K3-2: `scripts/sailor_eval.py::_pairs_same_space` now scores the
  (a) row with EMA-target latents on BOTH sides — JEPA = 1 − cos(predictor
  (state_t), EMA_{t+1}), persistence = 1 − cos(EMA_t, EMA_{t+1}); the old
  row paired the EMA JEPA error against `PersistenceBaseline(online
  projector)` over `encode_visits` (mixed space, the original-A8 error).
  New `--pairs` mode runs only that row. K3-11: `scripts/split_gate.py`
  now reports pooled means, patient-uniform means, patient win counts, and
  95% CIs from a 10k patient-level cluster bootstrap (`--boot`, seed 42);
  no new measurement, the instrument K3-11 asked for. Logs
  `logs_ss_pairs_*.log`; numbers → `metrics.json` (`sailor_transfer`,
  `sailor_reprocessed`, new `split_gate`).

- K3-2 numbers (same EMA space, 243 pairs each; mixed-space (a) in
  parentheses):

  | arm | JEPA | persist | ratio | old mixed ratio |
  |-----|------|---------|-------|-----------------|
  | derivatives | 0.0294 | 0.0041 | 7.11× | 4.90× (0.0294/0.0060) |
  | repro skull-in | 0.0244 | 0.0023 | 10.5× | 6.78× (0.0244/0.0036) |
  | repro skull-out | 0.0335 | 0.0037 | 8.95× | 5.58× (0.0335/0.0060) |

  The published "~5×" was the mixed-space online-persistence denominator;
  same-space the gap is 7–10.5×. Direction and verdict unchanged, but the
  D28 rule the repo imposes on itself is now honored in the headline rows,
  and the A16 EMA-persistence row (0.0039) is no longer silently a different
  space from A12's online row.

- K3-11 numbers (champion predictor over `horizon_cache.pt`, 547 pairs /
  91 patients; 95% patient-cluster CIs):

  | split | pooled JEPA | pooled persist | patient-u JEPA | patient-u persist | wins |
  |-------|-------------|----------------|----------------|-------------------|------|
  | train | 0.0082 [0.0069,0.0097] | 0.0065 [0.0057,0.0075] | 0.0083 [0.0071,0.0095] | 0.0071 [0.0059,0.0085] | 26/65 (40%) [28,52%] |
  | val | 0.0078 [0.0064,0.0095] | 0.0075 [0.0052,0.0107] | 0.0081 [0.0066,0.0098] | 0.0083 [0.0059,0.0112] | 7/13 (54%) [23,77%] |
  | test | 0.0070 [0.0059,0.0088] | 0.0088 [0.0056,0.0153] | 0.0074 [0.0050,0.0109] | 0.0160 [0.0075,0.0278] | 8/13 (62%) [38,85%] |
  | overall | 0.0080 [0.0070,0.0091] | 0.0069 [0.0060,0.0080] | 0.0081 [0.0072,0.0091] | 0.0086 [0.0069,0.0107] | 41/91 (45%) [35,55%] |

  Point estimates reproduce K3-11 to 4 decimals (train pooled 0.0082/0.0065,
  test 0.0070/0.0088; overall 41/91). Marginal CIs overlap everywhere, but
  the correct test is the PAIRED patient-resample difference (JEPA −
  persist; negative = JEPA better):
  - train pooled **+0.0017 [+0.0005,+0.0031] SIG** — JEPA is significantly
    WORSE than copying on its own training pairs (the regression-to-mean
    tax, cf. G7); train patient-u +0.0012 [−0.0000,+0.0023] n.s.
  - val: +0.0003 [−0.0015,+0.0017] n.s.; patient-u −0.0002 n.s.
  - test pooled −0.0017 [−0.0072,+0.0012] n.s. — the quoted "gate win" is
    NOT significant on the natural pair-pooled mean; only test
    patient-uniform is **−0.0086 [−0.0182,−0.0009] SIG**, and that is the
    aggregation val selection optimizes (G7), inflated by few-pair dynamic
    patients (persist 0.0160 [0.0075,0.0278]).
  - overall: pooled +0.0010 [−0.0000,+0.0022] n.s.; patient-u −0.0004
    [−0.0023,+0.0011] n.s.; wins 41/91 (45%) [35,55%].
- Inference: the in-domain gate is NOT a general win. The only significant
  cells are train-pooled (where the model is significantly WORSE than
  persistence) and test-patient-uniform (where it wins on the metric it was
  selected on, with a fragile margin). The pair-pooled test mean — the
  natural deployment average — does not separate. So "JEPA beats
  persistence" is unsupported as a general claim: it is a mean estimate on
  one aggregation, not a result, and the "82/91 wins" originally advertised
  is 41/91 (45%). Every gate ratio cited must carry its aggregation and a
  paired CI. Taken with K3-2, the honest program-level statement is: cross-
  site the learned dynamics are 7–10.5× WORSE than copying (same-space,
  unambiguous); in-domain they are indistinguishable from copying pooled
  and only win under the selected patient-uniform test metric.
- Caveat: `horizon_cache.pt` is LUMIERE and was not part of the A23 rebuild;
  these gate numbers are independent of the K3-1 guard.

### A23 addendum (2026-09-10) — downstream metric refresh from the rebuilt caches

- Refreshed the metrics that read the rebuilt SAILOR artifacts, so
  `metrics.json` no longer mixes pre/post-rebuild numbers.
  - CORAL (`freeze_battery.py --coral`, rebuilt `field_cache`): head
    unaligned 0.0294 → aligned 0.0140 vs persist 0.0041/0.0040; transductive
    0.0081. Recovery (0.0294−0.0140)/(0.0294−0.0041) ≈ 61% of the gap, down
    from ~59% at the old 0.0119/0.0039 — the "covariance explains ~2/3"
    reading (K3-4) survives but is now "~three-fifths".
  - PCA clouds recomputed from rebuilt caches (same SVD protocol):
    `site_shift` explained [33.7, 22.0] (was [34.2, 21.4]),
    `site_shift_aligned` [32.9, 24.5] (unchanged — exact replication),
    `three_way_shift` [39.3, 20.4] (was [39.8, 19.8]), `fourth_cloud`
    [40.5, 20.9] (was [40.1, 19.8]). Qualitative separation unchanged.
  - Still pre-rebuild (need a refit, not a re-encode): `gauntlet_probe`
    (A22 field_models) and `sailor_gap_bins.field_cond/field_uncond`
    (A15 field_scores) — flagged in `metrics.json._revision`.

## A25 — Locked evaluation protocol for the SOTA push (2026-09-11, local CPU)

- Step 1 of the SOTA plan: a frozen, pre-registered harness for downstream
  RANO classification, because every single-split headline so far (0.45→0.33,
  0.509→0.328) was the lucky tail and `probe_rano --cv` reshuffled all 91
  patients (K3-16: encoder saw 65 of them).
- Artifacts: `info/eval_protocol.md` (rules), `info/eval_folds.json` (frozen
  patient-wise fold assignment over the 26 encoder-unseen patients, fold seed
  2026, encoder-train list carried for the disjointness assert),
  `src/data/eval_protocol.py` (build/load/assert), `scripts/lock_eval.py`
  (materialize/verify), and `probe_rano.py --cv-unseen` (locked CV + transfer
  + paired patient-cluster bootstrap). Cohort: 26 unseen = 13 val (dev) +
  13 test (final); 22 contribute usable forecast-state rows, 95 labelled visits.
- Numbers (frozen champion `probe_cache.pt`; primary macro-F1; 95% patient-
  cluster bootstrap; `--compare` paired diff vs states_forecast-mlp):

  | config | within-unseen CV | vs states_forecast | transfer 65→26 |
  |--------|------------------|--------------------|----------------|
  | states_forecast-mlp | **0.309** [0.255,0.358] | — | **0.408** [0.299,0.473] |
  | vision | 0.257 | +0.052 [−0.036,+0.142] n.s. | — |
  | states_current | 0.243 | +0.066 [+0.021,+0.133] SIG | — |
  | fused (snapshot) | 0.241 | +0.069 [+0.014,+0.139] SIG | 0.234 |
  | clinical-only | 0.213 | +0.097 [+0.041,+0.158] SIG | — |

- Inference: (a) the locked, leak-free CV (0.309) is very close to the old
  leaky CV (0.33) — the K3-16 contamination was small, as suspected. (b) The
  **forecast state beats every snapshot with a CI excluding 0**, and the
  current-state readout too — the label signal lives in the trajectory, not
  the single-visit latent; clinical-only is weak, so it is not demographics.
  Transfer (readout on 65) reaches **0.408** on the 26 unseen — the honest
  headline for the frozen representation, above the old 0.33 CV. (c) These
  are the numbers Step 2 (ROI/mask pooling) must move; success = paired CI
  excluding 0 on this locked metric.
- Caveats: `dev` (val) was encoder-early-stopping-seen, so only `final` is
  fully clean; a pristine holdout for a *newly trained* encoder must be carved
  before Step 3. CIs reflect patient sampling, not readout training-seed noise.

## A26 — Representation-interface probes: ROI pooling, per-modality, volumetry all fail to move the locked metric (2026-09-11, local CPU)

- Step 2 of the SOTA plan, judged on the A25 locked protocol. New
  `scripts/encode_interface.py` reruns the frozen champion once, keeping the
  representations `probe_cache` discards: per-modality first-token latents
  (R9), DeepBraTumIA-atlas ROI-pooled patch latents (R5), and frozen-temporal
  states recomputed from ROI vision; plus DeepBraTumIA region volumes. 91
  patients / 599 ROI visits, 9 min CPU; provenance recorded in the cache
  (champion, config sha1, git sha `a0ee3db`, date) per the new convention.
- ROI alignment (the approximation risk) validated before use: the atlas mask
  is MNI152 1mm resized nearest to 96³ (registration transform not saved,
  K3-5). FLAIR z-score is +0.10…+0.38 inside ROI vs ≈0.00 in the rest of the
  brain, on 7/7 sampled studies, ROI = 0.4–6% of brain — consistent with
  edema, i.e. the mask lands on the tumour.
- Numbers (within-unseen CV and transfer, states_forecast-mlp baseline,
  macro-F1, 95% patient-cluster bootstrap, paired diff):

  | feature | within-unseen CV | vs state | transfer 65→26 | vs state |
  |---|---|---|---|---|
  | states_forecast (mean vision) | **0.309** | — | **0.408** | — |
  | states_roi (ROI vision) | 0.283 | +0.027 [−0.021,+0.078] n.s. | 0.357 | +0.052 [−0.024,+0.143] n.s. |
  | roi (snapshot) | 0.277 | — | — | — |
  | vision (snapshot, ref) | 0.267 | — | — | — |
  | roi_concat | 0.261 | — | — | — |
  | mod_concat (per-modality) | 0.239 | — | — | — |
  | volumes (DeepBraTumIA) | 0.236 | +0.073 [−0.002,+0.147] n.s. | 0.248 | +0.160 [+0.038,+0.266] SIG |
  | volumes_roi | 0.252 | +0.058 [−0.014,+0.130] n.s. | — | — |
  | volumes_clinical | 0.235 | +0.075 [+0.020,+0.127] SIG | — | — |

  Snapshot paired tests (vs `vision` 0.267): `roi` −0.010, `roi_concat` +0.007,
  `mod_concat` +0.028 — all n.s.
- Inference: **no interface lever moves the locked metric.** (a) ROI/mask
  pooling does not beat the whole-brain first-token mean, on the snapshot or
  the state, so the "global mean dilutes the focal lesion" hypothesis is not
  the binding constraint at this readout (the frozen state already encodes
  enough). (b) Per-modality concat (D9's modality mean being lossy) does not
  help either. (c) The decisive one: **explicit DeepBraTumIA volumetry — the
  feature SOTA leans on — is significantly WORSE than the learned trajectory
  state** (0.236/0.248 vs 0.309/0.408). The representation already carries more
  progression signal than the auto-mask volumes. Redirect effort to the
  objective/representation training (Step 3), not the interface.
- Caveats: ROI alignment is approximate (validated but not exact);
  22 patients / 95 rows give wide CIs, so this is "no detectable improvement",
  not "proven none"; the fixed 500-step MLP readout may underfit the 3072-d
  concat features; `states_roi` zero-fills the 17 visits without a mask; a
  ROI+global concat and precise mask registration remain untested refinements.

## A27 — Supervised task-training of the temporal stack fails; the frozen-readout headline is seed-sensitive (2026-09-11, local CPU)

- Step 3 of the SOTA plan. `scripts/task_train.py` trains on the cached fused
  tokens (no image forwards): vision backbone frozen, only the temporal
  transformer + a 4-class RANO head, class-weighted CE, on the 65 encoder-train
  patients, early stopping on the 13 `dev`, report-only on the 13 `final`.
  `--mode head` is the frozen-state reference under the same harness.
- Numbers (macro-F1):

  | config | dev 13 | final 13 |
  |---|---|---|
  | frozen-state head (frozen temporal) | 0.354 | 0.343 |
  | temporal + head task-trained | 0.345 | 0.311 |

  Task-training's best dev was epoch 1 (0.3647) then it degraded — immediate
  overfit, the D22/Run-6 signature again. The train loss fell ~3.5 → 0.8 while
  final dropped below the frozen head: a **fourth** instance of "gradient on
  the representation tilts it; the frozen readout wins" (Run 6, horizon leg,
  A22, now this).
- Readout-sensitivity audit (`probe_rano --readout-seed`, `--hidden`, final
  13): the same frozen `states_forecast`, same train pool, same features:

  | readout | final-13 macro-F1 |
  |---|---|
  | seed 0 / 1 / 7 / 42 / 123 (hidden 256) | 0.392 / 0.435 / 0.402 / **0.448** / 0.382 |
  | hidden 0 / 64 / 512 (seed 42) | 0.378 / 0.335 / 0.386 |

  A 0.11 range from readout optimizer noise alone — **the A25/A26 headline
  final 0.448 (and 0.408 transfer) is the lucky top of that range.** The honest
  frozen-representation final is ≈0.38–0.40; the readout seed and width must be
  pre-registered or ensembled (K3-14's winner's-curse, now measured).
- Inference: (a) supervised task-training of the temporal does not beat the
  frozen readout. (b) The frozen representation is ≈0.40 final, not ≈0.45, so
  the SOTA gap (0.50) is wider than the 0.448 point estimate implied. (c) The
  remaining levers are a genuinely better representation — vision-encoder
  task training (all three prior attempts damaged it) or external longitudinal
  data — not head/temporal tuning. Recommend pre-registering an ensembled
  readout in the protocol before any further claim.
- Caveats: one split, 10–12 patients/side, single seed set; "no improvement"
  means not detectable, and the task-train harness uses a weaker optimizer than
  `fit_linear`, so the temporal comparison is within-harness, not absolute.

## A28 — Supervised LoRA finetune of the vision tower: no locked-protocol gain (2026-09-11, Kaggle T4)

- Step 3's SOTA recipe: `scripts/finetune_lora.py` + `kaggle/kernel-lora`.
  Trainable = LoRA (1.18M) + projector + fusion + RANO head, temporal frozen;
  class-weighted CE on state_t -> RANO_{t+1}; JEPA distillation to the frozen
  champion (teacher fixed); augmentation; early stop on the 13 `dev`;
  locked-protocol eval with a retrained readout on the 26 unseen / final 13.
- **v1 (inconclusive):** head and LoRA shared lr 2e-5. The random head never
  learned (dev macro-F1 flat 0.1438 for 6 epochs), early stop fired at epoch 1,
  and the encoder barely moved (within-unseen 0.301 / final 0.420). Diagnosed
  as a mis-specified optimization, not evidence.
- **v2 (properly trained):** split LR — head 1e-2 (the `fit_linear` rate),
  representation 2e-4 — 14 epochs, patience 10, min-epochs 3. Dev learned but
  was very unstable (0.148 → 0.263 → 0.362(ep4) → 0.121 → … → 0.326), i.e.
  overfitting 65 patients; best-dev epoch 4. Locked-protocol:

  | model | within-unseen CV (26) | transfer -> final 13 |
  |---|---|---|
  | frozen champion | **0.3093** [0.2546,0.3575] | **0.4483** [0.2999,0.4987] (seed 42) |
  | LoRA v1 (untrained head) | 0.3014 [0.2431,0.3537] | 0.4199 [0.2750,0.4882] |
  | LoRA v2 (trained) | 0.2893 [0.2331,0.3457] | 0.4000 [0.2515,0.4661] |

- Inference: **supervised finetuning of the vision tower does not improve the
  locked representation** — within-unseen slightly *worse* (0.289 vs 0.309),
  final at the frozen readout's own seed-noise floor (0.400 vs 0.448 seed-42 /
  ~0.39 honest mean). This is the genuinely-tested version of the SOTA recipe,
  and it joins Run 6, the horizon leg, the temporal task-train (A27) and the
  interface probes (A26) as the fifth "training on 65 patients does not help"
  result. The JEPA regularizer held (jepa ≈ 0.015 throughout, no collapse), so
  the failure is overfitting the supervised signal, not representation
  destruction. Standing conclusion: the frozen BRAINIAC+LoRA trajectory
  representation is at its **data-limited ceiling** (~0.40 final); only adding
  information (external longitudinal data, Step 5) remains, not more training.
- Caveats: 65 train / 13 dev / 13 final, one seed, dev selection on an
  oscillating metric, and the readout seed alone spans 0.335–0.448 (A27), so
  the v2 deficit is within noise — the claim is "no detectable gain", and the
  honest direction is flat-to-slightly-negative. v1/v2 each cost ~0.5 T4-h.

## A29 — Cross-site adaptability: JEPA vs a from-scratch 3D CNN (2026-09-12, local CPU + Kaggle T4)

- Purpose: the frontier gap (LUMIERE→SAILOR generalization) as a *comparison*
  of task-specific vs task-agnostic pretraining. Supervised comparator =
  `scripts/train_supervised_cnn.py` (MONAI 3D ResNet-18, 8-channel visit-pair
  input, class-weighted CE, augmentation, from scratch) trained on the locked
  65/13 splits; features (512-d avgpool) → `scripts/cross_site_adapt.py`
  zero-shot + K-shot subject-wise CV on SAILOR, against the JEPA states.
- In-domain (locked protocol): CNN best dev 0.294 (ep13), **reserved final
  macro-F1 0.222** — well below the frozen JEPA final (0.448 seed-42 / ~0.39
  honest). The CNN train loss barely moved (0.42→0.36 over 20 epochs ≈ 1,300
  optimizer steps): it is **under-trained**, not converged.
- Cross-site (SAILOR subject-wise):

  | encoder | zero-shot | K=3 | K=5 | K=10 | K=15 | K=20 |
  |---|---|---|---|---|---|---|
  | JEPA | **0.355** | 0.269 | 0.283 | 0.326 | 0.333 | 0.313 |
  | CNN | 0.261 | 0.228 | 0.225 | 0.261 | 0.264 | 0.227 |

  JEPA is ahead at every point (zero-shot +0.09, K-shot +0.04…+0.08).
- **Confound (do not over-read):** the CNN is not a competitive supervised
  baseline — it loses in-domain to the frozen JEPA readout (0.22 vs 0.39). The
  comparison therefore shows only that "a from-scratch 3D CNN on 65 patients
  is worse in-domain and transfers worse than the frozen JEPA representation",
  which is expected and not the SOTA-family test. To make the adaptability
  claim, the supervised comparator must first be made competitive (2D
  axial-slice ResNet/DenseNet — the Matoso setup, ~20k slice samples/epoch —
  or many more epochs/LR schedule), then re-run the same harness.
- Harness + caches committed; CNN weights/features in `checkpoints/`
  (gitignored). Cost ~0.5 T4-h.

## A30 — 2D-slice supervised CNN also collapses; the SOTA-family comparison needs ROI crops (2026-09-12, Kaggle T4 + local CPU)

- Fix attempt for A29's under-trained 3D comparator: `train_supervised_cnn2d.py`,
  a MONAI 2D axial-slice ResNet-18 (~20k slice samples/epoch vs ~200 pairs),
  trained 9 epochs (early stop) on the locked 65, augmentation, class weights.
- In-domain: **dev macro-F1 pinned at 0.1790 for all 9 epochs** (the
  always-predict-PD collapse; 0.179 ≈ F1(PD)/4) while train loss fell
  1.79→1.42 — memorizing train, majority-predicting dev. Reserved final
  **0.182** (vs 3D CNN 0.222, JEPA ~0.39–0.45).
- Cross-site (SAILOR subject-wise):

  | encoder | zero-shot | K=3 | K=5 | K=10 | K=15 | K=20 |
  |---|---|---|---|---|---|---|
  | JEPA | **0.355** | 0.269 | 0.283 | 0.326 | 0.333 | 0.313 |
  | CNN-2D | 0.275 | 0.204 | 0.224 | 0.252 | 0.246 | 0.253 |
  | (CNN-3D, A29) | 0.261 | 0.228 | 0.225 | 0.261 | 0.264 | 0.227 |

  JEPA leads at every point; the two from-scratch CNNs are roughly tied with
  each other and both well below JEPA.
- Diagnosis: a from-scratch supervised CNN is not trainable to competitiveness
  on 91 patients. Two compounding causes: (i) no pretrained weights
  (torchvision is banned by D16, MONAI ships none — the field's ResNets are
  ImageNet/medical-initialised); (ii) whole-brain slices assign the visit's
  single RANO label to *every* slice, most of which contain no tumor — heavy
  label noise that drives the majority collapse. Tikhonov's ResNet is
  **ROI-cropped** and only reaches AUC 0.74 alone; its 0.50 hybrid is carried
  by >4,800 radiomic/growth features, not the CNN.
- Conclusion: **this is not yet the intended SOTA-family adaptability
  comparison.** What is established: on this cohort the frozen BRAINIAC-JEPA
  trajectory representation beats from-scratch supervised CNNs (3D and 2D) both
  in-domain and cross-site, and the supervised comparator must be ROI-cropped
  (using the DeepBraTumIA-atlas / SAILOR ONCO masks already on disk) and/or
  pretrained before any "SSL adapts better than supervised" claim is citable.
- Harness (`cross_site_adapt.py`) and both CNN variants committed; Kaggle legs
  ~0.5 T4-h total.
