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

  | weights | real-vs-noise | CT1-vs-FLAIR | output norm |
  |---|---|---|---|
  | HF port | 0.0002 | 0.0000 | 30.02 (identical for all inputs) |
  | official | 0.0709 | 0.0858 | 27.75 |
  | random-init | 0.8707 | — | — |

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

  | patient | JEPA | persistence | split |
  |---|---|---|---|
  | Patient-067 | 0.0100 | 0.0047 | train |
  | Patient-031 | 0.0051 | 0.0043 | train |
  | Patient-073 | 0.0042 | 0.0045 | train, JEPA wins |
  | Patient-078 | 0.0093 | 0.0041 | train |
  | Patient-029 | 0.0262 | 0.0040 | val (held-out) |
  | mean | 0.0110 | 0.0043 | — |

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
