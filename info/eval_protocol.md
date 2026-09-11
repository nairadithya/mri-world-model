# Locked evaluation protocol — downstream RANO classification

**Status:** v1, 2026-09-11. Frozen; changes require a version bump and a new
committed fold file. This is the pre-registered harness every representation
or readout change is judged on (Step 1 of the SOTA plan). It exists because
every single-split headline in this project so far (0.45 → 0.33 CV, 0.509 →
0.328 CV) was the lucky end of a wide spread.

## Cohort (from `info/eval_folds.json`)

- **`encoder_train` (65)** — hero-split train patients the champion encoder
  saw during SSL (`patient_splits`, seed 42).
- **`encoder_unseen` (26)** — the 13 val + 13 test patients the encoder never
  trained on (K3-16). **The probe cohort.** Probes are never trained or scored
  on `encoder_train` patients when making a representation claim.
- **`dev` (13)** — the val patients. Encoder checkpoint selection (early
  stopping) saw these, so they are encoder-unseen but not selection-clean.
- **`final` (13)** — the test patients. The only patients untouched by encoder
  training *and* early stopping; reserved for the release number.
- **`folds`** — fixed 5-fold partition of the 26 (seed 2026), materialized in
  the JSON. Never re-drawn.

## Metric and uncertainty

- Primary: **macro-F1** over the 4 clean RANO response classes {PD, SD, PR, CR}
  (operative states and missing excluded). Accuracy + per-class recall reported
  alongside. Macro-F1 because PD is ~64% of labels.
- Predictions are **out-of-fold** (CV) or single-net transfer; the metric is
  computed on pooled rows, then uncertainty is a **patient-cluster bootstrap**
  (resample patients with replacement, 10k; visits within a patient are
  correlated).
- Model comparison is a **paired** bootstrap: both models scored on the same
  resampled patients, CI on the difference. A result counts only when the
  paired 95% CI excludes 0.

## Modes

```bash
# within-unseen CV — readout trained and scored only on the 26 (representation claim)
python scripts/probe_rano.py --cache checkpoints/probe_cache.pt --cv-unseen \
    --feat states_forecast --hidden 256 --compare fused --boot 10000
# transfer — readout trained on the 65, scored on the 26
python scripts/probe_rano.py --cache checkpoints/probe_cache.pt --cv-unseen \
    --feat states_forecast --hidden 256 --train-pool train --boot 10000
```

`--compare <feat>` gives the paired difference against `--feat`.

## Rules

1. Selection uses the locked CV (or `dev`); `final` is report-only, touched
   once per release.
2. No fold re-drawing, no per-config reshuffling.
3. Every cited number carries its aggregation and a paired CI.
4. `assert_disjoint` (in `src/data/eval_protocol.py`) fails if any
   encoder-train patient enters a probe fold.

## First results (probe_cache, frozen champion, 2026-09-11)

Within-unseen CV, states_forecast-mlp vs each config (paired CI):

| config | macro-F1 | vs states_forecast |
|---|---|---|
| states_forecast (mlp) | **0.309** [0.255, 0.358] | — |
| vision | 0.257 | +0.052 [−0.036, +0.142] n.s. |
| states_current | 0.243 | +0.066 [+0.021, +0.133] SIG |
| fused (snapshot) | 0.241 | +0.069 [+0.014, +0.139] SIG |
| clinical-only | 0.213 | +0.097 [+0.041, +0.158] SIG |

Transfer (readout on 65 → 26 unseen): states_forecast **0.408** [0.299, 0.473],
fused 0.234 (paired +0.174 [+0.047, +0.278] SIG).

Readout: the **trajectory state** carries the signal, and the forecast framing
state_t → RANO_{t+1} beats reading the current snapshot — with CIs excluding 0.
Clinical-only is weak, so the label signal is not demographics. These are the
numbers Step 2 (ROI/mask pooling) must move.

## Known limits

- The encoder early-stopped on `dev`, so only `final` is fully clean; a
  pristine holdout for a *newly trained* encoder does not exist yet. When
  Step 3 trains the encoder, it must leave a fresh slice out from the start.
- 22 of the 26 unseen patients contribute usable labelled rows for the
  forecast state; the other 4 have no clean label at a forecastable step.
- Frozen-encoder bootstrap CIs reflect patient sampling, not training-seed
  variance of the readout.
