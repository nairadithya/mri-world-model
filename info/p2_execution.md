# P2 execution plan — lesion-aware forecasting

Status: **active** (2026-09-22). This plan supersedes the RANO-classification
execution order in `directions.md` where the two conflict. Historical RANO
experiments remain evidence about the old task, not gates for this one.

## Why P2 exists

P0 established a reproducible cross-cohort target: `lesion-state-v1`, the
necrotic/nonenhancing, enhancing, and edema/FLAIR compartment volumes in mm3.
Because exact dates are not consistently recoverable, its primary horizon is
the next observed visit. P1 then found that the current JEPA state does not
improve this target: relative patient-uniform log-volume MAE is 1.393 in
locked LUMIERE CV and 2.873 under unchanged SAILOR transfer, versus 1.000 for
persistence. P2 is therefore a repair experiment, not a scale-up of the
current model.

The scientific question is:

> Can an explicitly lesion-aware state and residual dynamics model improve
> future physical anatomy beyond persistence/current-volume baselines, and
> does a lesion-aware JEPA objective add an incremental benefit?

## Frozen scope

- Primary task: next-observed-visit anatomy forecasting.
- Primary target: the three `lesion-state-v1` log-volumes.
- Primary development population: encoder-unseen LUMIERE patients, using the
  existing patient folds and only eligible manifest rows.
- Developmental transfer: fit without SAILOR outcomes, then evaluate all
  eligible SAILOR rows; SAILOR is not a pristine external test.
- Primary measurement source: recovered DeepBraTumIA label maps for LUMIERE
  and ONCO masks for SAILOR. CL masks are a target-sensitivity analysis.
- Primary floor: persistence. Current-volume ridge is the learned scalar
  comparator.
- Timing: visit order is valid; `gap_days` is auxiliary. Do not claim a fixed
  horizon, growth rate, or calibrated time-to-event endpoint from current
  timing metadata.
- RANO: excluded from the primary target, model-selection gate, and cross-site
  claim. It may remain as historical or exploratory LUMIERE-only metadata.

## Stage P2.1 — physical lesion feature layer

Build a versioned per-visit state `u_t` from native masks and affines:

1. The three audited compartment log-volumes.
2. Shape/extent features that are well-defined in physical space: surface,
   compactness, principal extents, centroid, and compartment adjacency.
3. History-only features: previous change and current-to-nadir change.
4. Modality/mask availability and segmentation-source indicators.
5. Resection cavity only after a reproducible cross-cohort definition exists.
6. Treatment phase only when independently documented at prediction time.

Every feature cache must retain source paths, affine/voxel metadata, manifest
row IDs, configuration hash, git SHA, and creation date. Features derived from
resized identity-affine tensors are not physical measurements.

**Gate:** exact row alignment with the P0 manifest; finite features; no target
visit information in a forecast input; reproducible extraction from native
masks.

## Stage P2.2 — deterministic residual forecaster

Re-run persistence, mean delta, and patient-level trend, then add a small
zero-initialized residual model:

`u_hat_(t+1) = u_t + g(H_t) * delta(H_t)`

The zero initialization makes the starting prediction persistence. Inputs may
include structured lesion history, missingness, and documented treatment
history. The current global BRAINIAC/JEPA state must be a separate branch so
its incremental value can be removed and measured cleanly.

Do not condition the primary model on a supposedly exact future gap. A gap
sensitivity model may use the recorded interval, but it cannot support a
fixed-horizon claim under the current timing audit.

**Gate:** improve patient-uniform log-volume MAE over persistence with a paired
patient-cluster 95% interval excluding zero on locked LUMIERE development CV.
Also report pooled MAE, delta-log-volume MAE, signed bias, per-compartment
Spearman, stable/changing strata, and performance relative to persistence.

## Stage P2.3 — lesion-aware representation training

Only after the deterministic feature pipeline is sound, retrain the encoder
with lesion-aware information. The minimum ablation ladder is:

1. Structured lesion state alone.
2. Structured state plus frozen current-image BRAINIAC features.
3. Structured state plus the existing global JEPA state.
4. Structured state plus lesion-pooled image features.
5. Structured state plus a lesion-aware JEPA retrain.

The lesion-aware objective should predict fixed-teacher future lesion features
or compartment-specific representations while retaining the global latent
objective as a separately weighted term. It must not use a SAILOR RANO mapping,
the target visit mask as an input, or target-derived growth features.

**Gate:** an incremental paired improvement beyond the best structured model,
not merely a lower latent cosine loss. If the structured model wins, retain
that negative result and do not force JEPA into the final system.

## Stage P2.4 — transfer and uncertainty

After locking the LUMIERE model, run unchanged SAILOR transfer and ONCO-versus-
CL target sensitivity. Add probabilistic uncertainty only after a deterministic
model beats persistence. Then evaluate proper scores, empirical interval
coverage, and sharpness. Mask forecasting must be compared with an unchanged-
mask baseline.

Fixed-horizon progression risk, treatment-effect classification, and
counterfactual treatment claims are deferred until verified dates, outcome
definitions, censoring metadata, and suitable external supervision exist.

## Exit conditions

P2 succeeds only if a lesion-aware system improves physical lesion forecasts
over persistence/current-volume controls, survives locked patient-level
evaluation, and shows credible new-site behavior. P2 may also close with a
negative result if the structured and lesion-aware models cannot beat the
floor after the prespecified ablations. Either outcome is scientifically valid;
latent loss alone is not an exit criterion.

## Execution status — P2.1/P2.2 (2026-09-22)

`harness.py anatomy features` materialized `lesion-physical-v1`: 48 finite,
history-only features for all 738 eligible pairs (496 LUMIERE, 242 SAILOR).
Features include audited log-volumes/fractions, native-affine centroids,
physical extents, principal spatial spread, modality availability, prior
observed change, nadir-relative change, and history length. Future measurements
are stored separately from `x`; the cache records manifest SHA-256, source
paths through the manifest, git SHA, date, and the history-only cutoff.

`harness.py anatomy residual` then evaluated a patient-uniform,
zero-initialized 48→32→3 residual MLP and grouped-CV structured ridge. The MLP
starts exactly at persistence and uses only source history. Five locked outer
folds select stopping epochs without their test patients; the final model is
fit on all eligible encoder-unseen patients before unchanged SAILOR transfer.

| method | LUMIERE patient-uniform MAE | relative | SAILOR MAE | relative |
|---|---:|---:|---:|---:|
| persistence | 1.545 | 1.000 | 0.943 | 1.000 |
| structured ridge | 1.557 | 1.008 | 1.790 | 1.898 |
| residual MLP | 1.541 | 0.997 | 1.214 | 1.288 |

The residual difference from persistence is unresolved in LUMIERE (95% CI
−0.096 to +0.100) and harmful in SAILOR (+0.176 to +0.369). P2.2 therefore
fails its prespecified improvement gate. P2.3 lesion-aware JEPA training and
P2.4 uncertainty are not authorized by this result; first determine whether a
prespecified structured ablation or additional external supervision can make
the deterministic floor transferable.
