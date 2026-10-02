# R33 — Clean-data rescue reruns

Prespecified 2026-10-02 after the user reopened the pre-audit rescues, including
GPU training. This overrides the earlier stop on unchanged JEPA experiments
for this bounded diagnostic matrix. It does not reopen broad architecture or
hyperparameter sweeps.

## Question

Did audit defects invalidate the old rescue verdicts, and is the remaining
failure representation quality, forecast direction, dynamics gain, or site
shift? Historical A15/A16/A17/A33/A34/A37 are hypotheses to repeat, not current
evidence about the survival-free checkpoint or learned lesion forecaster.

## Control and contracts

- Control: `checkpoints/p0_clean_jepa/best.pt`, epoch index 19, validation
  loss 0.00915964925661683; optimizer-bearing and survival-free.
- Control SHA-256:
  `7186a6267d7aa42720381e6c2780f9c86fe20e71067074f8eacdf63d48199d0a`.
- The official BRAINIAC initialization, existing MRI data, 65/13/13 patient
  identities, seed 42, and locked unseen folds stay fixed.
- Primary downstream endpoint: next-observed-visit `lesion-state-v1`
  log-volume MAE, patient-uniform with paired patient bootstrap intervals.
- Every GPU arm is scored on the same unfiltered anatomy rows. Surgery
  exclusion applies to training only. Response labels may define the old
  weighting intervention; they never become treatment inputs or SAILOR gates.
- Latent error is a separate mechanism diagnostic. Each arm's own EMA gate
  compares its predictor with persistence in exactly that EMA space. A
  matched ridge state readout also predicts the fixed clean control teacher's
  future target, to detect gains created by moving the teacher geometry.
- SAILOR stays local and is developmental transfer. Gap days are recorded
  auxiliary metadata, not verified horizons or biological rates.
- Existing caches/checkpoints are immutable; reruns write new artifacts.

## CPU battery

`src/harness/analysis/clean_rescues.py` rebuilds SAILOR image/state/EMA features
with visit IDs, schema 2, survival-free provenance, and checkpoint hash. The
old separate target cache has no provenance and is deliberately not adopted.

Prespecified arms: raw prediction; support-fitted scalar displacement gain;
support-only mean, diagonal, and covariance (CORAL, epsilon 0.01) alignment;
and the original 256-hidden, 400-epoch conditioned/unconditioned field.
Five fixed patient folds reproduce the seed-42 SAILOR partition. Alignment
uses only support inputs and the 65 encoder-train LUMIERE reference patients;
predictions are inverse-mapped into the original SAILOR EMA space before
scoring. Query targets never fit a calibration. Field all-pairs integration
reproduces A15 as a recorded-gap sensitivity, not a rate claim. Persist both
weights and per-row predictions. Site discrimination uses patient-held-out
predictions, correcting the old in-sample separability probe.

The gain/direction decomposition includes the old future-magnitude oracle as
an explicitly ineligible diagnostic. A rescue passes only if its deployable
patient-uniform paired difference from untouched persistence excludes zero.

```bash
python scripts/harness.py legacy run clean-rescues --encode-sailor
python scripts/harness.py legacy run clean-rescues --field
```

## GPU matrix

1. Basin-hold: `{fresh optimizer, loaded optimizer}` × `{accumulation 1, 8}`,
   all resumed independently from the exact clean control for six epochs,
   learning rate 2e-5, no warmup. Save each best and last checkpoint.
2. Augmentation: 30 epochs from official initialization; augmented online
   MRI and clean EMA target, otherwise the clean P0 recipe.
3. Surgery filtering: 30 epochs from official initialization; the historical
   pre/post-operative and 91-day exclusion alone. This is a reproduction of
   R7, not a verified RANO 2.0 timing rule.
4. Transition weighting: 30 epochs from official initialization; old R13
   weights estimated only from encoder-training patients, with current clean
   missing/operative label handling.

The three scratch arms use batch 1, accumulation 8, learning rate 1e-4,
5-epoch warmup, and the same 30-epoch schedule as the completed P0 control.
They are separate runs. The earlier bundled A32 recipe cannot identify an
individual rescue, and a null short resume cannot replace this scratch test.

The run code is `src/harness/train/rescues.py`. Pushable notebook sources
live in `kaggle/kernel-clean-*`; each pins the trainer snapshot, checks for T4,
mounts the completed `nairadithya/p0-clean-jepa-retrain` as an input, validates
the baseline content hash and optimizer, records full configs and environment,
recomputes physical measurements from existing private LUMIERE masks, and
fails on mismatched anatomy rows. No new MRI or SAILOR upload is needed.

The user required private Kaggle delivery and no GitHub publication. The
notebooks clone the already-published base `768cd7e`, then apply an embedded
10-file code/test overlay from local commit `36b7078`; it contains no data,
weights, credentials or research notes. Overlay SHA-256 is
`fbb4d928bdb73717accc16b33214b3698950058ecb5794aa1c04fbd4c76170e8`.
Its digest is recorded in the run config, and a local Kaggle Git commit
records the applied files for cache provenance. The overlay also carries
`src/train/trainer.py` (the optimizer-resume device fix below; the pinned
public base commit has the pre-load move that Kaggle's torch undoes).
All five kernels have
`is_private: true`; the experiment branch remains local.

## Status

Implementation, cutoff tests and real one-epoch training smoke passed;
all 91 clinical vectors are finite and survival-free. SAILOR re-encoding
finished (27 patients, 270 visits, 243 next-visit image pairs). Gap/velocity
head reruns finished (A41); alignment/site-field reruns are in progress.
Private causal-lesion and basin kernels launched on 2026-10-02. Kaggle allows
two simultaneous batch GPU sessions; augmentation, surgery filtering and
transition weighting are queued in that order. A one-shot local coordinator
under `outputs/clean-rescues/` launches queued runs and saves logs, metrics,
configs and weights as slots free up. No old GPU rescue verdict is reinstated
until its cleaned rerun finishes and its gates are checked.

## Added representation-head reproductions

The user also named the gap-conditioned and velocity heads. The clean-target
battery therefore includes the original 1024-hidden two-layer gap MLP (300
fixed epochs, 1/n pair weighting), plus the 256-hidden velocity field (400
fixed epochs, three Euler steps). Fit on LUMIERE encoder-training patients and
report val/test and unchanged SAILOR results by horizon; separately refit the
gap head with fixed patient-held-out SAILOR folds. The original treatment
field CV is already in the CPU battery. Persist every trained head and target
normalizer. Supplying a retrospective target gap defines a conditional latent
reconstruction diagnostic, not an unqualified prospective next-visit claim.

```bash
python scripts/harness.py legacy run head-rescues
```

## A40 — Additional cutoff audit discovered during setup (2026-10-02)

The learned lesion forecaster formed transition `[observation[t],
observation[t+1]]`, integrated its GRU state, and added the residual to
`volume[t]` to predict `volume[t+1]`. Both the target image/lesion mask and
its ending gap therefore entered the forecast. This invalidates the reported
P2.4 forecasting gains and all adaptation based on these checkpoints.

Changing only the next-visit feature to repeat the current feature changed
the first prediction in **23/23 locked LUMIERE patients for every seed**:

| Seed | Median max-compartment change | Maximum change | Maximum future-gap effect |
|---:|---:|---:|---:|
| 42 | 0.4742 | 4.7007 | 0.0144 |
| 43 | 0.4919 | 3.6816 | 0.0063 |
| 44 | 0.4632 | 5.5787 | 0.0274 |

Units are absolute changes in predicted log-volume, not forecast errors.
The diagnostic keeps all past inputs and target volumes fixed. Raw audit
output is `outputs/lesion_causality_audit.json`.

A second mismatch: the old learned LUMIERE cache contains **123** pairs,
whereas the frozen unseen anatomy manifest has **117** usable consecutive
pairs. Dropping visits with no mask created cross-visit bridges. The fix
retains visit identities and explicit adjacency/positive-gap eligibility,
then filters losses and metrics rather than relabeling the next measured
visit as the next observed visit.

`history-only-lesion-v2` now uses `[observation[t-1], observation[t]]` and the
last elapsed gap. The first visit has a zero historical transition. Numerical
perturbation and gradient tests require zero dependence on every future
image, gap, and volume at every cutoff, with a nonzero residual head. Old
checkpoint tensors stay intact, but forecast/adaptation entry points reject
them until causal retraining. A new clean-initialized causal lesion run takes
priority over interpreting further lesion-model rescues.

## A41 — Clean gap/velocity representation-head reruns (2026-10-02)

Both heads fit only the 65 LUMIERE encoder-training patients. The gap head
uses 300 fixed epochs; the unconditional velocity field uses 400. Neither
uses test/SAILOR scores to select an epoch. Ratios below are patient-uniform
cosine error divided by unchanged same-space persistence; 1.0 is the floor.
Intervals bootstrap the paired error difference over patients (10,000 draws).
The historical 13-patient test split is repeatedly used development evidence,
not a fresh confirmation set.

| Cohort / horizon | Champion | Gap head | Velocity field |
|---|---:|---:|---:|
| LUMIERE historical test, 1 visit | 0.449 | 0.358 | 0.913 |
| LUMIERE historical test, 2 visits | 0.780 | 0.518 | 0.718 |
| LUMIERE historical test, 3 visits | 1.075 | 0.579 | 0.694 |
| SAILOR zero-shot, 1 visit | 7.409 | 6.056 | 1.051 |
| SAILOR zero-shot, 2 visits | 5.062 | 4.147 | 1.109 |
| SAILOR zero-shot, 3 visits | 3.939 | 3.213 | 1.156 |

At horizons above one visit, the champion column reuses its one-step output
as a reference; it is not a rollout or a fitted multi-horizon head. Patient
counts vary with available trajectory length (LUMIERE test 13/10/8 and
SAILOR 27/27/26 at horizons 1/2/3).

The gap head improves the historical in-domain one-step error (paired
patient-uniform difference CI [-0.021754, -0.002870], 12/13 patients), but
its SAILOR error stays far above persistence. The velocity head reduces the
SAILOR one-step error from 0.035023 to 0.004968, near the persistence error
0.004727, without demonstrating a win: paired patient-uniform CI
[-0.000117, +0.000550]; pooled CI [+0.000094, +0.000650] indicates pooled harm.
Both two- and three-step velocity patient-uniform intervals include zero.

The separately patient-held-out SAILOR gap refit gives ratios 1.127, 0.838
and 0.696 at horizons 1/2/3; all three paired patient-uniform intervals include
zero. This is developmental adaptation, not zero-shot transfer. The reported
target gaps are retrospective conditioning metadata, not verified prospective
horizons. No physical lesion-volume result follows from these latent metrics.

Weights, train-only normalizers, loss curves, fold support/query lists and
row-wise predictions are retained in `outputs/p0_head_rescue_models.pt`;
all horizons and both bootstrap aggregations are in
`outputs/p0_head_rescues.json`, copied to the main workspace's ignored
`outputs/clean-rescues/` for durable access.
