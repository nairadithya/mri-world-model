# P2.3 lesion-local architecture gate

The frozen representation experiments did not test a compartment-specific
observation encoder. They pooled the whole tumor on BRAINIAC's whole-volume
6×6×6 patch grid, and the matched forecasting ablation PCA-compressed global
features. A null result there cannot exclude local enhancing, necrotic, edema,
or peritumoral appearance signal.

The executable audit `python scripts/harness.py anatomy coverage` measured all
868 mask-bearing visits. Enhancing disease has fewer than two effective tokens
in 40.4% of LUMIERE and 42.4% of SAILOR visits. Necrotic/nonenhancing disease
has fewer than two in 31.7% and 64.7%, respectively. Median effective support
is only 2.53/2.32 tokens for enhancing and 2.62/1.65 for necrotic disease
(LUMIERE/SAILOR). Edema is better resolved (median 6.44/6.19), and the
one-token adjacent ring has median 78/84 patches.

The audit is descriptive; its threshold was not preregistered before looking at
these support statistics. We now freeze the operational rule for subsequent
comparisons: more than 25% of visits below two effective tokens for either core
compartment in either cohort forces a crop. It is exceeded in both cohorts, so
whole-volume patch pooling is rejected as the primary lesion representation.
P2.3 moves to a 64-voxel lesion-centred crop resized to the 96-voxel BRAINIAC
input. The model will retain separate compartment, adjacent-ring, and
compartment-minus-ring tokens by modality. Whole-volume/global features remain
controls. This decision changes spatial sampling, not the frozen endpoint,
patient folds, or persistence-centred evaluation.

## Implementation checkpoint

`harness.py anatomy lesion-encode` now materializes the proposed observation
state from the frozen champion backbone. For each visit it keeps modality ×
region tokens for the three compartments and adjacent ring, plus each
compartment-minus-ring contrast. The cache records source paths, mask-derived
patch weights, checkpoint/config/manifest hashes, crop geometry, and the
current-visit information cutoff. This is an implementation milestone only;
the TODO remains open until the full cache is evaluated on locked folds and
unchanged SAILOR transfer.

The first full-cache attempt was stopped before serialization because its
SAILOR path resolver checked existence but not content. SAILOR includes
present-yet-empty base volumes, so the encoder now mirrors the established
central finite-nonzero guard and falls back to `-icor` only when usable. No
scores or cache from the invalid partial attempt were retained.

## Frozen crop forecast result

The corrected cache contains all 868 visits. The matched evaluation used the
same 26 encoder-unseen LUMIERE patients, five patient-separated folds,
fold-local 16-component PCA, structured physical features, ridge selection,
zero-initialized residual MLP, and unchanged SAILOR transfer.

| Branch | LUMIERE relative MAE | SAILOR relative MAE |
|---|---:|---:|
| Persistence | 1.000 | 1.000 |
| Structured residual | 0.997 | 1.288 |
| Compartment residual | 1.001 | 1.263 |
| Compartment-minus-ring residual | 0.971 | 1.867 |
| Compartments + contrasts residual | 1.005 | 1.286 |

The 0.971 LUMIERE contrast result is not a pass: its paired patient-cluster
MAE-difference 95% CI is -0.164 to +0.092 log-volume units. All lesion-local
branches lose on unchanged SAILOR transfer. Spatial zoom and explicit regions
therefore do not rescue the frozen champion representation. The next planned
diagnostic asks whether these tokens can decode current anatomy at all; that
separates a representation failure from absence of prospective change signal.

## Current-anatomy diagnostic

On locked LUMIERE folds, combined compartment and contrast tokens decode the
current three log-volumes with patient-uniform MAE 1.141 and variance-weighted
R² 0.691. Compartment tokens alone reach R² 0.685; contrasts alone reach
0.414. Thus the frozen lesion-crop representation does contain current anatomy
in-domain even though it does not improve next-visit forecasting.

The unchanged combined decoder transfers poorly to SAILOR (MAE 2.179, R²
-0.155), although it is slightly better than the LUMIERE train-mean control
(MAE 2.289, R² -0.323). This exposes a separate site-invariance problem. The
next architecture test uses explicit consecutive-visit token differences and
history aggregation; merely strengthening a current-state decoder is not
supported by the forecast result.

## Frozen transition diagnostic

The cache was next evaluated with explicit consecutive-visit token changes:
the last transition, the last plus mean historical transition, and current
observation plus both transition summaries. These remain deterministic frozen
features; they are not the proposed learned GRU.

| Branch | LUMIERE relative MAE | SAILOR relative MAE |
|---|---:|---:|
| Last-transition residual | 0.994 | 1.368 |
| Transition-history residual | 0.991 | 1.288 |
| Observation + transitions residual | 0.987 | 1.423 |

Every LUMIERE paired CI crosses zero; every unchanged SAILOR result is harmful.
This closes further arithmetic recombination of frozen crop features. The
remaining P2 architecture branch is learned: anatomically supervise the
lesion observation space on the 65 encoder-train patients, learn transition
tokens over consecutive visits, aggregate them with a small elapsed-time-aware
GRU, and retain the persistence-centred residual decoder. The 26 encoder-unseen
patients remain outside representation training.

## Learned run deployment

The learned branch is implemented as a two-stage training procedure that fits
within a 16 GB T4. Stage 1 anatomy-supervises BRAINIAC LoRA plus a 256-d lesion
observation projection one mask-bearing visit at a time. Stage 2 freezes that
encoder, materializes adapted tokens, and trains a learned transition MLP,
elapsed-time-conditioned 256-d GRU, current-anatomy auxiliary head, and
zero-initialized persistence residual decoder. The 65-patient encoder-training
pool is split internally for selection; all 26 encoder-unseen patients remain
untouched until the final locked Kaggle evaluation.

The recovered LUMIERE masks were uploaded as the private Kaggle dataset
`nairadithya/lumiere-lesion-supervision` (mask supervision only; no SAILOR
content). The local one-patient structural smoke completed both stages and
confirmed that the initial forecast is persistence. Its score is not evidence.

Kaggle kernel `nairadithya/lesion-transition-leg` was deployed privately on a
Tesla T4. Versions 1–2 were mount-path diagnostics; version 3 is the completed
scientific run described below. SAILOR data never left the local machine.

## Learned run result

Kernel versions 1 and 2 failed before training because Kaggle exposed the
private mask dataset first as neither the assumed directory nor archive path.
The loader was corrected to discover Kaggle's extracted root-level `Patient-*`
layout and accept its `.nii` files. Version 3 completed on a Tesla T4 using
trainer commit `18d211d`; no failed-run scores were retained.

Anatomy LoRA ran for eight epochs. Its visit-level training loss was noisy but
fell from 2.152 at epoch 1 to 0.743 at epoch 6 before rising to 1.420 at epoch
8. The transition stage selected epoch 28 on the internal development split:
MAE 1.419 versus persistence 1.768 (relative 0.803). The downloaded checkpoint
is `outputs/lesion-transition-leg/lesion_learned.pt`.

Local re-evaluation produced paired patient bootstrap uncertainty:

| Cohort | Patients | Learned MAE | Persistence MAE | Relative | Difference 95% CI |
|---|---:|---:|---:|---:|---:|
| LUMIERE encoder-unseen, mask-eligible | 23 | 1.183 | 1.559 | 0.759 | [-0.638, -0.154] |
| SAILOR unchanged transfer | 27 | 0.987 | 0.943 | 1.047 | [-0.082, +0.166] |

This is the first learned lesion architecture to beat persistence with a paired
CI excluding zero in held-out LUMIERE patients. The result covers 23 rather
than all 26 encoder-unseen patients because three lack two recovered
mask-bearing visits; it must not be reported as a complete 26-patient result.
Unchanged SAILOR transfer is statistically tied with persistence and slightly
worse in point estimate, so the external-site go condition is not satisfied.
The next problem is site invariance/adaptation, not additional in-domain
transition capacity.

## SAILOR support-only adaptation

Zero-shot transfer remains frozen at relative MAE 1.047 on all 27 SAILOR
patients. A separate adaptation experiment fixed seven query patients
(`sub-01`, `05`, `09`, `13`, `17`, `21`, `25`) and used the other 20 only as
support. For each of 20 seeds, support order was randomized once and K=1, 3,
5, 10, and 20 sets were nested. No query target entered calibration.

Simple compartment-wise bias correction approaches persistence only at large
K (median relative MAE 0.992 at K=10 and 0.987 at K=20). A ridge calibrator of
predicted change using raw predicted delta, current anatomy, and elapsed gap is
more useful: median relative MAE is 0.968 at K=5, 0.901 at K=10, and 0.863 at
K=20. K=10 varies from 0.881 to 0.934 across the central 80% of support
orderings.

The full-support K=20 query result is MAE 0.764 versus persistence 0.885, but
the paired seven-patient bootstrap difference CI is -0.306 to +0.054 and still
crosses zero. Therefore support calibration identifies a correctable site
shift but does not satisfy the external gate. It is reported separately from
zero-shot and must not be described as site-generalization success.
