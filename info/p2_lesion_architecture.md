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
