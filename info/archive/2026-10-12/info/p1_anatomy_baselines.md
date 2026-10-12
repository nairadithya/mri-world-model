# P1 RANO-free prospective baseline gate

Status: **closed, negative for JEPA** (2026-09-22).

The executable evaluator is `src/harness/analysis/anatomy_baselines.py`:

```bash
python scripts/harness.py run anatomy-baselines --boot 10000
```

It consumes the frozen `lesion-state-v1` manifest and current schema-2,
survival-free caches. Every method predicts the three next-visit compartment
log-volumes on the same rows. Evaluation is five-fold patient CV on only the
26 LUMIERE patients never used to train the encoder. Ridge strength is selected
inside each training fold with patient-grouped CV. The transfer arm fits on
those encoder-unseen LUMIERE patients and applies the model to SAILOR without
using SAILOR outcomes for fitting or tuning.

## Result

| method | LUMIERE patient-uniform MAE | relative to persistence | SAILOR MAE | relative |
|---|---:|---:|---:|---:|
| persistence | 1.545 | 1.000 | **0.943** | **1.000** |
| mean delta | 1.552 | 1.005 | 0.947 | 1.005 |
| linear trend | 2.322 | 1.504 | 1.365 | 1.478 |
| current-volume ridge | **1.503** | **0.973** | 1.465 | 1.493 |
| volume-history ridge | 1.589 | 1.029 | 1.617 | 1.609 |
| current image | 2.047 | 1.325 | 3.182 | 3.295 |
| mean image history | 2.127 | 1.377 | 4.936 | 4.910 |
| JEPA state | 2.152 | 1.393 | 2.719 | 2.873 |

There are 496 matched LUMIERE rows in the full inventory. The locked
encoder-unseen evaluation contains 117 rows from 23 of 26 patients; three
patients have no eligible consecutive measured pair. The transfer evaluation
contains 242 SAILOR rows from all 27 subjects. The current-volume ridge's
LUMIERE improvement over persistence is not
resolved by the paired patient bootstrap (MAE difference −0.042; 95% CI
−0.227 to 0.148). JEPA is decisively worse (difference +0.607; 95% CI
+0.226 to +1.002). On SAILOR, both conclusions strengthen: current-volume
ridge is worse by +0.522 (95% CI +0.384 to +0.658) and JEPA by +1.776
(95% CI +1.444 to +2.171). Thus P1's question has an answer: the current JEPA state
does not add prospective value over the simple floor.

## Interpretation and scope

This is a next-observed-visit development benchmark, not a fixed-horizon or
clinical response claim. Log-volume level error and delta-log-volume error are
numerically identical when every method and target subtract the same observed
source volume; both names are retained in the artifact to make the target
contract explicit. SAILOR remains developmental transfer rather than a fresh
external test. Its ONCO masks are primary; the P0 CL-versus-ONCO audit bounds
measurement sensitivity but does not turn the two segmentations into
interchangeable truth.

The appropriate next move is therefore lesion-aware modeling only if it is
motivated as an attempt to repair this failure, with persistence as the gate.
The result does not support scaling the present JEPA formulation or claiming
temporal representation superiority.
