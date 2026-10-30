# Independent E observation-noise result review

## Review
- **Correct:** Saved results, provenance, row construction, calibration selection, CRPS calculations, patient aggregation, and paired bootstrap agree with independently implemented numerical references.
- **Fixed:** None; read-only review.
- **Findings:** No issues found.
- **Merge verdict:** **OK**, limited to the completed E ablation.
- **Result verdict:** **APPROVED_FOR_SCOPED_ABLATION_REPORTING**

## Reviewed scope

Reviewed `scripts/run_spatial_noise_ablation.py`, its baseline/history helpers, `info/METHODS.md:364–416`, and the v6 contract, preflight receipt, budget ledger, execution log, aggregate results, private row manifest, and saved numerical arrays.

Contract digest:

`1d0fc791c9340487670f951a88b5362095c9feb32e559a2e0347244667e0ba86`

This approval covers descriptive reporting of the two E observation-noise conditions—not full R37 completion, capability success, or selection of an E condition for subsequent capability comparisons.

## Verified checks

1. **Native coordinates and change direction.** Independently reconstructed every saved target from prepared native volumes using compartment order `necrotic_non_enhancing`, `enhancing`, `edema_flair`. Verified `source_changes = target_log1p − latest_decoded`, not measured-to-measured change or standardized residuals. Evidence: runner lines 90–98, 221–226, 244; baseline helper lines 127–134.

2. **Frozen decoder replay.** Independently applied the saved affine head and inverse target normalization to cached embeddings for all 262 source, 60 validation, 75 history-query, and 26 single-visit rows. Maximum discrepancy was `1.7763568394002505e-15`. No encoder forward pass or MRI loading occurred.

3. **Rows and separation.** Independently rebuilt the full eligible manifest from the 638-visit original chronology and prepared-pair metadata. It matched exactly: 423 eligible rows, including 262 source, 60 validation, and 101 assessment rows. The primary assessment contained **75 rows from 18 patients**, exceeding the 15-patient minimum. Original 52/13/26 patient memberships matched the frozen run and were mutually disjoint.

4. **History eligibility.** Reconstruction respected chronology adjacency, contiguous usable history, the four-visit maximum, and positive historical week-bin gaps. Same-week target gaps were not incorrectly used as eligibility exclusions. Evidence: runner lines 29–88; `spatial_dynamics_protocol.py:55–119`.

5. **Matched conditions and weighting.** Both conditions use identical source vectors, source patient weights, validation rows, and query rows. Source weight is `1/(number of represented source patients × rows for that patient)`. Calibration preserves the weighted source mean and jointly scales each three-coordinate change vector. Only Gaussian observation noise differs. Evidence: baseline helper lines 27–40, 103–150.

6. **Independent CRPS reference.** Recomputed every saved validation-grid and assessment score without importing project scoring helpers. Used
   \[
   \mathrm{CRPS}=E|X-y|-\tfrac12E|X-X'|
   \]
   with exact weighted absolute differences for discrete mixtures and an independent `erf`-based Gaussian absolute-moment formula for Gaussian mixtures. The Gaussian pair term used standard deviation `sqrt(2) × 0.05`. Maximum saved-array discrepancy was **`3.108624468950438e-14`**.

7. **Validation-only calibration.** Independently evaluated `[0.5, 1, 2]`; both conditions selected **k=1** on validation CRPS. Source fitting and calibration occur before query scoring. No outcome-dependent condition selection appears in the reviewed execution; the aggregate records `assessment_winner_selection: false`. Evidence: runner lines 228–232, 257; helper lines 137–160.

8. **Patient aggregation and bootstrap.** Independently averaged compartments and rows within each patient, then patients equally. Replayed both 10,000-draw paired patient bootstraps with seed 2026, including the continued RNG stream for the second contrast. Maximum bootstrap-array discrepancy was **`3.5680009294325465e-15`**.

9. **Integrity and completion.** Preflight and execution contracts matched exactly; canonical contract digest, runner/helper hashes, all eight recorded input hashes, manifest hash, and result bindings passed verification. Preflight status was `PASS`; aggregate and ledger status were `completed`.

## Aggregate result

Primary metric: patient-uniform marginal CRPS in `log1p(mm3)`; lower is better.

| Common k=1 | CRPS |
|---|---:|
| Discrete | 1.2230516797168902 |
| Gaussian observation | 1.2226323341078846 |

Gaussian minus discrete:

- Mean: **−0.0004193456090056126**
- Paired patient-bootstrap 95% interval: **[−0.0005599483034065046, −0.00026353003575198714]**
- Relative reduction: **approximately 0.0342868266285%**

Both validation-selected multipliers were k=1, so the selected-setting mean difference is identical. Its interval is slightly different because it uses the next bootstrap draws:

**[−0.0005597607656571536, −0.0002647988781426919]**

All grid scores and separately reported single-visit scores also replayed successfully.

This is a **tiny descriptive proper-score difference from Gaussian smoothing**. It is not evidence of meaningful anatomical improvement, validated uncertainty calibration, longitudinal-history benefit, spatial-history benefit, or world-model capability.

## Execution receipts

The reviewer authored the independent references. **The parent executed them** because this reviewer had no execution tool; the reviewer inspected the returned stdout and empty stderr. These were not reviewer-executed commands.

Commands:

```text
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/adithya/projects/world-model/.venv/bin/python -B - < /tmp/r37-independent-replay.py > /tmp/r37-independent-replay.json 2> /tmp/r37-independent-replay.stderr

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/adithya/projects/world-model/.venv/bin/python -B - < /tmp/r37-decoder-replay.py > /tmp/r37-decoder-replay.json 2> /tmp/r37-decoder-replay.stderr
```

Both returned **exit 0 / PASS**. Reported timed replay sections consumed respectively **0.88534 CPU / 0.89634 wall seconds** and **0.02256 CPU / 0.02307 wall seconds**. CPU and wall limits were installed. No training, re-encoding, network access, or scientific-file modification occurred.

Parent-reported receipt hashes:

| Artifact | SHA256 |
|---|---|
| Main reference script | `78cdc8466d6e4e5005f929a32e0a50ec907c66f0e8b0fb558ae9d162692c07a3` |
| Main stdout | `ed4e0517da6504728cf4075a1878f1118ed647113357e6392b35816df0e93114` |
| Decoder reference script | `f1ab20f17fb2e35e1e1c17c1b4749a11daf352b13d4156469883d2e9f985b70d` |
| Decoder stdout | `18fe09a3f468ada88586d1cbacc4498ebbda8435913fc451e8f275ab82f88a86` |

## Reviewed-file SHA256

Computed by the reviewer-authored reference during parent-host execution.

Paths below `outputs/r37-noise-ablation/v6/`:

| File | SHA256 |
|---|---|
| `run_contract.json` | `7695562d8000181e8981c9ab032860e9c81900207f4d277d06a1811601fde877` |
| `preflight_receipt.json` | `973e8810b0445ec18d48ee347e1ab77623accdf53d049e75bfd3fb2a9af836c6` |
| `budget_ledger.json` | `e0fc8314ca471ef5892cadff430ed3939fd9bdf138a5bd5434310d20f065ba55` |
| `execution.log` | `e62677d9250b3600a0bbc189d68a6a31d438fc0a28968e564fdb2a86314c8fd8` |
| `run/aggregate.json` | `9662947bd1cad598445e609bd2f856e570169219e824240b57871408df8c7585` |
| `run/contract.json` | `7695562d8000181e8981c9ab032860e9c81900207f4d277d06a1811601fde877` |
| `run/private_predictions_scores.npz` | `8081eaaceb63dd725ed949666369b646634425461f6c2a7f1e324f63a513db8a` |
| `run/private_row_manifest.json` | `8ed2b2c8b071c9d4342e1e354cb8306d0e0030f7346ef8537d61d177a84234e9` |

Code and protocol:

| File | SHA256 |
|---|---|
| `scripts/run_spatial_noise_ablation.py` | `251cb615e460a3f7d47d727c38223dd703bb5838c67956f89a0945809183d6a3` |
| `src/harness/analysis/spatial_dynamics_baselines.py` | `acbe7ddf99b3c39236cb1a99995c621ca08ad9e4bc5003421deb312ca70b8f37` |
| `src/harness/analysis/spatial_dynamics_protocol.py` | `9e29b9efc90699dde135b15d5705ef06bac1d5d473496daa2a3ead0edc358d19` |
| `src/data/representation.py` | `65537546257c70ce978fde9b254b7ef75c49b4797a8c279da3542c75f6dcee6a` |
| `info/METHODS.md` | `ac38cc4b7b3e501e325dbec71dc211d395e6bef5e4b8ed7d620abaaedc91e36d` |

Original input hashes independently recomputed and matched:

| Contract input key | SHA256 |
|---|---|
| `prepared` | `90e9b986bfc638866391ea0ab4113e978fcb9165fbd34acc0b681970086777f1` |
| `checkpoint` | `65a4c4db2ca1b44ebf29fa12659fbd09dea6ede606ce1d3279535144933a001a` |
| `features` | `0646a3be9b0bc7de6295adcea1b4aa3462b1ac6e630ad2f55374192a77026090` |
| `direct_predictions` | `9e7ecd09b8ce257d4c66bb23814850889520890da3fb0726e1b50d997ff8e311` |
| `run` | `f59820fb681aafc05b9112329aebbe1fe899bfa0fe831e503afb40a2353dc6e7` |
| `inventory` | `b1e668429cbfb5d15f680764b5a192bd256fcf1b10b03a1df081b58cbc9a5e08` |
| `anatomy_manifest` | `c65bdecfe9b6a9f3da548ea1249a95cb3037a3f6876c00f8bd1ba9613c148a57` |
| `chronology_manifest` | `5fc550c2edaf2cef8c432219b9ec4f8a08c04eb33dd0c54d140f35d631a300dc` |

The canonical eligible-row-manifest digest additionally matched:

`210c655e445fb33dc3e00bdce672ebcf521205d72ef22ee90577c499d01f32f9`

## Limits

- This is reused LUMIERE **development evidence**, scored against automated masks—not independent or expert-verified biological validation.
- Week-bin differences approximate historical timing; no exact physical-rate claim is supported.
- The intervals are descriptive, unadjusted 95% ablation intervals, not the full R37 five-contrast capability decision.
- The ledger records approximately **22.821 cumulative CPU seconds** for accounted work, with a **22.113-second execution wall time**. Earlier v1 expenditure is explicitly unavailable; complete historical resource consumption cannot be certified.
- Relevant test fixtures were inspected; the unit-test suite was not executed in this review.
- Raw-image geometry and mask validity were not re-audited. This review relies on the bound prepared artifacts and previously reviewed execution checks.
- No participant identifiers or individual measurements are included here. Approval applies only to the reviewed hashes and scoped aggregate comparison.