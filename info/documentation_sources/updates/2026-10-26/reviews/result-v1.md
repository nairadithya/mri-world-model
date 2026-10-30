# Independent R37 result/artifact review — v1

October 26, 2026. This is a fresh result review of the completed fixed pilot, not an extension of implementation approval and not a requirement to obtain a positive result. The implementation-v1/v2 reports are preserved unchanged.

APPROVED_FOR_SCOPED_R37_REPORTING

**Approved scope:** reporting the audited computational completion and **INCONCLUSIVE_FOR_FULL_CAPABILITY** outcome for the exact contract and aggregate below, with their controls, uncertainty, failed gates and limitations. This is not experiment success, biological validation, independent generalization, publication novelty/acceptance, permission for rescue training, or an authorization to retry or extend the experiment.

## Inherited decisions and reviewed identity

The fixed R37 question concerns next-observed automated anatomical prediction with a frozen original R35 seed-42 encoder/head, source52/internal13/unseen26 patient membership, H/L/G/P/E/R comparisons, three predictor seeds and twenty neural epochs. Historical gaps remain approximate positive week-bin differences; no future gap, treatment or target mask is a predictor input. Gaussian-observation E was locked after the already known ablation, not as untouched/blinded prespecification. Preserve its earlier discrete/noise companion descriptively; this review does not reopen that ablation.

The pilot retains five primary H contrasts with two-sided 99% Bonferroni-adjusted patient-bootstrap intervals, separate MAE/calibration/nuisance requirements, and one-analysis stopping. All outcomes are automated development proxies in `log1p(mm3)`; reused patients are not independent validation.

- Canonical run-contract digest: `6f73bb278bfbb4afd5778e24181f78485d80aa2cb792a82732900bfbc4dc9f1c`.
- Raw aggregate SHA256: `bf36341fbe581329a9bd2cc11a49deef8a552a455dcab0fb66b4b13074b12b9f`.
- Implementation-v2 SHA256: `72a85f07f597fff07bb55ed2b41ce362b6102265d48ea5066d8ba1ae3928a5bf`.
- Implementation-v1 SHA256: `bdb9a17a5ada777ee1481440fff655e156c62d34410f8030136571a1eba533de`.
- METHODS SHA256: `5dbfeed0697b1bf71079251ae5764471755b7882ef4eb4bd00ab750d25ea1026`.
- Continuation SHA256: `f34b0d9b669bde950c8e9e5d0ed8aa510cabfd83df8c3bd4cab0afe940e3504b`.

All thirteen current source hashes match implementation-v2, the contract and the locked source snapshot; local source hashes also match. Exact source mappings are retained in the contract and independent verification receipt. Protocol, continuation and implementation-review snapshot bytes match. Local aggregate and mirrored execution receipts match remote bytes.

## Executed independent verification

I authored and remotely compiled the standalone checker before scientific verification. It executed **once**, CPU-only, on the dedicated SSH environment using `/home/aditya/experiments/world-model-r37-20261026/.venv/bin/python`. The only project API called was `budget_stage`, solely for metering. No project score, aggregation, bootstrap or `analyze` function was reused. Deserialization of saved control objects resolves their class definitions but does not call their scoring methods.

The authoritative stage `independent-result-review-v1` completed with **PASS, 12,861 reported checks and no discrepancies**. Independent numerical work included:

1. **Bindings and complete artifacts.** Rehashed the contract, approved source/snapshot, original input metadata, feature cache, prediction PT/NPZ, fitting/assessment markers and all 29 required weight artifacts. Verified fitting-completion and artifact-manifest digests, explicit seed labels, exact canonical primary/single/nuisance rows, and PT/NPZ equality. Rehashed the reviewed artifacts after verification to detect changes.
2. **Chronology, targets and split.** Independently rebuilt all 423 eligible rows from the 638-visit original chronology and prepared adjacency/usability evidence, including contiguous histories, no bridging, positive retained week-bin gaps, target/cutoff log-volumes and shared prepared grids. Verified source/validation/assessment membership against the original run and patient disjointness.
3. **Normalizer and frozen decoding.** Recomputed channel mean/population standard deviation from the 353 unique source visits, verified original decoder weights/normalizers/compartment order, and compared pooled summaries of all 599 saved maps with the original pooled cache. This is a saved-map check, not a new image/encoder extraction.
4. **Predictive laws.** Independently reconstructed Gaussian CRPS and central 50/80/95% intervals from raw saved means/scales; checked calibrated and uncalibrated scores. Independently reconstructed E's equal-patient joint change weights, mean-preserving k transformation, fixed 0.05 Gaussian observation noise, analytic mixture CRPS and numerical marginal quantiles. Checked P residual RMS/persistence means and E/P validation grids/choices. Recomputed R means from saved coefficients and cutoff features, its validation grid/choice, grouped-fold partition, selected-alpha CV CRPS using recorded fold-training noise, and final selected-alpha OOF RMS.
5. **Neural matching and selection artifacts.** Independently regenerated each seed's initialization and twenty equal-patient replacement orders. Verified H/L/G initialization/order hashes, twenty complete epoch histories, earliest minimum validation-CRPS selections, grid argmins, selected/partial weight identity, final optimizer-bearing checkpoints, AdamW settings and 5,240 optimizer steps per fit: nine fits and 47,160 total updates.
6. **Bounded fixed-weight replay.** Independently implemented the predictor's tensor operations and frozen affine distribution propagation, without gradients. Replayed the predetermined first canonical primary row and first canonical single row for every neural arm/seed: **18 CPU forwards**, not a full assessment rerun. The pre-execution forward tolerance was `atol=rtol=1e-4`; maximum observed absolute discrepancy was `1.0730050625795684e-6`.
7. **Reporting.** Reconstructed every arm/seed summary, native-compartment MAE, coverage/width, empty/appearance/disappearance/both-empty partitions, single-visit companions, nuisance drift/error/coverage companions, all five paired primary contrasts and unadjusted companions, MAE margins and secondary L contrasts. Bootstrap used 10,000 paired patient resamples with seed 2026, after averaging seed scores rather than ensemble predictions.
8. **Decisions and resources.** Reconstructed each mechanical gate and benefit flag; independently checked authority/journal hash linkage, legacy spending retention, locked ancestry and six completed original stages with no failed/repeated scientific stage.

Aggregate-score discrepancies were at most `1.34e-14`; bootstrap discrepancies at most `4.50e-15`; all subset/seed summaries matched to numerical precision. Float32 uncalibrated-scale recovery differed by at most `1.19e-7`; saved-coefficient ridge checks by at most `5.60e-7`. The checker uses `1e-8` for direct summary/bootstrap comparisons, explicit `1e-6` allowances for float32/control recovery, and at most `1e-4` for fixed-weight/linear forward comparisons. No threshold was adjusted after seeing verification results.

### Executed versus inspected limits

Actual geometry/content checking for 2,396 MRI files and original-input reconstruction for 599 visits are supported by the executed encoding receipt, bound cache/source and geometry digests. Its fixed tolerances are `atol=rtol=1e-4`, maximum pooled absolute error `1.1920928955078125e-6`, maximum native decoded absolute error `2.8893163506182873e-6`. I inspected and verified those receipt associations, but did not reread/re-encode MRI or repeat registration/alignment checks.

Neural epoch/calibration choices were checked against complete saved validation histories/grids and unchanged selection code, **not** a new validation forward sweep. All-alpha ridge selection was checked against the recorded six-value CV grid; selected-alpha CV arithmetic and final OOF RMS were independently reconstructed. Fold fits, all-alpha fold-training residual estimates, gradients and the training trajectory were not refitted/reexecuted. These are explicit limits, not claims of a complete training replay.

The approval JSON has two byte representations: the contract's `review_gate_sha256` binds original `implementation-approval.json`; the runner's sorted copy is `implementation-gate.json`. Their raw hashes differ, but their parsed content equals the contract review exactly and the original bound file remains present. Both representations were checked; no review identity change was found.

## Independently reconstructed results

**Primary cohort:** 18 reused development patients, 75 history-eligible pairs. **Single-visit companion:** 23 patients, 26 pairs, kept separate; these patients can overlap the primary cohort and are not independent validation. **Nuisance:** one earliest eligible cutoff per primary patient, 18 patients/18 examples. Below, scores average compartments per example, examples within patient, equal patients, and seed scores—not predictions.

| Arm | Patient-uniform CRPS | Patient-uniform mean-prediction MAE |
|---|---:|---:|
| H | 1.30031836 | 1.69376433 |
| L | 1.29697350 | 1.67683953 |
| P | 1.26913951 | 1.65728793 |
| E, Gaussian observation | 1.22263233 | 1.69711601 |
| R | 1.28416460 | 1.68391809 |
| G | 1.31326489 | 1.69254350 |

Units throughout are native `log1p(mm3)`. E has the lowest point CRPS among these arms; this is not a tested significant superiority claim over every comparator.

### Paired primary CRPS contrasts

Negative differences favor H. Every adjusted interval spans zero; so do the unadjusted companions. The relative-reduction column is descriptive, not a significance test.

| Contrast | Difference | Adjusted 99% interval | Unadjusted 95% interval | H relative CRPS reduction |
|---|---:|---|---|---:|
| H − L | +0.00334486 | [−0.04732935, +0.05234837] | [−0.03511312, +0.04019126] | −0.2579% |
| H − P | +0.03117885 | [−0.03592705, +0.10415815] | [−0.02152134, +0.08475012] | −2.4567% |
| H − E | +0.07768603 | [−0.03430240, +0.20740137] | [−0.01075697, +0.17421197] | −6.3540% |
| H − R | +0.01615376 | [−0.05803496, +0.08182145] | [−0.03942636, +0.06769265] | −1.2579% |
| H − G | −0.01294653 | [−0.05938712, +0.02905705] | [−0.04807005, +0.01987582] | +0.9858% |

The primary benefit criterion is not met: H does not have a 5% point improvement over L, and none of the required adjusted H-versus-control intervals lies wholly below zero. The H-versus-G interval does not establish value from retaining earlier spatial detail.

### MAE, calibration and secondary comparison gates

- H−L MAE difference: `+0.01692480`, 95% interval `[−0.02721033, +0.06738806]`.
- H−P MAE difference: `+0.03647640`, 95% interval `[−0.05329185, +0.14005208]`.
- Both upper bounds exceed the required `+0.05` margin. This fails the noninferiority requirement; it is not proof of significant inferiority.
- H's 80% coverage by compartment is **84.36%, 66.16%, 87.82%**, in order necrotic/non-enhancing, enhancing, edema/FLAIR. Enhancing coverage falls below the required 70–90% range. This is a point calibration-gate failure, not a population-calibration significance claim.
- Secondary L−P CRPS: `+0.02783399`, 95% interval `[−0.01396873, +0.06935944]`; L−E: `+0.07434116`, `[−0.03036085, +0.18425660]`. Neither establishes the required latest-state benefit, and neither rescues the primary hypothesis.

### Nuisance findings and harms

On the predefined 18-patient nuisance set, intensity perturbation changes are approximately numerical noise: H CRPS change `1.20e-7`, MAE change `1.41e-7`; L changes are similarly tiny. Coverage changes are zero in the saved comparisons. The established standardization largely cancels this affine intensity transform, so this is **not evidence of learned intensity invariance**.

Half-resolution downsample/upsample gives the following paired changes versus identity:

| Arm | Relative CRPS worsening | CRPS increase, paired 95% interval | MAE increase, paired 95% interval |
|---|---:|---|---|
| H | +19.3641% | +0.37950101 [0.10504015, 0.72258824] | +0.37930948 [0.06658814, 0.75110040] |
| L | +18.6841% | +0.37233767 [0.11416493, 0.69375683] | +0.35931552 [0.07253778, 0.70238198] |

H exceeds both the +10% CRPS and +0.05 MAE nuisance margins. Mean absolute **log-volume** drift averaged across compartments is `0.79093224` for H, 95% interval `[0.53066429,1.09873528]`, and `0.75966924` for L, `[0.50350286,1.06064980]`. Per-compartment drifts and all coverage-difference companions were verified and remain in the raw aggregate/receipt. Despite the original key name `mean_absolute_volume_drift`, these quantities are in log-volume coordinates, not mm3.

The resolution check supports a scoped sensitivity/degradation finding for this particular synthetic perturbation. Its 95% companion intervals are not the adjusted five-contrast primary family and do not establish clinical scanner effects or general robustness failure. No primary H-minus-control adjusted interval establishes degradation; the primary `demonstrated_harms` list is correctly empty. An empty list is not proof of harmlessness or equivalence.

All four composite requirements—primary contrasts, MAE margin, coverage and nuisance—are false. Therefore the mechanical classification **INCONCLUSIVE_FOR_FULL_CAPABILITY** is correct. History, earlier-spatial-detail and secondary latest-state benefits are not established.

The checker also verified all central coverage/width and empty/appearance/disappearance summaries, without pooling their denominators incorrectly. Primary target-empty counts are 6/16/4 pairs across the three compartments; appearance 2/5/2 and disappearance 3/4/3. Such subsets are sparse and some are concentrated in very few patients. Their automated-mask summaries are descriptive, not biological transition validation or additional success endpoints.

## Resource/history review

The authority was initialized from the exact reviewed legacy ledger hash `cb717e6205a4af7e912e18854f19892c0a3f9169206cb48a22163aed3386c695`, preserving inventory `0.844038823` CPU seconds and the 900-second setup charge. Earlier expenditure remains separate; unknown original v1 expenditure is still unknown, not zero.

The six original stages—inventory, encode, lock, fit, evaluate, report—are complete, unique and within the authorized window. Pre-review GPU wall charge is `299.1753105649259` seconds; scientific-stage CPU `305.181922939` seconds plus setup 900 gives `1205.181922939` charged CPU seconds.

This independent review added **2.98843375 CPU seconds**, **2.2233256290201098 wall seconds**, and **zero GPU seconds** under the existing authority. Post-review charged CPU is `1208.170356689` seconds; GPU charge is unchanged. The current ledger/journal contain this seventh completed verification stage. No authority reset, retry, model update, data modification or unrelated process control occurred. Limits remain 28,800 GPU seconds / 57,600 CPU seconds and the original continuation deadline, October 28 at 10:00:04 +05:30; remaining allowance does not authorize more experiments.

## Exact artifact and evidence hashes

Unless noted, paths in this table are relative to the remote private output directory `/home/aditya/experiments/world-model-r37-20261026/repo/outputs/r37-full/continuation-20261026/`. Raw tensors, weights, participant inventories and predictions remain private.

| Artifact | SHA256 |
|---|---|
| `aggregate.json` | `bf36341fbe581329a9bd2cc11a49deef8a552a455dcab0fb66b4b13074b12b9f` |
| `contract.json` (raw bytes, distinct from canonical digest) | `1f8c4df94ddc4a2ec52949c26e131cbae2c826948323baf22b82143c79fcdabc` |
| `features.pt` | `cd0a12597b2fffc2cba2699a69aee6c6852dadef7941c0b57e256fea6d37b91e` |
| `features.json` | `719f868147ac7c776027c1a237f4f1276a555776b4b58dfafd9a4ef2c4308cd0` |
| `private_predictions.pt` | `b7d21485cab713df3901e032f4bd54a525c8eba6914ce6427319dc7d4ff883d8` |
| `private_predictions.npz` | `f7d2a6ee8394d5da30252717725281f4199a9053b143be2179cb5027760cc4bb` |
| `assessment-complete.json` | `408528aa014bef7b60f5eeb33b037c08cd0778e740a1cfd32c21dd3f83ae64cc` |
| `fitting-complete.json` | `92acfe746152192fb686dfbfbd3e8578ae350d74e300e2ea0a0d3b07fa5378ce` |
| `controls.pt` | `26c235cb43a0028665f6f5a056d3fbcf19878220c71f79a82225a889359cafa6` |
| `ridge/ridge.pt` | `eb0368955f112f96071de8a28f908dda7b04e15fcc7a785582ffc2b0936d5b6a` |
| `ridge/source_oof.npz` | `7b8e92c850dd9e070d0aafa9e057de82263eef0fefd5c93fbb7f6f41158b7ecc` |
| `ridge/complete.json` | `5ba98ca0c546f43f744acd60d5e8bf2d7ec01cdd09e5d947032719713a3282d0` |
| `implementation-approval.json` | `47927f66d3b2ca3b59c67d588387a6acb5049e9fc6d64b19f209c4d26d5bb51e` |
| `implementation-gate.json` | `fb637895bae23e3afad7e9098917ba853579688afaf02ec5a82157eb87ae8d55` |

The exact 29-weight mapping and hashes of all three saved order arrays are recorded in the independent receipt. Canonical SHA256 of the required weight manifest is `5fc1b3af0f2032b5091e68b11b1ca3ee54a00a023b0ed3d0ad7c2982ca3be45e`. Geometry-receipt digest is `ddc442302be27d18ac946c74f5573a71afb126b5529d170958334ae8404cb273`.

Execution receipts below are relative to the dedicated workspace's `receipts/` directory and match the local mirrored copies:

| Receipt | SHA256 |
|---|---|
| `authority-initialization.json` | `877da48294488bda31a95689b5342c3a1409eafe0b211ad01b278f5ac3ded558` |
| `encoding-gate.json` | `65766b582bf7186ba7921382310bc4f93dc85da51d8c26594e940660162073c6` |
| `fitting-gate.json` | `602832ed26d926820db5f43d15588e797616f254b5f037bf6c93b82f51b124eb` |
| `execution-complete-v1.json` | `adb5f8c04f88678875f61563bea07ed36e078fd97c6d6fed4f3d200454a00326` |
| `execution-summary-v1.json` | `2c31b0c0186f50ae9d390afc9ee597079ca8c081dd2d597344f6a3ec73b9de00` |

Post-review authority anchor SHA256 is `f4ed1e40adaa3460709132ebc1715d451fba5bd6cda42fe2c006e55062d8ddc8`; ledger snapshot `5edf94822ad83fb383edbea1c4799ae23483fc22f27da14a6d61f50e100b1f10`; journal snapshot `b1dd9bcd7c5996f0a09cecf22acf4a579585e3c3273976f02cdeaadcc21901dd`. Ledger/journal hashes describe this completed-review snapshot, not immutable future hashes.

## Recommendation and permitted claims

Report the exact completed negative/inconclusive pilot with its full comparisons and limits. A suitable conclusion is: **“The fixed R37 pilot did not establish added-history or earlier-spatial-detail benefit on the reused LUMIERE development cohort; uncertainty spans no difference, calibration and resolution-nuisance requirements were not met, and the full capability classification is inconclusive.”**

Do not claim equivalence, absence of useful MRI-history information in general, expert biological truth, independent generalization, clinical calibration, fixed-day/multistep prediction, treatment effects or significance from a point gate failure. Resolution sensitivity can be reported with its explicitly scoped paired companion intervals. Computational completion and scoped reporting approval are not a methodological novelty or publication-acceptance verdict.

Stop this formulation after this one analysis, regardless of the unused compute allowance. No rescue training, new condition selection, further calibration, new cohort/target or automatic recovery is authorized. No executor handoff is warranted. The parent may preserve the exact checker/receipts and document these aggregate results; no unresolved blocking decision is needed for this scoped reporting approval.

## Retained verification outputs

All files below are in the dedicated remote workspace, outside the repository, and must be preserved exactly. The bound receipt includes the complete reviewed artifact/source mapping and the aggregate numerical reconstruction; no participant identities or individual predictions are emitted in it.

```json
{
  "verdict": "APPROVED_FOR_SCOPED_R37_REPORTING",
  "experiment_success": false,
  "decision": "INCONCLUSIVE_FOR_FULL_CAPABILITY",
  "contract_digest": "6f73bb278bfbb4afd5778e24181f78485d80aa2cb792a82732900bfbc4dc9f1c",
  "aggregate_sha256": "bf36341fbe581329a9bd2cc11a49deef8a552a455dcab0fb66b4b13074b12b9f",
  "reviewed_code": {
    "scripts/run_spatial_dynamics.py": "d56053c5a987bcf51ab7589ae6f20432842f591180e1a8a762ed57a7e5b008bc",
    "scripts/run_spatial_noise_ablation.py": "251cb615e460a3f7d47d727c38223dd703bb5838c67956f89a0945809183d6a3",
    "src/model/spatial_dynamics.py": "c7c1b87d24de8b083569d29b5021cb994321cf068fd0f7dd1021df9fe103952a",
    "src/model/representation.py": "d73125b5f05dca864cb308773bcd1924a7f40552371374753d94c147bff547f6",
    "src/data/representation.py": "65537546257c70ce978fde9b254b7ef75c49b4797a8c279da3542c75f6dcee6a",
    "src/preprocessing/transforms.py": "c910a8ad41d62534fa3b9fe2f421be5260d91baed806ed679e03177dbb3a6bc1",
    "src/preprocessing/geometry.py": "f09b91bf2ec829297c64efb6606726a3206c98d7dacff4c2cc6db28c5d61fb2d",
    "src/harness/analysis/spatial_dynamics_data.py": "8a9a2d5db9bfc680f8a0cf73f4e8f18bb88b7daccf909aafa7a749ae8ece308c",
    "src/harness/analysis/spatial_dynamics_fit.py": "7f738e076c592d6bf045ccb1a835f10f81c16525ab32b7a37287529c7f5e629a",
    "src/harness/analysis/spatial_dynamics_report.py": "b1e4f14417fc89a4de64f8a0a648d50c0b75b9e880c63459a67cbb2e0865b7ad",
    "src/harness/analysis/spatial_dynamics_runtime.py": "3bbfa0f3d9ec82139b593b2a71f3e59b7dfe3b31eee746b9ff1932c35dd85638",
    "src/harness/analysis/spatial_dynamics_baselines.py": "acbe7ddf99b3c39236cb1a99995c621ca08ad9e4bc5003421deb312ca70b8f37",
    "src/harness/analysis/spatial_dynamics_protocol.py": "9e29b9efc90699dde135b15d5705ef06bac1d5d473496daa2a3ead0edc358d19"
  },
  "artifactPaths": [
    {
      "path": "/home/aditya/experiments/world-model-r37-20261026/verification/independent-result-review-v1/checker.py",
      "sha256": "0936e420322f8301f287e9f1d932af95893c81858f052d3f50a78dff6c93adee"
    },
    {
      "path": "/home/aditya/experiments/world-model-r37-20261026/verification/independent-result-review-v1/compile.log",
      "sha256": "85df89b07ca32c1bcd3b40a8c43e389da9adaf5ab4038d893dec29143c4424f0"
    },
    {
      "path": "/home/aditya/experiments/world-model-r37-20261026/verification/independent-result-review-v1/verification.log",
      "sha256": "65823f5fd152cffe635dc6c403c2a0cabf5970bde07cf16f1d6513c4ec35148e"
    },
    {
      "path": "/home/aditya/experiments/world-model-r37-20261026/verification/independent-result-review-v1/verification-staged.json",
      "sha256": "36abcbd6e2e398a3c7962fa1e7e13ffd353dcd180e6094af213fe51127be7f19"
    },
    {
      "path": "/home/aditya/experiments/world-model-r37-20261026/verification/independent-result-review-v1/verification-receipt.json",
      "sha256": "afb8555a2ce230d952d7d163b3cab6152a155c03b7d2a8d3e6e1acf682bd7919"
    }
  ],
  "outputReferences": {
    "private_run_directory": "/home/aditya/experiments/world-model-r37-20261026/repo/outputs/r37-full/continuation-20261026/",
    "aggregate": "aggregate.json",
    "contract": "contract.json",
    "fitting_completion": "fitting-complete.json",
    "assessment_completion": "assessment-complete.json",
    "complete_artifact_hash_mapping": "/home/aditya/experiments/world-model-r37-20261026/verification/independent-result-review-v1/verification-receipt.json#artifact_sha256",
    "verified_source_hash_mapping": "/home/aditya/experiments/world-model-r37-20261026/verification/independent-result-review-v1/verification-receipt.json#source_sha256",
    "weight_manifest_canonical_sha256": "5fc1b3af0f2032b5091e68b11b1ca3ee54a00a023b0ed3d0ad7c2982ca3be45e",
    "budget_stage": "independent-result-review-v1",
    "reporting_scope": "Exact audited development aggregate only; not scientific success or permission for further experiments"
  }
}
```
