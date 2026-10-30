# Independent R37 full-pilot implementation/design re-review — v2

October 26, 2026. This is a fresh independent review of the corrected full-pilot path and its integration, not result approval. The original `implementation-v1.md` is unchanged: SHA256 `bdb9a17a5ada777ee1481440fff655e156c62d34410f8030136571a1eba533de`. Its `CHANGES_REQUIRED` verdict remains applicable to its original code hashes.

APPROVED_FOR_R37_IMPLEMENTATION

**Scope of approval:** implementation/design of the thirteen current source entries in the final JSON, subject to successful production authority initialization, executed real geometry and reconstruction gates at the original fixed `atol=rtol=1e-4`, content-bound run-contract gates, and the unchanged operator allowance. This does not certify those unexecuted gates, approve results, authorize retries, extend the window, or authorize a different experiment. No remaining blocking design defect was found in the reviewed controlled execution path after the final B1 correction described below.

## Inherited decisions

- Keep the frozen original R35 seed-42 encoder, volume head and target normalizer; source52/internal13/encoder-unseen26 split; H/L/G/P/E/R arms; predictor seeds 42/43/44; twenty epochs; fixed optimization, selection and calibration; and the one-analysis stopping rule.
- Preserve patient-clean cutoff inputs. No future image, target mask, measured target volume, next gap or treatment enters a predictor. At most four consecutive usable history visits are retained. Primary outcomes remain automated developmental proxies in native `log1p(mm3)` coordinates.
- The continuation starts `2026-10-26T10:00:04+05:30` and ends `2026-10-28T10:00:04+05:30`, capped at eight additional GPU-hours and sixteen additional CPU-hours. Earlier expenditure remains separate, including unknown initial v1 cost. This review does not reset either allowance or clock.
- Science execution remains SSH `mosaic-gpu`, RTX 5090, in `/home/aditya/experiments/world-model-r37-20261026/`, using its dedicated `.venv/bin/python`. No hosted Kaggle execution, new data, SAILOR assessment, method sweep or R36 reopening follows.
- Implementation acceptance, executed reconstruction, computational completion and independent aggregate-result acceptance are distinct gates.

## Reviewed scope and method

I reviewed the guardrails, current R37 METHODS section, continuation, original review, final parent disposition, all thirteen source entries and their integration, and the six spatial test files listed below. Four source files changed since v1: the runner and `spatial_dynamics_data.py`, `spatial_dynamics_runtime.py`, `spatial_dynamics_report.py`. The other nine are byte-identical to the v1-reviewed sources; their retained fitting/model/data/control behavior was rechecked against the new boundaries.

The review used lightweight reads/hashes and read-only SSH metadata/receipt inspection. No Python, tests or scientific computation was run on the laptop. I did not independently rerun remote tests, initialize production authority, extract features, fit predictors, assess outcomes, acquire data or control scientific processes. No repository file was edited by this reviewer; only this requested new artifact was written.

I read `SOURCES` directly and used `sha256sum`, without importing the runner locally. All thirteen implementation files, all six test files, the reviewed protocol documents/disposition and the preserved v1 report matched local and remote SHA256 values. The final implementation receipt's source/test hashes match these actual bytes. The mirrored implementation log/manifest and documentation verification also match their remote copies.

## Diagnosis: B1–B3 disposition

### B1 — Actual MRI geometry and content: closed at implementation level

`validate_visit_geometry` now compares each present modality's actual shape/affine and canonical little-endian float32 content to the frozen inventory geometry evidence. Identity-affine tensors require verified pregrid reconstruction and the established resize convention. Native-affine files must match saved content/grid and have a consistent native pregrid; the previous native-affine early return is not substituted for validation. Each modality's physical pregrid must match the prepared visit pregrid, which also establishes modality agreement. Existing prepared-pair checks retain the longitudinal shared-grid requirement.

During this re-review I identified a remaining narrow boundary in the first correction: its initial raw digest was taken after the geometry read. I paused finalization rather than assume immutable staging. The parent then supplied a stable correction and new receipts. In the **final reviewed bytes**, the initial digest precedes `nib.load`, a second digest must match after geometry verification, and extraction checks that same initial digest again after encoding. The new regression mutates the on-disk affine during geometry reading while returning the old loaded header/array, verifies unchanged voxel values, and requires the post-read digest rejection. No OS-enforced immutability assumption is used to close this finding.

Per-file raw hashes, saved geometry-record hashes and prepared-pregrid hashes are retained privately; their aggregate digest is bound into the cache and contract. Contract verification checks the receipt/content association and current raw MRI bytes.

The separate original-cache gate still checks **every original visit's pooled feature and decoded native log-volume**, using the runtime transform plus float16 storage round-trip, at fixed absolute and relative tolerances of `1e-4`. It remains unexecuted. Synthetic geometry tests do not certify the real images, reconstructability, anatomical correspondence or biological validity.

### B2 — Authoritative budget/retry state: closed at implementation level

Production accepts only the fixed authoritative ledger path. Start, deadline and ceilings are pinned, and the 900-second setup charge cannot be silently changed. First initialization requires the exact independently reviewed legacy snapshot hash:

`cb717e6205a4af7e912e18854f19892c0a3f9169206cb48a22163aed3386c695`.

Inspection confirms that initialization retains the entire decoded legacy ledger and adds authority identity, rather than constructing an empty stage list. It requires the existing completed inventory stage. Thus the reviewed 0.844038823 CPU-second inventory charge and 900-second setup charge are retained, along with the separate-prior-expenditure statement. The raw snapshot hash also prevents substituting a different initial history. The original unknown v1 expenditure is not converted to zero or absorbed into a claim that old ceilings were met.

The authority anchor, chained journal and current-ledger digest reject an alternate ledger, changed limits, a ledger reset inconsistent with its journal, and a contract whose locked journal tip is not an ancestor. A nonblocking exclusive `flock` spans a whole stage and its ledger writes. Failed/running stages and repeated stage names block further execution; selecting another output directory does not reset this production authority. Inconsistent interrupted initialization/journal state fails closed rather than silently starting again.

**Production migration remains unexecuted.** Read-only remote inspection found the legacy ledger still at the exact hash above and no authority anchor or journal yet present. This establishes readiness for the reviewed initialization check, not successful migration. Preserve an actual initialization receipt before scientific execution; any mismatch or interruption must stop for explicit handling, not be repaired by resetting spending or weakening the pinned hash.

The mechanism is an integrity boundary for the controlled private execution path, not a cryptographic anti-rollback service against coordinated manual replacement/restoration of authority, ledger and journal files. Such restoration/recovery is not approved by this verdict. Whole-CUDA-stage wall charging, CPU limits, the preflight cap and continuation deadline remain in force.

### B3 — Assessment association, completeness and publication: closed at implementation level

Evaluation/reporting now validate fitting completion against the contract, exact required seed identities and the complete expected weight-artifact manifest, then verify artifact hashes. Assessment identity binds the contract, fitting-completion digest, artifact-manifest digest and canonical primary/single/nuisance cohort rows into both prediction payload and marker. PT and NPZ prediction digests are checked before reporting.

Expected cohorts are derived from the locked cache, whose adapter construction preserves patient/chronological cutoff order. Exact canonical row comparison requires the complete primary and single-visit sets and one earliest eligible nuisance cutoff per primary patient. All six cohort arms, ordered seeds 42/43/44 for H/L/G, H/L nuisance arms and all three nuisance conditions are required. Used score/mean/interval arrays must have finite `[N,3]` shape, with non-reversed bounds. A missing/substituted cohort, stale run/fitting identity or incomplete nuisance seed set blocks grading. Nuisance arrays, seed labels and cutoff measurements are also retained privately for replay.

`analyze` now stages `status=staged_not_graded`; it no longer writes an apparently completed canonical aggregate. The CLI publishes only after the authoritative report reservation completes, and `publish_aggregate` rechecks completed-stage state and final CPU/GPU/deadline limits. Publication remains `completed_not_scientifically_reviewed` with `reporting_approval=None`. Resource-failed reporting cannot publish a completed canonical aggregate through this path. Staged mechanical classifications are not scientific claims and must not be promoted separately.

## Drift / contradiction check

No scientific target, model capacity, initialization/order policy, fitting matrix, optimization setting, contrast, margin or stopping rule changed in these corrections.

- **Leakage/targets:** Source-only unique-visit normalization, separate target access, original native compartment order and frozen inverse target normalization remain consistent. No new future-dependent input or target-mask weighting was introduced.
- **H/L/G:** Identical initial parameters, saved patient-uniform replacement orders, unweighted sampled losses, optimizers and update counts remain matched. L removes only earlier standardized MRI maps; G pools only earlier maps; current spatial processing and cutoff covariates remain intact.
- **Uncertainty/calibration:** Shared spatial channel noise is propagated analytically without voxel-count variance reduction. Feature and target normalization are inverted. Calibration multiplies intrinsic uncertainty before the fixed 0.05 observation floor. Gaussian and empirical-mixture scoring remain unchanged.
- **Ridge:** Equal-patient mean-voxel residual moments, fold-local centering/scaling, unpenalized intercept, grouped source CV, fold-training CV noise and selected-alpha OOF final RMS remain intact. Validation selects only the permitted scalar uncertainty multiplier.
- **Aggregation/gates:** Seed scores, not predictions, are averaged before patient bootstrap. Patient-uniform aggregation, 10,000 draws/seed 2026, five adjusted H contrasts, the 5% H-versus-L threshold, MAE/coverage/nuisance requirements and separate H-versus-G spatial-detail claim remain unchanged. Secondary L comparisons do not rescue the primary hypothesis. Incomplete execution is not a partial-arm success.
- **Historical timing:** Positive week-bin differences times seven days remain an explicitly approximate history covariate; suffixes only order visits. Invalid retained histories are excluded without bridging unusable visits, and no future gap enters prediction. This is acceptable for the amended next-observed development pilot, not evidence of verified exact-date or fixed-horizon dynamics.
- **Primary E after known ablation:** `gaussian_observation` remains a coherent common-floor choice, but is not blinded or untouched prespecification. Preserve the already observed discrete/noise ablation descriptively. Exactly five fixed H comparisons with two-sided 99% adjusted intervals remains the declared analysis family; this is not pristine confirmatory inference, and later E switching or assessment-selected contrasts are not approved.

## Test/receipt scope

The final receipts supersede the initial v4 implementation receipts supplied at the start of re-review:

- `implementation-tests-v5.log`: **62 spatial tests PASS**, including the new during-geometry-read header mutation regression; command records the dedicated remote interpreter and exit status 0.
- `implementation-checks-v5.json`: binds all thirteen implementation files, six test files, the test-log digest, Python 3.12.14 interpreter path, and successful compilation list.
- `documentation-tests-v4.log`: **24 tests PASS**.
- `documentation-verify-v4.json`: binds documentation/source verification, coverage-render receipt, actual CLI help, disposition and preserved original review. These checks are not scientific acceptance or semantic-completeness proof.

The synthetic suite covers the principal mathematical/rejection cases and B1–B3 boundaries. It does not execute production migration, actual image reconstruction, the full fitting matrix, real nuisance assessment or independent aggregate-result replay. Test PASS and compilation are not substitutes for those remaining gates.

### Reviewed document and test hashes

| File | SHA256 |
|---|---|
| `info/research_guardrails.md` | `6e710be8c33b05d77bf8f13b415b64028fcc13dc03d71defb304e6a7631c78f4` |
| `info/METHODS.md` (whole file; R37 section reviewed) | `5dbfeed0697b1bf71079251ae5764471755b7882ef4eb4bd00ab750d25ea1026` |
| `info/documentation_sources/updates/2026-10-26/r37-continuation.md` | `f34b0d9b669bde950c8e9e5d0ed8aa510cabfd83df8c3bd4cab0afe940e3504b` |
| `info/documentation_sources/updates/2026-10-26/reviews/implementation-dispositions.md` | `996f5537547d9bf9ee632197b1b0b8f7981d4371fb55bbf2b381ee44d19e57d8` |
| `tests/test_spatial_dynamics.py` | `48bbf085ece902ac8453207c608f3fd07a1acae98d36e248259aac1df464c048` |
| `tests/test_spatial_dynamics_baselines.py` | `fb599ffb49cbc73b09ccabfd237339fea56d4d58fff95c534c0f05285ba62bd5` |
| `tests/test_spatial_dynamics_full.py` | `f78df289daa0d69e5f69784e4951f05971966bd708d9c1ac726dcd6b805efa2d` |
| `tests/test_spatial_dynamics_guards.py` | `c34ccac36c196728737678ae01a842a9716eff5fbae9c5c2bfb12e8de3e4f17c` |
| `tests/test_spatial_dynamics_protocol.py` | `07e7433ebc2523da27fcf0931f98b2f13fe6e1b7e8d6d7ec6fd66906d835a7a0` |
| `tests/test_spatial_noise_ablation.py` | `5798f539785cfb702ca0d6b8ef412d0ee5b109f73b2099c695b8cb7ef482e9eb` |

### Final remote receipt hashes

Paths below are relative to the dedicated workspace's `receipts/` directory. The first, second and fourth were also checked against their local mirrored copies.

| Receipt | SHA256 |
|---|---|
| `implementation-tests-v5.log` | `9b1149db0171f641c18289086115b557ca22aefea77b3461dce2b4e04e562a7f` |
| `implementation-checks-v5.json` | `011224511d72c67feb46772746a5e20a5d6f6840a62c1b7ff135ce92edfedeb8` |
| `documentation-tests-v4.log` | `684b2ca3fa2a68eda3f03da19d18ef3d1f1cac4b1ba0d157879d8c2a310e25e7` |
| `documentation-verify-v4.json` | `10c0d0a818fc3e0b604779d4c1bb73730e9aaab839283f37b612a84b7b38279b` |
| `source-verify-v4.log` | `f7daec116c4fd4abbe6c3a5cd32ec8177621b869aa56b93bf9adc8c2440fbd5f` |
| `coverage-render-v4.log` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `cli-help-v5.txt` | `accb7a73d273489a89c3686969b653d9dd71c700eca369ccda81b832036ac7ab` |

## Recommendation, risks and remaining gates

Proceed only through the already authorized gated execution path, within the unchanged remaining allowance:

1. Successfully initialize the authoritative production state from the exact reviewed legacy snapshot, preserving expenditure, and retain its receipt. Do not rerun the already completed inventory stage or reset authority/history.
2. Execute real geometry/content/stability checks and reconstruct every original cached pooled feature and decoded volume at `atol=rtol=1e-4`. Failure blocks fitting; this approval is not a retry or tolerance-relaxation authorization.
3. Lock and verify the content-bound run contract and exact implementation-review artifact before fitting. Keep code, inputs, normalization, rows/splits, geometry evidence, decoder, environment and budget identity bound.
4. Complete the fixed arms/seeds or retain an incomplete/inconclusive outcome without partial grading. Preserve weights, optimizer states, selections, orders, ridge/OOF artifacts, private predictions and resource receipts.
5. Obtain fresh independent aggregate-result review before any scientific success claim. No success is approved in advance.

No real reconstruction/full assessment has run in the reviewed status. The supplied inventory counts are eligibility evidence only. Automated targets, reused development patients, encoder-exposed internal validation, approximate historical timing, unknown future scheduling/interventions, restricted covariance and frozen-decoder bias remain interpretation limits. Passing these gates would not establish expert biological validity, independent generalization, fixed-day prediction, multistep dynamics or treatment effects.

## Need from main agent / suggested execution prompt

No unresolved design decision remains for this reviewed implementation. The parent owns recording the actual initialization/reconstruction/run-contract receipts and enforcing the existing authorization. No worker handoff or additional review tree is proposed by this read-only reviewer. This review did not execute any scientific stage or authorize exception recovery.

```json
{
  "verdict": "APPROVED_FOR_R37_IMPLEMENTATION",
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
  }
}
```
