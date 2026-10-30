# Independent R37 full-pilot implementation/design review

Date: October 26, 2026. Scope: the current R37 protocol and continuation, research guardrails, all 13 entries of `scripts.run_spatial_dynamics.SOURCES`, and the five spatial test files listed below. This is a fresh review of the full-pilot path, not an extension of the earlier E-only implementation or result approvals.

CHANGES_REQUIRED

The statistical/model implementation is substantially consistent with the declared pilot, but the current bytes have three blocking execution/provenance gaps. No encoder extraction, fitting, or full assessment is approved by this verdict. The reconstruction gate has **not** been executed and is **not** certified as passed.

## Inherited decisions

- Test next-observed anatomical prediction using the frozen original R35 seed-42 encoder, decoder and target normalizer; no new encoder objective, family search, future gap input, SAILOR assessment, data acquisition or rescue sweep.
- Preserve source52/internal13/encoder-unseen26 patient separation, H/L/G matched capacity, initialization, sampled orders and update counts, three predictor seeds, 20 epochs, and P/E/R controls. Primary outcomes are development-only automated native-volume proxies in `log1p(mm3)`.
- The operator's additional allowance begins `2026-10-26T10:00:04+05:30` and ends `2026-10-28T10:00:04+05:30`: at most eight additional GPU-hours and sixteen additional CPU-hours, with prior expenditure separate and initial v1 expenditure unknown, not zero. Science execution is confined to the dedicated environment on `mosaic-gpu` RTX 5090.
- Require fresh independent implementation review, executed rejection/reconstruction gates, a content-bound run contract, complete arms/seeds, and later independent aggregate-result review. Computational completion is not scientific acceptance.

## Diagnosis and blocking findings

### B1 — Reconstructed MRI geometry is not checked against the saved geometry evidence

**Locations:** `src/harness/analysis/spatial_dynamics_data.py:73–87,105–130`; `scripts/run_spatial_noise_ablation.py:build_rows/load_inputs`; `src/preprocessing/geometry.py:image_grid`.

`load_image()` finds current MRI files and applies the array-only runtime transform. Extraction compares pooled features and decoded volumes and then records *new* hashes of those current files. The adapter verifies stored inventory/prepared identities and equal stored adjacent-pair `pregrid` records, but the new path never resolves the actual reconstructed MRI's geometry against the saved inventory geometry manifest and prepared pregrid.

A NIfTI affine/header-only change leaves the loaded voxel array, pooled-feature gate and decoded-volume gate unchanged, yet can invalidate the image-to-target or longitudinal physical-grid contract. Hashing the changed file afterward binds the wrong input; it does not reject it. This is a missing rejection check, **not** a finding that an actual participant image is currently misregistered.

**Required correction:** Before accepting each reconstructed visit, validate every present modality against the existing content-bound geometry evidence, including current grid/header and verified pregrid; check agreement with the prepared visit's pregrid and between modalities. Cover identity-affine processed tensors and native-affine files rather than relying on `image_grid()`'s native-affine early return alone. Retain a bounded geometry-check receipt bound into the feature cache/run contract. This should reuse existing evidence, not launch new registration or target validation.

**Required rejection coverage:** An altered affine with unchanged voxel values must fail; stale manifest/content evidence, incompatible modality grids and mismatch with the prepared pregrid must fail. The independent pooled-feature and decoded-volume reconstruction checks must still apply to **every original cached visit** at the fixed `atol=rtol=1e-4`, with no tolerance relaxation after failure.

### B2 — The enforced resource/retry ledger is not the ledger bound by the contract

**Locations:** `scripts/run_spatial_dynamics.py:60–81,84–102,194–228`; `src/harness/analysis/spatial_dynamics_runtime.py:21–50`.

The contract records some budget fields, but `verify_contract()` receives no ledger and never compares the execution ledger to them. Every stage accepts an arbitrary `--ledger`; `budget_stage()` trusts its limits, deadline and stage history. A fresh ledger can therefore forget failed/running stages and prior spending, or supply a later deadline/larger allowance, while the same code/cache/contract digest continues to verify. Separate output directories do not solve the cumulative continuation-budget problem. The ledger also has no exclusive lock around its read/check/write sequence, so concurrent launches can lose stage reservations/history.

The same-ledger failure preservation, whole-CUDA-stage wall charging, CPU accounting, deadline signals and preflight cap are useful, but do not close this cross-ledger/concurrent-launch bypass.

**Required correction:** Establish one authoritative continuation-ledger identity/location for this authorization and bind it to the run. Validate the fixed authorization fields/ceilings and preserve the consumed inventory/setup history. Verify that execution uses that ledger and that its history extends the locked state, without resetting failed/running stages or spending. Use exclusive locking/reservation to prevent overlapping stages/ledger updates. A new output or copied ledger must not create another allowance or retry opportunity. Any exceptional recovery must require explicit approval rather than an automatic fresh ledger. Preserve the original expenditure ledger separately.

**Required rejection coverage:** Different/fresh ledger, reset history, changed limits/deadline, concurrent reservation, and attempted execution after a failed/running stage. Keep these tests synthetic and remote; no fitting is needed.

### B3 — Saved assessment predictions are not bound back to their run/fitted artifacts

**Locations:** `scripts/run_spatial_dynamics.py:176–191,238–261`; `src/harness/analysis/spatial_dynamics_report.py:122–133,143–166`.

Fitting completion correctly records a contract digest and weight hashes, and evaluation verifies them. However, neither the saved prediction payload nor `assessment-complete.json` carries the contract digest or fitting-completion identity. The report stage checks only that a prediction file matches its accompanying assessment hash, then labels the analysis with the currently supplied contract digest. It does not verify that the saved rows are the contract's full primary/single/nuisance cohorts. Thus an internally consistent stale/copied assessment bundle and marker can be graded and relabeled under another contract. Hash agreement establishes file integrity, not the missing provenance relationship.

Primary arm/seed counts are checked, but nuisance bundles are not required to contain all three matched seeds or the complete earliest-cutoff patient set; a shortened nuisance bundle can still enter the pass criteria.

**Required correction:** Put the contract digest, fitting-completion/artifact-manifest digest, and canonical cohort/row bindings in assessment outputs and verify them before reporting. Derive/validate the expected eligible rows from the locked cache; require all primary and companion rows and exactly the matched three nuisance seeds for H/L at each condition, on the prescribed one earliest eligible cutoff per primary patient. Keep private row identities out of public reports.

**Required rejection coverage:** Assessment from a different contract/fitting receipt, omitted or substituted cohort rows, and missing nuisance seeds/patients must block grading, not produce a capability classification. Also avoid publishing an apparently completed aggregate before the report stage's resource/completion checks succeed: `analyze()` currently writes `status=completed` before the caller adds provenance and before the final budget guard. A failed report may retain diagnostic/staged artifacts, but must have an unambiguous incomplete/not-graded state.

## Drift / contradiction check and mathematical assessment

No broad scientific pivot is recommended. The following choices are consistent with the continuation and do not themselves require changes:

- **Timing approximation:** The adapter uses ordered original chronology, does not bridge unusable visits, excludes invalid retained historical week-bin gaps, resets the first retained gap, and does not condition on the future gap. Same-week target adjacency can remain eligible; suffixes determine order, not elapsed days. This is defensible for the explicitly amended *next-observed* developmental pilot, but is not verified scan-date timing and cannot establish exact-time/irregular-gap forecasting. The earlier E-only timing approval is not being imported as full-path approval.
- **Primary E after the known ablation:** Locking `gaussian_observation` now has a coherent common-floor rationale. The empirical mean remains fixed under calibration; the 0.05 observation standard deviation is not multiplied by k. Keep the prior discrete/noise ablation and its outcomes descriptive and discoverable. This choice is not blinded/untouched prespecification. Exactly five primary H contrasts with two-sided 99% Bonferroni-adjusted bootstrap intervals remains consistent with the declared fixed family; do not later switch E, add an assessment-selected primary comparison, or present these reused-development intervals as pristine confirmatory inference.
- **Leakage and targets:** Predictor inputs resolve only cutoff history maps, availability, completed gaps and masks. Future map/volume targets and measured cutoff volumes are separate; cutoff measured volumes are used for reporting subsets, not as predictor inputs. Source-only feature normalization uses unique source visits. Native compartment order is `necrotic_non_enhancing`, `enhancing`, `edema_flair`; inverse target normalization is applied before anatomical scoring. I found no future-image, target-mask, treatment or next-gap input in the reviewed forward/fitting calls. B1 remains necessary to establish the physical input/target relationship for reconstruction.
- **Matched H/L/G:** One saved patient-uniform replacement-sampling order per seed/epoch is reused across arms. An unweighted sampled loss avoids double patient weighting. Initial parameter values are copied identically; optimizer, update counts, epoch selection and calibration rules match. L zeros only earlier standardized maps; G pools only earlier maps; the current spatial pathway and cutoff-available covariates remain unchanged. No stochastic layer creates an unpaired training-order difference.
- **Gaussian propagation/calibration:** Independent channel noise shared across space correctly produces a sum of squared affine channel contributions, with no division by voxel count. Source feature normalization and target normalization are inverted; signed decoder weights contribute squared variances. The fixed observation floor is added after scalar intrinsic-scale calibration. Gaussian CRPS, persistence RMS and the weighted Gaussian-mixture CRPS match the declared marginal laws.
- **Ridge weighting/CV:** Per-patient moments implement equal-patient, mean-example, mean-voxel squared residual loss. Fold-local design centering/scaling and an unpenalized intercept are implemented in float64. Patient-grouped folds select alpha on decoded CRPS using fold-training residual noise, not held-fold fitted responses; selected-alpha source OOF residuals set the final RMS. Validation selects only k. Precomputing assessment cutoff designs is unnecessary but does not enter the reviewed coefficient/noise/alpha selection calculations.
- **Aggregation/gates:** Scores, not seed predictions, are averaged. Examples are averaged within patient and patients weighted equally. Bootstrap resamples patients after seed averaging, with 10,000 draws and seed 2026. The five H contrasts, 5% H-versus-L improvement, MAE upper-bound margins, per-compartment 80% coverage, nuisance point margins and secondary L-versus-P/E rules are implemented consistently. H-versus-G is required for the spatial-detail claim, not silently required for history benefit. All six primary arms/three neural seeds are required, and adverse contrasts and the one-analysis stopping rule are retained. B2/B3 are needed so incomplete execution cannot bypass those protections through unrelated receipts.
- **Artifact retention:** The design saves selected and final neural weights, optimizer-bearing final checkpoints, sampling orders, normalization/cache, ridge coefficients/folds/OOF predictions, controls and private prediction arrays. These are useful and largely sufficient for replay; B3 addresses the missing end-to-end association, not an instruction to fit additional models.

## Reviewed evidence, hashes and remaining limitations

I performed source/document reads, lightweight hashes and read-only SSH receipt inspection only. I did not run Python, tests or scientific computation on the laptop, import the runner locally, run fitting, acquire data, use web research, or modify/control research processes. I did not independently rerun the remote tests. The 13 code files, the three documents below and five test files had matching local/remote SHA256 values when checked. The 13-entry mapping was obtained by reading `SOURCES` and using `sha256sum`, not importing it.

The supplied remote `implementation-tests-v3.log` reports **50 tests, all OK**, in the five spatial test files below. Scope includes synthetic forward/backward and frozen-decoder gradients, uncertainty formulas, H/L/G interventions, cutoff/chronology rejection, empirical-control weighting/noise, small ridge algebra, patient bootstrap, incomplete primary arms and same-ledger retry failure. It does **not** test actual image reconstruction, current MRI/manifest geometry rejection, the full fitting/assessment pipeline, cross-ledger budget binding or assessment-to-contract binding. It is not a result receipt. Current source hashes were compared with the remote tree, but the plain test log itself does not embed a complete code/interpreter hash manifest.

The remote documentation log reports **24 tests, all OK**; `documentation-verify-v2.json` reports PASS for its recorded integrity/link/coverage checks and expressly excludes scientific acceptance/semantic completeness. CLI help is present. Compilation success is author-reported; I did not find a separate hash-bound compilation receipt in the listed receipt directory and do not independently certify it.

| Reviewed document/test | SHA256 |
|---|---|
| `info/research_guardrails.md` | `6e710be8c33b05d77bf8f13b415b64028fcc13dc03d71defb304e6a7631c78f4` |
| `info/METHODS.md` (whole file; R37 section reviewed) | `5dbfeed0697b1bf71079251ae5764471755b7882ef4eb4bd00ab750d25ea1026` |
| `info/documentation_sources/updates/2026-10-26/r37-continuation.md` | `f34b0d9b669bde950c8e9e5d0ed8aa510cabfd83df8c3bd4cab0afe940e3504b` |
| `tests/test_spatial_dynamics_full.py` | `b4cb0bfebd3f4014519a0da419908d70ebd50c6d7b0fa545e9917122e8907b93` |
| `tests/test_spatial_dynamics.py` | `48bbf085ece902ac8453207c608f3fd07a1acae98d36e248259aac1df464c048` |
| `tests/test_spatial_dynamics_baselines.py` | `fb599ffb49cbc73b09ccabfd237339fea56d4d58fff95c534c0f05285ba62bd5` |
| `tests/test_spatial_dynamics_protocol.py` | `07e7433ebc2523da27fcf0931f98b2f13fe6e1b7e8d6d7ec6fd66906d835a7a0` |
| `tests/test_spatial_noise_ablation.py` | `5798f539785cfb702ca0d6b8ef412d0ee5b109f73b2099c695b8cb7ef482e9eb` |

Receipt hashes below refer to `/home/aditya/experiments/world-model-r37-20261026/receipts/` as inspected remotely; v3/v2 receipts need not yet be present in the local receipt copy.

| Receipt | SHA256 |
|---|---|
| `implementation-tests-v3.log` | `309996222b4e1ca69baac35dfb30ba66d12764fa3fdabe981d7975812da2eb42` |
| `documentation-tests.log` | `9674e237305645c240577133f0a417f845cbade29ad20d5a4d4966cc11186efb` |
| `documentation-verify-v2.json` | `f7daec116c4fd4abbe6c3a5cd32ec8177621b869aa56b93bf9adc8c2440fbd5f` |
| `cli-help.txt` | `accb7a73d273489a89c3686969b653d9dd71c700eca369ccda81b832036ac7ab` |
| `continuation_budget.json` (review-time snapshot only) | `cb717e6205a4af7e912e18854f19892c0a3f9169206cb48a22163aed3386c695` |

The inventory status supplied for review is 599 prepared visits, 423 eligible pairs, 262 source/60 validation/101 assessment pairs, and 18 history-eligible assessment patients. This is eligibility evidence, not an executed reconstruction or prediction result. The ledger snapshot contains completed inventory plus a conservative 900-second setup charge; it does not establish completion of the remaining stages or measure away the unknown original v1 expenditure.

## Recommendation / need from main agent

Correct B1–B3 narrowly, retain the same scientific question and stopping rule, execute the added synthetic rejection tests on the dedicated remote Python, and obtain fresh independent review of the resulting stable bytes. Do not use this `CHANGES_REQUIRED` artifact as an implementation-approval gate. No new scientific-target or method-family decision is requested here.

Any subsequent `APPROVED_FOR_R37_IMPLEMENTATION` must be explicitly limited to implementation/design, conditional on successful executed reconstruction at the fixed tolerances, geometry/rejection checks, the content-bound run contract, resource availability and the unchanged authorization. It must not approve results in advance. All primary outcomes remain automated developmental proxies; unknown scheduling/interventions, historical week-bin timing, restricted covariance, frozen-decoder bias, reused patients and lack of independent biological verification remain limits even after successful execution.

## Suggested execution prompt

No worker handoff or additional review tree is proposed by this read-only reviewer. The parent implementation author owns the bounded B1–B3 corrections and any follow-up orchestration. No executor action was taken during this review beyond writing this requested artifact.

```json
{
  "verdict": "CHANGES_REQUIRED",
  "reviewed_code": {
    "scripts/run_spatial_dynamics.py": "7efa91d63ef9939810e6df3cc77ce6cf82577d34701791d1f14aa0647e907aeb",
    "scripts/run_spatial_noise_ablation.py": "251cb615e460a3f7d47d727c38223dd703bb5838c67956f89a0945809183d6a3",
    "src/model/spatial_dynamics.py": "c7c1b87d24de8b083569d29b5021cb994321cf068fd0f7dd1021df9fe103952a",
    "src/model/representation.py": "d73125b5f05dca864cb308773bcd1924a7f40552371374753d94c147bff547f6",
    "src/data/representation.py": "65537546257c70ce978fde9b254b7ef75c49b4797a8c279da3542c75f6dcee6a",
    "src/preprocessing/transforms.py": "c910a8ad41d62534fa3b9fe2f421be5260d91baed806ed679e03177dbb3a6bc1",
    "src/preprocessing/geometry.py": "f09b91bf2ec829297c64efb6606726a3206c98d7dacff4c2cc6db28c5d61fb2d",
    "src/harness/analysis/spatial_dynamics_data.py": "807d6ad6a709d7d24aed43bc5c615deaab51a5ea6d5b16c50024a94b31f8c45b",
    "src/harness/analysis/spatial_dynamics_fit.py": "7f738e076c592d6bf045ccb1a835f10f81c16525ab32b7a37287529c7f5e629a",
    "src/harness/analysis/spatial_dynamics_report.py": "33cff462535462b9bc71b8fa9108a8421dba890c9b84f7b6dbea87a57504fc39",
    "src/harness/analysis/spatial_dynamics_runtime.py": "4772eaeba5d2a976f7b329a58b6da644ae38d534858c2d5a6691fa659f341793",
    "src/harness/analysis/spatial_dynamics_baselines.py": "acbe7ddf99b3c39236cb1a99995c621ca08ad9e4bc5003421deb312ca70b8f37",
    "src/harness/analysis/spatial_dynamics_protocol.py": "9e29b9efc90699dde135b15d5705ef06bac1d5d473496daa2a3ead0edc358d19"
  }
}
```
