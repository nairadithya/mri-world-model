# R34. Methodology validity repairs

## Scope

This bounded pass starts from `067ed6c` on `exp/clean-data-rescues`. Its branch is `exp/methodology-validity` in `/home/adithya/projects/world-model-validity`. It repairs coordinate handling and adds structural diagnostics. It does not run a new forecasting experiment. No GPU training, cohort acquisition, publication, historical-output overwrite, or main-checkout edit occurred.

## Measured evidence

The two baseline LUMIERE overlays showed a hemisphere mismatch between array-resized masks and MRI. Retained pre-resize MRI intermediates reconstructed both final tensors exactly. Those checks are in `outputs/validity/baseline.json` and the two baseline PNGs.

The repaired audit sampled three additional visits per cohort with seed 42. LUMIERE legacy-versus-aligned union Dice was 0.0000 for Patient-061/week-012, 0.0181 for Patient-008/week-066, and 0.0291 for Patient-068/week-010. SAILOR Dice was 1.0000 for sub-03/ses-02, sub-20/ses-05, and sub-24/ses-05. Native compartment volumes matched the frozen manifest in all six samples. All available image modalities in each sampled visit shared the verified physical reference grid.

All six three-plane overlays were rendered and inspected. Alignment changed the LUMIERE hemisphere and placement. It left the sampled SAILOR observations unchanged. Patient-068/week-010 retained an apparent extracranial segmentation label. That visual observation is a reason for further QA. It is not an expert adjudication.

Synthetic cutoff checks used nonzero residual weights. Future image-feature, time-gap, and volume gradients were zero at every lesion forecast cutoff. Equal-gap historical token swaps left the legacy temporal state unchanged. They changed the opt-in elapsed-time state. Both temporal modes passed prefix-causality tests. Default and explicit legacy outputs were exactly equal. Both modes share state-dictionary keys. The checkpoint loader rejects a time-basis mismatch even when weight shapes match. Thirteen raw JEPA load paths in ten analysis/training consumers also apply the shared semantic guard. Three notebook evaluation cells apply it before loading their checkpoint payloads.

The online projector received no gradient under the default JEPA loss. Backbone LoRA and predictor parameters did receive gradients. No projector objective was added. A fixed projection is the current behavior, not a new finding that the projector was trained.

A real CPU smoke used official BrainIAC weights and four CT1 visits from Patient-067. It performed one optimizer step and one EMA update. Loss changed from 1.00568 to 0.92494. Gradients and same-space persistence, mean, and shuffled-target baselines were finite. Target standard deviation was 0.11569 and effective rank was 1.60829. A teacher LoRA parameter changed after EMA. These are runtime checks on a tiny sample, not an out-of-sample forecasting result. A subsequent training-mode forward/backward was finite and kept the EMA teacher in evaluation mode. A real aligned crop produced finite region tensors with shape [4, 4, 768] and contrasts with shape [4, 3, 768]. Real-versus-noise encoder relative L2 drift was 0.23537. The report fingerprints the configured BrainIAC bytes with SHA-256. It does not independently certify their download origin. The transient smoke weights were discarded.

The clinical sweep covered all 91 demographic vectors. It found zero nonfinite values and zero nonzero eventual-survival inputs.

## Target-discontinuity audit

The frozen usable manifest gives 23 unseen LUMIERE patients and 117 rows. It gives 27 SAILOR patients and 242 rows. Each compartment keeps the same immutable row identities. The audit reports physical volumes, absolute log-volume changes, empty endpoints, and per-patient summaries. Pooled and patient-uniform errors remain separate.

| Cohort | Compartment | One empty endpoint | Both empty | Pooled error share from one-empty transitions |
|---|---|---:|---:|---:|
| LUMIERE | necrotic/nonenhancing | 6/117 | 3/117 | 26.1% |
| LUMIERE | enhancing | 14/117 | 14/117 | 32.7% |
| LUMIERE | edema/FLAIR | 6/117 | 1/117 | 44.0% |
| SAILOR | necrotic/nonenhancing | 42/242 | 34/242 | 54.9% |
| SAILOR | enhancing | 22/242 | 15/242 | 42.3% |
| SAILOR | edema/FLAIR | 1/242 | 0/242 | 8.0% |

Inference. A small set of appearance/disappearance events contributes substantial persistence error. These events could reflect biology, treatment, label conventions, or segmentation failure. The audit cannot distinguish those explanations. Repeat segmentations or adjudication are needed to estimate measurement repeatability. No noise ceiling was estimated.

## Accepted design

The primary design supplied the shared physical-grid boundary. The alternate design supplied the requirement to preserve time semantics in artifact configuration. The larger loader factory and changes to the generic horizon-gap encoder were rejected.

`Grid` owns shape and voxel-to-world affine. `align_masks` regrids masks to the verified MRI pre-resize grid, then applies nearest-neighbor tensor resizing. Native volumes are computed before regridding. Identity-affine MRI requires a content-bound sidecar, an explicit exported geometry manifest, or a retained intermediate that reconstructs the final tensor. Missing or stale provenance fails closed. Independent modality grids are rejected rather than averaged silently.

New preprocessing writes per-image geometry sidecars. An aggregate manifest is portable across root changes and `.nii.gz` decompression. The lesion encoder, trainer, SAILOR transfer, and adaptation checks use `affine-aligned-lesion-v1`. Crop features use `lesion-crop-brainiac-affine-v2`. Old lesion checkpoints and transfer caches are rejected by the new geometry consumers. Historical artifacts are preserved for historical analysis. They cannot be relabeled as aligned features.

The temporal option is `model.temporal.time_basis`. Its default remains `previous_gap`. `elapsed` computes cumulative day coordinates inside `TemporalTransformer`. `TimeDeltaEncoding` retains its generic gap semantics, including use by horizon heads. New elapsed-mode forecasting requires training and matched evaluation. Equal timestamps still give equal coordinates. The option does not recover missing or ambiguous visit dates.

## Verification and rerun

Run from the validity worktree with the repository environment.

```bash
.venv/bin/python tests/test_geometry_validity.py
.venv/bin/python tests/test_temporal_validity.py
.venv/bin/python tests/test_temporal_checkpoint_consumers.py
.venv/bin/python tests/test_validity_audit.py
.venv/bin/python tests/test_harness.py
.venv/bin/python tests/test_clean_rescues.py
.venv/bin/python scripts/audit_mri_validity.py \
  --manifest /home/adithya/projects/world-model/outputs/anatomy_harmonization.json
.venv/bin/python scripts/smoke_mri_validity.py \
  --manifest /home/adithya/projects/world-model/outputs/anatomy_harmonization.json
```

The audit writes `outputs/validity/audit.json`, overlays, and `sample_geometry_manifest.json`. The sample manifest covers only the sampled LUMIERE images. Full portable training needs an aggregate manifest for every image it will load. Export one with `src.preprocessing.geometry.export_geometry_manifest` over the selected final MRI paths. Pass it through `--geometry-manifest` to lesion encode, train, and transfer commands. Do not interpret sample coverage as whole-cohort certification.

The tests cover synthetic flips, translated grids, physical volumes, missing/stale geometry, portable manifests, sidecars without retained intermediates, modality-grid conflicts, modality ordering, old artifact rejection, chronology, causal prefixes, legacy outputs, and checkpoint semantics. They exercise an evaluation entry point, distinguish pooled from patient-uniform estimands, preserve row identities, and reject vacuous empty-cohort QA. The 9 geometry, 7 temporal, 6 checkpoint-consumer, and 2 audit tests pass. Existing 25 harness checks and clean-rescue contracts pass. All 25 changed non-notebook Python files compile. The two changed notebook sources were regenerated with jupytext and checked for cell parity. Their evaluation guards are parsed by tests. No notebook cells were executed. Independent review was pending at this record's creation. The review addendum below records closure.

The smoke harness initially used an incorrect collator import, an incomplete modality dictionary, and an incorrect crop-call signature. These were harness mistakes. It now uses the repository interfaces and marks non-CT1 smoke images absent. These failures were not model results.

An isolated loader worker initially introduced a pytest dependency and reported torch unavailable. Parent acceptance rejected that validation. The shared environment imported torch 2.14 CPU successfully. The tests were rewritten with unittest and then executed. No new dependency or execution-mode fallback was used.

## Scientific gates

Coordinate handling is structurally repaired. Full spatial/anatomical validity remains INCONCLUSIVE. The audit samples geometry and does not prove template equivalence, correct registration, or accurate segmentation.

No learned forecasting gain is established. The corrected historical causal-lesion result remains relative MAE 1.00595 with a paired difference interval spanning zero on 117 rows. Its source is `outputs/clean-rescues/clean-causal-lesion/causal_lesion_eval.json` in the main checkout. Historical leaked gains remain invalid. Aligned features and elapsed temporal semantics need retraining before comparison.

Next gates are whole-cohort geometry QA, review of the largest zero-endpoint transitions and visible segmentation failures, a measurement-repeatability estimate, and a prespecified matched history-only experiment. Independent confirmation needs a genuinely unexamined cohort. These gates precede any renewed forecasting-success claim.

## Independent review closure

The first two-model review independently found the raw-checkpoint bypass and the modality-order defect. Parent reproductions confirmed both. The next fresh review found an omitted LoRA caller. The worker had incorrectly classified that caller as vision-only. Parent reproduced the bypass, guarded both LoRA loads, and extended the inventory to discover whole-world receivers across the harness. Matching and mismatched LoRA entry points now have behavioral tests. Notebook evaluation loads were guarded in the same unit.

The final fresh review returned VERIFIED for both incremental repair and the overall bounded pass. Its source snapshot is `32310c166d79e892be629e977c6ab5e33a126df4`. All numerical evidence was regenerated at that commit with `tracked_dirty: false`. The final reports fingerprint the frozen manifest, configured weights, script, and smoke MRI files. The review identified no remaining implementation blocker. Its scientific limitations remain open.

Review copies are under `outputs/validity/reviews/`. The final report is `closure.md`. Test logs use `test_*.log`; compilation evidence is `compile.log`. A compact evidence summary is `review_evidence_summary.json`. Generated evidence stays local and ignored. The source commit and rerun commands identify how to regenerate it.

The exact-zero MRI proxy is now quantified. Patient-068/week-010 has enhancing and necrotic/nonenhancing fractions of 0.71127 and 0.56872. This is overlap with zero-valued MRI, not a validated anatomical brain mask. It cannot distinguish segmentation error, registration error, a resection cavity, or excessive skull stripping. Mask headers remain trusted source metadata. The lesion pipeline supports the BRAINIAC 96-cubed input contract, not arbitrary tensor sizes. These are QA and scope limits, not forecasting evidence.

## Decision principles

Attack the Premise moved the work from optimization rescues to observation and chronology checks. Model the Domain replaced array-only mask handling with explicit physical grids. Build the Lever produced rerunnable audit commands. Encode Lessons in Structure added artifact and temporal-semantic rejection gates. Prove It Works required real MRI overlays and a real CPU forward/backward/EMA check. Explain the Number kept error concentration separate from a measurement-noise claim.
