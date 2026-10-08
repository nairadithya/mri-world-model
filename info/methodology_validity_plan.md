# Methodology validity run

## Scope

This run starts from local commit `067ed6c` on `exp/clean-data-rescues`. It implements and verifies the cheap validity checks from the methodology audit. It does not launch GPU training, acquire cohorts, publish code, or overwrite historical artifacts. The main checkout and its existing edits remain untouched.

## Done predicate

A deterministic audit reproduces the legacy spatial and temporal failures, verifies accepted repairs on synthetic fixtures and local MRI, and reports target discontinuities with immutable row identities. Every result is VERIFIED, NOT VERIFIED, or INCONCLUSIVE. A repaired implementation does not establish better forecasting.

## Data shape

An image tensor has a pre-resize physical grid, an output tensor grid, and provenance linking them. A grid contains a three-dimensional shape and voxel-to-world affine. A mask retains its own physical grid until nearest-neighbor resampling onto the image grid. LUMIERE identity-affine tensors require retained pre-resize geometry or an explicit geometry manifest. SAILOR images and masks retain their physical affines. Missing geometry fails rather than falling back to array resizing.

A forecast row retains patient, source visit, target visit, eligibility, input cutoff, and segmentation source. The historical manifest is read without modification. The causal lesion contract remains `history-only-lesion-v2`.

## Designed workflow

1. Capture baseline behavior and add rerunnable validity tests before repairs.
2. Compare two independent geometry and temporal-interface designs. Select the smallest design that satisfies the done predicate.
3. Implement one shared image-mask geometry boundary. Migrate active lesion encoding and training callers. Version new feature and checkpoint contracts so old features cannot masquerade as aligned features.
4. Verify synthetic axis flips, translated grids, missing provenance, real image-mask overlays, and preservation of physical mask measurements.
5. Add an explicit elapsed-time temporal option while retaining legacy previous-gap behavior for old checkpoints. Verify causal prefixes, equal-gap chronology, and state-dictionary compatibility.
6. Record projector gradient routing without changing the objective. Check fixed-target semantics rather than inventing a training loss.
7. Produce a target audit by cohort, compartment, patient, and empty/nonempty transition. Keep pooled and patient-uniform estimands separate.
8. Run compilation, existing contract tests, one real CPU training smoke when feasible, and an independent code and decision-trail review.
9. Record results and remaining scientific gates in an append-only research note. Do not claim a forecast gain without retraining and matched evaluation.

## Throughput checkpoint

The parent owns the shared geometry boundary, caller migration, audit tools, and acceptance. Isolated workers own the temporal option and the later checkpoint-consumer repair. Each unit ends in executable tests before integration. Independent reviewers read pinned commit-range patches rather than a mutable working-tree diff. Source changes stop during final review. GPU work and full-cohort encoding remain outside this pass.

## Design selection

The primary design supplied the shared grid and mask-alignment boundary. The alternate supplied configuration-bound temporal semantics. The loader factory and generic gap-encoder rewrite were rejected. Existing MRI loading remains in place. Preprocessing now exports geometry sidecars. New caches and learned lesion checkpoints carry an aligned-geometry contract. The decision TSV records implementation, verification, and unresolved scientific gates.

## Gates after this run

Measurement repeatability needs alternate segmentations or adjudication. An incremental MRI forecasting claim needs matched history-only baselines and new training. Confirmation needs a genuinely unexamined cohort. None of those gates is satisfied by structural correctness tests.

## Tradeoffs

Retained local preprocessing intermediates can establish geometry for this run. Portable training needs an exported geometry manifest because Kaggle final tensors do not preserve their original affines. Affine-aware regridding fixes voxel orientation and field of view. It does not prove equivalence between different MNI templates or eliminate registration error. Visual QA remains required.

## Decision trail

The append-only trail is `info/methodology_validity_decisions.tsv`. Large generated arrays and screenshots stay under ignored `outputs/validity/`.
