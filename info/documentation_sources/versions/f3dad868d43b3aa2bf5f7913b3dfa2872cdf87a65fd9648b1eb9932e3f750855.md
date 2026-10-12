# Action-conditioned representation experiment brief

Status: proposed, not yet implemented. This is a new synthetic-mechanism experiment, separate from the closed P2 lesion-forecast track.

## Question and capability tested

Can a representation retain physical anatomical state, predict controlled physical changes from explicit actions, and distinguish those changes from acquisition nuisance? This tests controlled representations and dynamics on synthetic transformations of real MRI. It does not test whether synthetic actions reproduce glioma biology.

## Current candidate

Action-conditioned temporal JEPA: encode a source scan and predict the target scan representation conditioned on a declared action. Inputs are source scan plus action only. A predictor must not observe the target scan, target-derived labels, or future nuisance parameters.

## Materially different approach

A materially different per-scan encoder with direct state readouts. Encode each scan independently, read out its measured physical state, and predict target state from source state plus action. It has no temporal latent-prediction objective. Freeze the data generation, encoder initialization/compute budget, training examples, and evaluation splits across methods where technically possible.

## Simple baselines

Persistence predicts no state change. Current-scan-only prediction estimates the target state from the source scan without history or action. Also report a source-state-plus-action direct predictor where this is the natural low-capacity control. No learned baseline may access target information at inference.

## Rationale and assumptions

Synthetic paired views permit exact action labels and controlled nuisance-only pairs, which real longitudinal MRI lacks at adequate scale. A real scan is the content source; transforms create known interventions and nuisance. The test assumes transformations preserve a meaningful physical coordinate system and that measured readout targets change consistently with transformations. Synthetic transformations are proxies, not biological supervision.

## Matched comparisons

Use the same source scans, generated action pairs, train/validation/test examples, patient split, optimization budget, and physical-state targets for all learned methods. Hold out patients and transformation families. Keep a separate nuisance-only test set with intensity and resolution changes that preserve physical anatomy. Include identity/no-op and physically declared changes. Do not allow near-duplicate scans of held-out patients into training.

## Information cutoff and patient-clean split

Partition patients before generating any pairs. All views and transformations from a patient remain in one partition. The source scan and declared action are the only model inputs. Split transformation families independently so test performance measures generalization to unseen change types. Fit preprocessing statistics and readout calibration on training patients only. Lock test patients and action families before model selection.

## Evaluation measures and persistence comparison

Primary measures are error in predicted physical state changes and error in direct state readouts on held-out patients and held-out change families. Score action-conditioned prediction, persistence, and source-scan-only prediction on identical examples. Report patient-uniform aggregate metrics and per-action results with patient-cluster uncertainty. Separately report nuisance-only false-change magnitude and state-readout drift. Latent prediction loss is diagnostic only and cannot establish success.

## Costs

The first stage is a synthetic-pair generator, audit fixtures, and bounded CPU/GPU pilot using existing scans. No new cohort acquisition is required. Current transform preprocessing resizes to 96³ and z-scores nonzero voxels, while the geometry layer records verified pre-resize affine grids. Therefore intensity nuisance must be tested both before and after normalization, and physical changes must be applied in verified physical coordinates before the model resize. Estimate pair counts, storage, and training cost after confirming these operations. Do not launch full training before the generator passes physical-coordinate and nuisance invariance checks.

## Independent checks

Verify transformations on real image geometry and synthetic fixtures. Confirm known physical changes produce the expected physical target changes, while intensity-only and resolution-only actions preserve anatomy targets. Audit patient and transformation-family disjointness. Compare temporal JEPA against the per-scan direct-readout method and both simple baselines. Have an independent code/data audit inspect leakage, pair generation, and target calculations before the locked evaluation.

## Timing limits and cohort status

This is synthetic controlled-change evidence using local LUMIERE MRI as image content. It does not establish biological realism, treatment response, clinical utility, or independent cohort transfer. Real longitudinal validation would require independent repeat annotation or expert adjudication on real scans, then patient-held-out evaluation.

## Falsifiers and stop rules

Stop calling the result evidence for a useful world model if history adds no held-out value over source-scan-only prediction, if action-conditioned JEPA fails to improve physical-change prediction over persistence and direct readout, or if nuisance-only changes appear as anatomical change in the representation/readout. Stop before training if generated pairs fail geometry or target-consistency audits. A lower latent loss alone is not a success criterion.

## Repository evidence and implementation boundary

The current runtime path in `src/preprocessing/transforms.py` loads arrays, resizes to 96³, then z-scores nonzero voxels. It does not carry affine geometry through that resize. `src/preprocessing/geometry.py` provides verified physical grids and mask alignment, including explicit rejection of stale or missing geometry. The experiment must generate geometric actions in the verified pre-resize physical grid and derive physical targets before tensor resizing; it must not treat resized identity-affine arrays as physical space. Existing `outputs/anatomy_harmonization.json` provides measured native mask volumes and patient/visit identities, but those volumes are auto-segmentation targets with known repeatability and biological-validity limitations. There is no existing action-labeled synthetic-pair generator or direct physical-state readout task identified in this inspection.

The historical lesion-transition run is not reusable evidence: A40 found target-visit features and future gaps in its forecast input and withdrew its learned claims. The corrected causal replacement has no established gain. Whole-cohort geometry QA verifies grids and source volumes, but does not validate segmentation accuracy, repeatability, template equivalence, or biological changes.

## Expected unknowns

Whether physically defined synthetic transformations produce sufficiently diverse, plausible changes; whether the available auto-segmentation targets support reliable physical-change scoring; whether temporal context adds value beyond a scan and action; whether nuisance invariance transfers beyond the synthetic generator; and whether the proposed compute budget is adequate. Human expert adjudication or independent repeat annotation remains necessary before treating synthetic-change results as glioma-biological evidence.
