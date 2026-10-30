# R37 full implementation review dispositions

October 26, 2026. Parent-authored correction record for the independent [first review](implementation-v1.md), whose verdict remains `CHANGES_REQUIRED` for its original hashes. That report is preserved unchanged. These corrections do not approve extraction, fitting or results; stable corrected inputs require independent re-review.

## B1. Reconstructed MRI geometry

`spatial_dynamics_data.validate_visit_geometry` now checks actual NIfTI shape/affine and canonical float32 content against the already content-bound inventory geometry record for every present modality. Identity-affine tensors require the existing verified pregrid/reconstruction convention. Native-affine files must match their saved grid and content too; the old native-affine early return is not used as evidence. Every modality's physical pregrid must agree with the prepared visit's pregrid, and raw-file digests bracket the geometry read itself as well as encoding. The initial digest is taken before loading the NIfTI; a changed digest after geometry verification is rejected, and extraction compares the same initial digest again after encoding. No OS-enforced filesystem immutability is assumed. The private cache retains per-file geometry evidence hashes and its aggregate digest; the run binds that digest. Existing geometry evidence is reused; no registration, new mask audit or biological validation is added. Every original visit must still pass the separate fixed pooled-feature and decoded-volume tolerance gates.

Synthetic rejection tests cover header/affine changes with unchanged arrays, stale current content, missing reconstruction verification, incompatible modalities, prepared-pregrid disagreement, and native-affine stale records. A further regression changes the on-disk header during the geometry read while the loaded header and returned array remain old; the post-read digest must reject it. This closes the narrow digest-boundary issue raised during re-review. These tests do not certify the real staged MRI until the actual extraction gate executes.

## B2. Authoritative resource and retry state

Production execution now uses exactly `/home/aditya/experiments/world-model-r37-20261026/receipts/continuation_budget.json`. The authority pins the authorized start, deadline and resource limits. One-time migration requires the exact pre-migration ledger SHA256 `cb717e6205a4af7e912e18854f19892c0a3f9169206cb48a22163aed3386c695` inspected by the independent reviewer and preserves its inventory spending and 900-second setup charge. The original E ledger and unknown v1 expenditure stay separate.

A separate fixed authority record and chained spending journal bind ledger identity and history. The run contract records the authority and current journal ancestor. Each execution verifies that the authoritative history extends that ancestor. A copied/fresh ledger, altered limits, reset history or unrelated ancestor is rejected. An exclusive `flock` reservation spans the whole stage and its ledger writes. Failed or running stages block further stages and cannot be bypassed with another output directory or `--ledger`. An interrupted migration/journal requires explicit recovery; the code does not silently start a new allowance. This is an integrity mechanism for the controlled execution path, not a defense against an operator deliberately replacing all private authority files and code.

Synthetic tests cover copied/fresh ledgers, changed deadline/limits/setup charge, history reset, concurrent reservation, interrupted/failed stages, and an unrelated locked journal tip. Authority migration has not yet executed against the production ledger at this disposition.

## B3. Assessment provenance and incomplete grading

Assessment outputs now carry the contract digest, fitting-completion digest, complete fitted-artifact-manifest digest, canonical full primary/single/nuisance row bindings and explicit seed identities. Reporting revalidates the original fitted artifacts and requires the exact cache-defined cohorts, earliest eligible nuisance cutoff per primary patient, all nuisance conditions, and matched seeds 42/43/44. Private prediction arrays are finite and shape-checked. Public reports expose hashes and aggregate counts, not private row identities. The private NPZ adds nuisance arrays, seed labels and cutoff measurements for independent replay.

`analyze` produces only a staged/ungraded record. The canonical aggregate is published only after the authoritative report stage completed and final deadline, CPU and GPU checks passed. A failed report leaves a staged, explicitly ungraded artifact, never an apparently completed aggregate.

Synthetic tests reject another contract/fitting receipt/artifact manifest, omitted or substituted primary/single/nuisance rows, a non-earliest nuisance cutoff, missing/duplicate nuisance seeds, missing arms and truncated prediction arrays. They also show that a resource-failed report cannot publish a completed aggregate.

## Executed check scope

All checks ran on the dedicated SSH environment, not the laptop. `implementation-tests-v5.log` reports 62 spatial tests, all passing. `implementation-checks-v5.json` binds the tested code/test hashes, interpreter, test-log hash and successful compilation list. These receipts live in the private remote `receipts/` directory and are mirrored under ignored `outputs/r37-full/continuation-20261026/remote-receipts/` as needed. No predictor fitting, real encoder reconstruction or full assessment ran for these tests. No failed scientific arm was retried. Source files are held stable for re-review; old approvals are not inherited by these new bytes.
