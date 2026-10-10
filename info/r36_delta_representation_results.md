# R36 execution record

## First verified units, 2026-10-10

Status. Phase 1 is INCOMPLETE. Target QA and the local numerical checks are complete. The private capacity session is running. No new representation-quality evaluation has been scored. Phase 2 is not authorized.

Implementation lives in `/home/adithya/projects/world-model-r36` on `exp/r36-delta-objective`. Its base is `389b8ab`. R35 modules were copied selectively from `2f6e1f5`. The original main checkout's existing edits and intentional deletions were preserved. Numerical artifacts physically live under the original project's ignored `outputs/r36-delta-objective/` directory.

The question remains whether change-balanced supervision improves frozen local access to signed occupancy change. C versus B is the primary objective-only contrast. The original R35 encoder and continued state-only A are controls. D tests the nuisance tradeoff. Useful observed change does not establish future predictability.

### Contract and target QA

The immutable private contract is `outputs/r36-delta-objective/qa/r36_contract.json`, with its SHA256 receipt. It locks the four arms, three seeds, 20 final-graded epochs, coefficients, source52 readout budget, source-only normalizers, paired patient bootstrap, decision margins, and resource ceilings before new query results.

The audit preserved 599 visits and 496 usable consecutive rows. The development assessment remains 149 visits from 26 encoder-unseen patients and 117 pairs from 23 patients. No pair bridges or QA exclusions were introduced. All native labels were finite, declared compartment labels. All 599 native compartment-volume measurements matched the inventory. Pair geometry and endpoint identity passed.

The new prepared metadata initially failed the R35 byte-identity guard. The only content difference was the inventory hash. The cloud package used compact JSON, while the local inventory used indented JSON. Canonical compact serialization reproduced R35's exact inventory hash. The repair retains both prepared-metadata hashes and requires exact equality of every other field, including visits, rows, targets, geometry, and patient lists. It does not relabel the old metadata or relax tensor-byte checks.

Median changed occupancy was 1,239.5, 524.5, and 3,866 voxels for core, enhancement, and edema. These are 0.140%, 0.059%, and 0.437% of the 96-cubed grid. Counts include all 496 pairs and are descriptive, not biological measurements. At 96 cubed, there were 7, 40, and 5 compartment-pairs with empty masks at both visits. These are also the counts with all-zero signed differences. No occupied compartment-pair had an exactly unchanged target. One native enhancing-compartment disagreement disappeared after preparation at 96 cubed. This counts loss of a binary target difference, not all lost MRI detail.

Six model-blind overlays were rendered and inspected by the parent. They cover stable empty targets, a large directional change, a clinically annotated postoperative target, and the known suspect extracranial label. The suspect label remains visible and remains in the primary analysis. Matching grids and visual plausibility do not establish anatomical correspondence, accurate segmentation, or repeatability. No expert adjudication or noise ceiling is available.

Evidence is in `qa/audit.json`, `qa/target_rows.json`, `qa/native_targets.json`, `qa/tensor_manifest.json`, and `qa/overlays/`. These files contain private inventories and are not committed.

### Implementation and numerical verification

`DeltaRepresentation` shares the unchanged R35 `SpatialState` encoder. Its change decoder receives only the four signed feature differences. The state decoder and native-volume head remain per-scan controls. `LocalLinearProbe` uses one shared voxelwise linear map per spatial level and fixed trilinear resizing. It has no nonlinear output activation.

The CLI now has `delta-representation audit`, `smoke`, `train`, `probe`, `encode`, `eval`, `nuisance`, `transfer`, `ridge`, and `report`. Source52 ridge fits save actual scalers and coefficients. All probes save their trained weights. Final checkpoints, probes, tensors, contracts, and features carry content binding. A's untrained change head is not graded.

The balanced loss averages changed and unchanged region means only when the region exists. It handles empty partitions without an invented target or division by zero. Reductions remain float32. D accumulates the real-pair and nuisance gradients before one optimizer update. Its sequential backward matches a joint update in the behavioral test and avoids retaining both dense graphs. AMP backoff still precedes clipping. Persistent overflow and full-precision nonfinite gradients remain fatal.

The real CLI smoke made two finite optimizer updates on aligned MRI pairs. All four maps had the prescribed widths and sizes, ending at `[1,128,12,12,12]`. Identity feature differences were exactly zero. The new decoder's reversal errors were 0.3865 and 0.3814 after those updates. These are numerical checks on an untrained auxiliary decoder, not quality findings. Antisymmetry is not guaranteed by the nonlinear graph.

Evidence is `outputs/r36-delta-objective/two-pair-smoke.json`. The existing R35 CLI smoke also passed. Synthetic tests cover signed partitions, missing channels, source dependence, reversal, affine probe behavior, finite gradients, gradient accumulation, split isolation, immutable artifacts, source-only scalar fitting, and real CLI discovery.

### Capacity session

Quota was rechecked at 24.28 GPU-hours remaining. The experiment reserves at most 24 GPU-hours and nine hours per session. The capacity notebook trains seed42 A and B for exactly 20 epochs and fits both original-R35 source-only linear probes. It contains no encoder-unseen or SAILOR scoring.

Kaggle rejected the first title because it did not resolve to the kernel slug. After that repair, it rejected the oversized inline source. The executable notebook was about 1.2 MB. Two private CPU staging kernels now carry the unchanged binary snapshot below the source-upload limit. Their server metadata confirms private execution with GPU disabled. Reconstruction checks the complete snapshot SHA256. No dataset license was added, no SAILOR data were uploaded, and no QA or tensor guard was removed.

The capacity kernel is `nairadithya/r36-delta-capacity`. Server metadata confirms private execution and `NvidiaTeslaT4`. The live log also confirms Tesla T4. The first observed A epoch had 297 pair exposures, 295 successful updates, and two AMP skips. Later observed epochs had 297 successful updates. Completion and full-matrix affordability remain unobserved at this record's creation. Falling training loss is not a representation finding.

### Priors after QA

Measured evidence supports treating changed occupancy as sparse. That makes the balanced-versus-whole-grid contrast worth testing. It does not establish that background dilution caused R35's results.

The resizing audit does not support widespread complete disappearance of native binary change as the main aggregate explanation. Only one compartment-row lost all native disagreement. Partial loss of boundary detail remains unquantified.

Target quality and anatomical correspondence remain important limits. Exact-zero target differences occurred only in empty compartment-pairs. Clinically stable disease need not produce an exactly stable automated mask. The inspected suspect contour persists. Empty native compartments must remain visible in the report. Neither a stronger decoder nor a lower training loss can substitute for the frozen-probe contrast and state/nuisance companions.

No follow-up family is selected yet. Present all objective arms and controls before choosing a further experiment. Forecasting remains a separately authorized question.

### Remaining gates

- Finish the capacity receipt and project total cost from completed successful updates and source-probe overhead.
- Complete all twelve final arms only if the locked matrix fits the resource ceiling.
- Complete the source-fitted probes, full-row state/change comparisons, nuisance tests, and local developmental transfer.
- Verify artifact identity and publish negative, incomplete, and inconclusive outcomes.
- Obtain independent design and result review. Native subagent tools are not exposed in this session. Parent review is not independent review.
- Obtain an operator decision before Phase 2.

The decision trail is `.audit/r36-delta-objective.tsv` in the execution worktree. The workflow is `.audit/r36-workflow.md`. The current missing-review status also applies to the trail itself.
