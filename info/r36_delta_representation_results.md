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

## Completed capacity and matrix launch, 2026-10-10

The private capacity kernel completed. A and B each made 5,940 real-pair exposures over exactly 20 epochs. A made 5,935 successful updates and five AMP skips. B made 5,936 successful updates and four skips. Their epoch-wise real-pair and nuisance schedules have identical hashes. Both final state dictionaries were downloaded and checked finite. Both original-R35 source-only probes completed 20 epochs. No query outcome was scored.

A training took 2,221.05 seconds and B took 2,532.37 seconds. The original-R35 difference probe took 768.28 seconds and target-only took 754.83 seconds. The complete work block took 6,299.69 seconds. Quota accounting changed from 5.72 to 7.49 GPU-hours used, an observed 1.77-hour cost. The new remaining quota is 22.51 hours; R36's own remaining allowance is 22.23 hours.

Projection scales from completed successful updates and source-probe overhead. It assumes C costs like B and D costs 1.7 times B; D's runtime has not yet been measured. It projects 9.67 hours for all twelve training arms and 6.35 hours for all thirty probes. A 20% cost margin and one setup hour give 20.21 GPU-hours. This is a scheduling estimate, not a measured full-run cost.

Two private matrix jobs now cover the ten remaining arms exactly once. Their work limits are 4.5 and 5.0 hours; API session limits are 5.0 and 5.5 hours. Each remaining probe job has a three-hour work limit and a 3.5-hour API session limit. The five follow-on session ceilings sum to 21 hours. Adding observed capacity spending gives 22.77 hours, below the 24-hour experiment ceiling even if every session reaches its limit. Each incomplete arm remains incomplete rather than receiving a shorter graded recipe.

The matrix kernels are `nairadithya/r36-delta-matrix1` and `nairadithya/r36-delta-matrix2`. Their server metadata confirms private T4 settings. No further dataset source or license was created. Models still run the original private snapshot from `1daf27d`; model, data, and loss module hashes are unchanged. `d6adba7` adds only the orchestration packager.

Evidence is `capacity-complete/unpacked/results/capacity.json`, `capacity-checks.json`, `capacity-projection.json`, the quota receipts, and the server metadata under `outputs/r36-delta-objective/`. Source-only training curves do not establish encoder improvement. The complete matched assessment, nuisance tests, local transfer, independent review, and Phase 2 decision remain open.

## Check-in, 2026-10-11 04:04 UTC

Both private matrix jobs remain RUNNING and recovered live logs confirm Tesla T4. Seed42 C and seed43 D reached epoch20. Together with the downloaded capacity A/B checkpoints, four of twelve arms are log-complete; only the capacity checkpoints have been locally verified so far. Latest recovered progress is seed42 D epoch9 and seed44 A epoch8. No fatal training exception is visible in the recovered logs. D's initial AMP backoffs remain reported, not hidden. No new query assessment has started.

The original live-log consumers exited with HTTP500; they have been reconnected. Recovered timestamped events show roughly one hour of epoch work, despite about five hours since launch. Scheduling/startup delay is the likely explanation, not a demonstrated training stall. The local six-hour matrix wait could have expired before the remote work finished. Only the local bounded waits were extended (12 hours for matrices, 10 for probes), and the waiting monitor was restarted without resubmitting either matrix. Training recipes, 20-epoch grading, private settings, API GPU session ceilings, and the 24-GPU-hour experiment ceiling are unchanged. Failed jobs still stop continuation; no automatic failed-arm retry was added.

The quota check reports 17.99 hours used and 12.01 remaining. Relative to the pre-R36 5.72-hour receipt, accounting currently charges 12.27 hours, leaving 11.73 under R36's own ceiling. This exactly includes the two matrix session ceilings; it is quota accounting, not a measured completed training cost. The three planned probe session ceilings still fit (10.5 hours), and the driver rechecks quota before each launch.

Evidence: `outputs/r36-delta-objective/checkin-2026-10-11.json`, recovered and timestamped live logs, `quota-checkin.txt`, and `continuation.json`. Four continuation guard tests pass: idempotent artifact merging, conflicting-byte refusal, queued-job waiting, and bounded stop on failure or expiry. The next work remains complete matrix receipts, source-only probes, and local assessment. Independent review is unavailable; Phase2 is not authorized.
