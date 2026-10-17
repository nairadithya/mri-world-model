# R35 independent artifact verification — resumed audit

**Execution: PASS_FOR_SCOPED_REPORT.** The bounded artifact audit completed with no integrity/replay failures. This supports the saved R35 report’s scoped numerical summary; it is not scientific validation or a general representation-quality verdict. Known provenance and biological-validation limits below remain.

## Assignment/worktree state

- Repository: `/home/adithya/projects/world-model`; branch `integration/r36-report-closure`; HEAD `b6b20c8d8254788a8e6888bd795eda68ccb61867`.
- The parent’s authorized current edits are limited to `info/EXPERIMENTS.md`, `scripts/render_representation_report.py`, and its regression test. No repository files or result artifacts were edited by this verification.
- Rehashed all **228 private artifact paths** from the original handoff separately after the parent’s public-input changes: **0 mismatches**. The former 245-path full-input check and hash of `handoff.json` were already reported in the preceding artifact.
- One bounded audit source and its aggregate output: `/tmp/r35-closure-review/audit_r35.py` (SHA256 `fcf6f90a3b2702e643afe21c45ed6e123bb4ad2302a20f9f353397d348fbaa22`), `/tmp/r35-closure-review/audit-output.txt` (SHA256 `22b1cc9277a16c5555d5dad6f22b6bc1bf6baa0d3e05dcb75953f3dc64544ac0`). Additional independent metric/bootstrap calculations: `/tmp/r35-closure-review/metrics_r35.py`, `metrics-output.txt`, `contrast_extra.py`, and `correct_independent.py`. No participant IDs or raw arrays were emitted into these outputs.

## Executable audit

Executed `/home/adithya/projects/world-model/.venv/bin/python -B /tmp/r35-closure-review/audit_r35.py` (exit **0**; **0 failures**). The audit independently loaded saved files and checked:

- **12/12** trained `best.pt` and `features.pt` pairs; checkpoint digest equals feature checkpoint binding and summary digest. Model state tensors are finite. The saved corrected contract is `float32-reductions-v1`.
- Run split membership: 52 source-training, 13 validation, and 26 encoder-unseen patients; pairwise disjoint and matched to the saved protocol. Each checkpoint’s seed/arm/training/validation provenance matches the run.
- **240/240** training epoch records are ordered, finite, and complete (20 per arm). Each saved best epoch/validation MAE equals the minimum of that arm’s saved validation history, consistent with R35’s validation-selected-best recipe (not R36 final20). Totals recomputed from histories: **74,592 successful updates**, **48 AMP backoffs**.
- **48 prediction archives**: 12 LUMIERE direct, 12 SAILOR direct, 12 source26, and 12 source52 candidate transfer outputs. For each, independently checked exact visit/row order and identity, truth and observed-delta equality from metadata, patient vectors, no repeated visit/row IDs, and finiteness. Candidate transfer source-patient sets match the exact saved source26/source52 budgets and are disjoint from query patients. Prepared/checkpoint/training-metadata provenance and summary hashes match.
- Replayed all candidate saved ridge bundles and both frozen-reference readout budgets using saved query features and saved scalers/models; **122 view predictions** checked, maximum absolute error bounded by the requested `1e-6` (no failure). No fitting or image inference was run.
- Nuisance records: 12 candidate arms × 5 conditions = **60**, plus frozen-reference **5**, each with exact 26-patient coverage and finite per-patient and aggregate metrics. Identity-relative excess and saved summary means were recomputed and matched.
- **24** state-audit records bound to the actual direct prediction-file SHA256 with zero mismatches. Empty-target strata were independently recalculated from predictions and metadata and matched the saved patient-uniform audit results.

The audit’s final aggregate line is `COUNTS {"amp_backoffs":48,"audit_hash_mismatches":0,"checkpoints":12,"epoch_rows":240,"feature_files":12,"nuisance_conditions":65,"prediction_archives":48,"replayed_views":122,"state_audits":24,"successful_updates":74592}`; `FAILURES 0`.

## Independent saved-prediction results

Additional computations in `/tmp/r35-closure-review/metrics_r35.py` and `contrast_extra.py` implement patient-uniform absolute errors directly with NumPy. Seed-level patient errors are averaged across seeds; the predictions are not ensembled. Each paired interval resamples patients with replacement for **10,000** draws using seed **2026**. Ordinary 95% intervals are reported; family-12 intervals are used for the stated paired architecture controls.

- LUMIERE direct state MAE: **1.11449**; frozen scalar-source52 control: **1.49995**; difference **−0.38546**, ordinary 95% CI **[−0.51843, −0.24811]** (149 visits, 26 patients). The direct model’s run training-mean baseline is 1.70506; it is not the matched frozen scalar-source52 control and is not substituted for it.
- LUMIERE paired-head change MAE **1.26964**, zero-change **1.54461**, later-only **1.43811**, independent differences from the separately trained state arm **1.25889**. Paired-minus-zero **−0.27497**, ordinary 95% CI **[−0.49530, −0.09566]**; paired-minus-later-only **−0.16847**, ordinary CI **[−0.34381, −0.01547]**; paired-minus-independent-state **+0.01075**, ordinary CI **[−0.07912, +0.10662]**. The last contrast does not establish added paired-head benefit.
- SAILOR direct state MAE **2.24010** vs source-trained mean **2.11053**; difference **+0.12956**, ordinary CI **[−0.25715, +0.50985]**. Direct paired-head MAE **1.05077** vs independent differences of the separately trained state arm **0.96648**; difference **+0.08429**, family-12 CI **[+0.02948, +0.14402]**. Direct paired-head vs zero was +0.10777 (ordinary CI [+0.02384, +0.21518]; family-12 CI [−0.00617, +0.27333]).
- SAILOR source26/source52 ridge state MAEs recomputed as **1.97146/1.95248**; matched source means **2.23906/2.11053**. State-minus-mean differences: source26 **−0.26760**, ordinary CI **[−0.61896, +0.07604]**; source52 **−0.15805**, ordinary CI **[−0.49661, +0.17592]**. Both remain inconclusive against their means.
- For SAILOR transfer ridge, paired learned-view vs independent-state differences: source26 **+0.07455**, family-12 CI **[+0.04818, +0.10441]**; source52 **+0.08143**, family-12 CI **[+0.04012, +0.13796]**. Against zero, source26 +0.07617 (family-12 CI [−0.00568, +0.19285]) and source52 +0.11140 (family-12 CI [−0.01045, +0.29140]).
- Spatial-state resolution nuisance excess over identity independently recomputed as **+0.28308**, ordinary patient-bootstrap 95% CI **[+0.19859, +0.37466]**. Synthetic resolution degradation therefore remains a material false-change sensitivity; it does not show that discarded information was nuisance-only.

These values are descriptive saved-artifact evidence, not new prospective gates. Comparisons retain the stated development-cohort limitations and do not establish biological change or independent generalization.

## Limits retained

- The audit could bind the cached reference readouts to their recorded checkpoint and prepared-metadata provenance, but this cannot supply a **creation-time MRI-byte fingerprint** for the earlier frozen-reference cache. Completion-time tensor hashes are not retroactive MRI-byte provenance.
- Shuffle audit prediction-file hashes were checked against their saved direct-prediction inputs, but the within-patient shuffled prediction arrays were not saved as a separately replayable artifact; the shuffle values/associations were therefore **not independently replayed**. Empty-target strata, in contrast, were recomputed.
- No new dense prediction replay, expert annotations, segmentation repeatability assessment, or untouched test cohort was available/performed. SAILOR remains developmental transfer. No biological validation, future forecast, or universal quality claim follows.
- A single real CLI checkpoint inspection and the focused unit suites were completed in the original run; not repeated here. Checkpoint metadata is not behavioral evidence.
