# Independent R36 Phase 1 result review

## Review

**Scientific verdict: APPROVED_FOR_SCOPED_REPORT.**  
**Final reporting-patch verdict: OK.**

No issues found.

The final integration source closes both previously identified P2 reporting concerns. It does not change the scientific conclusions:

- **C–B frozen accessibility: INCONCLUSIVE**
- **C-versus-A scoped state retention: VERIFIED**
- **D nuisance/information-loss tradeoff: INCONCLUSIVE**
- **Incremental observed temporal information: INCONCLUSIVE**

This is a post-result review, **not retrospective pre-training approval**. Computational completion does not establish global representation quality, biological change, or future prediction. **Phase 2 remains unauthorized.**

No edits, execution commands, jobs, network access, commits, or delegation were performed.

## 1. Final integration correction

I read the **complete current files** in `/home/adithya/projects/world-model`:

- `src/harness/analysis/delta_evaluation.py`
- `tests/test_delta_reporting.py`
- `outputs/r36-delta-objective/closure/phase1-report-parent.json`
- `outputs/r36-delta-objective/closure/phase1-report.json`

I reviewed their contents, not an independently obtained committed-range diff. The parent identifies the integration correction as `ac931da`; branch/commit identity and documentation archive hashes were not independently recomputed.

### Correct

`validate_nuisance` now:

- constructs the exact patient/first-visit/condition inventory from the corresponding LUMIERE states and checks contract patient coverage;
- rejects missing, duplicate, substituted, or unexpected tuples;
- recomputes the canonical tensor-manifest hash from the contract prepared hash and corresponding evaluation tensor provenance;
- hashes actual final-checkpoint and difference-probe bytes and crossbinds both nuisance and evaluation receipts;
- requires finite false-change, state-drift, and excess measurements;
- rejects negative raw errors, inconsistent identity subtraction, and condition summaries inconsistent with rows.

Evidence: `delta_evaluation.py:36–76`. The byte hash reads file contents, and the manifest hash uses canonical JSON (`src/data/representation.py:21–30`).

The D classifier correctly distinguishes demonstrated harm/margin violation, verified benefit with both retention margins met, and unresolved intervals (`delta_evaluation.py:142–159`). Missing nuisance evidence leaves that hypothesis inconclusive rather than fabricating a null.

### Fixed by parent

- **Previous P2: incomplete nuisance manifest/visit binding — resolved.** Exact manifest equality and expected tuples replace the previous presence-only/coverage-only checks.
- **Previous P2: absent reporting regression tests — resolved.** Seven tests cover successful, adverse, margin-violating, boundary-overlap, and missing-evidence verdicts; 18 tamper subcases; and replacement of actual checkpoint/probe bytes (`tests/test_delta_reporting.py:65–129`).

The overlap fixture exercises a boundary interval rather than a broad heterogeneous crossing interval. Source inspection nevertheless confirms that genuinely crossing intervals retain INCONCLUSIVE. This is not a remaining correctness finding.

The parent reports **36 delta tests passing**. I inspected the tests but did not rerun them.

The two complete reports have the same displayed numerical endpoints, intervals, classifications, training statuses, contract hash, and missing-companion inventory. This was a manual comparison, not a newly executed report regeneration.

## 2. Protocol and estimand

The primary objective-only contrast is **C versus B**. Matched architecture, seed-specific R35 initialization, state/volume-delta terms, pair schedule, final20 selection, and probe recipe remain shared; mask-change reduction differs. A is continued state-only; R35 is the original reference; D adds nuisance supervision.

Evidence: previously inspected `src/harness/analysis/delta_representation.py:39–51,248–350` and `delta_protocol.py:42–58`.

The frozen local probe uses the locked source52 training budget and difference/target-only views. Ridge selection is source-only grouped three-fold selection, not query fitting.

The primary uncertainty calculation:

1. averages pair errors within each patient;
2. averages those patient errors across three seeds;
3. bootstraps paired patients with 10,000 draws, RNG2026 and family size3.

It does **not** ensemble predictions (`delta_evaluation.py:15–33,101–123`). Intervals describe patient uncertainty conditional on these three fitted seeds; they are not an independent estimate of unrestricted training-seed variability.

Both endpoint MRIs are observed. These are **observed-change accessibility tests, not forecasting tests**.

## 3. Numerical evidence

Lower is better. Local errors are balanced signed-occupancy MAE; native state/scalar errors are log1p-volume MAE.

### Every seed and arm

| Seed | Arm | LUMIERE frozen local | LUMIERE direct state | SAILOR frozen local | SAILOR direct state |
|---|---|---:|---:|---:|---:|
|42|A|0.369993|0.947258|0.422751|2.192607|
|42|B|0.376471|0.984611|0.438979|2.274750|
|42|C|0.380133|0.925122|0.439671|2.165047|
|42|D|0.379772|0.935592|0.440803|2.204108|
|42|R35|0.375607|1.151726|0.450109|2.316356|
|43|A|0.387839|0.936406|0.425926|1.792658|
|43|B|0.395070|0.940771|0.437992|2.132749|
|43|C|0.384049|0.871478|0.428024|1.853798|
|43|D|0.393108|0.934511|0.436519|1.870774|
|43|R35|0.378576|1.079970|0.432684|2.044965|
|44|A|0.373033|0.904152|0.422787|2.642922|
|44|B|0.374373|1.057255|0.432699|2.431649|
|44|C|0.372530|0.901893|0.426273|2.333084|
|44|D|0.371324|0.970904|0.433398|2.389886|
|44|R35|0.366007|1.111774|0.432995|2.358973|

Sources: `matched-final/seed{42,43,44}/{A,B,C,D,R35}/{lumiere,sailor}.json`, `local.frozen_probe` and `state`, inspected during the initial review segment.

### Frozen hypothesis contrasts

| Contrast | Difference | Family-adjusted interval | Interpretation |
|---|---:|---|---|
|C–B frozen local|−0.003067|[−0.007995, +0.002644]|INCONCLUSIVE|
|C–A frozen local companion|+0.001949|[−0.004490, +0.007941]|INCONCLUSIVE|
|D–C frozen local companion|+0.002498|[−0.000374, +0.005687]|INCONCLUSIVE|
|C–A direct state|−0.029774|[−0.073244, +0.017074]|VERIFIED against +0.05 margin|
|D–C trained-perturbation excess false change|−0.000220|[−0.000416, −0.000083]|Reduction on this endpoint|
|D–C direct state|+0.047505|[+0.004175, +0.098686]|Worsening; noninferiority unresolved|
|C versus target-only local probe|−0.058958|[−0.074268, −0.043641]|Better than target-only|
|C versus independent-state local difference|+0.003539|[−0.008029, +0.013840]|No established advantage|
|D versus target-only local probe|−0.053606|[−0.068688, −0.038017]|Better than target-only|
|D versus independent-state local difference|+0.005528|[−0.005650, +0.016473]|No established advantage|

Local comparisons contain23 patients; state/nuisance comparisons contain26.

C–B’s raw95% interval also crosses zero: **[−0.007162,+0.001430]**. D–C local’s raw interval is positive, but its family-adjusted interval crosses zero.

Source: complete `closure/phase1-report-parent.json`, corroborated against the complete original report.

**Interpretation:** C’s frozen-local mean is0.378904 versus B0.381971, A0.376955, and R350.373396. Balanced supervision has **not verified an encoder-accessibility gain**. C’s retention finding applies to the locked direct native-volume endpoint, not every anatomical or pathological capability.

### Decoder and temporal controls

Three-seed means:

| Cohort | Arm | Frozen difference | Independent-state difference | Target-only | Direct nonlinear change |
|---|---|---:|---:|---:|---:|
|LUMIERE|A|0.376955|0.376293|0.437612|not graded|
|LUMIERE|B|0.381971|0.376508|0.440397|0.483508|
|LUMIERE|C|0.378904|0.375365|0.437862|0.270405|
|LUMIERE|D|0.381402|0.375873|0.435008|0.274940|
|LUMIERE|R35|0.373396|0.375064|0.440490|not graded|
|SAILOR|A|0.423821|0.396249|0.480684|not graded|
|SAILOR|B|0.436557|0.395125|0.483279|0.468779|
|SAILOR|C|0.431323|0.391896|0.483676|0.342619|
|SAILOR|D|0.436907|0.394481|0.490904|0.340042|
|SAILOR|R35|0.438596|0.397776|0.470256|not graded|

Zero-change local error is0.484840 on LUMIERE and0.469092 on SAILOR.

C/D’s trained nonlinear heads show a substantial descriptive improvement over B, but this does not substitute for the frozen-probe primary contrast. On developmental SAILOR, every frozen difference probe has higher descriptive error than its independent-state control.

A/R35’s unused nonlinear change heads are correctly not graded. Evidence: cohort JSON `local` sections and `delta_representation.py:576–598`.

### Source52 ridge companions

| Arm | LUMIERE state | LUMIERE delta | SAILOR state | SAILOR delta |
|---|---:|---:|---:|---:|
|A|0.887015|1.189453|2.180392|0.983491|
|B|0.937617|1.230057|2.207182|1.006660|
|C|0.897726|1.165711|2.051489|0.969616|
|D|0.915186|1.171241|2.065202|0.963816|
|R35|1.031088|1.273334|1.952590|0.980395|

Zero observed native-volume change gives1.544612 on LUMIERE and0.943004 on SAILOR. **All SAILOR ridge-delta means exceed zero change.** These descriptive companions do not establish a transfer gain.

The source-only fitting, saved scalers/coefficients, and prediction-replay check remain present (`delta_evaluation.py:215–282`; source grouped-ridge implementation previously inspected). Sources: all thirty `ridge-{lumiere,sailor}.json`, `scores.state/delta`.

## 4. D nuisance/information-loss tradeoff

Three-seed means:

| Arm | Identity false change | Noise excess | Bias excess | Resolution excess | Resolution state drift |
|---|---:|---:|---:|---:|---:|
|A|0.001893|0.000231|0.004407|0.017287|0.477840|
|B|0.001269|0.000229|0.004958|0.017137|0.421469|
|C|0.001765|0.000185|0.005160|0.017680|0.430766|
|D|0.001462|0.000215|0.004469|0.017105|0.383501|
|R35|0.001202|0.000217|0.003928|0.014826|0.463730|

Sources: all fifteen `nuisance.json`, `conditions`.

Important negative evidence remains:

- Bias excess decreases: D–C **−0.000691**, adjusted interval **[−0.001278,−0.000277]**.
- Noise excess **increases**: **+0.0000301**, interval **[+0.0000182,+0.0000436]**.
- Held-out resolution excess is unresolved: **−0.000575**, interval **[−0.001358,+0.000106]**.
- Direct state error worsens with an entirely positive interval, whose upper bound exceeds the +0.05 allowance.

D therefore does **not** establish a successful nuisance/information-retention tradeoff. The positive state-error interval establishes worsening, but not a demonstrated margin violation: its lower bound remains below+0.05. The final INCONCLUSIVE classification is correct.

### Standardized intensity stress

The affine transform applies scale1.5/offset0.3 and then re-standardizes foreground intensities (`src/model/representation.py:121–131`, reread in integration). Positive affine intensity changes cancel under this normalization for ordinary nondegenerate foreground distributions.

Observed excess errors around10⁻⁹ and state drifts around10⁻⁶ across arms therefore **do not establish learned intensity robustness**. This is effectively a vacuous stress in the current prepared-data setting. Preserve its locked result and limitation; do not retrospectively replace it or reinterpret it as useful invariance.

## 5. State failures, strata, and target QA

Aggregate retention must not hide compartment failures:

- C42 LUMIERE empty-compartment state MAE is **3.8969/3.0033/7.9117**, versus nonempty **0.9094/1.0766/0.3545**.
- C43/44 LUMIERE empty edema errors are **6.5499/7.3611**.
- D44 LUMIERE empty edema error is **8.4235**.
- D44 SAILOR empty errors are **10.7436/9.9477/10.9577**.
- Empty-state coverage is7/22/5 LUMIERE visits and58/28/1 SAILOR visits. Sparse support limits interpretation.

Companions include both-empty, appearance/disappearance, changed/unchanged partitions, source52 change-size quartiles, scalar controls, and explicitly labeled post-hoc state permutation:

- C42 changed-partition MAEs are **0.8180/0.6597/0.7648**, versus unchanged **0.0374/0.0408/0.0331**.
- Changed supports are114/103/116 pairs; unchanged support117 each.
- These partition summaries are supported-pair pooled means, **not** the primary patient-uniform estimand.
- C42 edema frozen-probe error is0.4244 in the smallest-change stratum versus0.3529 in the largest.
- Its both-empty edema error is0.0271 versus zero0 and independent-state0.0000288.
- No-positive AUPRC cases are `null`, with support recorded; absent support must not be described as perfect performance.
- The within-patient permutation is a visit-specificity diagnostic, **not a biological-change test**.

Source52 quartile thresholds are fixed at `[396,1273,2724]`, `[138,573,1587]`, and `[2076,4372,6746]`; no query-derived threshold selection was found. Evidence: `qa/audit.json:115–131`, companion code, and sampled detailed cohort companions.

QA records599 visits/496 usable pairs; evaluation has149 visits/26 patients and117 pairs/23 patients. No QA exclusions or bridging changes were introduced. Median changed occupancies are1239.5/524.5/3866 voxels. Exactly-zero differences occur only in empty compartment-pairs. One native enhancing disagreement disappears at96³.

During the initial review segment I independently viewed:

- `qa/overlays/known_suspect_extracranial.png`
- `qa/overlays/most_changed.png`
- `qa/overlays/empty_compartment.png`

The suspect extracranial contour remains visible and retained in the primary analysis. This is not expert adjudication.

Matching grids, automated label integrity, and plausible overlays do not establish anatomical correspondence, segmentation accuracy, repeatability, or biological attribution. The resizing audit measures complete disappearance of binary disagreement—not all lost boundary detail or MRI information.

## 6. Completion and identity

The reports record twelve completed final20 training arms and no missing companions. Recovery receipts separately attest completion of seed44 C and D/R35 without GPU relaunches. Every arm records5940 real-pair exposures; successful updates range5928–5936 because AMP skips remain disclosed.

The inspected matrix metadata were internally consistent:

- contract and prepared hashes agree across corresponding artifacts;
- final-checkpoint identities agree across training, evaluation, probe, ridge, and matrix receipts;
- probe identities agree across evaluation/nuisance records;
- all thirty probe receipts record completion through epoch20 and the common final probe-order hash.

The frozen loaders enforce checkpoint/probe byte receipts, graph identity, final20, and finite weights. The final report guard additionally binds **actual C/D checkpoint/probe bytes** to nuisance and LUMIERE receipts.

Two distinctions remain important:

1. The report’s tensor check binds the **declared evaluation tensor provenance**; it is not a new report-time reread of every tensor file.
2. R35 creation-time MRI byte fingerprinting was unavailable; R36 records prepared tensor bytes. That historical limitation is not retrospectively repaired.

Artifact existence/status and prior behavioral checks support computational completion. They are not equivalent to independently replaying all predictions.

## 7. Checked scope and remaining verification limits

### Initial review segment

Read governing documents and locked protocol/source/tests in the R36 review context:

- `AGENTS.md`
- `info/research_guardrails.md`
- `info/r36_delta_representation_handoff.md`
- `info/r36_delta_representation_results.md`
- `info/methodology_validity.md`
- `info/eval_protocol.md`
- delta protocol, targets, training/evaluation, data/model, ridge, nuisance, and existing representation/protocol tests.

Inspected aggregate/status/provenance/training sections across seeds42/43/44 × A/B/C/D/R35 for:

- LUMIERE and SAILOR evaluation JSONs;
- nuisance JSONs;
- both ridge JSONs;
- difference and target-only probe receipts.

Also inspected matrix/recovery/continuation receipts, artifact inventories, sampled detailed companions/hash receipts, QA audit, and the three PNGs listed above. Large participant-level artifacts were **not** completely audited row by row.

### Final resumed review

In `/home/adithya/projects/world-model`, completely reread the final reporting source, new reporting tests, both generated reports, current `AGENTS.md`, guardrails, and R36 execution record. Also reread the evaluation/nuisance/report seam, standardized-intensity implementation, and digest/canonical-hash implementation.

`scripts/summarize_delta_representation.py` was **not used as evidence**.

### Not independently executed or verified

- binary SHA256 recomputation by this reviewer;
- loading/replaying `.pt` checkpoints, probes, or ridge models;
- `.npz` prediction replay;
- all tensor bytes, row inventories, PR/reversal/direction metrics;
- bootstrap regeneration or fresh report generation;
- runtime test execution;
- integration commit/branch state or archive/source hash recomputation.

The reported36-test success and documentation hash verification remain parent attestations. No participant identifiers or participant-level rows are included here.

## 8. Scoped conclusion

Balanced supervision produces a strong descriptive **nonlinear-decoder improvement**, but does **not verify improved frozen accessibility over ordinary delta supervision**. C passes the locked direct-volume retention margin. D improves mean trained-perturbation excess false change while worsening state error and noise sensitivity; its intended tradeoff remains unresolved. C/D outperform target-only local probes but do not establish superiority over independent decoded-state differences.

Repeatedly examined LUMIERE/SAILOR assessments, automated targets, sparse empty-compartment support, registration uncertainty, missing repeatability/biological attribution, and absent untouched external validation remain limitations. SAILOR is developmental transfer—not final independent validation.

**APPROVED_FOR_SCOPED_REPORT permits reporting these bounded results, including their negative findings. It does not authorize Phase 2, a new method family, or forecasting/global-quality claims.**