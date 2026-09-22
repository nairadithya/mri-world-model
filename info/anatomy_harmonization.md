# P0 anatomy harmonization — lesion-state-v1

Status: **P0 closed for next-observed-visit evaluation.** This is the RANO-free
target contract introduced by D45 and frozen by D46. It does not reinterpret
volume change as clinical response or RANO.

## Contract

The shared scalar state contains three physical-volume measurements:

- `necrotic_non_enhancing`
- `enhancing`
- `edema_flair`

All values are mm3. Primary prospective targets will be future `log1p` volume
and `delta log1p` volume. Growth/stable/shrinkage classes, if added later, are
derived imaging-change categories and must not be called progression or
response.

The manifest is generated with:

```bash
python scripts/harness.py run anatomy-manifest \
  --out outputs/anatomy_harmonization.json
```

## First full-tree audit (2026-09-22)

| cohort | patients | visits | measured visits | consecutive pairs | usable pairs | geometry-verified measurements | order-verified pairs |
|---|---:|---:|---:|---:|---:|---:|---:|
| LUMIERE | 91 | 638 | 599 | 547 | 496 | 599 | 547 |
| SAILOR | 27 | 270 | 269 | 243 | 242 | 269 | 243 |

LUMIERE has 39 visits without a lesion measurement. Those cause 32 source and
22 target exclusions across consecutive pairs; some pairs have both reasons.
The median usable next-visit gap is 84 days (range 1–1085).

All 599 shipped DeepBraTumIA atlas `seg_mask.nii.gz` files were recovered from
the local curated mask artifact. Affine-derived counts using the empirically
verified label map `{1: enhancing, 2: necrotic/nonenhancing, 3: edema}` match
the reported JSON exactly for all 1,797 compartment values. Physical geometry
is therefore independently reproducible for every measured LUMIERE visit.

SAILOR has one session without the ONCO edema mask, excluding one source pair.
All complete ONCO mask triplets are finite, have valid positive affine-derived
voxel volumes, and agree in array shape. The median usable gap is 77 days
(range 7–371).

The within-SAILOR target-source sensitivity audit found:

| compartment | overlapping sessions | Dice p10 / median / p90 | CL:ONCO volume ratio p10 / median / p90 |
|---|---:|---:|---:|
| enhancing | 239 | 0.000 / 0.650 / 0.842 | 0.620 / 0.911 / 10.551 |
| edema/FLAIR | 251 | 0.218 / 0.656 / 0.834 | 0.384 / 0.906 / 1.440 |

The median volumes are similar, but spatial agreement is only moderate and the
enhancing ratio has an extreme upper tail when ONCO is empty or nearly empty.
Consequently, ONCO remains the complete primary SAILOR measurement source and
CL is a required sensitivity arm; neither is treated as interchangeable truth.

## Timing scope

The LUMIERE paper documents week identifiers as rounded temporal bins: a visit
named week four can occur between days 28 and 35, and suffixes preserve order
for two scans in the same week. SAILOR `history.txt` documents that intervals
were manually derived from DICOM headers and/or a clinical spreadsheet, with
some intervals explicitly estimated to reconcile missing processed sessions.

This evidence verifies longitudinal ordering but not exact day-level horizons.
The frozen primary task is therefore **next observed visit**. `gap_days` is an
auxiliary covariate/sensitivity variable; it cannot support fixed-horizon or
precise rate claims. This restriction closes the P0 ambiguity without
fabricating date precision.

## Frozen eligibility and metrics

A pair is eligible when it joins consecutive curated visits, source and target
both have complete `lesion-state-v1` measurements, and the ordered gap is
positive. This yields 496 LUMIERE and 242 SAILOR pairs. Primary metrics are
log-volume MAE, delta-log-volume MAE, Spearman correlation, and improvement
over volume persistence, reported pooled and patient-uniform with paired
patient-cluster intervals.

Native target sources still differ by cohort (DeepBraTumIA versus ONCO).
Results must retain source-stratified reporting and the CL sensitivity arm;
these are not interchangeable manual ground truths.

The generated JSON lives under ignored `outputs/`; the code and this compact
audit are the reproducible record.
