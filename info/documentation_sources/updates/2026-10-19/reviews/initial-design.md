# R37 scientific protocol review

**Verdict: CHANGES_REQUIRED**

## Scope

Read-only review of the edited R37 protocol and decision brief in `info/METHODS.md`, associated TODO, interpretation, and index changes, against the research guardrails and existing R35/R36 findings. Inspected the two named source files for encoder, decoder, checkpoint, and normalization assumptions.

The supervisor confirmed HEAD equals base `ec0d9abe0b5d1a5103f6b0826e22f991cb039073`. Review used the working-tree diff. No files were edited, participant data loaded, models run, or network operations performed.

## Review

### Correct

- **Prediction cutoff is explicit.** Future scans, masks, features, availability, treatments, and next-visit gaps are targets only or excluded. Adjacency and future-dependent preprocessing are addressed (`info/METHODS.md:387,404,414`).
- **The probabilistic anatomy score is mathematically compatible with the frozen head.** Spatial averaging followed by an affine head preserves Gaussianity. The saved inverse target normalization is affine too (`src/model/representation.py:47,66–72`; `src/harness/analysis/delta_protocol.py:104–107`). Marginal CRPS is appropriate for the declared marginal distributions; it does not validate their joint dependence.
- **Evaluation and scope are bounded.** Patient-uniform aggregation, seed-score averaging, paired patient bootstrapping, five-comparison 99% intervals, calibration, error retention, and nuisance gates are specified. Budget ceilings include extraction and failed work; incomplete arms cannot be graded (`info/METHODS.md:404–412`). Execution feasibility remains unverified, but no larger experiment is required for design review.
- **Historical findings remain scoped.** The update does not recast R35/R36 observed-change results as forecasting success or authorize the historical Phase 2 recipe (`info/TODO.md:19–43`; `info/NOTES_AND_INSIGHTS.md:19–23`; `info/README.md:11–15`; `info/EXPERIMENTS.md:327–496`).
- The distinction between history benefit, spatial-specific benefit, and secondary current-state prediction benefit is useful. The blockers below concern whether the comparisons identify those claims, rather than the numerical thresholds themselves.

### Finding 1 — P1: H versus L cannot isolate information from earlier MRI observations

**Location:** `info/METHODS.md:374,385,387,396,412`

H receives historical MRI, completed gaps, and history masks. L receives only the current scan with gap zero. Consequently, their contrast changes both imaging history and observation-schedule information.

A positive contrast can arise when earlier images add no predictive information conditional on the current scan, but completed gaps or history length predict the next observation. P and E do not remove that explanation; linear R need not capture nonlinear scheduling effects. The MRI perturbation check also leaves this shortcut untouched.

This prevents the stated MRI-history interpretation and falls short of the matched-comparison and shortcut requirements in `info/research_guardrails.md:13–15`.

**Smallest correction:** Keep the existing arm count, but give L the same cutoff-available timing, history-length/mask, and explicitly supplied historical availability covariates as H while withholding earlier MRI-derived maps. Specify the ablation so no duplicated current scan becomes artificial imaging history. Alternatively, narrow the hypothesis and verdict consistently to **combined imaging-and-observation-history benefit**, without attributing a gain specifically to earlier MRI.

### Finding 2 — P1: H versus G does not isolate local spatial dynamics

**Location:** `info/METHODS.md:389,400,412`; `info/NOTES_AND_INSIGHTS.md:21`

G pools **every** map, including the current scan, before nonlinear processing. H retains spatial structure for the current scan as well as historical scans. Thus H-minus-G conflates local temporal modeling with richer nonlinear current-state processing.

Broadcasting G’s residual onto the current map does not repair this: the frozen anatomy head subsequently averages spatially and applies a linear map (`src/model/representation.py:47,66–72`).

A counterexample is an additive predictor combining:

1. better nonlinear spatial processing of the current scan; and
2. useful history of globally pooled states.

Such a predictor can beat both L and G without exploiting local spatial correspondence across visits. Therefore, the proposed conjunction does not establish the stated local-spatial-dynamics advantage.

**Smallest correction:** Either remove that attribution and label H-minus-G as a comparison between the two complete predictor formulations, or redefine G to preserve H’s current-scan spatial pathway while restricting only earlier observations to pooled summaries. This can replace the existing G arm; no additional arm or sweep is necessary.

### Fixed

None; review-only.

### Merge verdict

**BLOCK** on scientific design approval pending the two comparison/claim corrections above. Neither correction requires new data, expert annotation, or reopening R35/R36.

## Review limits

This is protocol-design review only—not implementation approval, data-validity certification, demonstrated computational feasibility, or scientific success.

The supervisor reported passing documentation integrity checks. I did not independently execute them, and they do not resolve the scientific findings above. The affine-intensity perturbation is largely canceled by the existing standardization and must not be interpreted as learned invariance (`src/model/representation.py:121–131`; `info/NOTES_AND_INSIGHTS.md:198`).

## Reviewed-file SHA256 receipt

Hashes below were supplied by the supervisor from parent-executed `sha256sum`, **not independently computed by this reviewer**. The supervisor confirmed the edited inputs remained unchanged.

| File | SHA256 |
|---|---|
| `info/METHODS.md` | `2abc0d033c5d94dba43680f08743f678ff48fca3cc3596079b895d2f1f3913c4` |
| `info/TODO.md` | `82f6bc8a68907a76ce5bb5ac337f21661b6ba978a1639f530287ff3dc06b4d57` |
| `info/NOTES_AND_INSIGHTS.md` | `206a4b8b1f986a9fe111dad7aa3ccc9cf95ef00b47b53ab20ed976276b26e903` |
| `info/README.md` | `e5951cd561404a4d0d116024ce1fe892baa96612bcd700ad31f16fead287479d` |
| `info/research_guardrails.md` | `6e710be8c33b05d77bf8f13b415b64028fcc13dc03d71defb304e6a7631c78f4` |
| `info/EXPERIMENTS.md` | `df61c56402b8a47a8c4556d20ebd5ba508f674ad2bb01e2587760a689ed78c55` |
| `src/model/representation.py` | `d73125b5f05dca864cb308773bcd1924a7f40552371374753d94c147bff547f6` |
| `src/harness/analysis/delta_protocol.py` | `f5e62a51ce0bec99b7d59000d01344209dc94dcf113e97178ef4837cb8e9ade0` |