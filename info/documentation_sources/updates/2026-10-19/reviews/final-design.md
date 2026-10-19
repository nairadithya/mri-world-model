# R37 bounded protocol-design re-review

**Verdict: APPROVED_FOR_PILOT_DESIGN**

## Reviewed scope

Fresh read-only review of:

- `info/documentation_sources/updates/2026-10-19/reviews/initial-design.md`
- `info/documentation_sources/updates/2026-10-19/reviews/dispositions.md`
- Revised R37 protocol, `info/METHODS.md:363–414`
- Associated interpretation, `info/NOTES_AND_INSIGHTS.md:19–23`
- R37 authorization and status in `info/TODO.md:7–17` and `info/README.md:13`
- Relevant research guardrails and `/tmp/r37-decisions.tsv`

The working-tree diff was inspected. Branch and base HEAD `ec0d9abe0b5d1a5103f6b0826e22f991cb039073` are task-supplied metadata, not independently verified Git state.

This review assesses the two initial P1 findings, their corrections, and contradictions or new blockers introduced by those corrections. It does not reopen settled protocol choices.

## Review

### Correct

- **Initial P1 finding 1 is resolved.** H and L now receive identical historical timing, masks, and availability covariates (`info/METHODS.md:389,396`). L removes only earlier standardized MRI maps, replacing them with constant zeros while retaining the actual current map. Earlier MRI summaries and duplicated current scans are explicitly excluded. The comparison now tests added MRI-history information conditional on the supplied observation-history covariates, rather than jointly removing imaging and scheduling information.
- **Initial P1 finding 2 is resolved.** G now uses H’s architecture and retains the current spatial map and its processing. Only earlier maps are pooled and broadcast (`info/METHODS.md:400`). This removes the previously identified current-scan-processing confound. The interpretation is appropriately limited to retaining earlier spatial detail; anatomical correspondence and physical mechanisms are explicitly not identified.
- **Claims match the revised comparisons.** The hypothesis, decision rule, and interpretation distinguish MRI-history benefit from the additional value of earlier spatial detail (`info/METHODS.md:385,412`; `info/NOTES_AND_INSIGHTS.md:21–23`).
- **Intensity-perturbation wording is corrected.** The protocol explicitly says standardization largely cancels the affine-intensity transform and prohibits interpreting a pass as learned intensity invariance (`info/METHODS.md:410`).
- **Scope remains bounded.** The six arms, three neural seeds, fixed fitting recipe, resource ceilings, and no-sweep stopping rule remain unchanged (`info/METHODS.md:406,412`). TODO and the documentation index retain implementation and reporting gates without claiming training completion or forecasting success.

### Fixed

Both initial P1 findings were corrected in the reviewed revision. No edits were made by this reviewer. The original `CHANGES_REQUIRED` report remains a valid receipt for its earlier inputs; this approval applies to the revised scope identified here.

### Findings

No issues found.

### Merge verdict

**OK — protocol design only.**

## Decision-log evidence note

The correction entry in `/tmp/r37-decisions.tsv` is supported by the reviewed protocol and dispositions. The protocol-first and separate-from-R36 decisions agree with the supplied authorization and current documentation.

The verification entry’s **“Integrity PASS”** is an externally reported execution result. Its cited `/tmp/r37-documentation-verify.log` was not inspected, and checks were not rerun. This is an unverified receipt in this review, not a scientific-design finding or evidence of design approval.

## Limits

This approval does **not** approve an implementation, certify data or geometry, establish computational feasibility, or approve results. Implementation review, future-input rejection tests, content-bound run provenance, bounded preflight checks, and independent aggregate-result review remain required (`info/METHODS.md:404,414`).

No project files were changed, commands or models run, private participant data accessed, or network operations performed. No subagents were used.

## Reviewed-input hash receipt

SHA256 values below are **parent-supplied and not independently computed by this reviewer**. The parent states that these inputs were held stable during review.

| File | SHA256 |
|---|---|
| `info/METHODS.md` | `e265b5789bf707b09ee2e137409a3cdd44e9993f0b064c1626035679df639dc5` |
| `info/NOTES_AND_INSIGHTS.md` | `3138d48f546ecf9c164cf215b77f24b7f0425fb0fedac9a55071469447b882ce` |
| `info/TODO.md` | `82f6bc8a68907a76ce5bb5ac337f21661b6ba978a1639f530287ff3dc06b4d57` |
| `info/README.md` | `e5951cd561404a4d0d116024ce1fe892baa96612bcd700ad31f16fead287479d` |