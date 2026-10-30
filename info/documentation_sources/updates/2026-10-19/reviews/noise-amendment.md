# R37 noise-ablation amendment review

## Scope and evidence

Read-only review of the working-tree amendment, actual untracked baseline module and tests, R37 Methods/TODO, amendment receipt, source coverage, prior simple-controls review, and original design-review boundary. The neural decoder’s floor was also inspected.

Branch/HEAD are task-supplied metadata; no committed range or hashes were independently verified. No edits, commands, tests, participant-data access, fitting, network operations, or subagents were used. **Thirty passing focused tests are parent-reported**, not independently reproduced.

## Review

### Correct

- **The predictive law is now explicit.** `info/METHODS.md:404–406` specifies SD 0 versus 0.05, independent Gaussian observation noise per compartment, unchanged joint source rows and patient weights, and centers `q + m + k(d − m)`. It explicitly exempts discrete E from the general floor.
- **Code matches that law.** Explicit condition selection determines observation SD (`src/harness/analysis/spatial_dynamics_baselines.py:94–101,127–134`). Patient-weighted source means stay fixed while one scalar rescales complete three-compartment deviations (`103–114`).
- **CRPS mathematics is correct.** The discrete condition uses absolute differences; Gaussian marginals use the normal absolute expectation, with `sqrt(2) × SD` for independent mixture-pair differences (`46–50,82–89,116–124`). These marginal scores cannot establish cross-compartment noise dependence; no joint sampler is implemented here.
- **Calibration preserves both conditions and every grid score.** Each condition is fitted from the same supplied source arrays and independently calibrated on supplied validation scores (`137–148`). Selection uses compartment-then-patient-uniform averaging and deterministic grid-order tie breaking (`37–43,152–159`). Actual split provenance remains a future integration obligation.
- **P and neural floors remain unchanged.** Both apply the fixed 0.05 observation SD after calibration (`spatial_dynamics_baselines.py:63–69`; `src/model/spatial_dynamics.py:117–129`).
- **Scientific choices remain open rather than silently selected.** Common-`k=1` paired differences isolate noise; independently calibrated and all-grid comparisons are also retained. The main E condition and multiplicity treatment explicitly require locking and independent review before fitting or assessment (`info/METHODS.md:406`; `info/TODO.md:14`).
- **Earlier approval is not extended retroactively.** The amendment and receipt preserve the original design review’s limited scope and distinguish synthetic tests from an executed ablation (`info/METHODS.md:406`; `info/documentation_sources/updates/2026-10-19/r37-noise-amendment.md:17–19`).

### Fixed since the prior review

No fixes were applied by this reviewer.

- The prior **P1 unspecified-E-law blocker is resolved for this bounded comparison**, not for choosing the eventual primary control.
- The calibrated E-floor test now checks a one-component Gaussian reference at all three multipliers (`tests/test_spatial_dynamics_baselines.py:53–58`).
- The calibration fixture now distinguishes patient-uniform from row-uniform selection: the former selects 1, whereas the latter would select 0.5 (`92–97`).

### Finding — P2: Nondegenerate calibrated CRPS remains unprotected

**Location:** `tests/test_spatial_dynamics_baselines.py:29–58,74–90`.

The nondegenerate reference checks CRPS only at `k=1`; other multipliers are checked only through component centers. The new singleton fixture necessarily has no empirical dispersion. Receipt tests check only grid membership, finiteness, and nonnegativity.

Consequently, incorrectly ignoring calibration inside `ChangeControl.crps()`, or repeating `k=1` scores across receipt grid entries, would escape these assertions.

**Smallest fix:** extend the nondegenerate direct reference across both conditions and all three multipliers. Compare receipt score arrays with those references and verify each independently selected patient-uniform minimum. The current implementation itself passes inspection; this is a remaining coverage gap.

### Finding — P2: Amendment disposition is missing from the discoverable source ledger

**Locations:** `info/SOURCE_COVERAGE.md:94–100`; `info/documentation_sources/updates/2026-10-19/coverage.json:5–32`.

The launch diff initially showed a SOURCE_COVERAGE addition, but subsequent reads and the final diff showed it absent. The existing dated coverage ledger contains only the original protocol dispositions and does not link the new amendment receipt.

**Smallest fix:** append an amendment disposition and receipt link to the existing dated `coverage.json`. SOURCE_COVERAGE already links that ledger. Avoid relying on a hand-added generated-page line: `scripts/render_documentation_coverage.py:32–39` regenerates that section.

## Merge verdict

**OK with notes — bounded noise-ablation design and partial baseline code only.**

**Full-pilot launch remains blocked.** Main-contrast designation/multiplicity, remaining implementation, chronology and historical-gap evidence, geometry checks, provenance contracts, and resource enforcement remain outstanding. These are explicit scope limits, not newly attributed regressions. No real ablation or scientific result is approved by this review.