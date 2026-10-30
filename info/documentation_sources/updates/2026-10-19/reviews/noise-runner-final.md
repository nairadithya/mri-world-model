## Review

**Scope:** Fresh read-only recheck of the final CPU-accounting fix and focused tests, after reading the v5 review. No commands, edits, data access, network, or assessments performed.

### Correct
- Run startup persists non-restartable `running` status, start timestamps, and reserved remaining CPU before work; interrupted runs cannot silently restart (`scripts/run_spatial_noise_ablation.py:169–178`).
- SIGXCPU raises a caught `TimeoutError`. The CPU soft limit uses absolute process CPU plus remaining allowance; the hard limit is unchanged, and prior limits/handlers are restored (`scripts/run_spatial_noise_ablation.py:181–195`).
- The failure handler records consumed CPU and `run_failed` (`scripts/run_spatial_noise_ablation.py:262–263`).
- Regression coverage delivers real SIGXCPU while run handlers are active, then checks positive charged CPU, failed status, and restart rejection. Transition coverage also verifies `running` is non-restartable (`tests/test_spatial_noise_ablation.py:81–111`).

**Prior P1 resolved. No issues found.**

**Verification limits:** Focused-test success and v6 preflight digest `1d0fc791c9340487670f951a88b5362095c9feb32e559a2e0347244667e0ba86` with 423 eligible pairs and 18 history patients are parent-reported, not independently reproduced. The runner is untracked; an exact v5→v6 textual comparison was unavailable. Scientific logic was not reassessed.

### Merge verdict: **OK — scoped E-only execution**

This is not scientific-result approval or broader experiment authorization. Execution still requires the user’s explicit post-review run approval.