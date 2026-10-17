# October 13 research-record update

This update incorporates the R36 Phase 1 closure from `36a9e28` into the four canonical documents. It adds the maintenance procedure to `AGENTS.md` and archives ten duplicate or completed standalone records without changing their bytes.

The recovered [session context](session-context.md), [plan](plan.md), [baseline](baseline.json), and [source dispositions](coverage.json) describe ownership and source scope. The [R36 execution diff](r36_delta_representation_results.diff) compares the newly completed record with S065. The original 72-version catalog, October 12 archive, review reports, and aggregate measurements remain unchanged.

## Reproduce the checks

From the repository root:

```bash
python3 -B info/documentation_sources/updates/2026-10-13/check_update.py
python3 -B scripts/documentation_sources.py verify
python3 -B -m unittest discover -s tests -p 'test_documentation*.py' -v
uv run python -B -m unittest discover -s tests -p 'test_delta_summary.py' -v
uv run python -B scripts/check_notebook_pairs.py
python3 -B scripts/render_documentation_coverage.py
python3 -B scripts/refresh_research_run.py --reviewed
```

The update checker compares all 116 baseline immutable hashes, all ten relocated source hashes and dispositions, ten cohort-specific arm rows, five ridge rows, six published contrasts, and four protected checkout HEADs/statuses. [Negative fixture checks](negative_checks.json) record rejection of changed archive bytes, missing immutable evidence, a changed published metric, and duplicate source coverage.

The original verifier separately checks all 72 frozen versions, three supplementary files, canonical links and IDs, and all 47 records in both archive manifests. The later coverage generator lists update ledgers separately rather than changing the old capture count. Its regression test checks portable links and idempotence.

The parent ran 22 documentation tests, three delta-summary tests, all 17 notebook-pair checks, and compiled the five touched ordinary Python files. The parent rendered `info/research_run.html` with Chromium and inspected the first 3,600 pixels, including both R36 tables, their qualifications, and changed navigation. The page's measured values and scientific text were not changed by this update.

These are focused update checks, not a new training smoke test or a rerun of the original artifact audit. HEAD/status equality does not certify ignored outputs or research process state. Number-to-table association checks do not establish semantic completeness or scientific acceptance.

## Independent review boundary

The reviewer prepared by reading the full archived Phase 1 report, scientific result review, artifact audit, and closure receipt. It was not released to inspect mutable targets before the explicit stable-input handoff.

The reviewer lacks shell and SHA256 tools. Parent-executed byte checks are not independently recomputed digests. After the [timeout diagnosis](failure_diagnosis.md), the rerun assigns executable byte checks to a separate fresh read-only verifier before resuming the reviewer. The reviewer compares sources with the documents, scientific qualifications, preservation, navigation, and maintenance rules. Each lane's exact result and scope will be recorded separately. The handoff hashes bind the reviewed input version; later changes require their own disposition.

No research checkout, ignored output, dataset, checkpoint, credential, or research process was modified. No new scientific method, Phase 2, network activity, push, or merge was performed for this update.
