# Build an anatomy manifest

The anatomy manifest command measures available lesion masks and volumes, records timing and provenance, and writes the RANO-free `lesion-state-v1` manifest used by downstream anatomy analyses.

## Sub-features

- `manifest-build` discovers supported LUMIERE and SAILOR inputs.
- `manifest-output` writes a versioned JSON manifest with visits and usable pairs.

## How to get to it (user POV)

- Run `python scripts/harness.py anatomy manifest` from the repository root.
- Supply `--lumiere-root`, `--lumiere-volumes`, `--lumiere-masks`, `--sailor-root`, and `--out` to select inputs and output.

## Driving it with the shell

Preconditions:

- The required preprocessed LUMIERE and/or SAILOR inputs exist and have been authorized for use.
- Set every input root explicitly for a disposable verification fixture. Do not run against controlled-access data unless access is authorized.
- Set `--out` to a new path inside `/tmp/verify-world-model-cli/<run-id>/`.

- Action: run `uv run python scripts/harness.py anatomy manifest --lumiere-root <fixture-lumiere> --lumiere-volumes <fixture-volumes> --lumiere-masks <fixture-masks> --sailor-root <fixture-sailor> --out /tmp/verify-world-model-cli/<run-id>/manifest.json`. Proof: command exits 0 and the output parses as JSON with `target_version` equal to `lesion-state-v1`, plus `visits` and `pairs` fields.
- Action: inspect the output without modifying it. Proof: the JSON has a nonempty source/provenance section and each usable pair references visits present in the manifest.

## Gotchas

- The default roots may not exist on a fresh checkout. Missing inputs make this workflow unverified, not passed.
- SAILOR is controlled-access. Keep its source data local and do not attach it to external viewers or include it in evidence.
- Never overwrite `outputs/anatomy_harmonization.json` during verification.
- The CLI flags are defined in `src/harness/data/anatomy_manifest.py`; consult its current help or parser if they change.
