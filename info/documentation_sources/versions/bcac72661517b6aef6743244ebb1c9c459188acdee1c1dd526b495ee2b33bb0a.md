# World-model CLI verification map

This map covers the short-lived command-line entry points in `scripts/harness.py`. Start from the repository root and run each command through `uv run`.

## Baseline preconditions

- The project environment is available through `uv run`.
- Run the read-only doctor checks in the parent skill when the target or environment looks wrong.
- Never run verification against a process or dataset session that this run did not start.
- Do not use real SAILOR data or print credentials in captured evidence.

## Driving conventions

- Capture the exact command, stdout, stderr, and exit status.
- Use existing read-only artifact paths only for inspection. Use fresh paths for generated output.
- Preserve captured evidence under `/tmp/verify-world-model-cli/<run-id>/`.
- Start every recipe from its stated preconditions.

## Features

- [List CLI capabilities](./list-capabilities.md) checks the registered anatomy and legacy commands.
- [Inspect a checkpoint](./checkpoint-inspection.md) checks JSON output from the real checkpoint inspector.
- [Build an anatomy manifest](./anatomy-manifest.md) checks the LUMIERE and SAILOR-derived lesion-state manifest command when its input datasets are available.
