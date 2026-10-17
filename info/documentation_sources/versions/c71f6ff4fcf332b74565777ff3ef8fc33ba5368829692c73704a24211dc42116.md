# List CLI capabilities

The harness list command prints the registered anatomy and legacy command families so a researcher can discover supported workflows without starting a data run.

## Sub-features

- `anatomy-list` prints active anatomy workflows.
- `legacy-list` prints legacy metrics, tasks, protocols, methods, and analyses.

## How to get to it (user POV)

- Run `python scripts/harness.py list` from the repository root.
- Run `python scripts/harness.py legacy list` to inspect the compatibility interface.

## Driving it with the shell

Preconditions:

- The repository root is the working directory.
- Project dependencies are available through `uv run`.

- Action: `uv run python scripts/harness.py list`. Proof: exit status is 0, stdout contains `harness:    anatomy-v2`, `anatomy:`, and `legacy:`.
- Action: `uv run python scripts/harness.py legacy list`. Proof: exit status is 0 and output includes `locked_unseen`, `rano4_forecast`, and `horizon-eval`.

## Gotchas

- `list` and `legacy list` are discovery commands. They do not prove that data-dependent workflows can run.
- Do not assume the older `AGENTS.md` command inventory exactly matches the current anatomy-v2 CLI.
