---
name: verify-world-model-cli
description: Verify the world-model research CLI through its real command-line entry points. Use when changing scripts/harness.py, the anatomy and legacy harness commands, or checkpoint inspection.
---

# Verify the world-model CLI

This skill covers the research CLI in `scripts/harness.py`. It is a short-lived command-line app, not a server. It reads local research data and artifacts. The repo also has a browser scan viewer, but this skill does not verify that UI.

Read `features/README.md` first, then select the matching feature recipe.

## Launch

Run commands from the repository root with `uv run`, which uses the pinned project environment. There is no server or readiness port. Each command starts and exits independently.

```bash
uv run python scripts/harness.py list
```

A ready environment prints `harness:    anatomy-v2` and the active anatomy commands. Stop a command with Ctrl-C only if this run started it. Do not kill an unrelated process. Commands that need datasets, checkpoints, or EBRAINS access state their preconditions in the feature recipe. No credentials are needed for local command discovery or checkpoint inspection.

## Doctor

Run these read-only checks before driving if the environment or target looks wrong.

```bash
pwd
git rev-parse --show-toplevel
git rev-parse --short HEAD
uv run python -c 'import torch; print("torch", torch.__version__)'
ps -eo pid,args | grep '[s]cripts/harness.py' || true
ss -ltnp 2>/dev/null | grep -E ':(8765)\b' || true
test -f .env && echo 'EBRAINS credential file present' || echo 'EBRAINS credential file absent'
```

The repo version is in `pyproject.toml`. There is no CLI daemon or required listening port. Port 8765 belongs to the separate scan viewer and should not be touched by this skill. The `.env` check reports presence only and must not print tokens. Local CLI checks do not require authentication. An absent `.env` only blocks workflows that explicitly use EBRAINS.

If `uv run` fails because dependencies are missing, use the documented setup (`uv sync`). Do not replace the project's CPU PyTorch setup or add `torchvision`.

## Drive

Use literal commands from the selected feature recipe. Capture stdout, stderr, and exit status. Prefer stable command names and explicit input paths. Do not run a command that can overwrite existing artifacts unless its recipe uses a fresh scratch output path.

## Evidence

Store proof under `/tmp/verify-world-model-cli/<run-id>/` and retain it after cleanup. Capture the command, stdout, stderr, and exit status. For commands that write an output file, use a disposable path and inspect that file after the command. A zero exit status alone does not prove the output is valid.

Do not mock local files or model outputs. Do not expose SAILOR data or credentials in captured logs. Do not run training or network acquisition as a smoke check.

## Cleanup

This skill's baseline commands are short-lived and should exit on their own. Stop only processes started by this run. Remove only scratch outputs created under the run's temporary directory. Keep evidence at the documented path. Do not delete or overwrite repository artifacts or user changes.

## Helpers

No helper scripts are required. The commands in this skill and its feature recipes can be run directly from the repository root.
