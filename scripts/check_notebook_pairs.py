#!/usr/bin/env python3
"""Fail if a Kaggle .py notebook source disagrees with its generated .ipynb."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pair_diff(source: Path, notebook: Path) -> str:
    """Return Jupytext's source-cell diff; empty means the pair is synchronized."""
    result = subprocess.run(
        [sys.executable, "-m", "jupytext", "--diff", str(source), str(notebook)],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    return result.stdout


def main() -> int:
    sources = sorted((ROOT / "kaggle").rglob("*.py"))
    pairs = [(source, source.with_suffix(".ipynb")) for source in sources
             if source.with_suffix(".ipynb").is_file()]
    failures = []
    for source, notebook in pairs:
        diff = pair_diff(source, notebook)
        if diff:
            failures.append((source.relative_to(ROOT), notebook.relative_to(ROOT), diff))
    if failures:
        for source, notebook, diff in failures:
            print(f"STALE NOTEBOOK PAIR: {source} != {notebook}\n{diff}", file=sys.stderr)
        print("Regenerate each .ipynb from its .py source with jupytext.", file=sys.stderr)
        return 1
    print(f"Notebook pairs synchronized: {len(pairs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
