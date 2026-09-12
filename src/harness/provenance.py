"""Artifact provenance: every cache records champion/config/git/date.

Repo convention 6: an encoded cache is only trustworthy if it carries the
champion path, config hash, git sha and creation date; loaders assert it.
Pre-2026-09-10 caches are filename-only and remain loadable (with a warning).
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import time


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def config_sha1(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def make_provenance(champion: str, config: str, **extra) -> dict:
    prov = {
        "champion": os.path.abspath(champion),
        "config": os.path.abspath(config),
        "config_sha1": config_sha1(config),
        "git_sha": git_sha(),
        "date": time.strftime("%Y-%m-%d"),
    }
    prov.update(extra)
    return prov


def load_provenance(cache: dict) -> dict | None:
    return cache.get("provenance")


def assert_provenance(cache: dict, *, warn: bool = True) -> None:
    """Warn (not raise) on legacy caches so old artifacts stay usable."""
    prov = load_provenance(cache)
    if prov:
        return
    if warn:
        print("WARNING: cache has no provenance block (pre-2026-09-10 legacy)")
