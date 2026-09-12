"""Train/eval harness: composable representation methods, tasks, metrics.

Two symmetric halves:

- :mod:`src.harness.train` — *methods* that learn or read out representations
  (JEPA objectives, velocity field, probes/readouts, supervised heads).
- :mod:`src.harness.eval` — *evaluations* that are just as pluggable: a task
  binds a feature view + label framing, a protocol picks cohorts/folds, and a
  registry of metrics scores it with shared CI/bootstrap machinery.

Artifact paths are frozen in :mod:`src.harness.paths`; the old scripts remain
behavior-compatible while they are migrated onto this library.
"""
from __future__ import annotations

from . import paths  # noqa: F401
from .registry import METHODS, METRICS, PROTOCOLS, TASKS, VIEWS, Registry

__all__ = [
    "METHODS",
    "METRICS",
    "PROTOCOLS",
    "TASKS",
    "VIEWS",
    "Registry",
    "paths",
]
