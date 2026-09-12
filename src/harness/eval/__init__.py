"""Pluggable evaluation: metrics, aggregation, tasks, protocols, reporting."""
from __future__ import annotations

from . import aggregate, metrics, report  # noqa: F401
from .evaluator import EvalResult, ReadoutEvaluator  # noqa: F401
