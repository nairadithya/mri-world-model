"""Structured run records: provenance + results, serialized on ``--out``.

Default stdout stays as before; this is purely additive. Keeping results in a
machine-readable artifact is what lets plots/info pull from one place instead
of regex-scraping logs.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

from .eval.evaluator import EvalResult


@dataclass
class RunRecord:
    kind: str
    results: list = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
    config: dict | None = None

    def add(self, result: EvalResult) -> None:
        entry = {
            "name": result.name,
            "metrics": result.metrics,
            "ci": {k: list(v) for k, v in result.ci.items()},
            "n_pat": result.n_pat,
            "n_rows": result.n_rows,
            "majority": result.majority,
            "fold_values": result.fold_values,
            "meta": result.meta,
        }
        self.results.append(entry)

    def summary(self) -> dict:
        return {
            "kind": self.kind,
            "provenance": self.provenance,
            "results": self.results,
        }

    def write(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.summary(), f, indent=2, sort_keys=True, default=str)
            f.write("\n")


def make_record(kind: str, *, provenance: dict | None = None,
                config: dict | None = None) -> RunRecord:
    rec = RunRecord(kind=kind, provenance=provenance or {}, config=config)
    return rec
