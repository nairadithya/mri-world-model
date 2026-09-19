"""Immutable row manifests for task-aligned longitudinal evaluation.

A manifest makes the information cutoff explicit.  The same row IDs must be
used by every method in a head-to-head comparison; only the allowed inputs
change between assessment and forecast tasks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

MANIFEST_VERSION = 1
LABEL_VERSION = "rano-clean-v1"
TASKS = {
    "assessment": "observed x_t and x_{t+1} -> response at t+1",
    "forecast": "history through t -> response at a future visit",
    "anatomy": "history through t -> lesion state at a future visit",
    "surprise": "history through t + observed x_{t+1} -> change/progression",
}


def _value(x, default=None):
    if torch.is_tensor(x):
        if x.ndim == 0:
            return x.item()
        return x.tolist()
    return default if x is None else x


def _row_id(pid: str, source_i: int, target_i: int, task: str) -> str:
    raw = json.dumps([pid, source_i, target_i, task, LABEL_VERSION],
                     separators=(",", ":"))
    return hashlib.sha1(raw.encode()).hexdigest()[:20]


def _patient_rows(pid: str, p: dict, task: str) -> list[dict]:
    visits = list(p.get("visits", []))
    labels = p.get("labels", p.get("response_labels"))
    if labels is None:
        raise ValueError(f"{pid}: cache has no response labels")
    labels = [int(x) for x in _value(labels)]
    T = len(labels)
    if visits and len(visits) != T:
        raise ValueError(f"{pid}: visits/labels length mismatch")
    visits = visits or [f"visit-{i:03d}" for i in range(T)]
    has_img = p.get("has_img")
    if has_img is None:
        has_img = [True] * T
    else:
        has_img = [bool(x) for x in _value(has_img)]
    if len(has_img) != T:
        raise ValueError(f"{pid}: has_img/labels length mismatch")
    deltas = p.get("deltas")
    deltas = [float(x) for x in _value(deltas)] if deltas is not None else [0.0] * T
    if len(deltas) != T:
        raise ValueError(f"{pid}: deltas/labels length mismatch")
    operative = p.get("operative_event", [False] * T)
    operative = [bool(x) for x in _value(operative)]
    treatment = p.get("treatment", [-1] * T)
    treatment = [int(x) for x in _value(treatment)]
    rows = []
    for t in range(T - 1):
        u = t + 1
        valid = labels[u] >= 0
        base = {
            "row_id": _row_id(pid, t, u, task),
            "task": task,
            "label_version": LABEL_VERSION,
            "patient_id": pid,
            "source_index": t,
            "target_index": u,
            "source_visit": visits[t],
            "target_visit": visits[u],
            "gap_days": float(sum(deltas[t + 1:u + 1])),
            "source_has_image": has_img[t],
            "target_has_image": has_img[u],
            "response_label": labels[u],
            "response_label_valid": valid,
            "source_operative_event": operative[t],
            "target_operative_event": operative[u],
            "source_treatment": treatment[t],
            "target_treatment": treatment[u],
            "allowed_inputs": {
                "assessment": ["history_through_t", "x_target"],
                "forecast": ["history_through_t", "metadata_through_t"],
                "anatomy": ["history_through_t", "metadata_through_t"],
                "surprise": ["history_through_t", "metadata_through_t", "x_target"],
            }[task],
        }
        # A target label is required for response rows. Anatomy can later use
        # a continuous target even when RANO is unavailable, but the current
        # cache builder only emits rows with a known response endpoint.
        if valid and has_img[t] and has_img[u]:
            rows.append(base)
    return rows


def build_manifest(caches: list[dict], *, source: str,
                   tasks: tuple[str, ...] = tuple(TASKS)) -> dict:
    """Merge compatible caches and emit one immutable manifest."""
    patients = {}
    for cache in caches:
        for pid, p in cache.get("patients", {}).items():
            patients.setdefault(pid, {}).update(p)
    unknown = set(tasks) - set(TASKS)
    if unknown:
        raise ValueError(f"unknown tasks: {sorted(unknown)}")
    out = {"manifest_version": MANIFEST_VERSION,
           "label_version": LABEL_VERSION,
           "source": source,
           "tasks": {task: [] for task in tasks}}
    for task in tasks:
        for pid in sorted(patients):
            out["tasks"][task].extend(_patient_rows(pid, patients[pid], task))
    return out


def assert_compatible(manifests: dict[str, dict], task: str) -> None:
    """Assert that methods share exactly the same task rows and labels."""
    if not manifests:
        return
    rows = {}
    for name, manifest in manifests.items():
        got = {r["row_id"]: (r["patient_id"], r["source_index"],
                              r["target_index"], r["response_label"],
                              r["response_label_valid"])
               for r in manifest["tasks"].get(task, [])}
        rows[name] = got
    first_name, first = next(iter(rows.items()))
    for name, got in rows.items():
        if got != first:
            missing = sorted(set(first) - set(got))[:5]
            extra = sorted(set(got) - set(first))[:5]
            raise AssertionError(
                f"manifest mismatch {first_name} vs {name} for {task}: "
                f"missing={missing} extra={extra}")


def write_manifest(manifest: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description="build immutable task row manifest")
    ap.add_argument("--cache", nargs="+", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", nargs="*", default=list(TASKS))
    args = ap.parse_args(argv)
    caches = [torch.load(p, map_location="cpu", weights_only=False)
              for p in args.cache]
    manifest = build_manifest(caches, source=args.source,
                              tasks=tuple(args.tasks))
    write_manifest(manifest, args.out)
    print(" ".join(f"{k}={len(v)}" for k, v in manifest["tasks"].items()))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
