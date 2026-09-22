"""Versioned checkpoint compatibility for the anatomy-first harness.

Historical checkpoints are immutable inputs.  This module normalizes their
containers and common data-parallel prefixes, then loads only name/shape
compatible tensors.  Every omission is reported; non-legacy shape conflicts
fail loudly instead of being hidden by ``strict=False``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

CHECKPOINT_API_VERSION = 2
LEGACY_OPTIONAL_PREFIXES = ("rano_heads.",)


@dataclass
class CheckpointReport:
    path: str | None
    container: str
    format_version: int
    loaded: int
    missing: list[str]
    unexpected: list[str]
    ignored_legacy: list[str]
    shape_mismatch: list[str]
    epoch: int | None = None
    val_loss: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def read(source: str | Path | Mapping[str, Any]) -> tuple[dict, str | None]:
    if isinstance(source, (str, Path)):
        path = str(source)
        payload = torch.load(path, map_location="cpu", weights_only=False)
    else:
        path, payload = None, dict(source)
    if not isinstance(payload, dict):
        raise TypeError("checkpoint must contain a mapping")
    return payload, path


def state_dict(payload: Mapping[str, Any]) -> tuple[dict[str, torch.Tensor], str]:
    """Extract weights from current, legacy, or raw-state-dict containers."""
    for key in ("model", "state_dict", "weights"):
        value = payload.get(key)
        if isinstance(value, Mapping) and value:
            return _normalize_keys(value), key
    if payload and all(torch.is_tensor(v) for v in payload.values()):
        return _normalize_keys(payload), "raw_state_dict"
    raise ValueError("checkpoint has no model/state_dict/weights tensor mapping")


def _normalize_keys(weights: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in weights.items():
        if not torch.is_tensor(value):
            continue
        name = str(key)
        while name.startswith("module."):
            name = name[len("module."):]
        out[name] = value
    return out


def inspect(source: str | Path | Mapping[str, Any]) -> dict:
    payload, path = read(source)
    weights, container = state_dict(payload)
    legacy = sorted(k for k in weights if k.startswith(LEGACY_OPTIONAL_PREFIXES))
    return {
        "checkpoint_api_version": CHECKPOINT_API_VERSION,
        "path": path, "container": container,
        "declared_format_version": int(payload.get("checkpoint_format_version", 1)),
        "tensor_count": len(weights),
        "parameter_count": int(sum(v.numel() for v in weights.values())),
        "legacy_tensor_count": len(legacy),
        "legacy_prefixes": sorted({k.split(".", 1)[0] for k in legacy}),
        "epoch": payload.get("epoch"), "val_loss": payload.get("val_loss"),
        "has_optimizer": bool(payload.get("opt")),
    }


def load_model(model: torch.nn.Module, source: str | Path | Mapping[str, Any], *,
               optional_missing_prefixes: tuple[str, ...] = (),
               legacy_prefixes: tuple[str, ...] = LEGACY_OPTIONAL_PREFIXES,
               fail_on_shape_mismatch: bool = True) -> tuple[dict, CheckpointReport]:
    """Load compatible weights and return the original payload plus audit.

    Old RANO heads are ignored only when the destination model has no matching
    tensor.  New lesion heads may be declared optional while loading an old
    encoder.  All other shape conflicts fail loudly by default.
    """
    payload, path = read(source)
    incoming, container = state_dict(payload)
    expected = model.state_dict()
    compatible, unexpected, ignored, mismatched = {}, [], [], []
    for key, value in incoming.items():
        if key not in expected:
            (ignored if key.startswith(legacy_prefixes) else unexpected).append(key)
        elif tuple(value.shape) != tuple(expected[key].shape):
            mismatched.append(
                f"{key}: checkpoint{tuple(value.shape)} != model{tuple(expected[key].shape)}")
        else:
            compatible[key] = value
    if mismatched and fail_on_shape_mismatch:
        raise ValueError("checkpoint shape mismatch: " + "; ".join(mismatched[:10]))
    result = model.load_state_dict(compatible, strict=False)
    missing = [k for k in result.missing_keys
               if not k.startswith(optional_missing_prefixes)]
    report = CheckpointReport(
        path=path, container=container,
        format_version=int(payload.get("checkpoint_format_version", 1)),
        loaded=len(compatible), missing=sorted(missing),
        unexpected=sorted(unexpected), ignored_legacy=sorted(ignored),
        shape_mismatch=sorted(mismatched), epoch=payload.get("epoch"),
        val_loss=payload.get("val_loss"))
    return payload, report


def format_report(report: CheckpointReport) -> str:
    return (f"checkpoint={report.path or '<memory>'} format=v{report.format_version} "
            f"container={report.container} loaded={report.loaded} "
            f"missing={len(report.missing)} unexpected={len(report.unexpected)} "
            f"ignored_legacy={len(report.ignored_legacy)} "
            f"shape_mismatch={len(report.shape_mismatch)}")
