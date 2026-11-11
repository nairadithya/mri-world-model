"""Load physical regional targets with explicit units and cohort ordering."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from src.preprocessing.regional_targets import REGIONAL_TARGET_CONTRACT, RegionalVolumes


def load_regional_targets(
    root: str | Path, group: str, visit_ids: Sequence[str]
) -> RegionalVolumes:
    root = Path(root)
    manifest = json.loads((root / "manifest-private.json").read_text())
    if (
        manifest.get("contract") != REGIONAL_TARGET_CONTRACT
        or manifest.get("units") != "mm3"
        or manifest.get("transformed_target") != "log1p(volumes_mm3 / 1000)"
    ):
        raise ValueError(
            "incompatible regional targets; rebuild with scripts/data/build_regional_targets.py"
        )
    if group not in ("fitting", "selection", "scoring"):
        raise ValueError("unknown regional target group")
    path = root / f"{group}.npz"
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != manifest["arrays_sha256"][group]:
        raise ValueError("regional target array hash does not match its manifest")
    with np.load(path, allow_pickle=False) as archive:
        if archive["visit_ids"].tolist() != list(visit_ids):
            raise ValueError(
                "regional target visits differ from the requested cohort order"
            )
        volumes = archive["volumes_mm3"]
        outside = archive["outside_mm3"]
        source = archive["source_mm3"]
        transformed = archive["log1p_ml"]
    n = len(visit_ids)
    if (
        volumes.shape != (n, 8, 3)
        or outside.shape != (n, 3)
        or source.shape != (n, 3)
        or transformed.shape != volumes.shape
    ):
        raise ValueError("regional target array shapes do not match the cohort")
    if any(
        not np.isfinite(a).all() or (a < 0).any()
        for a in (volumes, outside, source, transformed)
    ):
        raise ValueError("regional targets must be finite nonnegative physical volumes")
    if not np.allclose(
        volumes.sum(1) + outside, source, rtol=1e-12, atol=1e-8
    ) or not np.array_equal(transformed, np.log1p(volumes / 1000)):
        raise ValueError(
            "regional target conservation or log-millilitre transform failed"
        )
    return RegionalVolumes(volumes, outside, source)
