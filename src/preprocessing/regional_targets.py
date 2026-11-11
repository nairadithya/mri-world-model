"""Regional source-mask volumes without resampling foreground labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np

from .geometry import Grid


REGIONAL_TARGET_CONTRACT = "source-voxel-regional-mm3-v1"
REGION_ORDER = tuple((x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1))


@dataclass(frozen=True)
class RegionalVolumes:
    volumes_mm3: np.ndarray
    outside_mm3: np.ndarray
    source_mm3: np.ndarray

    @property
    def log1p_ml(self) -> np.ndarray:
        return np.log1p(self.volumes_mm3 / 1000.0)


def regional_volumes(
    mask_path: str | Path, labels: Sequence[int], reference: Grid
) -> RegionalVolumes:
    """Assign each labeled source voxel's full volume by its physical center.

    The reference field has voxel-coordinate bounds [-0.5, shape - 0.5).
    Bisect each axis to form eight regions in x-major, z-minor order. A center
    on an internal boundary belongs to the upper region. Centers outside the
    field contribute to outside_mm3 rather than being silently discarded.
    This conserves source voxel volume, not subvoxel geometric intersections.
    """
    if (
        not labels
        or any(int(label) != label or label <= 0 for label in labels)
        or len(set(labels)) != len(labels)
    ):
        raise ValueError("compartment labels must be distinct positive integers")
    image = nib.load(str(mask_path))
    source = Grid(tuple(image.shape), image.affine)
    values = np.asanyarray(image.dataobj)
    if (
        not np.isfinite(values).all()
        or not np.equal(values, np.floor(values)).all()
        or not np.isin(values, [0, *labels]).all()
    ):
        raise ValueError(
            "mask must contain only finite declared integer labels and background zero"
        )
    transform = np.linalg.solve(reference.affine, source.affine)
    shape = np.asarray(reference.shape, dtype=np.float64)
    counts = np.zeros((8, len(labels)), dtype=np.int64)
    outside = np.zeros(len(labels), dtype=np.int64)
    total = np.zeros(len(labels), dtype=np.int64)
    for compartment, label in enumerate(labels):
        centers = np.argwhere(values == label)
        total[compartment] = len(centers)
        coordinates = nib.affines.apply_affine(transform, centers)
        inside = ((coordinates >= -0.5) & (coordinates < shape - 0.5)).all(axis=1)
        halves = (coordinates[inside] >= shape / 2 - 0.5).astype(np.int64)
        region_ids = halves @ np.array([4, 2, 1])
        counts[:, compartment] = np.bincount(region_ids, minlength=8)
        outside[compartment] = int((~inside).sum())
    if not np.array_equal(counts.sum(axis=0) + outside, total):
        raise RuntimeError("source voxel partition did not conserve foreground")
    voxel_mm3 = abs(float(np.linalg.det(source.affine[:3, :3])))
    return RegionalVolumes(counts * voxel_mm3, outside * voxel_mm3, total * voxel_mm3)
