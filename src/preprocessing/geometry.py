"""Physical mask alignment for MRI tensors with verified pre-resize geometry."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import torch
import torch.nn.functional as F

from .transforms import resize_to, zscore_nonzero

GEOMETRY_CONTRACT = "affine-aligned-lesion-v1"
MANIFEST_SCHEMA = 1


@dataclass(frozen=True)
class Grid:
    shape: tuple[int, int, int]
    affine: np.ndarray

    def __post_init__(self):
        affine = np.array(self.affine, dtype=np.float64, copy=True)
        if (len(self.shape) != 3 or any(int(n) != n or n <= 0 for n in self.shape)
                or affine.shape != (4, 4) or not np.isfinite(affine).all()
                or abs(np.linalg.det(affine[:3, :3])) < 1e-12
                or not np.allclose(affine[3], [0, 0, 0, 1])):
            raise ValueError("expected a positive 3D shape and finite invertible voxel-to-world affine")
        affine.setflags(write=False)
        object.__setattr__(self, "affine", affine)
        object.__setattr__(self, "shape", tuple(int(n) for n in self.shape))

    def same_as(self, other: Grid) -> bool:
        return self.shape == other.shape and np.allclose(self.affine, other.affine, rtol=0, atol=1e-5)

    def record(self) -> dict:
        return {"shape": list(self.shape), "affine": self.affine.tolist()}


@dataclass(frozen=True)
class AlignedMasks:
    masks: torch.Tensor
    native_volumes_mm3: torch.Tensor
    provenance: dict


def require_geometry_contract(artifact: Mapping) -> None:
    if artifact.get("geometry_contract") != GEOMETRY_CONTRACT:
        raise ValueError("lesion artifact lacks verified aligned geometry; re-encode and retrain")


def _key(path: Path) -> str:
    stem = path.name.removesuffix(".gz").removesuffix(".nii")
    return "/".join([path.parent.parent.name, path.parent.name, stem])


def _signature(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return str(path.resolve()), stat.st_size, stat.st_mtime_ns


def _grid(image) -> Grid:
    return Grid(tuple(image.shape), image.affine)


def _content_hash(image, data: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(_grid(image).record(), sort_keys=True).encode())
    digest.update(np.ascontiguousarray(data, dtype="<f4").tobytes())
    return digest.hexdigest()


@lru_cache(maxsize=4096)
def _image_record(signature: tuple[str, int, int]) -> dict:
    image = nib.load(signature[0])
    data = image.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError(f"nonfinite MRI data: {signature[0]}")
    return {"tensor_grid": _grid(image).record(), "content_sha256": _content_hash(image, data)}


@lru_cache(maxsize=4096)
def _verified_pregrid(final_signature: tuple[str, int, int],
                      source_signature: tuple[str, int, int]) -> dict:
    final = nib.load(final_signature[0])
    source = nib.load(source_signature[0])
    original = source.get_fdata(dtype=np.float32)
    actual = final.get_fdata(dtype=np.float32)
    if not np.isfinite(original).all() or not np.isfinite(actual).all():
        raise ValueError("pre-resize reconstruction requires finite image data")
    expected = zscore_nonzero(resize_to(torch.from_numpy(original)[None], tuple(final.shape)))[0]
    if not torch.allclose(expected, torch.from_numpy(actual), rtol=1e-5, atol=1e-6):
        raise ValueError(f"pre-resize image does not reconstruct the final tensor: {final_signature[0]}")
    return {**_image_record(final_signature), "pregrid": _grid(source).record(),
            "pregrid_content_sha256": _content_hash(source, original),
            "reconstruction_verified": True,
            "resize": "trilinear-align_corners-false-then-nonzero-zscore"}


def load_geometry_manifest(path: str | Path | None) -> dict | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text())
    if (payload.get("schema_version") != MANIFEST_SCHEMA
            or payload.get("geometry_contract") != GEOMETRY_CONTRACT
            or not isinstance(payload.get("images"), dict)):
        raise ValueError("incompatible image geometry manifest")
    return payload


def image_grid(path: str | Path, manifest: Mapping | None = None) -> tuple[Grid, dict]:
    """Resolve physical image geometry, rejecting unproven identity-affine tensors."""
    path = Path(path)
    signature = _signature(path)
    image = nib.load(str(path))
    current = _grid(image)
    if not np.allclose(current.affine, np.eye(4), rtol=0, atol=1e-6):
        return current, {**_image_record(signature), "pregrid": current.record(), "source": "native-affine"}
    if manifest is None:
        stem = path.name.removesuffix(".gz").removesuffix(".nii")
        sidecar = path.with_name(stem + ".geometry.json")
        if sidecar.exists():
            manifest = load_geometry_manifest(sidecar)
    if manifest is not None:
        if (manifest.get("schema_version") != MANIFEST_SCHEMA
                or manifest.get("geometry_contract") != GEOMETRY_CONTRACT):
            raise ValueError("incompatible image geometry manifest")
        record = manifest.get("images", {}).get(_key(path))
        if (record is None or record.get("reconstruction_verified") is not True
                or record.get("resize") != "trilinear-align_corners-false-then-nonzero-zscore"
                or record.get("content_sha256") != _image_record(signature)["content_sha256"]
                or record.get("tensor_grid") != current.record()):
            raise ValueError(f"missing or stale verified image geometry: {_key(path)}")
    else:
        stem = path.name.removesuffix(".gz").removesuffix(".nii")
        candidates = [path.parent / "_tmp" / f"{stem}_bet{suffix}" for suffix in (".nii.gz", ".nii")]
        source = next((p for p in candidates if p.exists()), None)
        if source is None:
            raise ValueError(f"identity-affine image requires verified pre-resize geometry: {path}")
        record = _verified_pregrid(signature, _signature(source))
    physical = record["pregrid"]
    return Grid(tuple(physical["shape"]), physical["affine"]), dict(record)


def export_image_geometry(final_path: str | Path, pregrid_path: str | Path) -> Path:
    """Persist verified geometry alongside a newly created tensor NIfTI."""
    final = Path(final_path)
    record = _verified_pregrid(_signature(final), _signature(Path(pregrid_path)))
    stem = final.name.removesuffix(".gz").removesuffix(".nii")
    destination = final.with_name(stem + ".geometry.json")
    payload = {"schema_version": MANIFEST_SCHEMA, "geometry_contract": GEOMETRY_CONTRACT,
               "images": {_key(final): record}}
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return destination


def export_geometry_manifest(paths: Sequence[str | Path], out: str | Path) -> dict:
    """Export content-bound pre-resize geometry without changing image files."""
    records = {}
    for path in sorted(Path(p) for p in paths):
        key = _key(path)
        if key in records:
            raise ValueError(f"duplicate image geometry key: {key}")
        _, record = image_grid(path)
        records[key] = record
    payload = {"schema_version": MANIFEST_SCHEMA, "geometry_contract": GEOMETRY_CONTRACT,
               "images": records}
    dest = Path(out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return payload


def align_masks(specs: Sequence[tuple[str | Path, int | None]],
                image_paths: Sequence[str | Path], *, size: tuple[int, int, int] = (96, 96, 96),
                manifest: Mapping | None = None) -> AlignedMasks:
    """Align compartment masks to the physical image grid, then resize to its tensor grid.

    A label of None denotes a binary mask. Native physical volumes remain targets.
    Images that share a tensor must have the same pre-resize physical grid.
    """
    if not image_paths or not specs:
        raise ValueError("alignment requires image paths and mask specifications")
    resolved = [image_grid(p, manifest) for p in image_paths]
    reference = resolved[0][0]
    if any(record.get("reconstruction_verified") and tuple(record["tensor_grid"]["shape"]) != size
           for _, record in resolved):
        raise ValueError("processed tensor size differs from the requested mask grid")
    if not all(reference.same_as(grid) for grid, _ in resolved[1:]):
        raise ValueError("modality physical grids disagree; register them before lesion pooling")
    masks, volumes, sources = [], [], []
    for path, label in specs:
        image = nib.load(str(path))
        grid = _grid(image)
        data = np.asanyarray(image.dataobj)
        if not np.isfinite(data).all():
            raise ValueError(f"nonfinite lesion mask: {path}")
        foreground = data > 0 if label is None else data == label
        volumes.append(float(np.count_nonzero(foreground) * abs(np.linalg.det(grid.affine[:3, :3]))))
        binary = nib.Nifti1Image(foreground.astype(np.uint8), grid.affine)
        aligned = resample_from_to(binary, (reference.shape, reference.affine), order=0)
        masks.append(torch.from_numpy(np.array(aligned.dataobj, dtype=np.float32, copy=True)))
        sources.append({"path": str(path), "label": label, "grid": grid.record(),
                        "content_sha256": _content_hash(image, data)})
    tensor = F.interpolate(torch.stack(masks)[None], size=size, mode="nearest")[0] > 0
    provenance = {"geometry_contract": GEOMETRY_CONTRACT, "image_pregrid": reference.record(),
                  "images": {str(p): record for p, (_, record) in zip(image_paths, resolved)},
                  "masks": sources, "tensor_shape": list(size),
                  "registration_scope": "affine coordinate alignment; template equivalence is not established"}
    return AlignedMasks(tensor, torch.tensor(volumes, dtype=torch.float32), provenance)
