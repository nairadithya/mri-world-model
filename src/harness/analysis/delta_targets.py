"""Model-blind native segmentation integrity and resizing-loss audit."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import json

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np

from src.data.representation import COMPARTMENTS, digest
from src.preprocessing.geometry import Grid


def audit_native_targets(inventory_path: str, mask_root: str, metadata: dict, rows: list[dict]) -> dict:
    inventory = json.loads(Path(inventory_path).read_text())
    if digest(inventory_path) != metadata["inventory_sha256"]:
        raise ValueError("native audit inventory differs from tensor preparation")
    records = {v["visit_id"]: v for v in inventory["visits"]}
    hashes, volume_receipts, grid_receipts = {}, {}, {}

    @lru_cache(maxsize=4)
    def native(identifier):
        record = records[identifier]
        if "mask_specs" in record:
            specs = record["mask_specs"]
        else:
            specs = [{"relative": record["mask_relative"], "label": record["labels"][c]} for c in COMPARTMENTS]
        masks, grids = [], []
        for compartment, spec in zip(COMPARTMENTS, specs):
            path = Path(mask_root) / spec["relative"]
            if not path.exists():
                path = path.with_suffix("") if path.suffix == ".gz" else Path(str(path) + ".gz")
            if str(path) not in hashes:
                hashes[str(path)] = digest(path)
            image = nib.load(str(path))
            data = np.asarray(image.dataobj)
            grid = Grid(tuple(data.shape), image.affine)
            if not np.isfinite(data).all():
                raise ValueError("nonfinite native target")
            if spec["label"] is not None and not set(np.unique(data)) <= {0, *(s["label"] for s in specs)}:
                raise ValueError("native segmentation contains undeclared labels")
            mask = data > 0 if spec["label"] is None else data == spec["label"]
            measured = float(mask.sum() * abs(np.linalg.det(grid.affine[:3, :3])))
            if not np.isclose(measured, record["values_mm3"][compartment], atol=1, rtol=1e-6):
                raise ValueError("native volume disagrees with inventory")
            volume_receipts.setdefault(identifier, {})[compartment] = measured
            masks.append(mask); grids.append(grid)
        if not all(grids[0].same_as(g) for g in grids[1:]):
            raise ValueError("native compartment grids disagree")
        grid_receipts[identifier] = grids[0].record()
        return masks, grids[0]

    native_rows = []
    for index, row in enumerate(rows):
        source, grid = native(row["source_visit_id"])
        target, target_grid = native(row["target_visit_id"])
        if not grid.same_as(target_grid):
            target = [np.asarray(resample_from_to(nib.Nifti1Image(m.astype(np.uint8), target_grid.affine),
                                                (grid.shape, grid.affine), order=0).dataobj) > 0 for m in target]
        counts = [int(np.count_nonzero(a != b)) for a, b in zip(source, target)]
        lost = [count > 0 and resized == 0 for count, resized in zip(counts, row["changed_voxels"])]
        native_rows.append({"row_id": row["row_id"], "native_changed_voxels": counts,
                            "native_change_absent_at_96": lost, "same_native_grid": grid.same_as(target_grid)})
        if (index + 1) % 50 == 0:
            print(f"audit native pairs {index + 1}/{len(rows)}", flush=True)
    for identifier in records:
        native(identifier)
    return {"visits": len(volume_receipts), "pairs": len(native_rows), "native_volume_agreement": True,
            "lost_compartment_changes": np.asarray([r["native_change_absent_at_96"] for r in native_rows]).sum(0).tolist(),
            "changed_native_grid_pairs": sum(not r["same_native_grid"] for r in native_rows),
            "rows": native_rows, "native_volume_receipts": volume_receipts, "native_grids": grid_receipts,
            "native_mask_sha256": hashes,
            "comparison": "Binary compartments on the source native mask grid. Target is affine-aligned if grids differ.",
            "limits": "Native-to-96 loss includes field-of-view and resampling effects. Grid equality does not prove anatomy."}
