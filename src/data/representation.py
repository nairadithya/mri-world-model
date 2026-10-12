"""MRI-only visit and observed-pair data with portable geometry provenance."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.preprocessing.geometry import align_masks
from src.preprocessing.transforms import runtime_transform

SEQUENCES = ("CT1", "T1", "T2", "FLAIR")
COMPARTMENTS = ("necrotic_non_enhancing", "enhancing", "edema_flair")
CONTRACT = "mri-only-state-observed-change-v1"


def digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def patient_relative(path: str, patient: str) -> str:
    parts = Path(path).parts
    indexes = [i for i, part in enumerate(parts) if part == patient]
    if len(indexes) != 1:
        raise ValueError("expected one exact patient directory")
    return str(Path(*parts[indexes[0]:]))


def export_inventory(manifest_path: str, geometry_path: str, protocol_path: str, out: str,
                     cohort: str = "LUMIERE") -> dict:
    manifest = json.loads(Path(manifest_path).read_text())
    geometry = json.loads(Path(geometry_path).read_text())
    protocol = json.loads(Path(protocol_path).read_text())
    train = set(protocol["encoder_train"])
    unseen = set(protocol["encoder_unseen"])
    if train & unseen or set(protocol["folds"]) != unseen:
        raise ValueError("invalid encoder/probe patient partition")
    if cohort not in {"LUMIERE", "SAILOR"}:
        raise ValueError("unknown cohort")
    visits = []
    for visit in manifest["visits"]:
        if visit["cohort"] != cohort or not visit.get("measurement") or visit.get("exclusions"):
            continue
        measurement = visit["measurement"]
        if "mask" in measurement:
            spec = measurement["mask"]
            mask_specs = [{"relative": patient_relative(spec["path"], visit["patient_id"]),
                           "label": spec["labels"][name]} for name in COMPARTMENTS]
        else:
            mask_specs = [{"relative": patient_relative(measurement["masks"][name]["path"], visit["patient_id"]),
                           "label": None} for name in COMPARTMENTS]
        visits.append({"visit_id": visit["visit_id"], "patient_id": visit["patient_id"],
                       "visit": visit["visit"], "cohort": cohort, "mask_specs": mask_specs,
                       "values_mm3": measurement["values_mm3"]})
    identifiers = {v["visit_id"] for v in visits}
    if len(identifiers) != len(visits):
        raise ValueError("duplicate visit identity")
    pairs = [p for p in manifest["pairs"] if p["cohort"] == cohort and p["usable"]]
    if any(p["source_visit_id"] not in identifiers or p["target_visit_id"] not in identifiers for p in pairs):
        raise ValueError("usable pair has no measured endpoint")
    patients = {v["patient_id"] for v in visits}
    if cohort == "LUMIERE" and not patients <= train | unseen:
        raise ValueError("visit outside frozen patient partition")
    payload = {"contract": CONTRACT, "schema_version": 1, "target_version": manifest["target_version"],
               "visits": visits, "pairs": pairs, "protocol": protocol,
               "cohort": cohort,
               "geometry": {**geometry, "images": {k: v for k, v in geometry["images"].items()
                                                     if k.split("/")[0] in patients}},
               "source_hashes": {"manifest": digest(manifest_path), "geometry": digest(geometry_path),
                                  "protocol": digest(protocol_path)}}
    save_json(out, payload)
    return payload


def find_image(directory: Path, stem: str) -> Path | None:
    return next((directory / f"{stem}{suffix}" for suffix in (".nii.gz", ".nii")
                 if (directory / f"{stem}{suffix}").is_file()), None)


def prepare(inventory_path: str, mri_root: str, mask_root: str, out: str, limit: int | None = None) -> dict:
    inventory = json.loads(Path(inventory_path).read_text())
    if inventory.get("contract") != CONTRACT:
        raise ValueError("wrong representation inventory contract")
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    inventory_hash = digest(inventory_path)
    metadata_path = output / "prepared.json"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text())
        if existing["inventory_sha256"] != inventory_hash or existing.get("limit") != limit:
            raise ValueError("prepared directory belongs to another inventory or limit")
        if all((output / v["tensor_file"]).is_file() for v in existing["visits"]):
            return existing
        raise ValueError("prepared cache lost tensors")
    visits = []
    selected = inventory["visits"][:limit] if limit is not None else inventory["visits"]
    for index, visit in enumerate(selected, 1):
        directory = Path(mri_root) / visit["patient_id"] / visit["visit"]
        if inventory.get("cohort", "LUMIERE") == "SAILOR":
            from src.data.sailor import SAILOR_MODALITIES, _central_nzfrac, EMPTY_NZ_THRESHOLD
            paths = []
            for _, stems in SAILOR_MODALITIES:
                candidates = [find_image(directory, stem) for stem in stems]
                path = next((p for p in candidates if p is not None and _central_nzfrac(str(p)) >= EMPTY_NZ_THRESHOLD), None)
                paths.append(path)
        else:
            paths = [find_image(directory, name) for name in SEQUENCES]
        present = np.asarray([p is not None for p in paths], dtype=bool)
        if not present.any():
            raise ValueError(f"measured visit has no MRI: {visit['visit_id']}")
        declared_masks = visit.get("mask_specs")
        if declared_masks is None and inventory.get("cohort", "LUMIERE") == "LUMIERE":
            declared_masks = [{"relative": visit["mask_relative"], "label": visit["labels"][c]} for c in COMPARTMENTS]
        if declared_masks is None or len(declared_masks) != len(COMPARTMENTS):
            raise ValueError("visit must declare exactly three compartment masks")
        mask_specs = []
        for spec in declared_masks:
            mask_path = Path(mask_root) / spec["relative"]
            if not mask_path.exists():
                alternate = mask_path.with_suffix("") if mask_path.suffix == ".gz" else Path(str(mask_path) + ".gz")
                mask_path = alternate if alternate.exists() else mask_path
            mask_specs.append((mask_path, spec["label"]))
        aligned = align_masks(mask_specs, [p for p in paths if p is not None], manifest=inventory["geometry"])
        native = aligned.native_volumes_mm3.numpy()
        expected = np.asarray([visit["values_mm3"][name] for name in COMPARTMENTS])
        if not np.allclose(native, expected, rtol=1e-6, atol=1):
            raise ValueError("native target volumes disagree with frozen manifest")
        images = np.zeros((4, 96, 96, 96), dtype=np.float16)
        for slot, path in enumerate(paths):
            if path is not None:
                image = runtime_transform(str(path))[0].numpy()
                if not np.isfinite(image).all() or not np.any(image != 0):
                    raise ValueError(f"nonfinite or empty MRI: {path}")
                images[slot] = image.astype(np.float16)
        filename = f"{visit['visit_id']}.npz"
        temporary = output / f"{filename}.tmp"
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, image=images, masks=aligned.masks.numpy(),
                                availability=present, volume=np.log1p(expected).astype(np.float32))
        temporary.replace(output / filename)
        visits.append({**visit, "tensor_file": filename,
                       "pregrid": aligned.provenance["image_pregrid"]})
        print(f"prepare {index}/{len(selected)}", flush=True)
    lookup = {v["visit_id"]: v for v in visits}
    pairs = []
    for pair in inventory["pairs"]:
        if pair["source_visit_id"] not in lookup or pair["target_visit_id"] not in lookup:
            continue
        source, target = lookup[pair["source_visit_id"]], lookup[pair["target_visit_id"]]
        if source["patient_id"] != target["patient_id"] or source["patient_id"] != pair["patient_id"]:
            raise ValueError("pair crosses patients")
        if source["pregrid"] != target["pregrid"]:
            raise ValueError("spatial change requires a verified shared physical grid")
        pairs.append(pair)
    payload = {"contract": CONTRACT, "schema_version": 1, "inventory_sha256": inventory_hash,
               "source_hashes": inventory["source_hashes"], "visits": visits, "pairs": pairs,
               "protocol": inventory["protocol"], "cohort": inventory.get("cohort", "LUMIERE"), "limit": limit,
               "geometry_contract": inventory["geometry"]["geometry_contract"],
               "image_shape": [4, 96, 96, 96], "targets": "automated native log1p compartment volumes"}
    save_json(metadata_path, payload)
    return payload


@dataclass(frozen=True)
class Visit:
    visit_id: str
    patient_id: str
    tensor_file: str


class Visits(Dataset):
    def __init__(self, root: str | Path, patients: list[str] | None = None):
        self.root = Path(root)
        self.metadata = json.loads((self.root / "prepared.json").read_text())
        if self.metadata.get("contract") != CONTRACT:
            raise ValueError("wrong prepared tensor contract")
        allowed = set(patients) if patients is not None else None
        self.records = [v for v in self.metadata["visits"] if allowed is None or v["patient_id"] in allowed]
        self.lookup = {v["visit_id"]: v for v in self.records}
        if not self.records or len(self.lookup) != len(self.records):
            raise ValueError("empty or duplicated visit dataset")

    def __len__(self):
        return len(self.records)

    def load(self, identifier: str) -> dict:
        visit = self.lookup[identifier]
        with np.load(self.root / visit["tensor_file"], allow_pickle=False) as data:
            result = {key: torch.from_numpy(data[key].copy()).float()
                      for key in ("image", "masks", "availability", "volume")}
        if (result["image"].shape != (4, 96, 96, 96) or result["masks"].shape != (3, 96, 96, 96)
                or result["availability"].shape != (4,) or result["volume"].shape != (3,)):
            raise ValueError("prepared visit tensor shapes disagree with the contract")
        if not all(torch.isfinite(value).all() for value in result.values()):
            raise ValueError("nonfinite prepared visit")
        if not ((result["masks"] == 0) | (result["masks"] == 1)).all() or (result["volume"] < 0).any():
            raise ValueError("invalid compartment targets")
        result.update(visit_id=identifier, patient_id=visit["patient_id"])
        return result

    def __getitem__(self, index: int) -> dict:
        return self.load(self.records[index]["visit_id"])


class ObservedPairs(Dataset):
    def __init__(self, visits: Visits):
        self.visits = visits
        self.records = [p for p in visits.metadata["pairs"]
                        if p["source_visit_id"] in visits.lookup and p["target_visit_id"] in visits.lookup]
        if not self.records or len({p["row_id"] for p in self.records}) != len(self.records):
            raise ValueError("empty or duplicated pair dataset")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        pair = self.records[index]
        source = self.visits.load(pair["source_visit_id"])
        target = self.visits.load(pair["target_visit_id"])
        return {"source": source, "target": target,
                "change": target["volume"] - source["volume"],
                "mask_change": target["masks"] - source["masks"],
                "row_id": pair["row_id"], "patient_id": pair["patient_id"]}
