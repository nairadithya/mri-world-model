"""Versioned, decoder-complete patient state. No hidden image skip connections."""
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import math

import torch

SEQUENCES = ("CT1", "T1", "T2", "FLAIR")
LABELS = {1: "enhancing", 2: "necrotic_non_enhancing", 3: "edema_flair"}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


@dataclass
class PatientState:
    features: dict[str, torch.Tensor]
    metadata: dict
    memory: dict[str, torch.Tensor] = field(default_factory=dict)
    version: int = 1

    def validate(self):
        required = {"patient", "visit", "nominal_week", "visit_order", "availability",
                    "affine", "shape_xyz", "model_id", "backend", "time_unit"}
        if self.version != 1 or not required <= self.metadata.keys():
            raise ValueError("Unsupported or incomplete patient state")
        m = self.metadata
        if len(m["availability"]) != 4 or not any(m["availability"]):
            raise ValueError("At least one structural sequence is required")
        if m["time_unit"] == "nominal_week":
            time_valid = m["nominal_week"] is not None and math.isfinite(m["nominal_week"])
        elif m["time_unit"] == "diagnosis_relative_days":
            time_valid = m["nominal_week"] is None and m.get("diagnosis_relative_day") is not None and math.isfinite(m["diagnosis_relative_day"])
        else:
            time_valid = m["time_unit"] == "visit_order_only_not_elapsed_time" and m["nominal_week"] is None
        if not time_valid:
            raise ValueError("Invalid nominal time")
        affine = torch.tensor(m["affine"], dtype=torch.float64)
        if affine.shape != (4, 4) or not torch.isfinite(affine).all() or abs(torch.linalg.det(affine[:3, :3])) < 1e-8:
            raise ValueError("Invalid state geometry")
        if not self.features or any(v.ndim != 5 or v.shape[0] != 1 for v in self.features.values()):
            raise ValueError("Features must be batched spatial tensors for one patient")
        for v in (*self.features.values(), *self.memory.values()):
            if not torch.isfinite(v).all():
                raise ValueError("Nonfinite patient state")
        return self

    @property
    def payload_bytes(self):
        return sum(v.numel() * v.element_size() for v in (*self.features.values(), *self.memory.values()))

    def to(self, device):
        return PatientState({k: v.to(device) for k, v in self.features.items()}, dict(self.metadata),
                            {k: v.to(device) for k, v in self.memory.items()}, self.version)

    def save(self, path):
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        cpu = self.to("cpu")
        tmp = path.with_suffix(".tmp")
        torch.save({"version": self.version, "metadata": self.metadata,
                    "features": {k: v.detach() for k, v in cpu.features.items()},
                    "memory": {k: v.detach() for k, v in cpu.memory.items()}}, tmp)
        tmp.replace(path)

    @classmethod
    def load(cls, path, device="cpu"):
        value = torch.load(path, map_location="cpu", weights_only=True)
        return cls(**value).validate().to(device)


def check_update(previous, current):
    a, b = previous.metadata, current.metadata
    if a["patient"] != b["patient"]:
        raise ValueError("Cannot combine different patients")
    if tuple(b["visit_order"]) <= tuple(a["visit_order"]):
        raise ValueError("Visits must arrive in strictly increasing nominal order")
    if a["model_id"] != b["model_id"]:
        raise ValueError("Re-encode history when changing the observation model")
    if a['time_unit'] != b['time_unit']:
        raise ValueError('Observation time units changed')
    if a['time_unit'] == 'diagnosis_relative_days' and b['diagnosis_relative_day'] <= a['diagnosis_relative_day']:
        raise ValueError('Recorded MU elapsed time must increase')
    if a["shape_xyz"] != b["shape_xyz"] or not torch.allclose(torch.tensor(a["affine"]), torch.tensor(b["affine"]), atol=1e-5, rtol=0):
        raise ValueError("Register visits to the same geometry before updating")


def persistence(state, horizon):
    if state.metadata["time_unit"] not in ("nominal_week", "diagnosis_relative_days"):
        raise ValueError("Elapsed timing is required for a nominal-week forecast")
    if horizon not in (12, 24):
        raise ValueError("Supported provisional horizons are 12 and 24 nominal weeks")
    week = state.metadata['nominal_week'] if state.metadata['time_unit']=='nominal_week' else state.metadata['diagnosis_relative_day']/7
    metadata = dict(state.metadata, forecast_method="persistence", horizon_nominal_weeks=horizon,
                    forecast_week=week + horizon,
                    uncertainty_status="unavailable: persistence has no fitted uncertainty")
    return PatientState(state.features, metadata, state.memory)
