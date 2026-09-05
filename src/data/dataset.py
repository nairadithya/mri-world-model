"""LUMIERE longitudinal dataset.

One sample = one patient: variable-length visit sequence with per-visit
multi-sequence MRI paths, RANO action ids, time deltas, and static clinical
features. Actual NIfTI I/O happens in collate via runtime_transform so
workers do the heavy lifting.
"""
from __future__ import annotations

import os
import re

import pandas as pd
import torch
from torch.utils.data import Dataset

MODALITIES = ("CT1", "T1", "T2", "FLAIR")

# RANO rating -> categorical action id (config classes order).
RANO_ACTION_MAP = {
    "Pre-Op": 0,
    "Post-Op": 1,
    "SD": 2,
    "PD": 3,
    "CR": 4,
    "PR": 5,
    "Post-Op/PD": 3,  # treat as progression
    "None": 2,        # treat missing rating as stable
    "": 2,
}

IDH_MAP = {"WT": 0, "R132H mut": 1, "IDH1 neg, Sequencing required": 2, "na": 3, "": 3}
MGMT_MAP = {"methylated": 0, "not methylated": 1, "na": 2, "": 2}
SEX_MAP = {"female": 0, "male": 1}

WEEK_RE = re.compile(r"week-(\d+)(?:-(\d+))?")


def parse_week_to_days(timepoint: str) -> int:
    """week-NNN[-S] -> days since baseline (weeks*7; suffix adds sub-day epsilon)."""
    m = WEEK_RE.match(timepoint)
    if not m:
        return 0
    weeks = int(m.group(1))
    suffix = int(m.group(2)) if m.group(2) else 0
    return weeks * 7 + suffix  # suffix orders same-week scans


def sort_timepoints(tps: list[str]) -> list[str]:
    return sorted(tps, key=lambda t: (parse_week_to_days(t), t))


class LUMIEREDataset(Dataset):
    def __init__(
        self,
        meta_dir: str,
        processed_root: str,
        raw_root: str | None = None,
        patients: list[str] | None = None,
        modalities: tuple[str, ...] = MODALITIES,
        min_visits: int = 2,
    ):
        self.meta_dir = meta_dir
        self.processed_root = processed_root
        self.raw_root = raw_root
        self.modalities = modalities
        self.min_visits = min_visits

        demo = pd.read_csv(os.path.join(meta_dir, [f for f in os.listdir(meta_dir) if f.startswith("demographics")][0]))
        rano_files = [f for f in os.listdir(meta_dir) if f.startswith("rano")]
        rano = pd.read_csv(os.path.join(meta_dir, rano_files[0]))

        self.demographics = {row["Patient"]: row for _, row in demo.iterrows()}
        # (patient, date) -> rating string
        date_col = "Date" if "Date" in rano.columns else "Timepoint"
        rating_col = [c for c in rano.columns if "Rating" in c and "rationale" not in c.lower()][0]
        self.rano = {(r["Patient"], r[date_col]): str(r[rating_col]) for _, r in rano.iterrows()}

        # patient -> sorted visit list (from RANO rows + dirs on disk)
        visits: dict[str, set[str]] = {}
        for (p, d) in self.rano:
            visits.setdefault(p, set()).add(d)
        for root in (processed_root, raw_root or ""):
            if root and os.path.isdir(root):
                for p in os.listdir(root):
                    pdir = os.path.join(root, p)
                    if os.path.isdir(pdir):
                        for v in os.listdir(pdir):
                            if os.path.isdir(os.path.join(pdir, v)):
                                visits.setdefault(p, set()).add(v)

        self.patients = sorted(patients) if patients is not None else sorted(visits)
        self.visits = {p: sort_timepoints([v for v in visits.get(p, [])]) for p in self.patients}
        # Drop visits with no image file in any modality (e.g. Patient-025's
        # 8 upstream-missing weeks). Done eagerly so counts are exact even
        # with num_workers>0 (worker copies never report back).
        self.dropped_imageless = 0
        for p in self.patients:
            kept = [v for v in self.visits[p] if self._has_any_image(p, v)]
            self.dropped_imageless += len(self.visits[p]) - len(kept)
            self.visits[p] = kept
        if self.dropped_imageless:
            print(f"LUMIEREDataset: dropped {self.dropped_imageless} imageless visits")
        # drop patients with too few visits (after imageless-drop)
        self.patients = [p for p in self.patients if len(self.visits[p]) >= min_visits]

    def _has_any_image(self, patient: str, visit: str) -> bool:
        return any(self._image_path(patient, visit, m) is not None
                   for m in self.modalities)

    def __len__(self) -> int:
        return len(self.patients)

    def _image_path(self, patient: str, visit: str, mod: str) -> str | None:
        # Accept .nii.gz (pipeline output) and bare .nii: Kaggle's dataset
        # ingestion gunzips archives in place, so the same bytes may arrive
        # uncompressed. nibabel loads both transparently.
        for root in (self.processed_root, self.raw_root or ""):
            if not root:
                continue
            for ext in (".nii.gz", ".nii"):
                p = os.path.join(root, patient, visit, f"{mod}{ext}")
                if os.path.exists(p):
                    return p
        return None

    @staticmethod
    def _num(x, default: float = 0.0) -> float:
        """Float coercion that maps NaN / missing / junk to default.

        NB: bare `float(x) or default` does NOT catch NaN (NaN is truthy),
        which previously leaked NaNs into the clinical vector for patients
        missing survival (5) or MGMT-quantitative (26) values.
        """
        try:
            v = float(x)
        except (ValueError, TypeError):
            return default
        return default if pd.isna(v) else v

    @staticmethod
    def _clinical_vector(row: pd.Series) -> torch.Tensor:
        sex = SEX_MAP.get(str(row.get("Sex", "")), 0)
        age = LUMIEREDataset._num(row.get("Age at surgery (years)", 0)) / 100.0
        idh = IDH_MAP.get(str(row.get("IDH (WT: wild type)", "")), 3)
        mgmt = MGMT_MAP.get(str(row.get("MGMT qualitative", "")), 2)
        mgmt_q = LUMIEREDataset._num(
            str(row.get("MGMT quantitative", "na")).replace("%", "")) / 100.0
        surv = LUMIEREDataset._num(row.get("Survival time (weeks)", 0)) / 200.0
        return torch.tensor([sex, age, idh, mgmt, mgmt_q, surv], dtype=torch.float32)

    def __getitem__(self, idx: int) -> dict:
        patient = self.patients[idx]
        visits = self.visits[patient]

        paths = {m: [self._image_path(patient, v, m) for v in visits]
                 for m in self.modalities}
        # Safety net (init already filtered): drop visits with no pixels so
        # the predictor never trains toward the projector-bias constant.
        # Deltas recomputed below still span any gaps.
        kept_idx = [i for i in range(len(visits))
                    if any(paths[m][i] is not None for m in self.modalities)]
        if len(kept_idx) != len(visits):
            visits = [visits[i] for i in kept_idx]
            paths = {m: [paths[m][i] for i in kept_idx] for m in self.modalities}
        days = [parse_week_to_days(v) for v in visits]
        deltas = [0] + [b - a for a, b in zip(days, days[1:])]

        actions = []
        for v in visits:
            rating = self.rano.get((patient, v), "")
            actions.append(RANO_ACTION_MAP.get(rating, 2))

        clinical = self._clinical_vector(self.demographics[patient])

        return {
            "patient_id": patient,
            "visits": visits,
            "paths": paths,  # modality -> list[path|None] per visit
            "clinical": clinical,
            "actions": torch.tensor(actions, dtype=torch.long),
            "time_deltas": torch.tensor(deltas, dtype=torch.float32),
            "n_visits": len(visits),
        }
