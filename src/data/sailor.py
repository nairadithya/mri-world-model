"""SAILOR longitudinal dataset adapter (D26).

Same item contract as LUMIEREDataset (patient_id, visits, paths, clinical,
actions, time_deltas, n_visits) so collate/model/eval run unchanged.

Layout: data/sailor/.../mni2009c-n-s/sub-XX/ses-YY/ with MNI-registered,
skull-stripped volumes. Modality slots: T1c->CT1, T1->T1, T2->T2,
Flair->FLAIR (base variants first: -icor files carry background NaNs in
~200 sessions, D26 recon; -icor only as fallback; runtime_transform
z-scores at load). Sessions missing all quad
modalities are dropped (imageless-drop mirror).
Time: intervals-days.txt holds per-gap day counts (n_ses-1 values);
deltas = [0] + gaps (soft input only; source notes approximations).
RANO: ses-02+ RANO.txt holds numeric codes; empirical codebook (D26 recon
via enhancing-volume deltas): 1=PD, 2=SD, 3=PR, 5=CR. ses-01 (baseline)
has no RANO -> action default (cf. LUMIERE missing->SD); probes must use
sailor_rano raw codes and skip None.
Clinical 6-vector (LUMIERE-compatible order): sex unknown->0, age/100,
IDH/MGMT unknown->na defaults, MGMT-q 0, OS months->weeks/200.
"""
from __future__ import annotations

import json
import os

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

SAILOR_MODALITIES = (("CT1", ("T1c", "T1c-icor")),
                     ("T1", ("T1", "T1-icor")),
                     ("T2", ("T2", "T2-icor")),
                     ("FLAIR", ("Flair", "Flair-icor")))
SAILOR_SLOTS = tuple(s for s, _ in SAILOR_MODALITIES)

# K3-1 content guard. File existence is not presence: the derivatives tree
# holds all-zero (and off-center) T2/T1 volumes that the old adapter counted
# as present and fed to the encoder, so a constant "empty" embedding entered
# the per-visit modality mean. A volume counts as present only if the central
# 42-58% box (resolution-agnostic; matches the K3 probe's central slab to 0
# mismatches on the derivatives tree) has >= 5% finite nonzero voxels.
EMPTY_CENTRAL_LO = 0.42
EMPTY_NZ_THRESHOLD = 0.05
CONTENT_MANIFEST = "_volume_content.json"
CONTENT_METRIC = 2  # bump to invalidate cached manifests after a metric change


def _central_nzfrac(path: str) -> float:
    """Finite-nonzero fraction of the central 42-58% box (0.0 on any error).

    Denominator is the FULL box, so NaN-background -icor files (D26) score
    low: non-finite voxels count as absent, never as content.
    """
    d = np.asanyarray(nib.load(path).dataobj)
    sh = d.shape
    sl = tuple(slice(int(EMPTY_CENTRAL_LO * sh[i]),
                     int((1.0 - EMPTY_CENTRAL_LO) * sh[i])) for i in range(len(sh)))
    c = d[sl]
    if c.size == 0:
        return 0.0
    finite = np.isfinite(c)
    return float(np.count_nonzero(c[finite]) / c.size)


def scan_empty_modalities(root: str, thr: float = EMPTY_NZ_THRESHOLD) -> list[dict]:
    """List slots whose present files are all empty (content, not existence).

    Mirrors the adapter's candidate order (base variants, then -icor), but is
    an independent walk for the QA gate: per slot it reports the first usable
    file, or the first present file flagged when none is usable.
    """
    rows: list[dict] = []
    subs = sorted(d for d in os.listdir(root)
                  if d.startswith("sub-") and os.path.isdir(os.path.join(root, d)))
    names = dict(SAILOR_MODALITIES)
    for sub in subs:
        sdir = os.path.join(root, sub)
        for ses in sorted(d for d in os.listdir(sdir)
                          if d.startswith("ses-") and os.path.isdir(os.path.join(sdir, d))):
            vdir = os.path.join(sdir, ses)
            for slot in SAILOR_SLOTS:
                first_present = None
                first_nz = 0.0
                chosen = None
                for name in names[slot]:
                    for ext in (".nii.gz", ".nii"):
                        p = os.path.join(vdir, f"{name}{ext}")
                        if not os.path.exists(p):
                            continue
                        nz = _central_nzfrac(p)
                        if first_present is None:
                            first_present, first_nz = p, nz
                        if nz >= thr:
                            chosen = (p, nz)
                            break
                    if chosen is not None:
                        break
                if chosen is None and first_present is not None:
                    rows.append({"subject": sub, "session": ses, "slot": slot,
                                 "path": first_present, "nz": first_nz})
    return rows

# numeric RANO code -> LUMIERE action id (PD3/SD2/PR5/CR4); 3-vs-5 tentative.
SAILOR_RANO_TO_ACTION = {1: 3, 2: 2, 3: 5, 5: 4}

# Per-session treatment status (treatment.txt) -> phase id for conditioned
# dynamics. Distinct channel from RANO actions (response, not treatment).
# Unknown/missing -> 3 (own id, not silently merged into another phase).
SAILOR_TREATMENT_TO_PHASE = {"CRT": 0, "TMZ": 1, "no": 2, "unknown": 3}
TREATMENT_NAMES = ["CRT", "TMZ", "no", "unknown"]


def _ses_key(ses: str) -> int:
    try:
        return int(ses.split("-")[1])
    except (IndexError, ValueError):
        return 0


class SAILORDataset(Dataset):
    def __init__(self, root: str, subjects: list[str] | None = None, min_visits: int = 2):
        self.root = root
        self._content: dict[str, list] = {}
        self._content_dirty = False
        self._load_manifest()
        subs = sorted(d for d in os.listdir(root)
                      if d.startswith("sub-") and os.path.isdir(os.path.join(root, d)))
        if subjects is not None:
            subs = [s for s in subs if s in set(subjects)]
        self.subjects = subs
        self.sessions: dict[str, list[str]] = {}
        self.all_sessions: dict[str, list[str]] = {}  # unfiltered ordering
        self.sailor_rano: dict[tuple[str, str], int | None] = {}
        self.treatment: dict[tuple[str, str], int] = {}
        self.intervals: dict[str, list[float]] = {}
        self.age: dict[str, float] = {}
        self.os_months: dict[str, float] = {}
        for sub in subs:
            sdir = os.path.join(root, sub)
            ses = sorted([d for d in os.listdir(sdir)
                          if d.startswith("ses-") and os.path.isdir(os.path.join(sdir, d))],
                         key=_ses_key)
            self.all_sessions[sub] = ses
            kept = [s for s in ses if self._has_any_image(sub, s)]
            self.sessions[sub] = kept
            for s in kept:
                rp = os.path.join(sdir, s, "RANO.txt")
                code = None
                if os.path.exists(rp):
                    try:
                        code = int(open(rp).read().strip().split()[0])
                    except ValueError:
                        code = None
                self.sailor_rano[(sub, s)] = code
                tp = os.path.join(sdir, s, "treatment.txt")
                phase = 3
                if os.path.exists(tp):
                    with open(tp) as f:
                        tok = f.read().strip().split()
                    phase = SAILOR_TREATMENT_TO_PHASE.get(
                        tok[0] if tok else "unknown", 3)
                self.treatment[(sub, s)] = phase
            ip = os.path.join(sdir, "intervals-days.txt")
            gaps: list[float] = []
            if os.path.exists(ip):
                try:
                    gaps = [float(x) for x in open(ip).read().split()]
                except ValueError:
                    gaps = []
            self.intervals[sub] = gaps
            for fn, store in (("age-years.txt", self.age),
                              ("overall-survival-months.txt", self.os_months)):
                fp = os.path.join(sdir, fn)
                if os.path.exists(fp):
                    try:
                        store[sub] = float(open(fp).read().strip().split()[0])
                    except ValueError:
                        pass
        self._save_manifest()
        self.subjects = [s for s in subs if len(self.sessions.get(s, [])) >= min_visits]
        print(f"SAILORDataset: {len(self.subjects)} subjects "
              f"({sum(len(self.sessions[s]) for s in self.subjects)} sessions)")

    def _manifest_file(self) -> str:
        return os.path.join(self.root, CONTENT_MANIFEST)

    def _load_manifest(self) -> None:
        p = self._manifest_file()
        if not os.path.exists(p):
            return
        try:
            with open(p) as f:
                data = json.load(f)
        except (OSError, ValueError):
            return
        meta = data.get("_meta", {})
        if (meta.get("threshold") != EMPTY_NZ_THRESHOLD
                or meta.get("central_lo") != EMPTY_CENTRAL_LO
                or meta.get("metric") != CONTENT_METRIC):
            return
        self._content = {k: v for k, v in data.items() if k != "_meta"}

    def _save_manifest(self) -> None:
        if not self._content_dirty:
            return
        data = dict(self._content)
        data["_meta"] = {"threshold": EMPTY_NZ_THRESHOLD,
                         "central_lo": EMPTY_CENTRAL_LO,
                         "metric": CONTENT_METRIC}
        try:
            with open(self._manifest_file(), "w") as f:
                json.dump(data, f)
        except OSError:
            pass

    def _usable(self, path: str) -> bool:
        """True if a cached (size, mtime) content check says the volume is present."""
        key = os.path.relpath(path, self.root)
        try:
            st = os.stat(path)
        except OSError:
            return False
        hit = self._content.get(key)
        if hit and hit[0] == st.st_size and abs(hit[1] - st.st_mtime) < 1e-6:
            return hit[2] >= EMPTY_NZ_THRESHOLD
        nz = _central_nzfrac(path)
        self._content[key] = [st.st_size, st.st_mtime, nz]
        self._content_dirty = True
        return nz >= EMPTY_NZ_THRESHOLD

    def _image_path(self, sub: str, ses: str, slot: str) -> str | None:
        names = dict(SAILOR_MODALITIES)[slot]
        d = os.path.join(self.root, sub, ses)
        for name in names:
            for ext in (".nii.gz", ".nii"):
                p = os.path.join(d, f"{name}{ext}")
                if os.path.exists(p) and self._usable(p):
                    return p
        return None

    def _has_any_image(self, sub: str, ses: str) -> bool:
        return any(self._image_path(sub, ses, s) is not None for s in SAILOR_SLOTS)

    def __len__(self) -> int:
        return len(self.subjects)

    def _aligned_deltas(self, sub: str, visits: list[str]) -> list[float]:
        """Day gaps aligned to KEPT session positions (G4 fix).

        intervals-days holds per-gap counts over the FULL session ordering;
        a dropped middle session must span (sum) the gaps it covered, not
        shift every later gap. Identical to first-k-gaps when nothing is
        dropped (verified); correct when something is.
        """
        full = self.all_sessions[sub]
        gaps = self.intervals.get(sub, [])
        pos = {s: i for i, s in enumerate(full)}
        deltas = [0.0]
        for prev, cur in zip(visits[:-1], visits[1:]):
            ia, ib = pos[prev], pos[cur]
            deltas.append(sum(gaps[k] for k in range(ia, ib) if k < len(gaps)))
        return deltas

    def __getitem__(self, idx: int) -> dict:
        sub = self.subjects[idx]
        visits = self.sessions[sub]
        paths = {s: [self._image_path(sub, v, s) for v in visits] for s in SAILOR_SLOTS}
        deltas = self._aligned_deltas(sub, visits)
        actions = []
        for v in visits:
            code = self.sailor_rano.get((sub, v))
            actions.append(SAILOR_RANO_TO_ACTION.get(code, 2) if code else 2)
        treat = [self.treatment.get((sub, v), 3) for v in visits]
        age = self.age.get(sub, 0.0) / 100.0
        surv = self.os_months.get(sub, 0.0) * 4.345 / 200.0
        clinical = torch.tensor([0, age, 3, 2, 0.0, surv], dtype=torch.float32)
        return {
            "patient_id": sub,
            "visits": visits,
            "paths": paths,
            "clinical": clinical,
            "actions": torch.tensor(actions, dtype=torch.long),
            "treatment": torch.tensor(treat, dtype=torch.long),
            "time_deltas": torch.tensor(deltas, dtype=torch.float32),
            "n_visits": len(visits),
        }
