"""Tasks: bind a feature view + label framing (+ optional target transform).

A task answers "what rows / what label" and nothing else, so readouts,
protocols and metrics stay orthogonal. ``rows`` reproduces the exact row
semantics of the legacy probe functions:

- snapshot views (fused/vision/volumes/...): one row per clean-labelled visit;
- prefix views (states/states_roi): row ``s_t`` with label RANO_t
  (``*_current``) or RANO_{t+1} (``*_forecast``).
"""
from __future__ import annotations

import torch

from ..registry import TASKS
from ..encode import views

RANO_PROBE_MAP = {"PD": 0, "SD": 1, "PR": 2, "CR": 3}
RANO_PROBE_NAMES = ["PD", "SD", "PR", "CR"]

# prefix views whose row t carries the *next* visit's label
_FORECAST = {"states_forecast", "states_roi"}
_CURRENT = {"states_current"}


def _rows_from_view(p: dict, feat: str):
    """Per-patient rows for a view, or None. Mirrors probe_rano.patient_rows."""
    if feat in _FORECAST or feat in _CURRENT:
        src = "states_roi" if feat == "states_roi" else "states"
        if not views.has(p, src):
            return None
        x_src = views.get(p, src)
        off = 0 if feat in _CURRENT else 1
        idx = [t for t in range(len(x_src)) if p["labels"][t + off] >= 0]
        if not idx:
            return None
        return x_src[idx], p["labels"][[t + off for t in idx]]

    keep = p["labels"] >= 0
    if int(keep.sum()) == 0:
        return None
    n = int(keep.sum())
    if feat == "fused":
        x = views.get(p, "fused")[keep]
    elif feat == "vision":
        x = views.get(p, "vision")[keep]
    elif feat == "clinical":
        x = views.get(p, "clinical").unsqueeze(0).expand(n, -1)
    elif feat == "roi":
        x = views.get(p, "vision_roi")[keep]
    elif feat == "roi_concat":
        x = views.get(p, "vision_roi_mod")[keep].reshape(n, -1)
    elif feat == "mod_concat":
        x = views.get(p, "vision_mod")[keep].reshape(n, -1)
    elif feat == "volumes":
        x = views.get(p, "volumes")[keep]
    elif feat == "volumes_roi":
        x = torch.cat([views.get(p, "vision_roi")[keep], views.get(p, "volumes")[keep]], -1)
    elif feat == "volumes_clinical":
        x = torch.cat([views.get(p, "volumes")[keep],
                       views.get(p, "clinical").unsqueeze(0).expand(n, -1)], -1)
    elif feat in ("ema_z", "z", "vision_mod", "vision_roi_mod", "states", "states_roi"):
        x = views.get(p, feat)[keep]
    else:
        raise ValueError(f"unknown feat {feat}")
    return x, p["labels"][keep]


def patient_rows(patients: dict, pid: str, feat: str):
    """Feature rows + clean labels for one patient (None if no rows)."""
    r = _rows_from_view(patients[pid], feat)
    return r


def rows_for(patients: dict, splits: dict, split_names, feat: str):
    """Concatenated rows for one or more split names (legacy signature)."""
    wanted = [split_names] if isinstance(split_names, str) else split_names
    xs, ys = [], []
    for pid in [p for s in wanted for p in splits[s]]:
        r = patient_rows(patients, pid, feat)
        if r is not None:
            xs.append(r[0])
            ys.append(r[1])
    return torch.cat(xs), torch.cat(ys)


def rows(patients: dict, pids, view: str):
    """Concatenated rows for an arbitrary pid list (canonical task API)."""
    xs, ys = [], []
    for pid in pids:
        if pid not in patients:
            continue
        r = patient_rows(patients, pid, view)
        if r is not None:
            xs.append(r[0])
            ys.append(r[1])
    if not xs:
        return None
    return torch.cat(xs), torch.cat(ys)


class Task:
    name = "task"
    default_view = "fused"
    frame = "classification"
    class_names = RANO_PROBE_NAMES

    def rows(self, patients: dict, pids, view: str | None = None):
        return rows(patients, pids, view or self.default_view)

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        return y

    def filter(self, X: torch.Tensor, y: torch.Tensor):
        return X, y


@TASKS.register("rano4_forecast")
class RANO4Forecast(Task):
    name = "rano4_forecast"
    default_view = "states_forecast"


@TASKS.register("rano4_current")
class RANO4Current(Task):
    name = "rano4_current"
    default_view = "fused"


@TASKS.register("pd_forecast")
class PDForecast(Task):
    name = "pd_forecast"
    default_view = "states_forecast"
    frame = "binary"
    class_names = ["PD", "non-PD"]

    def transform(self, y):
        return (y == RANO_PROBE_MAP["PD"]).long()


@TASKS.register("response_forecast")
class ResponseForecast(Task):
    name = "response_forecast"
    default_view = "states_forecast"
    frame = "binary"
    class_names = ["non-response", "response"]

    def filter(self, X, y):
        keep = (y >= RANO_PROBE_MAP["SD"]) & (y <= RANO_PROBE_MAP["CR"])
        return X[keep], y[keep]

    def transform(self, y):
        return (y >= RANO_PROBE_MAP["PR"]).long()


@TASKS.register("latent_horizon")
class LatentHorizon(Task):
    """Pair task for representation error: state_t -> z_{t+n}.

    Has no classification rows; ``ReadoutEvaluator`` routes ``frame='latent'``
    to :class:`src.harness.eval.latent.LatentPairEvaluator`.
    """
    name = "latent_horizon"
    default_view = "z"
    frame = "latent"
