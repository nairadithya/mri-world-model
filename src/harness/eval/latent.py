"""Representation-error evaluation over latent (t, t+n) pairs.

The other half of the harness: instead of a classification readout, score how
well a prediction method reconstructs future EMA latents, against persistence
on the identical pair set. Methods are registered callables; metrics come from
the same registry as RANO eval (``cosine_error``, ``persistence_error``).
"""
from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field

import torch

from .aggregate import percentile_ci
from .metrics import cosine_error
from .. import paths
from ..encode import views
from ..train.latent import load_1step_predictor, load_gap_head


@dataclass
class LatentPair:
    pid: str
    split: str
    t: int
    n: int
    gap: float
    state: torch.Tensor
    z_t: torch.Tensor
    z_tgt: torch.Tensor


@dataclass
class LatentResult:
    methods: list = field(default_factory=list)
    overall: dict = field(default_factory=dict)
    per_horizon: dict = field(default_factory=dict)
    pairwise: dict = field(default_factory=dict)
    n_pairs: int = 0
    n_patients: int = 0
    horizon_counts: dict = field(default_factory=dict)


def build_pairs(patients: dict, pids=None, split: str | None = None) -> list[LatentPair]:
    want = set(pids) if pids else None
    rows: list[LatentPair] = []
    for pid, p in patients.items():
        if want is not None and pid not in want:
            continue
        s = p.get("split", "?")
        if split and s != split:
            continue
        try:
            z = views.get(p, "z")
            states = views.get(p, "states")
        except KeyError:
            continue
        has_img, deltas = p.get("has_img"), p.get("deltas")
        if has_img is None or deltas is None:
            continue
        T = len(z)
        for t in range(T - 1):
            if not bool(has_img[t]):
                continue
            gap = 0.0
            for u in range(t + 1, T):
                gap += float(deltas[u])
                if not bool(has_img[u]):
                    continue
                rows.append(LatentPair(pid, s, t, u - t, gap,
                                       states[t], z[t], z[u]))
    return rows


# --- prediction sources ----------------------------------------------------

def _persistence(rows, ctx):
    return torch.stack([r.z_t for r in rows])


def _champ(rows, ctx):
    S = torch.stack([r.state for r in rows])
    with torch.no_grad():
        return ctx["champ"](S)


def _gap_head(rows, ctx):
    net, mu, sd = ctx["gap_head"]
    S = torch.stack([r.state for r in rows])
    G = torch.tensor([r.gap for r in rows])
    g = (torch.log1p(G.clamp_min(0)) - mu) / sd
    with torch.no_grad():
        return net(S, g)


PREDICTIONS = {
    "persistence": _persistence,
    "champ": _champ,
    "gap_head": _gap_head,
}


class LatentPairEvaluator:
    def __init__(self, metrics=("cosine_error",), predictions=("persistence", "champ"),
                 champion_path: str = paths.CHAMPION,
                 gap_head_path: str = paths.PROBE_HEAD,
                 per_horizon: bool = True, boot: int = 0, seed: int = 42,
                 name: str = "latent_horizon"):
        self.metrics = list(metrics)
        self.predictions = list(predictions)
        self.champion_path = champion_path
        self.gap_head_path = gap_head_path
        self.per_horizon = per_horizon
        self.boot = boot
        self.seed = seed
        self.name = name

    def _ctx(self):
        ctx = {}
        if "champ" in self.predictions:
            ctx["champ"] = load_1step_predictor(self.champion_path)
        if "gap_head" in self.predictions:
            ctx["gap_head"] = load_gap_head(self.gap_head_path)
        return ctx

    def _paired(self, err_a, err_p, pids):
        if self.boot <= 0:
            d = err_a.mean().item() - err_p.mean().item()
            return (d, float("nan"), float("nan"), "n.s.")
        by_a, by_p = {}, {}
        for pid, a, p in zip(pids, err_a.tolist(), err_p.tolist()):
            by_a.setdefault(pid, []).append(a)
            by_p.setdefault(pid, []).append(p)
        uniq = sorted(by_a)
        rng = random.Random(self.seed)
        diffs = []
        for _ in range(self.boot):
            samp = [rng.choice(uniq) for _ in uniq]
            a = [x for pid in samp for x in by_a[pid]]
            p = [x for pid in samp for x in by_p[pid]]
            diffs.append(statistics.mean(a) - statistics.mean(p))
        lo, hi = percentile_ci(diffs)
        sig = "SIG" if hi < 0 or lo > 0 else "n.s."
        return (err_a.mean().item() - err_p.mean().item(), lo, hi, sig)

    def run(self, patients: dict, pids=None) -> LatentResult:
        rows = build_pairs(patients, pids=pids)
        if not rows:
            return LatentResult()
        ctx = self._ctx()
        targets = torch.stack([r.z_tgt for r in rows])
        errs = {}
        for m in self.predictions:
            errs[m] = cosine_error(PREDICTIONS[m](rows, ctx), targets)
        if "persistence" not in errs:
            errs["persistence"] = cosine_error(
                torch.stack([r.z_t for r in rows]), targets)

        overall = {m: e.mean().item() for m, e in errs.items()}
        per_horizon, counts = {}, {}
        if self.per_horizon:
            ns = torch.tensor([r.n for r in rows])
            for n in sorted(set(ns.tolist())):
                msk = ns == n
                counts[n] = int(msk.sum())
                for m, e in errs.items():
                    per_horizon[(n, m)] = e[msk].mean().item()

        pairwise = {}
        base = errs["persistence"]
        pids_col = [r.pid for r in rows]
        for m, e in errs.items():
            if m == "persistence":
                continue
            pairwise[m] = self._paired(e, base, pids_col)

        return LatentResult(
            methods=list(errs), overall=overall, per_horizon=per_horizon,
            pairwise=pairwise, n_pairs=len(rows),
            n_patients=len(set(pids_col)), horizon_counts=counts)
