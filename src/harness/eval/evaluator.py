"""Evaluator: task x protocol x readout x metrics x aggregator, per patient.

This is the composition point that makes evaluation as modular as training:
swap any one axis without touching the others. Readouts are trained inside
each protocol fold, predictions are pooled per patient, and any label metric
gets a patient-cluster bootstrap CI for free.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .aggregate import patient_bootstrap, percentile_ci
from .metrics import compute_metrics, macro_f1
from ..data.tasks import Task
from ..train.readout import fit_linear, fit_ridge_cv


@dataclass
class EvalResult:
    name: str
    metrics: dict = field(default_factory=dict)
    ci: dict = field(default_factory=dict)
    oof: dict = field(default_factory=dict)
    fold_values: list = field(default_factory=list)
    n_pat: int = 0
    n_rows: int = 0
    majority: dict | None = None
    meta: dict = field(default_factory=dict)

    def scalar(self, name: str) -> float:
        return self.metrics.get(name, float("nan"))


class _Standardized:
    """Fold train-stat standardization folded into the predictor (no leakage)."""

    def __init__(self, net: nn.Module, mu: torch.Tensor, sd: torch.Tensor):
        self.net, self.mu, self.sd = net, mu, sd

    def forward(self, x):
        return self.net((x - self.mu) / self.sd)


class ReadoutEvaluator:
    def __init__(self, task: Task, protocol, *, view: str | None = None,
                 readout: str = "mlp", hidden: int = 256, seed: int = 42,
                 standardize: bool = False, train_pool: str = "unseen",
                 cohort: str = "unseen", metrics=None, boot: int = 10000,
                 metric_seed: int = 42, name: str | None = None):
        self.task = task
        self.protocol = protocol
        self.view = view or task.default_view
        self.readout = readout
        self.hidden = hidden
        self.seed = seed
        self.standardize = standardize
        self.train_pool = train_pool
        self.cohort = cohort
        self.metrics = list(metrics or ["macro_f1", "accuracy"])
        self.boot = boot
        self.metric_seed = metric_seed
        self.name = name or f"{task.name}[{self.view}]"

    # ---------------------------------------------------------------- rows --
    def _rows(self, patients, pids):
        out = {}
        for pid in pids:
            if pid not in patients:
                continue
            r = self.task.rows(patients, [pid], self.view)
            if r is None:
                continue
            X, y = r
            X, y = self.task.filter(X, y)
            out[pid] = (X, self.task.transform(y))
        return out

    @property
    def n_cls(self) -> int:
        if self.task.frame == "binary":
            return 2
        return 4

    def _fit(self, X: torch.Tensor, y: torch.Tensor):
        if self.readout == "ridge":
            w, b, lam = fit_ridge_cv(X, y, seed=self.seed)
            return ("ridge", w, b, lam)
        hidden = self.hidden if self.readout == "mlp" else 0
        mu = sd = None
        Xin = X
        if self.standardize:
            mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
            Xin = (X - mu) / sd
        net = fit_linear(Xin, y, n_cls=self.n_cls, hidden=hidden, seed=self.seed)
        return ("net", net, mu, sd)

    def _predict(self, model, X: torch.Tensor):
        if model[0] == "ridge":
            _, w, b, _ = model
            return X @ w + b
        _, net, mu, sd = model
        with torch.no_grad():
            z = (X - mu) / sd if mu is not None else X
            logits = net(z)
        return logits.argmax(1), logits

    # ---------------------------------------------------------------- run ---
    def run(self, patients: dict, *, verbose: bool = False) -> EvalResult:
        oof = {}
        fold_values = []
        for fold, (tr, te) in enumerate(self.protocol.splits(self.train_pool,
                                                             self.cohort)):
            tr_rows = self._rows(patients, tr)
            te_rows = self._rows(patients, te)
            if not tr_rows or not te_rows:
                continue
            Xtr = torch.cat([v[0] for v in tr_rows.values()])
            ytr = torch.cat([v[1] for v in tr_rows.values()])
            model = self._fit(Xtr, ytr)
            fold_pred, fold_y = [], []
            for pid, (Xte, yte) in te_rows.items():
                pred = self._predict(model, Xte)
                if isinstance(pred, tuple):
                    pred = pred[0]
                oof[pid] = (pred, yte)
                fold_pred.append(pred)
                fold_y.append(yte)
            if fold_pred:
                fold_values.append(self._score(torch.cat(fold_pred),
                                               torch.cat(fold_y)))
            if verbose:
                print(f"  fold {fold}: n_te={len(te_rows)}", flush=True)
        if not oof:
            return EvalResult(name=self.name)

        pids = sorted(oof)
        pred = torch.cat([oof[p][0] for p in pids])
        y = torch.cat([oof[p][1] for p in pids])
        primary = self.metrics[0]
        metrics = compute_metrics(self.metrics, y, pred, n_cls=self.n_cls)
        ci = {}
        if self.boot and primary in ("macro_f1", "accuracy"):
            vals, _ = patient_bootstrap(
                oof, None, boot=self.boot, seed=self.metric_seed,
                metric_fn=lambda p, t: macro_f1(p, t, self.n_cls))
            ci[primary] = percentile_ci(vals)
        majority = None
        if self.task.frame != "latent":
            maj = int(torch.bincount(y, minlength=self.n_cls).argmax())
            majority = {"class": maj,
                        "accuracy": (y == maj).float().mean().item()}
        return EvalResult(
            name=self.name, metrics=metrics, ci=ci, oof=oof,
            fold_values=fold_values, n_pat=len(oof), n_rows=len(y),
            majority=majority,
            meta={"view": self.view, "readout": self.readout,
                  "train_pool": self.train_pool, "cohort": self.cohort,
                  "folds": len(fold_values), "n_cls": self.n_cls,
                  "class_names": list(self.task.class_names)})

    def _score(self, pred, y) -> float:
        return compute_metrics(self.metrics[:1], y, pred,
                               n_cls=self.n_cls)[self.metrics[0]]
