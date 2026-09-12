"""Frozen readouts: linear/MLP probes and ridge regression.

Canonical implementations (``fit_linear`` / ``scores`` are byte-compatible
with the legacy probe) plus standardisation helpers used for cross-space
comparison. These are the ``readout`` training methods in the harness.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..data.tasks import RANO_PROBE_NAMES
from .base import Method, register_method


def fit_linear(x_tr: torch.Tensor, y_tr: torch.Tensor, n_cls: int = 4,
               hidden: int = 0, steps: int = 500, lr: float = 1e-2,
               seed: int = 42) -> nn.Module:
    """Class-weighted Adam probe (legacy ``probe_rano.fit_linear``)."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    if hidden:
        net = nn.Sequential(nn.Linear(x_tr.shape[1], hidden), nn.GELU(),
                            nn.Linear(hidden, n_cls))
    else:
        net = nn.Linear(x_tr.shape[1], n_cls)
    counts = torch.bincount(y_tr, minlength=n_cls).float().clamp_min(1)
    weight = counts.sum() / (n_cls * counts)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(steps):
        idx = torch.randperm(len(x_tr), generator=g)
        opt.zero_grad()
        loss = nn.functional.cross_entropy(net(x_tr[idx]), y_tr[idx], weight=weight)
        loss.backward()
        opt.step()
    return net


def scores(net: nn.Module, x: torch.Tensor, y: torch.Tensor):
    """(acc, macro_f1, per-class recall dict, confusion) — legacy signature."""
    with torch.no_grad():
        pred = net(x).argmax(1)
    acc = (pred == y).float().mean().item()
    f1s, recs, cm = [], [], torch.zeros(4, 4, dtype=torch.long)
    for k in range(4):
        tp = int(((pred == k) & (y == k)).sum())
        fp = int(((pred == k) & (y != k)).sum())
        fn = int(((pred != k) & (y == k)).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1s.append(2 * prec * rec / max(1e-9, prec + rec))
        recs.append(rec)
        for j in range(4):
            cm[k, j] = int(((y == k) & (pred == j)).sum())
    return acc, sum(f1s) / 4, dict(zip(RANO_PROBE_NAMES, recs)), cm


def standardize(Xtr: torch.Tensor, Xte: torch.Tensor):
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    return (Xtr - mu) / sd, (Xte - mu) / sd, mu, sd


def fit_ridge_cv(x_tr: torch.Tensor, y_tr: torch.Tensor,
                 lams=(0.1, 1.0, 10.0, 100.0, 1000.0), seed: int = 42):
    """Ridge with standardized features; lambda picked by 5-fold CV (legacy)."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(x_tr), generator=g)
    folds = [idx[i::5] for i in range(5)]
    mu, sd = x_tr.mean(0), x_tr.std(0).clamp_min(1e-6)
    z = (x_tr - mu) / sd
    d = z.shape[1]
    I = torch.eye(d)
    best, best_mse = lams[0], float("inf")
    for lam in lams:
        mses = []
        for i in range(5):
            te, tr = folds[i], torch.cat([folds[j] for j in range(5) if j != i])
            w = torch.linalg.solve(z[tr].T @ z[tr] + lam * I,
                                   z[tr].T @ y_tr[tr].unsqueeze(1))
            pred = (z[te] @ w).squeeze(1)
            mses.append(((y_tr[te] - pred) ** 2).mean().item())
        m = sum(mses) / 5
        if m < best_mse:
            best, best_mse = lam, m
    w = torch.linalg.solve(z.T @ z + best * I, z.T @ y_tr.unsqueeze(1)).squeeze(1)
    b = y_tr.mean() - (mu / sd * w).sum()
    return w / sd, b, best


def fit_clf(X: torch.Tensor, y: torch.Tensor, seed: int = 0, steps: int = 400,
            lr: float = 1e-2) -> nn.Module:
    """Class-weighted full-batch linear classifier (legacy cross-site probe)."""
    torch.manual_seed(seed)
    net = nn.Linear(X.shape[1], 4)
    counts = torch.bincount(y, minlength=4).float().clamp_min(1)
    w = counts.sum() / (4 * counts)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        nn.functional.cross_entropy(net(X), y, weight=w).backward()
        opt.step()
    return net


@register_method("readout", role="readout", consumes=("features",), produces=())
class ReadoutMethod(Method):
    """Frozen probe/readout; fitting happens inside the evaluator per fold.

    Registered so the unified CLI lists it and so a future native ``train``
    path can reuse the same implementation the evaluator uses.
    """

    methods = ("linear", "mlp", "ridge")

    def fit(self, ctx):
        raise NotImplementedError(
            "readouts are fit per protocol fold by ReadoutEvaluator")

