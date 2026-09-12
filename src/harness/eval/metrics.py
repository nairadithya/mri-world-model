"""Composable metrics. Every metric returns ``{name: float}`` and declares
whether higher is better and what inputs it needs, so the evaluator can pair
any metric with any task/readout and attach uncertainty for free.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..registry import METRICS


# --- free functions (canonical implementations, imported by old scripts) ---

def macro_f1(pred: torch.Tensor, y: torch.Tensor, n_cls: int = 4) -> float:
    """Unweighted mean per-class F1 (identical to the legacy probe metric)."""
    f1s = []
    for k in range(n_cls):
        tp = int(((pred == k) & (y == k)).sum())
        fp = int(((pred == k) & (y != k)).sum())
        fn = int(((pred != k) & (y == k)).sum())
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f1s.append(2 * p * r / max(1e-9, p + r))
    return sum(f1s) / n_cls


def accuracy(pred: torch.Tensor, y: torch.Tensor) -> float:
    return (pred == y).float().mean().item()


def per_class_recall(pred, y, n_cls: int = 4, names=None) -> dict[str, float]:
    names = names or [f"class_{k}" for k in range(n_cls)]
    out = {}
    for k in range(n_cls):
        tp = int(((pred == k) & (y == k)).sum())
        fn = int(((pred != k) & (y == k)).sum())
        out[f"recall_{names[k]}"] = tp / max(1, tp + fn)
    return out


def confusion_matrix(pred, y, n_cls: int = 4) -> torch.Tensor:
    cm = torch.zeros(n_cls, n_cls, dtype=torch.long)
    for k in range(n_cls):
        for j in range(n_cls):
            cm[k, j] = int(((y == k) & (pred == j)).sum())
    return cm


def auc_mann_whitney(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """P(score_pos > score_neg) via rank sum. labels binary {0,1}."""
    pos = scores[labels == 1].sort().values
    neg = scores[labels == 0].sort().values
    order = torch.argsort(torch.cat([neg, pos]), stable=True).float() + 1
    n0, n1 = len(neg), len(pos)
    r1 = order[n0:].sum().item()
    return (r1 - n1 * (n1 + 1) / 2) / (n0 * n1)


def cosine_error(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """1 - cosine(a, b) along the last dim, per row. Representation error."""
    return 1 - F.cosine_similarity(a, b, dim=-1)


def persistence_error(z_t: torch.Tensor, z_u: torch.Tensor) -> torch.Tensor:
    """Representation error of the persistence predictor (z_hat = z_t)."""
    return cosine_error(z_t, z_u)


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    ss_res = ((y_true - y_pred) ** 2).sum().item()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum().item()
    return 1 - ss_res / max(1e-9, ss_tot)


def mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return (y_true - y_pred).abs().mean().item()


def mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return ((y_true - y_pred) ** 2).mean().item()


# --- registry -------------------------------------------------------------

class Metric:
    name = "metric"
    higher_is_better = True
    requires = "labels"  # labels | scores | latents

    def compute(self, y_true, y_pred, *, scores=None, **ctx) -> dict[str, float]:
        raise NotImplementedError


@METRICS.register("accuracy")
class AccuracyMetric(Metric):
    name = "accuracy"

    def compute(self, y_true, y_pred, *, scores=None, **ctx):
        return {self.name: accuracy(y_pred, y_true)}


@METRICS.register("macro_f1")
class MacroF1Metric(Metric):
    name = "macro_f1"

    def compute(self, y_true, y_pred, *, scores=None, n_cls=4, **ctx):
        return {self.name: macro_f1(y_pred, y_true, n_cls)}


@METRICS.register("per_class_recall")
class PerClassRecallMetric(Metric):
    name = "per_class_recall"

    def compute(self, y_true, y_pred, *, scores=None, names=None, n_cls=4, **ctx):
        return per_class_recall(y_pred, y_true, n_cls=n_cls, names=names)


@METRICS.register("auc")
class AucMetric(Metric):
    name = "auc"
    requires = "scores"

    def compute(self, y_true, y_pred, *, scores=None, **ctx):
        if scores is None:
            raise ValueError("auc requires continuous scores")
        return {self.name: auc_mann_whitney(scores, y_true)}


@METRICS.register("cosine_error")
class CosineErrorMetric(Metric):
    name = "cosine_error"
    requires = "latents"
    higher_is_better = False

    def compute(self, y_true, y_pred, *, scores=None, **ctx):
        return {self.name: cosine_error(y_pred, y_true).mean().item()}


@METRICS.register("persistence_error")
class PersistenceErrorMetric(Metric):
    name = "persistence_error"
    requires = "latents"
    higher_is_better = False

    def compute(self, y_true, y_pred, *, scores=None, **ctx):
        return {self.name: persistence_error(y_pred, y_true).mean().item()}


@METRICS.register("r2")
class R2Metric(Metric):
    name = "r2"
    requires = "latents"

    def compute(self, y_true, y_pred, *, scores=None, **ctx):
        yt = y_true.reshape(-1)
        yp = y_pred.reshape(-1)
        return {self.name: r2_score(yt, yp)}


@METRICS.register("mae")
class MaeMetric(Metric):
    name = "mae"
    requires = "latents"
    higher_is_better = False

    def compute(self, y_true, y_pred, *, scores=None, **ctx):
        return {self.name: mae(y_true.reshape(-1), y_pred.reshape(-1))}


def compute_metrics(names, y_true, y_pred, *, scores=None, **ctx) -> dict[str, float]:
    """Run a sequence of registered metrics and merge their outputs."""
    out: dict[str, float] = {}
    for name in names:
        metric = METRICS.get(name)
        if isinstance(metric, type):
            metric = metric()
        out.update(metric.compute(y_true, y_pred, scores=scores, **ctx))
    return out
