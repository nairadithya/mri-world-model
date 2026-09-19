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


def _binary_scores(scores: torch.Tensor) -> torch.Tensor:
    """Convert binary logits/probabilities to one positive-class score."""
    scores = scores.detach().float().reshape(-1, scores.shape[-1] if scores.ndim > 1 else 1)
    if scores.shape[1] == 1:
        return scores[:, 0]
    if scores.shape[1] == 2:
        return scores.softmax(dim=1)[:, 1]
    raise ValueError(f"binary metric received {scores.shape[1]} score columns")


def auc_mann_whitney(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Exact binary ROC-AUC with average ranks for ties.

    The old implementation used positions from ``argsort`` as ranks, which
    is wrong whenever positive and negative observations are interleaved and
    assigns tied scores different ranks.  This is the Mann--Whitney
    definition: P(score_pos > score_neg) + 0.5 P(tie).
    """
    scores = _binary_scores(scores).reshape(-1)
    labels = labels.reshape(-1).long()
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    n0, n1 = len(neg), len(pos)
    if n0 == 0 or n1 == 0:
        return float("nan")
    vals, order = torch.sort(torch.cat([neg, pos]))
    ranks = torch.arange(1, len(vals) + 1, dtype=torch.float32)
    # Assign each equal-score block its average rank.
    starts = torch.ones(len(vals), dtype=torch.bool)
    if len(vals) > 1:
        starts[1:] = vals[1:] != vals[:-1]
    group = starts.cumsum(0) - 1
    counts = torch.bincount(group)
    sums = torch.zeros_like(counts, dtype=torch.float32)
    sums.scatter_add_(0, group, ranks)
    avg = sums[group] / counts[group].float()
    original_ranks = torch.empty_like(avg)
    original_ranks[order] = avg
    r1 = original_ranks[n0:].sum().item()
    return (r1 - n1 * (n1 + 1) / 2) / (n0 * n1)


def auprc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Average precision for binary labels, using descending score thresholds."""
    s = _binary_scores(scores).reshape(-1)
    y = labels.reshape(-1).long()
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = torch.argsort(s, descending=True, stable=True)
    yy = (y[order] == 1).float()
    tp = yy.cumsum(0)
    fp = torch.arange(1, len(y) + 1, dtype=torch.float32) - tp
    precision = tp / (tp + fp).clamp_min(1.0)
    recall_delta = yy / n_pos
    return float((precision * recall_delta).sum())


def brier_score(scores: torch.Tensor, labels: torch.Tensor) -> float:
    p = _binary_scores(scores)
    y = labels.reshape(-1).float()
    return float(((p - y) ** 2).mean())


def log_loss(scores: torch.Tensor, labels: torch.Tensor) -> float:
    logits = scores.reshape(-1) if scores.ndim == 1 else scores
    y = labels.reshape(-1).long()
    if logits.ndim == 1:
        return float(torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), y.float()))
    return float(torch.nn.functional.cross_entropy(logits.float(), y))


def expected_calibration_error(scores: torch.Tensor, labels: torch.Tensor,
                              bins: int = 10) -> float:
    p = _binary_scores(scores)
    y = labels.reshape(-1).float()
    conf, pred = torch.maximum(p, 1 - p), (p >= 0.5).float()
    out = torch.zeros((), dtype=torch.float32)
    for lo, hi in zip(torch.linspace(0, 1, bins + 1)[:-1],
                      torch.linspace(0, 1, bins + 1)[1:]):
        keep = (conf >= lo) & ((conf < hi) if hi < 1 else (conf <= hi))
        if keep.any():
            out += keep.float().mean() * (pred[keep] == y[keep]).float().mean().sub(conf[keep].mean()).abs()
    return float(out)


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


@METRICS.register("sensitivity")
class SensitivityMetric(Metric):
    name = "sensitivity"

    def compute(self, y_true, y_pred, *, scores=None, **ctx):
        return {self.name: per_class_recall(y_pred, y_true, n_cls=2)["recall_class_1"]}


@METRICS.register("specificity")
class SpecificityMetric(Metric):
    name = "specificity"

    def compute(self, y_true, y_pred, *, scores=None, **ctx):
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        return {self.name: tn / max(1, tn + fp)}


@METRICS.register("prevalence")
class PrevalenceMetric(Metric):
    name = "prevalence"

    def compute(self, y_true, y_pred, *, scores=None, **ctx):
        return {self.name: (y_true == 1).float().mean().item()}


@METRICS.register("auc")
class AucMetric(Metric):
    name = "auc"
    requires = "scores"

    def compute(self, y_true, y_pred, *, scores=None, **ctx):
        if scores is None:
            raise ValueError("auc requires continuous scores")
        return {self.name: auc_mann_whitney(scores, y_true)}


@METRICS.register("auprc")
class AuprcMetric(Metric):
    name = "auprc"
    requires = "scores"

    def compute(self, y_true, y_pred, *, scores=None, **ctx):
        if scores is None:
            raise ValueError("auprc requires continuous scores")
        return {self.name: auprc(scores, y_true)}


@METRICS.register("brier")
class BrierMetric(Metric):
    name = "brier"
    requires = "scores"
    higher_is_better = False

    def compute(self, y_true, y_pred, *, scores=None, **ctx):
        if scores is None:
            raise ValueError("brier requires continuous scores")
        return {self.name: brier_score(scores, y_true)}


@METRICS.register("logloss")
class LogLossMetric(Metric):
    name = "logloss"
    requires = "scores"
    higher_is_better = False

    def compute(self, y_true, y_pred, *, scores=None, **ctx):
        if scores is None:
            raise ValueError("logloss requires continuous scores")
        return {self.name: log_loss(scores, y_true)}


@METRICS.register("ece")
class CalibrationMetric(Metric):
    name = "ece"
    requires = "scores"
    higher_is_better = False

    def compute(self, y_true, y_pred, *, scores=None, **ctx):
        if scores is None:
            raise ValueError("ece requires continuous scores")
        return {self.name: expected_calibration_error(scores, y_true)}


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
