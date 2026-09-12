"""Locked evaluation protocols: cohort and fold definitions.

Wraps :mod:`src.data.eval_protocol` so the harness has a single entry point.
Adding a protocol is a registration; the evaluator consumes only the
``folds()`` / ``cohort()`` interface.
"""
from __future__ import annotations

from ..registry import PROTOCOLS
from ...data import eval_protocol as _ep


def load(path: str = _ep.DEFAULT_PROTOCOL) -> dict:
    return _ep.load_protocol(path)


def fold_patients(protocol: dict, i: int) -> list[str]:
    return _ep.fold_patients(protocol, i)


class LockedProtocol:
    """The committed ``info/eval_folds.json`` unseen-patient CV protocol."""

    name = "locked_unseen"

    def __init__(self, path: str = _ep.DEFAULT_PROTOCOL):
        self.path = path
        self.protocol = load(path)

    @property
    def k(self) -> int:
        return self.protocol["k"]

    def cohort(self, which: str = "unseen") -> list[str]:
        if which == "unseen":
            return sorted(self.protocol["folds"])
        return sorted(self.protocol[which])

    def folds(self, which: str = "unseen"):
        """Yield ``(train_pids, eval_pids)`` for each locked fold."""
        unseen = self.cohort("unseen")
        for i in range(self.k):
            te = fold_patients(self.protocol, i)
            tr = [p for p in unseen if p not in set(te)]
            if which == "unseen":
                yield tr, te
            else:
                yield self.protocol["encoder_train"], te

    def encoder_train(self) -> list[str]:
        return list(self.protocol["encoder_train"])

    def splits(self, train_pool: str = "unseen", cohort: str = "unseen"):
        """List of ``(train_pids, eval_pids)`` for a train pool / cohort.

        ``train_pool='unseen'`` -> locked folds (readout trained within the
        unseen cohort); ``'train'`` -> transfer from encoder-train patients.
        """
        if train_pool == "unseen":
            return list(self.folds("unseen"))
        return [(self.encoder_train(), self.cohort(cohort))]


class HeroSplit:
    """The 65/13/13 encoder train/val/test patient split."""

    name = "hero_split"

    def __init__(self, cfg: dict):
        self.splits = _ep.hero_splits(cfg)

    def cohort(self, which: str = "test") -> list[str]:
        if which == "unseen":
            return sorted(self.splits["val"] + self.splits["test"])
        return sorted(self.splits[which])

    def folds(self, which: str = "test"):
        te = self.cohort(which)
        tr = [p for p in self.splits["train"] if p not in set(te)]
        yield tr, te

    def splits(self, train_pool: str = "hero", cohort: str = "test"):
        return [(self.cohort("train"), self.cohort(cohort))]


@PROTOCOLS.register("locked_unseen")
def _locked_unseen(path: str = _ep.DEFAULT_PROTOCOL) -> LockedProtocol:
    return LockedProtocol(path)


@PROTOCOLS.register("hero_split")
def _hero_split(cfg: dict) -> HeroSplit:
    return HeroSplit(cfg)
