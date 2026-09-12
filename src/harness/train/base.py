"""Method protocol + registry.

A method is anything the harness can *train* or *fit*. Two roles matter for
composition:

- ``ssl`` / ``supervised`` / ``dynamics``: learns a representation, exposed as
  named views (``produces``) into a cache.
- ``readout``: consumes feature views (``consumes``) and predicts a target.

The unified CLI trains a method; the evaluator scores a readout. Both are
registered here so new approaches slot in without touching the driver.
"""
from __future__ import annotations

from ..registry import METHODS


class Method:
    name = "method"
    role = "ssl"            # ssl | supervised | dynamics | readout
    produces: tuple = ()
    consumes: tuple = ("mri",)

    def fit(self, ctx):
        """Train/fit and return an artifact (or write a cache)."""
        raise NotImplementedError

    def predict(self, ctx, artifact):
        raise NotImplementedError


def register_method(name: str, **meta):
    def deco(cls):
        cls.name = name
        for k, v in meta.items():
            setattr(cls, k, v)
        return METHODS.add(name, cls)
    return deco
