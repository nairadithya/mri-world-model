"""Self-contained harness tests (no pytest dependency).

Run:  .venv/bin/python tests/test_harness.py
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.harness import METHODS, METRICS, PROTOCOLS, TASKS  # noqa: E402
from src.harness.data.tasks import (  # noqa: E402
    RANO_PROBE_NAMES, patient_rows, rows_for,
)
from src.harness.encode import cache as cache_mod  # noqa: E402
from src.harness.encode import views  # noqa: E402
from src.harness.eval.aggregate import patient_bootstrap, percentile_ci  # noqa: E402
from src.harness.eval.evaluator import ReadoutEvaluator  # noqa: E402
from src.harness.eval.latent import LatentPairEvaluator, build_pairs  # noqa: E402
from src.harness.eval.metrics import (  # noqa: E402
    accuracy, auc_mann_whitney, cosine_error, macro_f1, r2_score,
)
from src.harness import paths  # noqa: E402

PASSED, FAILED = [], []


def check(name, fn):
    try:
        fn()
        PASSED.append(name)
    except Exception as e:  # noqa: BLE001
        FAILED.append((name, e))


def test_registries():
    assert "macro_f1" in METRICS
    assert "rano4_forecast" in TASKS
    assert "locked_unseen" in PROTOCOLS


def test_metric_values():
    pred = torch.tensor([0, 1, 2, 3, 0, 1])
    y = torch.tensor([0, 1, 2, 3, 0, 1])
    assert abs(macro_f1(pred, y) - 1.0) < 1e-9
    assert abs(accuracy(pred, y) - 1.0) < 1e-9
    # imperfect: class 2 never predicted -> its F1 0; class 1 has 1 FP -> 0.8
    pred2 = torch.tensor([0, 1, 1, 3, 0, 1])
    y2 = torch.tensor([0, 1, 2, 3, 0, 1])
    assert abs(macro_f1(pred2, y2) - 0.7) < 1e-9


def test_auc():
    s = torch.tensor([0.1, 0.2, 0.8, 0.9])
    lab = torch.tensor([0, 0, 1, 1])
    assert abs(auc_mann_whitney(s, lab) - 1.0) < 1e-9


def test_latent_metrics():
    a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    b = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    err = cosine_error(a, b)
    assert abs(err[0].item()) < 1e-6
    assert abs(err[1].item() - 1.0) < 1e-6
    y = torch.tensor([1.0, 2.0, 3.0])
    assert abs(r2_score(y, y.clone()) - 1.0) < 1e-9


def _cache():
    return {
        "patients": {
            "a": {"split": "train", "labels": torch.tensor([0, 1, 2]),
                  "fused": torch.randn(3, 5, generator=torch.Generator().manual_seed(1))},
            "b": {"split": "train", "labels": torch.tensor([1, 2, 3]),
                  "fused": torch.randn(3, 5, generator=torch.Generator().manual_seed(2))},
            "c": {"split": "test", "labels": torch.tensor([0, 3]),
                  "fused": torch.randn(2, 5, generator=torch.Generator().manual_seed(3))},
        }
    }


def test_legacy_task_rows():
    pts = _cache()["patients"]
    splits = {"tr": ["a", "b"], "te": ["c"]}
    X, y = rows_for(pts, splits, "tr", "fused")
    assert X.shape == (6, 5) and len(y) == 6
    r = patient_rows(pts, "c", "fused")
    assert r is not None and r[0].shape == (2, 5)


def test_canonicalize_and_views():
    cache = _cache()
    can = cache_mod.canonicalize(cache)
    p = can["patients"]["a"]
    assert "fused" in p["features"]
    assert views.get(p, "fused").shape == (3, 5)
    # views.get must also read the legacy flat layout
    assert views.get(cache["patients"]["a"], "fused").shape == (3, 5)
    assert cache_mod.validate(can) == []


def test_bootstrap_reproducible():
    oof = {"a": (torch.tensor([0, 1]), torch.tensor([0, 1])),
           "b": (torch.tensor([2]), torch.tensor([2]))}
    a1, _ = patient_bootstrap(oof, None, boot=100, seed=1)
    a2, _ = patient_bootstrap(oof, None, boot=100, seed=1)
    assert a1 == a2
    lo, hi = percentile_ci(a1)
    assert lo <= hi


class _StubProtocol:
    def splits(self, train_pool, cohort):
        return [(["a", "b"], ["c"])]


def test_evaluator_composes():
    task = TASKS.get("rano4_current")()
    ev = ReadoutEvaluator(task, _StubProtocol(), view="fused", readout="linear",
                          metrics=["macro_f1", "accuracy"], boot=0)
    res = ev.run(_cache()["patients"])
    assert res.n_pat == 1 and "macro_f1" in res.metrics
    assert res.meta["class_names"] == RANO_PROBE_NAMES


def test_frozen_paths():
    assert paths.CHAMPION == "checkpoints/champion_0.0081.pt"
    assert paths.PROBE_CACHE == "checkpoints/probe_cache.pt"
    assert paths.INTERFACE_CACHE == "checkpoints/interface_cache.pt"
    assert paths.HORIZON_CACHE == "checkpoints/horizon_cache.pt"
    assert paths.FIELD_CACHE == "checkpoints/field_cache.pt"
    assert paths.FIELD_MODELS == "checkpoints/field_models.pt"
    assert paths.PROBE_HEAD == "checkpoints/probe_head.pt"
    assert paths.EVAL_FOLDS == "info/eval_folds.json"


def _latent_cache():
    g = torch.Generator().manual_seed(7)
    T = 4
    z = torch.randn(T, 8, generator=g)
    states = torch.randn(T - 1, 8, generator=g)
    return {"patients": {"a": {
        "split": "train", "z": z, "states": states,
        "deltas": torch.tensor([0.0, 10.0, 10.0, 10.0]),
        "has_img": torch.ones(T, dtype=torch.bool)}}}


def test_latent_pairs_and_eval():
    patients = _latent_cache()["patients"]
    rows = build_pairs(patients)
    assert len(rows) == 6  # (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
    res = LatentPairEvaluator(predictions=("persistence",), boot=0).run(patients)
    p = patients["a"]
    expected = cosine_error(torch.stack([r.z_t for r in rows]),
                            torch.stack([r.z_tgt for r in rows])).mean().item()
    assert res.n_pairs == 6
    assert abs(res.overall["persistence"] - expected) < 1e-9
    assert (1, "persistence") in res.per_horizon


def test_native_jepa_registered():
    assert "jepa" in METHODS
    assert getattr(METHODS.get("jepa"), "entry", None) is not None


def main():
    check("registries", test_registries)
    check("metric_values", test_metric_values)
    check("auc", test_auc)
    check("latent_metrics", test_latent_metrics)
    check("legacy_task_rows", test_legacy_task_rows)
    check("canonicalize_and_views", test_canonicalize_and_views)
    check("bootstrap_reproducible", test_bootstrap_reproducible)
    check("evaluator_composes", test_evaluator_composes)
    check("frozen_paths", test_frozen_paths)
    check("latent_pairs_and_eval", test_latent_pairs_and_eval)
    check("native_jepa_registered", test_native_jepa_registered)
    print(f"PASS {len(PASSED)}/{len(PASSED) + len(FAILED)}")
    for name, err in FAILED:
        print(f"FAIL {name}: {type(err).__name__}: {err}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
