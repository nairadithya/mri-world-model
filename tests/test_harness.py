"""Self-contained harness tests (no pytest dependency).

Run:  .venv/bin/python tests/test_harness.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.harness import METHODS, METRICS, PROTOCOLS, TASKS  # noqa: E402
from src.harness.data.tasks import (  # noqa: E402
    RANO_PROBE_NAMES, patient_rows, rows_for,
)
from src.harness.data.manifest import assert_compatible, build_manifest  # noqa: E402
from src.harness.data.anatomy_manifest import (  # noqa: E402
    COMPARTMENTS, build_pairs as build_anatomy_pairs, compare_masks,
    measure_label_map, measure_mask,
)
from src.harness.analysis.anatomy_baselines import (  # noqa: E402
    _metrics as anatomy_metrics,
)
from src.harness.checkpoints import inspect as inspect_checkpoint, load_model  # noqa: E402
from src.harness.cli import main as harness_main  # noqa: E402
from src.harness.encode.anatomy import (  # noqa: E402
    feature_names as anatomy_feature_names, history_feature_names,
    history_vector, visit_features,
)
from src.harness.analysis.anatomy_residual import ResidualMLP  # noqa: E402
from src.harness.analysis.lesion_coverage import (  # noqa: E402
    patch_occupancy, support_stats,
)
from src.model.lesion_tokens import (  # noqa: E402
    lesion_centered_crop, pool_region_tokens, region_patch_weights,
)
from src.model.lesion_forecaster import LesionTransitionForecaster  # noqa: E402
from src.harness.data.anatomy_representations import representation_rows  # noqa: E402
from src.data.labels import response_label, response_valid  # noqa: E402
from src.harness.encode import cache as cache_mod  # noqa: E402
from src.harness.encode import views  # noqa: E402
from src.harness.eval.aggregate import patient_bootstrap, percentile_ci  # noqa: E402
from src.harness.eval.evaluator import ReadoutEvaluator  # noqa: E402
from src.harness.train.readout import fit_ridge_cv  # noqa: E402
from src.harness.eval.latent import LatentPairEvaluator, build_pairs  # noqa: E402
from src.harness.eval.metrics import (  # noqa: E402
    accuracy, auc_mann_whitney, auprc, brier_score, compute_metrics,
    cosine_error, macro_f1, r2_score,
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
    interleaved = torch.tensor([0.2, 0.4, 0.1, 0.3])
    assert abs(auc_mann_whitney(interleaved, lab) - 0.25) < 1e-9
    assert abs(auc_mann_whitney(interleaved.flip(0), lab) - 0.75) < 1e-9
    assert abs(auc_mann_whitney(torch.ones(4), lab) - 0.5) < 1e-9
    assert abs(auprc(s, lab) - 1.0) < 1e-9
    assert abs(brier_score(torch.tensor([0.5, 0.5, 0.5, 0.5]), lab) - 0.25) < 1e-9
    assert torch.isnan(torch.tensor(auc_mann_whitney(torch.ones(2), torch.zeros(2, dtype=torch.long))))
    assert torch.isnan(torch.tensor(auc_mann_whitney(torch.empty(0), torch.empty(0, dtype=torch.long))))
    out = compute_metrics(["sensitivity", "specificity", "prevalence"],
                          lab, torch.tensor([0, 1, 1, 0]))
    assert out["sensitivity"] == 0.5 and out["specificity"] == 0.5
    assert out["prevalence"] == 0.5


def test_label_contract_and_manifest():
    assert response_label(" Post-Op ") == -1
    assert response_label("PD") == 0
    assert not response_valid("missing")
    a = {"patients": {"p": {
        "visits": ["v0", "v1"], "labels": torch.tensor([-1, 0]),
        "has_img": torch.tensor([True, True]),
        "deltas": torch.tensor([0.0, 14.0]),
    }}}
    b = build_manifest([a], source="test", tasks=("assessment", "forecast"))
    assert len(b["tasks"]["assessment"]) == 1
    assert b["tasks"]["assessment"][0]["allowed_inputs"] == ["history_through_t", "x_target"]
    assert_compatible({"a": b, "b": b}, "forecast")


def test_anatomy_measurement_and_pairs():
    import nibabel as nib
    import numpy as np

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "mask.nii.gz")
        data = np.zeros((2, 3, 4), dtype=np.uint8)
        data[0, 0, :2] = 1
        nib.save(nib.Nifti1Image(data, np.diag([2.0, 3.0, 4.0, 1.0])), path)
        got = measure_mask(path)
        assert got["foreground_voxels"] == 2
        assert abs(got["voxel_volume_mm3"] - 24.0) < 1e-9
        assert abs(got["volume_mm3"] - 48.0) < 1e-9
        assert got["geometry_valid"] and got["finite"]
        other = os.path.join(td, "other.nii.gz")
        nib.save(nib.Nifti1Image(data.copy(), np.diag([2.0, 3.0, 4.0, 1.0])), other)
        comparison = compare_masks(path, other)
        assert comparison["same_grid"] and comparison["dice"] == 1.0
        assert comparison["volume_ratio_cl_to_onco"] == 1.0
        labelled = data.copy()
        labelled[0, 0, 0] = 2
        label_path = os.path.join(td, "labels.nii.gz")
        nib.save(nib.Nifti1Image(labelled, np.diag([2.0, 3.0, 4.0, 1.0])),
                 label_path)
        measured = measure_label_map(label_path, {"a": 1, "b": 2})
        assert all(abs(measured["values_mm3"][k] - 24.0) < 1e-9
                   for k in ("a", "b"))
        assert measured["geometry_valid"]

    measurement = {
        "values_mm3": {k: 1.0 for k in COMPARTMENTS},
        "geometry_verified": True,
    }
    visits = [{"cohort": "x", "patient_id": "p", "visit": f"v{i}",
               "visit_id": f"id{i}", "day": float(i * 10),
               "timing_verified": True, "measurement": measurement,
               "exclusions": []} for i in range(3)]
    pairs = build_anatomy_pairs(visits)
    assert len(pairs) == 2 and all(p["usable"] for p in pairs)
    assert all(p["gap_days"] == 10 for p in pairs)


def test_anatomy_metrics_are_patient_uniform():
    import numpy as np

    y = np.zeros((3, 3))
    pred = np.array([[1.0] * 3, [1.0] * 3, [3.0] * 3])
    groups = np.array(["many", "many", "one"])
    got = anatomy_metrics(y, pred, groups)
    assert abs(got["log_volume_mae"] - 5 / 3) < 1e-9
    assert abs(got["patient_uniform_log_volume_mae"] - 2.0) < 1e-9


def test_checkpoint_compatibility_loader():
    model = torch.nn.Linear(3, 2)
    expected = {k: v.detach().clone() for k, v in model.state_dict().items()}
    payload = {"model": {**{f"module.{k}": v + 1 for k, v in expected.items()},
                         "module.rano_heads.flat.weight": torch.ones(4, 4)},
               "epoch": 7, "val_loss": 0.25}
    _, report = load_model(model, payload)
    assert report.loaded == 2 and report.epoch == 7
    assert report.ignored_legacy == ["rano_heads.flat.weight"]
    assert not report.missing and not report.unexpected
    for key, value in model.state_dict().items():
        assert torch.equal(value, expected[key] + 1)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "legacy.pt")
        torch.save(payload, path)
        got = inspect_checkpoint(path)
        assert got["container"] == "model" and got["legacy_tensor_count"] == 1


def test_anatomy_first_cli_routes():
    harness_main(["list"])
    harness_main(["anatomy", "--help"])
    harness_main(["legacy", "--help"])


def test_physical_anatomy_features_are_history_only():
    import nibabel as nib
    import numpy as np

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "labels.nii.gz")
        data = np.zeros((4, 5, 6), dtype=np.uint8)
        data[0, 0, 0] = 1
        data[1:3, 1, 1] = 2
        data[1:3, 2:4, 2] = 3
        affine = np.diag([2.0, 3.0, 4.0, 1.0])
        nib.save(nib.Nifti1Image(data, affine), path)
        visit = {
            "visit_id": "v0", "modalities": {"CT1": True, "T1": False,
                                                  "T2": True, "FLAIR": True},
            "measurement": {
                "values_mm3": {"necrotic_non_enhancing": 48.0,
                               "enhancing": 24.0, "edema_flair": 96.0},
                "mask": {"path": path, "labels": {
                    "necrotic_non_enhancing": 2, "enhancing": 1,
                    "edema_flair": 3}},
            },
        }
        current = visit_features(visit)
    assert len(current) == len(anatomy_feature_names())
    assert np.isfinite(current).all()
    x1 = history_vector([current])
    changed_target = current.copy(); changed_target[:3] += 100
    x2 = history_vector([current])  # target is intentionally not an argument
    assert np.array_equal(x1, x2)
    assert len(x1) == len(history_feature_names())
    assert not np.array_equal(x1[:3], changed_target[:3])


def test_residual_model_starts_at_persistence():
    model = ResidualMLP(7, hidden=4)
    delta = model(torch.randn(5, 7))
    assert torch.equal(delta, torch.zeros_like(delta))


def test_lesion_patch_coverage_support():
    import numpy as np

    mask = np.zeros((96, 96, 96), dtype=np.uint8)
    mask[:16, :16, :16] = 1
    occupancy = patch_occupancy(mask)
    stats = support_stats(occupancy)
    assert occupancy.shape == (6, 6, 6)
    assert stats["nonzero_patches"] == 1
    assert stats["effective_patches"] == 1.0
    assert stats["max_patch_fraction"] == 1.0


def test_lesion_crop_and_region_pooling():
    images = torch.zeros(1, 2, 80, 80, 80)
    masks = torch.zeros(1, 3, 80, 80, 80, dtype=torch.bool)
    masks[:, 1, 60:64, 60:64, 60:64] = True
    images[:, :, 60:64, 60:64, 60:64] = 2
    crop_images, crop_masks = lesion_centered_crop(images, masks)
    assert crop_images.shape == (1, 2, 96, 96, 96)
    assert crop_masks.shape == (1, 3, 96, 96, 96)
    assert bool(crop_masks[:, 1].any())
    weights = region_patch_weights(crop_masks)
    assert weights.shape == (1, 4, 216)
    tokens = torch.arange(216, dtype=torch.float32)[None, None, :, None]
    pooled, contrast = pool_region_tokens(tokens, weights)
    assert pooled.shape == (1, 1, 4, 1)
    assert contrast.shape == (1, 1, 3, 1)


def test_lesion_forecaster_starts_at_persistence():
    model = LesionTransitionForecaster(input_dim=14, observation_dim=8, hidden_dim=8)
    source = torch.randn(4, 3)
    result = model(torch.randn(4, 14), torch.tensor([0.0, 7.0, 14.0, 21.0]), source)
    assert torch.equal(result["prediction"], source[:-1])


def test_anatomy_representation_join_is_row_exact():
    import json

    with tempfile.TemporaryDirectory() as td:
        feature_path = os.path.join(td, "features.pt")
        manifest_path = os.path.join(td, "manifest.json")
        rep_path = os.path.join(td, "reps.pt")
        names = history_feature_names()
        torch.save({"feature_version": "lesion-physical-v1",
                    "feature_names": names,
                    "pairs": {"row": {"cohort": "LUMIERE", "patient_id": "p",
                                        "source_visit_id": "visit-id",
                                        "x": torch.zeros(len(names)),
                                        "target": torch.zeros(3)}}}, feature_path)
        with open(manifest_path, "w") as f:
            json.dump({"visits": [{"visit_id": "visit-id", "visit": "week-000"}]}, f)
        torch.save({"patients": {"p": {
            "visits": ["week-000", "week-010"],
            "vision": torch.tensor([[1.0, 2.0], [9.0, 9.0]]),
            "states": torch.tensor([[3.0, 4.0]])}}}, rep_path)
        vision = representation_rows(feature_cache=feature_path,
                                     manifest_path=manifest_path,
                                     representation_cache=rep_path,
                                     cohort="LUMIERE", view="vision")
        state = representation_rows(feature_cache=feature_path,
                                    manifest_path=manifest_path,
                                    representation_cache=rep_path,
                                    cohort="LUMIERE", view="jepa_state")
        assert torch.equal(vision["row"], torch.tensor([1.0, 2.0]))
        assert torch.equal(state["row"], torch.tensor([3.0, 4.0]))


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
    assert res.score_oof and res.meta["class_names"] == RANO_PROBE_NAMES


def test_multi_output_metric():
    task = TASKS.get("rano4_current")()
    ev = ReadoutEvaluator(task, _StubProtocol(), view="fused", readout="linear",
                          metrics=["macro_f1", "per_class_recall"], boot=0)
    res = ev.run(_cache()["patients"])
    assert "macro_f1" in res.metrics
    assert set(res.patient_metrics) >= {
        "macro_f1", "recall_PD", "recall_SD", "recall_PR", "recall_CR"
    }


def test_grouped_ridge_cv():
    x = torch.randn(10, 3, generator=torch.Generator().manual_seed(4))
    y = torch.randn(10, generator=torch.Generator().manual_seed(5))
    groups = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    w, b, lam = fit_ridge_cv(x, y, lams=(1.0,), groups=groups)
    assert w.shape == (3,) and torch.isfinite(w).all() and torch.isfinite(b)


def test_score_metrics():
    task = TASKS.get("pd_forecast")()
    ev = ReadoutEvaluator(task, _StubProtocol(), view="fused", readout="linear",
                          metrics=["auc", "auprc", "brier", "logloss", "ece"],
                          boot=20, seed=1, metric_seed=1)
    res = ev.run(_cache()["patients"])
    assert all(k in res.metrics for k in ("auc", "auprc", "brier", "logloss", "ece"))
    assert "auc" in res.ci


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
    check("label_contract_and_manifest", test_label_contract_and_manifest)
    check("anatomy_measurement_and_pairs", test_anatomy_measurement_and_pairs)
    check("anatomy_metrics_are_patient_uniform", test_anatomy_metrics_are_patient_uniform)
    check("checkpoint_compatibility_loader", test_checkpoint_compatibility_loader)
    check("anatomy_first_cli_routes", test_anatomy_first_cli_routes)
    check("physical_anatomy_features_are_history_only",
          test_physical_anatomy_features_are_history_only)
    check("residual_model_starts_at_persistence",
          test_residual_model_starts_at_persistence)
    check("lesion_patch_coverage_support", test_lesion_patch_coverage_support)
    check("lesion_crop_and_region_pooling", test_lesion_crop_and_region_pooling)
    check("lesion_forecaster_starts_at_persistence",
          test_lesion_forecaster_starts_at_persistence)
    check("anatomy_representation_join_is_row_exact",
          test_anatomy_representation_join_is_row_exact)
    check("latent_metrics", test_latent_metrics)
    check("legacy_task_rows", test_legacy_task_rows)
    check("canonicalize_and_views", test_canonicalize_and_views)
    check("bootstrap_reproducible", test_bootstrap_reproducible)
    check("evaluator_composes", test_evaluator_composes)
    check("multi_output_metric", test_multi_output_metric)
    check("grouped_ridge_cv", test_grouped_ridge_cv)
    check("score_metrics", test_score_metrics)
    check("frozen_paths", test_frozen_paths)
    check("latent_pairs_and_eval", test_latent_pairs_and_eval)
    check("native_jepa_registered", test_native_jepa_registered)
    print(f"PASS {len(PASSED)}/{len(PASSED) + len(FAILED)}")
    for name, err in FAILED:
        print(f"FAIL {name}: {type(err).__name__}: {err}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
