"""R37 full-pilot contracts and mathematical controls, with synthetic data only."""
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
import torch

from src.harness.analysis.spatial_dynamics_data import SpatialData, canonical_hash
from src.harness.analysis.spatial_dynamics_fit import (
    calibrate_gaussian, gaussian_scores, patient_mean, ridge_design, solve_ridge,
)
from src.harness.analysis.spatial_dynamics_report import (
    analyze, contrast, gaussian_bundle, interval_mixture, summarize_bundle,
)
from src.harness.analysis.spatial_dynamics_runtime import BudgetAuthority, budget_stage


def synthetic_payload():
    torch.manual_seed(8)
    from src.harness.analysis.spatial_dynamics_data import CHECKPOINT_SHA256
    return {"schema": "r37-spatial-cache-v1", "checkpoint_sha256": CHECKPOINT_SHA256,
            "visit_ids": ["a", "b", "c"], "maps": torch.randn(3, 128, 2, 2, 2),
            "availability": torch.ones(3, 4), "feature_mean": torch.zeros(128), "feature_std": torch.ones(128),
            "normalization": {"mean": [0., 0., 0.], "scale": [1., 1., 1.]},
            "decoder_weight": torch.randn(3, 128), "decoder_bias": torch.zeros(3),
            "splits": {"source52": ["p"], "validation13": ["v"], "unseen26": ["q"]},
            "rows": [{"row_id": "r", "patient_id": "p", "history_visit_ids": ["a", "b"],
                      "cutoff_visit_id": "b", "target_visit_id": "c", "primary_history": True,
                      "history_mask": [False, False, True, True], "gap_log1p": [0., 0., 0., 2.],
                      "truth": [1., 2., 3.], "cutoff_measured": [0., 1., 2.]}]}


class SpatialDynamicsFullTests(unittest.TestCase):
    def test_future_payload_cannot_change_inputs(self):
        data = SpatialData(synthetic_payload(), device="cpu")
        row = copy.deepcopy(data.rows[0])
        original = data.inputs(row)
        row.update(target_visit_id="not_in_cache", truth=[999., 999., 999.], future_gap=999., future_availability=[0]*4)
        for left, right in zip(original, data.inputs(row)):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        with self.assertRaises(KeyError):
            data.target(row)
        row["history_visit_ids"][0] = "c"
        with self.assertRaisesRegex(ValueError, "locked cutoff"):
            data.inputs(row)

    def test_cutoff_covariate_tampering_is_rejected(self):
        data = SpatialData(synthetic_payload(), device="cpu")
        for field, value in [("cutoff_visit_id", "c"), ("gap_log1p", [0., 0., 0., 9.]),
                             ("history_mask", [True]*4)]:
            row = copy.deepcopy(data.rows[0]); row[field] = value
            with self.assertRaises(ValueError):
                data.inputs(row)

    def test_ridge_design_order_masks_covariates_and_voxel_pooling(self):
        data = SpatialData(synthetic_payload(), device="cpu")
        row = data.rows[0]
        design = ridge_design(data, row)
        self.assertEqual(design.shape, (8, 537))
        np.testing.assert_array_equal(design[:, :256], 0)
        np.testing.assert_allclose(design[:, 256:384].mean(0), data.maps[0].double().mean((-3,-2,-1)))
        np.testing.assert_array_equal(design[:, -1], 1)
        beta = torch.randn(537, 128, dtype=torch.float64)
        torch.testing.assert_close((design @ beta).mean(0), design.mean(0) @ beta)

    def test_ridge_matches_explicit_weighted_centered_solution(self):
        torch.manual_seed(10)
        x = torch.randn(30, 4, dtype=torch.float64)
        x[:, 3] = 7.  # constant covariate must not cause a singular solve
        y = torch.randn(30, 2, dtype=torch.float64)
        weights = torch.arange(1., 31., dtype=torch.float64); weights /= weights.sum()
        design = torch.cat((x, torch.ones(30,1,dtype=torch.float64)), 1)
        actual = solve_ridge(design.T @ (weights[:,None] * design), design.T @ (weights[:,None] * y), .1)
        mx, my = weights @ x, weights @ y
        sx = torch.sqrt(weights @ ((x-mx)**2)).clamp_min(1e-6)
        xx = (x-mx) / sx
        coef = torch.linalg.solve(xx.T @ (weights[:,None]*xx) + .1*torch.eye(4,dtype=torch.float64),
                                  xx.T @ (weights[:,None]*(y-my))) / sx[:,None]
        expected = torch.cat((coef, (my-mx@coef)[None]), 0)
        torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-8)

    def test_patient_aggregation_and_bootstrap_never_pool_visits(self):
        rows = [{"patient_id": p} for p in ["a", "a", "a", "b"]]
        scores = np.repeat(np.asarray([0., 0., 0., 10.])[:,None], 3, 1)
        self.assertEqual(patient_mean(scores, rows).mean(), 5.)
        result = contrast(scores, np.zeros_like(scores), rows)
        self.assertEqual(result["difference"], 5.)
        self.assertEqual(result["patients"], 2)
        self.assertEqual(result["ci"], [0., 10.])
        self.assertEqual(contrast(scores, np.zeros_like(scores), rows), result)

    def test_calibration_tie_uses_first_grid_and_does_not_move_mean(self):
        rows = [{"patient_id": "v", "truth": [0., 0., 0.]}]
        prediction = {"mean": np.zeros((1,3)), "intrinsic_scale": np.zeros((1,3))}
        selected, scores = calibrate_gaussian(prediction, rows)
        self.assertEqual(selected, .5)
        self.assertEqual(scores[0], scores[2])
        bundle = gaussian_bundle(prediction, np.zeros((1,3)), 2.)
        np.testing.assert_array_equal(bundle["mean"], prediction["mean"])
        np.testing.assert_allclose(bundle["scale"], .05)

    def test_mixture_interval_matches_single_gaussian_and_brackets_median(self):
        centers = np.asarray([[[1., 2., 3.]]])
        low, high = interval_mixture(centers, np.asarray([1.]), .05, .8)
        np.testing.assert_allclose(low, centers[:,0] - 1.2815515655446004*.05)
        np.testing.assert_allclose(high, centers[:,0] + 1.2815515655446004*.05)
        centers = np.asarray([[[-2., -1., 0.], [2., 3., 4.]]])
        low, high = interval_mixture(centers, np.asarray([.5,.5]), .05, .5)
        self.assertTrue((low < np.asarray([[0.,1.,2.]])).all())
        self.assertTrue((high > np.asarray([[0.,1.,2.]])).all())

    def test_empty_targets_keep_compartment_denominators(self):
        rows = [{"patient_id": "a", "truth": [0., 1., 2.], "cutoff_measured": [1., 0., 0.]},
                {"patient_id": "b", "truth": [0., 0., 0.], "cutoff_measured": [0., 0., 1.]}]
        truth = np.asarray([r["truth"] for r in rows])
        bundle = gaussian_bundle({"mean": np.zeros((2,3)), "intrinsic_scale": np.ones((2,3))}, truth, 1.)
        summary = summarize_bundle(bundle, truth, rows)
        self.assertEqual(summary["subsets"]["empty_target"][0]["examples"], 2)
        self.assertEqual(summary["subsets"]["appearance"][1]["examples"], 1)
        self.assertEqual(summary["subsets"]["disappearance"][0]["examples"], 1)
        self.assertEqual(summary["subsets"]["both_empty"][2]["examples"], 0)
        self.assertIsNone(summary["subsets"]["both_empty"][2]["mae"])

    def test_seed_scores_not_ensemble_and_incomplete_arm_rejected(self):
        rows = [{"patient_id": f"p{i}", "truth": [1.,1.,1.], "cutoff_measured": [1.,1.,1.]} for i in range(15)]
        truth = np.ones((15,3))
        minus = gaussian_bundle({"mean": -np.ones((15,3)), "intrinsic_scale": np.ones((15,3))}, truth, 1.)
        plus = gaussian_bundle({"mean": 3*np.ones((15,3)), "intrinsic_scale": np.ones((15,3))}, truth, 1.)
        middle = gaussian_bundle({"mean": np.ones((15,3)), "intrinsic_scale": np.ones((15,3))}, truth, 1.)
        primary = {a: [minus, plus, middle] if a in {"H","L","G"} else [middle] for a in ["H","L","G","P","E","R"]}
        nuisance = {a: {c: [minus, plus, middle] for c in ["identity","intensity","resolution"]} for a in ["H","L"]}
        with tempfile.TemporaryDirectory() as directory:
            report = analyze(primary, rows, nuisance, rows, Path(directory)/"aggregate.json")
            self.assertAlmostEqual(report["arm_scores"]["H"]["mae"], 4/3)
            self.assertFalse(report["history_benefit"])
            self.assertFalse(report["automatic_followup"])
            self.assertEqual(report["decision"], "INCONCLUSIVE_FOR_FULL_CAPABILITY")
            del primary["R"]
            with self.assertRaises(ValueError):
                analyze(primary, rows, nuisance, rows, Path(directory)/"bad.json")

    def test_budget_preserves_failed_stage_and_disallows_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"ledger.json"
            authorization = {"cpu_seconds_limit": 1000., "gpu_seconds_limit": 1000., "deadline_unix": time.time()+1000}
            path.write_text(json.dumps({"stages": [], "setup_cpu_charge_seconds": 900., **authorization}))
            authority = BudgetAuthority(path, authorization)
            authority.initialize()
            with budget_stage(path, "test", authority=authority) as guard:
                guard()
            ledger = json.loads(path.read_text())
            self.assertEqual(ledger["stages"][0]["status"], "completed")
            self.assertGreater(ledger["stages"][0]["cpu_seconds"], 0)
            with self.assertRaises(FileExistsError):
                with budget_stage(path, "test", authority=authority):
                    pass
            with self.assertRaisesRegex(ValueError, "deliberate"):
                with budget_stage(path, "failure", authority=authority):
                    raise ValueError("deliberate")
            self.assertEqual(json.loads(path.read_text())["stages"][-1]["status"], "failed")
            with self.assertRaises(RuntimeError):
                with budget_stage(path, "next", authority=authority):
                    pass


if __name__ == "__main__":
    unittest.main()
