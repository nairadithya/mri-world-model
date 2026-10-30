import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.harness.analysis.treatment_forecast import (
    COMPARTMENTS,
    Rows,
    crps,
    features,
    fit_ridge,
    fit_transition,
    forecast,
    load_rows,
    patient_means,
    predict,
    prepare,
    run,
)


def synthetic_rows():
    rng = np.random.default_rng(11)
    current = rng.normal(3, 0.2, (120, 3))
    phase = np.tile(np.arange(4), 30)
    return Rows(
        current,
        current - 0.1,
        current + 0.2 * (phase[:, None] == 1),
        current + 0.01,
        rng.normal(size=(120, 5)),
        phase,
        np.repeat(np.arange(24), 5),
        np.ones(120, dtype=bool),
    )


def fixture(directory):
    visits, pairs, predictions = [], [], []
    for patient in range(21):
        pid = f"subject{patient}"
        (directory / pid).mkdir()
        (directory / pid / "intervals-days.txt").write_text("10 20 30 40 50")
        for t in range(6):
            name = f"ses-{t}"
            path = directory / pid / name
            path.mkdir()
            (path / "treatment.txt").write_text("TMZ" if t % 2 else "CRT")
            key = f"{pid}/{name}"
            values = [100 + patient + t, 200 + t, 300 + t]
            visits.append(
                dict(
                    visit_id=key,
                    visit=name,
                    patient_id=pid,
                    values_mm3=dict(zip(COMPARTMENTS, values)),
                )
            )
            predictions.append(np.log1p(values))
            if t:
                pairs.append(
                    dict(
                        source_visit=f"ses-{t - 1}",
                        target_visit=name,
                        source_visit_id=f"{pid}/ses-{t - 1}",
                        target_visit_id=key,
                        patient_id=pid,
                        usable=True,
                        ordering_verified=True,
                        row_id=f"row{len(pairs)}",
                    )
                )
    inventory = directory / "inventory.json"
    inventory.write_text(json.dumps(dict(cohort="SAILOR", visits=visits, pairs=pairs)))
    prediction = directory / "predictions.npz"
    np.savez_compressed(
        prediction,
        visit_ids=[v["visit_id"] for v in visits],
        state_prediction=predictions,
        state_truth=predictions,
    )
    return inventory, prediction


class TreatmentForecastTests(unittest.TestCase):
    def test_crps_known_value_and_patient_uniform_weighting(self):
        value = crps(np.zeros((1, 3)), np.zeros((1, 3)), np.ones(3))
        np.testing.assert_allclose(value, [[0.23369497725510913] * 3], atol=1e-14)
        np.testing.assert_array_equal(
            patient_means(np.array([0.0, 0.0, 0.0, 2.0]), np.array([0, 0, 0, 1])),
            [0.0, 2.0],
        )

    def test_future_targets_and_privileged_measurements_never_enter_features(self):
        rows = synthetic_rows()
        altered = Rows(
            **{
                **vars(rows),
                "truth": rows.truth + 999,
                "measured_current": rows.measured_current + 888,
            }
        )
        for arm in ("L", "LT", "H", "HT", "time_only", "phase_time"):
            np.testing.assert_array_equal(features(rows, arm), features(altered, arm))
        other_phase = Rows(**{**vars(rows), "phase": (rows.phase + 1) % 4})
        for arm in ("L", "H", "time_only"):
            np.testing.assert_array_equal(
                features(rows, arm), features(other_phase, arm)
            )

    def test_linear_fit_and_checkpoint_roundtrip(self):
        x = np.arange(20, dtype=float).reshape(-1, 1)
        y = np.tile(2 * x + 1, (1, 3))
        model = fit_ridge(x, y, np.arange(20), 1e-9)
        np.testing.assert_allclose(
            predict(model, np.array([[2.0], [4.0]])), [[5.0] * 3, [9.0] * 3], atol=1e-8
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model.npz"
            np.savez(path, **model)
            with np.load(path) as stored:
                restored = dict(stored)
            np.testing.assert_array_equal(predict(restored, x), predict(model, x))

    def test_heldout_targets_cannot_change_fitted_prediction(self):
        rows = synthetic_rows()
        train, test = rows.subset(rows.groups < 20), rows.subset(rows.groups >= 20)
        model = fit_transition(train, "HT")
        altered = Rows(**{**vars(test), "truth": test.truth - 999})
        np.testing.assert_array_equal(
            forecast(model, test, "HT"), forecast(model, altered, "HT")
        )
        self.assertTrue(np.isfinite(model["sd"]).all())

    def test_prepare_cutoff_timing_and_future_phase(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inventory, prediction = fixture(root)
            result = prepare(inventory, prediction, root, root / "first")
            self.assertEqual(
                (result["patients"], result["pairs"], result["history_pairs"]),
                (21, 105, 84),
            )
            first = load_rows(root / "first/rows.npz")
            np.testing.assert_array_equal(first.timing[0], [0.0] * 5)
            np.testing.assert_allclose(
                first.timing[1],
                [np.log1p(10), np.log1p(10) ** 2, np.log1p(10), np.log1p(1), 1],
            )
            (root / "subject0/ses-1/treatment.txt").write_text("no")
            (root / "subject0/intervals-days.txt").write_text("10 999 888 777 666")
            prepare(inventory, prediction, root, root / "second")
            second = load_rows(root / "second/rows.npz")
            self.assertEqual(first.phase[0], second.phase[0])
            self.assertNotEqual(first.phase[1], second.phase[1])
            np.testing.assert_array_equal(first.timing[:2], second.timing[:2])
            outcome = run(root / "first/rows.npz", root / "canary", canary=True)
            self.assertEqual(outcome["status"], "ENGINEERING_CANARY")
            self.assertIsNone(outcome["treatment_feasibility_supported"])

    def test_unrecognized_phase_and_target_order_fail(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inventory, prediction = fixture(root)
            (root / "subject0/ses-0/treatment.txt").write_text("response_PD")
            with self.assertRaisesRegex(ValueError, "unrecognized treatment"):
                prepare(inventory, prediction, root, root / "bad_phase")
            data = json.loads(inventory.read_text())
            data["visits"][0]["values_mm3"]["enhancing"] = 999
            inventory.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "target order or identity"):
                prepare(inventory, prediction, root, root / "bad_target")

    def test_original_nonadjacent_pair_is_excluded(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inventory, prediction = fixture(root)
            data = json.loads(inventory.read_text())
            data["pairs"][0]["target_visit"] = "ses-2"
            inventory.write_text(json.dumps(data))
            result = prepare(inventory, prediction, root, root / "out")
            self.assertEqual(result["excluded"], {"nonadjacent_or_unusable": 1})
            self.assertEqual(result["pairs"], 104)


if __name__ == "__main__":
    unittest.main()
