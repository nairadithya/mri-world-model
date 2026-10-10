from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scripts.report_representation_run import aggregate, compare, load_predictions


class RepresentationReportTests(unittest.TestCase):
    def records(self):
        truth = np.zeros((3, 3))
        groups = np.asarray(["a", "b", "b"])
        prediction = np.asarray([[3., 3., 3.], [0., 0., 0.], [0., 0., 0.]])
        return [{"state_truth": truth, "state_patients": groups, "prediction": sign * prediction}
                for sign in (1, -1, 1)]

    def test_average_errors_is_not_an_ensemble_or_visit_weighting(self):
        result = aggregate(self.records(), "prediction", True)
        self.assertEqual(result["seed_mean_patient_uniform_mae"], 1.5)
        self.assertEqual(result["seed_mean_pooled_mae"], 1)
        self.assertEqual(result["seed_patient_uniform_mae"], [1.5, 1.5, 1.5])
        self.assertEqual(result["n_patients"], 2)
        self.assertEqual(list(result["seed_mean_pooled_compartment_mae"].values()), [1., 1., 1.])

    def test_comparison_bootstraps_patients_not_seed_copies(self):
        records = self.records()
        baseline = [{**records[0], "prediction": np.zeros((3, 3))}]
        result = compare(records, "prediction", baseline, "prediction", True)
        self.assertEqual(result["difference"], 1.5)
        self.assertEqual(result["n_patients"], 2)
        self.assertEqual(result["paired_patient_95ci"], [0., 3.])

    def test_nonfinite_predictions_are_rejected(self):
        records = self.records()
        records[0]["prediction"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "invalid prediction"):
            aggregate(records, "prediction", True)

    def test_row_order_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = {"state_truth": np.zeros((2, 3)), "change_truth": np.zeros((1, 3)),
                    "state_patients": np.asarray(["a", "a"]), "pair_patients": np.asarray(["a"]),
                    "visit_ids": np.asarray(["v0", "v1"]), "row_ids": np.asarray(["r0"])}
            path = Path(tmp) / "predictions.npz"
            np.savez(path, **{**base, "visit_ids": np.asarray(["v1", "v0"])})
            with self.assertRaises(AssertionError):
                load_predictions(path, base)


if __name__ == "__main__":
    unittest.main()
