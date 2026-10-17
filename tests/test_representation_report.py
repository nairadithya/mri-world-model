import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scripts.report_representation_run import aggregate, compare, load_predictions
from scripts.render_representation_report import render


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

    def test_renderer_includes_independent_state_and_training_mean_controls(self):
        def metric(value):
            return {"seed_mean_patient_uniform_mae": value}

        state = metric(1.0)
        zero = metric(0.0)
        direct = {"arms": {"state": {"state": state, "state_mean": state,
                                    "independent_decoder_difference": zero, "zero": zero}},
                  "comparisons": {}, "frozen_source52": state}
        change = {name: zero for name in ("zero", "difference", "concatenation", "later_only")}
        change.update(independent_state_difference=metric(0.5678), train_mean_change=metric(0.6543))
        summary = {"LUMIERE_direct": direct, "SAILOR_direct": direct,
                   "SAILOR_source_readouts": {"26": {"state": {"state": state, "state_mean": state,
                                                              "observed_change": change}}},
                   "LUMIERE_nuisance": {}, "seeds": [42], "blocked": [],
                   "state_audits": {f"{cohort}/seed42/{arm}":
                                    {"state_patient_uniform_mae": 1.0, "within_patient_shuffled_mae": 2.0}
                                    for cohort in ("LUMIERE", "SAILOR")
                                    for arm in ("state", "paired", "later_only", "frozen")}}
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "summary.json", Path(tmp) / "report.html"
            source.write_text(json.dumps(summary))
            render(str(source), str(output))
            document = output.read_text()
        self.assertIn("<th>Independent state difference</th>", document)
        self.assertIn("<th>Train-mean change</th>", document)
        self.assertIn("<td>0.5678</td>", document)
        self.assertIn("<td>0.6543</td>", document)
        self.assertIn("family-12 intervals", document)

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
