"""Aggregate-only exports and the generated research page keep their scope."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.summarize_delta_representation import average
from scripts.refresh_research_run import section


class SummaryTests(unittest.TestCase):
    def test_seed_errors_are_averaged_without_patient_records(self):
        values = [{"score": 1.0, "per_patient": {"private": 1.0}, "values": [2.0, 4.0], "empty": None},
                  {"score": 3.0, "per_patient": {"private": 3.0}, "values": [4.0, 6.0], "empty": None}]
        self.assertEqual(average(values), {"score": 2.0, "values": [3.0, 5.0], "empty": None})

    def test_changed_support_and_nonfinite_values_fail(self):
        for values in ([{"x": 1}, {"y": 1}], [[1, 2], [1]], [None, 1], [float("nan"), 1]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                average(values)

    def test_page_does_not_turn_completion_into_success(self):
        local = {name: {"patient_uniform_balanced_mae": 0.4} for name in
                 ("frozen_probe", "independent_state", "target_only", "zero")}
        metrics = {"hypotheses": {"C_vs_B_frozen_accessibility": "INCONCLUSIVE", "state_retention": "VERIFIED",
                                  "D_nuisance_tradeoff": "INCONCLUSIVE", "incremental_temporal_information": "INCONCLUSIVE"},
                   "comparisons": {"C-B": {"difference": -0.003, "family_adjusted_ci": [-0.008, 0.003]}},
                   "state_C-A": {"family_adjusted_ci": [-0.073, 0.017]},
                   "nuisance_comparisons": {"state_D-C": {"difference": 0.048, "family_adjusted_ci": [0.004, 0.099]}},
                   "cohorts": {"lumiere": {"A": {"state": {"patient_uniform_mae": 0.9}, "local": local}}}}
        pending = section(metrics, False)
        reviewed = section(metrics, True)
        self.assertIn("INCONCLUSIVE", pending)
        self.assertIn("No quality verdict is accepted", pending)
        self.assertIn("not graded", pending)
        self.assertIn("0.9000", pending)
        self.assertNotIn("No quality verdict is accepted", reviewed)
        self.assertIn("Phase 2 remains unauthorized", reviewed)
        self.assertIn("not a verified encoder improvement", reviewed)
        metrics["hypotheses"]["C_vs_B_frozen_accessibility"] = "VERIFIED"
        with self.assertRaises(ValueError):
            section(metrics, True)


if __name__ == "__main__":
    unittest.main()
