"""Literal estimand and empty-cohort checks for the rerunnable validity audit."""
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_mri_validity import geometry_audit, target_audit
from src.harness.data.anatomy_manifest import COMPARTMENTS


class AuditTests(unittest.TestCase):
    def test_pooled_and_patient_uniform_estimands_keep_immutable_rows(self):
        visits = []
        pairs = []
        for row_id, patient, source, target, usable in [
                ("a", "p1", 0, 9, True), ("b", "p1", 2, 2, True),
                ("c", "p2", 1, 3, True), ("excluded", "p2", 0, 1000, False),
                ("seen", "encoder-trained", 0, 1000, True)]:
            for endpoint, value in (("source", source), ("target", target)):
                visits.append({"visit_id": f"{row_id}-{endpoint}",
                               "measurement": {"values_mm3": {name: value for name in COMPARTMENTS}}})
            pairs.append({"row_id": row_id, "patient_id": patient, "cohort": "LUMIERE",
                          "usable": usable, "source_visit_id": f"{row_id}-source",
                          "target_visit_id": f"{row_id}-target"})
        manifest = {"visits": visits, "pairs": pairs}
        before = json.dumps(manifest, sort_keys=True)
        result = target_audit(manifest, {"p1", "p2"})["LUMIERE"]
        self.assertEqual(result["n_patients"], 2)
        self.assertEqual(result["n_rows"], 3)
        for name in COMPARTMENTS:
            observed = result["compartments"][name]
            self.assertEqual([r["row_id"] for r in observed["rows"]], ["a", "b", "c"])
            self.assertAlmostEqual(observed["pooled_persistence_mae"], (math.log(10) + math.log(2)) / 3)
            self.assertAlmostEqual(observed["patient_uniform_persistence_mae"],
                                   (math.log(10) / 2 + math.log(2)) / 2)
            self.assertEqual(observed["one_endpoint_empty"], 1)
            self.assertEqual(observed["both_empty"], 0)
            self.assertAlmostEqual(observed["pooled_error_share_one_endpoint_empty"],
                                   math.log(10) / (math.log(10) + math.log(2)))
        self.assertEqual(json.dumps(manifest, sort_keys=True), before)

    def test_empty_geometry_cohorts_do_not_pass_vacuously(self):
        args = SimpleNamespace(seed=42, visits_per_cohort=3)
        with tempfile.TemporaryDirectory() as directory:
            rows = geometry_audit({"visits": []}, args, Path(directory))
        self.assertEqual([r["cohort"] for r in rows], ["LUMIERE", "SAILOR"])
        self.assertEqual([r["verdict"] for r in rows], ["NOT VERIFIED", "NOT VERIFIED"])


if __name__ == "__main__":
    unittest.main()
