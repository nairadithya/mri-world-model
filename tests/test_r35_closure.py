"""Check the public R35 closure receipt without accessing private artifacts."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
REVIEWS = ROOT / "info/reviews"
EVIDENCE = REVIEWS / "evidence/r35"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class R35ClosureTests(unittest.TestCase):
    def setUp(self):
        self.receipt = json.loads((REVIEWS / "r35_closure_receipt.json").read_text())

    def test_final_verdicts_bind_preserved_evidence(self):
        self.assertEqual(self.receipt["status"], "COMPUTATION_AND_SCOPED_REPORT_REVIEW_COMPLETE")
        for role, verdict in (("scientific_result_review", "APPROVED_FOR_SCOPED_REPORT"),
                              ("artifact_verification", "PASS_FOR_SCOPED_REPORT")):
            section = self.receipt[role]
            self.assertEqual(section["verdict"], verdict)
            self.assertIn(verdict, (REVIEWS / section["report"]).read_text())
        for name, expected in self.receipt["evidence_sha256"].items():
            with self.subTest(path=name):
                path = (REVIEWS / name).resolve()
                self.assertTrue(path.is_relative_to(REVIEWS.resolve()))
                self.assertEqual(digest(path), expected)

    def test_numerical_receipts_bind_actual_sources_and_exits(self):
        for name in ("metrics_r35.py", "contrast_extra.py", "correct_independent.py"):
            with self.subTest(script=name):
                receipt = json.loads((EVIDENCE / f"{name}.receipt.json").read_text())
                self.assertEqual(receipt["source_sha256"], digest(EVIDENCE / name))
                self.assertEqual(receipt["exit_status"], 0)
                self.assertEqual((EVIDENCE / Path(receipt["exit_path"]).name).read_text().strip(), "0")
                self.assertEqual((EVIDENCE / Path(receipt["stderr_path"]).name).read_bytes(), b"")
                self.assertGreater((EVIDENCE / Path(receipt["stdout_path"]).name).stat().st_size, 0)
        self.assertIn("independent_state_arm", (EVIDENCE / "correct_independent.py.stdout.txt").read_text())

    def test_supplement_and_history_are_not_confused(self):
        output = (EVIDENCE / "supplemental_r35.output.txt").read_text()
        self.assertIn("ASSERTION_FAILURES 0", output)
        counts = json.loads(next(line.removeprefix("COUNTS ") for line in output.splitlines()
                                 if line.startswith("COUNTS ")))
        self.assertEqual(counts, {"nuisance_maps": 65, "private_hashes": 228,
                                 "query_summary_digests": 24, "reference_budgets": 2,
                                 "source_summary_digests": 12})
        self.assertIn("BLOCKED", (EVIDENCE / "history/initial-result-review.md").read_text())
        self.assertIn("NEEDS_FIXES", (EVIDENCE / "history/audit-challenge-result-review.md").read_text())
        self.assertIn("if False else True", (EVIDENCE / "audit_r35.py").read_text())
        self.assertNotIn("if False else True", (EVIDENCE / "supplemental_r35.py").read_text())
        self.assertFalse(self.receipt["private_handoff"]["inventory_committed"])
        self.assertFalse(self.receipt["reference_feature_cache"]["original_manifest_member"])

    def test_canonical_documents_link_closure_without_validation_claim(self):
        for name in ("EXPERIMENTS.md", "METHODS.md", "NOTES_AND_INSIGHTS.md", "TODO.md", "README.md"):
            with self.subTest(document=name):
                text = (ROOT / "info" / name).read_text()
                self.assertIn("reviews/r35_closure_receipt.json", text)
                self.assertIn("reviews/r35_result_review.md", text)
        self.assertIn("not newly hashed checkpoint bytes", (ROOT / "info/EXPERIMENTS.md").read_text())
        self.assertIn("not R36's final-epoch-20 recipe", (ROOT / "info/METHODS.md").read_text())


if __name__ == "__main__":
    unittest.main()
