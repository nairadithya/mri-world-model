"""Current R37 presentation must use unchanged, independently reviewed evidence."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from scripts.refresh_research_run import r37_evidence, r37_section

ROOT = Path(__file__).resolve().parents[1]
CLOSURE = Path("info/documentation_sources/updates/2026-10-26/closure.json")


class R37DocumentationTests(unittest.TestCase):
    def test_current_table_and_intervals_come_from_bound_aggregate(self):
        metrics, closure = r37_evidence(ROOT)
        text = r37_section(metrics, closure)
        for arm in ("H", "L", "P", "E", "R", "G"):
            self.assertIn(f'{metrics["arm_scores"][arm]["crps"]:.6f}', text)
        for contrast in metrics["primary_comparisons"].values():
            self.assertIn(f'[{contrast["ci"][0]:+.6f}, {contrast["ci"][1]:+.6f}]', text)
        self.assertIn("not equivalence", text)
        self.assertIn("no rescue training", text)
        self.assertIn("not clinical scanner effects", text)

    def test_altered_raw_aggregate_is_not_presented_as_reviewed(self):
        closure = json.loads((ROOT/CLOSURE).read_text())
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            items = [closure[k]["path"] for k in ("aggregate", "implementation_review", "result_review")]
            items += [closure["independent_verification"][f"{k}_path"] for k in ("checker", "receipt")]
            for relative in [str(CLOSURE), *items]:
                dest = root/relative; dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT/relative, dest)
            aggregate = root/closure["aggregate"]["path"]
            aggregate.write_bytes(aggregate.read_bytes()+b"\n")
            with self.assertRaisesRegex(ValueError, "evidence bytes"):
                r37_evidence(root)

    def test_absent_reporting_approval_blocks_generation(self):
        closure = json.loads((ROOT/CLOSURE).read_text())
        closure["result_review"]["verdict"] = "CHANGES_REQUIRED"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); path = root/CLOSURE; path.parent.mkdir(parents=True)
            path.write_text(json.dumps(closure))
            with self.assertRaisesRegex(ValueError, "scoped inconclusive result approval"):
                r37_evidence(root)


if __name__ == "__main__":
    unittest.main()
