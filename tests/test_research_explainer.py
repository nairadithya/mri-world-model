import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_research_explainer.py"
REQUIRED_CLAIMS = (
    "0bed48b",
    "067ed6c",
    "768cd7e",
    "236352d",
    "A40",
    "A41",
    "0.334 ± 0.060",
    "0.309",
    "Readout seed and width alone",
    "pooled anatomy MAE",
    "No causal replacement score exists",
)


class ResearchExplainerVerifierTest(unittest.TestCase):
    def run_verifier(self, link):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            info = root / "info"
            info.mkdir()
            shutil.copy2(SCRIPT, root / "scripts" / SCRIPT.name)
            (info / "research_run.html").write_text(
                self.page(link), encoding="utf-8"
            )
            return subprocess.run(
                ["python", "-B", str(root / "scripts" / SCRIPT.name)],
                capture_output=True,
                text=True,
                check=False,
            )

    @staticmethod
    def page(link):
        ids = "".join(f'<td class="id">A{i}</td>' for i in range(1, 42))
        claims = " ".join(REQUIRED_CLAIMS)
        return f'<a href="{link}">source</a>{ids}{claims}'

    def test_accepts_existing_local_file_fragment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            info = root / "info"
            info.mkdir()
            shutil.copy2(SCRIPT, root / "scripts" / SCRIPT.name)
            (info / "source.md").write_text("# target", encoding="utf-8")
            (info / "research_run.html").write_text(
                self.page("source.md#target"), encoding="utf-8"
            )
            result = subprocess.run(
                ["python", "-B", str(root / "scripts" / SCRIPT.name)],
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            result.stdout,
            "PASS: all 41 experiment IDs, 1 local links, and cutoff/branch claims checked\n",
        )

    def test_reports_missing_local_file_fragment(self):
        result = self.run_verifier("missing.md#target")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("broken local links: ['missing.md#target']", result.stdout)


if __name__ == "__main__":
    unittest.main()
