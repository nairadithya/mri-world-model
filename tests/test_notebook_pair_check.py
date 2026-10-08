import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_notebook_pairs import pair_diff  # noqa: E402


class NotebookPairCheckTests(unittest.TestCase):
    def test_detects_stale_generated_notebook(self):
        # Regression fixture: the paired source was edited but the pushable
        # notebook (what Kaggle executes) was left at the prior version.
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "example.py"
            notebook = source.with_suffix(".ipynb")
            source.write_text("# %%\nprint('old')\n")
            subprocess.run(
                [sys.executable, "-m", "jupytext", "--to", "ipynb", str(source)],
                check=True, capture_output=True, text=True,
            )
            self.assertEqual(pair_diff(source, notebook), "")

            source.write_text("# %%\nprint('new')\n")
            diff = pair_diff(source, notebook)
            self.assertIn("print('new')", diff)
            self.assertIn("print('old')", diff)


if __name__ == "__main__":
    unittest.main()
