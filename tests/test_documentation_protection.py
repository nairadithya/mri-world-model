from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_documentation_protection import compare, git
from documentation_sources import sha


class DocumentationProtectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name)
        self.tree = self.parent / "world-model-r36"
        self.tree.mkdir()
        self.file = self.tree / "tracked.md"
        self.file.write_text("original\n")
        self.baseline = {"worktrees": [{"worktree": self.tree.name, "head": "baseline",
                                       "status": "", "tracked_sha256": {"tracked.md": sha(self.file.read_bytes())}}]}

    def test_unchanged_and_changed_files_are_reported_without_writes(self):
        with patch("check_documentation_protection.git", side_effect=["baseline\n", ""]):
            result = compare(self.baseline, self.parent)[0]
        self.assertTrue(result["head_equal"])
        self.assertTrue(result["status_equal"])
        self.assertEqual(result["changed_files"], [])
        self.file.write_text("owner change\n")
        with patch("check_documentation_protection.git", side_effect=["new-head\n", " M tracked.md\n"]):
            result = compare(self.baseline, self.parent)[0]
        self.assertFalse(result["head_equal"])
        self.assertFalse(result["status_equal"])
        self.assertEqual(result["changed_files"][0]["path"], "tracked.md")
        self.assertEqual(self.file.read_text(), "owner change\n")

    def test_missing_file_and_unexpected_checkout_are_detected(self):
        self.file.unlink()
        with patch("check_documentation_protection.git", side_effect=["baseline\n", ""]):
            result = compare(self.baseline, self.parent)[0]
        self.assertIsNone(result["changed_files"][0]["current_sha256"])
        self.baseline["worktrees"][0]["worktree"] = "../other"
        with self.assertRaises(ValueError):
            compare(self.baseline, self.parent)

    def test_git_optional_locks_are_disabled(self):
        with patch("check_documentation_protection.subprocess.check_output", return_value=b"ok\n") as command:
            self.assertEqual(git(self.tree, "status", "--short"), "ok\n")
        self.assertEqual(command.call_args.kwargs["env"]["GIT_OPTIONAL_LOCKS"], "0")


if __name__ == "__main__":
    unittest.main()
