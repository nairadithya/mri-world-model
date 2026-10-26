import json
import tempfile
import unittest
from pathlib import Path

from src.harness.analysis.joint_dynamics.authority import _private_test_authority


class AuthorityTests(unittest.TestCase):
    def test_initialization_is_once_and_snapshot_has_fixed_identity(self):
        with tempfile.TemporaryDirectory() as d:
            authority = _private_test_authority(Path(d) / "budget.json")
            snapshot = authority.initialize()
            self.assertEqual(snapshot["charged_cpu_seconds"], 900.0)
            self.assertEqual(snapshot["remaining_gpu_seconds"], 14400.0)
            with self.assertRaises(RuntimeError):
                authority.initialize()

    def test_stage_is_journaled_and_extension_is_verified(self):
        with tempfile.TemporaryDirectory() as d:
            authority = _private_test_authority(Path(d) / "budget.json")
            before = authority.initialize()
            with authority.stage("engineering-test") as guard:
                guard.check()
                self.assertEqual(guard.snapshot()["stages"], [])
            after = authority.snapshot()
            self.assertEqual(after["stages"][0]["status"], "succeeded")
            authority.verify_extension(before)
            authority.verify_extension(after)

    def test_failed_scientific_stage_blocks_continuation(self):
        with tempfile.TemporaryDirectory() as d:
            authority = _private_test_authority(Path(d) / "budget.json")
            authority.initialize()
            with self.assertRaisesRegex(ValueError, "stop"):
                with authority.stage("locked-fit", scientific=True):
                    raise ValueError("stop")
            with self.assertRaisesRegex(RuntimeError, "blocks"):
                with authority.stage("later-fit", scientific=True):
                    pass

    def test_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "budget.json"
            authority = _private_test_authority(path)
            authority.initialize()
            data = json.loads(path.read_text())
            data["charged_cpu_seconds"] = 0
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(RuntimeError, "totals"):
                authority.snapshot()


if __name__ == "__main__":
    unittest.main()
