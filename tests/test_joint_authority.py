from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from src.harness.analysis.joint_dynamics import authority as mod


class AuthorityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="r38-authority-test-")
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "receipts" / "budget.json"
        self.authority = mod._private_test_authority(self.path)
        self.clock = patch.object(mod, "_time", return_value=datetime.fromisoformat(mod.START).timestamp() + 1)
        self.now = self.clock.start()
        self.addCleanup(self.clock.stop)

    def initialized(self) -> dict:
        return self.authority.initialize()

    def run_child(self, code: str) -> subprocess.CompletedProcess:
        prefix = ("from pathlib import Path\n"
                  "from src.harness.analysis.joint_dynamics.authority import _private_test_authority\n"
                  f"a = _private_test_authority(Path({str(self.path)!r}))\n")
        return subprocess.run([sys.executable, "-B", "-c", prefix + code], capture_output=True, text=True)

    def test_fixed_production_and_explicit_temporary_test_paths(self) -> None:
        self.assertEqual(mod.production_authority().ledger, mod.LEDGER)
        with self.assertRaises(ValueError):
            mod.BudgetAuthority(self.path)
        with self.assertRaises(ValueError):
            mod._private_test_authority(mod.LEDGER)
        with self.assertRaises(ValueError):
            mod._private_test_authority(Path("/home/alternate-budget.json"))
        with self.assertRaises(RuntimeError):
            self.authority.adopt_initialized_authority()

    def test_initialization_once_and_fixed_identity(self) -> None:
        snap = self.initialized()
        self.assertEqual(snap["charged_cpu_seconds"], 900.0)
        self.assertEqual(snap["remaining_gpu_seconds"], 14400.0)
        self.assertEqual(snap["identity"]["cpu_limit"], 28800.0)
        self.assertEqual(snap["journal_sequence"], 0)
        with self.assertRaises(RuntimeError):
            self.authority.initialize()

    def test_initialize_rejects_anchor_or_journal_even_without_ledger(self) -> None:
        self.initialized()
        self.path.unlink()
        with self.assertRaises(RuntimeError):
            self.authority.initialize()
        self.authority.anchor_path.unlink()
        with self.assertRaises(RuntimeError):
            self.authority.initialize()
        with self.assertRaises(RuntimeError):
            self.authority.snapshot()

    def test_running_reservation_is_fsynced_before_yield_and_guard_expires(self) -> None:
        self.initialized()
        real_fsync = os.fsync
        with patch.object(mod.os, "fsync", wraps=real_fsync) as sync:
            with self.authority.stage("engineering") as guard:
                self.assertGreaterEqual(sync.call_count, 5)
                snap = guard.snapshot()
                self.assertEqual(snap["stages"][-1]["status"], "running")
                self.assertEqual(json.loads(self.path.read_bytes())["stages"][-1]["status"], "running")
                self.assertEqual(json.loads(self.authority.journal_path.read_text().splitlines()[-1])["reason"], "reserve")
                guard.check()
        self.assertEqual(self.authority.snapshot()["stages"][-1]["status"], "succeeded")
        with self.assertRaises(RuntimeError):
            guard.check()
        with self.assertRaises(RuntimeError):
            guard.snapshot()

    def test_crash_leaves_durable_unresolved_reservation(self) -> None:
        self.initialized()
        child = self.run_child("import os\nwith a.stage('crashed'):\n    os._exit(73)\n")
        self.assertEqual(child.returncode, 73, child.stderr)
        self.assertEqual(self.authority.snapshot()["stages"][-1]["status"], "running")
        with self.assertRaisesRegex(RuntimeError, "unresolved running"):
            with self.authority.stage("not-a-retry"):
                self.fail("crashed stage must block")
        with self.assertRaises(RuntimeError):
            self.authority.initialize()

    def test_whole_stage_exclusive_flock_and_no_reentrant_snapshot(self) -> None:
        self.initialized()
        other = mod._private_test_authority(self.path)
        with self.authority.stage("exclusive"):
            with self.assertRaisesRegex(RuntimeError, "concurrent"):
                other.snapshot()
            with self.assertRaisesRegex(RuntimeError, "concurrent"):
                self.authority.snapshot()
            child = self.run_child("with a.stage('concurrent'):\n    pass\n")
            self.assertNotEqual(child.returncode, 0)
            self.assertIn("concurrent", child.stderr)
        self.assertEqual(len(self.authority.snapshot()["stages"]), 1)

    def test_failed_engineering_preserved_new_unique_stage_allowed(self) -> None:
        self.initialized()
        with self.assertRaisesRegex(ValueError, "fixture failure"):
            with self.authority.stage("engineering-failed"):
                raise ValueError("fixture failure")
        with self.authority.stage("engineering-fixed"):
            pass
        snap = self.authority.snapshot()
        self.assertEqual([s["status"] for s in snap["stages"]], ["failed", "succeeded"])
        self.assertIn("fixture failure", snap["stages"][0]["failure"])
        with self.assertRaisesRegex(RuntimeError, "already used"):
            with self.authority.stage("engineering-failed"):
                pass

    def test_failed_scientific_attempt_blocks_all_continuation(self) -> None:
        self.initialized()
        with self.assertRaises(ValueError):
            with self.authority.stage("fit", scientific=True):
                raise ValueError("failed fit")
        for scientific in (False, True):
            with self.assertRaisesRegex(RuntimeError, "scientific attempt blocks"):
                with self.authority.stage("later", scientific=scientific):
                    pass

    def test_stage_local_cpu_across_sequential_stages(self) -> None:
        self.initialized()
        with patch.object(mod, "_cpu", side_effect=[(100., False), (102., False), (103., False),
                                                    (500., False), (501., False), (502., False)]):
            with self.authority.stage("first"):
                pass
            with self.authority.stage("second"):
                pass
        snap = self.authority.snapshot()
        self.assertEqual([s["cpu_seconds"] for s in snap["stages"]], [3., 2.])
        self.assertEqual(snap["charged_cpu_seconds"], 905.)

    def test_child_cpu_is_charged(self) -> None:
        self.initialized()
        with self.authority.stage("child-cpu"):
            child = subprocess.run([sys.executable, "-B", "-c",
                                    "import time\ns=time.process_time()\nwhile time.process_time()-s < .08: pass"],
                                   capture_output=True)
            self.assertEqual(child.returncode, 0)
        self.assertGreaterEqual(self.authority.snapshot()["stages"][-1]["cpu_seconds"], .08)

    def test_live_child_cpu_is_visible_and_unjoined_child_blocks(self) -> None:
        ready = Path(self.tmp.name) / "child-ready"
        child = subprocess.Popen([sys.executable, "-B", "-c",
                                  "import time\nfrom pathlib import Path\ns=time.process_time()\n"
                                  "while time.process_time()-s < .15: pass\n"
                                  f"Path({str(ready)!r}).write_text('ready')\ntime.sleep(30)"])
        try:
            until = time.monotonic() + 10
            while not ready.exists() and time.monotonic() < until:
                time.sleep(.01)
            self.assertTrue(ready.exists())
            cpu, live = mod._live_child_cpu()
            self.assertTrue(live)
            self.assertGreater(cpu, .10)
        finally:
            child.terminate()
            child.wait()
        self.initialized()
        values = [(100., False), (100., False), (101., True)]
        with patch.object(mod, "_cpu", side_effect=values):
            with self.assertRaisesRegex(RuntimeError, "unjoined"):
                with self.authority.stage("unjoined"):
                    pass
        snap = self.authority.snapshot()
        self.assertEqual(snap["stages"][-1]["status"], "running")
        self.assertEqual(snap["charged_cpu_seconds"], 901.)
        with self.assertRaisesRegex(RuntimeError, "unresolved running"):
            with self.authority.stage("blocked"):
                pass

    def test_gpu_is_full_monotonic_wall_not_wall_clock(self) -> None:
        self.initialized()
        with patch.object(mod.time, "monotonic", side_effect=[10., 12., 17.]):
            with self.authority.stage("gpu-wall-fixture", gpu=True):
                pass
        self.assertEqual(self.authority.snapshot()["charged_gpu_seconds"], 7.)

    def test_actual_cpu_over_limit_persisted_before_raising(self) -> None:
        self.initialized()
        with patch.object(mod, "_cpu", side_effect=[(0., False), (0., False), (mod.CPU_LIMIT + 1, False)]):
            with self.assertRaisesRegex(RuntimeError, "CPU allowance"):
                with self.authority.stage("over-cpu"):
                    pass
        snap = self.authority.snapshot()
        self.assertEqual(snap["charged_cpu_seconds"], 900. + mod.CPU_LIMIT + 1)
        self.assertEqual(snap["stages"][-1]["status"], "failed")
        self.assertLess(snap["remaining_cpu_seconds"], 0)
        with self.assertRaisesRegex(RuntimeError, "CPU allowance"):
            with self.authority.stage("blocked-by-limit"):
                pass

    def test_actual_gpu_over_limit_persisted_before_raising(self) -> None:
        self.initialized()
        with patch.object(mod.time, "monotonic", side_effect=[0., 0., mod.GPU_LIMIT + 1]):
            with self.assertRaisesRegex(RuntimeError, "GPU allowance"):
                with self.authority.stage("over-gpu", gpu=True):
                    pass
        snap = self.authority.snapshot()
        self.assertEqual(snap["charged_gpu_seconds"], mod.GPU_LIMIT + 1)
        self.assertEqual(snap["stages"][-1]["status"], "failed")

    def test_actual_charges_survive_deadline_and_original_exception(self) -> None:
        self.initialized()
        with self.assertRaisesRegex(ValueError, "original"):
            with self.authority.stage("deadline"):
                self.now.return_value = mod._deadline_epoch() + 1
                raise ValueError("original")
        snap = self.authority.snapshot()
        self.assertEqual(snap["stages"][-1]["status"], "failed")
        self.assertGreater(snap["charged_cpu_seconds"], 900.)
        with self.assertRaises(RuntimeError):
            with self.authority.stage("late"):
                pass

    def test_final_deadline_error_without_body_error_is_persisted(self) -> None:
        self.initialized()
        with self.assertRaisesRegex(RuntimeError, "window"):
            with self.authority.stage("late-final"):
                self.now.return_value = mod._deadline_epoch() + 1
        self.assertEqual(self.authority.snapshot()["stages"][-1]["status"], "failed")

    def test_projection_failure_charged_and_invalid_projections_rejected(self) -> None:
        self.initialized()
        with self.assertRaisesRegex(RuntimeError, "GPU allowance"):
            with self.authority.stage("projected") as guard:
                guard.check(projected_gpu_seconds=mod.GPU_LIMIT + 1)
        self.assertEqual(self.authority.snapshot()["stages"][-1]["status"], "failed")
        with self.authority.stage("invalid-values") as guard:
            for value in (-1, float("nan"), float("inf"), True):
                with self.assertRaises(RuntimeError):
                    guard.check(projected_cpu_seconds=value)
            with self.assertRaisesRegex(RuntimeError, "deadline"):
                guard.check(projected_cpu_seconds=48 * 3600)

    def test_deleted_ledger_snapshot_rejected(self) -> None:
        self.initialized()
        self.path.unlink()
        with self.assertRaises(RuntimeError):
            self.authority.snapshot()

    def test_ledger_and_journal_rollback_rejected_by_separate_anchor(self) -> None:
        self.initialized()
        old_ledger = self.path.read_bytes()
        old_journal = self.authority.journal_path.read_bytes()
        with self.authority.stage("spent"):
            pass
        self.path.write_bytes(old_ledger)
        with self.assertRaisesRegex(RuntimeError, "rollback"):
            self.authority.snapshot()
        self.authority.journal_path.write_bytes(old_journal)
        with self.assertRaisesRegex(RuntimeError, "rollback"):
            self.authority.snapshot()
        with self.assertRaises(RuntimeError):
            self.authority.initialize()

    def test_copied_authority_and_replaced_journal_rejected(self) -> None:
        self.initialized()
        copied = Path(self.tmp.name) / "copied" / "budget.json"
        copied.parent.mkdir()
        for source in self.path.parent.iterdir():
            shutil.copyfile(source, copied.parent / source.name)
        with self.assertRaises(RuntimeError):
            mod._private_test_authority(copied).snapshot()
        raw = self.authority.journal_path.read_bytes()
        self.authority.journal_path.rename(self.authority.journal_path.with_suffix(".saved"))
        self.authority.journal_path.write_bytes(raw)
        with self.assertRaisesRegex(RuntimeError, "anchor identity/path"):
            self.authority.snapshot()

    def test_allowance_identity_totals_and_journal_tampering_rejected(self) -> None:
        self.initialized()
        raw = self.path.read_bytes()
        for key, value in (("charged_cpu_seconds", 0), ("identity", {**self.authority._identity, "cpu_limit": 999999})):
            p = json.loads(raw)
            p[key] = value
            self.path.write_text(json.dumps(p))
            with self.assertRaises(RuntimeError):
                self.authority.snapshot()
        self.path.write_bytes(raw)
        anchor_raw = self.authority.anchor_path.read_bytes()
        anchor = json.loads(anchor_raw)
        anchor["identity"]["gpu_limit"] += 1
        self.authority.anchor_path.write_text(json.dumps(anchor))
        with self.assertRaises(RuntimeError):
            self.authority.snapshot()
        self.authority.anchor_path.write_bytes(anchor_raw)
        event = json.loads(self.authority.journal_path.read_text())
        event["parent"] = "f" * 64
        self.authority.journal_path.write_text(json.dumps(event) + "\n")
        with self.assertRaisesRegex(RuntimeError, "journal ancestry"):
            self.authority.snapshot()

    def test_extension_requires_exact_snapshot_and_authority_ancestry(self) -> None:
        initial = self.initialized()
        with self.authority.stage("ancestor") as guard:
            running = guard.snapshot()
        complete = self.authority.snapshot()
        for snapshot in (initial, running, complete):
            self.authority.verify_extension(snapshot)
        for key, value in (("journal_tip", "f" * 64), ("journal_sequence", 100),
                           ("authority_id", "unrelated"), ("charged_cpu_seconds", 0),
                           ("remaining_cpu_seconds", 999999)):
            bad = {**complete, key: value}
            with self.assertRaises(RuntimeError):
                self.authority.verify_extension(bad)
        other = mod._private_test_authority(Path(self.tmp.name) / "other" / "budget.json")
        other_snap = other.initialize()
        with self.assertRaises(RuntimeError):
            self.authority.verify_extension(other_snap)

    def test_interrupted_persistence_fails_closed_instead_of_disappearing(self) -> None:
        self.initialized()
        original = mod._replace
        def fail_ledger(path: Path, raw: bytes) -> None:
            if path == self.path:
                raise OSError("simulated disk failure")
            original(path, raw)
        with patch.object(mod, "_replace", side_effect=fail_ledger):
            with self.assertRaises(OSError):
                with self.authority.stage("interrupted-reserve"):
                    self.fail("must not yield")
        self.assertEqual(json.loads(self.authority.journal_path.read_text().splitlines()[-1])["reason"], "reserve")
        with self.assertRaises(RuntimeError):
            self.authority.snapshot()
        with self.assertRaises(RuntimeError):
            self.authority.initialize()

    def test_exact_byte_adoption_preserves_receipts_and_charges(self) -> None:
        self.path.parent.mkdir(parents=True)
        p = {"identity": self.authority._identity, "charged_cpu_seconds": 900.0457602360001,
             "charged_gpu_seconds": 0.0, "stages": [], "head": "0" * 64}
        for name, cpu in (("authority-tests-engineering-01", .029538136),
                          ("authority-compile-engineering-02", .0162221)):
            core = {"name": name, "scientific": False, "status": "succeeded", "started": 1.,
                    "ended": 2., "cpu_seconds": cpu, "gpu_seconds": 0., "failure": None, "parent": p["head"]}
            p["stages"].append({**core, "hash": mod._digest(core)})
            p["head"] = p["stages"][-1]["hash"]
        raw = (json.dumps(p, indent=2) + "\n").encode()
        self.path.write_bytes(raw)
        receipt = self.path.parent / "old-receipt.json"
        receipt.write_bytes(b"original receipt\n")
        self.authority._private = False
        with self.assertRaisesRegex(RuntimeError, "raw hash"):
            self.authority.adopt_initialized_authority()
        with patch.object(mod, "LEGACY_SHA256", hashlib.sha256(raw).hexdigest()):
            adopted = self.authority.adopt_initialized_authority()
            self.assertEqual(self.path.read_bytes(), raw)
            self.assertEqual(self.authority.legacy_path.read_bytes(), raw)
            self.assertEqual(receipt.read_bytes(), b"original receipt\n")
            self.assertEqual(adopted["stages"], p["stages"])
            self.assertEqual(adopted["charged_cpu_seconds"], 900.0457602360001)
            with self.assertRaises(RuntimeError):
                self.authority.adopt_initialized_authority()
            with self.authority.stage("post-adoption"):
                pass
            self.assertEqual(self.authority.snapshot()["stages"][:2], p["stages"])


if __name__ == "__main__":
    unittest.main()
