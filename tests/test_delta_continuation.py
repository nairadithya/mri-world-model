"""Mechanical orchestration guards, independent of representation-quality claims."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from scripts.continue_delta_representation_run import completed_probe_seed, ensure_probe_launch, merge, wait


class ContinuationTests(unittest.TestCase):
    def test_merge_is_idempotent_and_refuses_changed_bytes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source/seed42/A/final.pt"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"original")
            destination = root / "final"
            merge(root / "source", destination)
            merge(root / "source", destination)
            source.write_bytes(b"different")
            with self.assertRaises(ValueError):
                merge(root / "source", destination)
            self.assertEqual((destination / "seed42/A/final.pt").read_bytes(), b"original")

    def test_queued_jobs_wait_without_launching_or_retrying(self):
        records = []
        with patch("scripts.continue_delta_representation_run.status", side_effect=["QUEUED", "RUNNING", "COMPLETE"]), \
             patch("scripts.continue_delta_representation_run.time.sleep") as sleep:
            wait(["owned/job"], 3600, lambda *args: records.append(args))
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual([entry[1]["owned/job"] for entry in records], ["QUEUED", "RUNNING", "COMPLETE"])

    def test_failed_job_stops(self):
        with patch("scripts.continue_delta_representation_run.status", return_value="ERROR"), \
             patch("scripts.continue_delta_representation_run.time.sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "did not complete"):
                wait(["owned/job"], 3600, lambda *_: None)
        sleep.assert_not_called()

    def test_existing_launch_receipt_never_pushes_again(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "logs").mkdir()
            (root / "logs/push-probes42.log").write_text("legacy push receipt")
            with patch("scripts.continue_delta_representation_run.command") as command:
                ensure_probe_launch(root, 42, lambda *_: None)
            command.assert_not_called()

    def test_new_push_is_preceded_by_exclusive_intent_and_quota_check(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "logs").mkdir()
            calls = []
            def command(args, log):
                calls.append(args)
                if args[1] == "quota":
                    return "GPU 16.77h 13.23h 30.00h"
                self.assertTrue((root / "probes43-launch-intent.json").exists())
                return "pushed"
            with patch("scripts.continue_delta_representation_run.command", side_effect=command):
                ensure_probe_launch(root, 43, lambda *_: None)
                ensure_probe_launch(root, 43, lambda *_: None)
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[1][:3], ["kaggle", "kernels", "push"])
            self.assertIn("12600", calls[1])

    def test_ambiguous_push_failure_preserves_intent_and_forbids_resubmission(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "logs").mkdir()
            with patch("scripts.continue_delta_representation_run.command",
                       side_effect=["GPU 16.77h 13.23h 30.00h", RuntimeError("push response lost")]):
                with self.assertRaisesRegex(RuntimeError, "response lost"):
                    ensure_probe_launch(root, 43, lambda *_: None)
            with patch("scripts.continue_delta_representation_run.command") as command:
                ensure_probe_launch(root, 43, lambda *_: None)
            command.assert_not_called()

    def test_unaffordable_new_work_stops_before_intent_or_push(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("scripts.continue_delta_representation_run.command", return_value="GPU 28h 2h 30h") as command:
                with self.assertRaisesRegex(RuntimeError, "GPU ceiling"):
                    ensure_probe_launch(root, 43, lambda *_: None)
            self.assertEqual(command.call_count, 1)
            self.assertFalse((root / "probes43-launch-intent.json").exists())

    def test_completed_probe_skip_requires_bound_finite_matched_final_weights(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for arm in ("A", "B", "C", "D", "R35"):
                for view in ("difference", "target_only"):
                    path = root / f"matched-final/seed42/{arm}/{view}"
                    path.mkdir(parents=True)
                    (path / "probe.pt").write_bytes(b"fixture")
                    (path / "training.json").write_text(json.dumps({"status": "complete", "history": [
                        {"epoch": epoch, "pair_order_sha256": str(epoch)} for epoch in range(1, 21)]}))
            probe = torch.nn.Linear(1, 1)
            with patch("src.harness.analysis.delta_protocol.load_contract", return_value={}), \
                 patch("src.harness.analysis.delta_representation.load_probe", return_value=probe) as load:
                self.assertTrue(completed_probe_seed(root, 42, root / "r35"))
                self.assertEqual(load.call_count, 10)
                self.assertEqual(Path(load.call_args[0][1]), root / "r35/seed42/state/best.pt")
                with torch.no_grad():
                    probe.weight.fill_(float("nan"))
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    completed_probe_seed(root, 42, root / "r35")
            with patch("src.harness.analysis.delta_protocol.load_contract", return_value={}), \
                 patch("src.harness.analysis.delta_representation.load_probe", side_effect=ValueError("hash mismatch")):
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    completed_probe_seed(root, 42, root / "r35")

    def test_partial_probe_seed_does_not_authorize_skip(self):
        with TemporaryDirectory() as directory:
            self.assertFalse(completed_probe_seed(Path(directory), 43, Path(directory) / "r35"))

    def test_wait_remains_bounded(self):
        with patch("scripts.continue_delta_representation_run.status", return_value="RUNNING"), \
             patch("scripts.continue_delta_representation_run.time.sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "ceiling reached"):
                wait(["owned/job"], 0, lambda *_: None)
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
