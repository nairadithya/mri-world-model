"""Mechanical orchestration guards, independent of representation-quality claims."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.continue_delta_representation_run import merge, wait


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

    def test_wait_remains_bounded(self):
        with patch("scripts.continue_delta_representation_run.status", return_value="RUNNING"), \
             patch("scripts.continue_delta_representation_run.time.sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "ceiling reached"):
                wait(["owned/job"], 0, lambda *_: None)
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
