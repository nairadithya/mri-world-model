import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from src.harness.analysis.joint_dynamics.continuation import adopt_cell
from src.harness.analysis.joint_dynamics.fitting import tensor_hash
from src.harness.analysis.joint_dynamics.protocol import PLAN, digest, save_json


class ContinuationTests(unittest.TestCase):
    def fixture(self, root):
        original, output = root / "original", root / "new"
        (original / "OH_42").mkdir(parents=True)
        (output / "OH_42").mkdir(parents=True)
        order = [["row"] * PLAN.updates_per_epoch for _ in range(PLAN.epochs)]
        save_json(original / "order_42.json", {"orders": order})
        save_json(output / "order_42.json", {"orders": order})
        initialization = {"tensor_sha256": "a" * 64}
        saved = {
            "state_dict": {"weight": torch.tensor([2.0, 3.0])},
            "optimizer": {"state": {0: {"step": torch.tensor(5240.0)}}},
            "epoch": 20,
            "updates": 5240,
            "arm": "OH",
            "seed": 42,
            "normalizer": {"mean": [1, 2, 3]},
            "initialization": initialization,
            "order_sha256": digest(original / "order_42.json"),
            "contract_digest": "b" * 64,
            "implementation_source": {"original": True},
            "execution_scope": "fixed_r38_study",
        }
        checkpoint = original / "OH_42/epoch20.pt"
        torch.save(saved, checkpoint)
        record = {
            "row_id": "row",
            "cutoff_mask_sha256": "maskhash",
            "current_supervision_visit_id": "visit",
            "future_image_forwards": 0,
            "earlier_state_supervision": 0,
        }
        with (original / "OH_42/updates.jsonl").open("w") as stream:
            for index in range(5240):
                stream.write(
                    json.dumps(
                        {**record, "epoch": index // 262 + 1, "update": index % 262 + 1}
                    )
                    + "\n"
                )
        data = SimpleNamespace(
            normalizer=saved["normalizer"],
            by_row={"row": SimpleNamespace(row_id="row", cutoff_visit_id="visit")},
            visits={"visit": {"mask_sha256": "maskhash"}},
        )
        identity = {
            "contract_digest": "c" * 64,
            "implementation_source": {"continuation": True},
            "execution_scope": "fixed_r38_study",
        }
        return data, original, output, initialization, checkpoint, identity

    def test_adopts_without_weight_changes_or_new_updates_and_preserves_origin(self):
        with tempfile.TemporaryDirectory() as td:
            data, original, output, init, checkpoint, identity = self.fixture(Path(td))
            before = digest(checkpoint)
            result = adopt_cell(
                data, original, output, "OH_42", identity, init, before, "b" * 64
            )
            restored = torch.load(output / "OH_42/epoch20.pt", weights_only=False)
            torch.testing.assert_close(
                restored["state_dict"]["weight"],
                torch.tensor([2.0, 3.0]),
                rtol=0,
                atol=0,
            )
            self.assertEqual(result["updates"], 5240)
            self.assertEqual(restored["training_origin"]["new_optimizer_updates"], 0)
            self.assertEqual(restored["training_origin"]["contract_digest"], "b" * 64)
            self.assertEqual(restored["contract_digest"], "c" * 64)
            self.assertEqual(
                restored["training_origin"]["state_tensor_sha256"],
                tensor_hash(restored["state_dict"]),
            )
            self.assertEqual(digest(checkpoint), before)

    def test_changed_original_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            data, original, output, init, checkpoint, identity = self.fixture(Path(td))
            with self.assertRaisesRegex(ValueError, "checkpoint changed"):
                adopt_cell(
                    data, original, output, "OH_42", identity, init, "0" * 64, "b" * 64
                )

    def test_partial_optimizer_state_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            data, original, output, init, checkpoint, identity = self.fixture(Path(td))
            saved = torch.load(checkpoint, weights_only=False)
            saved["optimizer"]["state"][0]["step"] = torch.tensor(686.0)
            torch.save(saved, checkpoint)
            with self.assertRaisesRegex(ValueError, "optimizer counts"):
                adopt_cell(
                    data,
                    original,
                    output,
                    "OH_42",
                    identity,
                    init,
                    digest(checkpoint),
                    "b" * 64,
                )

    def test_future_supervision_in_original_log_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            data, original, output, init, checkpoint, identity = self.fixture(Path(td))
            path = original / "OH_42/updates.jsonl"
            path.write_text(
                path.read_text().replace(
                    '"future_image_forwards": 0', '"future_image_forwards": 1', 1
                )
            )
            with self.assertRaisesRegex(ValueError, "supervision"):
                adopt_cell(
                    data,
                    original,
                    output,
                    "OH_42",
                    identity,
                    init,
                    digest(checkpoint),
                    "b" * 64,
                )


if __name__ == "__main__":
    unittest.main()
