"""Behavioral checks for MRI-only signed feature differences."""
from pathlib import Path
import copy
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from src.model.representation import SpatialState
from src.model.delta_representation import DeltaRepresentation, LocalLinearProbe
from src.harness.analysis.delta_protocol import balanced_loss, partition_errors, objective, nuisance_loss
from src.harness.analysis.representation import optimizer_step


class DeltaTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.model = DeltaRepresentation(SpatialState((8, 16, 24, 32)))
        self.source = torch.randn(1, 4, 16, 16, 16)
        self.target = torch.randn_like(self.source)
        self.availability = torch.tensor([[1., 0., 1., 1.]])

    def forward(self, source=None, target=None):
        return self.model(self.source if source is None else source, self.availability,
                          self.target if target is None else target, self.availability)

    def test_identity_features_are_exact_zero_but_head_is_measured(self):
        output = self.forward(target=self.source)
        self.assertTrue(all(torch.count_nonzero(d) == 0 for d in output["differences"]))
        self.assertTrue(torch.isfinite(output["mask_change"]).all())
        self.assertGreater(float(output["mask_change"].abs().mean()), 0)

    def test_source_changes_predictions(self):
        first = self.forward()
        second = self.forward(source=self.source * 0)
        self.assertGreater(float((first["mask_change"] - second["mask_change"]).abs().mean()), 1e-5)
        self.assertTrue(torch.equal(first["target"]["volume"], second["target"]["volume"]))

    def test_missing_sequence_is_exactly_masked(self):
        changed = self.source.clone()
        changed[:, 1] = 1000
        first = self.forward()
        second = self.forward(source=changed)
        self.assertTrue(torch.equal(first["mask_change"], second["mask_change"]))

    def test_signed_targets_and_partition_balancing(self):
        target = torch.tensor([1., -1., 0., 0.]).reshape(1, 1, 1, 1, 4)
        prediction = torch.tensor([0., 0., 0.2, 0.4]).reshape_as(target)
        self.assertAlmostEqual(float(balanced_loss(prediction, target)), 0.65, places=6)
        self.assertAlmostEqual(float(partition_errors(prediction, target)["changed"]), 1)
        self.assertAlmostEqual(float(partition_errors(prediction, target)["unchanged"]), 0.3, places=6)

    def test_all_zero_and_all_changed_have_one_partition(self):
        prediction = torch.full((1, 3, 2, 2, 2), 0.25, requires_grad=True)
        self.assertEqual(float(balanced_loss(prediction, torch.zeros_like(prediction))), 0.25)
        loss = balanced_loss(prediction, torch.ones_like(prediction))
        self.assertEqual(float(loss), 0.75)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(float(prediction.grad.abs().sum()), 0)

    def test_half_partition_reduction_stays_finite(self):
        prediction = torch.full((1, 3, 96, 96, 96), 0.5, dtype=torch.float16, requires_grad=True)
        target = torch.ones_like(prediction)
        loss = balanced_loss(prediction, target)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertEqual(float(loss), 0.5)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_reversal_features_are_signed_not_head_guarantees(self):
        first, reverse = self.forward(), self.forward(source=self.target, target=self.source)
        for a, b in zip(first["differences"], reverse["differences"]):
            self.assertTrue(torch.equal(a, -b))
        self.assertGreater(float((first["mask_change"] + reverse["mask_change"]).abs().mean()), 0)

    def test_finite_backward_for_all_objectives_and_nuisance(self):
        a = {"masks": torch.zeros(1, 3, 16, 16, 16), "volume": torch.zeros(1, 3)}
        b = {"masks": torch.ones_like(a["masks"]), "volume": torch.ones_like(a["volume"])}
        pair = {"source": a, "target": b, "change": b["volume"] - a["volume"], "mask_change": b["masks"] - a["masks"]}
        normalization = {"mean": [0., 0., 0.], "scale": [1., 1., 1.], "change_scale": [1., 1., 1.]}
        for arm in ("A", "B", "C", "D"):
            self.model.zero_grad(set_to_none=True)
            output = self.forward()
            loss, _ = objective(output, pair, normalization, arm)
            if arm == "D":
                loss = loss + 0.25 * nuisance_loss(self.forward(target=self.source), normalization)
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None))
            self.assertGreater(float(self.model.encoder.blocks[0].body[0].weight.grad.abs().sum()), 0)
            self.assertEqual(self.model.change_head.weight.grad is None, arm == "A")

    def test_sequential_nuisance_backward_matches_joint_update(self):
        first, second = copy.deepcopy(self.model), copy.deepcopy(self.model)
        normalization = {"mean": [0., 0., 0.], "scale": [1., 1., 1.], "change_scale": [1., 1., 1.]}
        a = {"masks": torch.zeros(1, 3, 16, 16, 16), "volume": torch.zeros(1, 3)}
        b = {"masks": torch.ones_like(a["masks"]), "volume": torch.ones_like(a["volume"])}
        pair = {"source": a, "target": b, "change": b["volume"] - a["volume"], "mask_change": b["masks"] - a["masks"]}
        for index, model in enumerate((first, second)):
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
            scaler = torch.amp.GradScaler("cuda", enabled=False)
            output = model(self.source, self.availability, self.target, self.availability)
            real, _ = objective(output, pair, normalization, "C")
            if index == 1:
                scaler.scale(real).backward()
            synthetic = model(self.source, self.availability, self.source, self.availability)
            zero = 0.25 * nuisance_loss(synthetic, normalization)
            optimizer_step(real + zero if index == 0 else zero, model, optimizer, scaler)
        for a, b in zip(first.parameters(), second.parameters()):
            self.assertTrue(torch.allclose(a, b, atol=1e-6, rtol=1e-5))

    def test_probe_is_affine_with_fixed_resizing(self):
        widths = (8, 16, 24, 32)
        probe = LocalLinearProbe(widths)
        a = [torch.randn(1, w, 16 // 2**i, 16 // 2**i, 16 // 2**i) for i, w in enumerate(widths)]
        b = [torch.randn_like(x) for x in a]
        zero = [torch.zeros_like(x) for x in a]
        self.assertTrue(torch.allclose(probe([x + y for x, y in zip(a, b)]), probe(a) + probe(b) - probe(zero), atol=1e-6))
        with self.assertRaises(ValueError):
            probe(a[-1:])


if __name__ == "__main__":
    unittest.main()
