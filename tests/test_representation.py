from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nibabel as nib
import numpy as np
import torch

from src.data.representation import CONTRACT, ObservedPairs, Visits, prepare, save_json
from src.harness.analysis.representation import (cache_features, metrics, nested_predictions,
                                                optimizer_step, paired_interval, state_loss, transfer_features)
from src.model.representation import ObservedChange, SpatialState, nuisance


class RepresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def inputs(self):
        torch.manual_seed(42)
        return torch.randn(1, 4, 16, 16, 16), torch.ones(1, 4)

    def test_state_shape_backward_and_missing_sequence(self):
        image, availability = self.inputs()
        model = SpatialState()
        availability[:, 2] = 0
        output = model(image, availability)
        changed = image.clone(); changed[:, 2] = 10000
        actual = model(changed, availability)
        torch.testing.assert_close(output["volume"], actual["volume"], rtol=0, atol=0)
        self.assertEqual(tuple(output["embedding"].shape), (1, 128))
        self.assertEqual(tuple(output["mask_logits"].shape), (1, 3, 16, 16, 16))
        batch = {"masks": torch.zeros_like(output["mask_logits"]), "volume": torch.ones(1, 3)}
        loss = state_loss(output, batch, torch.zeros(3), torch.ones(3))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertGreater(float(model.blocks[0].body[0].weight.grad.norm()), 0)

    def test_amp_overflow_backs_off_without_changing_weights(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler("cpu", init_scale=65536)
        for _ in range(5):
            optimizer.zero_grad(set_to_none=True)
            loss = (model.weight * 10).half().float().sum()
            step = optimizer_step(loss, model, optimizer, scaler)
            if step["updated"]:
                break
            self.assertEqual(float(model.weight.detach()), 1)
            self.assertEqual(step["scale_after"], step["scale_before"] / 2)
        self.assertTrue(step["updated"])
        self.assertAlmostEqual(float(model.weight.detach()), 0.9, places=5)
        self.assertEqual(scaler.get_scale(), 32768)

    def test_full_precision_nonfinite_gradient_remains_fatal(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        with self.assertRaisesRegex(ValueError, "full-precision gradients"):
            optimizer_step(model.weight.sqrt().sum(), model, optimizer, scaler)
        self.assertEqual(float(model.weight.detach()), 0)

    def test_half_precision_dense_reductions_match_float(self):
        shape = (1, 3, 96, 96, 96)
        half = {"mask_logits": torch.zeros(shape, dtype=torch.float16, requires_grad=True), "volume": torch.zeros(1, 3)}
        full = {**half, "mask_logits": half["mask_logits"].detach().float()}
        batch = {"masks": torch.ones(shape), "volume": torch.zeros(1, 3)}
        actual = state_loss(half, batch, torch.zeros(3), torch.ones(3))
        expected = state_loss(full, batch, torch.zeros(3), torch.ones(3))
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
        actual.backward()
        self.assertTrue(torch.isfinite(half["mask_logits"].grad).all())
        self.assertGreater(float(half["mask_logits"].grad.abs().sum()), 0)

    def test_later_only_ignores_source_exactly(self):
        image, availability = self.inputs()
        model = ObservedChange(SpatialState(), "later_only")
        a = model(image, availability, image + 1, availability)
        b = model(image * 100, availability, image + 1, availability)
        torch.testing.assert_close(a["change"], b["change"], rtol=0, atol=0)
        torch.testing.assert_close(a["mask_change"], b["mask_change"], rtol=0, atol=0)

    def test_paired_model_has_source_dependence(self):
        image, availability = self.inputs()
        image.requires_grad_()
        model = ObservedChange(SpatialState(), "paired")
        output = model(image, availability, image.detach() + 1, availability, dense=False)
        gradient = torch.autograd.grad(output["change"].sum(), image)[0]
        self.assertGreater(float(gradient.abs().sum()), 0)

    def test_frozen_model_freezes_only_state_encoder(self):
        model = ObservedChange(SpatialState(), "frozen")
        self.assertTrue(all(not p.requires_grad for p in model.encoder.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.comparison.parameters()))

    def test_invalid_availability_rejected(self):
        image, availability = self.inputs()
        model = SpatialState()
        for invalid in (torch.zeros_like(availability), availability * 0.5):
            with self.assertRaises(ValueError):
                model(image, invalid)

    def test_nuisance_identity_background_and_repeatability(self):
        image, _ = self.inputs()
        image[:, :, :3] = 0
        torch.testing.assert_close(nuisance(image, "identity"), image, rtol=0, atol=0)
        for kind in ("intensity", "noise", "resolution", "bias"):
            actual = nuisance(image, kind)
            self.assertTrue(torch.isfinite(actual).all())
            self.assertEqual(int(torch.count_nonzero(actual[:, :, :3])), 0)
            torch.testing.assert_close(actual, nuisance(image, kind), rtol=0, atol=0)

    def test_patient_uniform_is_not_pooled(self):
        truth = np.zeros((3, 3))
        prediction = np.asarray([[3, 3, 3], [0, 0, 0], [0, 0, 0]])
        result = metrics(truth, prediction, np.asarray(["a", "b", "b"]))
        self.assertEqual(result["patient_uniform_mae"], 1.5)
        self.assertEqual(result["pooled_mae"], 1.0)
        interval = paired_interval(truth, truth, prediction, np.asarray(["a", "b", "b"]))
        self.assertEqual(interval["difference"], -1.5)

    def test_nested_probe_keeps_patients_together(self):
        groups = np.asarray([f"p{i}" for i in range(8) for _ in range(2)])
        x = np.arange(16, dtype=float)[:, None]
        y = np.repeat(x, 3, axis=1)
        protocol = {"k": 2, "folds": {f"p{i}": i % 2 for i in range(8)}}
        prediction, mean, alphas = nested_predictions(x, y, groups, protocol)
        self.assertEqual(len(alphas), 2)
        self.assertLess(float(np.abs(prediction - y).mean()), float(np.abs(mean - y).mean()))
        for patient in set(groups):
            self.assertTrue(np.all(mean[groups == patient] == mean[groups == patient][0]))

    def test_cache_without_visit_ids_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.pt"
            torch.save({"schema_version": 2, "provenance": {"clinical_schema": "survival_free_v1"},
                        "patients": {"p": {"vision": torch.ones(1, 2)}}}, path)
            with self.assertRaisesRegex(ValueError, "visit identifiers"):
                cache_features({"visits": []}, str(path))

    def test_transfer_rejects_overlapping_patients(self):
        metadata = {"protocol": {"encoder_unseen": ["p"]}, "visits": [{"patient_id": "p"}]}
        with self.assertRaisesRegex(ValueError, "overlap"):
            transfer_features(metadata, {}, metadata, {}, "unused.json", provenance={})

    def test_real_geometry_preparation_fixture_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            visits = []
            for index in (0, 1):
                directory = root / "mri" / "Patient-001" / f"week-00{index}"
                directory.mkdir(parents=True)
                image = np.arange(16 ** 3, dtype=np.float32).reshape(16, 16, 16)
                affine = np.diag([2.0, 2.0, 2.0, 1.0])
                nib.save(nib.Nifti1Image(image, affine), directory / "CT1.nii.gz")
                mask_dir = root / "masks" / "Patient-001" / f"week-00{index}"
                mask_dir.mkdir(parents=True)
                mask = np.zeros((16, 16, 16), np.uint8)
                mask[4:8, 4:8, 4:8] = 1
                nib.save(nib.Nifti1Image(mask, affine), mask_dir / "mask.nii.gz")
                visits.append({"visit_id": f"v{index}", "patient_id": "Patient-001", "visit": f"week-00{index}",
                               "mask_specs": [{"relative": f"Patient-001/week-00{index}/mask.nii.gz", "label": label}
                                              for label in (2, 1, 3)],
                               "values_mm3": {"necrotic_non_enhancing": 0, "enhancing": 512, "edema_flair": 0}})
            inventory = {"contract": CONTRACT, "visits": visits, "pairs": [{"row_id": "r", "patient_id": "Patient-001",
                         "source_visit_id": "v0", "target_visit_id": "v1"}], "protocol": {}, "source_hashes": {},
                         "geometry": {"schema_version": 1, "geometry_contract": "affine-aligned-lesion-v1", "images": {}}}
            save_json(root / "inventory.json", inventory)
            args = (str(root / "inventory.json"), str(root / "mri"), str(root / "masks"), str(root / "prepared"))
            first = prepare(*args)
            self.assertEqual(first, prepare(*args))
            data = Visits(root / "prepared")
            pair = ObservedPairs(data)[0]
            torch.testing.assert_close(pair["change"], torch.zeros(3), rtol=0, atol=0)
            self.assertEqual(tuple(data[0]["image"].shape), (4, 96, 96, 96))
            self.assertAlmostEqual(float(data[0]["volume"][1]), float(np.log1p(512)), places=5)


if __name__ == "__main__":
    unittest.main()
