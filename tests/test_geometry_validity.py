"""Run with .venv/bin/python tests/test_geometry_validity.py."""
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nibabel as nib
import numpy as np
import torch

from src.preprocessing.geometry import (GEOMETRY_CONTRACT, Grid, align_masks,
                                        export_geometry_manifest, export_image_geometry, image_grid,
                                        require_geometry_contract)
from src.preprocessing.transforms import resize_to, zscore_nonzero
from src.harness.train.lesion import load_masks, selected_image_paths
from src.harness.encode.lesion import mask_specs


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        torch.set_num_threads(2)

    def tearDown(self):
        self.tmp.cleanup()

    def save(self, name, values, affine):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(np.asarray(values), affine), path)
        return path

    def phantom(self):
        affine = np.diag([2., 3., 4., 1.]); affine[0, 3] = 10
        image = self.save("native.nii.gz", np.arange(125, dtype=np.float32).reshape(5, 5, 5), affine)
        reverse = affine.copy(); reverse[0, 0] = -2; reverse[0, 3] = 18
        labels = np.zeros((5, 5, 5), dtype=np.uint8); labels[0, 2, 2] = 1; labels[1, 2, 2] = 2
        mask = self.save("mask.nii.gz", labels, reverse)
        return image, mask

    def processed(self):
        affine = np.eye(4); affine[0, 3] = -20
        data = np.arange(125, dtype=np.float32).reshape(5, 5, 5)
        pre = self.save("source/Patient-001/week-000/_tmp/CT1_bet.nii.gz", data, affine)
        final_data = zscore_nonzero(resize_to(torch.from_numpy(data)[None], (4, 4, 4)))[0].numpy()
        final = self.save("source/Patient-001/week-000/CT1.nii.gz", final_data, np.eye(4))
        return final, pre

    def test_axis_flip_and_native_volume(self):
        image, mask = self.phantom()
        actual = align_masks([(mask, 1), (mask, 2), (mask, 3)], [image], size=(5, 5, 5))
        expected = torch.zeros(3, 5, 5, 5, dtype=torch.bool)
        expected[0, 4, 2, 2] = True; expected[1, 3, 2, 2] = True
        self.assertTrue(torch.equal(actual.masks, expected))
        self.assertEqual(actual.native_volumes_mm3.tolist(), [24., 24., 0.])
        self.assertEqual(actual.provenance["geometry_contract"], GEOMETRY_CONTRACT)
        self.assertFalse(torch.equal(actual.masks[0], torch.from_numpy(np.asarray(nib.load(mask).dataobj) == 1)))

    def test_translated_binary_grid(self):
        image, _ = self.phantom()
        affine = nib.load(image).affine.copy(); affine[0, 3] += 2
        binary = np.zeros((5, 5, 5), dtype=np.uint8); binary[0, 1, 1] = 1
        path = self.save("translated.nii.gz", binary, affine)
        actual = align_masks([(path, None)], [image], size=(5, 5, 5))
        expected = torch.zeros(1, 5, 5, 5, dtype=torch.bool); expected[0, 1, 1, 1] = True
        self.assertTrue(torch.equal(actual.masks, expected))
        self.assertEqual(actual.native_volumes_mm3.item(), 24.)

    def test_identity_requires_reconstructible_provenance(self):
        orphan = self.save("orphan.nii.gz", np.ones((4, 4, 4), dtype=np.float32), np.eye(4))
        with self.assertRaisesRegex(ValueError, "requires verified"):
            image_grid(orphan)
        final, pre = self.processed()
        grid, record = image_grid(final)
        self.assertEqual(grid.shape, (5, 5, 5))
        self.assertEqual(grid.affine[0, 3], -20.)
        self.assertTrue(record["reconstruction_verified"])
        image = nib.load(pre); data = image.get_fdata(dtype=np.float32); data[2, 2, 2] += 100
        nib.save(nib.Nifti1Image(data, image.affine), pre)
        with self.assertRaisesRegex(ValueError, "does not reconstruct"):
            image_grid(final)

    def test_manifest_survives_gzip_and_root_changes_but_rejects_stale_data(self):
        final, _ = self.processed()
        manifest = export_geometry_manifest([final], self.root / "geometry.json")
        copied = self.root / "copy/Patient-001/week-000/CT1.nii"
        copied.parent.mkdir(parents=True)
        original = nib.load(final)
        nib.save(nib.Nifti1Image(original.get_fdata(dtype=np.float32), original.affine), copied)
        grid, record = image_grid(copied, manifest)
        self.assertEqual(grid.shape, (5, 5, 5))
        self.assertTrue(record["reconstruction_verified"])
        changed = original.get_fdata(dtype=np.float32); changed[1, 1, 1] += 1
        nib.save(nib.Nifti1Image(changed, np.eye(4)), copied)
        with self.assertRaisesRegex(ValueError, "stale verified"):
            image_grid(copied, manifest)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            export_geometry_manifest([final, final], self.root / "bad.json")

    def test_per_image_sidecar_works_without_retained_pregrid(self):
        final, pre = self.processed()
        sidecar = export_image_geometry(final, pre)
        self.assertTrue(sidecar.exists())
        pre.unlink()
        grid, record = image_grid(final)
        self.assertEqual(grid.affine[0, 3], -20.)
        self.assertTrue(record["reconstruction_verified"])
        with self.assertRaisesRegex(ValueError, "tensor size differs"):
            align_masks([(final, None)], [final], size=(96, 96, 96))

    def test_mismatched_modality_grids_and_bad_masks_rejected(self):
        image, mask = self.phantom()
        affine = nib.load(image).affine.copy(); affine[1, 3] = 1
        other = self.save("other.nii.gz", np.ones((5, 5, 5), dtype=np.float32), affine)
        with self.assertRaisesRegex(ValueError, "modality physical grids disagree"):
            align_masks([(mask, 1)], [image, other])
        data = np.ones((5, 5, 5), dtype=np.float32); data[0, 0, 0] = np.nan
        bad = self.save("bad-mask.nii.gz", data, affine)
        with self.assertRaisesRegex(ValueError, "nonfinite lesion"):
            align_masks([(bad, None)], [image])
        with self.assertRaises(ValueError):
            Grid((5, 5, 5), np.zeros((4, 4)))

    def test_image_paths_follow_collator_columns_not_dictionary_order(self):
        sample = {"paths": {"T1": [None], "CT1": ["ct1.nii"], "T2": [None], "FLAIR": [None]}}
        actual = selected_image_paths(sample, [0], torch.tensor([[True, False, False, False]]))
        self.assertEqual(actual, [["ct1.nii"]])

    def test_training_uses_the_shared_boundary(self):
        image, mask = self.phantom()
        destination = self.root / "masks/Patient-001/week-000/DeepBraTumIA-segmentation/atlas/segmentation/seg_mask.nii.gz"
        destination.parent.mkdir(parents=True); shutil.copyfile(mask, destination)
        masks, volumes, geometry = load_masks(str(self.root / "masks"), "Patient-001", ["week-000"], [[str(image)]])
        direct = align_masks([(mask, 2), (mask, 1), (mask, 3)], [image])
        self.assertTrue(torch.equal(masks[0], direct.masks))
        self.assertTrue(torch.equal(volumes[0], torch.tensor([24., 24., 0.]).log1p()))
        self.assertEqual(geometry[0]["geometry_contract"], GEOMETRY_CONTRACT)
        visit = {"measurement": {"mask": {"path": str(mask), "labels": {
            "necrotic_non_enhancing": 2, "enhancing": 1, "edema_flair": 3}}}}
        self.assertEqual(mask_specs(visit), [(str(mask), 2), (str(mask), 1), (str(mask), 3)])

    def test_legacy_artifacts_rejected(self):
        with self.assertRaisesRegex(ValueError, "re-encode and retrain"):
            require_geometry_contract({"forecast_input_contract": "history-only-lesion-v2"})
        require_geometry_contract({"geometry_contract": GEOMETRY_CONTRACT})


if __name__ == "__main__":
    unittest.main()
