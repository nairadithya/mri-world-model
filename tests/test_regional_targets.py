"""Source-volume conservation, coarse-sampling regression, and cache CLI checks."""

from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np

from src.preprocessing.geometry import GEOMETRY_CONTRACT, Grid
from src.data.regional_targets import load_regional_targets
from src.preprocessing.regional_targets import (
    REGIONAL_TARGET_CONTRACT,
    regional_volumes,
)


class RegionalTargetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def mask(self, data, affine=None, name="mask.nii.gz"):
        path = self.root / name
        nib.save(
            nib.Nifti1Image(np.asarray(data), np.eye(4) if affine is None else affine),
            path,
        )
        return path

    def test_source_voxel_survives_when_coarse_sampling_erases_it(self):
        data = np.zeros((8, 8, 8), dtype=np.uint8)
        data[1, 1, 1] = 2
        source = self.mask(data, np.diag([2.0, 2.0, 2.0, 1.0]))
        coarse_affine = np.diag([8.0, 8.0, 8.0, 1.0])
        coarse_affine[:3, 3] = 3.0
        coarse = resample_from_to(nib.load(source), ((2, 2, 2), coarse_affine), order=0)
        self.assertEqual(int(np.asarray(coarse.dataobj).sum()), 0)
        result = regional_volumes(
            source, (2, 1, 3), Grid((8, 8, 8), np.diag([2.0, 2.0, 2.0, 1.0]))
        )
        expected = np.zeros((8, 3))
        expected[0, 0] = 8.0
        np.testing.assert_allclose(result.volumes_mm3, expected, rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            result.source_mm3, [8.0, 0.0, 0.0], rtol=0, atol=1e-12
        )
        np.testing.assert_array_equal(result.outside_mm3, [0.0, 0.0, 0.0])
        self.assertAlmostEqual(result.log1p_ml[0, 0], 0.007968169649176874, places=15)

    def test_flipped_anisotropic_grid_preserves_label_order_and_volume(self):
        data = np.zeros((5, 5, 5), dtype=np.uint8)
        data[0, 2, 2] = 1
        data[1, 2, 2] = 2
        affine = np.diag([2.0, 3.0, 4.0, 1.0])
        affine[0, 3] = 10.0
        reverse = affine.copy()
        reverse[0, 0] = -2.0
        reverse[0, 3] = 18.0
        result = regional_volumes(
            self.mask(data, reverse), (2, 1, 3), Grid((5, 5, 5), affine)
        )
        expected = np.zeros((8, 3))
        expected[7] = [24.0, 24.0, 0.0]
        np.testing.assert_allclose(result.volumes_mm3, expected, rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            result.source_mm3, [24.0, 24.0, 0.0], rtol=0, atol=1e-12
        )

    def test_half_open_boundaries_account_for_outside_foreground(self):
        affine = np.eye(4)
        affine[0, 3] = -0.5
        path = self.mask(np.ones((5, 1, 1), dtype=np.uint8), affine)
        result = regional_volumes(path, (1,), Grid((2, 2, 2), np.eye(4)))
        np.testing.assert_array_equal(
            result.volumes_mm3[:, 0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        )
        np.testing.assert_array_equal(result.outside_mm3, [3.0])
        np.testing.assert_array_equal(result.source_mm3, [5.0])

    def test_sheared_source_assigns_physical_centers(self):
        data = np.zeros((4, 4, 4), dtype=np.uint8)
        data[0, 0, 0] = 1
        data[0, 3, 0] = 1
        affine = np.eye(4)
        affine[0, 1] = 0.5
        result = regional_volumes(
            self.mask(data, affine), (1,), Grid((4, 4, 4), np.eye(4))
        )
        np.testing.assert_array_equal(
            result.volumes_mm3[:, 0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        )

    def test_reference_resolution_does_not_change_physical_regions(self):
        data = np.zeros((8, 8, 8), dtype=np.uint8)
        data[1, 1, 1] = 1
        data[4, 5, 7] = 1
        path = self.mask(data)
        coarse_affine = np.diag([4.0, 4.0, 4.0, 1.0])
        coarse_affine[:3, 3] = 1.5
        fine = regional_volumes(path, (1,), Grid((8, 8, 8), np.eye(4)))
        coarse = regional_volumes(path, (1,), Grid((2, 2, 2), coarse_affine))
        np.testing.assert_array_equal(
            fine.volumes_mm3[:, 0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        )
        np.testing.assert_array_equal(coarse.volumes_mm3, fine.volumes_mm3)

    def test_empty_mask_produces_exact_zero_volumes(self):
        result = regional_volumes(
            self.mask(np.zeros((4, 4, 4), dtype=np.uint8)),
            (2, 1, 3),
            Grid((4, 4, 4), np.eye(4)),
        )
        np.testing.assert_array_equal(result.volumes_mm3, np.zeros((8, 3)))
        np.testing.assert_array_equal(result.log1p_ml, np.zeros((8, 3)))
        np.testing.assert_array_equal(result.source_mm3, [0.0, 0.0, 0.0])

    def test_invalid_masks_and_label_declarations_are_rejected(self):
        for value in (np.nan, 0.5, 4.0):
            with self.subTest(value=value):
                data = np.zeros((2, 2, 2), dtype=np.float32)
                data[0, 0, 0] = value
                with self.assertRaisesRegex(ValueError, "finite declared integer"):
                    regional_volumes(
                        self.mask(data), (1, 2, 3), Grid((2, 2, 2), np.eye(4))
                    )
        path = self.mask(np.zeros((2, 2, 2), dtype=np.uint8))
        for labels in ((), (0,), (1, 1), (1.5,)):
            with self.assertRaisesRegex(ValueError, "distinct positive integers"):
                regional_volumes(path, labels, Grid((2, 2, 2), np.eye(4)))

    def fixture(self):
        data = np.zeros((8, 8, 8), dtype=np.uint8)
        data[1, 1, 1] = 2
        mask = self.mask(data)
        grid = Grid((8, 8, 8), np.eye(4)).record()
        prepared = {
            "schema": "r38-prepared-v3",
            "visits": {
                "v1": {
                    "raw_mask_sha256": hashlib.sha256(mask.read_bytes()).hexdigest(),
                    "visit": {
                        "patient_id": "p1",
                        "pregrid": grid,
                        "mask_relative": mask.name,
                        "labels": {
                            "necrotic_non_enhancing": 2,
                            "enhancing": 1,
                            "edema_flair": 3,
                        },
                        "values_mm3": {
                            "necrotic_non_enhancing": 1.0,
                            "enhancing": 0.0,
                            "edema_flair": 0.0,
                        },
                    },
                    "alignment": {
                        "geometry_contract": GEOMETRY_CONTRACT,
                        "image_pregrid": grid,
                    },
                }
            },
        }
        members = {
            "rows": {
                "fitting": [{"visit_id": "v1", "patient_id": "p1"}],
                "selection": [],
                "scoring": [],
            }
        }
        for name, value in (("prepared.json", prepared), ("membership.json", members)):
            (self.root / name).write_text(json.dumps(value))
        return prepared, members

    def command(self):
        script = (
            Path(__file__).resolve().parents[1] / "scripts/build_regional_targets.py"
        )
        return [
            sys.executable,
            "-B",
            str(script),
            "--prepared",
            str(self.root / "prepared.json"),
            "--membership",
            str(self.root / "membership.json"),
            "--mask-root",
            str(self.root),
            "--output",
            str(self.root / "targets-v1"),
        ]

    def test_cli_writes_physical_targets_and_refuses_overwrite(self):
        self.fixture()
        completed = subprocess.run(self.command(), capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        receipt = json.loads(completed.stdout)
        self.assertEqual(receipt["contract"], REGIONAL_TARGET_CONTRACT)
        out = self.root / "targets-v1"
        with np.load(out / "fitting.npz", allow_pickle=False) as a:
            self.assertEqual(a["visit_ids"].tolist(), ["v1"])
            expected = np.zeros((1, 8, 3))
            expected[0, 0, 0] = 1.0
            np.testing.assert_array_equal(a["volumes_mm3"], expected)
            self.assertAlmostEqual(
                a["log1p_ml"][0, 0, 0], 0.0009995003330835331, places=15
            )
            self.assertNotIn("y", a.files)
        before = (out / "manifest-private.json").read_bytes()
        loaded = load_regional_targets(out, "fitting", ["v1"])
        np.testing.assert_array_equal(loaded.volumes_mm3, expected)
        self.assertAlmostEqual(
            loaded.log1p_ml[0, 0, 0], 0.0009995003330835331, places=15
        )
        with self.assertRaisesRegex(ValueError, "cohort order"):
            load_regional_targets(out, "fitting", ["another-visit"])
        again = subprocess.run(self.command(), capture_output=True, text=True)
        self.assertNotEqual(again.returncode, 0)
        self.assertIn("already exists", again.stderr)
        self.assertEqual((out / "manifest-private.json").read_bytes(), before)

    def test_bad_source_hash_leaves_no_published_cache(self):
        prepared, _ = self.fixture()
        prepared["visits"]["v1"]["raw_mask_sha256"] = "0" * 64
        (self.root / "prepared.json").write_text(json.dumps(prepared))
        completed = subprocess.run(self.command(), capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("hash differs", completed.stderr)
        self.assertFalse((self.root / "targets-v1").exists())
        self.assertFalse(list(self.root.glob(".targets-v1-*")))

    def test_loader_rejects_legacy_target_units(self):
        (self.root / "manifest-private.json").write_text(
            json.dumps({"units": "log1p_counts"})
        )
        with self.assertRaisesRegex(ValueError, "incompatible regional targets"):
            load_regional_targets(self.root, "fitting", ["v1"])

    def test_loader_rejects_changed_array_bytes(self):
        self.fixture()
        completed = subprocess.run(self.command(), capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        out = self.root / "targets-v1"
        with (out / "fitting.npz").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "array hash"):
            load_regional_targets(out, "fitting", ["v1"])

    def test_patient_overlap_is_rejected(self):
        prepared, members = self.fixture()
        prepared["visits"]["v2"] = prepared["visits"]["v1"]
        members["rows"]["scoring"] = [{"visit_id": "v2", "patient_id": "p1"}]
        (self.root / "prepared.json").write_text(json.dumps(prepared))
        (self.root / "membership.json").write_text(json.dumps(members))
        completed = subprocess.run(self.command(), capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("patient groups overlap", completed.stderr)


if __name__ == "__main__":
    unittest.main()
