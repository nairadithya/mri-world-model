"""Synthetic remote rejection checks for R37 review findings B1 through B3."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np

from src.harness.analysis.spatial_dynamics_data import canonical_hash, validate_visit_geometry
from src.harness.analysis.spatial_dynamics_report import (
    assessment_identity, expected_cohorts, gaussian_bundle, publish_aggregate, verify_assessment,
)
from src.harness.analysis.spatial_dynamics_runtime import BudgetAuthority, budget_stage, production_authority
from src.preprocessing.geometry import GEOMETRY_CONTRACT, Grid, _content_hash


class GeometryGuards(unittest.TestCase):
    def fixture(self, root, *, native=False, stem="CT1", physical=None):
        directory = root / "synthetic-patient" / "week-0"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{stem}.nii.gz"
        affine = np.diag([2., 2., 2., 1.]) if native else np.eye(4)
        values = np.arange(64, dtype=np.float32).reshape(4,4,4)
        nib.save(nib.Nifti1Image(values, affine), path)
        image = nib.load(path)
        grid = Grid(tuple(image.shape), image.affine).record()
        pregrid = grid if native else (physical or Grid((8,8,8), np.diag([2.,2.,2.,1.])).record())
        record = {"tensor_grid": grid, "pregrid": pregrid, "content_sha256": _content_hash(image, values)}
        if not native:
            record.update(reconstruction_verified=True, resize="trilinear-align_corners-false-then-nonzero-zscore",
                          pregrid_content_sha256="frozen-existing-evidence")
        key = f"synthetic-patient/week-0/{stem}"
        manifest = {"schema_version": 1, "geometry_contract": GEOMETRY_CONTRACT, "images": {key: record}}
        return path, manifest, pregrid

    def test_identity_and_native_files_use_saved_content_and_pregrid(self):
        for native in (False, True):
            with tempfile.TemporaryDirectory() as td:
                path, manifest, physical = self.fixture(Path(td), native=native)
                result = validate_visit_geometry([path], manifest, physical)
                self.assertEqual(result[str(path)]["prepared_pregrid_sha256"], canonical_hash(physical))
                self.assertEqual(result[str(path)]["file_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
                image = nib.load(path); affine = image.affine.copy(); affine[0,3] += 1
                nib.save(nib.Nifti1Image(image.get_fdata(dtype=np.float32), affine), path)
                with self.assertRaisesRegex(ValueError, "grid/header or content"):
                    validate_visit_geometry([path], manifest, physical)

    def test_header_change_during_geometry_read_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path, manifest, physical = self.fixture(Path(td))
            image = nib.load(path)
            original_values = image.get_fdata(dtype=np.float32).copy()
            original_read = image.get_fdata
            def read_then_change(*args, **kwargs):
                values = original_read(*args, **kwargs)
                changed_affine = image.affine.copy()
                changed_affine[0,3] += 1
                nib.save(nib.Nifti1Image(values, changed_affine), path)
                return values
            # The loaded header and returned values remain valid old evidence,
            # while the on-disk file changes before geometry checking finishes.
            with patch.object(nib, "load", return_value=image), patch.object(image, "get_fdata", side_effect=read_then_change):
                with self.assertRaisesRegex(ValueError, "changed during geometry verification"):
                    validate_visit_geometry([path], manifest, physical)
            actual = nib.load(path)
            np.testing.assert_array_equal(actual.get_fdata(dtype=np.float32), original_values)
            self.assertFalse(np.array_equal(actual.affine, image.affine))

    def test_stale_content_missing_verification_and_pregrid_mismatch_fail(self):
        with tempfile.TemporaryDirectory() as td:
            path, manifest, physical = self.fixture(Path(td))
            key = next(iter(manifest["images"]))
            for field, value in [("content_sha256", "stale"), ("reconstruction_verified", False)]:
                changed = copy.deepcopy(manifest); changed["images"][key][field] = value
                with self.assertRaises(ValueError):
                    validate_visit_geometry([path], changed, physical)
            changed = copy.deepcopy(physical); changed["affine"][0][3] += 1
            with self.assertRaisesRegex(ValueError, "prepared visit pregrid"):
                validate_visit_geometry([path], manifest, changed)
            image = nib.load(path)
            nib.save(nib.Nifti1Image(image.get_fdata(dtype=np.float32)+1, image.affine), path)
            with self.assertRaisesRegex(ValueError, "grid/header or content"):
                validate_visit_geometry([path], manifest, physical)

    def test_incompatible_modalities_and_stale_native_record_fail(self):
        with tempfile.TemporaryDirectory() as td:
            first, manifest, physical = self.fixture(Path(td))
            altered = Grid((8,8,8), np.diag([3.,3.,3.,1.])).record()
            second, second_manifest, _ = self.fixture(Path(td), stem="T1", physical=altered)
            manifest["images"].update(second_manifest["images"])
            with self.assertRaisesRegex(ValueError, "prepared visit pregrid"):
                validate_visit_geometry([first, second], manifest, physical)
        with tempfile.TemporaryDirectory() as td:
            path, manifest, physical = self.fixture(Path(td), native=True)
            key = next(iter(manifest["images"])); manifest["images"][key]["content_sha256"] = "stale"
            with self.assertRaises(ValueError):
                validate_visit_geometry([path], manifest, physical)


class AuthorityGuards(unittest.TestCase):
    def authority(self, root):
        path = root / "continuation_budget.json"
        auth = {"started_at": "synthetic", "deadline_unix": time.time()+1000,
                "gpu_seconds_limit": 10000, "cpu_seconds_limit": 10000}
        path.write_text(json.dumps({**auth, "setup_cpu_charge_seconds": 900., "stages": []}))
        authority = BudgetAuthority(path, auth)
        authority.initialize()
        return authority

    def test_different_copied_fresh_or_reset_ledger_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            authority = self.authority(Path(td))
            copied = Path(td)/"copy.json"; copied.write_bytes(authority.path.read_bytes())
            with self.assertRaises(ValueError):
                with budget_stage(copied, "fit", authority=authority):
                    pass
            with self.assertRaises(ValueError):
                production_authority().require_path(copied)
            old = authority.path.read_bytes()
            with budget_stage(authority.path, "spent", authority=authority):
                pass
            locked = authority.snapshot(authority.path)
            authority.verify_extension(authority.path, locked)
            authority.path.write_bytes(old)
            with self.assertRaisesRegex(ValueError, "reset or changed"):
                authority.read_verified()

    def test_changed_limits_deadline_and_setup_charge_are_rejected(self):
        for key in ("deadline_unix", "cpu_seconds_limit", "gpu_seconds_limit", "setup_cpu_charge_seconds"):
            with tempfile.TemporaryDirectory() as td:
                authority = self.authority(Path(td))
                ledger = json.loads(authority.path.read_text()); ledger[key] += 1000
                authority.path.write_text(json.dumps(ledger))
                with self.assertRaises(ValueError):
                    with budget_stage(authority.path, "fit", authority=authority):
                        pass

    def test_exclusive_reservation_prevents_concurrent_stage(self):
        with tempfile.TemporaryDirectory() as td:
            authority = self.authority(Path(td))
            second = BudgetAuthority(authority.path, authority.authorization)
            with authority.lock():
                with self.assertRaisesRegex(RuntimeError, "exclusive budget reservation"):
                    with budget_stage(authority.path, "concurrent", authority=second):
                        pass
            self.assertEqual(authority.read_verified()[0]["stages"], [])

    def test_failed_report_never_publishes_a_completed_aggregate(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); authority = self.authority(root)
            record = {"status": "staged_not_graded", "reporting_approval": None}
            (root/"aggregate-staged.json").write_text(json.dumps(record))
            with self.assertRaises(TimeoutError):
                with budget_stage(authority.path, "report", authority=authority):
                    raise TimeoutError("synthetic report resource exhaustion")
            with self.assertRaises(RuntimeError):
                publish_aggregate(root/"aggregate.json", record, authority)
            self.assertFalse((root/"aggregate.json").exists())
            self.assertEqual(json.loads((root/"aggregate-staged.json").read_text())["status"], "staged_not_graded")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); authority = self.authority(root)
            with budget_stage(authority.path, "report", authority=authority):
                pass
            record = publish_aggregate(root/"aggregate.json", {"reporting_approval": None}, authority)
            self.assertEqual(record["status"], "completed_not_scientifically_reviewed")
            self.assertIsNone(record["reporting_approval"])

    def test_running_history_and_unrelated_locked_tip_block_execution(self):
        with tempfile.TemporaryDirectory() as td:
            authority = self.authority(Path(td))
            locked = authority.snapshot(authority.path)
            locked["locked_journal_tip"] = "not-an-ancestor"
            with self.assertRaises(ValueError):
                authority.verify_extension(authority.path, locked)
            with authority.lock():
                ledger, _ = authority.read_verified()
                ledger["stages"].append({"stage": "fit", "status": "running"})
                authority.persist(ledger, "synthetic-interrupted-reservation")
            with self.assertRaises(RuntimeError):
                with budget_stage(authority.path, "evaluate", authority=authority):
                    pass


class AssessmentGuards(unittest.TestCase):
    def fixture(self):
        rows = [{"row_id": f"r{i}", "patient_id": p, "primary_history": primary,
                 "truth": [1.,2.,3.], "cutoff_measured": [0.,1.,2.]}
                for i, (p, primary) in enumerate([("p",True),("p",True),("q",True),("q",False)])]
        cache = {"rows": rows, "splits": {"unseen26": ["p","q"]}}
        cohorts = expected_cohorts(cache)
        identity = assessment_identity(cache, "contract-a", "fitting-a", "manifest-a")
        def bundles(arm, selected):
            output = []
            for seed in ([42,43,44] if arm in {"H","L","G"} else [None]):
                truth = np.asarray([r["truth"] for r in selected])
                value = gaussian_bundle({"mean": truth, "intrinsic_scale": np.ones_like(truth)}, truth, 1.)
                value["seed"] = seed; output.append(value)
            return output
        result = {**identity, "primary_rows": copy.deepcopy(cohorts["primary"]), "single_rows": copy.deepcopy(cohorts["single"]),
                  "nuisance_rows": copy.deepcopy(cohorts["nuisance"]),
                  "cohorts": {name: {arm: bundles(arm, cohorts[name]) for arm in ("H","L","G","P","E","R")}
                              for name in ("primary","single")},
                  "nuisance": {arm: {condition: bundles(arm, cohorts["nuisance"]) for condition in ("identity","intensity","resolution")}
                               for arm in ("H","L")}}
        return cache, identity, result, copy.deepcopy(identity)

    def test_valid_bound_assessment_and_stale_contract_fitting_rejection(self):
        cache, identity, bundle, marker = self.fixture()
        self.assertTrue(verify_assessment(bundle, marker, cache, identity))
        for key in ("contract_digest", "fitting_completion_sha256", "fitted_artifact_manifest_sha256"):
            changed = copy.deepcopy(bundle); changed[key] = "stale"
            with self.assertRaises(ValueError):
                verify_assessment(changed, marker, cache, identity)
            changed_marker = copy.deepcopy(marker); changed_marker[key] = "stale"
            with self.assertRaises(ValueError):
                verify_assessment(bundle, changed_marker, cache, identity)

    def test_omitted_substituted_or_wrong_nuisance_rows_block_grading(self):
        for field in ("primary_rows", "single_rows", "nuisance_rows"):
            cache, identity, bundle, marker = self.fixture()
            bundle[field] = bundle[field][:-1]
            with self.assertRaises(ValueError):
                verify_assessment(bundle, marker, cache, identity)
        cache, identity, bundle, marker = self.fixture()
        bundle["primary_rows"][0]["truth"] = [999.]*3
        with self.assertRaises(ValueError):
            verify_assessment(bundle, marker, cache, identity)
        cache, identity, bundle, marker = self.fixture()
        bundle["nuisance_rows"][0] = copy.deepcopy(cache["rows"][1])
        with self.assertRaises(ValueError):
            verify_assessment(bundle, marker, cache, identity)

    def test_missing_duplicate_nuisance_seeds_and_arms_block_grading(self):
        for change in ("missing_seed", "duplicate_seed", "missing_arm", "bad_shape"):
            cache, identity, bundle, marker = self.fixture()
            if change == "missing_seed":
                bundle["nuisance"]["H"]["resolution"].pop()
            elif change == "duplicate_seed":
                bundle["nuisance"]["L"]["identity"][0]["seed"] = 43
            elif change == "missing_arm":
                del bundle["cohorts"]["single"]["R"]
            else:
                bundle["cohorts"]["primary"]["H"][0]["mean"] = np.zeros((1,3))
            with self.assertRaises(ValueError):
                verify_assessment(bundle, marker, cache, identity)


if __name__ == "__main__":
    unittest.main()
