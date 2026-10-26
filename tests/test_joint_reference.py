from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.harness.analysis.joint_dynamics import data
from src.harness.analysis.joint_dynamics.protocol import ArtifactRef, PLAN, digest


class CanonicalReferenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        path = Path(self.directory.name) / 'direct-reference.txt'
        path.write_text('synthetic reference artifact, not historical predictions\n')
        self.artifact = ArtifactRef(path, digest(path))
        self.ids = tuple(f'visit-{i:03d}' for i in range(599))
        self.values = np.arange(599 * 3, dtype=np.float64).reshape(599, 3) / 100
        self.canonical = data.CanonicalDecode(self.ids, self.values)

    def reference(self, indices):
        return data.PartialDirectReference(tuple(self.ids[i] for i in indices),
                                           self.values[list(indices)], self.artifact)

    def test_partial_subsets_and_order_use_explicit_id_lookup(self):
        for indices in ((0,), (598, 17, 2), tuple(reversed(range(599)))):
            with self.subTest(size=len(indices)):
                coverage = data.validate_direct(self.reference(indices), self.canonical)
                self.assertEqual(coverage.supplied_visit_ids, tuple(self.ids[i] for i in indices))
                self.assertEqual(set(coverage.unreferenced_visit_ids), set(self.ids) - set(coverage.supplied_visit_ids))
                self.assertEqual(coverage.max_native_absolute_error, 0)
                self.assertEqual(coverage.to_dict()['canonical_count'], 599)
                self.assertEqual(coverage.to_dict()['artifact'], self.artifact.to_dict())
        reordered = data.CanonicalDecode(self.ids[::-1], self.values[::-1])
        self.assertEqual(data.validate_direct(self.reference((25, 0, 598)), reordered).max_native_absolute_error, 0)

    def test_reference_rejects_empty_duplicate_unknown_nonfinite_and_wrong_shapes(self):
        for ids, values in (((), np.empty((0, 3))),
                            ((self.ids[0], self.ids[0]), self.values[:2]),
                            ((self.ids[0],), np.ones((1, 2))),
                            ((self.ids[0],), np.ones(3)),
                            ((self.ids[0],), np.full((1, 3), np.nan)),
                            ((self.ids[0],), np.full((1, 3), np.inf)),
                            ((self.ids[0],), np.full((1, 3), '1'))):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                data.PartialDirectReference(ids, values, self.artifact)
        unknown = data.PartialDirectReference(('unknown',), np.ones((1, 3)), self.artifact)
        with self.assertRaisesRegex(ValueError, 'unknown'):
            data.validate_direct(unknown, self.canonical)

    def test_every_supplied_value_is_checked_with_the_fixed_tolerance(self):
        reference = self.reference((598, 2, 17))
        for row in range(3):
            for compartment in range(3):
                changed = reference.native.copy()
                changed[row, compartment] += 1
                with self.subTest(row=row, compartment=compartment), self.assertRaises(ValueError):
                    data.validate_direct(replace(reference, native=changed), self.canonical)
        near = reference.native + PLAN.reconstruction_tolerance / 2
        coverage = data.validate_direct(replace(reference, native=near), self.canonical)
        self.assertLessEqual(coverage.max_native_absolute_error, PLAN.reconstruction_tolerance)
        swapped = reference.native[::-1]
        with self.assertRaises(ValueError):
            data.validate_direct(replace(reference, native=swapped), self.canonical)

    def test_full_599_decode_coverage_is_independent_of_direct_reference(self):
        data.validate_direct(self.reference((0,)), self.canonical)
        self.canonical.require_prepared(self.ids[::-1])
        for ids in (self.ids[:-1], self.ids[:-1] + (self.ids[0],),
                    self.ids[:-1] + ('unknown',), self.ids + ('extra',)):
            with self.subTest(size=len(ids)), self.assertRaises(ValueError):
                self.canonical.require_prepared(ids)
        for ids, native in ((self.ids[:-1], self.values[:-1]),
                            (self.ids[:-1] + (self.ids[0],), self.values),
                            (self.ids, np.zeros((599, 2))),
                            (self.ids, np.full((599, 3), np.nan))):
            with self.subTest(size=len(ids)), self.assertRaises(ValueError):
                data.CanonicalDecode(ids, native)

    def test_reference_artifact_and_constructor_copies_are_bound(self):
        native = self.values.copy()
        canonical = data.CanonicalDecode(self.ids, native)
        native[:] = 100
        np.testing.assert_array_equal(canonical.native, self.values)
        with self.assertRaises(ValueError):
            canonical.native[0, 0] = 1
        reference = self.reference((0,))
        self.artifact.path.write_text('changed artifact\n')
        with self.assertRaises(ValueError):
            data.validate_direct(reference, canonical)


class MaskTransferBindingTests(unittest.TestCase):
    def test_missing_wrong_and_missing_path_pins_reject_before_loading(self):
        for known, inputs in (({}, {}), ({'mask_transfer': '0' * 64}, {}),
                              ({'mask_transfer': data.MASK_TRANSFER_SHA256}, {})):
            with self.subTest(known=known), patch.object(data, 'digest', side_effect=AssertionError('no loading')), \
                 self.assertRaises(ValueError):
                data.load_metadata({'known_sha256': known, 'inputs': inputs})

    def test_required_pin_is_the_original_manifest_and_checks_actual_bytes(self):
        self.assertEqual(data.MASK_TRANSFER_SHA256,
                         'd42259bbb47f6a9926892d1203dce3300cdee098fee8bbf1109da70482274666')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mask-transfer.json'
            path.write_text('{}\n')
            bindings = {'known_sha256': {'mask_transfer': data.MASK_TRANSFER_SHA256},
                        'inputs': {'mask_transfer': str(path)}}
            with self.assertRaisesRegex(ValueError, 'changed frozen input: mask_transfer'):
                data.load_metadata(bindings)
            # Only manifest identity is under test; no study metadata or mask is loaded.
            with patch.object(data, 'digest', return_value=data.MASK_TRANSFER_SHA256), \
                 patch.object(data, 'read_json', side_effect=RuntimeError('past pin check')):
                bindings['inputs']['inventory'] = str(path)
                with self.assertRaisesRegex(RuntimeError, 'past pin check'):
                    data.load_metadata(bindings)


if __name__ == '__main__':
    unittest.main()
