from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
import tempfile
import unittest

from src.harness.analysis.joint_dynamics import authority as accounting
from src.harness.analysis.joint_dynamics.protocol import (
    ArtifactRef, Cost, SourceIdentity, StageLimits, StageRequest, canonical_hash, digest, save_json)


class AdditiveBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='r38-additive-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.authority = accounting._private_test_authority(self.root / 'receipts/budget.json')
        self.before = self.authority.initialize()
        self.current = SourceIdentity.capture().to_dict()
        files = dict(self.current['files'])
        files['src/harness/analysis/joint_dynamics/authority.py'] = '0' * 64
        self.measured = SourceIdentity(files, canonical_hash(files)).to_dict()

    def receipt(self, **changes):
        value = {'schema': 'r38-additive-authorization-v1', 'id': accounting.AMENDMENT_ID,
            'authority_id': self.before['authority_id'],
            'authorization_sha256': self.before['authorization_sha256'],
            'deadline': accounting.DEADLINE,
            'additional_cpu_seconds': 1600., 'additional_gpu_seconds': 5350.,
            'before': self.before, 'measured_source': self.measured,
            'implementation_source': self.current, 'operator_message': 'Go ahead',
            'approved_request': 'Add5350GPU/1600CPU; same science/deadline; preserve cumulative authority.'}
        value.update(changes)
        path = self.root / ('authorization-' + str(len(list(self.root.glob('authorization-*')))) + '.json')
        save_json(path, value)
        return ArtifactRef(path, digest(path))

    def apply(self, ref):
        with self.authority.stage(accounting.AMENDMENT_ID, gpu=False, scientific=False) as guard:
            guard.apply_budget_amendment(ref)

    def test_append_preserves_identity_costs_history_and_old_snapshots(self):
        raw = self.authority.ledger.read_bytes()
        journal = self.authority.journal_path.read_bytes()
        ref = self.receipt()
        self.apply(ref)
        final = self.authority.snapshot()
        self.assertEqual(final['identity'], self.before['identity'])
        self.assertEqual(final['authority_id'], self.before['authority_id'])
        self.assertEqual(final['authorization_sha256'], self.before['authorization_sha256'])
        self.assertEqual(final['effective_limits'],
            {'cpu_limit': 30400., 'gpu_limit': 19750., 'deadline': accounting.DEADLINE})
        self.assertEqual(final['charged_cpu_seconds'], 900. + final['stages'][-1]['cpu_seconds'])
        self.assertEqual(final['charged_gpu_seconds'], 0.)
        self.assertTrue(self.authority.journal_path.read_bytes().startswith(journal))
        self.assertNotEqual(self.authority.ledger.read_bytes(), raw)
        self.authority.verify_extension(self.before)
        legacy = copy.deepcopy(self.before)
        legacy.pop('effective_limits')
        self.authority.verify_extension(legacy)
        self.authority.verify_extension(final)
        missing = copy.deepcopy(final)
        missing.pop('effective_limits')
        with self.assertRaises(RuntimeError):
            self.authority.verify_extension(missing)
        self.assertEqual(accounting.verify_budget_only_sources(final, self.measured, self.current), ref.sha256)
        with self.assertRaises(RuntimeError):
            accounting.verify_budget_only_sources(final, self.current, self.measured)
        with self.assertRaises(RuntimeError):
            self.apply(ref)
        self.assertEqual(len(self.authority.snapshot()['budget_amendments']), 1)

    def test_effective_limits_are_enforced_without_replacing_original_limits(self):
        ref = self.receipt()
        self.apply(ref)
        self.assertEqual(accounting.CPU_LIMIT, 28800.)
        self.assertEqual(accounting.GPU_LIMIT, 14400.)
        with self.authority.stage('above-original-within-addition') as guard:
            guard.check(projected_gpu_seconds=19000.)
            from src.harness.analysis.joint_dynamics.execution import _Observer
            limits = StageLimits(False, False,
                {'startup': Cost(1000., 20., 0.), 'work': Cost(1000., 20., 0.),
                 'teardown': Cost(1000., 20., 0.)},
                Cost(3600., 19000., 19000.), Cost(300., 8., 0.),
                datetime.fromisoformat(accounting.DEADLINE))
            request = StageRequest('above-original-within-addition', 'verify', SourceIdentity.capture(),
                                   (ref,), limits, self.root / 'unused-output')
            observer = _Observer(guard, request)
            self.assertEqual(observer.entry_limits['gpu_limit'], 19750.)
            observer.observe()
        with self.assertRaisesRegex(RuntimeError, 'GPU allowance exceeded'):
            with self.authority.stage('above-effective-limit') as guard:
                guard.check(projected_gpu_seconds=19751.)
        self.assertEqual(self.authority.snapshot()['stages'][-1]['status'], 'failed')

    def test_rejects_unapproved_amount_deadline_and_changed_numerical_source(self):
        for changes in ({'additional_gpu_seconds': 5351.},
                        {'additional_cpu_seconds': 1601.},
                        {'deadline': '2026-10-29T14:14:16+05:30'}):
            with self.subTest(changes=changes):
                with self.assertRaises(RuntimeError):
                    accounting._amendment(self.receipt(**changes))
        files = dict(self.measured['files'])
        files['src/harness/analysis/joint_dynamics/model.py'] = '1' * 64
        changed = SourceIdentity(files, canonical_hash(files)).to_dict()
        with self.assertRaisesRegex(RuntimeError, 'cannot change numerical'):
            accounting._amendment(self.receipt(measured_source=changed))

    def test_grant_file_tampering_and_foreign_authority_are_rejected(self):
        ref = self.receipt(authority_id='another-authority')
        with self.assertRaises(RuntimeError):
            self.apply(ref)
        self.assertNotIn('budget_amendments', self.authority.snapshot())
        ref = self.receipt()
        ref.path.write_text('{}\n')
        with self.assertRaises(ValueError):
            accounting._amendment(ref)


if __name__ == '__main__':
    unittest.main()
