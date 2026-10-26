from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from src.harness.analysis.joint_dynamics.authority import _private_test_authority, LEDGER
from src.harness.analysis.joint_dynamics.execution import execute_stage
from src.harness.analysis.joint_dynamics.protocol import (
    ArtifactRef, Cost, SourceIdentity, StageLimits, StageRequest, digest, save_json)


class OwnedLimitTests(unittest.TestCase):
    def test_noncooperative_native_worker_ignores_term_is_killed_reaped_and_charged(self):
        # Only this limit test substitutes the worker: a real fresh C-sleep process,
        # not a model or scorer mock. The actual-data canary uses worker.py unchanged.
        fixture = '''
import ctypes, os, signal, socket
from pathlib import Path
from src.harness.analysis.joint_dynamics.execution import _Channel
signal.signal(signal.SIGTERM, signal.SIG_IGN)
with socket.socket(fileno=int(os.environ['R38_CONTROL_FD'])) as conn:
    channel = _Channel(conn)
    request = channel.receive()['request']
    Path(request['output'], 'native-entered').write_text(str(os.getpid()))
    channel.send({'sequence': 1, 'kind': 'enter', 'phase': 'native_work'})
    assert channel.receive() == {'ack': 1}
    ctypes.CDLL(None).sleep(30)
'''
        with tempfile.TemporaryDirectory(prefix='r38-owned-limit-') as temporary:
            root = Path(temporary)
            authority = _private_test_authority(root / 'receipts/budget.json')
            self.assertNotEqual(authority.ledger, LEDGER)
            authority.initialize()
            payload = root / 'payload.json'
            save_json(payload, {'purpose': 'noncooperative-native-limit-test-only'})
            cleanup_cpu = max(300., 3. * (os.cpu_count() or 1))
            limits = StageLimits(False, False,
                {'startup': Cost(1000., 20., 0.), 'native_work': Cost(1000., 4., 0.),
                 'teardown': Cost(1000., 20., 0.)},
                Cost(3600., 3600., 900.), Cost(cleanup_cpu, 8., 0.),
                datetime.fromisoformat(authority.snapshot()['identity']['deadline']))
            request = StageRequest('native-limit', 'verify', SourceIdentity.capture(),
                (ArtifactRef(payload, digest(payload)),), limits, root / 'out')
            actual_popen = subprocess.Popen
            children = []

            def launch(command, **kwargs):
                self.assertEqual(command[-1], 'src.harness.analysis.joint_dynamics.worker')
                process = actual_popen([sys.executable, '-B', '-c', fixture], **kwargs)
                children.append(process)
                return process

            started = time.monotonic()
            with patch('src.harness.analysis.joint_dynamics.execution.subprocess.Popen', side_effect=launch):
                with self.assertRaisesRegex(TimeoutError, 'phase stopping margin reached: native_work'):
                    execute_stage(authority, request)
            self.assertLess(time.monotonic() - started, 10.)
            self.assertTrue((root / 'out/native-entered').is_file())
            self.assertEqual(len(children), 1)
            self.assertEqual(children[0].returncode, -signal.SIGKILL)
            with self.assertRaises(ProcessLookupError):
                os.kill(children[0].pid, 0)
            receipt = authority.snapshot()
            self.assertEqual(receipt['stages'][-1]['status'], 'failed')
            self.assertGreater(receipt['stages'][-1]['cpu_seconds'], 0.)
            self.assertEqual(receipt['charged_cpu_seconds'], 900. + receipt['stages'][-1]['cpu_seconds'])
            self.assertFalse(any(s['status'] == 'running' for s in receipt['stages']))
            self.assertFalse((root / 'out.control/complete.json').exists())


if __name__ == '__main__':
    unittest.main()
