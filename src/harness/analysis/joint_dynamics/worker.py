from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import signal
import socket

from .execution import ArtifactManifest, StageWork, _Channel, _phases
from .protocol import StageRequest, digest, read_json


def execute_work(request: StageRequest, work: StageWork) -> ArtifactManifest:
    from scripts.run_joint_dynamics import perform_operation
    return perform_operation(request, work)


def _parent_death() -> None:
    expected = int(os.environ.pop('R38_PARENT_PID'))
    def orphaned(number, frame):
        # The fresh worker session is the only group this process may terminate.
        os.killpg(os.getpgrp(), signal.SIGKILL)
    signal.signal(signal.SIGTERM, orphaned)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'PR_SET_PDEATHSIG')
    if os.getppid() != expected:
        orphaned(signal.SIGTERM, None)


def _validate_output(request: StageRequest, manifest: ArtifactManifest) -> tuple[dict, dict]:
    if not isinstance(manifest, dict) or not manifest or 'operation_result.json' not in manifest:
        raise ValueError('operation must return complete relative artifact hashes')
    actual = {}
    for path in sorted(request.output.rglob('*')):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError('child output cannot contain links or special files')
        if path.is_file():
            name = path.relative_to(request.output).as_posix()
            if name == 'complete.json':
                raise ValueError('child cannot promote a complete artifact')
            actual[name] = digest(path)
    if actual != manifest:
        raise ValueError('operation manifest omits or misidentifies produced files')
    summary_path = request.output / 'operation_result.json'
    if summary_path.stat().st_size > 65500:
        raise ValueError('operation result is not a small CLI summary')
    summary = read_json(summary_path)
    if not isinstance(summary, dict):
        raise ValueError('operation result must be an object')
    request.source.require_current(Path.cwd())
    for ref in request.inputs:
        ref.verify()
    return actual, summary


def main() -> int:
    _parent_death()
    descriptor = int(os.environ.pop('R38_CONTROL_FD'))
    with socket.socket(fileno=descriptor) as connection:
        channel = _Channel(connection)
        incoming = channel.receive()
        if set(incoming) != {'request'}:
            raise ValueError('worker requires one stage request')
        request = StageRequest.from_dict(incoming['request'], repo=Path.cwd())
        _phases(request)
        for ref in request.inputs:
            ref.verify()
            if ref.path.resolve().is_relative_to(request.output.resolve()):
                raise ValueError('input/output overlap')
        if request.output.is_symlink() or not request.output.is_dir() or any(request.output.iterdir()):
            raise ValueError('worker requires a fresh empty output tree')
        work = StageWork(request, channel)
        manifest = execute_work(request, work)
        if work._sample is not None or work._inside or work._active != 'teardown':
            raise ValueError('operation returned inside unfinished phase/sample')
        actual, summary = _validate_output(request, manifest)
        work._event('result', manifest=actual, operation_result=summary)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
