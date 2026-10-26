from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import ctypes
import json
import math
import os
from pathlib import Path
import re
import select
import signal
import socket
import struct
import subprocess
import sys
import time
from types import MappingProxyType
from typing import Callable, ContextManager, Iterator, Mapping, TYPE_CHECKING

from .protocol import Cost, StageRequest, canonical_hash, read_json, save_json

if TYPE_CHECKING:
    from .authority import BudgetAuthority, BudgetGuard

ArtifactManifest = Mapping[str, str]
_MAX_FRAME = 1024 * 1024
_POLL = .02
_TERM_GRACE = .2
_KILL_GRACE = 2.
_UNIT = re.compile(r'[A-Za-z][A-Za-z0-9_./-]{0,127}')
_FIT = ('updates', 'source_cohort_mask_inference', 'calibration_controls',
        'nuisance_inference', 'serialization')


def _frame(value: dict) -> bytes:
    raw = json.dumps(value, allow_nan=False, separators=(',', ':')).encode()
    if len(raw) > _MAX_FRAME:
        raise ValueError('control frame too large')
    return struct.pack('!I', len(raw)) + raw


def _decode(raw: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate control key')
            result[key] = value
        return result
    result = json.loads(raw, object_pairs_hook=pairs,
                        parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if not isinstance(result, dict):
        raise ValueError('control message must be an object')
    return result


class _Channel:
    """Blocking child endpoint. Parent uses the same bounded framing incrementally."""
    def __init__(self, connection: socket.socket):
        self.connection = connection

    def send(self, value: dict) -> None:
        self.connection.sendall(_frame(value))

    def receive(self) -> dict:
        def exact(size):
            raw = bytearray()
            while len(raw) < size:
                block = self.connection.recv(size - len(raw))
                if not block:
                    raise EOFError('supervisor channel closed')
                raw.extend(block)
            return bytes(raw)
        length = struct.unpack('!I', exact(4))[0]
        if not 0 < length <= _MAX_FRAME:
            raise ValueError('invalid control frame length')
        return _decode(exact(length))


def _phases(request: StageRequest) -> tuple[str, ...]:
    caps = request.limits.phase_caps
    body = tuple(name for name in caps if name not in ('startup', 'teardown'))
    if request.operation == 'fit' and body != _FIT:
        raise ValueError('fit phase sequence differs from fixed production route')
    if request.operation == 'assess' and body != ('assessment',):
        raise ValueError('assessment must remain CPU-only scoring')
    if request.operation in ('prepare', 'preflight', 'fit') and not request.limits.gpu:
        raise ValueError('CUDA stages charge their whole wall time')
    if request.operation == 'assess' and request.limits.gpu:
        raise ValueError('CPU assessment cannot be GPU-retagged')
    if not body:
        raise ValueError('stage requires an explicitly capped work phase')
    return ('startup', *body, 'teardown')


class StageWork:
    """Child-only operation instrumentation; never an authority capability."""
    def __init__(self, request: StageRequest, channel: _Channel):
        self.request, self._channel = request, channel
        self._sequence = 0
        self._active = 'startup'
        self._inside = False
        self._counts: dict[str, int] = {}
        self._sample = None
        self._sample_id = 0

    def _event(self, kind: str, **fields) -> None:
        self._sequence += 1
        self._channel.send({'sequence': self._sequence, 'kind': kind, **fields})
        if self._channel.receive() != {'ack': self._sequence}:
            raise RuntimeError('supervisor rejected event')

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if self._inside or self._sample is not None:
            raise ValueError('nested phase or sample')
        self._event('enter', phase=name)
        self._active = name
        self._inside = True
        try:
            yield
        except BaseException:
            raise
        else:
            self._event('exit', phase=name)
            phases = _phases(self.request)
            self._active = phases[phases.index(name) + 1]
            self._inside = False

    def completed(self, unit: str, count: int = 1) -> None:
        if not isinstance(unit, str) or _UNIT.fullmatch(unit) is None or type(count) is not int or count <= 0:
            raise ValueError('invalid completed operation counter')
        self._event('completed', phase=self._active, unit=unit, count=count)
        self._counts[unit] = self._counts.get(unit, 0) + count

    def check(self, *, remaining_updates: int | None = None) -> None:
        if remaining_updates is not None and (type(remaining_updates) is not int or remaining_updates < 0):
            raise ValueError('invalid remaining updates')
        self._event('check', phase=self._active, remaining_updates=remaining_updates)

    def sample(self, unit: str, *, synchronize: Callable[[], None]) -> ContextManager[None]:
        return self._sample_operation(unit, synchronize)

    @contextmanager
    def _sample_operation(self, unit: str, synchronize: Callable[[], None]) -> Iterator[None]:
        if (self._sample is not None or not isinstance(unit, str) or _UNIT.fullmatch(unit) is None
                or not self._inside):
            raise ValueError('sample requires one active work phase/unit')
        synchronize()
        self._sample_id += 1
        if self._sample_id > 100000:
            raise ValueError('sample count bound exceeded')
        identifier = self._sample_id
        self._sample = unit
        initial = self._counts.get(unit, 0)
        self._event('sample_begin', phase=self._active, unit=unit, sample_id=identifier)
        cpu, wall = time.process_time(), time.monotonic()
        try:
            yield
            synchronize()
            if self._counts.get(unit, 0) != initial + 1:
                raise ValueError('sample must complete exactly one operation')
            self._event('sample_end', phase=self._active, unit=unit, sample_id=identifier,
                        cpu_seconds=time.process_time() - cpu, wall_seconds=time.monotonic() - wall)
        finally:
            self._sample = None


@dataclass(frozen=True)
class FinalizedStageReceipt:
    request: StageRequest
    authority: dict
    output: Path
    manifest: ArtifactManifest
    operation_result: Mapping[str, object]
    observation: Path
    completion: Path


@dataclass(frozen=True)
class _Process:
    pid: int
    parent: int
    group: int
    session: int
    start: int
    state: str


def _process(pid: int) -> _Process | None:
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return _Process(pid, int(fields[1]), int(fields[2]), int(fields[3]), int(fields[19]), fields[0])
    except FileNotFoundError:
        return None


class _OwnedWorker:
    def __init__(self, process: subprocess.Popen):
        self.process = process
        leader = _process(process.pid)
        if leader is None or leader.group != process.pid or leader.session != process.pid:
            raise RuntimeError('fresh worker session identity unavailable')
        self.leader = leader
        self.known = {leader.pid: leader}

    def members(self) -> list[_Process]:
        table = {}
        for path in Path('/proc').iterdir():
            if path.name.isdigit():
                entry = _process(int(path.name))
                if entry is not None:
                    table[entry.pid] = entry
        changed = True
        while changed:
            changed = False
            for pid, entry in table.items():
                if pid not in self.known and (entry.session == self.leader.session or entry.parent in self.known):
                    self.known[pid] = entry
                    changed = True
        members = []
        for pid, identity in self.known.items():
            current = table.get(pid)
            if current is not None:
                if current.start != identity.start:
                    raise RuntimeError('owned process PID identity changed')
                members.append(current)
        # Unknown children are not a signaling capability. Session members include
        # orphans adopted between walks; detached unobserved work fails unresolved.
        if any(e.parent == os.getpid() and e.pid not in self.known for e in table.values()):
            raise RuntimeError('unidentified adopted child; unresolved process exposure')
        return members

    def signal(self, number: int) -> None:
        members = self.members()
        leader = _process(self.leader.pid)
        if leader is not None:
            if (leader.start, leader.group, leader.session) != (self.leader.start, self.leader.group, self.leader.session):
                raise RuntimeError('refusing signal to changed worker identity')
            os.killpg(leader.group, number)
        for entry in members:
            current = _process(entry.pid)
            if current is not None and current.start == entry.start:
                try:
                    os.kill(entry.pid, number)
                except ProcessLookupError:
                    pass

    def reap(self) -> bool:
        members = self.members()
        for entry in members:
            if entry.pid == self.leader.pid:
                continue
            if entry.parent == os.getpid() and entry.state == 'Z':
                try:
                    os.waitpid(entry.pid, os.WNOHANG)
                except ChildProcessError:
                    raise RuntimeError('owned adopted descendant reaped elsewhere')
        # Reap the leader only after discovering/adopting descendants.
        self.process.poll()
        return self.process.returncode is not None and not self.members()

    def stop(self) -> bool:
        for number, grace in ((signal.SIGTERM, _TERM_GRACE), (signal.SIGKILL, _KILL_GRACE)):
            self.signal(number)
            until = time.monotonic() + grace
            while time.monotonic() < until:
                if self.reap():
                    return True
                time.sleep(_POLL)
        return self.reap()


@contextmanager
def _subreaper() -> Iterator[None]:
    import threading
    if sys.platform != 'linux' or threading.active_count() != 1:
        raise RuntimeError('supervision requires a dedicated single-threaded Linux parent')
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(previous), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'PR_GET_CHILD_SUBREAPER')
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'PR_SET_CHILD_SUBREAPER')
    try:
        yield
    finally:
        # Do not strand unresolved adopted children by relinquishing ownership.
        children = Path(f'/proc/{os.getpid()}/task/{os.getpid()}/children').read_text().strip()
        if not children and libc.prctl(36, previous.value, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), 'restore child subreaper')


def _subtract(a: Cost, b: Cost) -> Cost:
    return Cost(*(max(0., getattr(a, k) - getattr(b, k))
                  for k in ('cpu_seconds', 'wall_seconds', 'gpu_seconds')))


def _sum(costs) -> Cost:
    costs = tuple(costs)
    return Cost(*(sum(getattr(c, k) for c in costs)
                  for k in ('cpu_seconds', 'wall_seconds', 'gpu_seconds')))


class _Observer:
    def __init__(self, guard: BudgetGuard, request: StageRequest):
        self.guard, self.request = guard, request
        self.phases = _phases(request)
        self.index = 0
        self.entered = False
        self.origin = Cost(0., 0., 0.)
        self.phase_costs = {}
        self.counts = {name: {} for name in self.phases}
        self.samples = []
        self.sampling = None
        self.sequence = 0
        self.result = None
        self.last = Cost(0., 0., 0.)
        # Every runnable thread can spend CPU during detection/termination.
        slots = os.cpu_count() or 1
        wall = _POLL + _TERM_GRACE + _KILL_GRACE
        self.headroom = Cost(slots * wall, wall, wall if request.limits.gpu else 0.)
        cleanup = request.limits.cleanup
        if any(getattr(cleanup, k) < getattr(self.headroom, k)
               for k in ('cpu_seconds', 'wall_seconds', 'gpu_seconds')):
            raise ValueError('cleanup does not cover polling, CPU concurrency and termination grace')
        snapshot = guard.snapshot()
        original = datetime.fromisoformat(snapshot['identity']['deadline'])
        if request.limits.deadline > original:
            raise ValueError('stage cannot extend the original deadline')
        self.entry = snapshot

    @property
    def phase(self):
        return self.phases[self.index]

    def observe(self) -> Cost:
        measured = self.guard.cost().cost
        self.last = Cost(*(max(getattr(self.last, k), getattr(measured, k))
                           for k in ('cpu_seconds', 'wall_seconds', 'gpu_seconds')))
        elapsed = _subtract(self.last, self.origin)
        limits = self.request.limits
        cap = limits.phase_caps[self.phase]
        if any(getattr(elapsed, k) + getattr(self.headroom, k) >= getattr(cap, k)
               for k in ('cpu_seconds', 'wall_seconds') + (('gpu_seconds',) if limits.gpu else ())):
            raise TimeoutError(f'phase stopping margin reached: {self.phase}')
        obligations = _sum((limits.future, limits.cleanup, _subtract(cap, elapsed),
                            *(limits.phase_caps[p] for p in self.phases[self.index + 1:])))
        if (self.entry['charged_cpu_seconds'] + self.last.cpu_seconds + obligations.cpu_seconds >=
                self.entry['identity']['cpu_limit'] or
                self.entry['charged_gpu_seconds'] + self.last.gpu_seconds + obligations.gpu_seconds >=
                self.entry['identity']['gpu_limit']):
            raise TimeoutError('unfinished work or protected reserve would be consumed')
        if time.time() + obligations.wall_seconds >= limits.deadline.timestamp():
            raise TimeoutError('deadline cannot cover unfinished work and cleanup')
        return self.last

    def _close_phase(self, cost: Cost) -> None:
        self.phase_costs[self.phase] = _subtract(cost, self.origin).to_dict()
        self.origin = cost
        self.index += 1
        self.entered = False

    def event(self, event: dict) -> dict:
        cost = self.observe()
        if type(event.get('sequence')) is not int or event['sequence'] != self.sequence + 1:
            raise ValueError('nonconsecutive event sequence')
        kind = event.get('kind')
        base = {'sequence', 'kind'}
        fields = {'enter': {'phase'}, 'exit': {'phase'},
                  'completed': {'phase', 'unit', 'count'},
                  'check': {'phase', 'remaining_updates'},
                  'sample_begin': {'phase', 'unit', 'sample_id'},
                  'sample_end': {'phase', 'unit', 'sample_id', 'cpu_seconds', 'wall_seconds'},
                  'result': {'manifest', 'operation_result'}}
        if kind not in fields or set(event) != base | fields[kind] or self.result is not None:
            raise ValueError('invalid event fields or event after result')
        if kind == 'enter':
            if self.phase == 'startup':
                self._close_phase(cost)
            if self.entered or self.phase == 'teardown' or event['phase'] != self.phase:
                raise ValueError('wrong phase entry sequence')
            self.entered = True
        elif kind == 'result':
            if self.phase != 'teardown' or self.sampling is not None:
                raise ValueError('result before complete phase sequence')
            manifest, summary = event['manifest'], event['operation_result']
            if not isinstance(manifest, dict) or 'operation_result.json' not in manifest:
                raise ValueError('worker omitted operation result artifact')
            for name, sha in manifest.items():
                if (not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts
                        or str(Path(name)) != name or name == 'complete.json'
                        or not isinstance(sha, str) or re.fullmatch('[0-9a-f]{64}', sha) is None):
                    raise ValueError('unsafe or invalid artifact manifest')
            if not isinstance(summary, dict) or len(_frame(summary)) > 65536:
                raise ValueError('operation result must be small JSON')
            self.result = {'manifest': manifest, 'operation_result': summary}
        else:
            if event['phase'] != self.phase:
                raise ValueError('event outside active phase')
            if kind == 'exit':
                if self.sampling is not None or self.phase in ('startup', 'teardown') or not self.entered:
                    raise ValueError('invalid phase exit')
                self._close_phase(cost)
            elif kind == 'completed':
                unit, count = event['unit'], event['count']
                if not isinstance(unit, str) or _UNIT.fullmatch(unit) is None or type(count) is not int or count <= 0:
                    raise ValueError('invalid operation count')
                counts = self.counts[self.phase]
                if len(counts) >= 4096 and unit not in counts:
                    raise ValueError('too many operation dimensions')
                counts[unit] = counts.get(unit, 0) + count
            elif kind == 'check':
                remaining = event['remaining_updates']
                if remaining is not None and (type(remaining) is not int or remaining < 0):
                    raise ValueError('invalid remaining updates')
            elif kind == 'sample_begin':
                if (self.sampling is not None or not self.entered or self.phase in ('startup', 'teardown')
                        or type(event['sample_id']) is not int or event['sample_id'] != len(self.samples) + 1
                        or len(self.samples) >= 100000 or not isinstance(event['unit'], str)
                        or _UNIT.fullmatch(event['unit']) is None):
                    raise ValueError('invalid sample begin')
                self.sampling = (dict(event), cost, self.counts[self.phase].get(event['unit'], 0))
            elif kind == 'sample_end':
                if self.sampling is None:
                    raise ValueError('sample end without begin')
                start, initial, count = self.sampling
                unit = event['unit']
                if (event['sample_id'] != start['sample_id'] or unit != start['unit']
                        or self.counts[self.phase].get(unit, 0) != count + 1):
                    raise ValueError('sample/counter association mismatch')
                timing = Cost(event['cpu_seconds'], event['wall_seconds'], 0.)
                observed = _subtract(cost, initial)
                if (timing.wall_seconds <= 0 or timing.wall_seconds > observed.wall_seconds + _POLL
                        or timing.cpu_seconds > observed.cpu_seconds + .05):
                    raise ValueError('sample timing exceeds observed interval')
                self.samples.append({'sample_id': event['sample_id'], 'phase': self.phase, 'unit': unit,
                    'begin_sequence': start['sequence'], 'end_sequence': event['sequence'],
                    'completed_before': count, 'completed_after': count + 1,
                    'child': timing.to_dict(), 'observed': observed.to_dict()})
                self.sampling = None
        self.sequence = event['sequence']
        return {'ack': self.sequence}

    def finish(self) -> dict:
        cost = self.observe()
        if self.result is None or self.phase != 'teardown' or self.sampling is not None:
            raise ValueError('exit zero without complete stage result')
        self.phase_costs['teardown'] = _subtract(cost, self.origin).to_dict()
        return {'schema': 'r38-stage-observation-v4', 'stage_name': self.request.name,
                'operation': self.request.operation, 'request': self.request.to_dict(),
                'phase_costs': self.phase_costs, 'counts': self.counts, 'samples': self.samples,
                'last_sequence': self.sequence, **self.result}


def _supervise(owned: _OwnedWorker, connection: socket.socket, observer: _Observer) -> dict:
    connection.setblocking(False)
    outgoing = bytearray(_frame({'request': observer.request.to_dict()}))
    incoming = bytearray()
    eof = False
    while True:
        observer.observe()
        if owned.reap():
            if owned.process.returncode != 0:
                raise RuntimeError(f'numerical worker exited {owned.process.returncode}')
            if incoming or outgoing:
                raise ValueError('worker exited with partial or unacknowledged control frame')
            if observer.result is not None:
                return observer.finish()
            raise EOFError('worker exited without a result')
        ready, writable, _ = select.select([] if eof else [connection], [connection] if outgoing else [], [], _POLL)
        if writable:
            sent = connection.send(outgoing)
            del outgoing[:sent]
        if ready:
            raw = connection.recv(65536)
            if not raw:
                eof = True
                if incoming or observer.result is None:
                    raise EOFError('partial frame or EOF before result')
            incoming.extend(raw)
        while len(incoming) >= 4:
            length = struct.unpack('!I', incoming[:4])[0]
            if not 0 < length <= _MAX_FRAME:
                raise ValueError('invalid worker control frame length')
            if len(incoming) < length + 4:
                break
            event = _decode(bytes(incoming[4:4 + length]))
            del incoming[:4 + length]
            outgoing.extend(_frame(observer.event(event)))


def execute_stage(authority: BudgetAuthority, request: StageRequest) -> FinalizedStageReceipt:
    """Reserve once, supervise one fresh executable, finalize before completion."""
    _phases(request)
    controls = request.output.with_name(request.output.name + '.control')
    if request.output.exists() or request.output.is_symlink() or controls.exists() or controls.is_symlink():
        raise FileExistsError('partial or repeated stage/output; no renamed retry')
    entry = authority.snapshot()
    authority.verify_extension(entry)
    controls.mkdir(parents=True, exist_ok=False)
    observation_path = controls / 'observation.json'
    completion_path = controls / 'complete.json'
    observed = None
    with authority.stage(request.name, gpu=request.limits.gpu, scientific=request.limits.scientific) as guard:
        current = guard.snapshot()
        stages = current['stages']
        if (any(current[k] != entry[k] for k in ('authority_id', 'authorization_sha256', 'identity'))
                or len(stages) != len(entry['stages']) + 1 or stages[:-1] != entry['stages']
                or stages[-1]['name'] != request.name or stages[-1]['status'] != 'running'
                or stages[-1]['scientific'] != request.limits.scientific
                or stages[-1]['gpu'] != request.limits.gpu):
            raise RuntimeError('reservation differs from exact entry identity/prefix')
        observer = _Observer(guard, request)
        owned = None
        process = None
        with _subreaper(), _socket_pair() as pair:
            parent, child = pair
            try:
                observer.observe()
                request.output.mkdir(parents=True, exist_ok=False)
                with (controls / 'stdout').open('xb') as stdout, (controls / 'stderr').open('xb') as stderr:
                    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
                                       R38_CONTROL_FD=str(child.fileno()), R38_PARENT_PID=str(os.getpid()))
                    process = subprocess.Popen([sys.executable, '-B', '-m',
                        'src.harness.analysis.joint_dynamics.worker'], cwd=Path.cwd(),
                        env=environment, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                        close_fds=True, pass_fds=(child.fileno(),), start_new_session=True)
                    owned = _OwnedWorker(process)
                    child.close()
                    observed = _supervise(owned, parent, observer)
                    save_json(observation_path, observed)
                    observer.observe()
            except BaseException as exc:
                if owned is not None:
                    try:
                        joined = owned.stop()
                    except BaseException as cleanup_error:
                        guard.unresolved(f'{cleanup_error!r}; original={exc!r}')
                    else:
                        if not joined:
                            guard.unresolved(f'owned worker not joined; original={exc!r}')
                elif process is not None and process.poll() is None:
                    guard.unresolved(f'worker startup identity unavailable; original={exc!r}')
                save_json(controls / 'failure.json', {'error': repr(exc),
                    'lower_bound': guard.cost().cost.to_dict(), 'stage_name': request.name})
                raise
    final = authority.snapshot()
    authority.verify_extension(current)
    authority.verify_extension(entry)
    match = [stage for stage in final['stages'] if stage['name'] == request.name]
    if len(match) != 1 or match[0]['status'] != 'succeeded':
        raise RuntimeError('stage was not successfully finalized')
    # This small parent record binds the child manifest, not a second hash pass.
    completion = {'schema': 'r38-finalized-stage-v4', 'stage_name': request.name,
                  'request': request.to_dict(), 'authority': final,
                  'observation_sha256': canonical_hash(observed), 'observation': str(observation_path),
                  'manifest': observed['manifest'], 'operation_result': observed['operation_result']}
    save_json(completion_path, completion)
    return FinalizedStageReceipt(request, final, request.output,
        MappingProxyType(observed['manifest']), MappingProxyType(observed['operation_result']),
        observation_path, completion_path)


@contextmanager
def _socket_pair():
    parent, child = socket.socketpair()
    try:
        yield parent, child
    finally:
        parent.close()
        child.close()
