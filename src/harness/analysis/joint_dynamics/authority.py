from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import resource
import tempfile
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from .protocol import ArtifactRef, Cost, SourceIdentity, StageCost, read_json

START = "2026-10-26T14:14:16+05:30"
DEADLINE = "2026-10-28T14:14:16+05:30"
GPU_LIMIT = 14400.0
CPU_LIMIT = 28800.0
SETUP_SECONDS = 900.0
LEDGER = Path("/home/aditya/experiments/world-model-r38-20261026/receipts/budget.json")
WORKSPACE = LEDGER.parent.parent
IDENTITY = {"start": START, "deadline": DEADLINE, "gpu_limit": GPU_LIMIT,
            "cpu_limit": CPU_LIMIT, "setup_seconds": SETUP_SECONDS,
            "ledger": str(LEDGER), "workspace": str(WORKSPACE)}
LEGACY_SHA256 = "6b1a250adbe5ff38ecf209d213ee1891863b8d76fa4f7d909cbd32f00df9aa36"
_ZERO = "0" * 64
_TEST_TOKEN = object()
AMENDMENT_ID = 'r38-budget-amendment-01'
AMENDMENT_CPU = 1600.0
AMENDMENT_GPU = 5350.0
BUDGET_ONLY_FILES = frozenset((
    'src/harness/analysis/joint_dynamics/authority.py',
    'src/harness/analysis/joint_dynamics/execution.py',
    'scripts/run_joint_dynamics.py', 'tests/test_joint_budget_amendment.py'))
Record = dict[str, Any]


def _bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _digest(value: Any) -> str:
    # Preserve the original two stage hashes, which did not include a newline.
    return hashlib.sha256(_bytes(value).rstrip(b"\n")).hexdigest()


def _raw_hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _time() -> float:
    return time.time()


def _deadline_epoch() -> float:
    return datetime.fromisoformat(DEADLINE).timestamp()


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_new(path: Path, raw: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)


def _replace(path: Path, raw: bytes) -> None:
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _file_id(path: Path) -> list[int]:
    stat = path.stat(follow_symlinks=False)
    if path.is_symlink():
        raise RuntimeError("authority files cannot be symlinks")
    return [stat.st_dev, stat.st_ino]


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise RuntimeError("invalid resource seconds")
    return float(value)


def _live_child_cpu() -> tuple[float, bool]:
    """Linux live descendants complement getrusage's already reaped children."""
    pending, seen, ordered, total = [os.getpid()], set(), [], 0.0
    ticks = os.sysconf("SC_CLK_TCK")
    while pending:
        pid = pending.pop()
        try:
            tasks = list(Path(f"/proc/{pid}/task").iterdir())
        except FileNotFoundError:
            continue
        for task in tasks:
            try:
                children = (task / "children").read_text().split()
            except FileNotFoundError:
                continue
            for child in map(int, children):
                if child in seen:
                    continue
                seen.add(child)
                ordered.append(child)
                pending.append(child)
    # Read descendants before ancestors, so a wait during observation can only
    # duplicate a transition, not drop it. Final joined rusage is authoritative.
    for child in reversed(ordered):
        try:
            fields = Path(f"/proc/{child}/stat").read_text().rsplit(")", 1)[1].split()
            total += sum(int(fields[i]) for i in (11, 12, 13, 14)) / ticks
        except FileNotFoundError:
            pass
    return total, bool(seen)


def _cpu() -> tuple[float, bool]:
    # Bracket the walk: a concurrent wait can move CPU from /proc into rusage.
    # Taking the later rusage is conservative at that transition, never an omission.
    live, has_children = _live_child_cpu()
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return time.process_time() + children.ru_utime + children.ru_stime + live, has_children


def _amendment(ref: ArtifactRef) -> Record:
    ref.verify()
    value = read_json(ref.path)
    fields = {'schema', 'id', 'authority_id', 'authorization_sha256', 'deadline',
              'additional_cpu_seconds', 'additional_gpu_seconds', 'before',
              'measured_source', 'implementation_source', 'operator_message', 'approved_request'}
    if (not isinstance(value, dict) or set(value) != fields
            or value['schema'] != 'r38-additive-authorization-v1' or value['id'] != AMENDMENT_ID
            or value['additional_cpu_seconds'] != AMENDMENT_CPU
            or value['additional_gpu_seconds'] != AMENDMENT_GPU
            or value['deadline'] != DEADLINE or value['operator_message'] != 'Go ahead'
            or not isinstance(value['approved_request'], str) or not value['approved_request']):
        raise RuntimeError('not the exact operator-approved additive R38 amendment')
    identities = []
    for key in ('measured_source', 'implementation_source'):
        source = value[key]
        if (not isinstance(source, dict) or set(source) != {'schema', 'files', 'digest'}
                or source['schema'] != 'r38-source-identity-v4'):
            raise RuntimeError('amendment requires complete distinct source identities')
        identities.append(SourceIdentity(source['files'], source['digest']))
    old, new = (identity.files for identity in identities)
    changed = {name for name in old.keys() | new.keys() if old.get(name) != new.get(name)}
    if not changed <= BUDGET_ONLY_FILES:
        raise RuntimeError('budget amendment cannot change numerical, model, data or scoring source')
    return value


def effective_limits(payload: Record) -> Record:
    amendments = payload.get('budget_amendments', [])
    if not isinstance(amendments, list) or len(amendments) > 1:
        raise RuntimeError('only one exact additive authorization is permitted')
    cpu, gpu = CPU_LIMIT, GPU_LIMIT
    for record in amendments:
        if not isinstance(record, dict) or set(record) != {'authorization'}:
            raise RuntimeError('malformed additive authorization binding')
        value = _amendment(ArtifactRef.from_dict(record['authorization']))
        cpu += value['additional_cpu_seconds']
        gpu += value['additional_gpu_seconds']
    return {'cpu_limit': cpu, 'gpu_limit': gpu, 'deadline': DEADLINE}


def verify_budget_only_sources(snapshot: Record, measured: dict, current: dict) -> str:
    amendments = snapshot.get('budget_amendments', [])
    if len(amendments) != 1:
        raise RuntimeError('changed control source requires its exact operator-approved amendment')
    ref = ArtifactRef.from_dict(amendments[0]['authorization'])
    value = _amendment(ref)
    if value['measured_source'] != measured or value['implementation_source'] != current:
        raise RuntimeError('source differs from the exact budget-only change bound in the amendment')
    return ref.sha256


class BudgetGuard:
    def __init__(self, authority: BudgetAuthority, name: str, gpu: bool,
                 started: float, cpu_start: float):
        self.authority = authority
        self.name, self.gpu = name, gpu
        self.started, self.cpu_start = started, cpu_start
        self.active = True
        self._observed_cpu = 0.0
        self._unresolved: str | None = None

    def _require_active(self) -> None:
        if not self.active or self.authority._active is not self:
            raise RuntimeError("reservation is no longer held")

    def _cost(self) -> tuple[float, float, bool]:
        cpu, live = _cpu()
        measured = max(0.0, cpu - self.cpu_start)
        self._observed_cpu = max(self._observed_cpu, measured)
        return (self._observed_cpu if live else measured), max(0.0, time.monotonic() - self.started), live

    def cost(self) -> StageCost:
        """Read the held reservation without acquiring its non-reentrant lock."""
        self._require_active()
        cpu, wall, live = self._cost()
        return StageCost(Cost(cpu, wall, wall if self.gpu else 0.0), live)

    def unresolved(self, reason: str) -> None:
        """Keep durable lower-bound charges when owned work cannot be joined."""
        self._require_active()
        if not isinstance(reason, str) or not reason:
            raise ValueError('unresolved exposure requires a reason')
        self._unresolved = reason

    def check(self, projected_gpu_seconds: float = 0.0, projected_cpu_seconds: float = 0.0) -> None:
        self._require_active()
        projected_gpu_seconds = _number(projected_gpu_seconds)
        projected_cpu_seconds = _number(projected_cpu_seconds)
        p = self.authority._snapshot_locked()
        cpu, wall, _ = self._cost()
        self.authority._check_limits(p, cpu + projected_cpu_seconds,
                                     (wall if self.gpu else 0.0) + projected_gpu_seconds,
                                     max(projected_gpu_seconds, projected_cpu_seconds))

    def snapshot(self) -> Record:
        self._require_active()
        return self.authority._snapshot_locked()

    def apply_budget_amendment(self, ref: ArtifactRef) -> None:
        self._require_active()
        if self.name != AMENDMENT_ID or self.gpu:
            raise RuntimeError('additive authorization has one dedicated CPU control stage')
        value = _amendment(ref)
        SourceIdentity.from_dict(value['implementation_source'])
        authority = self.authority
        p, events, anchor = authority._read_verified()
        authority._verify_extension_locked(value['before'])
        if (p.get('budget_amendments') or any(s['scientific'] for s in p['stages'])
                or p['stages'][:-1] != value['before']['stages']
                or value['authority_id'] != anchor['authority_id']
                or value['authorization_sha256'] != _digest(authority._identity)):
            raise RuntimeError('amendment is repeated, stale or belongs to another authority')
        p['budget_amendments'] = [{'authorization': ref.to_dict()}]
        authority._persist(p, 'amendment', events, anchor)
        ref.verify()


class BudgetAuthority:
    def __init__(self, ledger: Path = LEDGER, *, _test_token: object = None):
        ledger = Path(ledger).resolve()
        self._private = _test_token is _TEST_TOKEN
        if self._private:
            if not ledger.is_relative_to(Path(tempfile.gettempdir()).resolve()) or ledger == LEDGER:
                raise ValueError("private test authority must be in a temporary directory")
        elif ledger != LEDGER:
            raise ValueError("production ledger path is fixed")
        self.ledger = ledger
        self.lock_path = ledger.with_name(ledger.name + ".lock")
        self.anchor_path = ledger.with_name("r38-authority-anchor.json")
        self.journal_path = ledger.with_name("r38-authority-journal.jsonl")
        self.legacy_path = ledger.with_name("r38-adopted-legacy-budget.raw.json")
        self._identity = {**IDENTITY, "ledger": str(ledger), "workspace": str(ledger.parent.parent)}
        self._active: BudgetGuard | None = None
        self._held = False

    @contextmanager
    def _lock(self) -> Iterator[None]:
        if self._held:
            raise RuntimeError("concurrent or nested authority reservation")
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("concurrent authority reservation") from exc
            self._held = True
            yield
        finally:
            self._held = False
            os.close(fd)

    def _check_window(self) -> None:
        if not datetime.fromisoformat(START).timestamp() <= _time() < _deadline_epoch():
            raise RuntimeError("outside R38 authorization window")

    def _check_limits(self, p: Record, cpu: float = 0.0, gpu: float = 0.0,
                      projected_wall: float = 0.0) -> None:
        self._check_window()
        if _time() + projected_wall > _deadline_epoch():
            raise RuntimeError("R38 deadline would be exceeded")
        limits = effective_limits(p)
        if p["charged_cpu_seconds"] + cpu > limits['cpu_limit']:
            raise RuntimeError("R38 CPU allowance exceeded")
        if p["charged_gpu_seconds"] + gpu > limits['gpu_limit']:
            raise RuntimeError("R38 GPU allowance exceeded")

    def _require_new(self) -> None:
        if any(p.exists() or p.is_symlink() for p in
               (self.ledger, self.anchor_path, self.journal_path, self.legacy_path)):
            raise RuntimeError("authority already exists; refusing reset")

    def initialize(self) -> Record:
        with self._lock():
            self._require_new()
            self._check_window()
            p = {"identity": self._identity, "charged_cpu_seconds": SETUP_SECONDS,
                 "charged_gpu_seconds": 0.0, "stages": [], "head": _ZERO}
            raw = _bytes(p)
            _write_new(self.ledger, raw)
            self._create_anchor(p, raw, adopted=False)
            return self._snapshot_locked()

    def adopt_initialized_authority(self) -> Record:
        """One-time exact-byte R38 recovery; no setup charge or legacy write."""
        with self._lock():
            if self._private:
                raise RuntimeError("legacy adoption is production-only")
            if any(p.exists() or p.is_symlink() for p in
                   (self.anchor_path, self.journal_path, self.legacy_path)):
                raise RuntimeError("authority already adopted; refusing reset")
            self._check_window()
            raw = self.ledger.read_bytes()
            if _raw_hash(raw) != LEGACY_SHA256:
                raise RuntimeError("legacy R38 raw hash mismatch")
            p = json.loads(raw)
            self._validate_payload(p)
            if (len(p["stages"]) != 2 or p["charged_cpu_seconds"] != 900.0457602360001
                    or p["charged_gpu_seconds"] != 0.0
                    or any(s["status"] != "succeeded" or s["scientific"] for s in p["stages"])):
                raise RuntimeError("legacy R38 expenditure mismatch")
            _write_new(self.legacy_path, raw)
            self._create_anchor(p, raw, adopted=True)
            return self._snapshot_locked()

    def _create_anchor(self, p: Record, raw: bytes, *, adopted: bool) -> None:
        _write_new(self.journal_path, b"")
        anchor = {"schema": "r38-authority-v2", "authority_id": str(uuid.uuid4()),
                  "identity": self._identity, "canonical_ledger": str(self.ledger),
                  "directory_id": _file_id(self.ledger.parent), "lock_id": _file_id(self.lock_path),
                  "journal_id": _file_id(self.journal_path), "initial_raw_sha256": _raw_hash(raw),
                  "initial_snapshot": p, "adopted": adopted, "sequence": -1, "tip": _ZERO}
        _write_new(self.anchor_path, _bytes(anchor))
        self._persist(p, "adopt" if adopted else "initialize", [], anchor, preserve_raw=raw)

    def _validate_payload(self, p: Record) -> None:
        if p.get("identity") != self._identity:
            raise RuntimeError("authority identity or allowance mismatch")
        effective_limits(p)
        cpu, gpu, head, names = SETUP_SECONDS, 0.0, _ZERO, set()
        for i, stage in enumerate(p["stages"]):
            core = {k: v for k, v in stage.items() if k != "hash"}
            if stage.get("parent") != head or stage.get("hash") != _digest(core):
                raise RuntimeError("stage ancestry/hash mismatch")
            if not isinstance(stage["name"], str) or not stage["name"] or stage["name"] in names:
                raise RuntimeError("duplicate or invalid stage name")
            names.add(stage["name"])
            if stage["status"] not in ("running", "succeeded", "failed"):
                raise RuntimeError("invalid stage status")
            if stage["status"] == "running" and i != len(p["stages"]) - 1:
                raise RuntimeError("unresolved running attempt")
            cpu += _number(stage["cpu_seconds"])
            gpu += _number(stage["gpu_seconds"])
            head = stage["hash"]
        if (p["head"] != head or p["charged_cpu_seconds"] != cpu
                or p["charged_gpu_seconds"] != gpu):
            raise RuntimeError("ledger totals/head mismatch")

    def _read_verified(self) -> tuple[Record, list[Record], Record]:
        try:
            _file_id(self.anchor_path)
            anchor = json.loads(self.anchor_path.read_bytes())
            if (anchor["schema"] != "r38-authority-v2" or anchor["identity"] != self._identity
                    or anchor["canonical_ledger"] != str(self.ledger)
                    or anchor["directory_id"] != _file_id(self.ledger.parent)
                    or anchor["lock_id"] != _file_id(self.lock_path)
                    or anchor["journal_id"] != _file_id(self.journal_path)):
                raise RuntimeError("authority anchor identity/path mismatch")
            self._validate_payload(anchor["initial_snapshot"])
            if anchor["adopted"]:
                if (anchor["initial_raw_sha256"] != LEGACY_SHA256
                        or _raw_hash(self.legacy_path.read_bytes()) != LEGACY_SHA256):
                    raise RuntimeError("adopted legacy bytes changed")
            raw = self.ledger.read_bytes()
            _file_id(self.ledger)
            p = json.loads(raw)
            self._validate_payload(p)
            events, previous = [], _ZERO
            for line in self.journal_path.read_bytes().splitlines():
                event = json.loads(line)
                core = {k: v for k, v in event.items() if k != "hash"}
                if (event["parent"] != previous or event["sequence"] != len(events)
                        or event["authority_id"] != anchor["authority_id"]
                        or event["hash"] != _digest(core)):
                    raise RuntimeError("journal ancestry/hash mismatch")
                self._validate_payload(event["payload"])
                self._validate_transition(events[-1]["payload"] if events else None, event, anchor)
                events.append(event)
                previous = event["hash"]
            if (not events or anchor["tip"] != previous or anchor["sequence"] != len(events) - 1
                    or events[-1]["payload"] != p or events[-1]["raw_sha256"] != _raw_hash(raw)):
                raise RuntimeError("ledger/journal rollback, reset or incomplete persistence")
            return p, events, anchor
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RuntimeError("missing or malformed authority; refusing recovery/reset") from exc

    def _validate_transition(self, prior: Record | None, event: Record, anchor: Record) -> None:
        p, reason = event["payload"], event["reason"]
        if prior is None:
            if p != anchor["initial_snapshot"] or event["raw_sha256"] != anchor["initial_raw_sha256"]:
                raise RuntimeError("initial authority history changed")
            return
        old, new = prior["stages"], p["stages"]
        if reason == 'amendment':
            before = prior.get('budget_amendments', [])
            added = p.get('budget_amendments', [])
            if (before or len(added) != 1 or old != new or not old
                    or old[-1]['name'] != AMENDMENT_ID or old[-1]['status'] != 'running'
                    or any(s['scientific'] for s in old)
                    or any(p[k] != prior[k] for k in ('identity', 'charged_cpu_seconds', 'charged_gpu_seconds', 'head'))):
                raise RuntimeError('invalid additive authorization transition')
            value = _amendment(ArtifactRef.from_dict(added[0]['authorization']))
            if (value['authority_id'] != anchor['authority_id']
                    or value['authorization_sha256'] != _digest(self._identity)
                    or value['before']['stages'] != old[:-1]
                    or value['before']['head'] != old[-1]['parent']
                    or any(value['before'][k] != prior[k] for k in ('charged_cpu_seconds', 'charged_gpu_seconds'))):
                raise RuntimeError('additive authorization does not preserve original history and expenditure')
            return
        if p.get('budget_amendments', []) != prior.get('budget_amendments', []):
            raise RuntimeError('additive authorization cannot be removed, reset or changed')
        if reason == "reserve":
            if (len(new) != len(old) + 1 or new[:-1] != old or new[-1]["status"] != "running"
                    or any(s["status"] == "running" or (s["scientific"] and s["status"] == "failed") for s in old)
                    or new[-1]["cpu_seconds"] != 0 or new[-1]["gpu_seconds"] != 0):
                raise RuntimeError("invalid reservation ancestry")
        elif reason in ("finalize", "abandon"):
            if (not old or len(new) != len(old) or old[:-1] != new[:-1]
                    or old[-1]["status"] != "running" or old[-1]["ended"] is not None
                    or new[-1]["ended"] is None
                    or (new[-1]["status"] == "running") != (reason == "abandon")
                    or any(new[-1][k] != old[-1][k] for k in
                           ("name", "scientific", "gpu", "started", "parent", "owner_pid"))):
                raise RuntimeError("invalid finalization ancestry")
        else:
            raise RuntimeError("unknown journal transition")

    def _persist(self, p: Record, reason: str, events: list[Record], anchor: Record,
                 *, preserve_raw: bytes | None = None) -> None:
        self._validate_payload(p)
        raw = preserve_raw if preserve_raw is not None else _bytes(p)
        core = {"sequence": len(events), "parent": events[-1]["hash"] if events else _ZERO,
                "authority_id": anchor["authority_id"], "reason": reason,
                "payload": p, "raw_sha256": _raw_hash(raw)}
        event = {**core, "hash": _digest(core)}
        self._validate_transition(events[-1]["payload"] if events else None, event, anchor)
        # Journal first: any interruption leaves a visible reservation or fails closed.
        fd = os.open(self.journal_path, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
        with os.fdopen(fd, "ab") as stream:
            stream.write(_bytes(event))
            stream.flush()
            os.fsync(stream.fileno())
        if preserve_raw is None:
            _replace(self.ledger, raw)
        anchor = {**anchor, "sequence": event["sequence"], "tip": event["hash"]}
        _replace(self.anchor_path, _bytes(anchor))

    def _snapshot_locked(self) -> Record:
        p, events, anchor = self._read_verified()
        limits = effective_limits(p)
        return {**p, "authority_id": anchor["authority_id"],
                "authorization_sha256": _digest(self._identity), 'effective_limits': limits,
                "journal_sequence": len(events) - 1, "journal_tip": events[-1]["hash"],
                "ledger_raw_sha256": events[-1]["raw_sha256"],
                "remaining_cpu_seconds": limits['cpu_limit'] - p["charged_cpu_seconds"],
                "remaining_gpu_seconds": limits['gpu_limit'] - p["charged_gpu_seconds"]}

    def snapshot(self) -> Record:
        with self._lock():
            return self._snapshot_locked()

    def verify_extension(self, snapshot: Record) -> None:
        with self._lock():
            self._verify_extension_locked(snapshot)

    def _verify_extension_locked(self, snapshot: Record) -> None:
        _, events, anchor = self._read_verified()
        index = snapshot.get('journal_sequence')
        if (snapshot.get('identity') != self._identity
                or snapshot.get('authority_id') != anchor['authority_id']
                or snapshot.get('authorization_sha256') != _digest(self._identity)
                or type(index) is not int or not 0 <= index < len(events)):
            raise RuntimeError('snapshot authority/allowance or ancestry mismatch')
        event = events[index]
        limits = effective_limits(event['payload'])
        if (snapshot.get('journal_tip') != event['hash']
                or snapshot.get('ledger_raw_sha256') != event['raw_sha256']
                or any(snapshot.get(k) != v for k, v in event['payload'].items())
                or snapshot.get('remaining_cpu_seconds') != limits['cpu_limit'] - event['payload']['charged_cpu_seconds']
                or snapshot.get('remaining_gpu_seconds') != limits['gpu_limit'] - event['payload']['charged_gpu_seconds']
                or ('effective_limits' in snapshot and snapshot['effective_limits'] != limits)
                or ('effective_limits' not in snapshot and event['payload'].get('budget_amendments'))):
            raise RuntimeError('snapshot is not an ancestor of current authority')

    @contextmanager
    def stage(self, name: str, gpu: bool = False, scientific: bool = False) -> Iterator[BudgetGuard]:
        if not isinstance(name, str) or not name:
            raise ValueError("stage name must be nonempty")
        if type(gpu) is not bool or type(scientific) is not bool:
            raise ValueError("stage flags must be boolean")
        with self._lock():
            cpu_start, children = _cpu()
            started = time.monotonic()
            if children:
                raise RuntimeError("stage must start without unjoined child processes")
            p, events, anchor = self._read_verified()
            if any(s["status"] == "running" for s in p["stages"]):
                raise RuntimeError("unresolved running attempt blocks continuation")
            if any(s["scientific"] and s["status"] == "failed" for s in p["stages"]):
                raise RuntimeError("failed scientific attempt blocks continuation")
            if any(s["name"] == name for s in p["stages"]):
                raise RuntimeError("stage name already used")
            self._check_limits(p)
            core = {"name": name, "scientific": scientific, "gpu": gpu, "status": "running",
                    "started": _time(), "owner_pid": os.getpid(), "ended": None,
                    "cpu_seconds": 0.0, "gpu_seconds": 0.0, "failure": None, "parent": p["head"]}
            p["stages"].append({**core, "hash": _digest(core)})
            p["head"] = p["stages"][-1]["hash"]
            self._persist(p, "reserve", events, anchor)
            guard = BudgetGuard(self, name, gpu, started, cpu_start)
            self._active = guard
            failure: BaseException | None = None
            try:
                guard.check()
                yield guard
            except BaseException as exc:
                failure = exc
                raise
            finally:
                try:
                    p, events, anchor = self._read_verified()
                    cpu, wall, children = guard._cost()
                    final_error: BaseException | None = None
                    try:
                        self._check_limits(p, cpu, wall if gpu else 0.0)
                        if children or guard._unresolved:
                            raise RuntimeError("stage ended with unjoined children or unresolved exposure; continuation blocked")
                    except RuntimeError as exc:
                        final_error = exc
                    core = {k: v for k, v in p["stages"][-1].items() if k != "hash"}
                    core.update(status="failed" if failure or final_error else "succeeded",
                                ended=_time(), cpu_seconds=cpu, gpu_seconds=wall if gpu else 0.0,
                                failure=repr(failure or final_error) if failure or final_error else None)
                    # Unjoined descendants can keep spending: leave a durable unresolved attempt.
                    if children or guard._unresolved:
                        core["status"] = "running"
                        core["failure"] = repr(guard._unresolved or failure or final_error)
                    p["stages"][-1] = {**core, "hash": _digest(core)}
                    p["head"] = p["stages"][-1]["hash"]
                    p["charged_cpu_seconds"] += cpu
                    p["charged_gpu_seconds"] += wall if gpu else 0.0
                    self._persist(p, "abandon" if children or guard._unresolved else "finalize", events, anchor)
                    if final_error is not None and failure is None:
                        raise final_error
                finally:
                    guard.active = False
                    self._active = None


def production_authority() -> BudgetAuthority:
    return BudgetAuthority()


def _private_test_authority(path: Path) -> BudgetAuthority:
    return BudgetAuthority(path, _test_token=_TEST_TOKEN)
