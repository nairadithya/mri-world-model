"""Authoritative, append-verified and exclusively reserved R37 continuation budget."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import time
import uuid

from .spatial_dynamics_data import canonical_hash, save_json

AUTHORIZED_LEDGER = Path("/home/aditya/experiments/world-model-r37-20261026/receipts/continuation_budget.json")
AUTHORIZATION = {
    "started_at": "2026-10-26T10:00:04+05:30",
    "deadline_unix": datetime.fromisoformat("2026-10-28T10:00:04+05:30").timestamp(),
    "gpu_seconds_limit": 28800,
    "cpu_seconds_limit": 57600,
}
# Exact pre-migration snapshot checked by the independent first implementation review.
LEGACY_LEDGER_SHA256 = "cb717e6205a4af7e912e18854f19892c0a3f9169206cb48a22163aed3386c695"


def process_cpu():
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return time.process_time() + children.ru_utime + children.ru_stime


class BudgetAuthority:
    """Reusable verifier; production CLI always uses the fixed authority below.

    Tests create a separate authority in a remote temporary directory. Neither
    the production CLI nor its contract accepts an alternate authority root.
    """
    def __init__(self, ledger_path, authorization, *, initial_sha256=None):
        self.path = Path(ledger_path).resolve()
        self.authorization = dict(authorization)
        self.initial_sha256 = initial_sha256
        self.anchor_path = self.path.with_name("continuation_authority.json")
        self.journal_path = self.path.with_name("continuation_journal.jsonl")
        self.lock_path = self.path.with_name("continuation_budget.lock")
        self.held = False

    def require_path(self, path):
        if Path(path).resolve() != self.path:
            raise ValueError("not the authoritative continuation ledger")

    @contextmanager
    def lock(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        previous_held = self.held
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("another continuation stage holds the exclusive budget reservation") from exc
            self.held = True
            yield
        finally:
            self.held = previous_held
            os.close(fd)

    def validate_limits(self, ledger):
        if any(ledger.get(key) != value for key, value in self.authorization.items()):
            raise ValueError("authorization limits or deadline changed")
        if ledger.get("setup_cpu_charge_seconds") != 900.:
            raise ValueError("original conservative setup charge was changed")
        for stage in ledger["stages"]:
            if stage["status"] not in {"running", "failed", "completed"}:
                raise ValueError("invalid stage history")
            for key in ("cpu_seconds", "gpu_seconds"):
                if key in stage and (not math.isfinite(stage[key]) or stage[key] < 0):
                    raise ValueError("invalid consumed resource history")

    def initialize(self):
        """One-time migration, retaining the entire reviewed inventory/setup state."""
        with self.lock():
            if self.anchor_path.exists():
                return self.read_verified()[0]
            raw = self.path.read_bytes()
            if self.initial_sha256 and hashlib.sha256(raw).hexdigest() != self.initial_sha256:
                raise ValueError("pre-migration ledger differs from the reviewed expenditure snapshot")
            ledger = json.loads(raw)
            self.validate_limits(ledger)
            if any(stage["status"] != "completed" for stage in ledger["stages"]):
                raise ValueError("cannot initialize over a failed/running stage")
            if self.initial_sha256 and [s["stage"] for s in ledger["stages"]] != ["inventory"]:
                raise ValueError("reviewed inventory expenditure was not retained")
            anchor = {"schema": "r37-budget-authority-v1", "identity": str(uuid.uuid4()),
                      "canonical_ledger": str(self.path), "authorization": self.authorization,
                      "initial_ledger_sha256": hashlib.sha256(raw).hexdigest(),
                      "initial_snapshot": ledger}
            fd = os.open(self.anchor_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as stream:
                json.dump(anchor, stream, sort_keys=True, indent=2, allow_nan=False)
                stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
            ledger["authority_sha256"] = canonical_hash(anchor)
            self.persist(ledger, "initialize")
            return ledger

    def events(self):
        result, previous = [], "0" * 64
        if not self.journal_path.is_file():
            return result
        for line in self.journal_path.read_text().splitlines():
            event = json.loads(line)
            saved = event.pop("event_sha256")
            if event["sequence"] != len(result) or event["previous"] != previous or canonical_hash(event) != saved:
                raise ValueError("invalid continuation journal chain")
            if canonical_hash(event["snapshot"]) != event["ledger_sha256"]:
                raise ValueError("invalid journal spending snapshot")
            event["event_sha256"] = saved
            result.append(event)
            previous = saved
        return result

    def read_verified(self):
        anchor = json.loads(self.anchor_path.read_text())
        if anchor.get("canonical_ledger") != str(self.path) or anchor.get("authorization") != self.authorization:
            raise ValueError("wrong continuation authority")
        ledger = json.loads(self.path.read_text())
        self.validate_limits(ledger)
        if ledger.get("authority_sha256") != canonical_hash(anchor):
            raise ValueError("ledger authority identity changed")
        events = self.events()
        if not events or events[-1]["ledger_sha256"] != canonical_hash(ledger):
            raise ValueError("ledger history was reset or changed outside its journal")
        if events[0]["snapshot"]["stages"] != anchor["initial_snapshot"]["stages"]:
            raise ValueError("original expenditure history was not retained")
        return ledger, events

    def persist(self, ledger, reason):
        if not self.held:
            raise RuntimeError("budget persistence requires an exclusive reservation")
        previous = self.events()
        event = {"sequence": len(previous), "previous": previous[-1]["event_sha256"] if previous else "0"*64,
                 "reason": reason, "ledger_sha256": canonical_hash(ledger), "snapshot": ledger}
        event["event_sha256"] = canonical_hash(event)
        save_json(self.path, ledger)
        fd = os.open(self.journal_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(fd, "a") as stream:
            stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
            stream.flush(); os.fsync(stream.fileno())

    def snapshot(self, path):
        self.require_path(path)
        ledger, events = self.read_verified()
        return {"ledger_path": str(self.path), "authority_sha256": ledger["authority_sha256"],
                "locked_journal_tip": events[-1]["event_sha256"]}

    def verify_extension(self, path, locked):
        self.require_path(path)
        ledger, events = self.read_verified()
        if locked["ledger_path"] != str(self.path) or locked["authority_sha256"] != ledger["authority_sha256"]:
            raise ValueError("run uses a different budget authority")
        if locked["locked_journal_tip"] not in {e["event_sha256"] for e in events}:
            raise ValueError("budget history does not extend the locked expenditure state")
        return ledger


def production_authority():
    return BudgetAuthority(AUTHORIZED_LEDGER, AUTHORIZATION, initial_sha256=LEGACY_LEDGER_SHA256)


def initialize_authority():
    return production_authority().initialize()


@contextmanager
def budget_stage(ledger_path, stage, *, gpu=False, preflight=False, authority=None):
    """Reserve the one authority for the whole process; never retry failed fitting."""
    authority = production_authority() if authority is None else authority
    authority.require_path(ledger_path)
    with authority.lock():
        ledger, _ = authority.read_verified()
        stages = ledger["stages"]
        if any(s["stage"] == stage for s in stages):
            raise FileExistsError("stage already started; automatic retries are forbidden")
        if any(s["status"] != "completed" for s in stages):
            raise RuntimeError("a prior continuation stage is incomplete or failed")
        spent_cpu = sum(s.get("cpu_seconds", 0.) for s in stages) + ledger["setup_cpu_charge_seconds"]
        spent_gpu = sum(s.get("gpu_seconds", 0.) for s in stages)
        remaining_cpu = ledger["cpu_seconds_limit"] - spent_cpu
        remaining_wall = ledger["deadline_unix"] - time.time()
        remaining_gpu = ledger["gpu_seconds_limit"] - spent_gpu
        if preflight:
            remaining_cpu = min(remaining_cpu, 4*3600 - sum(s.get("cpu_seconds", 0.) for s in stages if s.get("preflight")))
        limit_wall = min(remaining_wall, remaining_gpu) if gpu else remaining_wall
        if min(remaining_cpu, limit_wall) <= 0:
            raise TimeoutError("continuation resource allowance exhausted")
        started = time.monotonic()
        entry = {"stage": stage, "status": "running", "gpu": gpu, "preflight": preflight,
                 "pid": os.getpid(), "started_wall_unix": time.time(), "reserved_cpu_seconds": remaining_cpu,
                 "reserved_gpu_seconds": remaining_gpu if gpu else 0.}
        stages.append(entry)
        authority.persist(ledger, f"reserve:{stage}")
        old_handlers = {s: signal.getsignal(s) for s in (signal.SIGALRM, signal.SIGXCPU, signal.SIGTERM)}
        limits = resource.getrlimit(resource.RLIMIT_CPU)
        def expired(signum, frame):
            raise TimeoutError(f"resource or termination signal {signum}")
        def guard():
            if process_cpu() > remaining_cpu or time.monotonic() - started > limit_wall:
                raise TimeoutError("continuation stage resource allowance exhausted")
        for signum in old_handlers:
            signal.signal(signum, expired)
        signal.setitimer(signal.ITIMER_REAL, limit_wall)
        soft = math.ceil(time.process_time() + max(0., remaining_cpu - process_cpu()))
        if limits[1] != resource.RLIM_INFINITY:
            soft = min(soft, limits[1])
        resource.setrlimit(resource.RLIMIT_CPU, (max(1, soft), limits[1]))
        try:
            yield guard
            guard()
            entry["status"] = "completed"
        except BaseException as exc:
            entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
            resource.setrlimit(resource.RLIMIT_CPU, limits)
            entry["cpu_seconds"] = process_cpu()
            entry["wall_seconds"] = time.monotonic() - started
            entry["gpu_seconds"] = entry["wall_seconds"] if gpu else 0.
            authority.persist(ledger, f"finish:{stage}")
