"""Private, fixed R38 budget authority and append-only stage journal."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import resource
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

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


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _time() -> float:
    return time.time()


def _deadline_epoch() -> float:
    return datetime.fromisoformat(DEADLINE).timestamp()


class BudgetGuard:
    def __init__(self, authority: "BudgetAuthority", fd: int, started: float, child_cpu: float):
        self.authority, self.fd, self.started, self.child_cpu = authority, fd, started, child_cpu

    def check(self, projected_gpu_seconds: float = 0.0, projected_cpu_seconds: float = 0.0) -> None:
        snap = self.authority._snapshot_locked()
        elapsed = _time() - self.started
        cpu = time.process_time() - self.authority._process_start + self._child_cpu() - self.child_cpu
        gpu = elapsed if self.authority._active_gpu else 0.0
        if _time() + max(projected_gpu_seconds, projected_cpu_seconds) > _deadline_epoch():
            raise RuntimeError("R38 deadline would be exceeded")
        if snap["charged_gpu_seconds"] + gpu + projected_gpu_seconds > GPU_LIMIT:
            raise RuntimeError("R38 GPU allowance would be exceeded")
        if snap["charged_cpu_seconds"] + cpu + projected_cpu_seconds > CPU_LIMIT:
            raise RuntimeError("R38 CPU allowance would be exceeded")

    def _child_cpu(self) -> float:
        r = resource.getrusage(resource.RUSAGE_CHILDREN)
        return r.ru_utime + r.ru_stime

    def snapshot(self) -> dict:
        return self.authority._snapshot_locked()


class BudgetAuthority:
    def __init__(self, ledger: Path, *, _private_test: bool = False):
        if not _private_test and ledger != LEDGER:
            raise ValueError("production ledger path is fixed")
        self.ledger = ledger
        self.lock_path = ledger.with_suffix(ledger.suffix + ".lock")
        self._process_start = time.process_time()
        self._active_gpu = False

    def initialize(self) -> dict:
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with self._lock():
            if self.ledger.exists():
                raise RuntimeError("R38 authority already initialized; refusing reset")
            if _time() >= _deadline_epoch():
                raise RuntimeError("R38 authorization window has expired")
            payload = {"identity": IDENTITY, "charged_cpu_seconds": SETUP_SECONDS,
                       "charged_gpu_seconds": 0.0, "stages": [], "head": "0" * 64}
            self._write(payload)
            return self._snapshot_locked()

    @contextmanager
    def _lock(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield fd
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _read(self) -> dict:
        payload = json.loads(self.ledger.read_text())
        if payload.get("identity") != IDENTITY:
            raise RuntimeError("R38 authority identity mismatch")
        head = "0" * 64
        cpu, gpu = SETUP_SECONDS, 0.0
        names = set()
        for item in payload["stages"]:
            core = {k: v for k, v in item.items() if k != "hash"}
            if item.get("parent") != head or item.get("hash") != _digest(core):
                raise RuntimeError("R38 journal ancestry/hash mismatch")
            if item["name"] in names:
                raise RuntimeError("duplicate R38 stage name")
            names.add(item["name"])
            head = item["hash"]
            cpu += item["cpu_seconds"]
            gpu += item["gpu_seconds"]
        if payload["head"] != head or abs(payload["charged_cpu_seconds"] - cpu) > 1e-5 or abs(payload["charged_gpu_seconds"] - gpu) > 1e-5:
            raise RuntimeError("R38 ledger totals/head mismatch")
        return payload

    def _write(self, payload: dict) -> None:
        tmp = self.ledger.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
        os.chmod(tmp, 0o600)
        os.replace(tmp, self.ledger)

    def _snapshot_locked(self) -> dict:
        p = self._read()
        return {**p, "remaining_cpu_seconds": CPU_LIMIT-p["charged_cpu_seconds"],
                "remaining_gpu_seconds": GPU_LIMIT-p["charged_gpu_seconds"]}

    def snapshot(self) -> dict:
        with self._lock():
            return self._snapshot_locked()

    def verify_extension(self, snapshot: dict) -> None:
        current = self.snapshot()
        if snapshot.get("identity") != IDENTITY:
            raise RuntimeError("snapshot authority identity mismatch")
        old = snapshot.get("stages", [])
        if current["stages"][:len(old)] != old:
            raise RuntimeError("snapshot is not an ancestor of current authority")

    @contextmanager
    def stage(self, name: str, gpu: bool = False, scientific: bool = False) -> Iterator[BudgetGuard]:
        if not name or not isinstance(name, str):
            raise ValueError("stage name must be nonempty")
        with self._lock() as fd:
            p = self._read()
            if p["stages"] and p["stages"][-1].get("scientific") and p["stages"][-1].get("status") != "succeeded":
                raise RuntimeError("failed scientific stage blocks continuation")
            if any(x["name"] == name for x in p["stages"]):
                raise RuntimeError("stage name already used")
            if _time() >= _deadline_epoch():
                raise RuntimeError("R38 authorization window has expired")
            started = _time()
            process_start = time.process_time()
            child_start = resource.getrusage(resource.RUSAGE_CHILDREN)
            self._active_gpu = gpu
            guard = BudgetGuard(self, fd, started, child_start.ru_utime + child_start.ru_stime)
            status, failure = "succeeded", None
            try:
                guard.check()
                yield guard
                guard.check()
            except BaseException as exc:
                status, failure = "failed", repr(exc)
                raise
            finally:
                self._active_gpu = False
                ended = _time()
                child_end = resource.getrusage(resource.RUSAGE_CHILDREN)
                child_cpu = max(0.0, child_end.ru_utime + child_end.ru_stime - (child_start.ru_utime + child_start.ru_stime))
                cpu = max(0.0, time.process_time() - process_start) + child_cpu
                wall = ended - started
                stage_gpu = wall if gpu else 0.0
                # Record every attempt before releasing the reservation, including failures.
                p = self._read()
                core = {"name": name, "scientific": bool(scientific), "status": status,
                        "started": started, "ended": ended, "cpu_seconds": cpu,
                        "gpu_seconds": stage_gpu, "failure": failure, "parent": p["head"]}
                item = {**core, "hash": _digest(core)}
                p["stages"].append(item)
                p["head"] = item["hash"]
                p["charged_cpu_seconds"] += cpu
                p["charged_gpu_seconds"] += stage_gpu
                if p["charged_cpu_seconds"] > CPU_LIMIT or p["charged_gpu_seconds"] > GPU_LIMIT or ended > _deadline_epoch():
                    item["status"] = "failed"
                    item["failure"] = "final budget/deadline check failed"
                    item.pop("hash")
                    item["hash"] = _digest(item)
                    p["head"] = item["hash"]
                    raise RuntimeError("R38 final budget check failed")
                self._write(p)


def production_authority() -> BudgetAuthority:
    return BudgetAuthority(LEDGER)


def _private_test_authority(path: Path) -> BudgetAuthority:
    """Explicit factory reserved for isolated unit tests."""
    return BudgetAuthority(path, _private_test=True)
