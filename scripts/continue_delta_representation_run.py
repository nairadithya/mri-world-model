"""Finish the authorized fixed Phase 1 jobs; never launch Phase 2 or retry a failed arm."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import time


def command(args: list[str], log: Path | None = None) -> str:
    result = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
    if log is not None:
        log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"command failed: {args}; see {log}; response: {result.stdout[-4000:]}")
    return result.stdout


def status(kernel: str) -> str:
    text = command(["kaggle", "kernels", "status", "-k", kernel])
    match = re.search(r"KernelWorkerStatus\.(\w+)", text)
    if match is None:
        raise RuntimeError(f"unrecognized status for {kernel}: {text}")
    return match.group(1)


def wait(kernels: list[str], seconds: float, record) -> None:
    deadline = time.monotonic() + seconds
    while True:
        states = {k: status(k) for k in kernels}
        record("waiting", states)
        if all(s == "COMPLETE" for s in states.values()):
            return
        if any(s not in ("COMPLETE", "RUNNING", "QUEUED") for s in states.values()):
            raise RuntimeError(f"a fixed job did not complete: {states}")
        if time.monotonic() >= deadline:
            raise RuntimeError("mechanical wait ceiling reached; no automatic retry or further launch")
        time.sleep(120)


def collect(root: Path, job: str, kernel: str, filename: str) -> Path:
    directory = root / "jobs" / job; directory.mkdir(parents=True, exist_ok=True)
    command(["kaggle", "kernels", "output", kernel, "-p", str(directory),
             "--file-pattern", re.escape(filename) + "$"], root / "logs" / f"fetch-{job}.log")
    unpacked = directory / "unpacked"; unpacked.mkdir(exist_ok=True)
    with tarfile.open(directory / filename) as archive:
        archive.extractall(unpacked, filter="data")
    return unpacked / "results"


def merge(results: Path, destination: Path) -> None:
    for directory in results.glob("seed*"):
        for source in directory.rglob("*"):
            if not source.is_file():
                continue
            target = destination / source.relative_to(results)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != source.read_bytes():
                raise ValueError(f"refuse to overwrite differing artifact {target}")
            if not target.exists():
                shutil.copy2(source, target)


def completed_probe_seed(root: Path, seed: int, r35_results: Path) -> bool:
    """Only bound, complete local probes authorize skipping a seed on restart."""
    from src.harness.analysis.delta_protocol import load_contract
    from src.harness.analysis.delta_representation import load_probe
    import torch

    aggregate = root / "matched-final"
    paths = [aggregate / f"seed{seed}/{arm}/{view}/probe.pt"
             for arm in ("A", "B", "C", "D", "R35") for view in ("difference", "target_only")]
    if not all(path.exists() for path in paths):
        return False
    contract = load_contract(str(root / "qa/r36_contract.json"))
    schedules = []
    for path in paths:
        arm, view = path.parent.parent.name, path.parent.name
        checkpoint = (r35_results / f"seed{seed}/state/best.pt" if arm == "R35"
                      else path.parent.parent / "final.pt")
        probe = load_probe(str(path), str(checkpoint), contract, view)
        if not all(torch.isfinite(value).all() for value in probe.state_dict().values()):
            raise ValueError("nonfinite completed probe weights")
        receipt = json.loads((path.parent / "training.json").read_text())
        if (receipt["status"] != "complete" or len(receipt["history"]) != 20
                or [row["epoch"] for row in receipt["history"]] != list(range(1, 21))):
            raise ValueError("completed probe requires all twenty epochs")
        schedules.append([row["pair_order_sha256"] for row in receipt["history"]])
    if any(schedule != schedules[0] for schedule in schedules[1:]):
        raise ValueError("completed probes have differing training schedules")
    return True


def ensure_probe_launch(root: Path, seed: int, record) -> None:
    """A durable launch intent (including ambiguous failure) forbids another push."""
    job = f"probes{seed}"
    intent = root / f"{job}-launch-intent.json"
    log = root / "logs" / f"push-{job}.log"
    if intent.exists() or log.exists():
        record("reattaching_source_probes", {"seed": seed, "new_push": False})
        return
    quota = command(["kaggle", "quota"], root / f"quota-before-probes{seed}.txt")
    match = re.search(r"GPU\s+([\d.]+)h\s+([\d.]+)h", quota)
    if match is None:
        raise ValueError("quota receipt not understood")
    spent = float(match.group(1)) - 5.72
    remaining_ceiling = (44 - seed + 1) * 3.5
    if spent + remaining_ceiling > 24 or float(match.group(2)) < remaining_ceiling:
        raise RuntimeError("remaining fixed work does not fit the approved GPU ceiling")
    evidence = {"seed": seed, "spent_gpu_hours": spent,
                "remaining_session_ceiling_hours": remaining_ceiling}
    # Exclusive creation also refuses concurrent launchers; preserve intent on any failure.
    with intent.open("x") as stream:
        json.dump(evidence, stream)
    record("launching_source_probes", evidence)
    command(["kaggle", "kernels", "push", "-p", str(root / f"private-probes{seed}"),
             "-t", "12600", "--accelerator", "NvidiaTeslaT4"], log)


def run(root: Path, r35_results: Path, sailor: Path) -> None:
    root = root.resolve(); aggregate = root / "matched-final"; aggregate.mkdir(exist_ok=True)
    def record(state: str, evidence: dict) -> None:
        payload = {"state": state, "evidence": evidence, "time": datetime.now(timezone.utc).isoformat(),
                   "independent_review": "pending_not_yet_completed", "phase2": "not_authorized"}
        (root / "continuation.json").write_text(json.dumps(payload, indent=2) + "\n")
        with (root / "continuation-events.jsonl").open("a") as stream:
            stream.write(json.dumps(payload) + "\n")
    try:
        capacity = root / "capacity-complete/unpacked/results"
        if json.loads((capacity / "capacity.json").read_text())["status"] != "capacity_complete":
            raise ValueError("completed capacity receipt required")
        merge(capacity, aggregate)
        matrices = [f"nairadithya/r36-delta-matrix{i}" for i in (1, 2)]
        # Wall time includes Kaggle scheduling delay; GPU session ceilings stay unchanged.
        wait(matrices, 12 * 3600, record)
        for index, kernel in enumerate(matrices, 1):
            results = collect(root, f"matrix{index}", kernel, "r36-train-results.tar.gz")
            receipt = json.loads((results / "run.json").read_text())
            if receipt["status"] != "complete" or len(receipt["runs"]) != 5:
                raise ValueError("all fixed training work must complete; partial tables are not scored")
            merge(results, aggregate)
        for seed in (42, 43, 44):
            if completed_probe_seed(root, seed, r35_results):
                record("reusing_verified_source_probes", {"seed": seed, "new_push": False})
                continue
            job = f"probes{seed}"; kernel = f"nairadithya/r36-probes-seed{seed}"
            ensure_probe_launch(root, seed, record)
            command(["kaggle", "kernels", "pull", kernel, "-p", str(root / f"server-{job}"), "-m"],
                    root / "logs" / f"server-{job}.log")
            metadata = json.loads((root / f"server-{job}/kernel-metadata.json").read_text())
            if metadata["is_private"] is not True or metadata.get("machine_shape") != "NvidiaTeslaT4":
                raise RuntimeError("private T4 server settings required")
            wait([kernel], 10 * 3600, record)
            results = collect(root, job, kernel, "r36-probes-results.tar.gz")
            receipt = json.loads((results / "run.json").read_text())
            if receipt["status"] != "complete" or len(receipt["runs"]) != (8 if seed == 42 else 10):
                raise ValueError("all source-fitted readouts must complete before query assessment")
            merge(results, aggregate)
            if not completed_probe_seed(root, seed, r35_results):
                raise ValueError("complete bound source probes required after collection")
        for seed in (42, 43, 44):
            for arm in ("A", "B", "C", "D", "R35"):
                for view in ("difference", "target_only"):
                    if not (aggregate / f"seed{seed}/{arm}/{view}/probe.pt").exists():
                        raise ValueError("complete thirty-probe preflight failed")
        record("local_assessment_started", {"results": str(aggregate), "sailor_upload": False})
        with (root / "logs/evaluation-driver.log").open("w") as log:
            completed = subprocess.run([sys.executable, "-u", "scripts/evaluate_delta_representation_run.py",
                "--results", str(aggregate), "--prepared", str(root / "lumiere-tensors"),
                "--sailor-prepared", str(sailor.resolve()), "--contract", str(root / "qa/r36_contract.json"),
                "--audit", str(root / "qa/audit.json"), "--r35-results", str(r35_results.resolve()),
                "--probe-hours", "0"], stdout=log, stderr=subprocess.STDOUT, timeout=14 * 3600)
        if completed.returncode:
            raise RuntimeError("local assessment failed; inspect evaluation-driver.log")
        record("commands_complete_review_pending", {"report": str(aggregate / "phase1-report.json")})
    except Exception as error:
        record("incomplete_operator_or_parent_review_required", {"error": str(error)})
        raise


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--r35-results", type=Path, required=True)
    parser.add_argument("--sailor-prepared", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.r35_results, args.sailor_prepared)
