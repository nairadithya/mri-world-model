"""Reuse completed cells without changing the R38 scientific recipe."""

from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import time

import torch

from .protocol import PLAN, digest, read_json, save_json


def adopt_cell(
    data,
    original: Path,
    output: Path,
    label: str,
    identity: dict,
    initialization: dict,
    expected_sha256: str,
    old_contract_digest: str,
) -> dict:
    from .fitting import tensor_hash

    arm, seed_text = label.split("_")
    seed = int(seed_text)
    path = original / label / "epoch20.pt"
    if digest(path) != expected_sha256:
        raise ValueError("original completed checkpoint changed")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    count = PLAN.epochs * PLAN.updates_per_epoch
    if (
        saved["arm"] != arm
        or saved["seed"] != seed
        or saved["epoch"] != PLAN.epochs
        or saved["updates"] != count
        or saved["normalizer"] != data.normalizer
        or saved["contract_digest"] != old_contract_digest
        or saved["execution_scope"] != "fixed_r38_study"
        or saved["initialization"]["tensor_sha256"] != initialization["tensor_sha256"]
    ):
        raise ValueError("completed cell identity or initialization mismatch")
    if (
        not saved["optimizer"]["state"]
        or any(
            int(state["step"]) != count
            for state in saved["optimizer"]["state"].values()
        )
        or not all(
            torch.isfinite(value).all() for value in saved["state_dict"].values()
        )
    ):
        raise ValueError("checkpoint optimizer counts or finite weights invalid")
    order_path = original / f"order_{seed}.json"
    order = read_json(order_path)["orders"]
    if digest(order_path) != saved["order_sha256"]:
        raise ValueError("original training order changed")
    original_log = original / label / "updates.jsonl"
    with original_log.open() as stream:
        updates = [json.loads(line) for line in stream]
    if len(updates) != count:
        raise ValueError("completed update log has incorrect length")
    for index, record in enumerate(updates):
        epoch, step = divmod(index, PLAN.updates_per_epoch)
        row = data.by_row[order[epoch][step]]
        if (
            record["epoch"] != epoch + 1
            or record["update"] != step + 1
            or record["row_id"] != row.row_id
            or record["cutoff_mask_sha256"]
            != data.visits[row.cutoff_visit_id]["mask_sha256"]
            or record["current_supervision_visit_id"] != row.cutoff_visit_id
            or record["future_image_forwards"] != 0
            or record["earlier_state_supervision"] != 0
        ):
            raise ValueError(
                "original update log differs from bound rows or supervision"
            )
    tensor_digest = tensor_hash(saved["state_dict"])
    origin = {
        "path": str(path),
        "sha256": expected_sha256,
        "contract_digest": saved["contract_digest"],
        "implementation_source": saved["implementation_source"],
        "state_tensor_sha256": tensor_digest,
        "new_optimizer_updates": 0,
    }
    destination = output / label
    checkpoint = destination / "epoch20.pt"
    torch.save(
        {
            **saved,
            **identity,
            "initialization": initialization,
            "order_sha256": digest(output / f"order_{seed}.json"),
            "training_origin": origin,
        },
        checkpoint,
    )
    restored = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if tensor_hash(restored["state_dict"]) != tensor_digest:
        raise ValueError("adoption changed original weight tensors")
    shutil.copyfile(original_log, destination / "updates.jsonl")
    if digest(path) != expected_sha256:
        raise ValueError("original checkpoint changed during adoption")
    return {
        **identity,
        "epoch": PLAN.epochs,
        "updates": count,
        "initialization": initialization,
        "weights": {
            "path": str(checkpoint.relative_to(output)),
            "sha256": digest(checkpoint),
        },
        "updates_file": {
            "path": str((destination / "updates.jsonl").relative_to(output)),
            "sha256": digest(destination / "updates.jsonl"),
        },
        "training_origin": origin,
    }


class DirectExecution:
    """Record work in the caller process, without subprocess supervision or timers."""

    def __init__(self, out: Path):
        self.out = out
        self.phases = {}
        self.units = {}

    def check(self, remaining_updates=None):
        if remaining_updates is not None and remaining_updates < 0:
            raise ValueError("more updates than the fixed schedule")

    def completed(self, unit):
        self.units[unit] = self.units.get(unit, 0) + 1
        if self.units[unit] % PLAN.updates_per_epoch == 0:
            print(
                f"{unit}: {self.units[unit]}/{PLAN.epochs * PLAN.updates_per_epoch}",
                flush=True,
            )
            save_json(
                self.out / "progress.json", {"units": self.units, "phases": self.phases}
            )

    @contextmanager
    def phase(self, name):
        start, cpu = time.monotonic(), time.process_time()
        print(f"phase {name}: started", flush=True)
        try:
            yield
        finally:
            self.phases[name] = {
                "wall_seconds": time.monotonic() - start,
                "cpu_seconds": time.process_time() - cpu,
            }
            save_json(
                self.out / "progress.json", {"units": self.units, "phases": self.phases}
            )
