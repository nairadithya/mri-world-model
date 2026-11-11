"""Continue the fixed R38 matrix in one process, preserving the interrupted run."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.harness.analysis.joint_dynamics.protocol import (
    PLAN,
    REPO,
    WORKSPACE,
    SourceIdentity,
    canonical_hash,
    digest,
    read_json,
    save_json,
)


def prepare_continuation(pins_path: Path):
    from src.harness.analysis.joint_dynamics.operations import load_prepared

    pins = read_json(pins_path)
    payload_path = WORKSPACE / "study/requests/r38-fixed-study-02/payload.json"
    if digest(payload_path) != pins["original_payload_sha256"]:
        raise ValueError("original run payload changed")
    payload = read_json(payload_path)
    original = (
        Path(pins["reuse_from"])
        if "reuse_from" in pins
        else WORKSPACE / "study/stages/r38-fixed-study-02/fit"
    )
    if payload["contract"]["scientific_plan"] != json.loads(json.dumps(asdict(PLAN))):
        raise ValueError("scientific plan differs from original run")
    for file in ("model.py", "data.py", "protocol.py", "baselines.py", "assessment.py"):
        path = "src/harness/analysis/joint_dynamics/" + file
        if (
            digest(REPO / path)
            != payload["contract"]["implementation_source"]["files"][path]
        ):
            raise ValueError(f"scientific primitive changed: {path}")
    seed42_cells = {"OH_42", "OL_42", "JH_42", "JL_42"}
    seed42_43_cells = seed42_cells | {"OH_43", "OL_43", "JH_43", "JL_43"}
    if set(pins["reused_cells"]) not in (seed42_cells, seed42_43_cells):
        raise ValueError("continuation must reuse completed seed42 or seed42/43 cells")
    for label, sha in pins["reused_cells"].items():
        if digest(original / label / "epoch20.pt") != sha:
            raise ValueError("original checkpoint differs from preserved local backup")
    reused_fraction = len(pins["reused_cells"]) / (len(PLAN.arms) * len(PLAN.seeds))
    projection = read_json(WORKSPACE / "study/projection-r38-fixed-study-02.json")
    required_gpu = (
        projection["required_gpu_with_reserves_cleanup"]
        - projection["phases"]["updates"]["gpu_seconds"] * reused_fraction
    )
    required_cpu = (
        projection["required_cpu_with_reserves_cleanup"]
        - projection["phases"]["updates"]["cpu_seconds"] * reused_fraction
    )
    if (
        required_gpu > pins["remaining_allowance"]["gpu_seconds"]
        or required_cpu > pins["remaining_allowance"]["cpu_seconds"]
    ):
        raise ValueError(
            "remaining fixed workload exceeds the preserved planning allowance"
        )
    data, meta = load_prepared(payload)
    return pins, payload, original, data, meta


def run(pins_path: Path, output: Path, check_only: bool):
    if ROOT != REPO:
        raise RuntimeError("run from the existing mosaic-gpu R38 repository")
    started, cpu = time.monotonic(), time.process_time()
    pins, payload, original, data, meta = prepare_continuation(pins_path)
    from src.harness.analysis.joint_dynamics.continuation import (
        DirectExecution,
        adopt_cell,
    )
    from src.harness.analysis.joint_dynamics.fitting import fit_matrix, tensor_hash
    from src.harness.analysis.joint_dynamics.model import configure_torch
    import torch

    configure_torch()
    output.mkdir(parents=True, exist_ok=False)
    os.chmod(output, 0o700)
    source = SourceIdentity.capture()
    reused_contract_digest = pins.get(
        "reused_contract_digest", canonical_hash(payload["contract"])
    )
    contract = {
        **payload["contract"],
        "implementation_source": source.to_dict(),
        "continuation": {
            **pins,
            "original_contract_digest": reused_contract_digest,
            "operator_direction": "October30: finish unchanged R38, reusing completed cells; no custom supervisor",
            "entrypoint_sha256": digest(Path(__file__)),
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "prior_failure_retained": True,
            "restart_incomplete_OH43": "original initialization and order; no resumable OH43 checkpoint survived",
        },
    }
    save_json(output / "contract.json", contract)
    if check_only:
        from src.harness.analysis.joint_dynamics.fitting import (
            replacement_order,
            seed_all,
        )
        from src.harness.analysis.joint_dynamics.model import JointDynamics

        for seed in PLAN.seeds:
            seed_all(seed)
            model = JointDynamics(data.normalizer["mean"], data.normalizer["scale"])
            original_init = torch.load(
                original / f"initialization_{seed}.pt",
                map_location="cpu",
                weights_only=False,
            )
            if tensor_hash(model.state_dict()) != tensor_hash(
                original_init["state_dict"]
            ):
                raise ValueError(
                    "original initialization does not match the unchanged seeded model"
                )
            expected_order = [
                list(epoch)
                for epoch in replacement_order(data.select("source52"), seed)
            ]
            if read_json(original / f"order_{seed}.json")["orders"] != expected_order:
                raise ValueError(
                    "original sampling order differs from the unchanged schedule"
                )
        identity = {
            "contract_digest": canonical_hash(contract),
            "implementation_source": source.to_dict(),
            "execution_scope": "fixed_r38_study",
            "row_manifest_sha256": canonical_hash(data.payload["rows"]),
        }
        for label, sha in pins["reused_cells"].items():
            seed = label.split("_")[1]
            (output / label).mkdir()
            original_init = original / f"initialization_{seed}.pt"
            saved = torch.load(original_init, map_location="cpu", weights_only=False)
            initialization = {
                "path": original_init.name,
                "sha256": digest(original_init),
                "tensor_sha256": tensor_hash(saved["state_dict"]),
            }
            (output / f"order_{seed}.json").write_bytes(
                (original / f"order_{seed}.json").read_bytes()
            )
            adopt_cell(
                data,
                original,
                output,
                label,
                identity,
                initialization,
                sha,
                reused_contract_digest,
            )
        reused_updates = (
            len(pins["reused_cells"]) * PLAN.epochs * PLAN.updates_per_epoch
        )
        result = {
            "status": "CONTINUATION_INPUTS_AND_COMPLETED_CELLS_VERIFIED",
            "gpu_updates": 0,
            "reused_cells": list(pins["reused_cells"]),
            "remaining_updates": PLAN.total_updates - reused_updates,
            "cohort_rows": {
                name: len(data.select(name))
                for name in (
                    "source52",
                    "validation13",
                    "primary",
                    "single",
                    "nuisance",
                )
            },
            "wall_seconds": time.monotonic() - started,
            "cpu_seconds": time.process_time() - cpu,
        }
        save_json(output / "check.json", result)
        print(json.dumps(result, indent=2), flush=True)
        return
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("the fixed CUDA BF16 policy is unavailable")
    work = DirectExecution(output)
    fitted = fit_matrix(
        data,
        output / "fit",
        contract,
        work,
        meta["objects"]["geometry"],
        reuse_from=original,
    )
    from src.harness.analysis.joint_dynamics.operations import validate_fitted
    from src.harness.analysis.joint_dynamics.assessment import (
        AssessmentIdentity,
        analyze,
    )
    import numpy as np

    with work.phase("assessment"):
        root = output / "fit"
        validate_fitted(data, root, contract, canary=False)
        identity = AssessmentIdentity(
            source,
            canonical_hash(contract),
            digest(root / "fitted.json"),
            fitted["control_fits"],
            fitted["cohort_rows"],
            fitted["calibration"],
        )
        with np.load(root / "private_predictions.npz", allow_pickle=False) as stored:
            arrays = {key: stored[key] for key in stored.files}
        report = analyze(arrays, identity)
        save_json(output / "aggregate.json", report)
    data.verify_all_bytes(meta)
    save_json(
        output / "complete.json",
        {
            "status": "COMPUTATION_COMPLETE_PENDING_RESULT_REVIEW",
            "aggregate_sha256": digest(output / "aggregate.json"),
            "fitted_sha256": digest(root / "fitted.json"),
            "new_updates": fitted["new_updates"],
            "reused_updates": PLAN.total_updates - fitted["new_updates"],
            "total_updates": fitted["total_updates"],
            "phases": work.phases,
            "prior_charges": pins["prior_charges"],
            "wall_seconds": time.monotonic() - started,
            "cpu_seconds": time.process_time() - cpu,
        },
    )
    print("COMPUTATION_COMPLETE_PENDING_RESULT_REVIEW", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pins", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    try:
        run(args.pins, args.out, args.check_only)
    except Exception:
        if args.out.is_dir():
            save_json(
                args.out / "failure.json",
                {
                    "status": "INCOMPLETE_NOT_GRADED",
                    "traceback": traceback.format_exc(),
                },
            )
        raise


if __name__ == "__main__":
    main()
