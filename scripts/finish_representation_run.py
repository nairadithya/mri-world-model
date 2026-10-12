"""Run the fixed local R35 transfer matrix without fitting on SAILOR."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.representation import digest, save_json


def finish(prepared: str, results: str, training_metadata: str, out: str) -> dict:
    root, output = Path(results), Path(out)
    output.mkdir(parents=True, exist_ok=True)
    status = {"status": "running", "prepared_sha256": digest(Path(prepared) / "prepared.json"),
              "training_metadata_sha256": digest(training_metadata), "runs": []}
    for seed in (42, 43, 44):
        for arm in ("state", "paired", "later_only", "frozen"):
            source = root / f"seed{seed}" / arm
            target = output / f"seed{seed}" / arm / "transfer.json"
            record = {"seed": seed, "arm": arm, "status": "running", "output": str(target),
                      "checkpoint_sha256": digest(source / "best.pt")}
            status["runs"].append(record)
            save_json(output / "status.json", status)
            print(f"transfer seed={seed} arm={arm}", flush=True)
            try:
                subprocess.run([sys.executable, "-u", "scripts/harness.py", "representation", "transfer",
                                "--prepared", prepared, "--checkpoint", str(source / "best.pt"),
                                "--training-metadata", training_metadata, "--training-features", str(source / "features.pt"),
                                "--out", str(target)], check=True)
                record["status"] = "complete"
            except subprocess.CalledProcessError as error:
                record["status"] = "error"
                record["returncode"] = error.returncode
                status["status"] = "error"
                save_json(output / "status.json", status)
                raise
            save_json(output / "status.json", status)
    status["status"] = "complete"
    save_json(output / "status.json", status)
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--training-metadata", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(finish(args.prepared, args.results, args.training_metadata, args.out), sort_keys=True))
