"""Drive every R36 final encoder through the real CLI, with local-only transfer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.harness.analysis.delta_protocol import ARMS, load_contract
from src.harness.analysis.delta_representation import load_final, load_probe
from src.data.representation import save_json


def run(results: str, prepared: str, sailor_prepared: str, contract_path: str, audit: str,
        r35_results: str, probe_hours: float) -> dict:
    root = Path(results)
    contract = load_contract(contract_path)
    for seed in contract["seeds"]:
        for arm in ARMS:
            path = root / f"seed{seed}" / arm / "final.pt"
            load_final(str(path), contract)
            receipt = json.loads((path.parent / "training.json").read_text())
            if receipt["status"] != "complete" or len(receipt["history"]) != 20:
                raise ValueError("all twelve final arms must complete before comparative query evaluation")
    commands = []
    def cli(arguments):
        command = [sys.executable, "-u", "scripts/harness.py", "delta-representation", *arguments]
        directory = root / "evaluation-logs"; directory.mkdir(exist_ok=True)
        log_path = directory / f"{len(commands):03d}-{arguments[0]}.log"
        with log_path.open("w") as stream:
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        commands.append({"command": command, "exit_status": completed.returncode, "log": str(log_path)})
        save_json(root / "evaluation_commands.json", {"commands": commands, "status": "running"})
        if completed.returncode:
            raise RuntimeError(f"evaluation command failed; inspect {log_path}")
    for seed in contract["seeds"]:
        for arm in (*ARMS, "R35"):
            directory = root / f"seed{seed}" / arm; directory.mkdir(parents=True, exist_ok=True)
            checkpoint = (Path(r35_results) / f"seed{seed}" / "state/best.pt" if arm == "R35" else directory / "final.pt")
            base = ["--checkpoint", str(checkpoint), "--contract", contract_path, "--seed", str(seed)]
            for view in ("difference", "target_only"):
                probe = directory / view / "probe.pt"
                if probe.exists():
                    load_probe(str(probe), str(checkpoint), contract, view)
                else:
                    cli(["probe", "--prepared", prepared, *base, "--audit", audit, "--view", view,
                         "--hours", str(probe_hours), "--out", str(probe.parent)])
                    load_probe(str(probe), str(checkpoint), contract, view)
            for cohort, query_root in (("lumiere", prepared), ("sailor", sailor_prepared)):
                feature_path = directory / f"{cohort}-features.pt"
                cli(["encode", "--prepared", query_root, *base, "--out", str(feature_path)])
                cli(["eval" if cohort == "lumiere" else "transfer", "--prepared", query_root, *base,
                     "--audit", audit, "--probe", str(directory / "difference/probe.pt"),
                     "--target-probe", str(directory / "target_only/probe.pt"), "--out", str(directory / f"{cohort}.json")])
                cli(["ridge", "--prepared", prepared, "--query-prepared", query_root,
                     "--training-features", str(directory / "lumiere-features.pt"),
                     "--query-features", str(feature_path), "--contract", contract_path,
                     "--out", str(directory / f"ridge-{cohort}.json")])
            cli(["nuisance", "--prepared", prepared, *base, "--audit", audit,
                 "--probe", str(directory / "difference/probe.pt"), "--out", str(directory / "nuisance.json")])
    cli(["report", "--results", results, "--contract", contract_path, "--out", str(root / "phase1-report.json")])
    result = {"status": "commands_complete_review_pending", "commands": commands,
              "independent_review": "unavailable", "phase2": "operator_authorization_required"}
    save_json(root / "evaluation_commands.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True); parser.add_argument("--prepared", required=True)
    parser.add_argument("--sailor-prepared", required=True); parser.add_argument("--contract", required=True)
    parser.add_argument("--audit", required=True); parser.add_argument("--r35-results", required=True)
    parser.add_argument("--probe-hours", type=float, default=8)
    args = parser.parse_args()
    result = run(args.results, args.prepared, args.sailor_prepared, args.contract, args.audit, args.r35_results, args.probe_hours)
    print(result["status"])
