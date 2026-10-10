"""Build bounded follow-on private kernels from the verified capacity snapshot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.representation import save_json


def package(capacity: str, out: str, mode: str, seed: int | None, hours: float, shard: int | None = None) -> dict:
    if not 0 < hours <= 8.5:
        raise ValueError("reserve setup and finalization inside the nine-hour session ceiling")
    source = Path(capacity)
    text = (source / "capacity_leg.py").read_text()
    prefix = text.split("# %%\nstarted = time.monotonic()", 1)[0]
    if prefix == text:
        raise ValueError("capacity notebook prefix is not the expected version")
    prefix = prefix.replace("# # R36 capacity check", f"# # R36 locked {mode} session")
    prefix = prefix.replace("# Exactly 20 epochs of seed42 A and B. Final epochs only.",
                            "# Locked source-only work. Exactly 20 epochs. Final epochs only.")
    if mode == "train":
        if shard not in (1, 2) or seed is not None:
            raise ValueError("training requires one fixed shard and no probe seed")
        matrix = [(42, "C"), (42, "D"), *[(s, a) for s in (43, 44) for a in ("A", "B", "C", "D")]]
        work = matrix[(shard - 1) * 5:shard * 5]
        loop = f'''for seed, arm in {work!r}:
        remaining = {hours} - (time.monotonic() - started) / 3600
        if remaining <= 0:
            run["status"] = "incomplete_runtime_ceiling"
            break
        paths = [p for p in Path("/kaggle/input").glob(f"**/seed{{seed}}/state/best.pt")
                 if hashlib.sha256(p.read_bytes()).hexdigest() == contract["initial_checkpoint_sha256"][str(seed)]]
        assert len(paths) == 1, "Seed-matched initialization missing or ambiguous"
        subprocess.run(["python", "-u", "-m", "src.harness.analysis.delta_representation", "train",
                        "--prepared", str(prepared), "--checkpoint", str(paths[0]),
                        "--contract", str(qa / "r36_contract.json"), "--audit", str(qa / "audit.json"),
                        "--seed", str(seed), "--arm", arm, "--hours", str(remaining),
                        "--out", f"results/seed{{seed}}/{{arm}}"], check=True)
        receipt = json.loads(Path(f"results/seed{{seed}}/{{arm}}/training.json").read_text())
        run["runs"][f"{{seed}}/{{arm}}"] = {{"status": receipt["status"], "epochs": len(receipt["history"]), "seconds": receipt["seconds"]}}
        Path("results/run.json").write_text(json.dumps(run, indent=2))
        if receipt["status"] != "complete":
            run["status"] = "incomplete_runtime_ceiling"
            break
    else:
        run["status"] = "complete"
'''
        identity, title = f"r36-delta-matrix{shard}", f"R36 delta matrix{shard}"
        upstream = []
    else:
        if seed not in (42, 43, 44) or shard is not None:
            raise ValueError("probe jobs require one locked seed and no training shard")
        views = [(arm, view) for arm in ("A", "B", "C", "D", "R35") for view in ("difference", "target_only")]
        if seed == 42:
            views = [(a, v) for a, v in views if a != "R35"]
        work = [(seed, a, v) for a, v in views]
        loop = f'''for seed, arm, view in {work!r}:
        remaining = {hours} - (time.monotonic() - started) / 3600
        if remaining <= 0:
            run["status"] = "incomplete_runtime_ceiling"
            break
        if arm == "R35":
            paths = [p for p in Path("/kaggle/input").glob(f"**/seed{{seed}}/state/best.pt")
                     if hashlib.sha256(p.read_bytes()).hexdigest() == contract["initial_checkpoint_sha256"][str(seed)]]
        else:
            paths = list(Path("/kaggle/input").glob(f"**/seed{{seed}}/{{arm}}/final.pt"))
        assert len(paths) == 1, "Frozen final encoder missing or ambiguous"
        probe_started = time.monotonic()
        subprocess.run(["python", "-u", "-m", "src.harness.analysis.delta_representation", "probe",
                        "--prepared", str(prepared), "--checkpoint", str(paths[0]),
                        "--contract", str(qa / "r36_contract.json"), "--audit", str(qa / "audit.json"),
                        "--seed", str(seed), "--view", view, "--hours", str(remaining),
                        "--out", f"results/seed{{seed}}/{{arm}}/{{view}}"], check=True)
        receipt = json.loads(Path(f"results/seed{{seed}}/{{arm}}/{{view}}/training.json").read_text())
        run["runs"][f"{{seed}}/{{arm}}/{{view}}"] = {{"status": receipt["status"], "epochs": len(receipt["history"]), "seconds": time.monotonic() - probe_started}}
        Path("results/run.json").write_text(json.dumps(run, indent=2))
        if receipt["status"] != "complete":
            run["status"] = "incomplete_runtime_ceiling"
            break
    else:
        run["status"] = "complete"
'''
        identity, title = f"r36-probes-seed{seed}", f"R36 probes seed{seed}"
        upstream = ["nairadithya/r36-delta-capacity", "nairadithya/r36-delta-matrix1", "nairadithya/r36-delta-matrix2"]
    suffix = f'''# %%
started = time.monotonic()
run = {{"status": "running", "runs": {{}}, "source_commit": snapshot["commit"], "mode": "{mode}"}}
Path("results").mkdir(exist_ok=True)
try:
    {loop}except Exception as error:
    run["status"] = "failed"
    run["error"] = str(error)
    raise
finally:
    run["seconds"] = time.monotonic() - started
    Path("results/run.json").write_text(json.dumps(run, indent=2))
    subprocess.run(["tar", "-czf", "/kaggle/working/r36-{mode}-results.tar.gz", "results", "source_snapshot.json", "environment.txt"], check=True)
print(json.dumps(run, indent=2))
'''
    directory = Path(out); directory.mkdir(parents=True, exist_ok=True)
    filename = f"{mode}_leg.py"
    (directory / filename).write_text(prefix + suffix)
    metadata = json.loads((source / "kernel-metadata.json").read_text())
    metadata.update(id=f"nairadithya/{identity}", title=title, code_file=filename.replace(".py", ".ipynb"),
                    kernel_sources=list(dict.fromkeys([*metadata["kernel_sources"], *upstream])))
    save_json(directory / "kernel-metadata.json", metadata)
    receipt = {"mode": mode, "work": work, "hours_ceiling": hours, "private": True,
               "query_scoring": False, "snapshot": json.loads((source / "package_receipt.json").read_text())}
    save_json(directory / "package_receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity", required=True); parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=("train", "probes"), required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44)); parser.add_argument("--hours", type=float, default=8.5)
    parser.add_argument("--shard", type=int, choices=(1, 2))
    args = parser.parse_args()
    print(json.dumps(package(args.capacity, args.out, args.mode, args.seed, args.hours, args.shard), sort_keys=True))
