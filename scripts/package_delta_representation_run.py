"""Package a bounded private LUMIERE-only R36 capacity session."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.representation import digest, save_json
from src.harness.analysis.delta_protocol import load_contract, verify_tensor_manifest
from src.harness.analysis.delta_representation import SOURCE_FILES


def package(prepared: str, qa: str, out: str) -> dict:
    qa_root = Path(qa)
    if qa_root.is_absolute() or ".." in qa_root.parts:
        raise ValueError("QA directory must be a repository-relative private output path")
    contract = load_contract(qa_root / "r36_contract.json")
    audit = json.loads((qa_root / "audit.json").read_text())
    if audit["visual_review_status"] != "reviewed":
        raise ValueError("review QA pixels before packaging")
    verify_tensor_manifest(prepared, audit["tensor_manifest"])
    metadata = json.loads((Path(prepared) / "prepared.json").read_text())
    if metadata["cohort"] != "LUMIERE" or any(not v["patient_id"].startswith("Patient-") for v in metadata["visits"]):
        raise ValueError("GPU inputs must contain LUMIERE only")
    files = (*SOURCE_FILES, "src/preprocessing/transforms.py", "tests/test_delta_representation.py")
    for filename in files:
        subprocess.run(["git", "ls-files", "--error-unmatch", filename], check=True, stdout=subprocess.DEVNULL)
    if subprocess.check_output(["git", "diff", "HEAD", "--", *files], text=True):
        raise ValueError("commit experiment code before packaging")
    archive_data = io.BytesIO()
    with zipfile.ZipFile(archive_data, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename in files:
            archive.writestr(filename, Path(filename).read_bytes())
        for path in qa_root.rglob("*"):
            if path.is_file():
                archive.writestr(str(path), path.read_bytes())
        archive.writestr("source_snapshot.json", json.dumps({"commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(), "hashes": {f: digest(f) for f in files}}))
    payload = archive_data.getvalue()
    sha = hashlib.sha256(payload).hexdigest()
    encoded = base64.b64encode(payload).decode()
    notebook = f'''# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # R36 capacity check
# Private LUMIERE only. No encoder-unseen or SAILOR scoring.
# Exactly 20 epochs of seed42 A and B. Final epochs only.

# %%
import torch
assert torch.cuda.is_available(), "GPU required"
assert "T4" in torch.cuda.get_device_name(0), "Tesla T4 required"
print("device", torch.cuda.get_device_name(0))

# %%
!pip uninstall --quiet -y torchao torchvision
!pip install --quiet "monai==1.6.0" "nibabel==5.4.2" "scikit-learn==1.9.1"

# %%
import base64, hashlib, io, json, os, subprocess, time, zipfile
from pathlib import Path
payload = base64.b64decode("{encoded}")
assert hashlib.sha256(payload).hexdigest() == "{sha}"
root = Path("/kaggle/working/r36-run")
root.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    assert all(not Path(p).is_absolute() and ".." not in Path(p).parts for p in archive.namelist())
    archive.extractall(root)
os.chdir(root)
snapshot = json.loads(Path("source_snapshot.json").read_text())
for filename, expected in snapshot["hashes"].items():
    assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected
subprocess.run(["git", "init"], check=True)
Path(".gitignore").write_text("outputs/\\nresults/\\n__pycache__/\\n*.pyc\\n")
subprocess.run(["git", "add", "."], check=True)
subprocess.run(["git", "-c", "user.name=R36 experiment", "-c", "user.email=r36@example.invalid",
                "commit", "--no-gpg-sign", "-m", "install private R36 snapshot"], check=True)
with Path("environment.txt").open("w") as stream:
    subprocess.run(["pip", "freeze"], stdout=stream, check=True)
subprocess.run(["python", "tests/test_delta_representation.py"], check=True)
qa = Path("{qa}")
contract = json.loads((qa / "r36_contract.json").read_text())
expected_metadata = contract["r35_prepared_sha256"]
prepared_candidates = [p.parent for p in Path("/kaggle/input").glob("**/prepared/prepared.json")
                       if hashlib.sha256(p.read_bytes()).hexdigest() == expected_metadata]
assert len(prepared_candidates) == 1, "Expected exactly one hash-matched R35 prepared input"
original_prepared = prepared_candidates[0]
prepared = Path("inputs/prepared")
prepared.mkdir(parents=True, exist_ok=True)
Path(prepared / "prepared.json").write_bytes((qa / "prepared.json").read_bytes())
assert hashlib.sha256((prepared / "prepared.json").read_bytes()).hexdigest() == contract["prepared_sha256"]
local_metadata = json.loads((prepared / "prepared.json").read_text())
original_metadata = json.loads((original_prepared / "prepared.json").read_text())
assert {{**local_metadata, "inventory_sha256": original_metadata["inventory_sha256"]}} == original_metadata
for visit in local_metadata["visits"]:
    (prepared / visit["tensor_file"]).symlink_to(original_prepared / visit["tensor_file"])
checkpoint_candidates = [p for p in Path("/kaggle/input").glob("**/seed42/state/best.pt")
                         if hashlib.sha256(p.read_bytes()).hexdigest() == contract["initial_checkpoint_sha256"]["42"]]
assert len(checkpoint_candidates) == 1, "Expected exactly one seed42 R35 state checkpoint"
checkpoint = checkpoint_candidates[0]
print("prepared inventory hash", expected_metadata)
print("initial checkpoint hash", contract["initial_checkpoint_sha256"]["42"])

# %%
started = time.monotonic()
capacity = {{"status": "running", "runs": {{}}, "source_commit": snapshot["commit"], "snapshot_sha256": "{sha}"}}
Path("results").mkdir(exist_ok=True)
try:
    for arm in ("A", "B"):
        remaining = 8.5 - (time.monotonic() - started) / 3600
        if remaining <= 0:
            capacity["status"] = "incomplete_runtime_ceiling"
            break
        subprocess.run(["python", "-u", "-m", "src.harness.analysis.delta_representation", "train",
                        "--prepared", str(prepared), "--checkpoint", str(checkpoint),
                        "--contract", str(qa / "r36_contract.json"), "--audit", str(qa / "audit.json"),
                        "--seed", "42", "--arm", arm, "--hours", str(remaining),
                        "--out", f"results/seed42/{{arm}}"], check=True)
        receipt = json.loads(Path(f"results/seed42/{{arm}}/training.json").read_text())
        capacity["runs"][arm] = {{"status": receipt["status"], "epochs": len(receipt["history"]), "seconds": receipt["seconds"]}}
        if receipt["status"] != "complete":
            capacity["status"] = "incomplete_runtime_ceiling"
            break
    else:
        capacity["status"] = "capacity_training_complete"
finally:
    capacity["seconds"] = time.monotonic() - started
    Path("results/capacity.json").write_text(json.dumps(capacity, indent=2))
    subprocess.run(["tar", "-czf", "/kaggle/working/r36-capacity.tar.gz", "results", "source_snapshot.json", "environment.txt"], check=True)
print(json.dumps(capacity, indent=2))
'''
    directory = Path(out); directory.mkdir(parents=True, exist_ok=True)
    (directory / "capacity_leg.py").write_text(notebook)
    save_json(directory / "kernel-metadata.json", {
        "id": "nairadithya/r36-delta-capacity", "title": "R36 delta objective capacity",
        "code_file": "capacity_leg.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [], "kernel_sources": ["nairadithya/r35-representation-retry", "nairadithya/r35-representation-comparison-fixed"],
        "competition_sources": [], "model_sources": []})
    receipt = {"archive_sha256": sha, "source_files": list(files), "private": True, "cohort": "LUMIERE",
               "arms": ["42/A", "42/B"], "epochs": 20, "max_session_hours": 9,
               "query_evaluation": False, "notebook_source": str(directory / "capacity_leg.py")}
    save_json(directory / "package_receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True); parser.add_argument("--qa", required=True); parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.prepared, args.qa, args.out), sort_keys=True))
