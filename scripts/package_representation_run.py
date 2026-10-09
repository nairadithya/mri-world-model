"""Build a private source-and-data notebook under ignored outputs; never commit it."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile

FILES = ("src/data/representation.py", "src/model/representation.py", "src/model/brainiac.py",
         "src/harness/analysis/representation.py", "src/preprocessing/geometry.py",
         "src/preprocessing/transforms.py", "tests/test_representation.py")


def package(inventory_path: str, out: str) -> dict:
    inventory = json.loads(Path(inventory_path).read_text())
    if (any(not v["patient_id"].startswith("Patient-") for v in inventory["visits"])
            or any(p["cohort"] != "LUMIERE" for p in inventory["pairs"])
            or any(not k.startswith("Patient-") for k in inventory["geometry"]["images"])):
        raise ValueError("private GPU inventory must contain LUMIERE only")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "diff", "HEAD", "--", *FILES], text=True)
    if dirty:
        raise ValueError("commit experiment code before packaging")
    for filename in FILES:
        subprocess.run(["git", "ls-files", "--error-unmatch", filename], check=True, stdout=subprocess.DEVNULL)
    archive_data = io.BytesIO()
    source_hashes = {}
    with zipfile.ZipFile(archive_data, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename in FILES:
            content = Path(filename).read_bytes()
            source_hashes[filename] = hashlib.sha256(content).hexdigest()
            info = zipfile.ZipInfo(filename, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)
        archive.writestr("inventory.json", json.dumps(inventory, sort_keys=True, separators=(",", ":")))
        archive.writestr("source_snapshot.json", json.dumps({"local_commit": commit, "source_hashes": source_hashes}, sort_keys=True))
    payload = archive_data.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
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
# # R35 representation comparison
#
# Private LUMIERE-only state and observed-change experiments.
# All selected checkpoints and per-row predictions remain available.
# SAILOR is evaluated locally, never uploaded.

# %%
import torch
assert torch.cuda.is_available(), "No GPU allocated"
assert "T4" in torch.cuda.get_device_name(0), "Wrong GPU"
print("device", torch.cuda.get_device_name(0))

# %%
!pip uninstall --quiet -y torchao torchvision
!pip install --quiet "monai==1.6.0" "nibabel==5.4.2" "peft==0.20.0" "transformers==4.57.6" "einops==0.8.2" "scikit-learn==1.9.1" "safetensors==0.8.0"

# %%
import base64, hashlib, io, json, os, subprocess, zipfile
from pathlib import Path
snapshot = base64.b64decode("{encoded}")
assert hashlib.sha256(snapshot).hexdigest() == "{digest}"
root = Path("/kaggle/working/representation-run")
root.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(snapshot)) as archive:
    assert all(not Path(name).is_absolute() and ".." not in Path(name).parts for name in archive.namelist())
    archive.extractall(root)
os.chdir(root)
record = json.loads(Path("source_snapshot.json").read_text())
for filename, expected in record["source_hashes"].items():
    assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected
subprocess.run(["git", "init"], check=True)
Path(".gitignore").write_text("prepared/\\nresults/\\n__pycache__/\\n*.pyc\\nenvironment.txt\\n")
subprocess.run(["git", "add", "."], check=True)
subprocess.run(["git", "-c", "user.name=R35 experiment", "-c", "user.email=r35@example.invalid",
                "commit", "--no-gpg-sign", "-m", "install private R35 snapshot"], check=True)
with Path("environment.txt").open("w") as stream:
    subprocess.run(["pip", "freeze"], stdout=stream, check=True)
subprocess.run(["python", "tests/test_representation.py"], check=True)
print("source snapshot", record["local_commit"], "{digest}")

# %%
mri_roots = sorted(Path("/kaggle/input").glob("**/lumiere_preprocessed"))
assert len(mri_roots) == 1, "MRI mount missing or ambiguous"
masks = sorted(Path("/kaggle/input").glob("**/Patient-*/week-*/DeepBraTumIA-segmentation/atlas/segmentation/seg_mask.nii*"))
if not masks:
    archives = sorted(Path("/kaggle/input").glob("**/Imaging.zip"))
    assert len(archives) == 1, "Mask archive missing or ambiguous"
    with zipfile.ZipFile(archives[0]) as archive:
        assert all(not Path(name).is_absolute() and ".." not in Path(name).parts for name in archive.namelist())
        archive.extractall("/kaggle/working/lesion-supervision")
    masks = sorted(Path("/kaggle/working/lesion-supervision").glob("**/Patient-*/week-*/DeepBraTumIA-segmentation/atlas/segmentation/seg_mask.nii*"))
assert masks, "Native masks missing"
mask_root = str(masks[0]).split("/Patient-", 1)[0]
checkpoints = sorted(Path("/kaggle/input").glob("**/p0_clean_best.pt"))
assert len(checkpoints) == 1, "Clean checkpoint mount missing or ambiguous"
subprocess.run(["python", "-u", "-m", "src.harness.analysis.representation", "prepare",
                "--inventory", "inventory.json", "--mri-root", str(mri_roots[0]),
                "--mask-root", mask_root, "--out", "prepared"], check=True)
prepared = json.loads(Path("prepared/prepared.json").read_text())
assert len(prepared["visits"]) == 599 and len(prepared["pairs"]) == 496

# %%
subprocess.run(["python", "-u", "-m", "src.harness.analysis.representation", "smoke",
                "--prepared", "prepared", "--out", "results/real-gpu-smoke.json"], check=True)
subprocess.run(["python", "-u", "-m", "src.harness.analysis.representation", "reference",
                "--prepared", "prepared", "--checkpoint", str(checkpoints[0]),
                "--out", "results/reference"], check=True)
subprocess.run(["python", "-u", "-m", "src.harness.analysis.representation", "train",
                "--prepared", "prepared", "--out", "results/comparison",
                "--epochs", "20", "--hours", "6", "--seeds", "42", "43", "44"], check=True)

# %%
run = json.loads(Path("results/comparison/run.json").read_text())
print("run status", run.get("status", "UNKNOWN"))
print(json.dumps(run["runs"], indent=2))
subprocess.run(["tar", "-czf", "/kaggle/working/r35-results.tar.gz", "results", "source_snapshot.json", "environment.txt", "prepared/prepared.json"], check=True)
'''
    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "representation_leg.py").write_text(notebook)
    metadata = {"id": "nairadithya/r35-representation-comparison-fixed", "title": "R35 representation comparison fixed",
                "code_file": "representation_leg.ipynb", "language": "python", "kernel_type": "notebook",
                "is_private": True, "enable_gpu": True, "enable_internet": True,
                "machine_shape": "NvidiaTeslaT4",
                "dataset_sources": ["nairadithya/preprocessed-mri-data", "nairadithya/lumiere-lesion-supervision"],
                "kernel_sources": ["nairadithya/p0-clean-jepa-retrain"], "competition_sources": [], "model_sources": []}
    (directory / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    receipt = {"source_commit": commit, "archive_sha256": digest, "archive_bytes": len(payload),
               "visits": len(inventory["visits"]), "pairs": len(inventory["pairs"]), "private": True,
               "source_files": list(FILES)}
    print(json.dumps(receipt, sort_keys=True))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--out", default="outputs/private-kernel")
    args = parser.parse_args()
    package(args.inventory, args.out)
