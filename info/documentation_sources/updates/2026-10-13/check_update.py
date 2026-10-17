"""Check the October 13 documentation update without reading private artifacts."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[4])
args = parser.parse_args()
root = args.root.resolve()
package = root / "info/documentation_sources/updates/2026-10-13"
baseline = json.loads((package / "baseline.json").read_text())
coverage = json.loads((package / "coverage.json").read_text())["records"]
metrics = json.loads((root / "info/r36_phase1_metrics.json").read_text())
experiments = (root / "info/EXPERIMENTS.md").read_text()
completed = experiments.split('<a id="r36-completed"></a>', 1)[1].split("### Historical execution", 1)[0]
errors = []

for name, expected in baseline["immutable_sha256"].items():
    path = root / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        errors.append(f"Immutable evidence changed or missing: {name}")

origins = [row["original_path"] for row in coverage]
expected_origins = [row["original_path"] for row in baseline["sources"]]
if sorted(origins) != sorted(expected_origins):
    errors.append("Update coverage must name every captured source exactly once")
captured = {row["original_path"]: row for row in baseline["sources"]}
for row in coverage:
    expected = captured.get(row["original_path"])
    if expected is None:
        errors.append(f"Uncaptured source disposition: {row['original_path']}")
        continue
    if row["archive_path"] != expected["archive_path"] or row["sha256"] != expected["sha256"]:
        errors.append(f"Source disposition differs from captured identity: {row['original_path']}")
    path = root / expected["archive_path"]
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
        errors.append(f"Relocated source changed or missing: {expected['archive_path']}")
    if (root / row["original_path"]).exists():
        errors.append(f"Duplicate active narrative remains: {row['original_path']}")
    if not row.get("read_mode") or not row.get("disposition") or not row.get("destinations"):
        errors.append(f"Incomplete source disposition: {row['original_path']}")
    for destination in row.get("destinations", []):
        if not (root / "info" / destination).is_file():
            errors.append(f"Missing canonical destination: {destination}")

local_tables = {
    "lumiere": completed.split("LUMIERE has zero-change occupancy MAE", 1)[1].split("SAILOR has zero-change occupancy MAE", 1)[0],
    "sailor": completed.split("SAILOR has zero-change occupancy MAE", 1)[1].split("#### Source52 ridge companions", 1)[0],
}
for cohort, arms in metrics["cohorts"].items():
    for arm, values in arms.items():
        local = values["local"]
        direct = local.get("direct_change", {}).get("patient_uniform_balanced_mae")
        cells = [arm, f'{values["state"]["patient_uniform_mae"]:.6f}']
        cells.extend(f'{local[k]["patient_uniform_balanced_mae"]:.6f}'
                     for k in ("frozen_probe", "independent_state", "target_only"))
        cells.append("not graded" if direct is None else f"{direct:.6f}")
        expected = "| " + " | ".join(cells) + " |"
        if expected not in local_tables[cohort]:
            errors.append(f"Missing or changed published arm association: {cohort}/{arm}")

for arm in ("A", "B", "C", "D", "R35"):
    cells = [arm]
    for cohort in ("lumiere", "sailor"):
        cells.extend(f'{metrics["cohorts"][cohort][arm]["ridge"][kind]["patient_uniform_mae"]:.6f}'
                     for kind in ("state", "delta"))
    if "| " + " | ".join(cells) + " |" not in completed:
        errors.append(f"Missing or changed published ridge association: {arm}")

contrasts = [("C minus B frozen local occupancy MAE", metrics["comparisons"]["C-B"]),
             ("C minus A direct native log-volume state MAE", metrics["state_C-A"])]
for arm in ("C", "D"):
    contrasts.extend([(f"{arm} minus target-only", metrics["control_comparisons"][arm]["target_only"]),
                      (f"{arm} minus independent-state difference", metrics["control_comparisons"][arm]["independent_state"])])
for label, value in contrasts:
    low, high = value["family_adjusted_ci"]
    expected = f'| {label} | {value["difference"]:+.6f} | [{low:+.6f}, {high:+.6f}] |'
    if expected not in completed:
        errors.append(f"Missing or changed published contrast: {label}")

for checkout in baseline["protected_checkouts"]:
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    head = subprocess.check_output(["git", "-C", checkout["path"], "rev-parse", "HEAD"], env=env).decode().strip()
    status = subprocess.check_output(["git", "-C", checkout["path"], "status", "--porcelain=v1", "--untracked-files=all"], env=env).decode()
    if head != checkout["head"] or status != checkout["status"]:
        errors.append(f"Protected checkout HEAD or status changed: {checkout['path']}")

print(json.dumps({"status": "FAIL" if errors else "PASS", "relocated_sources": len(coverage),
                  "immutable_files": len(baseline["immutable_sha256"]),
                  "published_arm_rows": 10, "published_ridge_rows": 5, "published_contrast_rows": len(contrasts),
                  "protected_checkouts": len(baseline["protected_checkouts"]), "errors": errors,
                  "scope": "File integrity, source dispositions, published metric associations, and protected HEAD/status equality. Not semantic completeness, ignored-output integrity, process state, or scientific acceptance."}, indent=2))
raise SystemExit(bool(errors))
