"""Reconcile P1 aggregation without modifying the historical result artifact."""

import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "outputs/p1_anatomy_baselines.json"
    original = json.loads(source.read_text())
    result = {
        "source": str(source.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "correction_date": "2026-11-02",
        "reason": "SAILOR saved relative MAE uses pair-weighted errors; its displayed absolute MAEs use patient-uniform errors. LUMIERE saved ratios already use patient-uniform errors.",
        "units": "log1p(mm3)",
        "cohorts": {},
    }
    for name, key in (("LUMIERE", "within_lumiere"), ("SAILOR", "sailor_transfer")):
        rows = original[key]
        baseline = rows["persistence"]
        corrected = {}
        for arm, row in rows.items():
            patient_mae = row["patient_uniform_log_volume_mae"]
            assert abs(mean(row["per_patient_mae"].values()) - patient_mae) < 1e-12
            patient_ratio = patient_mae / baseline["patient_uniform_log_volume_mae"]
            pair_ratio = row["log_volume_mae"] / baseline["log_volume_mae"]
            saved = row["persistence_relative_mae"]
            expected = patient_ratio if name == "LUMIERE" else pair_ratio
            assert abs(saved - expected) < 1e-12
            corrected[arm] = {
                "patient_uniform_mae": patient_mae,
                "patient_uniform_relative_mae": patient_ratio,
                "pair_weighted_mae": row["log_volume_mae"],
                "pair_weighted_relative_mae": pair_ratio,
                "historical_saved_relative_mae": saved,
                "paired_patient_difference": row["mae_difference_vs_persistence"],
                "paired_patient_difference_ci95": row["mae_difference_95ci"],
            }
        result["cohorts"][name] = corrected
    out = ROOT / "info/p1_anatomy_reconciled.json"
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(out.relative_to(ROOT))


if __name__ == "__main__":
    main()
