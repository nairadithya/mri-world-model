"""Replay R39 saved predictions and independently reconstruct aggregate scores."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    rows = dict(np.load(args.inputs, allow_pickle=False))
    saved = dict(np.load(args.result / "predictions.npz", allow_pickle=False))
    report = json.loads((args.result / "metrics.json").read_text())
    np.testing.assert_array_equal(saved["truth"], rows["truth"])
    np.testing.assert_array_equal(saved["groups"], rows["groups"])
    assert set(saved["fold"].tolist()) == set(range(5))
    for group in np.unique(rows["groups"]):
        assert len(set(saved["fold"][rows["groups"] == group])) == 1
    reconstructed = {}
    arms = ("L", "LT", "H", "HT", "time_only", "phase_time")
    for arm in (*arms, "persistence", "measured_persistence"):
        if arm in arms:
            phase = np.eye(4)[rows["phase"]]
            if arm == "time_only":
                x = rows["timing"]
            elif arm == "phase_time":
                x = np.c_[rows["timing"], phase]
            else:
                parts = [rows["current"], rows["timing"], rows["has_history"][:, None]]
                if arm in ("H", "HT"):
                    parts.append(rows["current"] - rows["previous"])
                if arm in ("LT", "HT"):
                    parts.extend(
                        [
                            phase,
                            (phase[:, :, None] * rows["current"][:, None, :]).reshape(
                                -1, 12
                            ),
                        ]
                    )
                x = np.concatenate(parts, axis=1)
            base = (
                rows["current"]
                if arm in ("L", "LT", "H", "HT")
                else np.zeros_like(rows["truth"])
            )
        else:
            base = rows["current"] if arm == "persistence" else rows["measured_current"]
        for fold in range(5):
            test = saved["fold"] == fold
            model = dict(
                np.load(args.result / f"fold{fold}-{arm}.npz", allow_pickle=False)
            )
            expected = base[test]
            if arm in arms:
                expected = (
                    expected
                    + ((x[test] - model["center"]) / model["scale"]) @ model["coef"].T
                    + model["intercept"]
                )
                train = ~test
                _, inv, count = np.unique(
                    rows["groups"][train], return_inverse=True, return_counts=True
                )
                weight = 1 / count[inv]
                np.testing.assert_allclose(
                    model["center"],
                    np.average(x[train], axis=0, weights=weight),
                    atol=1e-12,
                )
            np.testing.assert_allclose(saved[f"mean_{arm}"][test], expected, atol=1e-12)
            np.testing.assert_allclose(
                saved[f"sd_{arm}"][test], np.broadcast_to(model["sd"], expected.shape)
            )
        sigma = saved[f"sd_{arm}"]
        mean = saved[f"mean_{arm}"]
        z = (saved["truth"] - mean) / sigma
        crps = (
            sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
        ).mean(1)
        mae = np.abs(saved["truth"] - mean).mean(1)
        patients = np.unique(saved["groups"])
        per_crps = np.array([crps[saved["groups"] == p].mean() for p in patients])
        per_mae = np.array([mae[saved["groups"] == p].mean() for p in patients])
        np.testing.assert_allclose(
            report["scores"][arm]["crps"], per_crps.mean(), atol=1e-12
        )
        np.testing.assert_allclose(
            report["scores"][arm]["mae"], per_mae.mean(), atol=1e-12
        )
        reconstructed[arm] = per_crps
    for contrast, original in report["comparisons"].items():
        candidate, baseline = contrast.split("_vs_")
        delta = reconstructed[candidate] - reconstructed[baseline]
        sample = np.random.default_rng(2026).integers(
            0, len(delta), (10000, len(delta))
        )
        np.testing.assert_allclose(original["difference"], delta.mean(), atol=1e-12)
        np.testing.assert_allclose(
            original["ci99"],
            np.quantile(delta[sample].mean(1), [0.005, 0.995]),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            original["relative_reduction"],
            1 - reconstructed[candidate].mean() / reconstructed[baseline].mean(),
            atol=1e-12,
        )
    print(
        json.dumps(
            {
                "status": "PASS",
                "patients": len(patients),
                "pairs": len(saved["groups"]),
                "replayed_models": 40,
                "scored_arms": 8,
                "recomputed_contrasts": 5,
                "scope": "saved-weight predictions, source-only standardization, CRPS, MAE and paired bootstrap; not independent scientific review",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
