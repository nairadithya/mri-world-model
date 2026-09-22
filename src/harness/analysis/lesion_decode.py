"""Diagnose whether frozen lesion-crop tokens encode current anatomy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from src.data.eval_protocol import fold_patients, load_protocol
from src.harness.analysis.anatomy_residual import ALPHAS, _patient_mae, _scale_fit
from src.harness.analysis.lesion_fusion import OBSERVATION_VIEWS, lesion_rows
from src.harness.data.anatomy_tasks import rows


def _arrays(feature_cache, manifest, lesion_cache, cohort, patients, view):
    data = rows(feature_cache, cohort=cohort, patients=patients)
    mapping = lesion_rows(feature_cache, manifest, lesion_cache, cohort, view)
    keep = [i for i, row_id in enumerate(data.row_ids) if row_id in mapping]
    x = np.stack([mapping[data.row_ids[i]].numpy() for i in keep])
    y = data.source.numpy()[keep]
    groups = np.asarray(data.patient_ids)[keep]
    return x, y, groups


def _transform_fit(x, components):
    pca = PCA(n_components=min(components, len(x) - 1, x.shape[1]),
              svd_solver="full").fit(x)
    z = pca.transform(x)
    mean, scale = _scale_fit(z)
    return pca, mean, scale


def _fit(x, y, groups, components):
    scores = {alpha: [] for alpha in ALPHAS}
    splitter = GroupKFold(min(3, len(np.unique(groups))))
    for train, valid in splitter.split(x, y, groups):
        pca, mean, scale = _transform_fit(x[train], components)
        za = (pca.transform(x[train]) - mean) / scale
        zb = (pca.transform(x[valid]) - mean) / scale
        for alpha in ALPHAS:
            prediction = Ridge(alpha=alpha, solver="lsqr").fit(
                za, y[train]).predict(zb)
            scores[alpha].append(np.mean(list(_patient_mae(
                y[valid], prediction, groups[valid]).values())))
    alpha = min(ALPHAS, key=lambda value: np.mean(scores[value]))
    pca, mean, scale = _transform_fit(x, components)
    model = Ridge(alpha=alpha, solver="lsqr").fit(
        (pca.transform(x) - mean) / scale, y)
    return model, pca, mean, scale, alpha


def _predict(fitted, x):
    model, pca, mean, scale, _ = fitted
    return model.predict((pca.transform(x) - mean) / scale)


def _metrics(y, prediction, groups):
    per = _patient_mae(y, prediction, groups)
    denominator = float(np.sum((y - y.mean(0)) ** 2))
    return {"patient_uniform_log_volume_mae": float(np.mean(list(per.values()))),
            "log_volume_mae": float(np.abs(y - prediction).mean()),
            "variance_weighted_r2": (1.0 - float(np.sum((y - prediction) ** 2)) /
                                      denominator if denominator > 0 else None),
            "n_rows": len(y), "n_patients": len(per)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="checkpoints/anatomy_features.pt")
    ap.add_argument("--manifest", default="outputs/anatomy_harmonization.json")
    ap.add_argument("--lesion-cache", default="checkpoints/lesion_crop_features.pt")
    ap.add_argument("--protocol", default="info/eval_folds.json")
    ap.add_argument("--out", default="outputs/p2_lesion_decode.json")
    ap.add_argument("--components", type=int, default=16)
    args = ap.parse_args(argv)
    protocol = load_protocol(args.protocol)
    unseen = sorted(protocol["encoder_unseen"])
    output = {"schema_version": 1, "components": args.components,
              "target": "current source-visit log compartment volumes",
              "views": {}}
    for view in OBSERVATION_VIEWS:
        predictions, truths, group_rows, alphas = [], [], [], []
        for fold in range(protocol["k"]):
            test_patients = fold_patients(protocol, fold)
            train_patients = [p for p in unseen if p not in set(test_patients)]
            xa, ya, ga = _arrays(args.features, args.manifest, args.lesion_cache,
                                 "LUMIERE", train_patients, view)
            xb, yb, gb = _arrays(args.features, args.manifest, args.lesion_cache,
                                 "LUMIERE", test_patients, view)
            fitted = _fit(xa, ya, ga, args.components)
            predictions.append(_predict(fitted, xb)); truths.append(yb)
            group_rows.append(gb); alphas.append(fitted[-1])
        y = np.concatenate(truths); prediction = np.concatenate(predictions)
        groups = np.concatenate(group_rows)
        within = _metrics(y, prediction, groups)
        # The mean-only control is fit fold-wise to avoid using held-out means.
        within["selected_alphas"] = alphas
        xa, ya, ga = _arrays(args.features, args.manifest, args.lesion_cache,
                             "LUMIERE", unseen, view)
        xb, yb, gb = _arrays(args.features, args.manifest, args.lesion_cache,
                             "SAILOR", None, view)
        fitted = _fit(xa, ya, ga, args.components)
        transfer = _metrics(yb, _predict(fitted, xb), gb)
        mean_transfer = _metrics(yb, np.broadcast_to(ya.mean(0), yb.shape), gb)
        output["views"][view] = {"within_lumiere": within,
                                  "sailor_transfer": transfer,
                                  "sailor_train_mean_control": mean_transfer,
                                  "final_alpha": fitted[-1]}
    Path(args.out).write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    for view, result in output["views"].items():
        lum = result["within_lumiere"]; sailor = result["sailor_transfer"]
        print(f"{view:30s} LUM MAE={lum['patient_uniform_log_volume_mae']:.4f} "
              f"R2={lum['variance_weighted_r2']:.3f} | "
              f"SAILOR MAE={sailor['patient_uniform_log_volume_mae']:.4f} "
              f"R2={sailor['variance_weighted_r2']:.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
