"""Matched structured + frozen-representation ablations for P2."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

from src.data.eval_protocol import fold_patients, load_protocol
from src.harness.analysis.anatomy_residual import (
    _bootstrap, _predict, _ridge, _subset, _summarize, _train_mlp,
)
from src.harness.data.anatomy_representations import representation_rows
from src.harness.data.anatomy_tasks import AnatomyRows, rows
from src.harness.encode.anatomy import FEATURE_VERSION, load_cache

REPRESENTATIONS = ("vision", "jepa_state")
CHANGE_THRESHOLD = float(np.log(1.25))


def _strata(y, pred, source, groups):
    changing = np.max(np.abs(y - source), axis=1) >= CHANGE_THRESHOLD
    out = {}
    for name, mask in (("stable", ~changing), ("changing", changing)):
        if not mask.any():
            out[name] = {"n_rows": 0, "n_patients": 0,
                         "patient_uniform_log_volume_mae": None}
            continue
        per = [np.abs(y[(groups == p) & mask] - pred[(groups == p) & mask]).mean()
               for p in np.unique(groups[mask])]
        out[name] = {"n_rows": int(mask.sum()),
                     "n_patients": len(np.unique(groups[mask])),
                     "patient_uniform_log_volume_mae": float(np.mean(per))}
    return out


def _append(train: AnatomyRows, test: AnatomyRows, train_rep: dict, test_rep: dict,
            components: int):
    if set(train.row_ids) - set(train_rep) or set(test.row_ids) - set(test_rep):
        raise ValueError("representation cache does not cover the anatomy rows")
    a = np.stack([train_rep[r] for r in train.row_ids])
    b = np.stack([test_rep[r] for r in test.row_ids])
    n = min(components, len(a) - 1, a.shape[1])
    pca = PCA(n_components=n, svd_solver="full").fit(a)
    xa = torch.cat([train.x, torch.from_numpy(pca.transform(a)).float()], dim=1)
    xb = torch.cat([test.x, torch.from_numpy(pca.transform(b)).float()], dim=1)
    return replace(train, x=xa), replace(test, x=xb), pca


def _join_maps(feature_cache, manifest, lum_cache, sailor_cache, view):
    lum = representation_rows(feature_cache=feature_cache, manifest_path=manifest,
                              representation_cache=lum_cache, cohort="LUMIERE", view=view)
    sailor = representation_rows(feature_cache=feature_cache, manifest_path=manifest,
                                 representation_cache=sailor_cache, cohort="SAILOR", view=view)
    return lum, sailor


def main(argv=None):
    ap = argparse.ArgumentParser(description="P2 structured + frozen representation ablation")
    ap.add_argument("--features", default="checkpoints/anatomy_features.pt")
    ap.add_argument("--manifest", default="outputs/anatomy_harmonization.json")
    ap.add_argument("--lumiere-cache", default="checkpoints/p0_interface_cache.pt")
    ap.add_argument("--sailor-cache", default="checkpoints/p0_sailor_cache.pt")
    ap.add_argument("--protocol", default="info/eval_folds.json")
    ap.add_argument("--out", default="outputs/p2_anatomy_fusion.json")
    ap.add_argument("--weights", default="checkpoints/anatomy_fusion_models.pt")
    ap.add_argument("--components", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    feature_cache = load_cache(args.features); protocol = load_protocol(args.protocol)
    unseen = sorted(protocol["encoder_unseen"])
    maps = {view: _join_maps(args.features, args.manifest, args.lumiere_cache,
                             args.sailor_cache, view) for view in REPRESENTATIONS}
    methods = ["structured_ridge", "structured_residual"]
    for view in REPRESENTATIONS:
        methods += [f"{view}_ridge", f"{view}_residual"]
    predictions = {method: [] for method in methods}
    selected_epochs = {m: [] for m in methods if m.endswith("residual")}
    selected_alphas = {m: [] for m in methods if m.endswith("ridge")}
    for i in range(protocol["k"]):
        test_pids = fold_patients(protocol, i)
        train_pids = [p for p in unseen if p not in set(test_pids)]
        base_train = _subset(args.features, "LUMIERE", train_pids)
        base_test = _subset(args.features, "LUMIERE", test_pids)
        datasets = {"structured": (base_train, base_test, None)}
        for view, (lum_map, _) in maps.items():
            datasets[view] = _append(base_train, base_test, lum_map, lum_map,
                                     args.components)
        for name, (train, test, _) in datasets.items():
            ridge = _ridge(train)
            mlp = _train_mlp(train, seed=args.seed + i, hidden=args.hidden,
                             epochs=args.epochs, patience=args.patience)
            for suffix, fitted in (("ridge", ridge), ("residual", mlp)):
                method = f"{name}_{suffix}"
                value = _predict(
                    test, "ridge" if suffix == "ridge" else "residual_mlp", fitted)
                predictions[method].append((*value, test.source.numpy()))
            selected_alphas[f"{name}_ridge"].append(ridge[-1])
            selected_epochs[f"{name}_residual"].append(mlp[-1])
    within = {}
    for method, values in predictions.items():
        y = np.concatenate([v[0] for v in values])
        pred = np.concatenate([v[1] for v in values])
        groups = np.concatenate([v[2] for v in values])
        source = np.concatenate([v[3] for v in values])
        within[method] = _summarize(y, pred, groups)
        within[method]["strata"] = _strata(y, pred, source, groups)
    # Persistence is the common paired reference on exactly the same OOF rows.
    base = _subset(args.features, "LUMIERE", unseen)
    base_prediction = _predict(base, "persistence", None)
    persistence = _summarize(*base_prediction)
    persistence["strata"] = _strata(*base_prediction[:2], base.source.numpy(),
                                    base_prediction[2])
    within = {"persistence": persistence, **within}
    _bootstrap(within, args.boot, args.seed)

    sailor_base = rows(args.features, cohort="SAILOR")
    sailor_prediction = _predict(sailor_base, "persistence", None)
    persistence = _summarize(*sailor_prediction)
    persistence["strata"] = _strata(*sailor_prediction[:2], sailor_base.source.numpy(),
                                    sailor_prediction[2])
    transfer = {"persistence": persistence}
    saved = {"checkpoint_format_version": 2, "feature_version": FEATURE_VERSION,
             "feature_names": feature_cache["feature_names"], "models": {}}
    for name in ("structured", *REPRESENTATIONS):
        train, test, pca = base, sailor_base, None
        if name != "structured":
            lum_map, sailor_map = maps[name]
            train, test, pca = _append(base, sailor_base, lum_map, sailor_map,
                                       args.components)
        ridge = _ridge(train)
        epochs = max(1, int(np.median(selected_epochs[f"{name}_residual"])))
        mlp = _train_mlp(train, seed=args.seed, hidden=args.hidden,
                         epochs=args.epochs, patience=args.patience,
                         fixed_epochs=epochs)
        for suffix, fitted in (("ridge", ridge), ("residual", mlp)):
            method = f"{name}_{suffix}"
            value = _predict(test, "ridge" if suffix == "ridge" else "residual_mlp", fitted)
            transfer[method] = _summarize(*value)
            transfer[method]["strata"] = _strata(
                *value[:2], test.source.numpy(), value[2])
        ridge_model, ridge_mean, ridge_scale, alpha = ridge
        mlp_model, mlp_mean, mlp_scale, _ = mlp
        saved["models"][name] = {
            "pca_mean": None if pca is None else pca.mean_,
            "pca_components": None if pca is None else pca.components_,
            "ridge_coef": ridge_model.coef_, "ridge_intercept": ridge_model.intercept_,
            "ridge_mean": ridge_mean, "ridge_scale": ridge_scale, "ridge_alpha": alpha,
            "residual_model": mlp_model.state_dict(), "residual_mean": mlp_mean,
            "residual_scale": mlp_scale, "residual_epochs": epochs,
        }
    _bootstrap(transfer, args.boot, args.seed + 1)
    output = {"schema_version": 1, "feature_version": FEATURE_VERSION,
              "components": args.components, "within_lumiere": within,
              "sailor_transfer": transfer, "selected_alphas": selected_alphas,
              "selected_epochs": selected_epochs,
              "strata_definition": (
                  "changing iff any compartment absolute delta-log-volume is "
                  f">= log(1.25)={CHANGE_THRESHOLD:.6f}; reporting only, no row selection"),
              "representation_provenance": {
                  "lumiere": args.lumiere_cache, "sailor": args.sailor_cache}}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    weights = Path(args.weights); weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, weights)
    for section, result in (("within_lumiere", within), ("sailor_transfer", transfer)):
        print(section)
        for method, metric in result.items():
            print(f"  {method:22s} MAE={metric['patient_uniform_log_volume_mae']:.4f} "
                  f"rel={metric['persistence_relative_mae']:.3f} "
                  f"CI={metric['mae_difference_95ci']}")
    print(f"wrote {out} and {weights}")


if __name__ == "__main__":
    main()
