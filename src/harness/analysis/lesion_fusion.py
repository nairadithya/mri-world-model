"""Matched forecasting ablation for lesion-centred compartment tokens."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data.eval_protocol import fold_patients, load_protocol
from src.harness.analysis.anatomy_fusion import _append, _strata
from src.harness.analysis.anatomy_residual import (
    _bootstrap, _predict, _ridge, _subset, _summarize, _train_mlp,
)
from src.harness.data.anatomy_tasks import rows
from src.harness.encode.anatomy import FEATURE_VERSION, load_cache
from src.harness.encode.lesion import FEATURE_VERSION as LESION_FEATURE_VERSION

OBSERVATION_VIEWS = ("compartments", "contrasts", "compartments_and_contrasts")
TRANSITION_VIEWS = ("transition_last", "transition_history",
                    "observation_and_transitions")
VIEWS = OBSERVATION_VIEWS + TRANSITION_VIEWS


def _observation(item: dict) -> tuple[torch.Tensor, torch.Tensor]:
    present = torch.as_tensor(item["modalities"]).bool()
    if not bool(present.any()):
        raise ValueError("visit has no usable modalities")
    regions = torch.as_tensor(item["regions"]).float()[present].mean(0).flatten()
    contrasts = torch.as_tensor(item["contrasts"]).float()[present].mean(0).flatten()
    return regions, contrasts


def lesion_rows(feature_cache: str, manifest_path: str, lesion_cache: str,
                cohort: str, view: str) -> dict[str, torch.Tensor]:
    if view not in VIEWS:
        raise ValueError(f"unknown lesion view {view!r}")
    anatomy = load_cache(feature_cache)
    manifest = json.loads(Path(manifest_path).read_text())
    encoded = torch.load(lesion_cache, map_location="cpu", weights_only=False)
    if encoded.get("feature_version") != LESION_FEATURE_VERSION:
        raise ValueError("lesion feature cache version mismatch")
    visits = {visit["visit_id"]: visit for visit in manifest["visits"]}
    out = {}
    for row_id, row in anatomy["pairs"].items():
        if row["cohort"] != cohort:
            continue
        source_id = row["source_visit_id"]
        item = encoded["visits"].get(source_id)
        if item is None or source_id not in visits:
            continue
        try:
            regions, contrasts = _observation(item)
        except ValueError:
            continue
        if view == "compartments":
            value = regions
        elif view == "contrasts":
            value = contrasts
        elif view == "compartments_and_contrasts":
            value = torch.cat([regions, contrasts])
        else:
            patient_visits = [candidate for candidate in manifest["visits"]
                              if candidate["cohort"] == cohort
                              and candidate["patient_id"] == row["patient_id"]]
            patient_visits.sort(key=lambda candidate: (
                candidate["day"] if candidate["day"] is not None else float("inf"),
                candidate["visit"]))
            source_index = next(i for i, candidate in enumerate(patient_visits)
                                if candidate["visit_id"] == source_id)
            history = []
            for candidate in patient_visits[:source_index + 1]:
                encoded_visit = encoded["visits"].get(candidate["visit_id"])
                if encoded_visit is not None:
                    try:
                        r, c = _observation(encoded_visit)
                        history.append(torch.cat([r, c]))
                    except ValueError:
                        pass
            current = torch.cat([regions, contrasts])
            transitions = [later - earlier for earlier, later in
                           zip(history[:-1], history[1:])]
            last = transitions[-1] if transitions else torch.zeros_like(current)
            mean = (torch.stack(transitions).mean(0) if transitions
                    else torch.zeros_like(current))
            if view == "transition_last":
                value = last
            elif view == "transition_history":
                value = torch.cat([last, mean])
            else:
                value = torch.cat([current, last, mean])
        if torch.isfinite(value).all():
            out[row_id] = value
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="checkpoints/anatomy_features.pt")
    ap.add_argument("--manifest", default="outputs/anatomy_harmonization.json")
    ap.add_argument("--lesion-cache", default="checkpoints/lesion_crop_features.pt")
    ap.add_argument("--protocol", default="info/eval_folds.json")
    ap.add_argument("--out", default="outputs/p2_lesion_fusion.json")
    ap.add_argument("--weights", default="checkpoints/p2_lesion_fusion_models.pt")
    ap.add_argument("--components", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    feature_cache = load_cache(args.features)
    protocol = load_protocol(args.protocol)
    unseen = sorted(protocol["encoder_unseen"])
    maps = {view: {
        cohort: lesion_rows(args.features, args.manifest, args.lesion_cache,
                            cohort, view)
        for cohort in ("LUMIERE", "SAILOR")} for view in VIEWS}
    methods = ["structured_ridge", "structured_residual"]
    for view in VIEWS:
        methods += [f"{view}_ridge", f"{view}_residual"]
    predictions = {method: [] for method in methods}
    selected_epochs = {method: [] for method in methods if method.endswith("residual")}
    selected_alphas = {method: [] for method in methods if method.endswith("ridge")}
    for fold in range(protocol["k"]):
        test_patients = fold_patients(protocol, fold)
        train_patients = [p for p in unseen if p not in set(test_patients)]
        base_train = _subset(args.features, "LUMIERE", train_patients)
        base_test = _subset(args.features, "LUMIERE", test_patients)
        datasets = {"structured": (base_train, base_test, None)}
        for view in VIEWS:
            datasets[view] = _append(base_train, base_test,
                                     maps[view]["LUMIERE"],
                                     maps[view]["LUMIERE"], args.components)
        for name, (train, test, _) in datasets.items():
            ridge = _ridge(train)
            mlp = _train_mlp(train, seed=args.seed + fold, hidden=args.hidden,
                             epochs=args.epochs, patience=args.patience)
            for suffix, fitted in (("ridge", ridge), ("residual", mlp)):
                method = f"{name}_{suffix}"
                kind = "ridge" if suffix == "ridge" else "residual_mlp"
                value = _predict(test, kind, fitted)
                predictions[method].append((*value, test.source.numpy()))
            selected_alphas[f"{name}_ridge"].append(ridge[-1])
            selected_epochs[f"{name}_residual"].append(mlp[-1])
    within = {}
    for method, values in predictions.items():
        y = np.concatenate([value[0] for value in values])
        pred = np.concatenate([value[1] for value in values])
        groups = np.concatenate([value[2] for value in values])
        source = np.concatenate([value[3] for value in values])
        within[method] = _summarize(y, pred, groups)
        within[method]["strata"] = _strata(y, pred, source, groups)
    base = _subset(args.features, "LUMIERE", unseen)
    value = _predict(base, "persistence", None)
    persistence = _summarize(*value)
    persistence["strata"] = _strata(*value[:2], base.source.numpy(), value[2])
    within = {"persistence": persistence, **within}
    _bootstrap(within, args.boot, args.seed)

    sailor = rows(args.features, cohort="SAILOR")
    value = _predict(sailor, "persistence", None)
    persistence = _summarize(*value)
    persistence["strata"] = _strata(*value[:2], sailor.source.numpy(), value[2])
    transfer = {"persistence": persistence}
    saved = {"checkpoint_format_version": 2, "feature_version": FEATURE_VERSION,
             "lesion_feature_version": LESION_FEATURE_VERSION, "models": {}}
    for name in ("structured", *VIEWS):
        train, test, pca = base, sailor, None
        if name != "structured":
            train, test, pca = _append(base, sailor, maps[name]["LUMIERE"],
                                       maps[name]["SAILOR"], args.components)
        ridge = _ridge(train)
        epochs = max(1, int(np.median(selected_epochs[f"{name}_residual"])))
        mlp = _train_mlp(train, seed=args.seed, hidden=args.hidden,
                         epochs=args.epochs, patience=args.patience,
                         fixed_epochs=epochs)
        for suffix, fitted in (("ridge", ridge), ("residual", mlp)):
            method = f"{name}_{suffix}"
            kind = "ridge" if suffix == "ridge" else "residual_mlp"
            value = _predict(test, kind, fitted)
            transfer[method] = _summarize(*value)
            transfer[method]["strata"] = _strata(
                *value[:2], test.source.numpy(), value[2])
        ridge_model, ridge_mean, ridge_scale, alpha = ridge
        mlp_model, mlp_mean, mlp_scale, _ = mlp
        saved["models"][name] = {
            "pca_mean": None if pca is None else pca.mean_,
            "pca_components": None if pca is None else pca.components_,
            "ridge_coef": ridge_model.coef_, "ridge_intercept": ridge_model.intercept_,
            "ridge_mean": ridge_mean, "ridge_scale": ridge_scale,
            "ridge_alpha": alpha, "residual_model": mlp_model.state_dict(),
            "residual_mean": mlp_mean, "residual_scale": mlp_scale,
            "residual_epochs": epochs,
        }
    _bootstrap(transfer, args.boot, args.seed + 1)
    output = {"schema_version": 1, "feature_version": FEATURE_VERSION,
              "lesion_feature_version": LESION_FEATURE_VERSION,
              "components": args.components, "within_lumiere": within,
              "sailor_transfer": transfer, "selected_alphas": selected_alphas,
              "selected_epochs": selected_epochs,
              "representation_provenance": args.lesion_cache}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    Path(args.weights).parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, args.weights)
    for section, result in (("within_lumiere", within),
                            ("sailor_transfer", transfer)):
        print(section)
        for method, metric in result.items():
            print(f"  {method:36s} MAE={metric['patient_uniform_log_volume_mae']:.4f} "
                  f"rel={metric['persistence_relative_mae']:.3f} "
                  f"CI={metric['mae_difference_95ci']}")
    print(f"wrote {args.out} and {args.weights}")


if __name__ == "__main__":
    main()
