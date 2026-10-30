"""Locked R37 neural and voxelwise-ridge fitting; source/validation only."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.model.spatial_dynamics import SpatialDynamics, gaussian_crps
from .spatial_dynamics_baselines import CALIBRATION_GRID, patient_weights
from .spatial_dynamics_data import canonical_hash, save_json

SEEDS = (42, 43, 44)
EPOCHS = 20
RIDGE_ALPHAS = (.1, 1., 10., 100., 1000., 10000.)


def patient_mean(values, rows):
    array = np.asarray(values)
    return np.sum(patient_weights([r["patient_id"] for r in rows])[:, None] * array, axis=0)


def gaussian_scores(mean, scale, truth):
    m, s, y = [torch.as_tensor(x, dtype=torch.float64) for x in np.broadcast_arrays(mean, scale, truth)]
    return gaussian_crps(m, s, y).numpy()


def neural_predictions(model, data, rows, arm, *, overrides=None):
    means, intrinsic, latent_mse = [], [], []
    model.eval()
    with torch.inference_mode():
        for i, row in enumerate(rows):
            override = None if overrides is None else overrides[i]
            pred = model(*data.inputs(row, latest_override=override), mode=arm)
            mean, scale = data.decoder(pred.feature_mean, pred.feature_scale)
            target, _ = data.target(row)
            means.append(mean[0].cpu().numpy())
            intrinsic.append(torch.sqrt((scale[0].square() - .05 ** 2).clamp_min(0)).cpu().numpy())
            latent_mse.append(float((pred.feature_mean - target).square().mean()))
    return {"mean": np.asarray(means), "intrinsic_scale": np.asarray(intrinsic), "latent_mse": np.asarray(latent_mse)}


def calibrate_gaussian(prediction, rows):
    truth = np.asarray([r["truth"] for r in rows])
    grid = [gaussian_scores(prediction["mean"], np.sqrt((k * prediction["intrinsic_scale"]) ** 2 + .05 ** 2), truth)
            for k in CALIBRATION_GRID]
    means = [float(patient_mean(scores, rows).mean()) for scores in grid]
    selected = min(range(3), key=lambda i: (means[i], i))
    return CALIBRATION_GRID[selected], means


def fit_neural(data, out, seed, guard):
    """Fit all matched arms for a seed; no assessment row enters this function."""
    out = Path(out)
    if out.exists():
        raise FileExistsError("refusing to retry or overwrite a neural seed")
    out.mkdir(parents=True)
    train, validation = data.select("source52"), data.select("validation13")
    weights = patient_weights([r["patient_id"] for r in train])
    rng = np.random.default_rng(seed)
    orders = np.stack([rng.choice(len(train), len(train), replace=True, p=weights) for _ in range(EPOCHS)])
    np.save(out / "sample_order.npy", orders)
    torch.manual_seed(seed)
    initial = SpatialDynamics().state_dict()
    initial = {k: v.detach().clone() for k, v in initial.items()}
    initial_hash = canonical_hash({k: v.tolist() for k, v in initial.items()})
    receipts = {}
    for arm in ("H", "L", "G"):
        guard()
        model = SpatialDynamics().to(data.device)
        model.load_state_dict(initial, strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        best, best_epoch, selected, history = float("inf"), None, None, []
        for epoch in range(EPOCHS):
            guard()
            model.train()
            total = 0.
            for index in orders[epoch]:
                row = train[int(index)]
                optimizer.zero_grad(set_to_none=True)
                pred = model(*data.inputs(row), mode=arm)
                target_map, truth = data.target(row)
                mean, scale = data.decoder(pred.feature_mean, pred.feature_scale)
                loss = gaussian_crps(mean, scale, truth).mean() + .1 * (pred.feature_mean - target_map).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite R37 training loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                if not torch.isfinite(norm):
                    raise FloatingPointError("nonfinite R37 gradients")
                optimizer.step()
                total += float(loss.detach())
            prediction = neural_predictions(model, data, validation, arm)
            truth = np.asarray([r["truth"] for r in validation])
            scores = gaussian_scores(prediction["mean"], np.sqrt(prediction["intrinsic_scale"] ** 2 + .05 ** 2), truth)
            value = float(patient_mean(scores, validation).mean())
            if value < best:  # strict inequality retains earliest exact tie
                best, best_epoch = value, epoch + 1
                selected = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save({"state_dict": selected, "status": "partial_not_graded", "selected_epoch": best_epoch}, out / f"{arm}-partial.pt")
            history.append({"epoch": epoch + 1, "training_loss": total / len(train), "validation_crps": value})
            print(f"seed={seed} arm={arm} epoch={epoch+1}/{EPOCHS} val_crps={value:.8f}", flush=True)
            save_json(out / f"{arm}-progress.json", {"history": history, "status": "running", "selected_epoch": best_epoch})
        if selected is None:
            raise RuntimeError("no finite selected checkpoint")
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "epoch": EPOCHS, "optimizer_state_dict": optimizer.state_dict()}, out / f"{arm}-last.pt")
        model.load_state_dict(selected)
        validation_predictions = neural_predictions(model, data, validation, arm)
        calibration, grid_scores = calibrate_gaussian(validation_predictions, validation)
        receipt = {"arm": arm, "seed": seed, "selected_epoch": best_epoch, "selection_validation_crps": best,
                   "calibration": calibration, "validation_grid_crps": grid_scores,
                   "initialization_sha256": initial_hash, "sample_order_sha256": canonical_hash(orders.tolist()),
                   "epochs": EPOCHS, "updates": EPOCHS * len(train), "history": history,
                   "sampling": "patient-uniform weighted replacement; identical saved order for H/L/G; unweighted stochastic loss"}
        torch.save({"state_dict": selected, "receipt": receipt}, out / f"{arm}.pt")
        save_json(out / f"{arm}.json", receipt)
        receipts[arm] = receipt
        del model, optimizer
    save_json(out / "complete.json", receipts)
    return receipts


def ridge_design(data, row):
    maps, available, gaps, mask = data.inputs(row)
    maps = maps[0].permute(2, 3, 4, 0, 1).reshape(-1, 512).double()
    cov = torch.cat((available.reshape(-1), gaps.reshape(-1), mask.reshape(-1))).double()
    return torch.cat((maps, cov[None].expand(len(maps), -1), maps.new_ones((len(maps), 1))), dim=1)


def solve_ridge(gram, cross, alpha):
    """Fold-local weighted centering/scaling, unpenalized intercept."""
    mx, my = gram[:-1, -1], cross[-1]
    covariance = gram[:-1, :-1] - mx[:, None] * mx[None]
    variance = covariance.diag().clamp_min(0)
    constant = variance < 1e-12
    # Moment subtraction can leave roundoff in exactly constant mask/covariate columns.
    covariance[constant, :] = 0
    covariance[:, constant] = 0
    scale = torch.where(constant, torch.ones_like(variance), variance.sqrt())
    normalized = covariance / (scale[:, None] * scale[None])
    rhs = (cross[:-1] - mx[:, None] * my[None]) / scale[:, None]
    rhs[constant] = 0
    coef = torch.linalg.solve(normalized + alpha * torch.eye(len(mx), device=gram.device, dtype=gram.dtype), rhs) / scale[:, None]
    return torch.cat((coef, (my - mx @ coef)[None]), dim=0)


def ridge_volume_means(data, rows, beta, pooled_designs):
    indices = [data.index[r["cutoff_visit_id"]] for r in rows]
    current = data.maps[indices].mean((-3, -2, -1)).double()
    x = torch.stack([pooled_designs[r["row_id"]] for r in rows])
    pooled = (current + x @ beta).float() * data.std + data.mean
    weight, bias = data.payload["decoder_weight"].to(data.device), data.payload["decoder_bias"].to(data.device)
    norm = data.payload["normalization"]
    result = torch.nn.functional.linear(pooled, weight, bias)
    return result.cpu().numpy() * np.asarray(norm["scale"]) + np.asarray(norm["mean"])


def fit_ridge(data, out, guard):
    """Select alpha on grouped source folds; fit final scale on chosen OOF errors."""
    from sklearn.model_selection import GroupKFold
    out = Path(out)
    if out.exists():
        raise FileExistsError("refusing to retry or overwrite ridge fitting")
    out.mkdir(parents=True)
    rows = data.select("source52")
    by_patient = {}
    for row in rows:
        by_patient.setdefault(row["patient_id"], []).append(row)
    moments, pooled = {}, {}
    with torch.inference_mode():
        for patient, patient_rows in sorted(by_patient.items()):
            guard()
            gram = cross = None
            for row in patient_rows:
                x = ridge_design(data, row)
                target, _ = data.target(row)
                latest = data.maps[data.index[row["cutoff_visit_id"]]]
                y = (target[0] - latest).permute(1, 2, 3, 0).reshape(-1, 128).double()
                weight = 1. / (len(patient_rows) * len(x))
                if gram is None:
                    gram = x.T @ x * weight
                    cross = x.T @ y * weight
                else:
                    gram.addmm_(x.T, x, beta=1., alpha=weight)
                    cross.addmm_(x.T, y, beta=1., alpha=weight)
                pooled[row["row_id"]] = x.mean(0)
            moments[patient] = (gram, cross)
        all_rows = data.select("validation13") + data.select("unseen26")
        for row in all_rows:
            pooled[row["row_id"]] = ridge_design(data, row).mean(0)
        def fit(patient_ids, alpha):
            return solve_ridge(torch.stack([moments[p][0] for p in patient_ids]).mean(0),
                               torch.stack([moments[p][1] for p in patient_ids]).mean(0), alpha)
        groups = np.asarray([r["patient_id"] for r in rows])
        truth = np.asarray([r["truth"] for r in rows])
        folds = list(GroupKFold(n_splits=3).split(np.zeros(len(rows)), groups=groups))
        cv_scores, oof_means, fold_receipts = [], [], []
        for alpha in RIDGE_ALPHAS:
            oof = np.full_like(truth, np.nan)
            fold_scores = np.full_like(truth, np.nan)
            details = []
            for fitting, held in folds:
                guard()
                fitting_rows = [rows[i] for i in fitting]
                held_rows = [rows[i] for i in held]
                beta = fit(sorted(set(groups[fitting])), alpha)
                train_mean = ridge_volume_means(data, fitting_rows, beta, pooled)
                # CV predictive variance is fitted on that fold's training responses only.
                rms = np.sqrt(patient_mean((truth[fitting] - train_mean) ** 2, fitting_rows))
                mean = ridge_volume_means(data, held_rows, beta, pooled)
                oof[held] = mean
                fold_scores[held] = gaussian_scores(mean, np.sqrt(rms ** 2 + .05 ** 2), truth[held])
                details.append({"fitting_patients": sorted(set(groups[fitting])), "held_patients": sorted(set(groups[held])),
                                "training_residual_rms": rms.tolist()})
            if not np.isfinite(oof).all():
                raise ValueError("incomplete source OOF predictions")
            cv_scores.append(float(patient_mean(fold_scores, rows).mean()))
            oof_means.append(oof)
            fold_receipts.append(details)
        chosen = min(range(len(RIDGE_ALPHAS)), key=lambda i: (cv_scores[i], i))
        beta = fit(sorted(by_patient), RIDGE_ALPHAS[chosen])
        rms = np.sqrt(patient_mean((truth - oof_means[chosen]) ** 2, rows))
        validation = data.select("validation13")
        pred = {"mean": ridge_volume_means(data, validation, beta, pooled),
                "intrinsic_scale": np.broadcast_to(rms, (len(validation), 3)).copy()}
        calibration, validation_grid = calibrate_gaussian(pred, validation)
    receipt = {"alpha": RIDGE_ALPHAS[chosen], "alpha_grid": list(RIDGE_ALPHAS), "source_cv_crps": cv_scores,
               "calibration": calibration, "validation_grid_crps": validation_grid, "residual_rms": rms.tolist(),
               "objective": "equal-patient mean voxel squared residual plus alpha coefficient penalty; fold-local design centering/scaling; unpenalized intercept",
               "cv_scale": "training-fold residual RMS, never held-fold response fitting; final RMS from chosen-alpha source OOF predictions",
               "folds": fold_receipts[chosen]}
    torch.save({"beta": beta.cpu(), "residual_rms": torch.from_numpy(rms), "receipt": receipt}, out / "ridge.pt")
    np.savez_compressed(out / "source_oof.npz", means=np.stack(oof_means), truth=truth,
                        row_ids=np.asarray([r["row_id"] for r in rows]), patient_ids=groups)
    save_json(out / "complete.json", receipt)
    return receipt


def predict_ridge(data, rows, bundle):
    with torch.inference_mode():
        pooled = {r["row_id"]: ridge_design(data, r).mean(0) for r in rows}
        means = ridge_volume_means(data, rows, bundle["beta"].to(data.device), pooled)
    return {"mean": means, "intrinsic_scale": np.broadcast_to(bundle["residual_rms"].numpy(), means.shape).copy()}
