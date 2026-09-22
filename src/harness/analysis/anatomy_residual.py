"""P2 deterministic residual forecaster over physical lesion features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from src.data.eval_protocol import fold_patients, load_protocol
from src.harness.data.anatomy_tasks import AnatomyRows, rows
from src.harness.encode.anatomy import FEATURE_VERSION, load_cache

ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)


class ResidualMLP(nn.Module):
    """Zero-initialized delta head: its initial forecast is persistence."""
    def __init__(self, n_features: int, hidden: int = 32, n_targets: int = 3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, hidden), nn.GELU(),
                                 nn.Linear(hidden, n_targets))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


def _arrays(data: AnatomyRows):
    return (data.x.numpy().astype(np.float64), data.source.numpy().astype(np.float64),
            data.target.numpy().astype(np.float64), np.asarray(data.patient_ids))


def _scale_fit(x):
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def _patient_mae(y, pred, groups):
    return {str(p): float(np.abs(y[groups == p] - pred[groups == p]).mean())
            for p in np.unique(groups)}


def _ridge(train: AnatomyRows):
    x, _, y, groups = _arrays(train)
    scores = {a: [] for a in ALPHAS}
    for tr, va in GroupKFold(min(3, len(np.unique(groups)))).split(x, y, groups):
        mean, scale = _scale_fit(x[tr])
        for alpha in ALPHAS:
            model = Ridge(alpha=alpha, solver="lsqr").fit((x[tr] - mean) / scale, y[tr])
            pred = model.predict((x[va] - mean) / scale)
            scores[alpha].append(np.mean(list(_patient_mae(y[va], pred, groups[va]).values())))
    alpha = min(ALPHAS, key=lambda a: np.mean(scores[a]))
    mean, scale = _scale_fit(x)
    model = Ridge(alpha=alpha, solver="lsqr").fit((x - mean) / scale, y)
    return model, mean, scale, alpha


def _weights(groups: np.ndarray) -> torch.Tensor:
    counts = {p: int(np.sum(groups == p)) for p in np.unique(groups)}
    value = np.array([1.0 / counts[p] for p in groups], dtype=np.float32)
    value /= value.mean()
    return torch.from_numpy(value)


def _train_mlp(train: AnatomyRows, *, seed: int, hidden: int, epochs: int,
               patience: int, fixed_epochs: int | None = None):
    x, _, y, groups = _arrays(train)
    delta = y - train.source.numpy()
    unique = np.array(sorted(set(groups)))
    val_patients = set(unique[::5]) if fixed_epochs is None else set()
    va = np.array([p in val_patients for p in groups])
    tr = ~va if va.any() else np.ones(len(x), dtype=bool)
    mean, scale = _scale_fit(x[tr])
    xt = torch.from_numpy(((x - mean) / scale).astype(np.float32))
    yt = torch.from_numpy(delta.astype(np.float32))
    weights = _weights(groups)
    torch.manual_seed(seed)
    model = ResidualMLP(x.shape[1], hidden=hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    best, best_state, best_epoch, stale = float("inf"), None, 0, 0
    limit = fixed_epochs or epochs
    for epoch in range(1, limit + 1):
        model.train(); opt.zero_grad()
        per = F.smooth_l1_loss(model(xt[tr]), yt[tr], reduction="none").mean(1)
        loss = (per * weights[tr]).mean(); loss.backward(); opt.step()
        if fixed_epochs is not None:
            continue
        model.eval()
        with torch.no_grad():
            val = F.l1_loss(model(xt[va]), yt[va]).item()
        if val < best - 1e-5:
            best, best_epoch, stale = val, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, mean, scale, (limit if fixed_epochs is not None else best_epoch)


def _predict(test: AnatomyRows, method, fitted):
    x, source, target, groups = _arrays(test)
    if method == "persistence":
        pred = source
    elif method == "ridge":
        model, mean, scale, _ = fitted
        pred = model.predict((x - mean) / scale)
    else:
        model, mean, scale, _ = fitted
        model.eval()
        with torch.no_grad():
            delta = model(torch.from_numpy(((x - mean) / scale).astype(np.float32))).numpy()
        pred = source + delta
    return target, pred, groups


def _summarize(y, pred, groups):
    per = _patient_mae(y, pred, groups)
    return {"log_volume_mae": float(np.abs(y - pred).mean()),
            "patient_uniform_log_volume_mae": float(np.mean(list(per.values()))),
            "signed_bias": float((pred - y).mean()), "n_rows": len(y),
            "n_patients": len(per), "per_patient_mae": per}


def _bootstrap(result, boot, seed):
    base = result["persistence"]["per_patient_mae"]
    pids = np.array(sorted(base)); rng = np.random.default_rng(seed)
    for method, metric in result.items():
        diff = np.array([metric["per_patient_mae"][p] - base[p] for p in pids])
        draws = rng.choice(diff, (boot, len(diff)), replace=True).mean(1)
        metric["mae_difference_vs_persistence"] = float(diff.mean())
        metric["mae_difference_95ci"] = [float(v) for v in np.quantile(draws, [0.025, 0.975])]
        metric["persistence_relative_mae"] = (
            metric["patient_uniform_log_volume_mae"] /
            result["persistence"]["patient_uniform_log_volume_mae"])


def _subset(path, cohort, pids):
    return rows(path, cohort=cohort, patients=list(pids))


def main(argv=None):
    ap = argparse.ArgumentParser(description="evaluate P2 structured residual forecast")
    ap.add_argument("--features", default="checkpoints/anatomy_features.pt")
    ap.add_argument("--protocol", default="info/eval_folds.json")
    ap.add_argument("--out", default="outputs/p2_anatomy_residual.json")
    ap.add_argument("--weights", default="checkpoints/anatomy_residual.pt")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    cache = load_cache(args.features); protocol = load_protocol(args.protocol)
    unseen = sorted(protocol["encoder_unseen"])
    fold_predictions = {m: [] for m in ("persistence", "ridge", "residual_mlp")}
    selected_epochs, selected_alphas = [], []
    for i in range(protocol["k"]):
        test_pids = fold_patients(protocol, i)
        train_pids = [p for p in unseen if p not in set(test_pids)]
        train, test = _subset(args.features, "LUMIERE", train_pids), _subset(
            args.features, "LUMIERE", test_pids)
        ridge = _ridge(train); selected_alphas.append(ridge[-1])
        mlp = _train_mlp(train, seed=args.seed + i, hidden=args.hidden,
                         epochs=args.epochs, patience=args.patience)
        selected_epochs.append(mlp[-1])
        for method, fitted in (("persistence", None), ("ridge", ridge),
                               ("residual_mlp", mlp)):
            fold_predictions[method].append(_predict(test, method, fitted))
    within = {}
    for method, values in fold_predictions.items():
        y = np.concatenate([v[0] for v in values]); pred = np.concatenate([v[1] for v in values])
        groups = np.concatenate([v[2] for v in values]); within[method] = _summarize(y, pred, groups)
    _bootstrap(within, args.boot, args.seed)
    train_all = _subset(args.features, "LUMIERE", unseen)
    sailor = rows(args.features, cohort="SAILOR")
    final_ridge = _ridge(train_all)
    final_epochs = max(1, int(np.median(selected_epochs)))
    final_mlp = _train_mlp(train_all, seed=args.seed, hidden=args.hidden,
                           epochs=args.epochs, patience=args.patience,
                           fixed_epochs=final_epochs)
    transfer = {}
    for method, fitted in (("persistence", None), ("ridge", final_ridge),
                           ("residual_mlp", final_mlp)):
        transfer[method] = _summarize(*_predict(sailor, method, fitted))
    _bootstrap(transfer, args.boot, args.seed + 1)
    output = {"schema_version": 1, "feature_version": FEATURE_VERSION,
              "feature_provenance": cache["provenance"],
              "within_lumiere": within, "sailor_transfer": transfer,
              "selected_alphas": selected_alphas,
              "selected_epochs": selected_epochs, "final_epochs": final_epochs,
              "model": {"hidden": args.hidden, "zero_initialized_residual": True}}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    model, mean, scale, _ = final_mlp
    weight_path = Path(args.weights); weight_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"checkpoint_format_version": 2, "model": model.state_dict(),
                "scaler_mean": mean, "scaler_scale": scale,
                "feature_names": cache["feature_names"],
                "feature_provenance": cache["provenance"],
                "hidden": args.hidden, "epochs": final_epochs}, weight_path)
    for section, result in (("within_lumiere", within), ("sailor_transfer", transfer)):
        print(section)
        for method, metric in result.items():
            print(f"  {method:12s} MAE={metric['patient_uniform_log_volume_mae']:.4f} "
                  f"rel={metric['persistence_relative_mae']:.3f} "
                  f"CI={metric['mae_difference_95ci']}")
    print(f"wrote {out} and {weight_path}")


if __name__ == "__main__":
    main()
