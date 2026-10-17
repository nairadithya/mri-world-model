"""Clean-target gap-conditioned and velocity-head representation rescues.

The supplied target interval is a retrospective timing-sensitivity input.
Image/clinical history remains strictly past-only. This diagnostic does not
establish a prospective fixed-day or anatomical forecasting result.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.harness.analysis.clean_rescues import checked_cache, sha256, summary
from src.harness.provenance import git_sha
from src.harness.train.field import fit_field, pairs_of
from src.harness.train.latent import GapHeadPredictor, load_1step_predictor
from src.model.dynamics_field import integrate


def rows_for(cache, patients):
    rows, groups, ids = [], [], []
    for pid in patients:
        p = cache[pid]
        for t in range(len(p["z"]) - 1):
            for u in range(t + 1, len(p["z"])):
                if not (p["has_img"][t] and p["has_img"][u]): continue
                rows.append({"z0": p["z"][t], "h": p["states"][t], "tgt": p["z"][u],
                    "clin": p["clinical"], "phase": max(int(p["treatment"][t]), 0),
                    "gap": float(p["deltas"][t + 1:u + 1].sum()), "n": u - t})
                groups.append(pid); ids.append(f"{pid}|{p['visits'][t]}|{p['visits'][u]}|ema-horizon-v1")
    return rows, np.array(groups), ids


def tensors(rows):
    return {"source": torch.stack([r["z0"] for r in rows]),
            "target": torch.stack([r["tgt"] for r in rows]),
            "states": torch.stack([r["h"] for r in rows]),
            "gap": torch.tensor([r["gap"] for r in rows]),
            "n": torch.tensor([r["n"] for r in rows])}


def fit_gap(rows, seed, epochs=300, hidden=1024):
    torch.manual_seed(seed)
    x = tensors(rows); loggap = torch.log1p(x["gap"])
    mu, sd = loggap.mean(), loggap.std().clamp_min(1e-6)
    gap = (loggap - mu) / sd; weight = 1. / x["n"].float()
    model = GapHeadPredictor(hidden=hidden); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(seed)
    curve = []
    for epoch in range(epochs):
        model.train(); order = torch.randperm(len(rows), generator=generator); losses = []
        for start in range(0, len(rows), 512):
            idx = order[start:start + 512]
            prediction = model(x["states"][idx], gap[idx])
            error = 1 - F.cosine_similarity(prediction, x["target"][idx], dim=1)
            w = weight[idx]; loss = (error * w).sum() / w.sum()
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss.detach()))
        curve.append(float(np.mean(losses)))
        if (epoch + 1) % 50 == 0:
            print(f"gap-head seed={seed} epoch={epoch + 1}/{epochs} loss={curve[-1]:.6f}", flush=True)
    return model.eval(), mu, sd, curve


def predict_gap(model, mu, sd, rows):
    x = tensors(rows)
    with torch.no_grad():
        return model(x["states"], (torch.log1p(x["gap"]) - mu) / sd)


def predict_field(field, tempo, rows):
    x = tensors(rows); clinical = torch.stack([r["clin"] for r in rows])
    with torch.no_grad():
        return integrate(field, tempo(clinical), x["source"], x["states"],
                         torch.zeros(len(rows), dtype=torch.long), clinical, x["gap"], steps=3)


def report(rows, groups, predictions, boot=10000):
    x = tensors(rows); out = {}
    for horizon in sorted(x["n"].unique().tolist()):
        mask = x["n"] == horizon
        subset = {"source": x["source"][mask], "target": x["target"][mask], "groups": groups[mask.numpy()]}
        out[str(horizon)] = {name: summary(subset, value[mask], boot, 42)
                             for name, value in predictions.items()}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--champion", default="checkpoints/p0_clean_jepa/best.pt")
    ap.add_argument("--lumiere-cache", default="checkpoints/p0_probe_cache.pt")
    ap.add_argument("--sailor-cache", default="checkpoints/p0_rescue_sailor_cache.pt")
    ap.add_argument("--gap-epochs", type=int, default=300)
    ap.add_argument("--field-epochs", type=int, default=400)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--out", default="outputs/p0_head_rescues.json")
    ap.add_argument("--models", default="outputs/p0_head_rescue_models.pt")
    args = ap.parse_args(argv); torch.set_num_threads(2)
    lum = checked_cache(args.lumiere_cache, args.champion)["patients"]
    sailor = checked_cache(args.sailor_cache, args.champion)["patients"]
    train = sorted(p for p in lum if lum[p]["split"] == "train")
    tr, _, _ = rows_for(lum, train)
    gap, mu, sd, curve = fit_gap(tr, 42, args.gap_epochs, args.hidden)
    field, tempo, field_curve = fit_field(tr, hidden=256, max_epochs=args.field_epochs, use_phase=False)
    predictor = load_1step_predictor(args.champion)
    result = {"schema_version": 1, "champion_sha256": sha256(args.champion), "git_sha": git_sha(),
              "input_contract": "history through source + retrospectively supplied target gap",
              "selection": "fixed original training budgets; no test or SAILOR selection",
              "limitations": ["recorded intervals include rounded/estimated values",
                              "latent error is separate from physical anatomy forecasts"], "sections": {}}
    saved = {"provenance": {"champion_sha256": result["champion_sha256"], "git_sha": git_sha()},
             "gap": {"net": gap.state_dict(), "mu": mu, "sd": sd, "hidden": args.hidden, "layers": 2, "curve": curve},
             "field": {"field": field.state_dict(), "tempo": tempo.state_dict(), "curve": field_curve},
             "predictions": {}, "sailor_gap_folds": {}}
    for label, cache, patients in (("lumiere_val", lum, sorted(p for p in lum if lum[p]["split"] == "val")),
                                   ("lumiere_test", lum, sorted(p for p in lum if lum[p]["split"] == "test")),
                                   ("sailor_zero_shot", sailor, sorted(sailor))):
        rows, groups, ids = rows_for(cache, patients)
        with torch.no_grad(): raw = predictor(tensors(rows)["states"])
        predictions = {"champion_1step": raw, "gap_head": predict_gap(gap, mu, sd, rows),
                       "velocity_field": predict_field(field, tempo, rows)}
        result["sections"][label] = report(rows, groups, predictions)
        saved["predictions"][label] = {"row_ids": ids, "groups": groups.tolist(),
                                        "source": tensors(rows)["source"], "target": tensors(rows)["target"], **predictions}
        print(label, result["sections"][label].get("1"), flush=True)
    all_rows, groups, ids = rows_for(sailor, sorted(sailor)); out = torch.empty_like(tensors(all_rows)["target"])
    patients = sorted(sailor); random.Random(42).shuffle(patients)
    fold_of = {p: i % 5 for i, p in enumerate(patients)}
    for fold in range(5):
        support = sorted(p for p in sailor if fold_of[p] != fold); query = sorted(p for p in sailor if fold_of[p] == fold)
        tr, _, _ = rows_for(sailor, support); te, _, _ = rows_for(sailor, query)
        net, norm_mu, norm_sd, history = fit_gap(tr, 42 + fold, args.gap_epochs, args.hidden)
        out[np.isin(groups, query)] = predict_gap(net, norm_mu, norm_sd, te)
        saved["sailor_gap_folds"][fold] = {"net": net.state_dict(), "mu": norm_mu, "sd": norm_sd,
                                           "support": support, "query": query, "curve": history}
    result["sections"]["sailor_gap_refit_oof"] = report(all_rows, groups, {"gap_head": out})
    saved["predictions"]["sailor_gap_refit_oof"] = {"row_ids": ids, "groups": groups.tolist(),
                                                     "prediction": out, "source": tensors(all_rows)["source"],
                                                     "target": tensors(all_rows)["target"]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    torch.save(saved, args.models)


if __name__ == "__main__": main()
