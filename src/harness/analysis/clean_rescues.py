"""Re-run A15/A16/A37 transfer mechanisms against a survival-free champion.

Latent error is a mechanism diagnostic, separate from physical anatomy MAE.
Every adapted SAILOR prediction is scored in the original EMA space. Support
patients alone supply calibration labels or alignment moments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.data.collate import make_collate
from src.data.eval_protocol import load_protocol
from src.data.sailor import SAILORDataset
from src.harness.encode.cache import load as load_cache
from src.harness.encode.encoders import features_for_batch, load_champion
from src.harness.provenance import git_sha, make_provenance
from src.harness.train.field import fit_field, pairs_of
from src.harness.train.latent import load_1step_predictor
from src.model.dynamics_field import integrate

SAILOR_ROOT = "data/sailor/sailor_ebrains_pseud/derivatives/mni2009c-n-s"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encode_sailor(args):
    cfg = yaml.safe_load(Path(args.config).read_text())
    if cfg["data"].get("include_survival", False):
        raise ValueError("clean rescues require survival-free clinical inputs")
    ds = SAILORDataset(args.root, include_survival=False)
    loader = DataLoader(ds, batch_size=1, num_workers=0,
                        collate_fn=make_collate(tuple(cfg["preprocessing"]["target_size"])))
    model, checkpoint = load_champion(cfg, args.champion)
    cache = {"schema_version": 2, "patients": {}, "provenance": make_provenance(
        args.champion, args.config, clinical_schema="survival_free_v1",
        champion_sha256=sha256(args.champion), champion_epoch=checkpoint.get("epoch"),
        champion_val=checkpoint.get("val_loss"), target_space="ema",
        views=["vision", "clinical", "states", "ema_z", "z"], site="SAILOR")}
    with torch.no_grad():
        for batch in loader:
            pid = batch["patient_id"][0]
            item = ds[ds.subjects.index(pid)]
            features, meta = features_for_batch(model, batch, item, ds,
                                                ("vision", "clinical", "states", "ema_z", "z"))
            cache["patients"][pid] = {"split": "sailor", **features, **meta,
                                      "codes": [None] * len(meta["visits"])}
            print(f"clean SAILOR encoded {len(cache['patients'])}/{len(ds)}", flush=True)
    Path(args.sailor_cache).parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, args.sailor_cache)


def checked_cache(path, champion):
    cache = load_cache(path, require_current=True)
    prov = cache["provenance"]
    if prov.get("clinical_schema") != "survival_free_v1":
        raise ValueError(f"{path}: not survival_free_v1")
    if Path(prov["champion"]).resolve() != Path(champion).resolve():
        raise ValueError(f"{path}: champion mismatch")
    if prov.get("champion_sha256") and prov["champion_sha256"] != sha256(champion):
        raise ValueError(f"{path}: checkpoint content changed")
    for pid, row in cache["patients"].items():
        for key in ("z", "states", "clinical", "deltas"):
            if not torch.isfinite(row[key]).all():
                raise ValueError(f"{path}: nonfinite {pid}/{key}")
        if len(row["visits"]) != len(row["z"]) or len(set(row["visits"])) != len(row["visits"]):
            raise ValueError(f"{path}: ambiguous visit identities for {pid}")
    return cache


def pair_tensors(cache, patients, predictor):
    source, target, states, groups, ids = [], [], [], [], []
    for pid in patients:
        p = cache[pid]
        for t in range(len(p["z"]) - 1):
            if not (p["has_img"][t] and p["has_img"][t + 1]):
                continue
            source.append(p["z"][t]); target.append(p["z"][t + 1])
            states.append(p["states"][t]); groups.append(pid)
            ids.append(f"{pid}|{p['visits'][t]}|{p['visits'][t + 1]}|ema-next-visit-v1")
    z0, z1, h = map(torch.stack, (source, target, states))
    with torch.no_grad():
        raw = predictor(h)
    return {"source": z0, "target": z1, "states": h, "raw": raw,
            "groups": np.array(groups), "row_ids": ids}


def error(prediction, target):
    return (1 - F.cosine_similarity(prediction, target, dim=-1)).numpy().astype(float)


def summary(rows, prediction, boot, seed):
    e = error(prediction, rows["target"])
    b = error(rows["source"], rows["target"])
    groups = rows["groups"]; patients = np.unique(groups)
    per = np.array([e[groups == p].mean() for p in patients])
    base = np.array([b[groups == p].mean() for p in patients])
    diff = per - base
    sample = np.random.default_rng(seed).integers(len(patients), size=(boot, len(patients)))
    counts = np.array([(groups == p).sum() for p in patients])
    pooled_diff = (diff[sample] * counts[sample]).sum(1) / counts[sample].sum(1)
    return {"n_patients": len(patients), "n_rows": len(e),
            "pooled_cosine_error": float(e.mean()), "pooled_persistence": float(b.mean()),
            "patient_uniform_cosine_error": float(per.mean()),
            "patient_uniform_persistence": float(base.mean()),
            "relative_patient_uniform_error": float(per.mean() / base.mean()),
            "paired_patient_uniform_difference_95ci": np.quantile(diff[sample].mean(1), [.025, .975]).tolist(),
            "paired_pooled_difference_95ci": np.quantile(pooled_diff, [.025, .975]).tolist(),
            "patient_wins": int((diff < 0).sum())}


def fit_gain(rows):
    # Patient-uniform objective and fixed grid, calibrated only on support.
    delta = rows["raw"] - rows["source"]
    gains = torch.linspace(0, 1, 101)
    costs = []
    for gain in gains:
        e = error(rows["source"] + gain * delta, rows["target"])
        costs.append(np.mean([e[rows["groups"] == p].mean() for p in np.unique(rows["groups"])]))
    return float(gains[int(np.argmin(costs))])


def alignment(source, reference, kind, eps=.01):
    source, reference = source.double(), reference.double()
    ms, mr = source.mean(0), reference.mean(0)
    if kind == "mean":
        a = torch.eye(source.shape[1], dtype=torch.float64)
    elif kind == "diagonal":
        a = torch.diag(((reference.var(0) + eps) / (source.var(0) + eps)).sqrt())
    elif kind == "coral":
        eye = torch.eye(source.shape[1], dtype=torch.float64)
        ds, vs = torch.linalg.eigh(torch.cov(source.T) + eps * eye)
        dr, vr = torch.linalg.eigh(torch.cov(reference.T) + eps * eye)
        a = ((vs * ds.clamp_min(1e-12).rsqrt()) @ vs.T) @ ((vr * dr.clamp_min(1e-12).sqrt()) @ vr.T)
    else:
        raise ValueError(kind)
    return a, ms, mr


def visit_pool(cache, patients, key):
    return torch.cat([p[key][:len(p[key])][p["has_img"][:len(p[key])]]
                      for pid in patients for p in [cache[pid]]])


def aligned_prediction(predictor, query, zmap, hmap):
    az, mz, rz = zmap; ah, mh, rh = hmap
    with torch.no_grad():
        prediction = predictor(((query["states"].double() - mh) @ ah + rh).float()).double()
    # Inverse target map: compare to untouched native EMA targets/persistence.
    return (torch.linalg.solve(az.T, (prediction - rz).T).T + mz).float()


def decomposition(rows):
    predicted = rows["raw"] - rows["source"]; actual = rows["target"] - rows["source"]
    pn, an = predicted.norm(dim=1), actual.norm(dim=1)
    direction = F.cosine_similarity(predicted, actual, dim=1)
    oracle = rows["source"] + predicted * (an / pn.clamp_min(1e-12)).unsqueeze(1)
    return {"mean_predicted_displacement_norm": float(pn.mean()),
            "mean_actual_displacement_norm": float(an.mean()),
            "norm_ratio": float(pn.mean() / an.mean()),
            "mean_direction_cosine": float(direction.mean()),
            "oracle_magnitude_cosine_error": float(error(oracle, rows["target"]).mean()),
            "oracle_status": "uses future target; diagnostic only, excluded from rescue gate"}


def site_probe(lum, sailor, key):
    xs, ys, groups = [], [], []
    for site, cache in enumerate((lum, sailor)):
        for pid, row in cache.items():
            x = row[key][row["has_img"][:len(row[key])]].numpy()
            xs.extend(x); ys.extend([site] * len(x)); groups.extend([f"{site}:{pid}"] * len(x))
    x, y, groups = np.array(xs), np.array(ys), np.array(groups)
    logits = np.empty(len(y))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(x, y, groups):
        net = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=2000))
        net.fit(x[tr], y[tr]); logits[te] = net.decision_function(x[te])
    return {"patient_grouped_oof_auc": float(roc_auc_score(y, logits)),
            "patient_grouped_oof_accuracy": float(((logits > 0) == y).mean()),
            "n_visits": len(y), "n_patients": len(np.unique(groups))}


def run(args):
    lum_payload = checked_cache(args.lumiere_cache, args.champion)
    sailor_payload = checked_cache(args.sailor_cache, args.champion)
    lum, sailor = lum_payload["patients"], sailor_payload["patients"]
    protocol = load_protocol(args.protocol); predictor = load_1step_predictor(args.champion)
    reference = protocol["encoder_train"]
    result = {"schema_version": 1, "git_sha": git_sha(), "seed": args.seed,
              "champion": str(Path(args.champion).resolve()), "champion_sha256": sha256(args.champion),
              "cache_provenance": {"lumiere": lum_payload["provenance"], "sailor": sailor_payload["provenance"]},
              "limitations": ["latent mechanism diagnostic, not an anatomy forecast claim",
                              "SAILOR development adaptation, not zero-shot external validation",
                              "recorded gaps auxiliary; field is a timing-sensitivity reproduction"],
              "lumiere": {}, "sailor": {}, "folds": [], "site_probe": {}}
    saved = {"provenance": {"champion_sha256": result["champion_sha256"], "git_sha": git_sha()}, "folds": {}}
    for split in ("train", "val", "test"):
        ids = sorted(p for p, row in lum.items() if row["split"] == split)
        rows = pair_tensors(lum, ids, predictor)
        result["lumiere"][split] = {"raw": summary(rows, rows["raw"], args.boot, args.seed),
                                     "decomposition": decomposition(rows)}
    result["site_probe"] = {key: site_probe(lum, sailor, key) for key in ("z", "states")}
    ids = sorted(sailor); random.Random(42).shuffle(ids)
    fold_of = {pid: i % 5 for i, pid in enumerate(ids)}
    all_rows = pair_tensors(sailor, sorted(sailor), predictor)
    predictions = {name: torch.empty_like(all_rows["target"]) for name in
                   ("raw", "support_gain", "mean", "diagonal", "coral")}
    if args.field:
        predictions.update({name: torch.empty_like(all_rows["target"]) for name in ("field_cond", "field_uncond")})
    partial_path = Path(args.models + ".partial")
    run_contract = {"champion_sha256": result["champion_sha256"],
                    "row_ids": all_rows["row_ids"], "fold_of": fold_of,
                    "field": args.field, "field_epochs": args.field_epochs,
                    "hidden": args.hidden, "seed": args.seed,
                    "cache_provenance": result["cache_provenance"]}
    if args.resume and partial_path.exists():
        partial = torch.load(partial_path, map_location="cpu", weights_only=False)
        if partial["contract"] != run_contract:
            raise ValueError("partial rescue run has a different input/config contract")
        predictions = partial["predictions"]
        saved = partial["saved"]
        result["folds"] = partial["folds"]
    for fold in range(5):
        if fold in saved["folds"]:
            print(f"clean rescues fold {fold}: restored completed fit", flush=True)
            continue
        support = sorted(p for p in sailor if fold_of[p] != fold)
        query = sorted(p for p in sailor if fold_of[p] == fold)
        tr = pair_tensors(sailor, support, predictor); te = pair_tensors(sailor, query, predictor)
        mask = np.isin(all_rows["groups"], query)
        gain = fit_gain(tr)
        predictions["raw"][mask] = te["raw"]
        predictions["support_gain"][mask] = te["source"] + gain * (te["raw"] - te["source"])
        fold_record = {"fold": fold, "support": support, "query": query, "gain": gain}
        fitted = {"gain": gain, "alignments": {}}
        for kind in ("mean", "diagonal", "coral"):
            zmap = alignment(visit_pool(sailor, support, "z"), visit_pool(lum, reference, "z"), kind)
            hmap = alignment(visit_pool(sailor, support, "states"), visit_pool(lum, reference, "states"), kind)
            predictions[kind][mask] = aligned_prediction(predictor, te, zmap, hmap)
            fitted["alignments"][kind] = {"z": zmap, "states": hmap}
        if args.field:
            tr_field = [r for p in support for r in pairs_of(sailor[p])]
            te_field = [r for p in query for r in pairs_of(sailor[p], one_step=True)]
            for name, phase in (("field_cond", True), ("field_uncond", False)):
                field, tempo, curve = fit_field(tr_field, hidden=args.hidden, max_epochs=args.field_epochs,
                                                seed=args.seed + fold, use_phase=phase)
                with torch.no_grad():
                    z0 = torch.stack([r["z0"] for r in te_field]); h = torch.stack([r["h"] for r in te_field])
                    clin = torch.stack([r["clin"] for r in te_field])
                    ph = torch.tensor([r["phase"] if phase else 0 for r in te_field])
                    out = integrate(field, tempo(clin), z0, h, ph, clin,
                                    torch.tensor([r["gap"] for r in te_field]), steps=3)
                predictions[name][mask] = out
                fitted[name] = {"field": field.state_dict(), "tempo": tempo.state_dict(), "curve": curve}
        saved["folds"][fold] = fitted
        result["folds"].append(fold_record)
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(str(partial_path) + ".tmp")
        torch.save({"contract": run_contract, "predictions": predictions,
                    "saved": saved, "folds": result["folds"]}, temporary)
        temporary.replace(partial_path)
        print(f"clean rescues fold {fold}: support={len(support)} query={len(query)} gain={gain:.2f}", flush=True)
    result["sailor"]["decomposition"] = decomposition(all_rows)
    for name, pred in predictions.items():
        if not torch.isfinite(pred).all():
            raise ValueError(f"nonfinite rescue {name}")
        result["sailor"][name] = summary(all_rows, pred, args.boot, args.seed)
        print(name, result["sailor"][name], flush=True)
    saved["row_ids"] = all_rows["row_ids"]; saved["predictions"] = predictions
    saved["source"] = all_rows["source"]; saved["target"] = all_rows["target"]; saved["groups"] = all_rows["groups"].tolist()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    torch.save(saved, args.models)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--champion", default="checkpoints/p0_clean_jepa/best.pt")
    ap.add_argument("--lumiere-cache", default="checkpoints/p0_probe_cache.pt")
    ap.add_argument("--sailor-cache", default="checkpoints/p0_rescue_sailor_cache.pt")
    ap.add_argument("--root", default=SAILOR_ROOT)
    ap.add_argument("--protocol", default="info/eval_folds.json")
    ap.add_argument("--encode-sailor", action="store_true")
    ap.add_argument("--field", action="store_true")
    ap.add_argument("--resume", action="store_true", help="reuse completed folds after interruption")
    ap.add_argument("--field-epochs", type=int, default=400)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/p0_clean_rescues.json")
    ap.add_argument("--models", default="outputs/p0_clean_rescue_models.pt")
    args = ap.parse_args(argv); torch.set_num_threads(2)
    if args.encode_sailor:
        encode_sailor(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
