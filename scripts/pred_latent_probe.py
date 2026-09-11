"""Predicted-latent vs EMA-latent RANO probes across the dynamics gauntlet.

Question: do the SAILOR-era dynamics improvements (gap-conditioned head,
velocity field) preserve clinically relevant signal better than the champion
1-step head? Metric is discriminative, not cosine error (classifiers absorb
the global gain that dominates JEPA error -- K3-3): for each method M, train
a probe RANO_{t+1} classifier on the PREDICTED latent z_hat_M and on the
ACTUAL EMA latent z_true, and compare. Run on LUMIERE (clean labels) and
SAILOR (transfer protocol mirroring A12-b).

Phases:
  --refit : rerun the A15 5-fold field protocol (cond + uncond, identical
            hyperparams/seeds) and save models + held-out 1-step predicted
            vectors -> checkpoints/field_models.pt (CPU ~40-60 min).
            Field weights were never saved by train_field.py (scores only),
            so this phase is mandatory before --probe.
  --probe : build per-pair rows for M in {champ1step, gaphead, field} plus
            z_true; run 4-class linear/MLP probes (probe_rano protocol):
            LUMIERE hero-split + 5-fold patient CV; SAILOR transfer
            (train on LUMIERE-train rows) + SAILOR-fit ceiling + subject CV.
            Also prints per-method cosine error vs persistence on the same
            labelled pairs (backfills the never-reported LUMIERE gap-head
            error table).

Usage:
    python scripts/pred_latent_probe.py --refit
    python scripts/pred_latent_probe.py --probe
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model.dynamics_field import PatientTempo, VelocityField, integrate  # noqa: E402

from horizon_probe import HorizonPredictor  # noqa: E402  (probe head class)
from probe_rano import fit_linear, scores  # noqa: E402
from split_gate import load_predictor  # noqa: E402  (champion 1-step head)
from train_field import fit_field, pairs_of  # noqa: E402

FIELD_MODELS = "checkpoints/field_models.pt"
SAILOR_PROBE_MAP = {1: 0, 2: 1, 3: 2, 5: 3}


# ---------------------------------------------------------------- refit ---

@torch.no_grad()
def _heldout_preds(field, tempo, rows, steps=3):
    """Held-out 1-step rows -> (pred vectors, persist errs, codes)."""
    out = []
    for r in rows:
        tp = tempo(r["clin"].unsqueeze(0))
        pred = integrate(field, tp, r["z0"].unsqueeze(0), r["h"].unsqueeze(0),
                         torch.tensor([r["phase"]]),
                         r["clin"].unsqueeze(0),
                         torch.tensor([r["gap"]]), steps=steps).squeeze(0)
        pe = 1 - (F.normalize(r["z0"], dim=0) *
                  F.normalize(r["tgt"], dim=0)).sum().item()
        out.append((pred.clone(), r["tgt"], pe, r["phase"], r["code"]))
    return out


def refit(cache_path= "checkpoints/field_cache.pt",
          save_path=FIELD_MODELS, hidden=256, dropout=0.2, wd=0.1,
          lr=1e-3, max_epochs=400, steps=3, seed=42):
    """Mirror train_field.run_cv exactly, but persist models + predictions."""
    cache = torch.load(cache_path, map_location="cpu",
                       weights_only=False)["patients"]
    saved = {"folds": {}, "heldout": {"cond": [], "uncond": []}}
    t0 = time.time()
    for f in range(5):
        tr_rows = [r for _, p in cache.items() if p["fold"] != f
                   for r in pairs_of(p)]
        te_pids = [pid for pid, p in cache.items() if p["fold"] == f]
        if not tr_rows or not te_pids:
            continue
        for tag, use_phase in (("cond", True), ("uncond", False)):
            torch.manual_seed(seed + f)
            field, tempo, _ = fit_field(
                tr_rows, hidden=hidden, dropout=dropout, wd=wd, lr=lr,
                max_epochs=max_epochs, seed=seed + f, use_phase=use_phase,
                steps=steps)
            field.eval()
            tempo.eval()
            saved["folds"].setdefault(str(f), {})[tag] = {
                "field": {k: v.clone() for k, v in field.state_dict().items()},
                "tempo": {k: v.clone() for k, v in tempo.state_dict().items()},
            }
            n_pred = 0
            jes = []
            for pid in te_pids:
                p = cache[pid]
                te1 = pairs_of(p, one_step=True)
                # pairs_of(one_step) emits ascending valid t; reconstruct t
                # for the (pid, t) join key used by --probe.
                T = len(p["z"])
                valid_ts = [t for t in range(T - 1)
                            if p["has_img"][t] and p["has_img"][t + 1]]
                assert len(valid_ts) == len(te1), (pid, f)
                preds = _heldout_preds(field, tempo, te1, steps=steps)
                for t, (pred, tgt, pe, phase, code) in zip(valid_ts, preds):
                    saved["heldout"][tag].append({
                        "pid": pid, "t": t, "pred": pred, "tgt": tgt,
                        "persist": pe, "phase": phase, "code": code})
                    jes.append(1 - (F.normalize(pred, dim=0) *
                                    F.normalize(tgt, dim=0)).sum().item())
                    n_pred += 1
            print(f"fold {f} {tag}: held_n={n_pred} "
                  f"jepa={statistics.mean(jes):.4f} "
                  f"({(time.time()-t0)/60:.0f}min elapsed)", flush=True)
    torch.save(saved, save_path)
    print(f"field models + held-out preds -> {save_path}")


# ----------------------------------------------------------------- rows ---

def _lumiere_pairs():
    """Join horizon_cache (z/states/deltas/split) + probe_cache (labels/clinical).

    Returns list of dicts, one per valid 1-step pair with a clean label:
    pid, split, t, state, z0, zt(true next), gap, clin384, label.
    """
    hz = torch.load("checkpoints/horizon_cache.pt", map_location="cpu",
                    weights_only=False)["patients"]
    pc = torch.load("checkpoints/probe_cache.pt", map_location="cpu",
                    weights_only=False)["patients"]
    rows = []
    for pid, p in hz.items():
        if "z" not in p or pid not in pc:
            continue
        q = pc[pid]
        T = len(p["z"])
        for t in range(T - 1):
            if t + 1 >= T or t >= len(p["states"]):
                continue
            if not (p["has_img"][t] and p["has_img"][t + 1]):
                continue
            lab = int(q["labels"][t + 1])
            if lab < 0:
                continue
            rows.append({"pid": pid, "split": p["split"], "t": t,
                         "state": p["states"][t],
                         "z0": p["z"][t], "zt": p["z"][t + 1],
                         "gap": float(p["deltas"][t + 1]),
                         "clin": q["clinical"], "label": lab})
    return rows


def _sailor_pairs():
    """field_cache rows: valid 1-step pairs with RANO codes -> probe labels."""
    fl = torch.load("checkpoints/field_cache.pt", map_location="cpu",
                    weights_only=False)["patients"]
    rows = []
    for pid, p in fl.items():
        T = len(p["z"])
        for t in range(T - 1):
            if t + 1 >= T or t >= len(p["states"]):
                continue
            if not (p["has_img"][t] and p["has_img"][t + 1]):
                continue
            code = p["codes"][t + 1]
            if code not in SAILOR_PROBE_MAP:
                continue
            rows.append({"pid": pid, "fold": p["fold"], "t": t,
                         "state": p["states"][t],
                         "z0": p["z"][t], "zt": p["z"][t + 1],
                         "gap": float(p["deltas"][t + 1]),
                         "clin": p["clinical"],
                         "phase": int(p["treatment"][t]),
                         "label": SAILOR_PROBE_MAP[code]})
    return rows


@torch.no_grad()
def _predict_all(rows, champ, gapnet, gmu, gsd, fields, steps=3):
    """z_hat per method for a row list. fields = list of (field, tempo)
    uncond models (ensemble-mean for cross-cohort rows)."""
    S = torch.stack([r["state"] for r in rows])
    G = torch.tensor([r["gap"] for r in rows])
    Z0 = torch.stack([r["z0"] for r in rows])
    ZT = torch.stack([r["zt"] for r in rows])
    H = S
    C = torch.stack([r["clin"] for r in rows])
    PH = torch.zeros(len(rows), dtype=torch.long)  # uncond channel
    out = {"champ": champ(S)}
    gg = (torch.log1p(G.clamp_min(0)) - gmu) / gsd
    out["gap"] = gapnet(S, gg)
    if fields:
        acc = torch.zeros_like(ZT)
        for field, tempo in fields:
            field.eval()
            tempo.eval()
            tp = tempo(C)
            acc += integrate(field, tp, Z0, H, PH, C, G, steps=steps)
        out["field"] = acc / len(fields)
    errs = {}
    for k, P in out.items():
        e = 1 - (F.normalize(P, dim=-1) *
                 F.normalize(ZT, dim=-1)).sum(dim=-1)
        errs[k] = e
    pe = 1 - (F.normalize(Z0, dim=-1) *
              F.normalize(ZT, dim=-1)).sum(dim=-1)
    return out, errs, pe


def _std_fit_predict(x_tr, y_tr, x_te, hidden):
    """Probe with train-stat standardization folded into the scorer.

    Mandatory for cross-space comparison: feature norms differ wildly
    (EMA z ~28, champion head ~33, gap head ~134 -- cosine training is
    scale-blind), and the fixed-lr unstandardized fit_linear degenerates on
    large-norm spaces (measured: gap ceiling F1 0.05 -> 0.72 on
    standardizing). Train stats only -- no leakage. Dev note: this deviates
    from the A9 probe protocol (unstandardized); A9 compared LayerNorm'd
    spaces of similar scale, so its numbers stand.
    """
    import torch.nn as nn
    mu, sd = x_tr.mean(0), x_tr.std(0).clamp_min(1e-6)
    net = fit_linear((x_tr - mu) / sd, y_tr, hidden=hidden)

    class W(nn.Module):
        def __init__(self):
            super().__init__()
            self.n = net

        def forward(self, t):
            return self.n((t - mu) / sd)

    return W()


def _run_probes(rows_tr, rows_te, feats_tr, feats_te, tag):
    """4-class linear + MLP probes, train rows -> test rows. Returns (acc, F1)."""
    res = {}
    for hidden, nm in ((0, "linear"), (256, "mlp")):
        net = _std_fit_predict(feats_tr, rows_tr, feats_te, hidden)
        acc, f1, rec, _ = scores(net, feats_te, rows_te)
        res[nm] = (acc, f1)
        print(f"  {tag}-{nm}: acc={acc:.4f} macro-F1={f1:.4f} "
              f"recall={ {k: round(v, 3) for k, v in rec.items()} }",
              flush=True)
    return res


def probe():
    champ = load_predictor("checkpoints/champion_0.0081.pt")
    saved = torch.load("checkpoints/probe_head.pt", map_location="cpu",
                       weights_only=False)
    gapnet = HorizonPredictor(hidden=saved.get("hidden", 1024),
                              layers=saved.get("layers", 2))
    gapnet.load_state_dict(saved["net"])
    gapnet.eval()
    gmu, gsd = saved["mu"], saved["sd"]
    fm = torch.load(FIELD_MODELS, map_location="cpu",
                    weights_only=False)["folds"]
    uncond_fields = []
    for f in sorted(fm):
        fld = VelocityField(hidden_dim=256)  # must match --refit width
        fld.load_state_dict(fm[f]["uncond"]["field"])
        tpo = PatientTempo()
        tpo.load_state_dict(fm[f]["uncond"]["tempo"])
        uncond_fields.append((fld, tpo))

    lrows = _lumiere_pairs()
    srows = _sailor_pairs()
    print(f"pairs: LUMIERE {len(lrows)}, SAILOR {len(srows)}", flush=True)

    # SAILOR field predictions: honest held-out (the fold model that held
    # the subject out), joined on (pid, t); cond + uncond variants.
    held = torch.load(FIELD_MODELS, map_location="cpu",
                      weights_only=False)["heldout"]
    field_map = {}
    for tag in ("cond", "uncond"):
        field_map[tag] = {(h["pid"], h["t"]): h["pred"]
                          for h in held[tag]}
    with torch.no_grad():
        # error-context predictions for all rows (batched, fast)
        lpred, lerr, lpe = _predict_all(lrows, champ, gapnet, gmu, gsd,
                                        uncond_fields)
        spred, serr, spe = _predict_all(srows, champ, gapnet, gmu, gsd,
                                        None)  # field via heldout maps
    # SAILOR field error context from the saved held-out predictions
    serr["field_cond"], serr["field_uncond"] = [], []
    for tag, key in (("cond", "field_cond"), ("uncond", "field_uncond")):
        m = field_map[tag]
        errs = []
        for r in srows:
            pr = m[(r["pid"], r["t"])]
            errs.append(1 - (F.normalize(pr, dim=0) *
                             F.normalize(r["zt"], dim=0)).sum().item())
        serr[key] = torch.tensor(errs)

    print("\n== error context on labelled pairs (cos err) ==")
    print(f"{'cohort':>8} {'method':>12} {'n':>6} {'jepa':>8} {'persist':>8}")
    for tag, err, pe, methods in (
            ("LUM", lerr, lpe, ("champ", "gap", "field")),
            ("SAI", serr, spe,
             ("champ", "gap", "field_cond", "field_uncond"))):
        n = len(pe)
        for m in methods:
            print(f"{tag:>8} {m:>12} {n:>6} "
                  f"{err[m].mean():>8.4f} {pe.mean():>8.4f}")

    feats_l = {"true": torch.stack([r["zt"] for r in lrows]),
               "champ": lpred["champ"], "gap": lpred["gap"],
               "field": lpred["field"]}
    y_l = torch.tensor([r["label"] for r in lrows])
    sp_l = [r["split"] for r in lrows]
    feats_s = {"true": torch.stack([r["zt"] for r in srows]),
               "champ": spred["champ"], "gap": spred["gap"],
               "field_cond": torch.stack(
                   [field_map["cond"][(r["pid"], r["t"])] for r in srows]),
               "field_uncond": torch.stack(
                   [field_map["uncond"][(r["pid"], r["t"])] for r in srows])}
    y_s = torch.tensor([r["label"] for r in srows])

    print("\n== LUMIERE hero-split (train patients -> test patients) ==")
    for m, F_ in feats_l.items():
        itr = [i for i, s in enumerate(sp_l) if s == "train"]
        ite = [i for i, s in enumerate(sp_l) if s == "test"]
        print(f"-- feature: {m} (n_tr={len(itr)}, n_te={len(ite)})")
        _run_probes(y_l[itr], y_l[ite], F_[itr], F_[ite], m)

    print("\n== LUMIERE 5-fold patient CV ==")
    import random
    pids = sorted({r["pid"] for r in lrows})
    rng = random.Random(42)
    rng.shuffle(pids)
    folds = [pids[i::5] for i in range(5)]
    for m, F_ in feats_l.items():
        accs, f1s = [], []
        for i in range(5):
            te = set(folds[i])
            itr = [j for j, r in enumerate(lrows) if r["pid"] not in te]
            ite = [j for j, r in enumerate(lrows) if r["pid"] in te]
            net = _std_fit_predict(F_[itr], y_l[itr], F_[ite], 256)
            acc, f1, _, _ = scores(net, F_[ite], y_l[ite])
            accs.append(acc)
            f1s.append(f1)
        print(f"-- {m}-mlp CV: acc={statistics.mean(accs):.4f}±"
              f"{statistics.pstdev(accs):.4f} "
              f"macro-F1={statistics.mean(f1s):.4f}±"
              f"{statistics.pstdev(f1s):.4f}", flush=True)

    print("\n== SAILOR transfer (LUMIERE-train rows -> SAILOR rows) ==")
    for m, ms in (("true", "true"), ("champ", "champ"), ("gap", "gap"),
                  ("field_uncond", "field")):
        Ftr = feats_l[ms][[i for i, s in enumerate(sp_l) if s == "train"]]
        ytr = y_l[[i for i, s in enumerate(sp_l) if s == "train"]]
        print(f"-- feature: {m} (n_tr={len(ytr)}, n_sa={len(y_s)})")
        _run_probes(ytr, y_s, Ftr, feats_s[m], f"{m}-transfer")

    print("\n== SAILOR-fit ceiling (train=test) ==")
    for m, F_ in feats_s.items():
        print(f"-- feature: {m} (n={len(y_s)})")
        _run_probes(y_s, y_s, F_, F_, f"{m}-ceiling")

    print("\n== SAILOR 5-fold subject CV (field folds) ==")
    for m, F_ in feats_s.items():
        accs, f1s = [], []
        for f in range(5):
            itr = [j for j, r in enumerate(srows) if r["fold"] != f]
            ite = [j for j, r in enumerate(srows) if r["fold"] == f]
            if not itr or not ite:
                continue
            net = _std_fit_predict(F_[itr], y_s[itr], F_[ite], 256)
            acc, f1, _, _ = scores(net, F_[ite], y_s[ite])
            accs.append(acc)
            f1s.append(f1)
        print(f"-- {m}-mlp CV: acc={statistics.mean(accs):.4f}±"
              f"{statistics.pstdev(accs):.4f} "
              f"macro-F1={statistics.mean(f1s):.4f}±"
              f"{statistics.pstdev(f1s):.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()
    if args.refit:
        refit()
    if args.probe or not args.refit:
        probe()


if __name__ == "__main__":
    main()
