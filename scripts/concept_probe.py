"""Concept probe (A36): is the SOTA's measurement/growth signal in the latent?

The 0.50 frontier is carried by hand-engineered longitudinal volumetry/radiomics
(Tikhonov 2025), not the CNN. Before committing to a concept-bottleneck
architecture (a representation whose dimensions ARE the RANO measurement
variables), test how much of that signal the frozen champion already holds.

Rows are the A35 cache (`checkpoints/radiomics_features.pt`) so concept values,
labels, and the pair index are byte-identical to the radiomics comparator; the
per-pair latent is indexed by `visit_idx` into `interface_cache.pt`. Two locked
questions (info/eval_folds.json, within-unseen CV + transfer to the reserved
final):

 1. Recoverability -- can the frozen latent (vision / fused / states /
    fused_next) linearly predict the 22-d DeepBraTumIA volumetry+growth concept
    vector? Per-target R2, with the persistence (no-change) floor.
 2. Bottleneck -- does concatenating the *true* concepts to the latent move the
    RANO readout past latent-only and concepts-only? If combined >> both, the
    representation is missing measurement info a concept head would supply; if
    ~= max, it already holds it.

Usage:
    python scripts/concept_probe.py
    python scripts/concept_probe.py --sources states fused
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from probe_rano import _bootstrap, _ci, _pooled, fit_linear, macro_f1  # noqa: E402
from src.data.eval_protocol import assert_disjoint, fold_patients, load_protocol  # noqa: E402

REGIONS = ("nec", "enh", "edema")
CONCEPT_NAMES = []
for _r in REGIONS:
    CONCEPT_NAMES += [f"{_r}_logv_t", f"{_r}_logv_next", f"{_r}_dlog",
                      f"{_r}_ratio", f"{_r}_rel_t", f"{_r}_rel_next"]
CONCEPT_NAMES += ["tot_logv_t", "tot_logv_next", "tot_dlog", "tot_ratio"]
GROWTH_IDX = [i for i, n in enumerate(CONCEPT_NAMES) if n.endswith(("_dlog", "_ratio"))]

RIDGE_LAMS = (0.1, 1.0, 10.0, 100.0, 1000.0)


# --------------------------------------------------------------------------- #
# data assembly (A35 rows x interface latents)
# --------------------------------------------------------------------------- #
def assemble(rf, lat):
    """pid -> list of {src latent, concept, label}; only patients in both caches."""
    out = {}
    for pid, r in rf.items():
        lp = lat.get(pid)
        if lp is None:
            continue
        idx = r["visit_idx"]
        rows = []
        for j in range(len(idx)):
            t = int(idx[j])
            rows.append({
                "vision": lp["vision"][t],
                "fused": lp["fused"][t],
                "fused_next": lp["fused"][t + 1],
                "states": lp["states"][t],
                "concept": r["features"][j],
                "label": int(r["labels"][j]),
            })
        out[pid] = {"split": lp["split"], "rows": rows}
    return out


def _stack_key(data, pids, key):
    vals = [data[p]["rows"][i][key] for p in pids if p in data
            for i in range(len(data[p]["rows"]))]
    if not vals:
        return None
    return torch.stack(vals)


# --------------------------------------------------------------------------- #
# 1. recoverability: latent -> concept regression
# --------------------------------------------------------------------------- #
def fit_ridge_multi(X, Y, lams=RIDGE_LAMS, seed=42):
    """Standardized-X/Y ridge, lambda by inner 5-fold on train rows."""
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
    Z = (X - mu) / sd
    ymu, ysd = Y.mean(0), Y.std(0).clamp_min(1e-6)
    Yz = (Y - ymu) / ysd
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(X), generator=g)
    folds = [idx[i::5] for i in range(5)]
    I = torch.eye(Z.shape[1])

    def _solve(tr, lam):
        return torch.linalg.solve(Z[tr].T @ Z[tr] + lam * I, Z[tr].T @ Yz[tr])

    best_lam, best_err = lams[0], float("inf")
    for lam in lams:
        errs = []
        for i in range(5):
            te = folds[i]
            tr = torch.cat([folds[j] for j in range(5) if j != i])
            W = _solve(tr, lam)
            errs.append(((Yz[te] - Z[te] @ W) ** 2).mean().item())
        m = sum(errs) / 5
        if m < best_err:
            best_lam, best_err = lam, m
    W = _solve(torch.arange(len(X)), best_lam)

    def predict(Xn):
        return ((Xn - mu) / sd) @ W * ysd + ymu

    return predict, best_lam


def r2_per_target(Yt, Yp):
    ss_res = ((Yt - Yp) ** 2).sum(0)
    ss_tot = ((Yt - Yt.mean(0)) ** 2).sum(0)
    return 1.0 - ss_res / ss_tot.clamp_min(1e-9)


def _r2_row(tag, Yt, Yp):
    r2 = r2_per_target(Yt, Yp)
    g = GROWTH_IDX
    ss_res0 = (Yt[:, g] ** 2).sum(0)
    ss_tot0 = ((Yt[:, g] - Yt[:, g].mean(0)) ** 2).sum(0)
    r2_pers = 1.0 - ss_res0 / ss_tot0.clamp_min(1e-9)
    print(f"  {tag:26s} n={len(Yt):4d}  mean R2={r2.mean():+.3f}  "
          f"growth R2={r2[g].mean():+.3f} (persist {r2_pers.mean():+.3f})  "
          f"tot_logv_next R2={r2[CONCEPT_NAMES.index('tot_logv_next')]:+.3f}  "
          f"tot_dlog R2={r2[CONCEPT_NAMES.index('tot_dlog')]:+.3f}")


def recoverability(data, proto, sources):
    unseen = sorted(p for p in proto["folds"] if p in data)
    train = [p for p in proto["encoder_train"] if p in data]
    print("\n== 1. recoverability: latent -> 22-d concept ==")
    print("   [CV] within-unseen out-of-fold; [TR] train 65 -> unseen; "
          "[ORACLE] fused_next (sees visit t+1)")
    for src in sources:
        # [CV] within-unseen
        Yt, Yp = [], []
        for i in range(proto["k"]):
            te = fold_patients(proto, i)
            tr = [p for p in unseen if p in data and p not in set(te)]
            pred, _ = fit_ridge_multi(_stack_key(data, tr, src), _stack_key(data, tr, "concept"))
            Yp.append(pred(_stack_key(data, te, src)))
            Yt.append(_stack_key(data, te, "concept"))
        _r2_row(f"{src} [CV]", torch.cat(Yt), torch.cat(Yp))
        # [TR] train 65 -> unseen
        pred, lam = fit_ridge_multi(_stack_key(data, train, src),
                                    _stack_key(data, train, "concept"))
        _r2_row(f"{src} [TR lam={lam:g}]", _stack_key(data, unseen, "concept"),
                pred(_stack_key(data, unseen, src)))


# --------------------------------------------------------------------------- #
# 2. bottleneck: RANO readout on [latent (+ true concept)]
# --------------------------------------------------------------------------- #
def _rows_clf(data, pids, keys):
    xs, ys = [], []
    for p in pids:
        if p not in data:
            continue
        for r in data[p]["rows"]:
            if r["label"] < 0:
                continue
            x = r[keys[0]]
            for k in keys[1:]:
                x = torch.cat([x.reshape(-1), r[k].reshape(-1)])
            xs.append(x.reshape(-1))
            ys.append(r["label"])
    if not xs:
        return None, None
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


def _fit_readout(X, y, hidden, standardize=True, seed=42):
    if standardize:
        mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
        X = (X - mu) / sd
    else:
        mu = sd = None
    return fit_linear(X, y, hidden=hidden, seed=seed), mu, sd


def _pred(net, mu, sd, X):
    with torch.no_grad():
        return net((X - mu) / sd if mu is not None else X).argmax(1)


def readout(data, proto, configs, hidden, standardize, boot, seed=42):
    """Return {name: {"oof":..., "per":...}} with a printed table."""
    unseen = sorted(p for p in proto["folds"] if p in data)
    final = sorted(p for p in proto["final"] if p in data)
    print(f"\n== 2. RANO readout (hidden={hidden}, standardize={standardize}) ==")
    out = {}
    for name, keys in configs.items():
        oof, per = {}, {}
        for i in range(proto["k"]):
            te = fold_patients(proto, i)
            tr = [p for p in unseen if p not in set(te)]
            Xtr, ytr = _rows_clf(data, tr, keys)
            net, mu, sd = _fit_readout(Xtr, ytr, hidden, standardize, seed)
            for p in te:
                X, y = _rows_clf(data, [p], keys)
                if X is None:
                    continue
                oof[p] = (_pred(net, mu, sd, X), y)
        f1, _, y = _pooled(oof, sorted(oof))
        a_vals, _ = _bootstrap(oof, None, boot=boot, seed=seed)
        lo, hi = _ci(a_vals)
        # transfer -> reserved final
        Xtr, ytr = _rows_clf(data, proto["encoder_train"], keys)
        net, mu, sd = _fit_readout(Xtr, ytr, hidden, standardize, seed)
        for p in final:
            X, y = _rows_clf(data, [p], keys)
            if X is None:
                continue
            per[p] = (_pred(net, mu, sd, X), y)
        tf1, _, _ = _pooled(per, sorted(per))
        t_vals, _ = _bootstrap(per, None, boot=boot, seed=seed)
        tlo, thi = _ci(t_vals)
        n_rows = sum(len(v[1]) for v in oof.values())
        print(f"  {name:18s} CV {f1:.4f} [{lo:.4f},{hi:.4f}] (n_pat={len(oof)}, "
              f"n_rows={n_rows})   transfer {tf1:.4f} [{tlo:.4f},{thi:.4f}] "
              f"(n_pat={len(per)})")
        out[name] = {"oof": oof, "per": per}
    return out


def paired(name_a, name_b, res, regime="CV"):
    a = res[name_a]["oof" if regime == "CV" else "per"]
    b = res[name_b]["oof" if regime == "CV" else "per"]
    _, d_vals = _bootstrap(a, b, boot=10000)
    dlo, dhi = _ci(d_vals)
    fa, _, _ = _pooled(a, sorted(a))
    fb, _, _ = _pooled(b, sorted(b))
    sig = "SIG" if (dhi < 0 or dlo > 0) else "n.s."
    print(f"  [{regime}] {name_a} vs {name_b}: {fa:.4f} vs {fb:.4f}  "
          f"diff {fa - fb:+.4f} [{dlo:+.4f},{dhi:+.4f}] {sig}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radiomics", default="checkpoints/radiomics_features.pt")
    ap.add_argument("--latents", default="checkpoints/interface_cache.pt")
    ap.add_argument("--protocol", default="info/eval_folds.json")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--sources", nargs="*",
                    default=["vision", "fused", "states", "fused_next"])
    args = ap.parse_args()

    proto = load_protocol(args.protocol)
    assert_disjoint(proto)
    rf = torch.load(args.radiomics, map_location="cpu", weights_only=False)["lum"]
    lat_cache = torch.load(args.latents, map_location="cpu", weights_only=False)
    lat = lat_cache["patients"]
    if "provenance" in lat_cache:
        p = lat_cache["provenance"]
        print(f"latent provenance: champion={os.path.basename(p['champion'])} "
              f"git={str(p.get('git_sha'))[:10]} date={p.get('date')}")
    data = assemble(rf, lat)
    n_rows = sum(len(v["rows"]) for v in data.values())
    n_unseen = sum(len(data[p]["rows"]) for p in proto["folds"] if p in data)
    n_train = sum(len(data[p]["rows"]) for p in proto["encoder_train"] if p in data)
    print(f"aligned rows: {n_rows} total (unseen {n_unseen} / encoder_train {n_train}) "
          f"across {len(data)} patients, concept dim {len(CONCEPT_NAMES)}")

    recoverability(data, proto, args.sources)

    reg = {
        "states": ["states"],
        "concept": ["concept"],
        "states+concept": ["states", "concept"],
        "vision": ["vision"],
        "vision+concept": ["vision", "concept"],
    }
    for hidden in (0, 256):
        res = readout(data, proto, reg, hidden=hidden, standardize=True, boot=args.boot)
        print("  paired tests:")
        for reg_name in ("CV", "transfer"):
            paired("states+concept", "states", res, reg_name)
            paired("states+concept", "concept", res, reg_name)
            paired("vision+concept", "vision", res, reg_name)


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
