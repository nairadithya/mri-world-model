"""Prospective forecast baselines on one immutable row manifest.

All methods use rows ``t -> t+1`` but only features available through ``t``:
majority, last-RANO, a smoothed RANO transition model, clinical-only,
current-image-only, current volume, and past volume trend.  This deliberately
comes before another JEPA training sweep.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from src.data.eval_protocol import fold_patients, load_protocol
from src.harness.data.manifest import LABEL_VERSION
from src.harness.eval.aggregate import patient_bootstrap, percentile_ci
from src.harness.eval.metrics import macro_f1
from src.harness.train.readout import fit_linear


FEATURES = ("clinical", "current_image", "volume", "trend")
METHODS = ("majority", "last_rano", "transition", *FEATURES)


def _tensor(p, key, default=None):
    x = p.get(key, default)
    if x is None:
        return None
    return x if torch.is_tensor(x) else torch.as_tensor(x)


def rows(cache: dict, pids: list[str]) -> dict[str, list[dict]]:
    """Build forecast rows, retaining patient boundaries and source indices."""
    out = {}
    for pid in pids:
        p = cache["patients"].get(pid)
        if p is None:
            continue
        y = _tensor(p, "response_labels", p.get("labels")).long()
        T = len(y)
        has = _tensor(p, "has_img", torch.ones(T, dtype=torch.bool)).bool()
        vision = _tensor(p, "vision")
        clinical = _tensor(p, "clinical")
        volumes = _tensor(p, "volumes")
        labels = y.tolist()
        if vision is None:
            continue
        rr = []
        for t in range(T - 1):
            u = t + 1
            if not bool(has[t]) or not bool(has[u]) or labels[u] < 0:
                continue
            vol = volumes[t].float() if volumes is not None else None
            if vol is not None and t:
                prev = volumes[t - 1].float()
                delta = vol - prev
                past = volumes[:t + 1].float()
                nadir = past.min(dim=0).values
                trend = torch.cat([vol, delta, vol - nadir])
            elif vol is not None:
                trend = torch.cat([vol, torch.zeros_like(vol), torch.zeros_like(vol)])
            else:
                trend = None
            rr.append({
                "source_index": t,
                "target_index": u,
                "y": labels[u],
                "source_label": labels[t],
                "current_image": vision[t].float(),
                "clinical": clinical.float() if clinical is not None else None,
                "volume": vol,
                "trend": trend,
            })
        if rr:
            out[pid] = rr
    return out


def _stack(data, pids, key):
    vals = [r[key] for pid in pids for r in data.get(pid, [])
            if r[key] is not None]
    ys = [r["y"] for pid in pids for r in data.get(pid, [])
          if r[key] is not None]
    if not vals:
        return None, None
    return torch.stack(vals), torch.tensor(ys, dtype=torch.long)


def _labels(data, pids):
    return [(r["source_label"], r["y"])
            for pid in pids for r in data.get(pid, [])]


def _majority(y):
    return int(torch.bincount(y, minlength=4).argmax()) if len(y) else 0


def _transition(data, pids, alpha=1.0):
    counts = torch.full((4, 4), float(alpha))
    for a, b in _labels(data, pids):
        if 0 <= a < 4 and 0 <= b < 4:
            counts[a, b] += 1
    return counts.argmax(1), _majority(torch.tensor([b for _, b in _labels(data, pids)]))


def _fit_feature(data, pids, key, seed=42):
    X, y = _stack(data, pids, key)
    if X is None:
        return None
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
    net = fit_linear((X - mu) / sd, y, hidden=0, seed=seed)
    return net, mu, sd


def _predict(method, model, data, pid):
    rr = data.get(pid, [])
    if not rr:
        return None
    y = torch.tensor([r["y"] for r in rr], dtype=torch.long)
    if method == "majority":
        return torch.full_like(y, int(model))
    if method == "last_rano":
        majority = model
        return torch.tensor([a if 0 <= a < 4 else majority
                             for a in [r["source_label"] for r in rr]], dtype=torch.long)
    if method == "transition":
        table, majority = model
        return torch.tensor([int(table[a]) if 0 <= a < 4 else majority
                             for a in [r["source_label"] for r in rr]], dtype=torch.long)
    net, mu, sd = model
    X = torch.stack([r[method] for r in rr])
    with torch.no_grad():
        return net((X - mu) / sd).argmax(1)


def evaluate(data, train_pids, test_pids, method, seed=42, boot=10000):
    train_y = torch.tensor([r["y"] for p in train_pids for r in data.get(p, [])],
                           dtype=torch.long)
    if method == "majority":
        model = _majority(train_y)
    elif method == "last_rano":
        model = _majority(train_y)
    elif method == "transition":
        model = _transition(data, train_pids)
    else:
        model = _fit_feature(data, train_pids, method, seed=seed)
        if model is None:
            return None
    per = {}
    for pid in test_pids:
        pred = _predict(method, model, data, pid)
        if pred is not None:
            per[pid] = (pred, torch.tensor([r["y"] for r in data[pid]], dtype=torch.long))
    if not per:
        return None
    pred = torch.cat([per[p][0] for p in sorted(per)])
    y = torch.cat([per[p][1] for p in sorted(per)])
    pooled = macro_f1(pred, y)
    uniform = sum(macro_f1(per[p][0], per[p][1]) for p in per) / len(per)
    vals, _ = patient_bootstrap(per, None, boot=boot, seed=seed)
    return {"pooled_macro_f1": pooled, "patient_uniform_macro_f1": uniform,
            "ci": list(percentile_ci(vals)), "n_pat": len(per), "n_rows": len(y)}


def run(data, protocol, method, regime, seed, boot):
    unseen = sorted(protocol["folds"])
    if regime == "within_unseen":
        fold_results = []
        all_per = {}
        for i in range(protocol["k"]):
            te = fold_patients(protocol, i)
            tr = [p for p in unseen if p not in set(te)]
            result = evaluate(data, tr, te, method, seed=seed, boot=boot)
            if result:
                fold_results.append(result)
        if not fold_results:
            return None
        return {"folds": fold_results,
                "pooled_macro_f1": sum(x["pooled_macro_f1"] for x in fold_results) / len(fold_results),
                "n_folds": len(fold_results)}
    tr = list(protocol["encoder_train"])
    te = list(protocol["final"])
    return evaluate(data, tr, te, method, seed=seed, boot=boot)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="checkpoints/p0_interface_cache.pt")
    ap.add_argument("--protocol", default="info/eval_folds.json")
    ap.add_argument("--out", default="results/p1_forecast_baselines.json")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-legacy-clinical", action="store_true")
    args = ap.parse_args(argv)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    protocol = load_protocol(args.protocol)
    data = rows(cache, sorted(set(protocol["encoder_train"]) | set(protocol["encoder_unseen"])))
    all_y = torch.tensor([r["y"] for rr in data.values() for r in rr], dtype=torch.long)
    prevalence = torch.bincount(all_y, minlength=4).float() / max(1, len(all_y))
    print(f"cache={args.cache} schema={cache.get('schema_version', 1)} "
          f"patients={len(data)} label_version={LABEL_VERSION}")
    print(f"forecast prevalence PD/SD/PR/CR={['%.3f' % x for x in prevalence.tolist()]}")
    clinical_schema = (cache.get("provenance") or {}).get("clinical_schema")
    if "clinical" in FEATURES and clinical_schema != "survival_free_v1" \
            and not args.allow_legacy_clinical:
        print("clinical: BLOCKED (cache is not marked survival_free_v1; "
              "eventual OS status unknown)")
    output = {"cache": args.cache, "label_version": LABEL_VERSION,
              "prevalence": prevalence.tolist(), "methods": {}, "blocked": {}}
    for method in METHODS:
        if method == "clinical" and clinical_schema != "survival_free_v1" \
                and not args.allow_legacy_clinical:
            output["blocked"][method] = "cache does not prove survival-free clinical schema"
            continue
        output["methods"][method] = {}
        for regime in ("within_unseen", "transfer_final"):
            result = run(data, protocol, method, regime, args.seed, args.boot)
            output["methods"][method][regime] = result
            print(f"{method:>14} {regime:>14}: {result}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
