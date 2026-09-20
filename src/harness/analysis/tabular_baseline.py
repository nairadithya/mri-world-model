"""Tree-based observed-scan tabular baseline on the radiomics feature cache.

This is a dependency-light CatBoost-family control using sklearn's histogram
GradientBoosting classifier. It is an assessment baseline: radiomics rows
include the observed follow-up scan and realized pair change. It must not be
compared to the past-only forecast rows as if they were the same task.
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from sklearn.ensemble import HistGradientBoostingClassifier

from src.data.eval_protocol import fold_patients, load_protocol
from src.harness.eval.aggregate import patient_bootstrap, percentile_ci
from src.harness.eval.metrics import macro_f1


def rows(cache, pids):
    out = {}
    for pid in pids:
        p = cache.get(pid)
        if p is None:
            continue
        x, y = p["features"].float(), p["labels"].long()
        if len(y):
            out[pid] = (x, y)
    return out


def fit(train, seed):
    x = torch.cat([v[0] for v in train.values()]).numpy()
    y = torch.cat([v[1] for v in train.values()]).numpy()
    counts = torch.bincount(torch.as_tensor(y), minlength=4).float().clamp_min(1)
    weights = counts.sum() / (4 * counts)
    sample_weight = weights[torch.as_tensor(y)].numpy()
    model = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
        l2_regularization=1.0, random_state=seed)
    model.fit(x, y, sample_weight=sample_weight)
    return model


def evaluate(model, test, boot, seed):
    per = {}
    for pid, (x, y) in test.items():
        pred = torch.as_tensor(model.predict(x.numpy()), dtype=torch.long)
        per[pid] = (pred, y)
    pooled = macro_f1(torch.cat([v[0] for v in per.values()]),
                      torch.cat([v[1] for v in per.values()]))
    uniform = sum(macro_f1(p, y) for p, y in per.values()) / len(per)
    vals, _ = patient_bootstrap(per, None, boot=boot, seed=seed)
    return {"pooled_macro_f1": pooled, "patient_uniform_macro_f1": uniform,
            "ci": list(percentile_ci(vals)), "n_pat": len(per),
            "n_rows": sum(len(y) for _, y in per.values())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="checkpoints/p0_radiomics_features.pt")
    ap.add_argument("--protocol", default="info/eval_folds.json")
    ap.add_argument("--out", default="outputs/p1_tabular_baseline.json")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args(argv)
    raw = torch.load(args.cache, map_location="cpu", weights_only=False)
    cache = raw.get("lum", raw)
    proto = load_protocol(args.protocol)
    all_pids = sorted(set(proto["encoder_train"]) |
                      set(proto["encoder_unseen"]) | set(proto["final"]))
    data = rows(cache, all_pids)
    output = {"cache": args.cache, "task": "observed_pair_assessment",
              "methods": {}}
    folds = []
    for i in range(proto["k"]):
        te = set(fold_patients(proto, i))
        tr = {p: data[p] for p in data if p in set(proto["encoder_unseen"]) - te}
        test = {p: data[p] for p in data if p in te}
        result = evaluate(fit(tr, 42), test, args.boot, 42)
        folds.append(result)
    output["methods"]["hist_gradient_boosting"] = {
        "within_unseen_folds": folds,
        "within_unseen_macro_f1": sum(x["pooled_macro_f1"] for x in folds) / len(folds)}
    tr = {p: data[p] for p in data if p in proto["encoder_train"]}
    test = {p: data[p] for p in data if p in proto["final"]}
    output["methods"]["hist_gradient_boosting"]["transfer_final"] = \
        evaluate(fit(tr, 42), test, args.boot, 42)
    for k, v in output["methods"]["hist_gradient_boosting"].items():
        print(k, v)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
        f.write("\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
