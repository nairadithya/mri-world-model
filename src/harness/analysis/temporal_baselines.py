"""Matched past-only temporal baselines for the prospective RANO task.

All rows are the same ``t -> t+1`` rows used by forecast-baselines. Sequence
features are restricted to visits through ``t``. This is intentionally a small
baseline suite: mean-history pooling, a last-visit MLP, and a GRU, plus
scan-count and order/truncation controls.
"""
from __future__ import annotations

import argparse
import json
import os
import random

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

from src.data.eval_protocol import fold_patients, load_protocol
from src.harness.eval.aggregate import patient_bootstrap, percentile_ci
from src.harness.eval.metrics import macro_f1


class Row:
    __slots__ = ("pid", "seq", "y")

    def __init__(self, pid, seq, y):
        self.pid, self.seq, self.y = pid, seq, int(y)


def build_rows(cache, pids):
    out = []
    for pid in pids:
        p = cache["patients"].get(pid)
        if p is None or p.get("fused") is None:
            continue
        seq = torch.as_tensor(p["fused"]).float()
        labels = torch.as_tensor(p.get("response_labels", p["labels"])).long()
        has = torch.as_tensor(p.get("has_img", torch.ones(len(labels)))).bool()
        n = min(len(seq), len(labels), len(has))
        for t in range(n - 1):
            if not has[t] or not has[t + 1] or labels[t + 1] < 0:
                continue
            out.append(Row(pid, seq[:t + 1].clone(), labels[t + 1]))
    return out


def normalizer(rows):
    x = torch.cat([r.seq for r in rows])
    return x.mean(0), x.std(0).clamp_min(1e-6)


def _weight(rows):
    y = torch.tensor([r.y for r in rows])
    c = torch.bincount(y, minlength=4).float().clamp_min(1)
    return c.sum() / (4 * c)


def fit_mlp(rows, mu, sd, seed=42, steps=500):
    torch.manual_seed(seed)
    x = torch.stack([(r.seq[-1] - mu) / sd for r in rows])
    y = torch.tensor([r.y for r in rows])
    net = nn.Sequential(nn.Linear(x.shape[1], 128), nn.GELU(),
                        nn.Linear(128, 4))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randperm(len(x), generator=g)
        opt.zero_grad()
        loss = nn.functional.cross_entropy(net(x[idx]), y[idx],
                                           weight=_weight(rows))
        loss.backward()
        opt.step()
    return net


def fit_gru(rows, mu, sd, seed=42, steps=300, shuffle=False, truncate=False):
    torch.manual_seed(seed)
    seqs = []
    for r in rows:
        s = (r.seq - mu) / sd
        if truncate:
            s = s[-2:]
        if shuffle and len(s) > 1:
            # Keep the current visit fixed; only historical order is shuffled.
            g = torch.Generator().manual_seed(seed + len(seqs) * 7919)
            hist = s[:-1][torch.randperm(len(s) - 1, generator=g)]
            s = torch.cat([hist, s[-1:]])
        seqs.append(s)
    x = pad_sequence(seqs, batch_first=True)
    lengths = torch.tensor([len(s) for s in seqs])
    y = torch.tensor([r.y for r in rows])
    model = nn.GRU(x.shape[-1], 64, batch_first=True)
    head = nn.Linear(64, 4)
    opt = torch.optim.Adam([*model.parameters(), *head.parameters()], lr=3e-3)
    weight = _weight(rows)
    for _ in range(steps):
        opt.zero_grad()
        packed = pack_padded_sequence(x, lengths, batch_first=True,
                                       enforce_sorted=False)
        _, h = model(packed)
        loss = nn.functional.cross_entropy(head(h[-1]), y, weight=weight)
        loss.backward()
        opt.step()
    return model, head


def fit_mean(rows, mu, sd):
    # A class-weighted linear head on mean history, preserving the same
    # classifier family as the frozen current-image baseline.
    from src.harness.train.readout import fit_linear
    x = torch.stack([((r.seq - mu) / sd).mean(0) for r in rows])
    y = torch.tensor([r.y for r in rows])
    return fit_linear(x, y, hidden=0, seed=42)


def predict(method, model, rows, mu, sd):
    if method == "mean_history":
        x = torch.stack([((r.seq - mu) / sd).mean(0) for r in rows])
        with torch.no_grad():
            return model(x).argmax(1)
    if method == "last_mlp":
        x = torch.stack([(r.seq[-1] - mu) / sd for r in rows])
        with torch.no_grad():
            return model(x).argmax(1)
    seqs = [(r.seq[-2:] if method == "truncated_gru" else r.seq).clone()
            for r in rows]
    if method == "shuffle_gru":
        for i, s in enumerate(seqs):
            if len(s) > 1:
                g = torch.Generator().manual_seed(42 + i * 7919)
                seqs[i] = torch.cat([s[:-1][torch.randperm(len(s)-1,
                                      generator=g)], s[-1:]])
    seqs = [(s - mu) / sd for s in seqs]
    x = pad_sequence(seqs, batch_first=True)
    lengths = torch.tensor([len(s) for s in seqs])
    with torch.no_grad():
        packed = pack_padded_sequence(x, lengths, batch_first=True,
                                       enforce_sorted=False)
        _, h = model[0](packed)
        return model[1](h[-1]).argmax(1)


def evaluate(train, test, method, seed=42, boot=1000, steps=300):
    mu, sd = normalizer(train)
    if method == "mean_history":
        model = fit_mean(train, mu, sd)
    elif method == "last_mlp":
        model = fit_mlp(train, mu, sd, seed=seed, steps=steps)
    else:
        model = fit_gru(train, mu, sd, seed=seed, steps=steps,
                        shuffle=method == "shuffle_gru",
                        truncate=method == "truncated_gru")
    by_patient = {}
    for pid in sorted({r.pid for r in test}):
        rr = [r for r in test if r.pid == pid]
        pred = predict(method, model, rr, mu, sd)
        y = torch.tensor([r.y for r in rr])
        by_patient[pid] = (pred, y)
    pooled = macro_f1(torch.cat([v[0] for v in by_patient.values()]),
                      torch.cat([v[1] for v in by_patient.values()]))
    uniform = sum(macro_f1(p, y) for p, y in by_patient.values()) / len(by_patient)
    vals, _ = patient_bootstrap(by_patient, None, boot=boot, seed=seed)
    return {"pooled_macro_f1": pooled, "patient_uniform_macro_f1": uniform,
            "ci": list(percentile_ci(vals)), "n_pat": len(by_patient),
            "n_rows": sum(len(y) for _, y in by_patient.values())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="checkpoints/p0_interface_cache.pt")
    ap.add_argument("--protocol", default="info/eval_folds.json")
    ap.add_argument("--out", default="outputs/p1_temporal_baselines.json")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args(argv)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    proto = load_protocol(args.protocol)
    pids = sorted(set(proto["encoder_train"]) | set(proto["encoder_unseen"]))
    data = build_rows(cache, pids)
    methods = ("mean_history", "last_mlp", "gru", "shuffle_gru", "truncated_gru")
    output = {"cache": args.cache, "methods": {}}
    print(f"temporal rows={len(data)} patients={len(set(r.pid for r in data))}")
    for method in methods:
        folds = []
        for i in range(proto["k"]):
            te_p = set(fold_patients(proto, i))
            tr_p = set(proto["encoder_unseen"]) - te_p
            tr = [r for r in data if r.pid in tr_p]
            te = [r for r in data if r.pid in te_p]
            result = evaluate(tr, te, method, boot=args.boot, steps=args.steps)
            folds.append(result)
        output["methods"][method] = {"folds": folds,
            "pooled_macro_f1": sum(x["pooled_macro_f1"] for x in folds) / len(folds)}
        print(method, output["methods"][method])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
        f.write("\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
