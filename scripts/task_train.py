"""Task-train the temporal stack on cached fused tokens (Step 3).

The vision backbone is frozen (no image forwards); only the temporal
transformer + a 4-class RANO head train, on the 65 encoder-train patients.
Early stopping uses the 13 `dev` patients; the 13 `final` patients are
report-only (locked protocol). Gate: beat the frozen-state readout baseline
final macro-F1 0.448 [0.304, 0.505] (A25/A26).

Usage:
    python scripts/task_train.py --mode temporal --epochs 30 --lr 1e-4
    python scripts/task_train.py --mode head          # frozen-state reference
"""
from __future__ import annotations

import argparse
import copy
import os
import random
import statistics
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset import parse_week_to_days
from src.model.jepa_model import JEPAWorldModel

RANO_NAMES = ["PD", "SD", "PR", "CR"]


def macro_f1(pred, y, n_cls=4):
    f1s = []
    for k in range(n_cls):
        tp = int(((pred == k) & (y == k)).sum())
        fp = int(((pred == k) & (y != k)).sum())
        fn = int(((pred != k) & (y == k)).sum())
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f1s.append(2 * p * r / max(1e-9, p + r))
    return sum(f1s) / n_cls


def _ci(samples, lo=2.5, hi=97.5):
    s = sorted(samples)
    n = len(s)
    return s[int(lo / 100 * n)], s[min(n - 1, int(hi / 100 * n))]


def pairs_for(patients, pid):
    """(features (T-1,d), labels_next (T-1,), valid (T-1,)) from the cache."""
    import torch
    p = patients[pid]
    st = p["states"]                       # (T-1, 1152)
    labels = p["labels"]
    n = st.shape[0]
    lab = labels[1:1 + n]
    valid = lab >= 0
    return st, lab, valid


def eval_macro(patients, pids, predict_fn):
    preds, ys, per = [], [], {}
    for pid in pids:
        st, lab, valid = pairs_for(patients, pid)
        if int(valid.sum()) == 0:
            continue
        with torch.no_grad():
            pr = predict_fn(st)[valid]
        yy = lab[valid]
        per[pid] = (pr, yy)
        preds.append(pr)
        ys.append(yy)
    if not preds:
        return 0.0, per
    return macro_f1(torch.cat(preds), torch.cat(ys)), per


def bootstrap(per, boot=10000, seed=42):
    pids = sorted(per)
    rng = random.Random(seed)
    vals = []
    for _ in range(boot):
        samp = [rng.choice(pids) for _ in pids]
        vals.append(macro_f1(torch.cat([per[p][0] for p in samp]),
                             torch.cat([per[p][1] for p in samp])))
    return _ci(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--champion", default="checkpoints/champion_0.0081.pt")
    ap.add_argument("--cache", default="checkpoints/interface_cache.pt")
    ap.add_argument("--protocol", default="info/eval_folds.json")
    ap.add_argument("--mode", choices=["head", "temporal"], default="temporal")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--hidden", type=int, default=0, help="MLP head width (0=linear)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    from src.data.eval_protocol import load_protocol

    proto = load_protocol(args.protocol)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    patients = cache["patients"]
    train_pool, dev_pool, final_pool = (proto["encoder_train"], proto["dev"], proto["final"])
    print(f"mode={args.mode} train={len(train_pool)} dev={len(dev_pool)} final={len(final_pool)}")

    model = JEPAWorldModel(cfg)
    ckpt = torch.load(args.champion, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    for p in model.parameters():
        p.requires_grad = False

    if args.mode == "temporal":
        for p in model.temporal.parameters():
            p.requires_grad = True
        head = model.rano_heads.flat
        for p in head.parameters():
            p.requires_grad = True
        params = list(model.temporal.parameters()) + list(head.parameters())
    else:
        d = next(iter(patients.values()))["states"].shape[-1]
        head = nn.Sequential(nn.Linear(d, args.hidden), nn.GELU(),
                             nn.Linear(args.hidden, 4)) if args.hidden \
            else nn.Linear(d, 4)
        params = list(head.parameters())

    # class weights from train pairs
    counts = torch.zeros(4)
    for pid in train_pool:
        _, lab, valid = pairs_for(patients, pid)
        for c in lab[valid]:
            counts[int(c)] += 1
    counts = counts.clamp_min(1)
    cw = counts.sum() / (4 * counts)
    print(f"train class counts {counts.tolist()} weights {[round(x,2) for x in cw.tolist()]}")

    def maybe_temporal(pid):
        """Return the (T-1,d) state tensor under the current temporal weights."""
        if args.mode != "temporal":
            return patients[pid]["states"]
        p = patients[pid]
        visits = p["visits"]
        days = [parse_week_to_days(v) for v in visits]
        deltas = torch.tensor([[0.0] + [float(b - a) for a, b in zip(days, days[1:])]])
        mask = torch.ones(1, len(visits), dtype=torch.bool)
        st, _ = model.temporal.forward_prefixes(p["fused"].unsqueeze(0), deltas, mask)
        return st[0]

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    def train_epoch():
        order = list(train_pool)
        random.shuffle(order)
        tot, nb = 0.0, 0
        for pid in order:
            st, lab, valid = pairs_for(patients, pid)
            if int(valid.sum()) == 0:
                continue
            inp = maybe_temporal(pid) if args.mode == "temporal" else st
            # align: maybe_temporal returns (T-1,d); pairs_for valid mask is (T-1,)
            if inp.shape[0] != valid.shape[0]:
                m = min(inp.shape[0], valid.shape[0])
                inp, v = inp[:m], valid[:m]
                l = lab[:m]
            else:
                v, l = valid, lab
            if int(v.sum()) == 0:
                continue
            logits = head(inp)[v]
            loss = F.cross_entropy(logits, l[v], weight=cw)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        return tot / max(1, nb)

    best_dev, best_state, best_ep = -1.0, None, -1
    for ep in range(1, args.epochs + 1):
        loss = train_epoch()
        # eval on dev (and final for monitoring only)
        dev_preds, dev_ys, dev_per = [], [], {}
        for pid in dev_pool:
            st, lab, valid = pairs_for(patients, pid)
            if int(valid.sum()) == 0:
                continue
            with torch.no_grad():
                inp = maybe_temporal(pid) if args.mode == "temporal" else st
                m = min(inp.shape[0], valid.shape[0])
                pr = head(inp[:m]).argmax(1)[valid[:m]]
                yy = lab[:m][valid[:m]]
            dev_per[pid] = (pr, yy)
            dev_preds.append(pr)
            dev_ys.append(yy)
        dev_f1 = macro_f1(torch.cat(dev_preds), torch.cat(dev_ys)) if dev_preds else 0.0
        print(f"epoch {ep}: loss={loss:.4f} dev_macroF1={dev_f1:.4f}", flush=True)
        if dev_f1 > best_dev:
            best_dev, best_ep = dev_f1, ep
            best_state = copy.deepcopy({"temporal": model.temporal.state_dict(),
                                        "head": head.state_dict()})

    if best_state is not None:
        model.temporal.load_state_dict(best_state["temporal"])
        head.load_state_dict(best_state["head"])
    print(f"best dev macro-F1 {best_dev:.4f} @ epoch {best_ep}")

    # report final (report-only) + dev, with bootstrap CIs
    for name, pool in (("dev", dev_pool), ("final", final_pool)):
        preds, ys, per = [], [], {}
        for pid in pool:
            st, lab, valid = pairs_for(patients, pid)
            if int(valid.sum()) == 0:
                continue
            with torch.no_grad():
                inp = maybe_temporal(pid) if args.mode == "temporal" else st
                m = min(inp.shape[0], valid.shape[0])
                pr = head(inp[:m]).argmax(1)[valid[:m]]
                yy = lab[:m][valid[:m]]
            per[pid] = (pr, yy)
            preds.append(pr)
            ys.append(yy)
        f1 = macro_f1(torch.cat(preds), torch.cat(ys)) if preds else 0.0
        lo, hi = bootstrap(per, boot=args.boot, seed=args.seed)
        print(f"{name}: {len(per)} patients, {sum(len(y) for _, y in per.values())} rows, "
              f"macro-F1 {f1:.4f} [{lo:.4f},{hi:.4f}]")


if __name__ == "__main__":
    main()
