"""Frozen RANO probe (D23): does the champion representation encode progression?

Phase 1 --encode: freeze champion, encode every visit (vision 768-d +
fused 1152-d latents), cache per-patient tensors + clean RANO labels.
Phase 2 --probe: train linear/MLP probes on train-split patients, score
macro-F1/accuracy/confusion on test-split patients, with majority and
clinical-only baselines.

Shared eval primitives now live in ``src.harness`` and are re-exported here so
the existing importers (radiomics_probe, concept_probe, sailor_eval,
surprise_signal, leadtime, horizon_probe) keep working unchanged.

Usage:
    python scripts/probe_rano.py --champion <best.pt> --cache probe_cache.pt --encode
    python scripts/probe_rano.py --cache probe_cache.pt --probe
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.collate import make_collate
from src.model.jepa_model import JEPAWorldModel

# --- shared harness primitives (single source of truth) -------------------
from src.harness.data.builder import build_datasets  # noqa: E402,F401
from src.harness.data.tasks import (  # noqa: E402,F401
    RANO_PROBE_MAP,
    RANO_PROBE_NAMES,
    patient_rows,
    rows_for,
)
from src.harness.eval.aggregate import (  # noqa: E402
    patient_bootstrap as _bootstrap,
    percentile_ci as _ci,
    pooled as _pooled,
)
from src.harness.eval.metrics import macro_f1  # noqa: E402,F401
from src.harness.train.readout import fit_linear, scores  # noqa: E402,F401


def encode_all(cfg, champion_path, cache_path):
    device = torch.device("cpu")
    datasets, splits = build_datasets(cfg)
    size = tuple(cfg["preprocessing"].get("target_size", [96, 96, 96]))
    collate = make_collate(size)

    model = JEPAWorldModel(cfg)
    ckpt = torch.load(champion_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"champion: {champion_path} (epoch {ckpt.get('epoch')}, "
          f"val {ckpt.get('val_loss')})")

    cache = {"patients": {}}
    t0 = time.time()
    done = 0
    total = sum(len(ds) for ds in datasets.values())
    with torch.no_grad():
        for split, ds in datasets.items():
            loader = DataLoader(ds, batch_size=1, shuffle=False,
                                num_workers=0, collate_fn=collate)
            for batch in loader:
                pid = batch["patient_id"][0]
                item = ds[ds.patients.index(pid)]
                v = model.encode_visits(batch["mri"], batch["mri_mask"])  # (1,T,768)
                c = model.clinical(batch["clinical"])                     # (1,384)
                tok = model.fusion(v, c.unsqueeze(1).expand(-1, v.shape[1], -1))
                n = int(batch["n_visits"][0])
                labels = []
                for t in range(n):
                    rating = ds.rano.get((pid, item["visits"][t]), "")
                    labels.append(RANO_PROBE_MAP.get(rating, -1))
                cache["patients"][pid] = {
                    "split": split,
                    "vision": v[0, :n].clone(),
                    "fused": tok[0, :n].clone(),
                    "clinical": c[0].clone(),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "n_labeled": sum(l >= 0 for l in labels),
                }
                # Temporal states: state s_t summarizes visits 0..t.
                # Cached alongside so probes test the history summary
                # (thesis: RANO/volume read out from temporal latent).
                states, _ = model.temporal.forward_prefixes(
                    tok, batch["time_deltas"], batch["visit_mask"])
                cache["patients"][pid]["states"] = states[0, :n - 1].clone()
                done += 1
                if done == 1 or done % 10 == 0 or done == total:
                    el = time.time() - t0
                    print(f"encoded {done}/{total} ({el / done:.1f}s/patient, "
                          f"ETA {el / done * (total - done) / 60:.0f}min)",
                          flush=True)
    n_lab = sum(int((p["labels"] >= 0).sum()) for p in cache["patients"].values())
    print(f"cache: {total} patients, {n_lab} RANO-labelled visits -> {cache_path}")
    torch.save(cache, cache_path)


def _train_readout(patients, pids, feat, hidden, seed=42):
    xs, ys = [], []
    for pid in pids:
        r = patient_rows(patients, pid, feat)
        if r is not None:
            xs.append(r[0])
            ys.append(r[1])
    if not xs:
        return None
    return fit_linear(torch.cat(xs), torch.cat(ys), hidden=hidden, seed=seed)


def _predict(net, patients, pids, feat):
    out = {}
    for pid in pids:
        r = patient_rows(patients, pid, feat)
        if r is None or net is None:
            continue
        with torch.no_grad():
            out[pid] = (net(r[0]).argmax(1), r[1])
    return out


def run_locked(cache_path, protocol_path, feat="states_forecast", hidden=256,
               compare=None, train_pool="unseen", boot=10000, seed=42,
               cohort="unseen", readout_seed=42):
    """Locked-protocol evaluation over the encoder-unseen cohort (Step 1).

    ``train_pool='unseen'``: 5-fold patient-wise CV within the 26 (readout
    trained only on encoder-unseen patients). ``'train'``: readout trained on
    the 65 encoder-train patients and scored on the unseen cohort (transfer).
    ``cohort`` restricts the scored patients to ``unseen`` (all 26), ``dev``
    (13 val) or ``final`` (13 test). ``compare`` runs a second feature config
    with a paired bootstrap.
    """
    from src.data.eval_protocol import assert_disjoint, fold_patients, load_protocol

    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    patients = cache["patients"]
    proto = load_protocol(protocol_path)
    assert_disjoint(proto)
    unseen = sorted(proto["folds"])
    k = proto["k"]
    if cohort != "unseen":
        eval_pool = sorted(proto[cohort])
    else:
        eval_pool = unseen
    n_lab = sum(int((patients[p]["labels"] >= 0).sum()) for p in eval_pool)
    prov = cache.get("provenance")
    if prov:
        print(f"cache provenance: champion={os.path.basename(prov['champion'])} "
              f"git={str(prov.get('git_sha'))[:10]} date={prov.get('date')}")
    else:
        print("WARNING: cache has no provenance block (pre-2026-09-10 legacy)")
    print(f"locked protocol v{proto['version']} ({protocol_path}): cohort={cohort} "
          f"{len(eval_pool)} patients, {n_lab} labelled visits; "
          f"train_pool={train_pool}, feat={feat}-{'mlp' if hidden else 'linear'}")

    def oof_for(cfg_feat):
        oof = {}
        if train_pool == "unseen":
            for i in range(k):
                te = fold_patients(proto, i)
                tr = [p for p in unseen if p not in set(te)]
                net = _train_readout(patients, tr, cfg_feat, hidden, seed=readout_seed)
                oof.update(_predict(net, patients, te, cfg_feat))
        else:
            net = _train_readout(patients, proto["encoder_train"], cfg_feat, hidden, seed=readout_seed)
            oof = _predict(net, patients, unseen, cfg_feat)
        return {p: v for p, v in oof.items() if p in set(eval_pool)}

    oof_a = oof_for(feat)
    f1_a, pred_a, y_a = _pooled(oof_a, sorted(oof_a))
    a_vals, d_vals = _bootstrap(oof_a, None, boot=boot, seed=seed)
    lo, hi = _ci(a_vals)
    print(f"  pooled macro-F1 {f1_a:.4f} [{lo:.4f},{hi:.4f}]  "
          f"(n_pat={len(oof_a)}, n_rows={len(y_a)})")
    maj = int(torch.bincount(y_a, minlength=4).argmax())
    print(f"  majority ({RANO_PROBE_NAMES[maj]}) acc={(y_a == maj).float().mean():.4f}")

    if compare and compare != feat:
        oof_b = oof_for(compare)
        f1_b, _, _ = _pooled(oof_b, sorted(oof_b))
        _, d_vals = _bootstrap(oof_a, oof_b, boot=boot, seed=seed)
        dlo, dhi = _ci(d_vals)
        sig = "SIG" if dhi < 0 or dlo > 0 else "n.s."
        print(f"  compare {compare}: pooled {f1_b:.4f}  "
              f"paired diff {f1_a - f1_b:+.4f} [{dlo:+.4f},{dhi:+.4f}] {sig}")


def run_probe(cache_path):
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    patients = cache["patients"]
    splits = {}
    for pid, p in patients.items():
        splits.setdefault(p["split"], []).append(pid)
    n_lab = lambda s: sum(int((patients[p]["labels"] >= 0).sum()) for p in splits[s])
    print(f"labelled visits: train={n_lab('train')} val={n_lab('val')} test={n_lab('test')}")

    maj = torch.bincount(rows_for(cache["patients"], splits, ["test"], "fused")[1],
                         minlength=4).argmax().item()
    _, y_te = rows_for(cache["patients"], splits, ["test"], "fused")
    print(f"majority baseline (predict {RANO_PROBE_NAMES[maj]}): "
          f"acc={(y_te == maj).float().mean():.4f} (macro-F1 ~0 by construction)")

    for feat in ["fused", "vision", "clinical", "states_current", "states_forecast"]:
        x_tr, y_tr = rows_for(cache["patients"], splits, ["train"], feat)
        x_te, y_te = rows_for(cache["patients"], splits, ["test"], feat)
        for hidden in [0, 256]:
            tag = f"{feat}-{'mlp' if hidden else 'linear'}"
            net = fit_linear(x_tr, y_tr, hidden=hidden)
            acc, f1, rec, cm = scores(net, x_te, y_te)
            print(f"{tag}: acc={acc:.4f} macro-F1={f1:.4f} "
                  f"recall={ {k: round(v, 3) for k, v in rec.items()} }")
            print(f"  confusion (true x pred):\n{cm}")


def run_cv(cache_path, k=5, feat="states_forecast", hidden=256, seed=42):
    """k-fold patient-wise CV of one probe config (field protocol)."""
    import random
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    patients = cache["patients"]
    pids = sorted(patients)
    rng = random.Random(seed)
    rng.shuffle(pids)
    folds = [pids[i::k] for i in range(k)]
    accs, f1s = [], []
    for i in range(k):
        te, tr = folds[i], [p for j in range(k) if j != i for p in folds[j]]
        sp = {"tr": tr, "te": te}
        x_tr, y_tr = rows_for(patients, sp, "tr", feat)
        x_te, y_te = rows_for(patients, sp, "te", feat)
        net = fit_linear(x_tr, y_tr, hidden=hidden)
        acc, f1, _, _ = scores(net, x_te, y_te)
        accs.append(acc)
        f1s.append(f1)
        print(f"fold {i}: n_te={len(y_te)} acc={acc:.4f} macro-F1={f1:.4f}", flush=True)
    import statistics
    print(f"CV {feat}-{'mlp' if hidden else 'linear'}: "
          f"acc={statistics.mean(accs):.4f}±{statistics.pstdev(accs):.4f} "
          f"macro-F1={statistics.mean(f1s):.4f}±{statistics.pstdev(f1s):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--champion", default="checkpoints/champion_0.0081.pt")
    ap.add_argument("--cache", default="checkpoints/probe_cache.pt")
    ap.add_argument("--encode", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--cv", action="store_true",
                    help="legacy 5-fold CV over all 91 patients (K3-16 leaky)")
    ap.add_argument("--cv-unseen", action="store_true",
                    help="locked-protocol CV over the encoder-unseen cohort")
    ap.add_argument("--protocol", default="info/eval_folds.json")
    ap.add_argument("--feat", default="states_forecast")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--compare", default=None,
                    help="second feature config for a paired bootstrap")
    ap.add_argument("--train-pool", choices=["unseen", "train"], default="unseen")
    ap.add_argument("--cohort", choices=["unseen", "dev", "final"], default="unseen")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--readout-seed", type=int, default=42)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.encode:
        encode_all(cfg, args.champion, args.cache)
    if args.cv_unseen:
        run_locked(args.cache, args.protocol, feat=args.feat, hidden=args.hidden,
                   compare=args.compare, train_pool=args.train_pool,
                   boot=args.boot, cohort=args.cohort, readout_seed=args.readout_seed)
        return
    if args.probe or (not args.encode and not args.cv):
        run_probe(args.cache)
    if args.cv:
        run_cv(args.cache)


if __name__ == "__main__":
    main()
