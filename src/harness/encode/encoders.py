"""Frozen representation encoding pass.

Produces the legacy top-level cache schema (``patient[view]``) so every
committed reader keeps working; :mod:`harness.encode.cache` can lift it to the
canonical nested schema when needed. Generalises the near-identical encode
loops in ``probe_rano``/``horizon_probe``.
"""
from __future__ import annotations

import time

import torch
from torch.utils.data import DataLoader

from .. import provenance as prov_mod
from ..paths import DEFAULT_CONFIG
from ..data.builder import build_datasets
from ..data.tasks import RANO_PROBE_MAP
from ...data.collate import make_collate
from ...model.jepa_model import JEPAWorldModel

ALL_VIEWS = ("vision", "fused", "states", "clinical", "ema_z", "z")


def load_champion(cfg: dict, champion_path: str):
    model = JEPAWorldModel(cfg)
    ckpt = torch.load(champion_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"champion: {champion_path} (epoch {ckpt.get('epoch')}, "
          f"val {ckpt.get('val_loss')})", flush=True)
    return model, ckpt


@torch.no_grad()
def features_for_batch(model, batch, item, ds, views):
    """Return (features dict, meta dict) for a single-patient batch."""
    pid = batch["patient_id"][0]
    n = int(batch["n_visits"][0])
    v = model.encode_visits(batch["mri"], batch["mri_mask"])          # (1,T,768)
    c = model.clinical(batch["clinical"])                             # (1,384)
    tok = model.fusion(v, c.unsqueeze(1).expand(-1, v.shape[1], -1))  # (1,T,1152)
    feats = {}
    if "vision" in views:
        feats["vision"] = v[0, :n].clone()
    if "fused" in views:
        feats["fused"] = tok[0, :n].clone()
    if "clinical" in views:
        feats["clinical"] = c[0].clone()
    if "states" in views:
        states, _ = model.temporal.forward_prefixes(
            tok, batch["time_deltas"], batch["visit_mask"])
        feats["states"] = states[0, :n - 1].clone()
    if "ema_z" in views or "z" in views:
        z = model.encode_target_visit(batch["mri"], batch["mri_mask"])[0, :n].clone()
        if "ema_z" in views:
            feats["ema_z"] = z.clone()
        if "z" in views:
            feats["z"] = z.clone()
    labels = batch.get("response_labels", torch.full_like(batch["actions"], -1))[0, :n]
    valid = batch.get("response_valid", (labels >= 0).unsqueeze(0))[0, :n]
    operative = batch.get("operative_event",
                          torch.zeros((1, n), dtype=torch.bool))[0, :n]
    treatment = batch.get("treatment", torch.full((1, n), -1, dtype=torch.long))[0, :n]
    meta = {
        "labels": labels.clone(),
        "response_labels": labels.clone(),
        "response_valid": valid.clone(),
        "operative_event": operative.clone(),
        "treatment": treatment.clone(),
        "visits": list(item["visits"][:n]),
        "deltas": batch["time_deltas"][0, :n].clone(),
        "has_img": batch["mri_mask"][0, :n].any(dim=-1).clone(),
    }
    return feats, meta


def encode_lumiere(cfg: dict, champion_path: str, cache_path: str,
                   views=ALL_VIEWS, patients=None, write_provenance=True,
                   config_path: str | None = None) -> dict:
    datasets, _ = build_datasets(cfg, patients=patients)
    size = tuple(cfg["preprocessing"].get("target_size", [96, 96, 96]))
    collate = make_collate(size)
    model, ckpt = load_champion(cfg, champion_path)

    cache = {"schema_version": 2, "patients": {}}
    t0 = time.time()
    done = 0
    total = sum(len(ds) for ds in datasets.values())
    for split, ds in datasets.items():
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                            collate_fn=collate)
        for batch in loader:
            pid = batch["patient_id"][0]
            item = ds[ds.patients.index(pid)]
            feats, meta = features_for_batch(model, batch, item, ds, views)
            cache["patients"][pid] = {"split": split, **feats, **meta}
            done += 1
            if done == 1 or done % 10 == 0 or done == total:
                el = time.time() - t0
                print(f"encoded {done}/{total} ({el / done:.1f}s/patient, "
                      f"ETA {el / done * (total - done) / 60:.0f}min)",
                      flush=True)
    if write_provenance:
        cache["provenance"] = prov_mod.make_provenance(
            champion_path, config_path or DEFAULT_CONFIG,
            champion_epoch=ckpt.get("epoch"),
            champion_val=ckpt.get("val_loss"),
            views=list(views),
            clinical_schema=("survival_free_v1"
                             if not cfg.get("data", {}).get("include_survival", False)
                             else "retrospective_survival_v1"),
        )
    n_lab = sum(int((p["labels"] >= 0).sum()) for p in cache["patients"].values())
    torch.save(cache, cache_path)
    print(f"cache: {total} patients, {n_lab} RANO-labelled visits -> {cache_path}")
    return cache
