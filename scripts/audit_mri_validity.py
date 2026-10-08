"""Audit geometry, temporal cutoffs, gradient routing, and lesion targets locally."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from src.harness.data.anatomy_manifest import COMPARTMENTS
from src.data.dataset import LUMIEREDataset
from src.harness.encode.lesion import image_paths, mask_specs
from src.model.lesion_forecaster import LesionTransitionForecaster
from src.model.temporal import TemporalTransformer
from src.preprocessing.geometry import GEOMETRY_CONTRACT, align_masks, export_geometry_manifest
from src.preprocessing.transforms import runtime_transform


def target_audit(manifest: dict, unseen: set[str]) -> dict:
    visits = {v["visit_id"]: v for v in manifest["visits"]}
    result = {}
    for cohort in ("LUMIERE", "SAILOR"):
        pairs = [p for p in manifest["pairs"] if p["usable"] and p["cohort"] == cohort
                 and (cohort != "LUMIERE" or p["patient_id"] in unseen)]
        compartments = {}
        for name in COMPARTMENTS:
            rows = []
            for pair in pairs:
                source = visits[pair["source_visit_id"]]["measurement"]["values_mm3"][name]
                target = visits[pair["target_visit_id"]]["measurement"]["values_mm3"][name]
                rows.append({"row_id": pair["row_id"], "patient_id": pair["patient_id"],
                             "source_mm3": source, "target_mm3": target,
                             "absolute_log_change": float(abs(np.log1p(target) - np.log1p(source))),
                             "one_endpoint_empty": bool((source == 0) != (target == 0)),
                             "both_empty": bool(source == 0 and target == 0)})
            total = sum(r["absolute_log_change"] for r in rows)
            by_patient = {}
            for row in rows:
                by_patient.setdefault(row["patient_id"], []).append(row)
            per = {p: {"n_rows": len(rr),
                       "mae": float(np.mean([r["absolute_log_change"] for r in rr])),
                       "one_endpoint_empty": sum(r["one_endpoint_empty"] for r in rr)}
                   for p, rr in by_patient.items()}
            compartments[name] = {
                "n_rows": len(rows), "one_endpoint_empty": sum(r["one_endpoint_empty"] for r in rows),
                "both_empty": sum(r["both_empty"] for r in rows),
                "pooled_persistence_mae": total / len(rows) if rows else None,
                "patient_uniform_persistence_mae": float(np.mean([r["mae"] for r in per.values()])) if per else None,
                "pooled_error_share_one_endpoint_empty":
                    sum(r["absolute_log_change"] for r in rows if r["one_endpoint_empty"]) / total if total else 0.,
                "per_patient": per, "rows": rows}
        result[cohort] = {"n_rows": len(pairs), "n_patients": len({p["patient_id"] for p in pairs}),
                          "compartments": compartments}
    return result


def clinical_audit(meta_dir: str, image_root: str) -> dict:
    dataset = LUMIEREDataset(meta_dir, image_root, min_visits=1)
    vectors = torch.stack([dataset._clinical_vector(row) for row in dataset.demographics.values()])
    return {"n_patients": len(vectors), "nonfinite_values": int((~torch.isfinite(vectors)).sum()),
            "eventual_survival_values_nonzero": int(torch.count_nonzero(vectors[:, -1])),
            "verdict": "VERIFIED" if torch.isfinite(vectors).all() and not vectors[:, -1].any()
                       else "NOT VERIFIED"}


def structural_audit() -> dict:
    torch.manual_seed(42)
    gaps = torch.tensor([[0., 30., 30., 30.]])
    tokens = torch.randn(1, 4, 8)
    valid = torch.ones(1, 4, dtype=torch.bool)
    legacy = TemporalTransformer(d_model=8, nhead=2, num_layers=1, dim_feedforward=16, dropout=0).eval()
    elapsed = TemporalTransformer(d_model=8, nhead=2, num_layers=1, dim_feedforward=16,
                                  dropout=0, time_basis="elapsed").eval()
    elapsed.load_state_dict(legacy.state_dict(), strict=True)
    with torch.no_grad():
        legacy_equal = torch.allclose(legacy(tokens, gaps, valid), legacy(tokens[:, [0, 2, 1, 3]], gaps, valid), atol=1e-6)
        elapsed_equal = torch.allclose(elapsed(tokens, gaps, valid), elapsed(tokens[:, [0, 2, 1, 3]], gaps, valid), atol=1e-6)
    forecaster = LesionTransitionForecaster(input_dim=8, observation_dim=4, hidden_dim=4).eval()
    torch.nn.init.normal_(forecaster.delta.weight)
    features = torch.randn(5, 8, requires_grad=True)
    time = torch.tensor([0., 10., 20., 30., 40.], requires_grad=True)
    volumes = torch.randn(5, 3, requires_grad=True)
    pred = forecaster(features, time, volumes)["prediction"]
    causal = True
    for cutoff in range(4):
        gradients = torch.autograd.grad(pred[cutoff].sum(), (features, time, volumes), retain_graph=True)
        causal &= all(torch.count_nonzero(g[cutoff + 1:]).item() == 0 for g in gradients)
    from src.model import jepa_model

    class TinyEncoder(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.linear = torch.nn.Linear(8, 768)

        def forward(self, x):
            return self.linear(x.flatten(1))

    cfg = {"model": {"brainiac": {}, "clinical": {}, "fusion": {},
                     "temporal": {"num_layers": 1, "dim_feedforward": 32},
                     "predictor": {"hidden_dim": 16}, "target": {}}}
    with patch.object(jepa_model, "BrainiacEncoder", TinyEncoder):
        model = jepa_model.JEPAWorldModel(cfg)
    batch = {"mri": torch.randn(1, 3, 2, 1, 2, 2, 2),
             "mri_mask": torch.ones(1, 3, 2, dtype=torch.bool),
             "visit_mask": torch.ones(1, 3, dtype=torch.bool), "clinical": torch.zeros(1, 6),
             "time_deltas": torch.tensor([[0., 30., 60.]])}
    model(batch)["loss"].backward()
    gradients = {name: any(p.grad is not None for p in module.parameters())
                 for name, module in (("backbone", model.backbone), ("projector", model.projector),
                                      ("predictor", model.predictor))}
    verified = bool(legacy_equal and not elapsed_equal and causal
                    and gradients == {"backbone": True, "projector": False, "predictor": True})
    return {"verdict": "VERIFIED" if verified else "NOT VERIFIED",
            "legacy_equal_gap_history_order_invariant": bool(legacy_equal),
            "elapsed_equal_gap_history_order_invariant": bool(elapsed_equal),
            "lesion_future_gradients_zero_at_every_cutoff": bool(causal),
            "default_jepa_gradient_routing": gradients,
            "scope": "synthetic structural checks; not trained MRI forecasting"}


def legacy_masks(specs):
    channels = []
    for path, label in specs:
        data = np.asanyarray(nib.load(path).dataobj)
        channels.append(torch.from_numpy((data > 0 if label is None else data == label).astype(np.float32)))
    return F.interpolate(torch.stack(channels)[None], size=(96, 96, 96), mode="nearest")[0] > 0


def render_overlay(image, legacy, aligned, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    foreground = torch.nonzero(aligned.any(0))
    if not len(foreground):
        raise ValueError("cannot render an empty aligned lesion")
    position = foreground.float().mean(0).round().long().tolist()
    labels = torch.arange(1, 4)[:, None, None, None]
    fig, axes = plt.subplots(3, 2, figsize=(8, 10), layout="constrained")
    for dimension, coordinate in enumerate(position):
        background = np.take(image.numpy(), coordinate, axis=dimension).T
        for column, masks in enumerate((legacy, aligned)):
            colored = (masks * labels).amax(0).numpy()
            overlay = np.ma.masked_equal(np.take(colored, coordinate, axis=dimension).T, 0)
            axes[dimension, column].imshow(background, origin="lower", cmap="gray", vmin=-1, vmax=3)
            axes[dimension, column].imshow(overlay, origin="lower", cmap="autumn", vmin=1, vmax=3, alpha=.5)
            axes[dimension, column].set_title(f'{("Legacy array resize", "Affine-aligned")[column]}, axis {dimension}')
            axes[dimension, column].axis("off")
    fig.suptitle(title)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def geometry_audit(manifest, args, out):
    chosen, records, portable_paths = [], [], []
    rng = np.random.default_rng(args.seed)
    for cohort in ("LUMIERE", "SAILOR"):
        candidates = [v for v in manifest["visits"] if v["cohort"] == cohort and v.get("measurement")
                      and not v.get("exclusions") and v["measurement"]["total_mm3"] > 0]
        if not candidates:
            records.append({"cohort": cohort, "verdict": "NOT VERIFIED",
                            "error": "no eligible visits available for geometry QA"})
            continue
        selected = rng.choice(len(candidates), min(args.visits_per_cohort, len(candidates)), replace=False)
        chosen.extend(candidates[int(i)] for i in selected)
    for visit in chosen:
        paths = image_paths(visit, Path(args.lumiere_root), Path(args.sailor_root))
        present = [p for p in paths.values() if p is not None]
        specs = mask_specs(visit)
        try:
            aligned = align_masks(specs, present)
            old = legacy_masks(specs)
            aa, bb = old.any(0), aligned.masks.any(0)
            denom = int(aa.sum() + bb.sum())
            dice = 2 * int((aa & bb).sum()) / denom if denom else 1.
            native = torch.tensor([visit["measurement"]["values_mm3"][c] for c in COMPARTMENTS])
            if not torch.allclose(native, aligned.native_volumes_mm3, atol=1e-4):
                raise ValueError("native volumes disagree with the frozen target manifest")
            image_tensor = runtime_transform(str(present[0]))[0]
            zero_overlap = {}
            for i, name in enumerate(COMPARTMENTS):
                count = int(aligned.masks[i].sum())
                zero_overlap[name] = float(((image_tensor == 0) & aligned.masks[i]).sum()) / count if count else None
            png = out / f'geometry_{visit["cohort"]}_{visit["patient_id"]}_{visit["visit"]}.png'
            render_overlay(image_tensor, old, aligned.masks, png,
                           f'{visit["cohort"]} {visit["patient_id"]}/{visit["visit"]}')
            records.append({"visit_id": visit["visit_id"], "cohort": visit["cohort"],
                            "patient_id": visit["patient_id"], "visit": visit["visit"],
                            "verdict": "VERIFIED", "legacy_vs_aligned_union_dice": dice,
                            "native_volumes_preserved": True, "overlay": str(png),
                            "aligned_mask_fraction_on_zero_mri": zero_overlap,
                            "zero_mri_overlap_scope": "exact-zero MRI is a QA proxy, not an anatomical brain mask",
                            "geometry": aligned.provenance})
            if visit["cohort"] == "LUMIERE":
                portable_paths.extend(present)
        except (ValueError, OSError) as exc:
            records.append({"visit_id": visit["visit_id"], "cohort": visit["cohort"],
                            "verdict": "NOT VERIFIED", "error": str(exc)})
    export_geometry_manifest(sorted(set(portable_paths)), out / "sample_geometry_manifest.json")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="outputs/anatomy_harmonization.json")
    parser.add_argument("--protocol", default="info/eval_folds.json")
    parser.add_argument("--lumiere-root", default="data/lumiere_preprocessed")
    parser.add_argument("--meta-dir", default="data/lumiere_meta")
    parser.add_argument("--sailor-root", default="data/sailor/sailor_ebrains_pseud/derivatives/mni2009c-n-s")
    parser.add_argument("--out", default="outputs/validity")
    parser.add_argument("--visits-per-cohort", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.visits_per_cohort < 1:
        parser.error("visits-per-cohort must be positive")
    torch.set_num_threads(2)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text())
    protocol = json.loads(Path(args.protocol).read_text())
    tracked_diff = subprocess.check_output(["git", "diff", "HEAD"])
    report = {"schema_version": 1, "geometry_contract": GEOMETRY_CONTRACT,
              "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "tracked_dirty": bool(tracked_diff),
              "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
              "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
              "seed": args.seed, "structural": structural_audit(),
              "clinical": clinical_audit(args.meta_dir, args.lumiere_root),
              "targets": target_audit(manifest, set(protocol["encoder_unseen"])),
              "geometry": geometry_audit(manifest, args, out),
              "limitations": ["sampled geometry QA; not full-cohort certification",
                              "affine alignment does not prove template equivalence",
                              "target discontinuity is descriptive; no expert truth or noise ceiling",
                              "no trained forecasting gain or independent confirmation"]}
    passed = (report["structural"]["verdict"] == "VERIFIED"
              and report["clinical"]["verdict"] == "VERIFIED"
              and all(r["verdict"] == "VERIFIED" for r in report["geometry"]))
    report["verdict"] = "VERIFIED" if passed else "NOT VERIFIED"
    (out / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f'{report["verdict"]}: structural checks and {len(report["geometry"])} sampled geometry visits')
    for cohort, targets in report["targets"].items():
        print(f'{cohort}: {targets["n_patients"]} patients, {targets["n_rows"]} frozen eligible rows')
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
