"""Bounded real-MRI CPU forward/backward/EMA check, not a training experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

from src.data.collate import collate_fn
from src.data.dataset import LUMIEREDataset
from src.model.jepa_model import JEPAWorldModel
from src.model.jepa import jepa_loss
from src.harness.encode.lesion import encode_visit


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/pilot.yaml")
    parser.add_argument("--patient", default="Patient-067")
    parser.add_argument("--manifest", default="outputs/anatomy_harmonization.json")
    parser.add_argument("--out", default="outputs/validity/cpu_smoke.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    cfg = yaml.safe_load(Path(args.config).read_text())
    ds = LUMIEREDataset(cfg["data"]["meta_dir"], cfg["data"]["root"],
                        patients=[args.patient], min_visits=4)
    ds.visits[args.patient] = [v for v in ds.visits[args.patient]
                              if ds._image_path(args.patient, v, "CT1") is not None][:4]
    if len(ds.visits[args.patient]) != 4:
        raise ValueError("smoke needs four real CT1 visits")
    sample = ds[0]
    for modality in sample["paths"]:
        if modality != "CT1":
            sample["paths"][modality] = [None] * 4
    batch = collate_fn([sample])
    assert torch.isfinite(batch["clinical"]).all() and torch.isfinite(batch["mri"]).all()
    model = JEPAWorldModel(cfg)
    model.eval()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    out = model(batch)
    out["loss"].backward()
    gradients_finite = all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
    routing = {name: any(p.grad is not None for p in module.parameters())
               for name, module in (("backbone", model.backbone), ("projector", model.projector),
                                    ("predictor", model.predictor))}
    with torch.no_grad():
        all_targets = model.encode_target_visit(batch["mri"], batch["mri_mask"])[0]
        targets = all_targets[1:]
        baseline = {"persistence": float(jepa_loss(all_targets[:-1], targets)),
                    "mean": float(jepa_loss(targets.mean(0).expand_as(targets), targets)),
                    "shuffle": float(jepa_loss(targets.roll(1, 0), targets))}
    before = float(out["loss"].detach())
    optimizer.step()
    changed_name = next(name for name, p in model.backbone.named_parameters()
                        if p.grad is not None and torch.count_nonzero(p.grad).item())
    teacher = dict(model.target.backbone.named_parameters())[changed_name]
    teacher_before = teacher.detach().clone()
    model.update_target()
    teacher_changed = not torch.equal(teacher_before, teacher)
    with torch.no_grad():
        after = model(batch)
        real = batch["mri"][0, 0, :1]
        real_latent = model.backbone(real)
        noise_latent = model.backbone(torch.randn_like(real))
        drift = float((real_latent - noise_latent).norm() / real_latent.norm().clamp_min(1e-9))
    manifest = json.loads(Path(args.manifest).read_text())
    visit = next(v for v in manifest["visits"] if v["cohort"] == "LUMIERE"
                 and v["patient_id"] == args.patient and v["visit"] in sample["visits"]
                 and v.get("measurement") and not v.get("exclusions"))
    crop = encode_visit(model, visit, lumiere_root=Path(cfg["data"]["root"]), sailor_root=Path("unused"))
    crop_finite = all(torch.isfinite(crop[key]).all().item()
                      for key in ("regions", "contrasts", "region_weights"))
    optimizer.zero_grad(set_to_none=True)
    model.train()
    training = model(batch)
    training["loss"].backward()
    train_finite = all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
    report = {"scope": "one optimizer/EMA step on four real CT1 visits; not trained forecasting evidence",
              "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "tracked_dirty": bool(subprocess.check_output(["git", "diff", "HEAD"])),
              "mri_sha256": {path: file_sha256(path) for path in sample["paths"]["CT1"]},
              "config_sha256": file_sha256(args.config),
              "brainiac_sha256": file_sha256(cfg["model"]["brainiac"]["checkpoint"]),
              "source_sha256": file_sha256(__file__),
              "manifest_sha256": file_sha256(args.manifest),
              "seed": 42, "patient_id": args.patient, "visits": sample["visits"],
              "device": "cpu", "time_basis": model.temporal.time_basis,
              "loss_before": before, "loss_after": float(after["loss"]),
              "gradient_routing": routing, "gradients_finite": gradients_finite,
              "teacher_parameter_changed": teacher_changed, "baselines": baseline,
              "ema_checked_parameter": changed_name,
              "training_mode_loss": float(training["loss"].detach()),
              "training_mode_gradients_finite": train_finite,
              "teacher_remains_eval_in_training_mode": not model.target.training,
              "real_vs_noise_encoder_relative_l2": drift,
              "real_crop_encoding": {"visit_id": visit["visit_id"], "regions_shape": list(crop["regions"].shape),
                                     "contrasts_shape": list(crop["contrasts"].shape), "finite": crop_finite,
                                     "geometry_contract": crop["geometry"]["geometry_contract"]},
              "target_std": float(out["target_std"]), "target_eff_rank": float(out["target_eff_rank"])}
    passed = (gradients_finite and teacher_changed and train_finite and crop_finite
              and not model.target.training and drift > 1e-3 and torch.isfinite(training["loss"]).item()
              and torch.isfinite(after["loss"]).item()
              and all(torch.isfinite(torch.tensor(value)).item() for value in baseline.values())
              and routing == {"backbone": True, "projector": False, "predictor": True}
              and report["target_std"] > 0 and report["target_eff_rank"] > 1)
    report["verdict"] = "VERIFIED" if passed else "NOT VERIFIED"
    dest = Path(args.out); dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
