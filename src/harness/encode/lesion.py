"""Encode lesion-centred BRAINIAC compartment and context tokens for P2.3."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml

from src.data.sailor import (EMPTY_NZ_THRESHOLD, SAILOR_MODALITIES,
                             _central_nzfrac)
from src.harness.checkpoints import format_report, load_model
from src.harness.data.anatomy_manifest import COMPARTMENTS
from src.harness.provenance import git_sha
from src.model.jepa_model import JEPAWorldModel
from src.model.lesion_tokens import (REGIONS, lesion_centered_crop,
                                     pool_region_tokens, region_patch_weights)
from src.preprocessing.transforms import runtime_transform

SCHEMA_VERSION = 1
FEATURE_VERSION = "lesion-crop-brainiac-v1"
MODALITIES = ("CT1", "T1", "T2", "FLAIR")
SAILOR_ROOT = Path("data/sailor/sailor_ebrains_pseud/derivatives/mni2009c-n-s")


def _find(directory: Path, stems: tuple[str, ...]) -> Path | None:
    for stem in stems:
        for suffix in (".nii.gz", ".nii"):
            path = directory / f"{stem}{suffix}"
            if path.exists():
                return path
    return None


def _find_sailor(directory: Path, stems: tuple[str, ...]) -> Path | None:
    """Mirror SAILOR's content-aware candidate selection exactly."""
    for stem in stems:
        for suffix in (".nii.gz", ".nii"):
            path = directory / f"{stem}{suffix}"
            if path.exists() and _central_nzfrac(str(path)) >= EMPTY_NZ_THRESHOLD:
                return path
    return None


def image_paths(visit: dict, lumiere_root: Path,
                sailor_root: Path) -> dict[str, Path | None]:
    if visit["cohort"] == "LUMIERE":
        directory = lumiere_root / visit["patient_id"] / visit["visit"]
        return {name: _find(directory, (name,)) for name in MODALITIES}
    directory = sailor_root / visit["patient_id"] / visit["visit"]
    stems = {slot: tuple(names) for slot, names in SAILOR_MODALITIES}
    return {name: _find_sailor(directory, stems[name]) for name in MODALITIES}


def compartment_masks(visit: dict, size: int = 96) -> torch.Tensor:
    measurement = visit["measurement"]
    if "mask" in measurement:
        spec = measurement["mask"]
        data = np.asanyarray(nib.load(spec["path"]).dataobj)
        masks = np.stack([data == spec["labels"][name] for name in COMPARTMENTS])
    else:
        masks = np.stack([
            np.asanyarray(nib.load(measurement["masks"][name]["path"]).dataobj) > 0
            for name in COMPARTMENTS
        ])
    tensor = torch.from_numpy(masks.astype(np.float32))[None]
    return torch.nn.functional.interpolate(
        tensor, size=(size, size, size), mode="nearest")[0] > 0


@torch.no_grad()
def encode_visit(model: JEPAWorldModel, visit: dict, *, lumiere_root: Path,
                 sailor_root: Path, crop_size: int = 64) -> dict:
    paths = image_paths(visit, lumiere_root, sailor_root)
    present = torch.tensor([paths[name] is not None for name in MODALITIES])
    images = torch.zeros(len(MODALITIES), 96, 96, 96)
    for index, name in enumerate(MODALITIES):
        if paths[name] is not None:
            images[index] = runtime_transform(str(paths[name]), (96, 96, 96))[0]
    masks = compartment_masks(visit)
    crop_images, crop_masks = lesion_centered_crop(
        images[None], masks[None], crop_size=crop_size, output_size=96)
    tokens = torch.zeros(len(MODALITIES), 216, 768)
    if bool(present.any()):
        hidden = model.backbone.encode_tokens(crop_images[0, present, None].float())
        if hidden.ndim == 2:
            hidden = hidden[:, None]
        hidden = hidden[:, 1:] if hidden.shape[1] == 217 else hidden
        if hidden.shape[1] != 216:
            raise ValueError(f"expected 216 patch tokens, got {hidden.shape[1]}")
        tokens[present] = hidden.cpu()
    weights = region_patch_weights(crop_masks.cpu())
    pooled, contrast = pool_region_tokens(tokens[None], weights)
    return {"modalities": present, "regions": pooled[0].half(),
            "contrasts": contrast[0].half(), "region_weights": weights[0].half(),
            "crop_size": crop_size,
            "paths": {name: str(path) if path else None for name, path in paths.items()}}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--champion", default="checkpoints/champion_0.0081.pt")
    ap.add_argument("--manifest", default="outputs/anatomy_harmonization.json")
    ap.add_argument("--out", default="checkpoints/lesion_crop_features.pt")
    ap.add_argument("--lumiere-root", default="data/lumiere_preprocessed")
    ap.add_argument("--sailor-root", default=str(SAILOR_ROOT))
    ap.add_argument("--crop-size", type=int, default=64)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    manifest = json.loads(Path(args.manifest).read_text())
    model = JEPAWorldModel(cfg)
    checkpoint, report = load_model(model, args.champion)
    print(format_report(report))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    visits = [visit for visit in manifest["visits"]
              if visit.get("measurement") and not visit.get("exclusions")]
    if args.limit is not None:
        visits = visits[:args.limit]
    encoded = {}
    started = time.time()
    for index, visit in enumerate(visits, 1):
        encoded[visit["visit_id"]] = encode_visit(
            model, visit, lumiere_root=Path(args.lumiere_root),
            sailor_root=Path(args.sailor_root), crop_size=args.crop_size)
        elapsed = time.time() - started
        print(f"lesion crop {index}/{len(visits)} "
              f"({elapsed / index:.1f}s/visit, ETA "
              f"{elapsed / index * (len(visits) - index) / 60:.0f}min)", flush=True)
    artifact = {
        "schema_version": SCHEMA_VERSION, "feature_version": FEATURE_VERSION,
        "target_version": manifest["target_version"], "modalities": MODALITIES,
        "regions": REGIONS, "visits": encoded,
        "provenance": {
            "champion": str(Path(args.champion).resolve()),
            "champion_epoch": checkpoint.get("epoch"),
            "champion_val": checkpoint.get("val_loss"),
            "config": str(Path(args.config).resolve()),
            "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
            "manifest": str(Path(args.manifest).resolve()),
            "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
            "git_sha": git_sha(), "date": time.strftime("%Y-%m-%d"),
            "information_cutoff": "current/source visit image and mask only",
            "crop_size": args.crop_size, "output_size": 96,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.out)
    print(f"wrote {args.out}: visits={len(encoded)}")


if __name__ == "__main__":
    main()
