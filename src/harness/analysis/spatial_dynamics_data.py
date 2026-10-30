"""Content-bound R37 inputs. Targets stay separate from cutoff-only tensors."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import torch

from src.data.representation import COMPARTMENTS, SEQUENCES, find_image
from src.model.representation import SpatialState, nuisance
from src.model.spatial_dynamics import FrozenAffineVolumeDecoder
from src.preprocessing.transforms import runtime_transform
from src.preprocessing.geometry import Grid, GEOMETRY_CONTRACT


CHECKPOINT_SHA256 = "65a4c4db2ca1b44ebf29fa12659fbd09dea6ede606ce1d3279535144933a001a"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def configure_torch():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def load_bound_inputs():
    # Reuse the independently reviewed E adapter without altering its approved bytes.
    from scripts.run_spatial_noise_ablation import load_inputs, make_contract, gap_days
    data = load_inputs()
    old_contract = make_contract(data)
    visits = {v["visit_id"]: v for v in data["objects"]["prepared"]["visits"]}
    rows = []
    for old in data["rows"]:
        ids = list(old["history_visit_ids"])
        gaps = [0.] + [math.log1p(gap_days(visits[a]["visit"], visits[b]["visit"]))
                         for a, b in zip(ids[:-1], ids[1:])]
        if any(not math.isfinite(g) or g <= 0 for g in gaps[1:]):
            raise ValueError("nonpositive retained historical week-bin gap")
        row = {k: old[k] for k in ("row_id", "patient_id", "cutoff_visit_id", "target_visit_id", "primary_history")}
        row.update(history_visit_ids=ids, gap_log1p=[0.] * (4 - len(ids)) + gaps,
                   history_mask=[False] * (4 - len(ids)) + [True] * len(ids),
                   truth=[math.log1p(old["target"]["values_mm3"][c]) for c in COMPARTMENTS],
                   cutoff_measured=[math.log1p(old["cutoff"]["values_mm3"][c]) for c in COMPARTMENTS])
        rows.append(row)
    return data, old_contract, visits, rows


def validate_visit_geometry(paths, manifest, prepared_pregrid):
    """Compare actual arrays AND physical grids to the already bound evidence."""
    import nibabel as nib
    if manifest.get("schema_version") != 1 or manifest.get("geometry_contract") != GEOMETRY_CONTRACT:
        raise ValueError("incompatible saved geometry evidence")
    reference = Grid(tuple(prepared_pregrid["shape"]), prepared_pregrid["affine"])
    if not paths:
        raise ValueError("no present modalities for geometry verification")
    receipt = {}
    for path in paths:
        path = Path(path)
        key = "/".join((path.parent.parent.name, path.parent.name, path.name.removesuffix(".gz").removesuffix(".nii")))
        record = manifest["images"].get(key)
        if record is None:
            raise ValueError("missing frozen modality geometry evidence")
        initial_sha256 = digest(path)
        image = nib.load(str(path))
        current = Grid(tuple(image.shape), image.affine)
        values = image.get_fdata(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("nonfinite MRI during geometry verification")
        content = hashlib.sha256()
        content.update(json.dumps(current.record(), sort_keys=True).encode())
        content.update(np.ascontiguousarray(values, dtype="<f4").tobytes())
        if record.get("tensor_grid") != current.record() or record.get("content_sha256") != content.hexdigest():
            raise ValueError("current MRI grid/header or content differs from frozen geometry evidence")
        physical = Grid(tuple(record["pregrid"]["shape"]), record["pregrid"]["affine"])
        if np.allclose(current.affine, np.eye(4), rtol=0, atol=1e-6):
            if record.get("reconstruction_verified") is not True or record.get("resize") != "trilinear-align_corners-false-then-nonzero-zscore":
                raise ValueError("identity-affine tensor lacks verified pregrid reconstruction")
        elif not physical.same_as(current):
            raise ValueError("native-affine MRI and its saved pregrid disagree")
        if not physical.same_as(reference):
            raise ValueError("modality physical grid differs from prepared visit pregrid")
        if digest(path) != initial_sha256:
            raise ValueError("MRI changed during geometry verification")
        receipt[str(path)] = {"file_sha256": initial_sha256, "geometry_record_sha256": canonical_hash(record),
                              "prepared_pregrid_sha256": canonical_hash(prepared_pregrid)}
    return receipt


def load_image(root, visit):
    directory = Path(root) / visit["patient_id"] / visit["visit"]
    paths = [find_image(directory, name) for name in SEQUENCES]
    availability = torch.tensor([p is not None for p in paths], dtype=torch.float32)
    if not availability.bool().any():
        raise ValueError("prepared visit has no MRI")
    image = torch.zeros(4, 96, 96, 96)
    for i, path in enumerate(paths):
        if path is not None:
            # R35 prepared tensors were stored float16 then loaded float32.
            image[i] = runtime_transform(str(path))[0].half().float()
    if not torch.isfinite(image).all():
        raise ValueError("nonfinite reconstructed prepared image")
    return image, availability, [p for p in paths if p is not None]


def load_encoder(checkpoint, device):
    model = SpatialState().to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval().requires_grad_(False)
    return model


def extract_cache(root, out, device="cuda"):
    """Encode once; refuse incompatible cached inputs and never refit the encoder."""
    root, out = Path(root), Path(out)
    if out.exists():
        raise FileExistsError("feature extraction output already exists")
    configure_torch()
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("R37 execution requires the operator-selected remote CUDA GPU")
    data, old, visits, rows = load_bound_inputs()
    encoder = load_encoder(data["checkpoint"], device)
    identifiers = sorted(visits)
    maps, available, input_hashes = [], [], {}
    geometry_receipts = {}
    errors, decoded_errors = [], []
    with torch.inference_mode():
        for i, vid in enumerate(identifiers):
            visit = visits[vid]
            directory = root / "data/lumiere_preprocessed" / visit["patient_id"] / visit["visit"]
            paths = [p for name in SEQUENCES if (p := find_image(directory, name)) is not None]
            geometry = validate_visit_geometry(paths, data["objects"]["inventory"]["geometry"], visit["pregrid"])
            image, availability, paths = load_image(root / "data/lumiere_preprocessed", visit)
            feature = encoder.encode(image[None].to(device), availability[None].to(device))[-1]
            pooled = feature.mean((-3, -2, -1))[0].cpu().numpy()
            reference = np.asarray(data["feature_obj"]["features"][vid])
            error = float(np.max(np.abs(pooled - reference)))
            errors.append(error)
            # A cross-hardware tolerance, fixed before seeing reconstructed outputs.
            if not np.allclose(pooled, reference, rtol=1e-4, atol=1e-4):
                raise ValueError("reconstructed input does not reproduce frozen pooled features")
            predicted = encoder.volume_head(feature.mean((-3, -2, -1)))[0].cpu().numpy()
            norm = data["objects"]["run"]["normalization"]
            predicted = predicted * np.asarray(norm["scale"]) + np.asarray(norm["mean"])
            decoded_errors.append(float(np.max(np.abs(predicted - data["predicted"][vid]))))
            if not np.allclose(predicted, data["predicted"][vid], rtol=1e-4, atol=1e-4):
                raise ValueError("reconstructed input does not reproduce frozen volume decoding")
            maps.append(feature[0].cpu())
            available.append(availability)
            for path in paths:
                relative = str(path.relative_to(root))
                checked = geometry[str(path)]
                if digest(path) != checked["file_sha256"]:
                    raise ValueError("MRI changed between geometry verification and encoding")
                input_hashes[relative] = checked["file_sha256"]
                geometry_receipts[relative] = checked
            if (i + 1) % 50 == 0:
                print(f"encode {i+1}/{len(identifiers)}", flush=True)
    all_maps = torch.stack(maps)
    train = set(old["splits"]["source52"])
    source_indexes = [i for i, vid in enumerate(identifiers) if visits[vid]["patient_id"] in train]
    source = all_maps[source_indexes].double()
    mean = source.mean((0, 2, 3, 4)).float()
    std = source.std((0, 2, 3, 4), correction=0).float().clamp_min(1e-6)
    encoding_config = {"encoder_sha256": CHECKPOINT_SHA256, "input_roundtrip": "runtime-transform/float16/float32",
                       "dtype": "float32", "normalization": "source52-unique-visits/population-std/floor-1e-6"}
    provenance = {"created_at": datetime.now(timezone.utc).isoformat(),
                  "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                  "config_sha256": canonical_hash(encoding_config), "config": encoding_config,
                  "encoder_path": str(data["files"]["checkpoint"].relative_to(root)),
                  "runtime_source_sha256": {p: digest(root / p) for p in (
                      "src/harness/analysis/spatial_dynamics_data.py", "src/model/representation.py",
                      "src/preprocessing/transforms.py", "scripts/run_spatial_noise_ablation.py")},
                  "original_cache_provenance": data["feature_obj"]["provenance"]}
    payload = {"schema": "r37-spatial-cache-v1", "provenance": provenance, "compartment_order": list(COMPARTMENTS), "visit_ids": identifiers,
               "maps": all_maps, "availability": torch.stack(available), "feature_mean": mean,
               "feature_std": std, "source_normalizer_visit_ids": [identifiers[i] for i in source_indexes],
               "source_patient_ids": sorted(train), "rows": rows, "splits": old["splits"],
               "input_hashes": {k: digest(p) for k, p in data["files"].items()},
               "image_hashes": input_hashes, "geometry_receipts": geometry_receipts,
               "geometry_receipts_sha256": canonical_hash(geometry_receipts), "checkpoint_sha256": CHECKPOINT_SHA256,
               "normalization": data["objects"]["run"]["normalization"],
               "decoder_weight": data["checkpoint"]["state_dict"]["volume_head.weight"].cpu(),
               "decoder_bias": data["checkpoint"]["state_dict"]["volume_head.bias"].cpu(),
               "reconstruction": {"max_pooled_absolute_error": max(errors),
                                  "max_native_decoded_absolute_error": max(decoded_errors),
                                  "atol": 1e-4, "rtol": 1e-4},
               "timing": old["timing"], "exclusion_counts": old["exclusion_counts"]}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    save_json(out.with_suffix(".json"), {k: v for k, v in payload.items()
              if k not in {"maps", "availability", "feature_mean", "feature_std", "decoder_weight", "decoder_bias"}})
    return payload


class SpatialData:
    def __init__(self, payload, device="cuda"):
        if payload.get("schema") != "r37-spatial-cache-v1" or payload["checkpoint_sha256"] != CHECKPOINT_SHA256:
            raise ValueError("wrong spatial cache schema or encoder")
        self.payload = payload
        self.device = device
        self.ids = payload["visit_ids"]
        self.index = {v: i for i, v in enumerate(self.ids)}
        self.mean = payload["feature_mean"].to(device)
        self.std = payload["feature_std"].to(device)
        self.maps = (payload["maps"].to(device) - self.mean[None, :, None, None, None]) / self.std[None, :, None, None, None]
        self.availability = payload["availability"].to(device)
        norm = payload["normalization"]
        self.decoder = FrozenAffineVolumeDecoder(payload["decoder_weight"].to(device), payload["decoder_bias"].to(device),
            self.mean, self.std, torch.tensor(norm["mean"], device=device), torch.tensor(norm["scale"], device=device))
        self.rows = payload["rows"]
        self.history_signatures = {r["row_id"]: self.history_signature(r) for r in self.rows}
        self.splits = {k: set(v) for k, v in payload["splits"].items()}
        if any(self.splits[a] & self.splits[b] for a in self.splits for b in self.splits if a < b):
            raise ValueError("overlapping patient splits")

    def select(self, split, history=None):
        return [r for r in self.rows if r["patient_id"] in self.splits[split]
                and (history is None or r["primary_history"] == history)]

    @staticmethod
    def history_signature(row):
        return canonical_hash({k: row[k] for k in ("history_visit_ids", "cutoff_visit_id", "gap_log1p", "history_mask")})

    def inputs(self, row, *, latest_override=None):
        """Only history identities resolve to image tensors. Never read row targets."""
        if self.history_signatures.get(row["row_id"]) != self.history_signature(row):
            raise ValueError("history/covariates differ from the locked cutoff row")
        ids = row["history_visit_ids"]
        if not ids or len(ids) > 4 or ids[-1] != row["cutoff_visit_id"]:
            raise ValueError("history must terminate at the cutoff")
        indexes = [self.index[v] for v in ids]
        maps = self.maps.new_zeros((1, 4, 128, *self.maps.shape[-3:]))
        availability = self.availability.new_zeros((1, 4, 4))
        maps[0, -len(ids):] = self.maps[indexes]
        availability[0, -len(ids):] = self.availability[indexes]
        if latest_override is not None:
            maps[0, -1] = latest_override
        gaps = torch.tensor([row["gap_log1p"]], device=self.device, dtype=torch.float32)
        mask = torch.tensor([row["history_mask"]], device=self.device, dtype=torch.float32)
        return maps, availability, gaps, mask

    def target(self, row):
        return (self.maps[self.index[row["target_visit_id"]]][None],
                torch.tensor([row["truth"]], device=self.device, dtype=torch.float32))

    def latest_decoded(self, rows):
        raw = self.payload["maps"][[self.index[r["cutoff_visit_id"]] for r in rows]].mean((-3, -2, -1))
        norm = self.payload["normalization"]
        prediction = torch.nn.functional.linear(raw, self.payload["decoder_weight"], self.payload["decoder_bias"])
        return prediction.numpy() * np.asarray(norm["scale"]) + np.asarray(norm["mean"])

    def changed_latest(self, row, encoder, root, condition):
        from scripts.run_spatial_noise_ablation import BASE
        prepared = json.loads((BASE / "training_prepared.json").read_text())
        visit = next(v for v in prepared["visits"] if v["visit_id"] == row["cutoff_visit_id"])
        image, availability, _ = load_image(Path(root) / "data/lumiere_preprocessed", visit)
        with torch.inference_mode():
            altered = nuisance(image[None].to(self.device), condition)
            feature = encoder.encode(altered, availability[None].to(self.device))[-1][0]
        return (feature - self.mean[:, None, None, None]) / self.std[:, None, None, None]
