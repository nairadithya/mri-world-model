"""Use selected R69 encoder/image decoder and R71 learned-state readout unchanged."""
from pathlib import Path
import json

import torch
from torch import nn
from torch.nn import functional as F

from .legacy.model66 import Decoder, Encoder, _crop, _upsample
from .legacy.model69 import SpatialAccess
from .state import PatientState, digest


def decode(decoder, spatial, features):
    value = decoder.projection(features["compact"].float())
    for i, (projection, block) in enumerate(zip(spatial, decoder.blocks)):
        feature = features[f"feature{3-i}"]
        value = block(_upsample(value, feature.shape[2:]) + projection(feature))
    return _crop(decoder.head(value))


class SegmentationReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = Decoder(8, 4)
        self.spatial = nn.ModuleList([nn.Conv3d(a, b, 1) for a, b in
                                     ((64, 64), (32, 64), (16, 32), (8, 16))])

    def forward(self, features):
        return decode(self.decoder, self.spatial, features)


class LegacyObservation(nn.Module):
    backend = "r69-r71"

    def __init__(self, r69, r71, seed=17, device="cuda"):
        super().__init__()
        r69, r71 = Path(r69), Path(r71)
        self.image_model = SpatialAccess(1)
        self.segmentation = SegmentationReadout()
        image_fit = json.loads((r69 / f"fits-v1/{seed}-1/complete.json").read_text())
        seg_fit = json.loads((r71 / f"fits-v1/{seed}-learned/complete.json").read_text())
        for root, fit, net in ((r69, image_fit, self.image_model), (r71, seg_fit, self.segmentation)):
            path = root / fit["checkpoint"]
            if digest(path) != fit["checkpoint_sha256"]:
                raise ValueError("Checkpoint identity changed")
            saved = torch.load(path, map_location="cpu", weights_only=False)
            net.load_state_dict(saved["model"], strict=True)
        self.model_id = image_fit["checkpoint_sha256"] + ":" + seg_fit["checkpoint_sha256"]
        self.to(device).eval().requires_grad_(False)

    @torch.inference_mode()
    def encode(self, images, metadata):
        if metadata["availability"][:3] != [True, True, True]:
            raise ValueError("R69 bridge requires CT1, T1 and T2; use the joint model for other panels")
        features = self.image_model.encoder.features(images[:, :3])
        compact = self.image_model.encoder.projection(F.adaptive_avg_pool3d(features[-1], 16)).half()
        metadata = dict(metadata, backend=self.backend, model_id=self.model_id,
                        availability=[True, True, True, False],
                        uncertainty_status="forecast uncertainty unavailable")
        return PatientState({"compact": compact, **{f"feature{i}": f for i, f in enumerate(features)}}, metadata)

    @torch.inference_mode()
    def read(self, state):
        if state.metadata["model_id"] != self.model_id:
            raise ValueError("State and decoder model identities differ")
        image = decode(self.image_model.decoder, self.image_model.spatial, state.features)
        return {"logits": self.segmentation(state.features), "images": image[:, :1],
                "image_sequences": ["FLAIR"], "image_log_scale": image[:, 1:2].clamp(-4, 2)}


class LegacyAdaptedAnatomy(nn.Module):
    """Selected R71 direct-image reference, used only as a matched comparator."""
    def __init__(self, r71, seed=17, device="cuda"):
        super().__init__()
        self.encoder = Encoder(8)
        self.readout = SegmentationReadout()
        root = Path(r71)
        fit = json.loads((root / f"fits-v1/{seed}-direct/complete.json").read_text())
        path = root / fit["checkpoint"]
        if digest(path) != fit["checkpoint_sha256"]:
            raise ValueError("Adapted reference checkpoint changed")
        self.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"], strict=True)
        self.to(device).eval().requires_grad_(False)

    @torch.inference_mode()
    def forward(self, images):
        features = self.encoder.features(images[:, :3])
        state = {"compact": self.encoder.projection(F.adaptive_avg_pool3d(features[-1], 16)).half(),
                 **{f"feature{i}": f for i, f in enumerate(features)}}
        return self.readout(state)
