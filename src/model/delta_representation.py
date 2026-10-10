"""MRI-only signed multiscale differences and frozen linear readouts."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from src.model.representation import SpatialDecoder, SpatialState, WIDTHS


class DeltaRepresentation(nn.Module):
    def __init__(self, encoder: SpatialState | None = None):
        super().__init__()
        self.encoder = encoder if encoder is not None else SpatialState()
        self.change_decoder = SpatialDecoder(self.encoder.widths)
        self.change_head = nn.Linear(self.encoder.widths[-1], 3)

    def decode_difference(self, differences: list[torch.Tensor]) -> dict:
        return {"mask_change": self.change_decoder(differences).float().tanh(),
                "change": self.change_head(self.encoder.summarize(differences))}

    def state(self, features: list[torch.Tensor]) -> dict:
        return {"volume": self.encoder.volume_head(self.encoder.summarize(features)),
                "mask_logits": self.encoder.decoder(features)}

    def forward(self, source: torch.Tensor, source_availability: torch.Tensor,
                target: torch.Tensor, target_availability: torch.Tensor) -> dict:
        source_features = self.encoder.encode(source, source_availability)
        target_features = self.encoder.encode(target, target_availability)
        differences = [b - a for a, b in zip(source_features, target_features)]
        return {"source": self.state(source_features), "target": self.state(target_features),
                "differences": differences, **self.decode_difference(differences)}


class LocalLinearProbe(nn.Module):
    def __init__(self, widths: tuple[int, ...] = WIDTHS):
        super().__init__()
        self.widths = widths
        self.levels = nn.ModuleList([nn.Conv3d(width, 3, 1) for width in widths])

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        if len(features) != len(self.levels):
            raise ValueError("probe requires every spatial level")
        size = features[0].shape[-3:]
        return torch.stack([F.interpolate(layer(feature).float(), size=size,
                                         mode="trilinear", align_corners=False)
                            for layer, feature in zip(self.levels, features)]).mean(0)
