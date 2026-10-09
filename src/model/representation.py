"""Spatial MRI state and observed-pair encoders without mask inputs."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

WIDTHS = (16, 32, 64, 128)


class ResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, output_channels), nn.SiLU(),
            nn.Conv3d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, output_channels))
        self.skip = nn.Conv3d(input_channels, output_channels, 1) if input_channels != output_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.body(x) + self.skip(x))


class SpatialDecoder(nn.Module):
    def __init__(self, widths: tuple[int, ...] = WIDTHS):
        super().__init__()
        self.blocks = nn.ModuleList([ResidualBlock(widths[i + 1] + widths[i], widths[i])
                                     for i in reversed(range(len(widths) - 1))])
        self.output = nn.Conv3d(widths[0], 3, 1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        x = features[-1]
        for block, skip in zip(self.blocks, reversed(features[:-1])):
            x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        return self.output(x)


class SpatialState(nn.Module):
    def __init__(self, widths: tuple[int, ...] = WIDTHS):
        super().__init__()
        self.widths = widths
        self.blocks = nn.ModuleList([ResidualBlock(8 if i == 0 else widths[i - 1], width)
                                     for i, width in enumerate(widths)])
        self.decoder = SpatialDecoder(widths)
        self.volume_head = nn.Linear(widths[-1], 3)

    def encode(self, image: torch.Tensor, availability: torch.Tensor) -> list[torch.Tensor]:
        if image.ndim != 5 or image.shape[1] != 4 or availability.shape != image.shape[:2]:
            raise ValueError("expected MRI [B,4,D,H,W] and availability [B,4]")
        if not torch.isfinite(image).all() or not torch.isfinite(availability).all():
            raise ValueError("nonfinite model input")
        if not ((availability == 0) | (availability == 1)).all() or not availability.bool().any(1).all():
            raise ValueError("availability must describe at least one present sequence")
        available = availability[..., None, None, None]
        x = torch.cat([image * available, available.expand_as(image)], dim=1)
        features = []
        for i, block in enumerate(self.blocks):
            if i:
                x = F.avg_pool3d(x, 2)
            x = block(x)
            features.append(x)
        return features

    def summarize(self, features: list[torch.Tensor]) -> torch.Tensor:
        return features[-1].mean(dim=(-3, -2, -1))

    def forward(self, image: torch.Tensor, availability: torch.Tensor) -> dict:
        features = self.encode(image, availability)
        embedding = self.summarize(features)
        return {"embedding": embedding, "volume": self.volume_head(embedding),
                "mask_logits": self.decoder(features)}


class ObservedChange(nn.Module):
    def __init__(self, encoder: SpatialState, mode: str = "paired"):
        super().__init__()
        if mode not in {"paired", "later_only", "frozen"}:
            raise ValueError("unknown observed-change mode")
        self.encoder = encoder
        self.mode = mode
        self.comparison = nn.ModuleList([nn.Conv3d(3 * width, width, 1) for width in encoder.widths])
        self.decoder = SpatialDecoder(encoder.widths)
        self.change_head = nn.Sequential(nn.Linear(encoder.widths[-1], 128), nn.SiLU(), nn.Linear(128, 3))
        if mode == "frozen":
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def forward(self, source: torch.Tensor, source_availability: torch.Tensor,
                target: torch.Tensor, target_availability: torch.Tensor, *, dense: bool = True) -> dict:
        target_features = self.encoder.encode(target, target_availability)
        source_features = (target_features if self.mode == "later_only" else
                           self.encoder.encode(source, source_availability))
        compared = [layer(torch.cat([a, b, b - a], dim=1))
                    for layer, a, b in zip(self.comparison, source_features, target_features)]
        embedding = self.encoder.summarize(compared)
        current_embedding = self.encoder.summarize(target_features)
        output = {"embedding": embedding, "change": self.change_head(embedding),
                  "volume": self.encoder.volume_head(current_embedding)}
        if dense:
            output.update(mask_logits=self.encoder.decoder(target_features),
                          mask_change=torch.tanh(self.decoder(compared)))
        return output


def nuisance(image: torch.Tensor, kind: str, *, seed: int = 2026) -> torch.Tensor:
    if kind == "identity":
        return image.clone()
    foreground = image != 0
    generator = torch.Generator(device=image.device).manual_seed(seed)
    if kind == "noise":
        changed = image + 0.1 * torch.randn(image.shape, device=image.device, dtype=image.dtype, generator=generator) * foreground
    elif kind == "resolution":
        coarse = tuple(max(2, side // 2) for side in image.shape[-3:])
        changed = F.interpolate(F.interpolate(image, size=coarse, mode="trilinear", align_corners=False),
                                size=image.shape[-3:], mode="trilinear", align_corners=False) * foreground
    elif kind == "bias":
        coordinate = torch.linspace(-0.4, 0.4, image.shape[-1], device=image.device, dtype=image.dtype)
        changed = image * coordinate.exp()[None, None, None, None, :]
    elif kind == "intensity":
        changed = (image * 1.5 + 0.3) * foreground
    else:
        raise ValueError("unknown nuisance transform")
    output = torch.zeros_like(changed)
    for batch in range(len(changed)):
        for sequence in range(changed.shape[1]):
            mask = foreground[batch, sequence]
            if mask.any():
                values = changed[batch, sequence][mask]
                output[batch, sequence][mask] = (values - values.mean()) / values.std().clamp_min(1e-6)
    return output
