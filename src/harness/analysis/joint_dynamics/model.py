"""Named cutoff-only prediction boundary and the fixed R38 gradient factorial."""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F

from src.model.representation import SpatialState
from .protocol import ARMS, OBSERVATION_SD, PLAN, Arm


@dataclass(frozen=True)
class ForecastInputs:
    images: torch.Tensor
    availability: torch.Tensor
    gap_log1p_days: torch.Tensor
    history_valid: torch.Tensor
    image_present: torch.Tensor

    def validate(self) -> None:
        if (self.images.shape != (1, 4, 4, 96, 96, 96)
                or self.availability.shape != (1, 4, 4)
                or any(t.shape != (1, 4) for t in
                       (self.gap_log1p_days, self.history_valid, self.image_present))):
            raise ValueError('expected batch-one, four-slot 96-cubed forecast tensors')
        if any(t.dtype != torch.float32 for t in
               (self.images, self.availability, self.gap_log1p_days)):
            raise ValueError('forecast arrays must reconstruct to FP32 before autocast')
        if not all(torch.isfinite(t).all() for t in
                   (self.images, self.availability, self.gap_log1p_days)):
            raise ValueError('nonfinite forecast input')
        if (self.history_valid.dtype != torch.bool or self.image_present.dtype != torch.bool
                or not ((self.availability == 0) | (self.availability == 1)).all()
                or not self.history_valid[:, -1].all() or not self.image_present[:, -1].all()
                or (self.image_present & ~self.history_valid).any()
                or (self.history_valid[:, :-1] & ~self.history_valid[:, 1:]).any()
                or (self.gap_log1p_days < 0).any()):
            raise ValueError('invalid right-aligned history masks or covariates')
        for i in range(4):
            valid = bool(self.history_valid[0, i])
            if valid and not self.availability[0, i].bool().any():
                raise ValueError('valid visit has no available modality metadata')
            if not valid and (self.availability[0, i].any() or self.gap_log1p_days[0, i] != 0):
                raise ValueError('padding has nonzero metadata')
        first = int(self.history_valid[0].nonzero()[0])
        if self.gap_log1p_days[0, first] != 0 or (self.gap_log1p_days[0, first + 1:] <= 0).any():
            raise ValueError('first retained gap must be zero and later history gaps positive')


@dataclass(frozen=True)
class AnatomyTargets:
    native_volume: torch.Tensor
    masks: torch.Tensor

    def validate(self) -> None:
        if self.native_volume.shape != (1, 3) or self.masks.shape != (1, 3, 96, 96, 96):
            raise ValueError('invalid anatomy target shapes')
        if self.native_volume.dtype != torch.float32 or self.masks.dtype != torch.float32:
            raise ValueError('anatomy targets must be FP32')
        if (not torch.isfinite(self.native_volume).all() or (self.native_volume < 0).any()
                or not torch.isfinite(self.masks).all()
                or not ((self.masks == 0) | (self.masks == 1)).all()):
            raise ValueError('invalid native log-volume or automated mask target')


@dataclass(frozen=True)
class Prediction:
    mean: torch.Tensor
    intrinsic_scale: torch.Tensor
    current_mean: torch.Tensor
    current_standardized: torch.Tensor
    current_logits: torch.Tensor
    future_logits: torch.Tensor

    @property
    def total_scale(self) -> torch.Tensor:
        return (self.intrinsic_scale.float().square() + OBSERVATION_SD ** 2).sqrt()


class ForecastHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.recurrent = nn.GRUCell(247, 64)
        self.change = nn.Linear(64, 3)
        self.intrinsic = nn.Linear(64, 3)
        self.dense = nn.Sequential(nn.Conv3d(128, 32, 3, padding=1), nn.SiLU(), nn.Conv3d(32, 3, 1))

    def forward(self, pooled: list[torch.Tensor], cutoff_map: torch.Tensor,
                anchor: torch.Tensor, inputs: ForecastInputs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = pooled[-1].new_zeros((1, 64))
        for i in range(4):
            covariates = torch.cat([pooled[i], inputs.availability[:, i],
                inputs.gap_log1p_days[:, i:i + 1], inputs.history_valid[:, i:i + 1].float(),
                inputs.image_present[:, i:i + 1].float()], dim=1)
            updated = self.recurrent(covariates, hidden)
            hidden = torch.where(inputs.history_valid[:, i:i + 1], updated, hidden)
        mean = anchor.float() + self.change(hidden).float()
        intrinsic = F.softplus(self.intrinsic(hidden).float())
        spatial_hidden = hidden[..., None, None, None].expand(-1, -1, 24, 24, 24)
        logits = self.dense(torch.cat([cutoff_map, spatial_hidden], dim=1))
        logits = F.interpolate(logits, size=(96, 96, 96), mode='trilinear', align_corners=False)
        return mean, intrinsic, logits


class JointDynamics(nn.Module):
    def __init__(self, mean: Iterable[float], scale: Iterable[float]) -> None:
        super().__init__()
        self.state = SpatialState(widths=(16, 32, 64, 128))
        self.head = ForecastHead()
        self.register_buffer('volume_mean', torch.tensor(tuple(mean), dtype=torch.float32))
        self.register_buffer('volume_scale', torch.tensor(tuple(scale), dtype=torch.float32))
        if (self.volume_mean.shape != (3,) or self.volume_scale.shape != (3,)
                or not torch.isfinite(self.volume_mean).all()
                or not torch.isfinite(self.volume_scale).all() or (self.volume_scale < 0.5).any()):
            raise ValueError('invalid source normalizer')

    def forward(self, inputs: ForecastInputs, arm: Arm) -> Prediction:
        if type(inputs) is not ForecastInputs or arm not in ARMS:
            raise TypeError('prediction accepts only named label-free ForecastInputs and a fixed arm')
        inputs.validate()
        expected_present = inputs.history_valid.clone()
        if not arm.history:
            expected_present[:, :-1] = False
        if not torch.equal(inputs.image_present, expected_present):
            raise ValueError('image-present metadata differs from the fixed arm')
        pooled = []
        current_features = None
        for i in range(4):
            if bool(expected_present[0, i]):
                features = self.state.encode(inputs.images[:, i], inputs.availability[:, i])
                pooled.append(torch.cat([f.mean((-3, -2, -1)) for f in features], dim=1))
                if i == 3:
                    current_features = features
            else:
                pooled.append(inputs.images.new_zeros((1, 240)))
        assert current_features is not None
        current_standardized = self.state.volume_head(current_features[-1].mean((-3, -2, -1)))
        current_mean = current_standardized.float() * self.volume_scale + self.volume_mean
        current_logits = self.state.decoder(current_features)
        cutoff_map, anchor = current_features[2], current_mean
        if not arm.joint:
            pooled = [p.detach() for p in pooled]
            cutoff_map, anchor = cutoff_map.detach(), anchor.detach()
        mean, intrinsic, future_logits = self.head(pooled, cutoff_map, anchor, inputs)
        return Prediction(mean, intrinsic, current_mean, current_standardized,
                          current_logits, future_logits)


def soft_dice_loss(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    probability, masks = logits.float().sigmoid(), masks.float()
    axes = (-3, -2, -1)
    return 1 - ((2 * (probability * masks).sum(axes) + 1) /
                (probability.sum(axes) + masks.sum(axes) + 1)).mean()


def gaussian_crps(mean: torch.Tensor, scale: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    mean, scale, truth = mean.float(), scale.float(), truth.float()
    if (scale <= 0).any():
        raise ValueError('CRPS requires positive total scales')
    z = (truth - mean) / scale
    phi = torch.exp(-0.5 * z.square()) / math.sqrt(2 * math.pi)
    cdf = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
    return scale * (z * (2 * cdf - 1) + 2 * phi - 1 / math.sqrt(math.pi))


def losses(prediction: Prediction, current: AnatomyTargets, future: AnatomyTargets,
           mean: torch.Tensor, scale: torch.Tensor) -> dict[str, torch.Tensor]:
    current.validate(); future.validate()
    # Autocast is explicitly disabled for every loss reduction, including BCE.
    with torch.autocast(device_type=prediction.mean.device.type, enabled=False):
        current_loss = (F.binary_cross_entropy_with_logits(prediction.current_logits.float(), current.masks.float())
            + soft_dice_loss(prediction.current_logits, current.masks)
            + F.l1_loss(prediction.current_standardized.float(), (current.native_volume.float() - mean.float()) / scale.float()))
        forecast_loss = gaussian_crps(prediction.mean, prediction.total_scale, future.native_volume).mean()
        forecast_loss = forecast_loss + 0.1 * soft_dice_loss(prediction.future_logits, future.masks)
        return {'current': current_loss, 'forecast': forecast_loss, 'total': current_loss + forecast_loss}


def binary_dice(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    prediction, target = logits.float() >= 0, masks.bool()
    axes = (-3, -2, -1)
    denominator = prediction.sum(axes) + target.sum(axes)
    return torch.where(denominator == 0, torch.ones_like(denominator, dtype=torch.float32),
                       2 * (prediction & target).sum(axes).float() / denominator.clamp_min(1))


def configure_torch() -> None:
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (None, ':4096:8'):
        raise ValueError('incompatible CUBLAS determinism configuration')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def finite_step(model: JointDynamics, optimizer: torch.optim.Optimizer,
                loss: torch.Tensor) -> dict[str, float]:
    if not torch.isfinite(loss):
        raise ValueError('nonfinite loss; update is not successful')
    loss.backward()
    norms = {}
    for name, module in (('state', model.state), ('head', model.head)):
        parameters = [p for p in module.parameters() if p.grad is not None]
        if not parameters or not all(torch.isfinite(p.grad).all() for p in parameters):
            raise ValueError(f'missing/nonfinite {name} gradients')
        norms[name] = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True))
    optimizer.step()
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('nonfinite parameters after update')
    return norms
