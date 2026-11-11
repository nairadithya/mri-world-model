import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


CHANNELS = (2, 8, 32)
STATE_SIZE = 16
INPUT_SHAPE = (182, 218, 182)


def _block(inputs, outputs):
    return nn.Sequential(
        nn.Conv3d(inputs, outputs, 3, padding=1),
        nn.GroupNorm(4, outputs),
        nn.SiLU(),
        nn.Conv3d(outputs, outputs, 3, padding=1),
        nn.GroupNorm(4, outputs),
        nn.SiLU(),
    )


def _crop(tensor):
    return tensor[..., 5:-5, 3:-3, 5:-5]


def _upsample(tensor, size):
    return F.interpolate(tensor, size=size, mode="trilinear", align_corners=False)


class Encoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        if channels not in CHANNELS:
            raise ValueError(f"unsupported state channels: {channels}")
        self.blocks = nn.ModuleList(
            [_block(3, 8), _block(8, 16), _block(16, 32), _block(32, 64)]
        )
        self.projection = nn.Conv3d(64, channels, 1)

    def features(self, images):
        if images.ndim != 5 or tuple(images.shape[1:]) != (3, *INPUT_SHAPE):
            raise ValueError("images must have shape [N, 3, 182, 218, 182]")
        tensor = F.pad(images, (5, 5, 3, 3, 5, 5))
        features = []
        for index, block in enumerate(self.blocks):
            if index:
                tensor = F.avg_pool3d(tensor, 2)
            tensor = block(tensor)
            features.append(tensor)
        return features

    def forward(self, images):
        bottleneck = self.features(images)[-1]
        state = self.projection(F.adaptive_avg_pool3d(bottleneck, STATE_SIZE))
        return state.to(torch.float16)


class Decoder(nn.Module):
    def __init__(self, channels, outputs):
        super().__init__()
        if channels not in CHANNELS:
            raise ValueError(f"unsupported state channels: {channels}")
        self.projection = nn.Conv3d(channels, 64, 1)
        self.blocks = nn.ModuleList(
            [_block(64, 64), _block(64, 32), _block(32, 16), _block(16, 8)]
        )
        self.head = nn.Conv3d(8, outputs, 1)

    def forward(self, state):
        if (
            state.ndim != 5
            or state.dtype != torch.float16
            or tuple(state.shape[2:]) != (16, 16, 16)
        ):
            raise ValueError(
                "decoder input must be a batched Float16 state on the fixed 16-cubed grid"
            )
        tensor = self.projection(state.float())
        for size, block in zip(
            ((24, 28, 24), (48, 56, 48), (96, 112, 96), (192, 224, 192)),
            self.blocks,
        ):
            tensor = block(_upsample(tensor, size))
        return _crop(self.head(tensor))


class Representation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.encoder = Encoder(channels)
        self.decoder = Decoder(channels, 2)

    def forward(self, images):
        output = self.decoder(self.encoder(images))
        return output[:, :1], output[:, 1:2].clamp(-4, 2)


class DirectSegmentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(32)
        self.encoder.projection = nn.Identity()
        self.blocks = nn.ModuleList([_block(96, 32), _block(48, 16), _block(24, 8)])
        self.head = nn.Conv3d(8, 4, 1)

    def forward(self, images):
        features = self.encoder.features(images)
        tensor = features[-1]
        for skip, block in zip(reversed(features[:-1]), self.blocks):
            tensor = block(torch.cat((_upsample(tensor, skip.shape[2:]), skip), dim=1))
        return _crop(self.head(tensor))


def initialize(module, seed):
    for module_name, layer in module.named_modules():
        for parameter_name, parameter in layer.named_parameters(recurse=False):
            name = f"{module_name}.{parameter_name}" if module_name else parameter_name
            key = f"{seed}:{name}:{tuple(parameter.shape)}".encode()
            parameter_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
            generator = torch.Generator(device=parameter.device).manual_seed(
                parameter_seed
            )
            with torch.no_grad():
                if isinstance(layer, nn.Conv3d):
                    if parameter_name == "weight":
                        nn.init.kaiming_uniform_(
                            parameter, a=math.sqrt(5), generator=generator
                        )
                    else:
                        nn.init.zeros_(parameter)
                elif isinstance(layer, nn.GroupNorm):
                    nn.init.ones_(
                        parameter
                    ) if parameter_name == "weight" else nn.init.zeros_(parameter)
                else:
                    raise TypeError(f"unsupported parameter: {name}")


def write_state(path, state_without_batch):
    state = state_without_batch.detach().cpu()
    if (
        state.ndim != 4
        or state.shape[0] not in CHANNELS
        or tuple(state.shape[1:]) != (STATE_SIZE,) * 3
    ):
        raise ValueError("state must have shape [2, 8, or 32, 16, 16, 16]")
    if state.dtype != torch.float16:
        raise ValueError("state must have dtype float16")
    if not torch.isfinite(state).all():
        raise ValueError("state must be finite")
    payload = state.numpy().astype("<f2", copy=False).tobytes(order="C")
    with Path(path).open("xb") as file:
        file.write(payload)


def read_state(path, channels, device):
    if channels not in CHANNELS:
        raise ValueError(f"unsupported state channels: {channels}")
    payload = Path(path).read_bytes()
    if len(payload) != channels * STATE_SIZE**3 * 2:
        raise ValueError("invalid state file length")
    state = np.frombuffer(payload, dtype="<f2").reshape(
        channels, STATE_SIZE, STATE_SIZE, STATE_SIZE
    )
    if not np.isfinite(state).all():
        raise ValueError("state must be finite")
    return (
        torch.from_numpy(state.astype(np.float16, copy=True))
        .unsqueeze(0)
        .to(device=device)
    )
