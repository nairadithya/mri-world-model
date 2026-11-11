import torch
from torch import nn
from torch.nn import functional as F

from .model66 import Encoder, Decoder, initialize, _upsample, _crop


class SpatialAccess(nn.Module):
    def __init__(self, gate):
        super().__init__()
        if gate not in (0, 1):
            raise ValueError("Gate must be fixed at zero or one")
        self.gate = int(gate)
        self.encoder = Encoder(8)
        self.decoder = Decoder(8, 2)
        self.spatial = nn.ModuleList(
            [nn.Conv3d(a, b, 1) for a, b in ((64, 64), (32, 64), (16, 32), (8, 16))]
        )

    def forward(self, images):
        features = self.encoder.features(images)
        state = self.encoder.projection(F.adaptive_avg_pool3d(features[-1], 16)).to(
            torch.float16
        )
        tensor = self.decoder.projection(state.float())
        for feature, projection, block in zip(
            reversed(features), self.spatial, self.decoder.blocks
        ):
            tensor = (
                _upsample(tensor, feature.shape[2:]) + projection(feature) * self.gate
            )
            tensor = block(tensor)
        output = _crop(self.decoder.head(tensor))
        return output[:, :1], output[:, 1:2].clamp(-4, 2)


def fresh(gate, seed):
    model = SpatialAccess(gate)
    initialize(model, seed)
    return model.cuda()
