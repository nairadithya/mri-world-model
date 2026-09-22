"""Anatomically supervised lesion observation and transition model."""
from __future__ import annotations

import torch
import torch.nn as nn


class LesionTransitionForecaster(nn.Module):
    def __init__(self, input_dim: int = 7 * 768, observation_dim: int = 256,
                 hidden_dim: int = 256):
        super().__init__()
        self.observation = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, observation_dim),
            nn.GELU(), nn.Dropout(0.1), nn.Linear(observation_dim, observation_dim),
        )
        self.anatomy = nn.Linear(observation_dim, 3)
        self.transition = nn.Sequential(
            nn.LayerNorm(3 * observation_dim + 1),
            nn.Linear(3 * observation_dim + 1, hidden_dim), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(hidden_dim, hidden_dim),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.delta = nn.Linear(hidden_dim, 3)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def encode_observation(self, features: torch.Tensor) -> torch.Tensor:
        return self.observation(features)

    def decode_anatomy(self, observations: torch.Tensor) -> torch.Tensor:
        return self.anatomy(observations)

    def forward(self, features: torch.Tensor, gaps: torch.Tensor,
                source: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forecast every next visit in one patient trajectory.

        features: ``(T,F)``, gaps: ``(T,)`` with gaps[0]=0, source: ``(T,3)``.
        """
        observation = self.encode_observation(features)
        previous, current = observation[:-1], observation[1:]
        transition = self.transition(torch.cat(
            [previous, current, current - previous,
             torch.log1p(gaps[1:].clamp_min(0))[:, None] / 8.0], dim=-1))
        state, _ = self.gru(transition[None])
        prediction = source[:-1] + self.delta(state[0])
        return {"observation": observation, "current_anatomy": self.anatomy(observation),
                "transitions": transition, "states": state[0],
                "prediction": prediction}
