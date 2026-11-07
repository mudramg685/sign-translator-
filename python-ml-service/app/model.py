"""Small feed-forward classifier for normalized landmark features."""

from __future__ import annotations

import torch
from torch import nn


class LandmarkClassifier(nn.Module):
    """Classify a static one- or two-hand landmark feature vector."""

    def __init__(
        self,
        input_size: int,
        class_count: int,
        hidden_sizes: tuple[int, int] = (128, 64),
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        first_hidden, second_hidden = hidden_sizes
        self.network = nn.Sequential(
            nn.Linear(input_size, first_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(first_hidden, second_hidden),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(second_hidden, class_count),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)
