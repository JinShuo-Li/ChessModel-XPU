from __future__ import annotations

import torch
from torch import nn


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=(2, 3))
        scale, bias = self.fc2(torch.relu(self.fc1(pooled))).chunk(2, dim=1)
        return x * torch.sigmoid(scale)[:, :, None, None] + bias[:, :, None, None]


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, se: bool = False, se_hidden: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SqueezeExcitation(channels, se_hidden) if se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.se(self.bn2(self.conv2(x)))
        return torch.relu(x + residual)

