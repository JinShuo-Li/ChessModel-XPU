from __future__ import annotations

import torch
from torch import nn

from chess_ai.board.encoding import INPUT_PLANES
from .heads import PolicyHead, ValueHead
from .residual import ResidualBlock


class ChessNetwork(nn.Module):
    def __init__(self, blocks: int = 12, channels: int = 192, se: bool = True, se_hidden: int = 32, moves_left: bool = False):
        super().__init__()
        self.architecture = {"blocks": blocks, "channels": channels, "se": se, "se_hidden": se_hidden, "moves_left": moves_left}
        self.stem = nn.Sequential(nn.Conv2d(INPUT_PLANES, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels), nn.ReLU())
        self.tower = nn.Sequential(*(ResidualBlock(channels, se, se_hidden) for _ in range(blocks)))
        self.policy = PolicyHead(channels)
        self.value = ValueHead(channels, moves_left)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        x = self.tower(self.stem(x))
        policy = self.policy(x)
        wdl, moves_left = self.value(x)
        return {"policy": policy, "wdl": wdl, "moves_left": moves_left}


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

