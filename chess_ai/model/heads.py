from __future__ import annotations

from torch import nn

from chess_ai.board.moves import POLICY_SIZE


class PolicyHead(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(channels, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(), nn.Flatten(), nn.Linear(32 * 64, POLICY_SIZE))

    def forward(self, x):
        return self.net(x)


class ValueHead(nn.Module):
    def __init__(self, channels: int, moves_left: bool):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(), nn.Flatten(), nn.Linear(32 * 64, 256), nn.ReLU())
        self.wdl = nn.Linear(256, 3)
        self.moves_left = nn.Linear(256, 1) if moves_left else None

    def forward(self, x):
        x = self.body(x)
        return self.wdl(x), None if self.moves_left is None else self.moves_left(x).squeeze(1)

