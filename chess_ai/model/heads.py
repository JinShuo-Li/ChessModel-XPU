from __future__ import annotations

from torch import nn

from chess_ai.board.moves import POLICY_PLANES


class PolicyHead(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.logits = nn.Conv2d(32, POLICY_PLANES, 1)

    def forward(self, x):
        # NHWC flattening gives policy order square * 73 + move_plane.
        return self.logits(self.body(x)).permute(0, 2, 3, 1).reshape(x.shape[0], -1)


class ValueHead(nn.Module):
    def __init__(self, channels: int, moves_left: bool):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(), nn.Flatten(), nn.Linear(32 * 64, 256), nn.ReLU())
        self.wdl = nn.Linear(256, 3)
        self.moves_left = nn.Linear(256, 1) if moves_left else None

    def forward(self, x):
        x = self.body(x)
        return self.wdl(x), None if self.moves_left is None else self.moves_left(x).squeeze(1)
