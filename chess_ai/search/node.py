from __future__ import annotations

from dataclasses import dataclass, field

import chess


@dataclass
class Node:
    board: chess.Board
    prior: float = 0.0
    move: chess.Move | None = None
    parent: "Node | None" = None
    visits: int = 0
    value_sum: float = 0.0
    children: dict[chess.Move, "Node"] = field(default_factory=dict)
    expanded: bool = False

    @property
    def q(self):
        return self.value_sum / self.visits if self.visits else 0.0

    def terminal_value(self):
        outcome = self.board.outcome(claim_draw=True)
        if outcome is None:
            return None
        if outcome.winner is None:
            return 0.0
        return 1.0 if outcome.winner == self.board.turn else -1.0

