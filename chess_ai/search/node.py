from __future__ import annotations

from dataclasses import dataclass, field

import chess


@dataclass
class Node:
    board: chess.Board | None
    prior: float = 0.0
    move: chess.Move | None = None
    parent: "Node | None" = None
    visits: int = 0
    value_sum: float = 0.0
    children: dict[chess.Move, "Node"] = field(default_factory=dict)
    expanded: bool = False
    _terminal_checked: bool = field(default=False, init=False, repr=False)
    _terminal_value: float | None = field(default=None, init=False, repr=False)

    @property
    def q(self):
        return self.value_sum / self.visits if self.visits else 0.0

    def materialize_board(self) -> chess.Board:
        if self.board is None:
            if self.parent is None or self.move is None:
                raise ValueError("a lazy node requires both parent and move")
            self.board = self.parent.materialize_board().copy(stack=True)
            self.board.push(self.move)
        return self.board

    def terminal_value(self):
        if self._terminal_checked:
            return self._terminal_value
        board = self.materialize_board()
        outcome = board.outcome(claim_draw=True)
        self._terminal_checked = True
        if outcome is None:
            return None
        if outcome.winner is None:
            self._terminal_value = 0.0
        else:
            self._terminal_value = 1.0 if outcome.winner == board.turn else -1.0
        return self._terminal_value
