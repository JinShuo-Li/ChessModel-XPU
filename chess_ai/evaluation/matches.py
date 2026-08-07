from __future__ import annotations

import math
import time

import chess
import chess.pgn


def play_game(white, black, max_plies=300):
    board = chess.Board(); times = []
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        started = time.perf_counter(); result = (white if board.turn else black)(board.copy(stack=True)); times.append(time.perf_counter() - started)
        move = result.move if hasattr(result, "move") else result
        if move not in board.legal_moves:
            raise ValueError(f"engine returned illegal move {move}")
        board.push(move)
    outcome = board.outcome(claim_draw=True)
    return board, outcome, sum(times) / max(len(times), 1)


def elo_from_score(score):
    if score <= 0 or score >= 1:
        return float("-inf") if score <= 0 else float("inf")
    return -400 * math.log10(1 / score - 1)

