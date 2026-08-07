from __future__ import annotations

import random

import chess
import chess.pgn


def random_positions(count: int, seed: int = 7, min_ply: int = 6, max_ply: int = 80):
    rng = random.Random(seed)
    while count > 0:
        board = chess.Board()
        target = rng.randint(min_ply, max_ply)
        for _ in range(target):
            if board.is_game_over(claim_draw=True):
                break
            board.push(rng.choice(list(board.legal_moves)))
        if not board.is_game_over(claim_draw=True):
            yield board
            count -= 1


def fen_positions(path: str, count: int):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if count <= 0:
                break
            line = line.strip()
            if line:
                yield chess.Board(line)
                count -= 1


def pgn_positions(path: str, count: int, every: int = 8):
    with open(path, encoding="utf-8") as handle:
        while count > 0 and (game := chess.pgn.read_game(handle)) is not None:
            board = game.board()
            for ply, move in enumerate(game.mainline_moves()):
                board.push(move)
                if ply >= 5 and ply % every == 0 and not board.is_game_over(claim_draw=True):
                    yield board.copy(stack=True)
                    count -= 1
                    if count <= 0:
                        return

