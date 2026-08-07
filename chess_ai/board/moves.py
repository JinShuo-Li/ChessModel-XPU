from __future__ import annotations

import chess
import numpy as np

POLICY_PLANES = 73
POLICY_SIZE = 64 * POLICY_PLANES

_QUEEN_DIRS = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))
_KNIGHT_DIRS = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
_UNDERPROMOS = (chess.KNIGHT, chess.BISHOP, chess.ROOK)
_PROMO_DIRS = (-1, 0, 1)


def _canonical(square: int, turn: chess.Color) -> int:
    return square if turn == chess.WHITE else chess.square_mirror(square)


def move_to_index(board: chess.Board, move: chess.Move) -> int:
    if move not in board.legal_moves:
        raise ValueError(f"illegal move for position: {move.uci()}")
    src, dst = _canonical(move.from_square, board.turn), _canonical(move.to_square, board.turn)
    sf, sr, df, dr = chess.square_file(src), chess.square_rank(src), chess.square_file(dst), chess.square_rank(dst)
    dx, dy = df - sf, dr - sr
    if move.promotion in _UNDERPROMOS:
        try:
            direction = _PROMO_DIRS.index(dx)
        except ValueError as exc:
            raise ValueError("invalid underpromotion direction") from exc
        plane = 64 + _UNDERPROMOS.index(move.promotion) * 3 + direction
    elif (dx, dy) in _KNIGHT_DIRS:
        plane = 56 + _KNIGHT_DIRS.index((dx, dy))
    else:
        distance = max(abs(dx), abs(dy))
        if distance < 1 or distance > 7 or (dx and dy and abs(dx) != abs(dy)):
            raise ValueError(f"move cannot be encoded: {move.uci()}")
        unit = (0 if dx == 0 else dx // abs(dx), 0 if dy == 0 else dy // abs(dy))
        plane = _QUEEN_DIRS.index(unit) * 7 + distance - 1
    return src * POLICY_PLANES + plane


def index_to_move(board: chess.Board, index: int) -> chess.Move:
    if not 0 <= index < POLICY_SIZE:
        raise ValueError(f"policy index out of range: {index}")
    src_c, plane = divmod(index, POLICY_PLANES)
    sf, sr = chess.square_file(src_c), chess.square_rank(src_c)
    promotion = None
    if plane < 56:
        direction, distance0 = divmod(plane, 7)
        ux, uy = _QUEEN_DIRS[direction]
        df, dr = sf + ux * (distance0 + 1), sr + uy * (distance0 + 1)
    elif plane < 64:
        dx, dy = _KNIGHT_DIRS[plane - 56]
        df, dr = sf + dx, sr + dy
    else:
        promo, direction = divmod(plane - 64, 3)
        df, dr = sf + _PROMO_DIRS[direction], sr + 1
        promotion = _UNDERPROMOS[promo]
    if not (0 <= df < 8 and 0 <= dr < 8):
        raise ValueError(f"policy index {index} points off board")
    dst_c = chess.square(df, dr)
    src = _canonical(src_c, board.turn)
    dst = _canonical(dst_c, board.turn)
    if promotion is None and board.piece_type_at(src) == chess.PAWN and chess.square_rank(dst) in (0, 7):
        promotion = chess.QUEEN
    move = chess.Move(src, dst, promotion=promotion)
    if move not in board.legal_moves:
        raise ValueError(f"policy index {index} is not legal in this position")
    return move


def legal_move_mask(board: chess.Board) -> np.ndarray:
    mask = np.zeros(POLICY_SIZE, dtype=np.bool_)
    for move in board.legal_moves:
        mask[move_to_index(board, move)] = True
    return mask

