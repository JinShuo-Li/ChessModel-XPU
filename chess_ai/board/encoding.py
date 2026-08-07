from __future__ import annotations

import chess
import numpy as np

HISTORY = 8
PLANES_PER_HISTORY = 13  # 12 pieces + repetition indicator
AUX_PLANES = 8
INPUT_PLANES = HISTORY * PLANES_PER_HISTORY + AUX_PLANES  # 112


def _canonical_square(square: int, turn: chess.Color) -> int:
    return square if turn == chess.WHITE else chess.square_mirror(square)


def _piece_plane(piece: chess.Piece, root_turn: chess.Color) -> int:
    relative_color = 0 if piece.color == root_turn else 1
    return relative_color * 6 + piece.piece_type - 1


def _history_boards(board: chess.Board) -> list[chess.Board]:
    current = board.copy(stack=True)
    result = []
    for _ in range(HISTORY):
        result.append(current.copy(stack=True))
        if not current.move_stack:
            break
        current.pop()
    return result


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode from side-to-move perspective into documented 112 binary/scalar planes.

    Planes 0..103 are eight history frames. Each frame contains six friendly and
    six enemy piece planes (P,N,B,R,Q,K), plus a twofold-repetition plane.
    Planes 104..107 are our/opp king/queen-side castling rights; 108 is the
    canonical en-passant target square; 109 is side-to-move (white=1); 110 is
    halfmove clock / 100; 111 is fullmove number / 200 (both clipped).
    """
    root_turn = board.turn
    planes = np.zeros((INPUT_PLANES, 8, 8), dtype=np.float32)
    for h, historical in enumerate(_history_boards(board)):
        base = h * PLANES_PER_HISTORY
        for square, piece in historical.piece_map().items():
            sq = _canonical_square(square, root_turn)
            planes[base + _piece_plane(piece, root_turn), chess.square_rank(sq), chess.square_file(sq)] = 1.0
        if historical.is_repetition(2):
            planes[base + 12].fill(1.0)
    us, them = root_turn, not root_turn
    for offset, color, kingside in ((0, us, True), (1, us, False), (2, them, True), (3, them, False)):
        has_right = board.has_kingside_castling_rights(color) if kingside else board.has_queenside_castling_rights(color)
        if has_right:
            planes[104 + offset].fill(1.0)
    if board.ep_square is not None:
        sq = _canonical_square(board.ep_square, root_turn)
        planes[108, chess.square_rank(sq), chess.square_file(sq)] = 1.0
    if root_turn == chess.WHITE:
        planes[109].fill(1.0)
    planes[110].fill(min(board.halfmove_clock, 100) / 100.0)
    planes[111].fill(min(board.fullmove_number, 200) / 200.0)
    return planes


def pack_encoded(encoded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if encoded.shape != (INPUT_PLANES, 8, 8):
        raise ValueError(f"expected {(INPUT_PLANES, 8, 8)}, got {encoded.shape}")
    binary = np.packbits((encoded[:110] > 0).reshape(110, 64), axis=1, bitorder="little")
    scalars = np.asarray([encoded[110, 0, 0], encoded[111, 0, 0]], dtype=np.float16)
    return binary, scalars


def unpack_encoded(binary: np.ndarray, scalars: np.ndarray) -> np.ndarray:
    planes = np.zeros((INPUT_PLANES, 8, 8), dtype=np.float32)
    planes[:110] = np.unpackbits(binary, axis=1, count=64, bitorder="little").reshape(110, 8, 8)
    planes[110].fill(float(scalars[0]))
    planes[111].fill(float(scalars[1]))
    return planes
