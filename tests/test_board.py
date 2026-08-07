import chess
import numpy as np

from chess_ai.board.encoding import INPUT_PLANES, encode_board, pack_encoded, unpack_encoded


def test_encoding_shape_and_determinism():
    board = chess.Board()
    first = encode_board(board)
    assert first.shape == (INPUT_PLANES, 8, 8)
    assert first.dtype == np.float32
    np.testing.assert_array_equal(first, encode_board(board.copy(stack=True)))


def test_canonical_orientation_changes_side_plane():
    board = chess.Board(); board.push_uci("e2e4")
    encoded = encode_board(board)
    assert encoded[109].sum() == 0
    assert encoded[0, 1, 4] == 1  # Black's e7 pawn is canonical friendly pawn on e2.


def test_history_castling_ep_and_clocks():
    board = chess.Board(); board.push_uci("e2e4")
    encoded = encode_board(board)
    assert encoded[108].sum() == 1
    assert all(encoded[i].all() for i in range(104, 108))
    assert encoded[111, 0, 0] > 0


def test_pack_roundtrip():
    board = chess.Board(); board.push_uci("g1f3"); board.push_uci("g8f6")
    encoded = encode_board(board); packed, scalars = pack_encoded(encoded)
    np.testing.assert_allclose(unpack_encoded(packed, scalars), encoded, atol=3e-4)
