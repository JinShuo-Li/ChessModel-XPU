import random

import chess

from chess_ai.board.moves import POLICY_SIZE, index_to_move, legal_move_mask, legal_moves_and_indices, move_to_index


def assert_roundtrip(board):
    legal = list(board.legal_moves); ids = [move_to_index(board, move) for move in legal]
    fast_moves, fast_ids = legal_moves_and_indices(board)
    assert fast_moves == legal
    assert fast_ids.tolist() == ids
    assert len(ids) == len(set(ids))
    assert all(0 <= index < POLICY_SIZE for index in ids)
    assert {index_to_move(board, index) for index in ids} == set(legal)
    mask = legal_move_mask(board)
    assert mask.sum() == len(legal)
    assert all(mask[index] for index in ids)


def test_start_and_random_game_roundtrips():
    rng = random.Random(11); board = chess.Board()
    for _ in range(120):
        assert_roundtrip(board)
        if board.is_game_over(claim_draw=True): board = chess.Board()
        board.push(rng.choice(list(board.legal_moves)))


def test_castling_roundtrip():
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    assert chess.Move.from_uci("e1g1") in board.legal_moves
    assert chess.Move.from_uci("e1c1") in board.legal_moves
    assert_roundtrip(board)


def test_en_passant_roundtrip():
    board = chess.Board(); [board.push_uci(move) for move in ("e2e4", "a7a6", "e4e5", "d7d5")]
    assert chess.Move.from_uci("e5d6") in board.legal_moves
    assert_roundtrip(board)


def test_all_promotions_roundtrip():
    for turn, fen in ((chess.WHITE, "7k/P7/8/8/8/8/8/7K w - - 0 1"), (chess.BLACK, "k7/8/8/8/8/8/p7/7K b - - 0 1")):
        board = chess.Board(fen); assert board.turn == turn
        promotions = {move.promotion for move in board.legal_moves if move.promotion}
        assert promotions == {chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT}
        assert_roundtrip(board)
