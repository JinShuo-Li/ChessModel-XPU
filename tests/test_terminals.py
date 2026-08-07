import chess

from chess_ai.search.node import Node


def test_checkmate_and_stalemate():
    mate = Node(chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"))
    stale = Node(chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"))
    assert mate.board.is_checkmate() and mate.terminal_value() == -1
    assert stale.board.is_stalemate() and stale.terminal_value() == 0


def test_insufficient_and_fifty_move():
    insufficient = Node(chess.Board("8/8/8/8/8/8/2k5/K7 w - - 0 1"))
    fifty = Node(chess.Board("8/8/8/8/8/5k2/7R/7K b - - 100 75"))
    assert insufficient.board.is_insufficient_material() and insufficient.terminal_value() == 0
    assert fifty.board.can_claim_fifty_moves() and fifty.terminal_value() == 0


def test_repetition_claim():
    board = chess.Board()
    for move in ("g1f3", "g8f6", "f3g1", "f6g8") * 2: board.push_uci(move)
    assert board.can_claim_threefold_repetition()
    assert Node(board).terminal_value() == 0

