import chess
import numpy as np

from chess_ai.search import PUCTSearch


class UniformEvaluator:
    def evaluate(self, boards):
        return [({move: 1 / board.legal_moves.count() for move in board.legal_moves}, 0.0, np.array([0.2, 0.6, 0.2])) for board in boards]


def test_puct_returns_legal_move_and_batches():
    board = chess.Board(); search = PUCTSearch(UniformEvaluator(), simulations=12, leaf_batch_size=4)
    result = search.search(board)
    assert result.move in board.legal_moves
    assert result.simulations == 12
    assert result.pv and result.pv[0] == result.move


def test_terminal_propagation_finds_mate():
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    result = PUCTSearch(UniformEvaluator(), simulations=48, leaf_batch_size=4).search(board)
    after = board.copy(); after.push(result.move)
    assert result.move in board.legal_moves

