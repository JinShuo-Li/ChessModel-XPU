import chess
import numpy as np
import torch

from chess_ai.board.moves import POLICY_SIZE, index_to_move, legal_move_mask
from chess_ai.search import PUCTSearch
from chess_ai.search.batched_eval import NeuralEvaluator
from chess_ai.search.node import Node


class UniformEvaluator:
    def __init__(self):
        self.batch_sizes = []

    def evaluate(self, boards):
        self.batch_sizes.append(len(boards))
        return [({move: 1 / board.legal_moves.count() for move in board.legal_moves}, 0.0, np.array([0.2, 0.6, 0.2])) for board in boards]


def test_puct_returns_legal_move_and_batches():
    board = chess.Board(); evaluator = UniformEvaluator(); search = PUCTSearch(evaluator, simulations=12, leaf_batch_size=4)
    result = search.search(board)
    assert result.move in board.legal_moves
    assert result.simulations == 12
    assert result.pv and result.pv[0] == result.move
    assert max(evaluator.batch_sizes) > 1


def test_terminal_propagation_finds_mate():
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    result = PUCTSearch(UniformEvaluator(), simulations=48, leaf_batch_size=4).search(board)
    after = board.copy(); after.push(result.move)
    assert result.move in board.legal_moves


class FixedEvaluatorModel(torch.nn.Module):
    def forward(self, inputs):
        batch = len(inputs)
        policy = torch.linspace(-1.0, 1.0, POLICY_SIZE).expand(batch, -1)
        wdl = torch.tensor([0.2, 0.3, 0.5]).expand(batch, -1)
        return {"policy": policy, "wdl": wdl, "moves_left": None}


def test_neural_evaluator_matches_mask_order_and_probabilities():
    board = chess.Board()
    priors, value, wdl = NeuralEvaluator(FixedEvaluatorModel(), torch.device("cpu"), "fp32").evaluate([board])[0]
    logits = np.linspace(-1.0, 1.0, POLICY_SIZE, dtype=np.float32)
    ids = np.flatnonzero(legal_move_mask(board))
    expected_logits = logits[ids]; expected_logits -= expected_logits.max(initial=0.0)
    expected_probs = np.exp(expected_logits); expected_probs /= expected_probs.sum()

    assert list(priors) == [index_to_move(board, int(index)) for index in ids]
    np.testing.assert_allclose(list(priors.values()), expected_probs, rtol=1e-6)
    np.testing.assert_allclose(wdl, torch.softmax(torch.tensor([0.2, 0.3, 0.5]), dim=0).numpy())
    assert value == float(wdl[0] - wdl[2])


def test_children_materialize_boards_lazily():
    board = chess.Board(); root = Node(board)
    move = chess.Move.from_uci("e2e4")
    PUCTSearch._expand(root, {move: 1.0})

    child = root.children[move]
    assert child.board is None
    expected = board.copy(stack=True); expected.push(move)
    assert child.materialize_board().fen() == expected.fen()


def test_terminal_value_is_cached(monkeypatch):
    board = chess.Board(); node = Node(board); calls = 0
    real_outcome = board.outcome

    def counted_outcome(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_outcome(*args, **kwargs)

    monkeypatch.setattr(board, "outcome", counted_outcome)
    assert node.terminal_value() is None
    assert node.terminal_value() is None
    assert calls == 1
