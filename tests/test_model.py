import torch

from chess_ai.board.moves import POLICY_SIZE
from chess_ai.model import ChessNetwork, count_parameters
from chess_ai.training.losses import soft_cross_entropy


def test_model_shapes_and_wdl():
    model = ChessNetwork(blocks=2, channels=16, se=True, se_hidden=4, moves_left=True)
    output = model(torch.randn(3, 112, 8, 8))
    assert output["policy"].shape == (3, POLICY_SIZE)
    assert output["wdl"].shape == (3, 3)
    assert output["moves_left"].shape == (3,)
    assert count_parameters(model) > 0
    assert torch.allclose(torch.softmax(output["wdl"], 1).sum(1), torch.ones(3))


def test_soft_policy_loss():
    logits = torch.tensor([[2.0, 0.0]])
    target = torch.tensor([[0.75, 0.25]])
    loss = soft_cross_entropy(logits, target)
    assert loss.isfinite() and loss > 0

