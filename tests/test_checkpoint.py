import torch

from chess_ai.model import ChessNetwork
from chess_ai.training.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_save_resume(tmp_path):
    model = ChessNetwork(blocks=1, channels=8, se=False); optimizer = torch.optim.AdamW(model.parameters()); scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    original = next(model.parameters()).detach().clone(); path = tmp_path / "model.pt"
    save_checkpoint(path, model, optimizer, scheduler, step=3, epoch=2, config={"test": True})
    with torch.no_grad(): next(model.parameters()).add_(1)
    state = load_checkpoint(path, model, optimizer, scheduler)
    assert state["global_step"] == 3 and state["epoch"] == 2
    assert torch.equal(next(model.parameters()), original)

