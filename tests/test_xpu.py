import pytest
import torch

from chess_ai.model import ChessNetwork


@pytest.mark.skipif(not hasattr(torch, "xpu") or not torch.xpu.is_available(), reason="real Intel XPU unavailable")
def test_real_xpu_forward_backward():
    device = torch.device("xpu"); model = ChessNetwork(blocks=1, channels=8, se=False).to(device); x = torch.randn(2, 112, 8, 8, device=device)
    output = model(x); (output["policy"].mean() + output["wdl"].mean()).backward(); torch.xpu.synchronize()
    assert output["policy"].device.type == "xpu"
