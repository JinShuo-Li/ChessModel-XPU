import pytest
import torch

from chess_ai.model import ChessNetwork


@pytest.mark.skipif(not hasattr(torch, "cuda") or not torch.cuda.is_available(), reason="real NVIDIA CUDA unavailable")
def test_real_cuda_forward_backward():
    device = torch.device("cuda"); model = ChessNetwork(blocks=1, channels=8, se=False).to(device); x = torch.randn(2, 112, 8, 8, device=device)
    output = model(x); (output["policy"].mean() + output["wdl"].mean()).backward(); torch.cuda.synchronize()
    assert output["policy"].device.type == "cuda"
