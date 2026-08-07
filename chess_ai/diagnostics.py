from __future__ import annotations

import argparse

import torch

from chess_ai.model import ChessNetwork
from chess_ai.runtime import resolve_device


def run(device_name="xpu"):
    print("PyTorch:", torch.__version__)
    print("PyTorch XPU build:", hasattr(torch, "xpu"))
    available = hasattr(torch, "xpu") and torch.xpu.is_available()
    print("XPU available:", available)
    if device_name == "xpu":
        device = resolve_device("xpu")
        print("XPU count:", torch.xpu.device_count())
        print("XPU device:", torch.xpu.get_device_name(0))
    else:
        device = resolve_device("cpu")
    tensor = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
    result = tensor @ tensor.T
    print("Tensor compute:", float(result.sum().cpu()))
    model = ChessNetwork(blocks=1, channels=8, se=False).to(device)
    inputs = torch.randn(2, 112, 8, 8, device=device)
    outputs = model(inputs); loss = outputs["policy"].mean() + outputs["wdl"].mean(); loss.backward()
    print("Neural forward/backward: PASS", tuple(outputs["policy"].shape), tuple(outputs["wdl"].shape))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--device", choices=("cpu", "xpu"), default="xpu"); args = parser.parse_args(); run(args.device)


if __name__ == "__main__": main()

