from __future__ import annotations

import argparse

import torch

from chess_ai.model import ChessNetwork
from chess_ai.runtime import resolve_device


def run(device_name="cuda"):
    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA build:", torch.version.cuda or "none")
    print("CUDA available:", torch.cuda.is_available())
    if device_name == "cuda":
        device = resolve_device("cuda")
        print("CUDA device count:", torch.cuda.device_count())
        print("CUDA device:", torch.cuda.get_device_name(0))
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
    parser = argparse.ArgumentParser(); parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda"); args = parser.parse_args(); run(args.device)


if __name__ == "__main__": main()

