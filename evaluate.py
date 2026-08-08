from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from chess_ai.data import TeacherDataset
from chess_ai.evaluation.metrics import neural_metrics
from chess_ai.model import ChessNetwork
from chess_ai.runtime import resolve_device
from chess_ai.training.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Validate neural metrics on teacher shards"); parser.add_argument("--checkpoint", required=True); parser.add_argument("--dataset", required=True); parser.add_argument("--device", choices=("cpu", "cuda"), required=True); parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args(); state = torch.load(args.checkpoint, map_location="cpu", weights_only=False); model = ChessNetwork(**state["architecture"]); load_checkpoint(args.checkpoint, model); device = resolve_device(args.device); model.to(device)
    metrics = neural_metrics(model, DataLoader(TeacherDataset(args.dataset), batch_size=args.batch_size), device); print(json.dumps(metrics, indent=2))


if __name__ == "__main__": main()

