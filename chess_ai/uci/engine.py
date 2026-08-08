from __future__ import annotations

import argparse
import sys

import chess
import torch

from chess_ai.model import ChessNetwork
from chess_ai.runtime import resolve_device
from chess_ai.search import PUCTSearch
from chess_ai.search.batched_eval import NeuralEvaluator
from chess_ai.training.checkpoint import load_checkpoint


def build_engine(checkpoint, device_name, simulations, leaf_batch_size):
    device = resolve_device(device_name)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = ChessNetwork(**state["architecture"]); load_checkpoint(checkpoint, model); model.to(device)
    return PUCTSearch(NeuralEvaluator(model, device), simulations, leaf_batch_size)


def loop(search, input_stream=None, output_stream=None):
    input_stream = input_stream or sys.stdin
    output_stream = output_stream or sys.stdout
    board = chess.Board()
    for raw in input_stream:
        parts = raw.strip().split()
        if not parts: continue
        command = parts[0]
        if command == "uci": print("id name ChessModel-XPU\nid author JinShuo-Li\nuciok", file=output_stream, flush=True)
        elif command == "isready": print("readyok", file=output_stream, flush=True)
        elif command == "ucinewgame": board = chess.Board()
        elif command == "position":
            if parts[1] == "startpos": board = chess.Board(); offset = 2
            else:
                moves_at = parts.index("moves") if "moves" in parts else len(parts)
                board = chess.Board(" ".join(parts[2:moves_at])); offset = moves_at
            if offset < len(parts) and parts[offset] == "moves":
                for uci in parts[offset + 1:]: board.push_uci(uci)
        elif command == "go":
            result = search.search(board)
            print(f"info nodes {result.simulations} time {int(result.elapsed_s*1000)} pv {' '.join(m.uci() for m in result.pv)}", file=output_stream, flush=True)
            print(f"bestmove {result.move.uci()}", file=output_stream, flush=True)
        elif command == "stop": pass
        elif command == "quit": break


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", required=True); parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda"); parser.add_argument("--simulations", type=int, default=400); parser.add_argument("--leaf-batch-size", type=int, default=64)
    args = parser.parse_args(); loop(build_engine(args.checkpoint, args.device, args.simulations, args.leaf_batch_size))


if __name__ == "__main__": main()
