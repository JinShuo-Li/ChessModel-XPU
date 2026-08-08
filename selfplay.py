from __future__ import annotations

import argparse
from pathlib import Path

import chess
import numpy as np
import torch

from chess_ai.board.encoding import encode_board
from chess_ai.board.moves import move_to_index
from chess_ai.data.format import TeacherRecord, write_shard
from chess_ai.model import ChessNetwork
from chess_ai.runtime import resolve_device
from chess_ai.search import PUCTSearch
from chess_ai.search.batched_eval import NeuralEvaluator
from chess_ai.training.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", required=True); parser.add_argument("--output", required=True); parser.add_argument("--device", choices=("cpu", "cuda"), required=True); parser.add_argument("--games", type=int, default=1); parser.add_argument("--simulations", type=int, default=400); parser.add_argument("--leaf-batch-size", type=int, default=64); parser.add_argument("--temperature", type=float, default=1.0); parser.add_argument("--max-plies", type=int, default=300)
    args = parser.parse_args(); state = torch.load(args.checkpoint, map_location="cpu", weights_only=False); model = ChessNetwork(**state["architecture"]); load_checkpoint(args.checkpoint, model); device = resolve_device(args.device); model.to(device); search = PUCTSearch(NeuralEvaluator(model, device), args.simulations, args.leaf_batch_size)
    records = []
    for game_id in range(args.games):
        board = chess.Board(); game_records = []
        while not board.is_game_over(claim_draw=True) and board.ply() < args.max_plies:
            result = search.search(board); moves = list(result.visits); visits = np.array([result.visits[m] for m in moves], dtype=np.float64); probs = visits ** (1 / max(args.temperature, 1e-6)); probs /= probs.sum(); game_records.append((board.turn, TeacherRecord(encode_board(board), np.array([move_to_index(board, m) for m in moves]), probs.astype(np.float32), np.array([0, 1, 0], dtype=np.float32), metadata={"fen": board.fen(), "source": "selfplay", "game_id": game_id}))); board.push(np.random.choice(moves, p=probs))
        outcome = board.outcome(claim_draw=True)
        for color, record in game_records:
            if outcome is None or outcome.winner is None: record.wdl[:] = (0, 1, 0)
            elif outcome.winner == color: record.wdl[:] = (1, 0, 0)
            else: record.wdl[:] = (0, 0, 1)
            records.append(record)
    write_shard(Path(args.output) / "selfplay-00000.npz", records, {"games": args.games, "split_unit": "game"}); print(f"wrote {len(records)} positions from {args.games} games")


if __name__ == "__main__": main()

