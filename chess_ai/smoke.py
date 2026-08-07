from __future__ import annotations

import argparse
import io
from itertools import cycle
from pathlib import Path

import chess
import torch
from torch.utils.data import DataLoader

from chess_ai.config import load_config
from chess_ai.data import TeacherDataset
from chess_ai.data.sampler import random_positions
from chess_ai.diagnostics import run as diagnostic
from chess_ai.model import ChessNetwork, count_parameters
from chess_ai.runtime import resolve_device
from chess_ai.search import PUCTSearch
from chess_ai.search.batched_eval import NeuralEvaluator
from chess_ai.teacher.generation import generate
from chess_ai.teacher.stockfish import StockfishTeacher
from chess_ai.training.checkpoint import load_checkpoint, save_checkpoint
from chess_ai.training.trainer import Trainer, cosine_schedule
from chess_ai.uci.engine import loop as uci_loop


def main():
    parser = argparse.ArgumentParser(description="Bounded end-to-end Intel XPU smoke pipeline")
    parser.add_argument("--stockfish", required=True); parser.add_argument("--workdir", required=True); parser.add_argument("--config", default="configs/smoke.yaml")
    args = parser.parse_args(); cfg = load_config(args.config); root = Path(args.workdir); data_root = root / "data"; checkpoint = root / "smoke.pt"
    diagnostic("xpu"); device = resolve_device("xpu"); torch.manual_seed(cfg["seed"])
    sf = cfg["stockfish"]
    with StockfishTeacher(args.stockfish, threads=sf["threads"], hash_mb=sf["hash_mb"], multipv=2, nodes=100, temperature=sf["temperature"]) as teacher:
        generation = generate(random_positions(16, cfg["seed"]), teacher, data_root, shard_size=16, metadata={"purpose": "smoke"})
        print({"stockfish": teacher.version, **generation})
    dataset = TeacherDataset(data_root); print({"dataset_positions": len(dataset), **dataset.profile})
    loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=True); batches = cycle(loader)
    model = ChessNetwork(**cfg["model"]).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"]); scheduler = cosine_schedule(optimizer, 1, 5); trainer = Trainer(model, optimizer, scheduler, device, cfg["runtime"]["precision"], cfg["loss"])
    print(f"parameters={count_parameters(model):,} device={device}")
    for _ in range(4): print({"step": trainer.global_step + 1, **trainer.train_step(next(batches))})
    save_checkpoint(checkpoint, model, optimizer, scheduler, step=trainer.global_step, epoch=2, config=cfg)
    resumed = ChessNetwork(**cfg["model"]).to(device); resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"]); resumed_scheduler = cosine_schedule(resumed_optimizer, 1, 5); state = load_checkpoint(checkpoint, resumed, resumed_optimizer, resumed_scheduler, map_location=device)
    resumed_trainer = Trainer(resumed, resumed_optimizer, resumed_scheduler, device, cfg["runtime"]["precision"], cfg["loss"]); resumed_trainer.global_step = state["global_step"]; print({"resumed_from_step": resumed_trainer.global_step, "step": 5, **resumed_trainer.train_step(next(batches))})
    search = PUCTSearch(NeuralEvaluator(resumed, device, cfg["runtime"]["precision"]), simulations=4, leaf_batch_size=2, cpuct=cfg["search"]["cpuct"]); board = chess.Board(); result = search.search(board); assert result.move in board.legal_moves; print({"puct_move": result.move.uci(), "legal": True, "simulations": result.simulations, "pv": [move.uci() for move in result.pv]})
    commands = io.StringIO("uci\nisready\nposition startpos\ngo\nquit\n"); replies = io.StringIO(); uci_loop(search, commands, replies); transcript = replies.getvalue(); assert "uciok" in transcript and "readyok" in transcript and "bestmove " in transcript; print(transcript, end="")
    print("END-TO-END SMOKE: PASS")


if __name__ == "__main__": main()
