from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from chess_ai.config import load_config
from chess_ai.data import TeacherDataset
from chess_ai.model import ChessNetwork, count_parameters
from chess_ai.runtime import resolve_device
from chess_ai.training.checkpoint import load_checkpoint, save_checkpoint
from chess_ai.training.trainer import Trainer, cosine_schedule


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--dataset", required=True); parser.add_argument("--device", choices=("cpu", "xpu"), required=True); parser.add_argument("--resume"); parser.add_argument("--output", default="checkpoints/latest.pt"); parser.add_argument("--max-steps", type=int); parser.add_argument("--logdir")
    args = parser.parse_args(); cfg = load_config(args.config); device = resolve_device(args.device)
    dataset = TeacherDataset(args.dataset); training = cfg["training"]; loader = DataLoader(dataset, batch_size=training["batch_size"], shuffle=True, num_workers=0)
    model = ChessNetwork(**cfg["model"]).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    total = args.max_steps or training.get("steps") or training["epochs"] * len(loader); scheduler = cosine_schedule(optimizer, training["warmup_steps"], total)
    trainer = Trainer(model, optimizer, scheduler, device, cfg["runtime"]["precision"], cfg["loss"]); epoch0 = 0; writer = SummaryWriter(args.logdir) if args.logdir else None
    if args.resume:
        state = load_checkpoint(args.resume, model, optimizer, scheduler, map_location=device); trainer.global_step = state["global_step"]; epoch0 = state["epoch"]
    print(f"parameters={count_parameters(model):,} device={device} positions={len(dataset)}")
    configured_epochs = training.get("epochs", 1)
    epoch = epoch0
    while trainer.global_step < total and (args.max_steps is not None or epoch < configured_epochs):
        model.train()
        for batch in loader:
            metrics = trainer.train_step(batch)
            if trainer.global_step % 10 == 0 or total <= 10: print({"step": trainer.global_step, **metrics})
            if writer:
                for name, value in metrics.items(): writer.add_scalar(f"train/{name}", value, trainer.global_step)
            if trainer.global_step >= total: break
        epoch += 1
        save_checkpoint(args.output, model, optimizer, scheduler, step=trainer.global_step, epoch=epoch, config=cfg)
    if writer: writer.close()


if __name__ == "__main__": main()
