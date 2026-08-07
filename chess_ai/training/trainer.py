from __future__ import annotations

import math
import time

import torch

from chess_ai.runtime import autocast_context
from .losses import training_loss


def cosine_schedule(optimizer, warmup: int, total: int):
    def scale(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


class Trainer:
    def __init__(self, model, optimizer, scheduler, device, precision="bf16", loss_weights=None):
        self.model, self.optimizer, self.scheduler = model, optimizer, scheduler
        self.device, self.precision = device, precision
        self.loss_weights = loss_weights or {"policy": 1.0, "value": 1.0, "moves_left": 0.05}
        self.global_step = 0

    def train_step(self, batch):
        transfer_start = time.perf_counter()
        batch = {key: value.to(self.device, non_blocking=True) for key, value in batch.items()}
        transfer_s = time.perf_counter() - transfer_start
        self.optimizer.zero_grad(set_to_none=True)
        forward_start = time.perf_counter()
        with autocast_context(self.device, self.precision):
            outputs = self.model(batch["position"])
            loss, components = training_loss(outputs, batch, self.loss_weights)
        forward_s = time.perf_counter() - forward_start
        backward_start = time.perf_counter()
        loss.backward()
        backward_s = time.perf_counter() - backward_start
        optimizer_start = time.perf_counter()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        optimizer_s = time.perf_counter() - optimizer_start
        self.global_step += 1
        return {"loss": float(loss.detach().cpu()), "transfer_s": transfer_s, "forward_s": forward_s, "backward_s": backward_s, "optimizer_s": optimizer_s, **{key: float(value.cpu()) for key, value in components.items()}}

