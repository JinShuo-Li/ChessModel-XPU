from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, legal_mask: torch.Tensor | None = None) -> torch.Tensor:
    if legal_mask is not None:
        logits = logits.masked_fill(~legal_mask, torch.finfo(logits.dtype).min)
    return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def training_loss(outputs, batch, weights):
    policy = soft_cross_entropy(outputs["policy"], batch["policy"], batch.get("legal_mask"))
    value = soft_cross_entropy(outputs["wdl"], batch["wdl"])
    moves = torch.zeros((), device=policy.device)
    if outputs["moves_left"] is not None:
        moves = F.huber_loss(outputs["moves_left"], batch["moves_left"])
    total = weights.get("policy", 1.0) * policy + weights.get("value", 1.0) * value + weights.get("moves_left", 0.05) * moves
    return total, {"policy": policy.detach(), "value": value.detach(), "moves_left": moves.detach()}
