from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.inference_mode()
def neural_metrics(model, loader, device):
    totals = {key: 0.0 for key in ("policy_cross_entropy", "policy_kl", "top1", "top3", "wdl_cross_entropy", "wdl_accuracy", "wdl_brier")}
    count = 0
    model.eval()
    for batch in loader:
        position, target_p, target_v = batch["position"].to(device), batch["policy"].to(device), batch["wdl"].to(device)
        out = model(position); masked_logits = out["policy"].masked_fill(~batch["legal_mask"].to(device), torch.finfo(out["policy"].dtype).min); logp = F.log_softmax(masked_logits, 1); pred_v = torch.softmax(out["wdl"], 1)
        batch_n = position.shape[0]; count += batch_n
        totals["policy_cross_entropy"] += float((-(target_p * logp).sum(1)).sum().cpu())
        target_log = torch.where(target_p > 0, target_p.log(), torch.zeros_like(target_p))
        totals["policy_kl"] += float((target_p * (target_log - logp)).sum().cpu())
        teacher_top = target_p.argmax(1); top3 = masked_logits.topk(3, 1).indices
        totals["top1"] += float((masked_logits.argmax(1) == teacher_top).sum().cpu())
        totals["top3"] += float((top3 == teacher_top[:, None]).any(1).sum().cpu())
        totals["wdl_cross_entropy"] += float((-(target_v * torch.log(pred_v.clamp_min(1e-9))).sum(1)).sum().cpu())
        totals["wdl_accuracy"] += float((pred_v.argmax(1) == target_v.argmax(1)).sum().cpu())
        totals["wdl_brier"] += float(((pred_v - target_v) ** 2).sum(1).sum().cpu())
    return {key: value / max(count, 1) for key, value in totals.items()}
