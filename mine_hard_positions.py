from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from chess_ai.data import TeacherDataset
from chess_ai.model import ChessNetwork
from chess_ai.runtime import resolve_device
from chess_ai.training.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", required=True); parser.add_argument("--dataset", required=True); parser.add_argument("--output", required=True); parser.add_argument("--device", choices=("cpu", "xpu"), required=True); parser.add_argument("--top-k", type=int, default=10000); parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args(); state = torch.load(args.checkpoint, map_location="cpu", weights_only=False); model = ChessNetwork(**state["architecture"]); load_checkpoint(args.checkpoint, model); device = resolve_device(args.device); model.to(device).eval()
    scored, offset = [], 0
    with torch.inference_mode():
        for batch in DataLoader(TeacherDataset(args.dataset), batch_size=args.batch_size):
            target_p, target_v = batch["policy"].to(device), batch["wdl"].to(device); out = model(batch["position"].to(device)); pred_p = torch.softmax(out["policy"], 1); pred_v = torch.softmax(out["wdl"], 1)
            kl = (target_p * (target_p.clamp_min(1e-9).log() - pred_p.clamp_min(1e-9).log())).sum(1); value_error = (target_v - pred_v).abs().sum(1); entropy = -(pred_p * pred_p.clamp_min(1e-9).log()).sum(1); disagreement = (pred_p.argmax(1) != target_p.argmax(1)).float()
            score = kl + value_error + 0.05 * entropy + disagreement
            for i, (s, k, v, e, d) in enumerate(zip(score, kl, value_error, entropy, disagreement)):
                scored.append({"index": offset + i, "score": float(s.cpu()), "policy_kl": float(k.cpu()), "wdl_l1": float(v.cpu()), "entropy": float(e.cpu()), "top1_disagreement": bool(d.cpu())})
            offset += len(score)
    scored.sort(key=lambda row: row["score"], reverse=True)
    with open(args.output, "w", encoding="utf-8") as handle: json.dump(scored[:args.top_k], handle, indent=2)
    print(f"wrote {min(args.top_k, len(scored))} ranked positions to {args.output}")


if __name__ == "__main__": main()

