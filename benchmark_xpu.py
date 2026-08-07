from __future__ import annotations

import argparse
import json
import time

import torch

from chess_ai.config import load_config
from chess_ai.model import ChessNetwork, count_parameters
from chess_ai.runtime import autocast_context, resolve_device, synchronize


def bench(config, batch, steps, precision, layout, compile_model):
    cfg = load_config(config); device = resolve_device("xpu")
    model = ChessNetwork(**cfg["model"]).to(device); model.train()
    if layout == "channels_last": model = model.to(memory_format=torch.channels_last)
    if compile_model: model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    shape = (batch, 112, 8, 8)
    try:
        x = torch.randn(shape, device=device); x = x.contiguous(memory_format=torch.channels_last) if layout == "channels_last" else x
        for _ in range(2):
            with autocast_context(device, precision): out = model(x); loss = out["policy"].mean() + out["wdl"].mean()
            loss.backward(); optimizer.zero_grad(set_to_none=True)
        synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            for _ in range(steps):
                with autocast_context(device, precision): model(x)
        synchronize(device); forward_s = time.perf_counter() - started
        started = time.perf_counter()
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, precision): out = model(x); loss = out["policy"].mean() + out["wdl"].mean()
            loss.backward(); optimizer.step()
        synchronize(device); train_s = time.perf_counter() - started
        memory = None
        if hasattr(torch.xpu, "memory_allocated"): memory = torch.xpu.memory_allocated() / 2**20
        return {"status": "ok", "batch": batch, "parameters": count_parameters(model), "precision": precision, "layout": layout, "compile": compile_model, "forward_positions_s": batch * steps / forward_s, "training_positions_s": batch * steps / train_s, "step_latency_ms": train_s * 1000 / steps, "memory_mib": memory}
    except RuntimeError as error:
        if "memory" in str(error).lower() or "alloc" in str(error).lower():
            return {"status": "oom", "batch": batch, "error": str(error)}
        raise


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--batches", type=int, nargs="+", default=[128, 256, 512, 1024, 2048]); parser.add_argument("--steps", type=int, default=5); parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16"); parser.add_argument("--layout", choices=("contiguous", "channels_last"), default="contiguous"); parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    for batch in args.batches:
        result = bench(args.config, batch, args.steps, args.precision, args.layout, args.compile); print(json.dumps(result), flush=True)
        if result["status"] == "oom": break


if __name__ == "__main__": main()

