from __future__ import annotations

import contextlib

import torch


def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not hasattr(torch, "cuda") or not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but torch.cuda is unavailable")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError("device must be explicitly 'cpu' or 'cuda'")


def autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return contextlib.nullcontext()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    return torch.autocast(device_type=device.type, dtype=dtype)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)

