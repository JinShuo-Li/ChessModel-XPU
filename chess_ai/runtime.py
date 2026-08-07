from __future__ import annotations

import contextlib

import torch


def resolve_device(name: str) -> torch.device:
    if name == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("XPU was explicitly requested but torch.xpu is unavailable")
        return torch.device("xpu")
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError("device must be explicitly 'cpu' or 'xpu'")


def autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return contextlib.nullcontext()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    return torch.autocast(device_type=device.type, dtype=dtype)


def synchronize(device: torch.device) -> None:
    if device.type == "xpu":
        torch.xpu.synchronize(device)

