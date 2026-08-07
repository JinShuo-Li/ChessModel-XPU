from __future__ import annotations

import random
import subprocess
from pathlib import Path

import numpy as np
import torch


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted"


def save_checkpoint(path, model, optimizer=None, scheduler=None, *, step=0, epoch=0, config=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"model": model.state_dict(), "optimizer": None if optimizer is None else optimizer.state_dict(), "scheduler": None if scheduler is None else scheduler.state_dict(), "global_step": step, "epoch": epoch, "config": config or {}, "architecture": model.architecture, "git_commit": git_commit(), "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}}
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu"):
    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    # The global default RNG is CPU-based even when checkpoint tensors were
    # remapped directly to XPU for model/optimizer resume.
    torch.set_rng_state(state["rng"]["torch"].cpu())
    return state
