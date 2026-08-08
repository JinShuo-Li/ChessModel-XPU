#!/usr/bin/env bash
# Re-runnable Linux setup: creates the `chessmodel` conda env, installs torch
# only from the official CUDA index, installs the remaining dependencies, and
# executes real CUDA compute. Adjust the --index-url to match the target
# machine's driver/CUDA version if cu126 is not appropriate.
set -euo pipefail

EnvironmentName="chessmodel"

if ! conda env list | grep -q "^${EnvironmentName}[[:space:]]"; then
    conda create -n "$EnvironmentName" python=3.11 -y
fi
conda run -n "$EnvironmentName" python -m pip install --upgrade pip
conda run -n "$EnvironmentName" python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
conda run -n "$EnvironmentName" python -m pip install -r requirements.txt
conda run -n "$EnvironmentName" python -m chess_ai.diagnostics --device cuda
