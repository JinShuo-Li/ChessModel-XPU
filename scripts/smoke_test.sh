#!/usr/bin/env bash
# Usage: ./scripts/smoke_test.sh <path-to-stockfish-binary>
# Runs the bounded end-to-end CUDA smoke pipeline and removes tmp/smoke on exit.
set -euo pipefail

Stockfish="${1:?usage: smoke_test.sh <stockfish-binary>}"
ProjectRoot="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SmokeRoot="$ProjectRoot/tmp/smoke"

cleanup() { rm -rf "$SmokeRoot"; }
trap cleanup EXIT

cd "$ProjectRoot"
conda run -n chessmodel python -m chess_ai.smoke \
    --stockfish "$Stockfish" \
    --workdir "$SmokeRoot" \
    --config configs/smoke.yaml
