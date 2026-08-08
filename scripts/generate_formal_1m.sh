#!/usr/bin/env bash
# Generates the formal 1M-position teacher dataset (plus 50k validation) using
# Stockfish, then verifies every shard checksum, position total, and shard
# total. Defaults mirror the original Windows script; every value can be
# overridden with an environment variable, e.g.:
#   STOCKFISH=path ./scripts/generate_formal_1m.sh
set -euo pipefail

Stockfish="${STOCKFISH:-tools/stockfish/unpacked/stockfish/stockfish-ubuntu-x86-64-avx2}"
TrainPgn="${TRAIN_PGN:-datasets/formal_train.pgn}"
ValidationPgn="${VALIDATION_PGN:-datasets/formal_validation.pgn}"
TrainOutput="${TRAIN_OUTPUT:-data/formal_1m_train}"
ValidationOutput="${VALIDATION_OUTPUT:-data/formal_50k_validation}"
TrainPositions="${TRAIN_POSITIONS:-1000000}"
ValidationPositions="${VALIDATION_POSITIONS:-50000}"
Nodes="${NODES:-10000}"
MultiPv="${MULTIPV:-8}"
ShardSize="${SHARD_SIZE:-4096}"

ProjectRoot="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ProjectRoot"

if [[ "$TrainPgn" == "$ValidationPgn" ]]; then
    echo "Training and validation PGNs must be different files split at game level." >&2
    exit 1
fi
for dir in "$TrainOutput" "$ValidationOutput"; do
    if [[ -e "$dir" ]]; then
        echo "Training output already exists: $dir" >&2
        exit 1
    fi
done

conda run -n chessmodel python generate_teacher_data.py \
    --config configs/formal_1m.yaml \
    --stockfish "$Stockfish" \
    --output "$TrainOutput" \
    --positions "$TrainPositions" \
    --nodes "$Nodes" \
    --multipv "$MultiPv" \
    --shard-size "$ShardSize" \
    --pgn "$TrainPgn"

conda run -n chessmodel python generate_teacher_data.py \
    --config configs/formal_1m.yaml \
    --stockfish "$Stockfish" \
    --output "$ValidationOutput" \
    --positions "$ValidationPositions" \
    --nodes "$Nodes" \
    --multipv "$MultiPv" \
    --shard-size "$ShardSize" \
    --pgn "$ValidationPgn"

TrainShards="$(find "$TrainOutput" -name '*.npz' -type f | wc -l)"
ValidationShards="$(find "$ValidationOutput" -name '*.npz' -type f | wc -l)"
ExpectedTrainShards=$(( (TrainPositions + ShardSize - 1) / ShardSize ))
ExpectedValidationShards=$(( (ValidationPositions + ShardSize - 1) / ShardSize ))
if [[ "$TrainShards" -ne "$ExpectedTrainShards" ]]; then
    echo "Expected $ExpectedTrainShards training shards, found $TrainShards" >&2
    exit 1
fi
if [[ "$ValidationShards" -ne "$ExpectedValidationShards" ]]; then
    echo "Expected $ExpectedValidationShards validation shards, found $ValidationShards" >&2
    exit 1
fi

conda run -n chessmodel python scripts/verify_teacher_dataset.py \
    --dataset "$TrainOutput" \
    --expected-positions "$TrainPositions" \
    --expected-shards "$ExpectedTrainShards"

conda run -n chessmodel python scripts/verify_teacher_dataset.py \
    --dataset "$ValidationOutput" \
    --expected-positions "$ValidationPositions" \
    --expected-shards "$ExpectedValidationShards"

echo "Formal teacher data complete: $TrainPositions training positions in $TrainShards shards; $ValidationPositions validation positions in $ValidationShards shards."
