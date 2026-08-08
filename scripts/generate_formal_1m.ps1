param(
    [string]$Stockfish = ".\tools\stockfish\unpacked\stockfish\stockfish-windows-x86-64-avx2.exe",
    [string]$TrainPgn = ".\datasets\formal_train.pgn",
    [string]$ValidationPgn = ".\datasets\formal_validation.pgn",
    [string]$TrainOutput = ".\data\formal_1m_train",
    [string]$ValidationOutput = ".\data\formal_50k_validation",
    [int]$TrainPositions = 1000000,
    [int]$ValidationPositions = 50000,
    [int]$Nodes = 10000,
    [int]$MultiPv = 8,
    [int]$ShardSize = 4096
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

$StockfishPath = (Resolve-Path -LiteralPath $Stockfish).Path
$TrainPgnPath = (Resolve-Path -LiteralPath $TrainPgn).Path
$ValidationPgnPath = (Resolve-Path -LiteralPath $ValidationPgn).Path

if ($TrainPgnPath -eq $ValidationPgnPath) {
    throw "Training and validation PGNs must be different files split at game level."
}
if (Test-Path -LiteralPath $TrainOutput) {
    throw "Training output already exists: $TrainOutput"
}
if (Test-Path -LiteralPath $ValidationOutput) {
    throw "Validation output already exists: $ValidationOutput"
}

conda run -n chessmodel python generate_teacher_data.py `
    --config configs\formal_1m.yaml `
    --stockfish $StockfishPath `
    --output $TrainOutput `
    --positions $TrainPositions `
    --nodes $Nodes `
    --multipv $MultiPv `
    --shard-size $ShardSize `
    --pgn $TrainPgnPath
if ($LASTEXITCODE -ne 0) { throw "Training data generation failed with exit code $LASTEXITCODE" }

conda run -n chessmodel python generate_teacher_data.py `
    --config configs\formal_1m.yaml `
    --stockfish $StockfishPath `
    --output $ValidationOutput `
    --positions $ValidationPositions `
    --nodes $Nodes `
    --multipv $MultiPv `
    --shard-size $ShardSize `
    --pgn $ValidationPgnPath
if ($LASTEXITCODE -ne 0) { throw "Validation data generation failed with exit code $LASTEXITCODE" }

$TrainShards = (Get-ChildItem -LiteralPath $TrainOutput -Filter "*.npz" -File).Count
$ValidationShards = (Get-ChildItem -LiteralPath $ValidationOutput -Filter "*.npz" -File).Count
$ExpectedTrainShards = [math]::Ceiling($TrainPositions / $ShardSize)
$ExpectedValidationShards = [math]::Ceiling($ValidationPositions / $ShardSize)
if ($TrainShards -ne $ExpectedTrainShards) { throw "Expected $ExpectedTrainShards training shards, found $TrainShards" }
if ($ValidationShards -ne $ExpectedValidationShards) { throw "Expected $ExpectedValidationShards validation shards, found $ValidationShards" }

conda run -n chessmodel python scripts\verify_teacher_dataset.py `
    --dataset $TrainOutput `
    --expected-positions $TrainPositions `
    --expected-shards $ExpectedTrainShards
if ($LASTEXITCODE -ne 0) { throw "Training dataset verification failed with exit code $LASTEXITCODE" }

conda run -n chessmodel python scripts\verify_teacher_dataset.py `
    --dataset $ValidationOutput `
    --expected-positions $ValidationPositions `
    --expected-shards $ExpectedValidationShards
if ($LASTEXITCODE -ne 0) { throw "Validation dataset verification failed with exit code $LASTEXITCODE" }

Write-Output "Formal teacher data complete: $TrainPositions training positions in $TrainShards shards; $ValidationPositions validation positions in $ValidationShards shards."
