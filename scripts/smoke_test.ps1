param([Parameter(Mandatory=$true)][string]$Stockfish)
$ErrorActionPreference = "Stop"
$SmokeRoot = Join-Path $PSScriptRoot "..\tmp\smoke"
$DataRoot = Join-Path $SmokeRoot "data"
$Checkpoint = Join-Path $SmokeRoot "smoke.pt"
try {
    conda run -n chessmodel python -m chess_ai.diagnostics --device xpu
    conda run -n chessmodel python generate_teacher_data.py --config configs\smoke.yaml --stockfish $Stockfish --output $DataRoot --positions 16 --nodes 100 --multipv 2 --shard-size 16
    conda run -n chessmodel python train.py --config configs\smoke.yaml --dataset $DataRoot --device xpu --max-steps 4 --output $Checkpoint
    conda run -n chessmodel python train.py --config configs\smoke.yaml --dataset $DataRoot --device xpu --resume $Checkpoint --max-steps 5 --output $Checkpoint
    conda run -n chessmodel python evaluate.py --checkpoint $Checkpoint --dataset $DataRoot --device xpu --batch-size 8
    $Uci = "uci`nisready`nposition startpos`ngo`nquit`n"
    $Uci | conda run -n chessmodel python -m chess_ai.uci.engine --checkpoint $Checkpoint --device xpu --simulations 4 --leaf-batch-size 2
    Write-Host "END-TO-END SMOKE: PASS"
}
finally {
    if (Test-Path -LiteralPath $SmokeRoot) { Remove-Item -LiteralPath $SmokeRoot -Recurse -Force }
}
