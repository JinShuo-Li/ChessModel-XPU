param([Parameter(Mandatory=$true)][string]$Stockfish)
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$SmokeRoot = Join-Path $PSScriptRoot "..\tmp\smoke"
$DataRoot = Join-Path $SmokeRoot "data"
$Checkpoint = Join-Path $SmokeRoot "smoke.pt"
try {
    conda run -n chessmodel python -m chess_ai.smoke --stockfish $Stockfish --workdir $SmokeRoot --config configs\smoke.yaml
}
finally {
    if (Test-Path -LiteralPath $SmokeRoot) { Remove-Item -LiteralPath $SmokeRoot -Recurse -Force }
}
