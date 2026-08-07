$ErrorActionPreference = "Stop"
$EnvironmentName = "chessmodel"

$Exists = conda env list | Select-String -Pattern "^$EnvironmentName\s"
if (-not $Exists) {
    conda create -n $EnvironmentName python=3.11 -y
}
conda run -n $EnvironmentName python -m pip install --upgrade pip
conda run -n $EnvironmentName python -m pip install torch --index-url https://download.pytorch.org/whl/xpu
conda run -n $EnvironmentName python -m pip install -r requirements.txt
conda run -n $EnvironmentName python -m chess_ai.diagnostics --device xpu

