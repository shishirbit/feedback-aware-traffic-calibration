$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
foreach ($seed in 11, 22, 33) {
    & .\.venv\Scripts\python.exe -m ps1.cli train --dataset sumo --seed $seed
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
