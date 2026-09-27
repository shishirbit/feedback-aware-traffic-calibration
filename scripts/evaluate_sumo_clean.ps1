$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
foreach ($seed in 11, 22, 33) {
    foreach ($partition in 'calibration', 'test') {
        & .\.venv\Scripts\python.exe -m ps1.cli cache-predictions --dataset sumo --seed $seed --partition $partition
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    & .\.venv\Scripts\python.exe -m ps1.cli evaluate-clean --dataset sumo --seed $seed
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & .\.venv\Scripts\python.exe -m ps1.cli report-clean --dataset sumo --seed $seed
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
