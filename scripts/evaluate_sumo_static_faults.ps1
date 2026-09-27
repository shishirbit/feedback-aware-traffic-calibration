$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
foreach ($modelSeed in 11, 22, 33) {
    foreach ($scenario in 'C3', 'C4') {
        foreach ($faultSeed in 101, 202, 303) {
            & .\.venv\Scripts\python.exe -m ps1.cli cache-faulted --dataset sumo --model-seed $modelSeed --scenario $scenario --fault-seed $faultSeed
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
            & .\.venv\Scripts\python.exe -m ps1.cli evaluate-static-fault --dataset sumo --model-seed $modelSeed --scenario $scenario --fault-seed $faultSeed
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        }
    }
}
