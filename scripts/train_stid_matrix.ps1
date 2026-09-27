$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv-staeformer\Scripts\python.exe"
$trainer = Join-Path $root "scripts\run_stid_ps1.py"
$log = Join-Path $root "artifacts\stid-training.log"
$runs = @(
    @{Dataset="metr_la"; Seed=11; Batch=64; Decay="0.0003"},
    @{Dataset="metr_la"; Seed=22; Batch=64; Decay="0.0003"},
    @{Dataset="metr_la"; Seed=33; Batch=64; Decay="0.0003"},
    @{Dataset="pems_bay"; Seed=11; Batch=64; Decay="0.0001"},
    @{Dataset="pems_bay"; Seed=22; Batch=64; Decay="0.0001"},
    @{Dataset="pems_bay"; Seed=33; Batch=64; Decay="0.0001"}
)
foreach ($run in $runs) {
    $output = Join-Path $root ("artifacts\training\stid-{0}-seed{1}" -f $run.Dataset,$run.Seed)
    if (Test-Path (Join-Path $output "training_summary.json")) { continue }
    & $python $trainer --prepared (Join-Path $root ("data\prepared\{0}.npz" -f $run.Dataset)) `
        --output $output --seed $run.Seed --batch-size $run.Batch --weight-decay 0.0001 --learning-rate 0.002 `
        --epochs 100 --patience 30 --device cuda *>> $log
    if ($LASTEXITCODE -ne 0) { throw "STID training failed for $($run.Dataset) seed $($run.Seed)" }
}
@{status="completed"; completed_at=(Get-Date).ToString("o")} | ConvertTo-Json |
    Set-Content (Join-Path $root "artifacts\stid-training-matrix.json")


