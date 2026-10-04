<#
Runs one SEAM main-result sweep on Modal, detached from this PC.
Corrected training uses FP32 weights and CPU offload on one H100 with 320 GiB RAM.
Use -TrainOnly -TrainSteps 2 for an operational check before a full 500-step run.
The official SEAM README accepts the public RepNoise refusal dataset. This
does not prove the refusals came from GPT-4o as described in the paper.

Example:
  powershell -ExecutionPolicy Bypass -File .\scripts\Run-SEAMFull.ps1 -Seed 42

The command exits after Modal accepts the job. It is then safe to shut down
the local PC; the job executes in Modal. It does NOT run 20 independent seeds.
#>
[CmdletBinding()]
param(
    [ValidateRange(0, 2147483647)][int]$Seed = 42,
    [ValidateRange(0, 1000000)][int]$TestSize = 100,
    [string]$ModalProfile = '',
    [switch]$TrainOnly,
    [ValidateRange(1, 500)][int]$TrainSteps = 500,
    [string]$EvalConfigPath = '',
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$python = (Get-Command python -ErrorAction Stop).Source
if ($TrainSteps -ne 500 -and -not $TrainOnly) { throw 'Short training requires -TrainOnly.' }
if ($EvalConfigPath -and -not $TrainOnly) { throw '-EvalConfigPath requires -TrainOnly.' }
$runId = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 8)
$env:PYTHONIOENCODING = 'utf-8'
$runDirectory = "seam-full-seed$Seed-$runId"
$commandArgs = @('-X', 'utf8', '-m', 'modal', 'run')
if ($ModalProfile) { $commandArgs += @('--profile', $ModalProfile) }
$commandArgs += @('--detach', 'scripts/modal_seam_full.py::main', '--seed', "$Seed",
    '--run-id', $runId, '--test-size', "$TestSize", '--train-steps', "$TrainSteps")
if ($TrainOnly) { $commandArgs += '--train-only' }

Write-Host "SEAM run ID: $runId"
Write-Host 'Data: public BeaverTails-with-refusals from RepNoise (official SEAM README option 1).'
Write-Host "Defense: $TrainSteps steps; FP32 on one H100, CPU offload with 320 GiB requested RAM."
Write-Host "Train only: $TrainOnly. CPU offload can be substantially slower than the old BF16 run."
Write-Host 'This is one seed, not the paper table averaged over 20 seeds.'
Write-Host 'The command will return when Modal accepts the detached job.'
if ($DryRun) {
    Write-Host "Dry run: python $($commandArgs -join ' ')"
    return
}

Push-Location -LiteralPath $repoRoot
try {
    if ($TrainOnly) {
        if (-not $EvalConfigPath) { $EvalConfigPath = "config/defense_eval_$runDirectory.yaml" }
        & $python -X utf8 'scripts/create_seam_eval_config.py' --template 'config/defense_eval.yaml' `
            --output $EvalConfigPath --run-dir $runDirectory --steps $TrainSteps --seed $Seed
        if ($LASTEXITCODE -ne 0) { throw 'Failed to create the new checkpoint eval config.' }
    }
    & $python @commandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Modal submission failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

Write-Host "Submitted. Remote results: seam-full-results/seam-full-seed$Seed-$runId"
Write-Host "Status file: seam-full-results/$runDirectory/status.json"
if ($TrainOnly) {
    Write-Host "After training status is complete, eval config: $EvalConfigPath"
    Write-Host "Checkpoint: seam-full-results/$runDirectory/defense/checkpoint-$TrainSteps"
}
