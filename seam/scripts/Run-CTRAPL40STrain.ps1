<#
Submit CTRAP defense training only: one L40S worker, 2 CPU, 16 GiB RAM.
The detached Modal job continues after this PC shuts down. Default: 500 steps.

Example:
  powershell -ExecutionPolicy Bypass -File .\scripts\Run-CTRAPL40STrain.ps1 -Seed 42
  powershell -ExecutionPolicy Bypass -File .\scripts\Run-CTRAPL40STrain.ps1 -Seed 42 -DryRun

Requires Modal credentials, the huggingface-secret Modal secret containing
HF_TOKEN, and Llama-2-7b-chat-hf access. This does not perform evaluation.
#>
[CmdletBinding()]
param(
    [ValidateRange(0, 2147483647)][int]$Seed = 42,
    [ValidateRange(1, 2147483647)][int]$MaxSteps = 500,
    [string]$ModalProfile = '',
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$runner = Join-Path $repoRoot 'scripts\modal_ctrap_l40s.py'
if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) {
    throw "Missing Modal runner: $runner"
}
$runId = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 8)
$commandArgs = @('-X', 'utf8', '-m', 'modal', 'run')
if ($ModalProfile) {
    $commandArgs += @('--profile', $ModalProfile)
}
$commandArgs += @('--detach', 'scripts/modal_ctrap_l40s.py::train_main',
    '--seed', "$Seed", '--run-id', $runId, '--max-steps', "$MaxSteps")

Write-Host "CTRAP L40S train-only run ID: $runId"
Write-Host "Defense: $MaxSteps optimizer steps; LoRA BF16; one L40S worker with 2 CPU / 16 GiB RAM."
Write-Host 'A small CPU-only coordinator remains active; no attack/evaluation is run.'
Write-Host "Remote results: defense-compare-results/ctrap-l40s-seed$Seed-$runId"
if ($DryRun) {
    Write-Host "Dry run only: python $($commandArgs -join ' ')"
    return
}

$pythonExecutable = Get-Command python -ErrorAction Stop
$env:PYTHONIOENCODING = 'utf-8'
Push-Location -LiteralPath $repoRoot
try {
    & $pythonExecutable.Source @commandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Modal submission failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
Write-Host 'Submitted. You may now shut down this PC. Check status: modal app list'
