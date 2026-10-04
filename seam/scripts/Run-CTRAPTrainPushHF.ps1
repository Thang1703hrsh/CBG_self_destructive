<#
Train CTRAP for the requested optimizer steps on one L40S (2 CPU, 16 GiB RAM),
then upload the merged full model and LoRA adapter to a NEW private HF repo.
No attack or utility evaluation is run. Detached execution survives PC shutdown.

Example:
  powershell -ExecutionPolicy Bypass -File .\scripts\Run-CTRAPTrainPushHF.ps1 `
    -Seed 42 -HfRepo ducthang1703/ctrap-llama2-7b-seed42-step500
#>
[CmdletBinding()]
param(
    [ValidateRange(0, 2147483647)][int]$Seed = 42,
    [ValidateRange(1, 2147483647)][int]$MaxSteps = 500,
    [Parameter(Mandatory)][ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$')][string]$HfRepo,
    [string]$RunId = '',
    [string]$ModalProfile = '',
    [switch]$UploadOnly,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$runner = Join-Path $repoRoot 'scripts\modal_ctrap_l40s.py'
if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) {
    throw "Missing Modal runner: $runner"
}
if ($UploadOnly -and -not $RunId) {
    throw '-UploadOnly requires -RunId from the original CTRAP train-and-push run.'
}
if ($RunId -and $RunId -notmatch '^[A-Za-z0-9_-]{1,64}$') {
    throw 'RunId must contain 1-64 letters, digits, _ or -.'
}
if (-not $RunId) {
    $RunId = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 8)
}
$commandArgs = @('-X', 'utf8', '-m', 'modal', 'run')
if ($ModalProfile) {
    $commandArgs += @('--profile', $ModalProfile)
}
$entrypoint = if ($UploadOnly) { 'scripts/modal_ctrap_l40s.py::retry_ctrap_push_main' }
              else { 'scripts/modal_ctrap_l40s.py::train_main' }
$commandArgs += @('--detach', $entrypoint,
    '--seed', "$Seed", '--run-id', $RunId, '--hf-repo', $HfRepo)
if (-not $UploadOnly) {
    $commandArgs += @('--max-steps', "$MaxSteps")
}

Write-Host "CTRAP train-and-push run ID: $RunId"
if ($UploadOnly) {
    Write-Host 'Retry HF upload only from the existing 500-step checkpoint; no GPU training.'
}
else {
    Write-Host "Train only: $MaxSteps steps, one L40S worker with 2 CPU / 16 GiB RAM."
}
Write-Host 'A small CPU-only coordinator remains active; no attack/utility evaluation.'
Write-Host "Private HF repo: https://huggingface.co/$HfRepo"
Write-Host "Remote results: defense-compare-results/ctrap-l40s-seed$Seed-$RunId"
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
Write-Host 'Submitted. You may shut down this PC; check Modal status and HF upload later.'
