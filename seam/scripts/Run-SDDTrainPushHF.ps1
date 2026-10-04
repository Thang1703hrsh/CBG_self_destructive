<#
Train SDD for exactly 500 optimizer steps from a completed data preflight,
then upload the full model to a NEW private Hugging Face model repository.
No attack or utility evaluation is run. Detached Modal execution survives PC shutdown.

Example:
  powershell -ExecutionPolicy Bypass -File .\scripts\Run-SDDTrainPushHF.ps1 `
    -Seed 42 -AnswerPool AlpacaOnly -PreparedRunId RUN_ID `
    -HfRepo ducthang1703/sdd-llama2-7b-seed42-step500
#>
[CmdletBinding()]
param(
    [ValidateRange(0, 2147483647)][int]$Seed = 42,
    [ValidateSet('AlpacaOnly', 'AlpacaLima')][string]$AnswerPool = 'AlpacaOnly',
    [Parameter(Mandatory)][ValidatePattern('^[A-Za-z0-9_-]{1,64}$')][string]$PreparedRunId,
    [Parameter(Mandatory)][ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$')][string]$HfRepo,
    [string]$ModalProfile = '',
    [switch]$UploadOnly,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$runner = Join-Path $repoRoot 'scripts\modal_sdd_full.py'
if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) {
    throw "Missing Modal runner: $runner"
}
$commandArgs = @('-X', 'utf8', '-m', 'modal', 'run')
if ($ModalProfile) {
    $commandArgs += @('--profile', $ModalProfile)
}
$entrypoint = if ($UploadOnly) { 'scripts/modal_sdd_full.py::retry_sdd_push_main' }
              else { 'scripts/modal_sdd_full.py::train_push_main' }
$commandArgs += @('--detach', $entrypoint,
    '--seed', "$Seed", '--run-id', $PreparedRunId, '--hf-repo', $HfRepo)
if (-not $UploadOnly) {
    $commandArgs += @('--answer-pool', $AnswerPool)
}

Write-Host "SDD train-and-push run ID: $PreparedRunId"
if ($UploadOnly) {
    Write-Host 'Retry HF upload only from the existing 500-step checkpoint; no GPU training.'
}
else {
    Write-Host 'Train only: 500 steps, one H100. No attack/utility evaluation.'
}
if ($AnswerPool -eq 'AlpacaOnly') {
    Write-Warning 'AlpacaOnly differs from the paper Alpaca+LIMA answer pool.'
}
Write-Host "Private HF repo: https://huggingface.co/$HfRepo"
Write-Host "Remote results: defense-compare-results/sdd-seed$Seed-$PreparedRunId"
if ($DryRun) {
    Write-Host "Dry run only: python $($commandArgs -join ' ')"
    return
}

$pythonExecutable = Get-Command python -ErrorAction Stop
$env:PYTHONIOENCODING = 'utf-8'
if (-not $UploadOnly) {
    $statusFile = Join-Path ([System.IO.Path]::GetTempPath()) (
        'sdd-preflight-status-' + [guid]::NewGuid().ToString('N') + '.json'
    )
    $checkArgs = @('-X', 'utf8', '-m', 'modal', 'volume', 'get',
        'defense-compare-results', "sdd-seed$Seed-$PreparedRunId/status.json", $statusFile)
    if ($ModalProfile) {
        $checkArgs += @('--profile', $ModalProfile)
    }
    try {
        & $pythonExecutable.Source @checkArgs
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $statusFile -PathType Leaf)) {
            throw 'SDD preflight status not found. Wait for Run-SDDPreflight.ps1 to finish.'
        }
        $preflight = Get-Content -LiteralPath $statusFile -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($preflight.mode -ne 'data_preflight' -or $preflight.state -ne 'complete' -or
            [int]$preflight.seed -ne $Seed -or [string]$preflight.run_id -ne $PreparedRunId -or
            [string]$preflight.answer_pool -ne $AnswerPool) {
            $details = "mode=$($preflight.mode), state=$($preflight.state), stage=$($preflight.stage)"
            if ($preflight.error) {
                $details += ", remote error=$($preflight.error)"
            }
            throw "SDD preflight is not reusable or does not match this seed/run ID/answer pool ($details). Inspect the first Modal job before starting another 500-step run."
        }
    }
    finally {
        if (Test-Path -LiteralPath $statusFile -PathType Leaf) {
            Remove-Item -LiteralPath $statusFile -Force
        }
    }
    Write-Host 'Validated completed SDD data preflight.'
}
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
