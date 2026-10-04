<#
Run one SDD comparison seed on Modal using a completed CPU data preflight.
The local PC may be shut down after submission succeeds. SDD training is
reconstructed because SDD/train is absent.

Example:
  powershell -ExecutionPolicy Bypass -File .\scripts\Run-SDDPreflight.ps1 -Seed 42
  powershell -ExecutionPolicy Bypass -File .\scripts\Run-SDDFull.ps1 -Seed 42 -PreparedRunId RUN_ID_FROM_PREFLIGHT
  powershell -ExecutionPolicy Bypass -File .\scripts\Run-SDDFull.ps1 -PreparedRunId RUN_ID_FROM_PREFLIGHT -ValidateOnly

Prerequisites: `modal setup`, Modal secret `huggingface-secret` with HF_TOKEN,
and the HF account must have accepted Llama-2 terms. AlpacaLima additionally
requires access to GAIR/lima. Data preparation runs before requesting an H100.
#>
[CmdletBinding()]
param(
    [ValidateRange(0, 2147483647)][int]$Seed = 42,
    [ValidateRange(0, 1000000)][int]$TestSize = 100,
    [ValidateSet('AlpacaOnly', 'AlpacaLima')][string]$AnswerPool = 'AlpacaOnly',
    [Parameter(Mandatory)][ValidatePattern('^[A-Za-z0-9_-]{1,64}$')][string]$PreparedRunId,
    [string]$ModalProfile = '',
    [switch]$ValidateOnly,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$runner = Join-Path $repoRoot 'scripts\modal_sdd_full.py'
if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) {
    throw "Missing Modal runner: $runner"
}
$runId = $PreparedRunId
$commandArgs = @('-X', 'utf8', '-m', 'modal', 'run')
if ($ModalProfile) {
    $commandArgs += @('--profile', $ModalProfile)
}
$commandArgs += @('--detach', 'scripts/modal_sdd_full.py::main', '--seed', "$Seed", '--run-id', $runId, '--test-size', "$TestSize", '--answer-pool', $AnswerPool)

Write-Host "SDD comparison run ID: $runId"
Write-Host 'Paper-derived SFT: 8k pairs, 500 steps, batch 24, LR 5e-7.'
if ($AnswerPool -eq 'AlpacaOnly') {
    Write-Warning "SDD answer pool: AlpacaOnly; this differs from the paper's Alpaca+LIMA pool."
}
else {
    Write-Host 'SDD answer pool: AlpacaLima; GAIR/lima access is required.'
}
Write-Host 'Shared SEAM-style attack/evaluation sweep; one H100 at a time.'
Write-Host 'This is one seed, not a published multi-seed table.'
Write-Host "Prepared data: defense-compare-data/sdd-seed$Seed-$runId"
Write-Host "Remote results: defense-compare-results/sdd-seed$Seed-$runId"

if ($DryRun) {
    Write-Host "Dry run only: python $($commandArgs -join ' ')"
    return
}

$pythonExecutable = Get-Command python -ErrorAction Stop
$env:PYTHONIOENCODING = 'utf-8'
$statusFile = Join-Path ([System.IO.Path]::GetTempPath()) ("sdd-preflight-status-" + [guid]::NewGuid().ToString('N') + '.json')
$checkArgs = @('-X', 'utf8', '-m', 'modal', 'volume', 'get',
    'defense-compare-results', "sdd-seed$Seed-$runId/status.json", $statusFile)
if ($ModalProfile) {
    $checkArgs += @('--profile', $ModalProfile)
}
try {
    & $pythonExecutable.Source @checkArgs
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $statusFile -PathType Leaf)) {
        throw 'SDD data preflight not found. Run Run-SDDPreflight.ps1 and wait for completion.'
    }
    $preflight = Get-Content -LiteralPath $statusFile -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($preflight.mode -ne 'data_preflight' -or $preflight.state -ne 'complete' -or
        [int]$preflight.seed -ne $Seed -or [string]$preflight.run_id -ne $runId -or
        [string]$preflight.answer_pool -ne $AnswerPool) {
        throw "SDD preflight is not complete or does not match seed/run ID/answer pool (state=$($preflight.state), stage=$($preflight.stage))."
    }
}
finally {
    if (Test-Path -LiteralPath $statusFile -PathType Leaf) {
        Remove-Item -LiteralPath $statusFile -Force
    }
}
Write-Host 'Validated completed CPU data preflight.'
if ($ValidateOnly) {
    Write-Host 'Validation only: no GPU job was submitted.'
    return
}
Write-Host 'Submitting full run.'
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
