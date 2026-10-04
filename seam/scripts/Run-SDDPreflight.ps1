<#
Prepare and validate all 8,000 SDD training pairs on Modal CPU only.
The job is detached; the local PC may be shut down after submission.
Use the printed run ID with Run-SDDFull.ps1 once status.json is complete.

Example:
  powershell -ExecutionPolicy Bypass -File .\scripts\Run-SDDPreflight.ps1 -Seed 42
#>
[CmdletBinding()]
param(
    [ValidateRange(0, 2147483647)][int]$Seed = 42,
    [ValidateSet('AlpacaOnly', 'AlpacaLima')][string]$AnswerPool = 'AlpacaOnly',
    [string]$ModalProfile = '',
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$runner = Join-Path $repoRoot 'scripts\modal_sdd_full.py'
if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) {
    throw "Missing Modal runner: $runner"
}
$runId = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 8)
$commandArgs = @('-X', 'utf8', '-m', 'modal', 'run')
if ($ModalProfile) {
    $commandArgs += @('--profile', $ModalProfile)
}
$commandArgs += @('--detach', 'scripts/modal_sdd_full.py::preflight_main',
    '--seed', "$Seed", '--run-id', $runId, '--answer-pool', $AnswerPool)

Write-Host "SDD CPU preflight run ID: $runId"
Write-Host "Answer pool: $AnswerPool"
Write-Host "Remote status: defense-compare-results/sdd-seed$Seed-$runId/status.json"
Write-Host 'No GPU is requested by this preflight.'
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
Write-Host 'Submitted. Wait until status.json says state=complete before full training.'
Write-Host "Then run: powershell -ExecutionPolicy Bypass -File .\scripts\Run-SDDFull.ps1 -Seed $Seed -AnswerPool $AnswerPool -PreparedRunId $runId"
