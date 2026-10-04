<#
Run one CTRAP comparison seed on Modal, detached from this PC.

Example:
  powershell -ExecutionPolicy Bypass -File .\scripts\Run-CTRAPFull.ps1 -Seed 42
  powershell -ExecutionPolicy Bypass -File .\scripts\Run-CTRAPFull.ps1 -Seed 42 -MaxSteps 500
  powershell -ExecutionPolicy Bypass -File .\scripts\Run-CTRAPFull.ps1 -DryRun

Prerequisites: `modal setup`, Modal secret `huggingface-secret` with HF_TOKEN,
and access to meta-llama/Llama-2-7b-chat-hf. The public RepNoise refusal file
is not verified as GPT-4o-generated. Default CTRAP training is 20 epochs;
-MaxSteps 500 runs a step-matched comparison variant instead.
#>
[CmdletBinding()]
param(
    [ValidateRange(0, 2147483647)][int]$Seed = 42,
    [ValidateRange(0, 1000000)][int]$TestSize = 100,
    [ValidateScript({ $_ -eq -1 -or $_ -gt 0 })][int]$MaxSteps = -1,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$runner = Join-Path $repoRoot 'scripts\modal_ctrap_full.py'
if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) {
    throw "Missing Modal runner: $runner"
}
$runId = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 8)
$commandArgs = @('-X', 'utf8', '-m', 'modal', 'run', '--detach', 'scripts/modal_ctrap_full.py::main', '--seed', "$Seed", '--run-id', $runId, '--test-size', "$TestSize", '--max-steps', "$MaxSteps")

Write-Host "CTRAP comparison run ID: $runId"
if ($MaxSteps -gt 0) {
    Write-Host "Original CTRAPTrainer with a $MaxSteps-optimizer-step cap (comparison variant; not the original 20-epoch budget)."
}
else {
    Write-Host 'Original CTRAPTrainer: 5k safe + 5k harmful + 5k UltraChat; 20 epochs.'
}
Write-Host 'Shared SEAM-style attack/evaluation sweep; one H100 at a time.'
Write-Host 'This is one seed, not a published multi-seed table.'
Write-Host "Remote results: defense-compare-results/ctrap-seed$Seed-$runId"

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
