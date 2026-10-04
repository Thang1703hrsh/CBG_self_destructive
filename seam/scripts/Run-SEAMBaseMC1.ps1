<#
Evaluate truthfulqa_mc1 on the unmodified Llama-2-7b-chat-hf base model.
No SEAM defense or harmful fine-tuning is run. The Modal job is detached, so
the local PC may be shut down after successful submission.

Prerequisites: Modal login/profile, secret `huggingface-secret` containing
HF_TOKEN for an account with access to meta-llama/Llama-2-7b-chat-hf.

Examples:
  powershell -ExecutionPolicy Bypass -File .\scripts\Run-SEAMBaseMC1.ps1 -DryRun
  powershell -ExecutionPolicy Bypass -File .\scripts\Run-SEAMBaseMC1.ps1
  powershell -ExecutionPolicy Bypass -File .\scripts\Run-SEAMBaseMC1.ps1 -ModalProfile myprofile
#>
[CmdletBinding()]
param(
    [string]$ModalProfile = '',
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$runner = Join-Path $repoRoot 'scripts\modal_seam_truthfulqa_mc1.py'
if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) {
    throw "Missing Modal runner: $runner"
}
$runId = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 8)
$modalArgs = @('-X', 'utf8', '-m', 'modal', 'run')
if ($ModalProfile) {
    $modalArgs += @('--profile', $ModalProfile)
}
$modalArgs += @('--detach', 'scripts/modal_seam_truthfulqa_mc1.py::main', '--run-id', $runId)

Write-Host 'Evaluation: Base Llama-2-7b-chat-hf, truthfulqa_mc1, zero-shot, no attack.'
Write-Host 'GPU: one H100. No defense training or harmfulness evaluation.'
Write-Host "Run ID: $runId"
Write-Host "Remote results: seam-mc1-results/base-truthfulqa-mc1-$runId"
if ($DryRun) {
    Write-Host "Dry run only: python $($modalArgs -join ' ')"
    return
}

$pythonExecutable = Get-Command python -ErrorAction Stop
$env:PYTHONIOENCODING = 'utf-8'
if (-not $ModalProfile) {
    $activeProfile = & $pythonExecutable.Source -X utf8 -m modal profile current
    if ($LASTEXITCODE -ne 0 -or -not $activeProfile) {
        throw 'Could not identify active Modal profile. Run modal setup or pass -ModalProfile.'
    }
    Write-Host "Active Modal profile: $activeProfile"
}
else {
    Write-Host "Requested Modal profile: $ModalProfile"
}

Push-Location -LiteralPath $repoRoot
try {
    & $pythonExecutable.Source @modalArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Modal submission failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
Write-Host 'Submitted. You may shut down this PC; check progress with: python -X utf8 -m modal app list'
