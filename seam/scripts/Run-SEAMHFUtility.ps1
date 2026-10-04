<#
Evaluate the private SEAM checkpoint on four zero-shot utility tasks.
This submits a detached Modal job, so the PC may be shut down afterward.

Prerequisites in the selected Modal workspace:
  - A secret named huggingface-secret containing HF_TOKEN with access to
    ducthang1703/seam-llama2-7b-seed42-checkpoint500.
  - An available H100 GPU quota.

Examples:
  powershell -ExecutionPolicy Bypass -File .\scripts\Run-SEAMHFUtility.ps1 -DryRun
  powershell -ExecutionPolicy Bypass -File .\scripts\Run-SEAMHFUtility.ps1
  powershell -ExecutionPolicy Bypass -File .\scripts\Run-SEAMHFUtility.ps1 -ModalProfile myprofile
#>
[CmdletBinding()]
param(
    [string]$ModalProfile = '',
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$runner = Join-Path $repoRoot 'scripts\modal_seam_hf_utility.py'
if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) {
    throw "Missing Modal runner: $runner"
}
$runId = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 8)
$modalArgs = @('-X', 'utf8', '-m', 'modal', 'run')
if ($ModalProfile) {
    $modalArgs += @('--profile', $ModalProfile)
}
$modalArgs += @('--detach', 'scripts/modal_seam_hf_utility.py::main', '--run-id', $runId)

Write-Host 'Model: ducthang1703/seam-llama2-7b-seed42-checkpoint500 (private).'
Write-Host 'Tasks: truthfulqa_mc1, hellaswag, mmlu, arc_easy; zero-shot; no attack.'
Write-Host 'GPU: one H100. Each task is saved when it finishes.'
Write-Host "Run ID: $runId"
Write-Host "Remote results: seam-hf-utility-results/seam-hf-utility-$runId"
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
