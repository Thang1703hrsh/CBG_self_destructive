<#
Evaluate existing checkpoints; never runs defense training or pushes to HF.
Example: .\scripts\Run-DefenseEval.ps1 -Suite Core -Models "base,seam" -DryRun
Configure config/defense_eval.yaml before selecting SDD and CTRAP.
#>
[CmdletBinding()]
param(
    [string]$ConfigPath = '',
    [ValidateSet('Clean', 'Smoke', 'Core', 'Robustness', 'Full', 'Paper')][string]$Suite = 'Core',
    [string]$Models = 'sdd,ctrap,seam',
    [ValidatePattern('^[A-Za-z0-9_-]{1,80}$')][string]$RunId = '',
    [string]$ModalProfile = '',
    [switch]$PrepareOnly,
    [switch]$Resume,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if (-not $ConfigPath) { $ConfigPath = Join-Path $repoRoot 'config\defense_eval.yaml' }
$config = (Resolve-Path -LiteralPath $ConfigPath -ErrorAction Stop).Path
if ($Resume -and -not $RunId) { throw '-Resume requires the original -RunId.' }
if (-not $RunId) {
    $RunId = 'defense-eval-' + $Suite.ToLowerInvariant() + '-' +
        (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ') + '-' +
        [guid]::NewGuid().ToString('N').Substring(0, 8)
}
$python = (Get-Command python -ErrorAction Stop).Source
$env:PYTHONIOENCODING = 'utf-8'
# Local validation does not import Modal, download models, or request a GPU.
& $python -X utf8 (Join-Path $repoRoot 'scripts\defense_eval_common.py') `
    --config $config --suite $Suite.ToLowerInvariant() --models $Models
if ($LASTEXITCODE -ne 0) { throw 'Configuration validation failed; fill checkpoint and defense-data paths.' }

$commandArgs = @('-X', 'utf8', '-m', 'modal', 'run')
if ($ModalProfile) { $commandArgs += @('--profile', $ModalProfile) }
$commandArgs += @('--detach', 'scripts/modal_defense_eval.py::main', '--run-id', $RunId)
if ($Resume) { $commandArgs += '--resume' }
if ($PrepareOnly) { $commandArgs += '--prepare-only' }
Write-Host "Run ID: $RunId"
Write-Host "Config: $config"
Write-Host 'Checkpoint eval: one L40S per case, sequential; original checkpoints are preserved.'
if ($PrepareOnly) { Write-Host 'CPU preparation only; no GPU case will run.' }
Write-Host "Results: defense-eval-results/$RunId"
if ($DryRun) {
    Write-Host "Dry run: python $($commandArgs -join ' ')"
    return
}

$previous = @{}
foreach ($key in @('DEFENSE_EVAL_CONFIG', 'DEFENSE_EVAL_SUITE', 'DEFENSE_EVAL_MODELS')) {
    $previous[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
}
$env:DEFENSE_EVAL_CONFIG = $config
$env:DEFENSE_EVAL_SUITE = $Suite.ToLowerInvariant()
$env:DEFENSE_EVAL_MODELS = $Models
Push-Location -LiteralPath $repoRoot
try {
    & $python @commandArgs
    if ($LASTEXITCODE -ne 0) { throw "Modal submission failed (exit $LASTEXITCODE)." }
}
finally {
    Pop-Location
    foreach ($key in $previous.Keys) {
        [Environment]::SetEnvironmentVariable($key, $previous[$key], 'Process')
    }
}
Write-Host 'Submitted. The job continues remotely after this PC shuts down.'
Write-Host "Read status: .\scripts\Get-DefenseEval.ps1 -RunId $RunId -ModalProfile '$ModalProfile'"
