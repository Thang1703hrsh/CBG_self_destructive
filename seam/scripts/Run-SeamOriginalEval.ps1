<#
Eval Base + SEAM + SDD + CTRAP (Table 1 + Table 6: clean + harmful 5 LR) theo đúng
protocol của evaluator gốc SEAM (config\defense_eval_seam_original.yaml), trên
oanhtranbachkhoa, mỗi case một L40S (4 CPU, 16 GB RAM), tối đa -Parallel case song song.
Job chạy --detach: Ctrl+C / tắt máy không dừng job.

  .\scripts\Run-SeamOriginalEval.ps1 -Step Check                 # kiểm tra dữ liệu + config, không tốn tiền
  .\scripts\Run-SeamOriginalEval.ps1 -Step Run                   # gửi job, in Run ID
  .\scripts\Run-SeamOriginalEval.ps1 -Step Status -RunId <id>
  .\scripts\Run-SeamOriginalEval.ps1 -Step Download -RunId <id>  # -> results\<id>\comparison.md
  .\scripts\Run-SeamOriginalEval.ps1 -Step Run -RunId <id>       # chạy tiếp, bỏ qua case đã xong
#>
[CmdletBinding()]
param(
    [ValidateSet('Check', 'Run', 'Status', 'Download')][string]$Step = 'Check',
    [string]$ModalProfile = 'oanhtranbachkhoa',
    [ValidatePattern('^[A-Za-z0-9_-]{1,80}$')][string]$RunId = '',
    [string]$Models = 'base,seam,sdd,ctrap',
    # Số L40S chạy cùng lúc; tổng chi phí GPU không đổi, chỉ nhanh hơn.
    [ValidateRange(1, 24)][int]$Parallel = 6
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$config = Join-Path $repoRoot 'config\defense_eval_seam_original.yaml'
$python = (Get-Command python -ErrorAction Stop).Source
$env:PYTHONIOENCODING = 'utf-8'

if ($Step -in 'Status', 'Download') {
    if (-not $RunId) { throw "-Step $Step requires -RunId." }
    $getArgs = @{ RunId = $RunId; ModalProfile = $ModalProfile }
    if ($Step -eq 'Download') { $getArgs.Download = $true }
    & (Join-Path $PSScriptRoot 'Get-DefenseEval.ps1') @getArgs
    return
}

# Every volume input in the config must exist on this profile before any GPU is used.
$required = @(
    @('seam-full-data', 'seam-full-seed42-20260930T104315Z-a64e5914', 'defense.json'),
    @('defense-compare-data', 'ctrap-l40s-seed42-20261001T030609Z-306acd8b', 'ctrap_defense.json'),
    @('defense-compare-data', 'sdd-seed42-20261003T135304Z-977553aa', 'sdd_pairs.jsonl'),
    @('defense-compare-data', 'sdd-seed42-20261003T135304Z-977553aa', 'attack.json'),
    @('defense-compare-results', 'sdd-seed42-20261003T135304Z-977553aa/defense/model', 'model.safetensors.index.json')
)
foreach ($item in $required) {
    $listing = & $python -X utf8 -m modal volume ls $item[0] $item[1] --profile $ModalProfile 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0 -or $listing -notmatch [regex]::Escape($item[2])) {
        throw "Missing $($item[0]):$($item[1])/$($item[2]) on $ModalProfile.`n$listing"
    }
    Write-Host "OK  $($item[0]):$($item[1])/$($item[2])"
}

$env:DEFENSE_EVAL_PARALLEL = "$Parallel"
$evalArgs = @{ ConfigPath = $config; Suite = 'Robustness'; Models = $Models; ModalProfile = $ModalProfile }
if ($RunId) { $evalArgs.RunId = $RunId; $evalArgs.Resume = $true }
if ($Step -eq 'Check') { $evalArgs.DryRun = $true }
& (Join-Path $PSScriptRoot 'Run-DefenseEval.ps1') @evalArgs
