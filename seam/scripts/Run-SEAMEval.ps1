<#
Eval riêng SEAM (tùy chọn kèm Base) cho Table 1 + Table 6: suite Robustness =
clean + harmful 5 LR (2e-5, 5e-5, 8e-5, 1e-4, 2e-4). Không train, không push HF.

Job chạy trên Modal ở chế độ --detach + spawn: Ctrl+C, đóng terminal hay tắt máy
đều KHÔNG dừng job. Muốn dừng thật: python -m modal app stop <app-id> --profile <profile>.

  .\scripts\Run-SEAMEval.ps1 -Step Check                 # kiểm tra, không tốn tiền
  .\scripts\Run-SEAMEval.ps1 -Step Run                   # gửi job, in Run ID
  .\scripts\Run-SEAMEval.ps1 -Step Status -RunId <id>    # xem tiến độ
  .\scripts\Run-SEAMEval.ps1 -Step Download -RunId <id>  # tải về results\<id>
  .\scripts\Run-SEAMEval.ps1 -Step Run -RunId <id>       # chạy tiếp run bị lỗi, bỏ qua case đã xong
#>
[CmdletBinding()]
param(
    [ValidateSet('Check', 'Run', 'Status', 'Download')][string]$Step = 'Check',
    [string]$ModalProfile = 'chichichanhchanh222',
    [ValidatePattern('^[A-Za-z0-9_-]{1,80}$')][string]$RunId = '',
    # Thêm Llama-2-7b-chat gốc làm dòng so sánh trong Table 1/6 (thêm 6 case).
    [switch]$IncludeBase
)

$ErrorActionPreference = 'Stop'
$python = (Get-Command python -ErrorAction Stop).Source
$env:PYTHONIOENCODING = 'utf-8'
$models = if ($IncludeBase) { 'base,seam' } else { 'seam' }

if ($Step -in 'Status', 'Download') {
    if (-not $RunId) { throw "-Step $Step requires -RunId." }
    $getArgs = @{ RunId = $RunId; ModalProfile = $ModalProfile }
    if ($Step -eq 'Download') { $getArgs.Download = $true }
    & (Join-Path $PSScriptRoot 'Get-DefenseEval.ps1') @getArgs
    return
}

# Dữ liệu phòng thủ của SEAM phải có để loại prompt trùng khỏi tập attack/test.
$listing = & $python -X utf8 -m modal volume ls seam-full-data `
    seam-full-seed42-20260930T104315Z-a64e5914 --profile $ModalProfile 2>&1 | Out-String
if ($LASTEXITCODE -ne 0 -or $listing -notmatch 'defense\.json') {
    throw "Missing seam-full-data:.../defense.json on profile $ModalProfile.`n$listing"
}
Write-Host 'OK  SEAM defense.json found'

$evalArgs = @{ Suite = 'Robustness'; Models = $models; ModalProfile = $ModalProfile }
if ($RunId) { $evalArgs.RunId = $RunId; $evalArgs.Resume = $true }
if ($Step -eq 'Check') { $evalArgs.DryRun = $true }
& (Join-Path $PSScriptRoot 'Run-DefenseEval.ps1') @evalArgs
