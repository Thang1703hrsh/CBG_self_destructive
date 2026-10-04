<#
Eval ducthang1703/ctrap-llama2-7b-seed42-step500 trên 6 case (suite Robustness:
clean + harmful 5 LR = Table 1 + Table 6), Modal profile oanhtranbachkhoa.
Job chạy --detach: Ctrl+C / tắt máy không dừng job.

  .\scripts\Run-CTRAPEval.ps1 -Step CopyData              # chép ctrap_defense.json sang profile (1 lần)
  .\scripts\Run-CTRAPEval.ps1 -Step Check                 # kiểm tra, không tốn tiền
  .\scripts\Run-CTRAPEval.ps1 -Step Run                   # gửi job, in Run ID
  .\scripts\Run-CTRAPEval.ps1 -Step Status -RunId <id>
  .\scripts\Run-CTRAPEval.ps1 -Step Download -RunId <id>
  .\scripts\Run-CTRAPEval.ps1 -Step Run -RunId <id>       # chạy tiếp, bỏ qua case đã xong
  -IncludeBase: thêm Llama-2-7b-chat gốc làm mốc (12 case); giữ nguyên ở mọi bước.
#>
[CmdletBinding()]
param(
    [ValidateSet('CopyData', 'Check', 'Run', 'Status', 'Download')][string]$Step = 'Check',
    [string]$ModalProfile = 'oanhtranbachkhoa',
    [ValidatePattern('^[A-Za-z0-9_-]{1,80}$')][string]$RunId = '',
    [switch]$IncludeBase
)

$ErrorActionPreference = 'Stop'
$evalArgs = @{ Step = $Step; ModalProfile = $ModalProfile
               Models = $(if ($IncludeBase) { 'base,ctrap' } else { 'ctrap' }) }
if ($RunId) { $evalArgs.RunId = $RunId }
& (Join-Path $PSScriptRoot 'Run-OanhEval.ps1') @evalArgs
