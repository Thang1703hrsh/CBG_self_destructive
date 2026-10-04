<#
Eval SEAM, SDD, CTRAP trên suite Core (6 kịch bản/model) trong MỘT run duy nhất,
để cả ba dùng chung một tập attack/test/benign đã loại trùng với dữ liệu phòng thủ
của mọi model. Không train, không push HF; checkpoint gốc giữ nguyên.

6 kịch bản Core: clean, harmful-lr2e-5, harmful-lr5e-5, harmful-lr1e-4,
                 benign-lr2e-5, mixed10-lr5e-5 (250 bước, 1000 mẫu, L40S/case).

Quy trình khuyến nghị:
  1) .\scripts\Run-Core3Eval.ps1 -ModalProfile <profile> -SddRunId <id> -Step Check
  2) .\scripts\Run-Core3Eval.ps1 -ModalProfile <profile> -SddRunId <id> -Step Prepare
  3) .\scripts\Run-Core3Eval.ps1 -ModalProfile <profile> -SddRunId <id> -Step Run -RunId <id từ bước 2>
  4) .\scripts\Get-DefenseEval.ps1 -RunId <id> -ModalProfile <profile>            # xem trạng thái
     .\scripts\Get-DefenseEval.ps1 -RunId <id> -ModalProfile <profile> -Download  # tải kết quả
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$ModalProfile,
    # Phần sau "sdd-seed42-" trong thư mục volume defense-compare-data.
    [Parameter(Mandatory)][ValidatePattern('^[A-Za-z0-9_-]{1,64}$')][string]$SddRunId,
    [ValidateSet('Check', 'Prepare', 'Run')][string]$Step = 'Check',
    [ValidatePattern('^[A-Za-z0-9_-]{1,80}$')][string]$RunId = '',
    # Thêm Llama-2-7b-chat gốc làm mốc tham chiếu (thêm 6 case).
    [switch]$IncludeBase,
    # Robustness = 6 case: clean (Table 1) + harmful 5 LR (Table 6).
    # Paper = 8 case (Table 1 + Table 6 đủ 5 LR + benign + LoRA, không có mixed/Table 2).
    # Core = 6 case; Full = 9 case (Paper + mixed10).
    [ValidateSet('Robustness', 'Paper', 'Core', 'Full')][string]$Suite = 'Robustness'
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$python = (Get-Command python -ErrorAction Stop).Source
$env:PYTHONIOENCODING = 'utf-8'
$models = if ($IncludeBase) { 'base,sdd,ctrap,seam' } else { 'sdd,ctrap,seam' }

# 1. Tạo bản config riêng với đường dẫn SDD thật; không sửa config gốc.
$source = Join-Path $repoRoot 'config\defense_eval.yaml'
$config = Join-Path $repoRoot 'config\defense_eval_core3.yaml'
$text = Get-Content -LiteralPath $source -Raw -Encoding UTF8
$sddPath = "sdd-seed42-$SddRunId/sdd_pairs.jsonl"
$text = [regex]::Replace($text, 'sdd-seed42-[A-Za-z0-9_-]+/sdd_pairs\.jsonl', $sddPath)
if ($text -notmatch [regex]::Escape($sddPath)) { throw 'Could not set the SDD defense_data path.' }
[IO.File]::WriteAllText($config, $text, (New-Object Text.UTF8Encoding $false))
Write-Host "Config: $config"

# 2. Kiểm tra dữ liệu phòng thủ tồn tại trên Modal. Thiếu file thì việc loại trùng
#    prompt sai, nên dừng tại đây.
$env:MODAL_PROFILE = $ModalProfile
$required = @(
    @('defense-compare-data', "sdd-seed42-$SddRunId", 'sdd_pairs.jsonl'),
    @('defense-compare-data', 'ctrap-l40s-seed42-20261001T034515Z-cd52d171', 'ctrap_defense.json'),
    @('seam-full-data', 'seam-full-seed42-20260930T104315Z-a64e5914', 'defense.json')
)
foreach ($item in $required) {
    $listing = & $python -X utf8 -m modal volume ls $item[0] $item[1] 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0 -or $listing -notmatch [regex]::Escape($item[2])) {
        throw "Missing $($item[0]):$($item[1])/$($item[2]) (profile $ModalProfile).`n$listing"
    }
    Write-Host "OK  $($item[0]):$($item[1])/$($item[2])"
}

$evalArgs = @{ ConfigPath = $config; Suite = $Suite; Models = $models; ModalProfile = $ModalProfile }
if ($RunId) { $evalArgs.RunId = $RunId }
switch ($Step) {
    # Kiểm tra cấu hình cục bộ + in lệnh Modal, không chạy gì trên cloud.
    'Check'   { & (Join-Path $PSScriptRoot 'Run-DefenseEval.ps1') @evalArgs -DryRun }
    # Chỉ chuẩn bị dữ liệu trên CPU (rẻ); xem số prompt/overlap trước khi thuê GPU.
    'Prepare' { & (Join-Path $PSScriptRoot 'Run-DefenseEval.ps1') @evalArgs -PrepareOnly }
    # Chạy GPU. Dùng lại RunId của bước Prepare để giữ nguyên tập dữ liệu đã chuẩn bị.
    'Run' {
        if (-not $RunId) { throw '-Step Run requires -RunId from the Prepare step.' }
        & (Join-Path $PSScriptRoot 'Run-DefenseEval.ps1') @evalArgs -Resume
    }
}
