<#
Chạy tiếp run eval SEAM (Base + SEAM, suite Robustness) đã dừng trên
chichichanhchanh222, trên tài khoản oanhtranbachkhoa. 8 case đã xong được giữ lại;
chỉ chạy các case còn thiếu, cùng tập dữ liệu, cùng config và code.

  1) .\scripts\Resume-SEAMEvalOanh.ps1 -Step Transfer   # tải run về, chép run + defense.json sang oanh, nạp cache HF
  2) .\scripts\Resume-SEAMEvalOanh.ps1 -Step Resume     # gửi job chạy tiếp (--detach, tắt máy không sao)
  3) .\scripts\Resume-SEAMEvalOanh.ps1 -Step Status
  4) .\scripts\Resume-SEAMEvalOanh.ps1 -Step Download   # -> results\<RunId>\comparison.md
#>
[CmdletBinding()]
param(
    [ValidateSet('Transfer', 'Resume', 'Status', 'Download')][string]$Step = 'Transfer',
    [ValidatePattern('^[A-Za-z0-9_-]{1,80}$')][string]$RunId = 'defense-eval-robustness-20261003T131958Z-bab8c7b7',
    [string]$SourceProfile = 'chichichanhchanh222',
    [string]$ModalProfile = 'oanhtranbachkhoa'
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$python = (Get-Command python -ErrorAction Stop).Source
$env:PYTHONIOENCODING = 'utf-8'
$local = Join-Path $repoRoot "results\$RunId"

function Invoke-Modal([string[]]$Arguments) {
    & $python -X utf8 -m modal @Arguments
    if ($LASTEXITCODE -ne 0) { throw "modal $($Arguments -join ' ') failed (exit $LASTEXITCODE)" }
}

switch ($Step) {
    'Transfer' {
        # Fresh copy of the stopped run, including inputs and finished cases.
        & (Join-Path $PSScriptRoot 'Get-DefenseEval.ps1') -RunId $RunId -ModalProfile $SourceProfile -Download -Force
        if (-not (Test-Path -LiteralPath (Join-Path $local 'manifest.json'))) { throw "Missing $local\manifest.json" }
        $existing = & $python -X utf8 -m modal volume ls defense-eval-results / --profile $ModalProfile 2>&1 | Out-String
        if ($existing -match [regex]::Escape($RunId)) {
            throw "$RunId already exists on $ModalProfile; skip Transfer and use -Step Resume."
        }
        Invoke-Modal @('volume', 'put', 'defense-eval-results', $local, $RunId, '--profile', $ModalProfile)
        # SEAM defense data is mounted as a source volume by the eval app.
        & (Join-Path $PSScriptRoot 'Run-OanhEval.ps1') -Step CopyData -Models seam `
            -ModalProfile $ModalProfile -SourceProfile $SourceProfile
        # Resume re-hashes each checkpoint at its pinned cache path; restore and verify them now (CPU).
        Push-Location -LiteralPath $repoRoot
        try { Invoke-Modal @('run', '--profile', $ModalProfile, 'scripts/modal_hf_cache_warm.py::main', '--run-id', $RunId) }
        finally { Pop-Location }
        Write-Host 'Transfer complete. Next: -Step Resume'
    }
    'Resume' {
        & (Join-Path $PSScriptRoot 'Run-DefenseEval.ps1') -ConfigPath (Join-Path $repoRoot 'config\defense_eval_oanh.yaml') `
            -Suite Robustness -Models 'base,seam' -RunId $RunId -Resume -ModalProfile $ModalProfile
    }
    'Status'   { & (Join-Path $PSScriptRoot 'Get-DefenseEval.ps1') -RunId $RunId -ModalProfile $ModalProfile }
    'Download' {
        & (Join-Path $PSScriptRoot 'Get-DefenseEval.ps1') -RunId $RunId -ModalProfile $ModalProfile `
            -Download -Destination (Join-Path $repoRoot "results\$RunId-oanh") -Force
    }
}
