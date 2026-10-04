<#
Train lại SDD (H100, 500 bước) rồi eval 6 case (suite Robustness: clean + harmful
5 LR = Table 1 + Table 6) trên Modal profile oanhtranbachkhoa. Mọi job chạy --detach:
Ctrl+C / tắt máy không dừng job.

  1) .\scripts\Run-SDDTrainEval.ps1 -Step Preflight                    # CPU: tạo 8000 cặp SDD, in SddRunId
  2) .\scripts\Run-SDDTrainEval.ps1 -Step TrainStatus -SddRunId <id>   # chờ state = complete
  3) .\scripts\Run-SDDTrainEval.ps1 -Step Train -SddRunId <id>         # H100: train 500 bước + push HF
  4) .\scripts\Run-SDDTrainEval.ps1 -Step TrainStatus -SddRunId <id>   # chờ có model (stage hf_upload/complete)
  5) .\scripts\Run-SDDTrainEval.ps1 -Step Check -SddRunId <id>         # kiểm tra eval, không tốn tiền
  6) .\scripts\Run-SDDTrainEval.ps1 -Step Eval  -SddRunId <id>         # gửi eval, in Run ID
  7) .\scripts\Run-SDDTrainEval.ps1 -Step Status   -SddRunId <id> -RunId <eval id>
     .\scripts\Run-SDDTrainEval.ps1 -Step Download -SddRunId <id> -RunId <eval id>
     .\scripts\Run-SDDTrainEval.ps1 -Step Eval     -SddRunId <id> -RunId <eval id>  # chạy tiếp eval lỗi

Eval đọc model SDD thẳng từ volume Modal, nên không phụ thuộc vào việc push HF.
#>
[CmdletBinding()]
param(
    [ValidateSet('Preflight', 'Train', 'TrainStatus', 'Check', 'Eval', 'Status', 'Download')]
    [string]$Step = 'Preflight',
    [string]$ModalProfile = 'oanhtranbachkhoa',
    [string]$SddRunId = '',
    # Repo HF phải MỚI: repo cũ ...-step500 đã bị lần chạy trước giữ chỗ.
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$')]
    [string]$HfRepo = 'ducthang1703/sdd-llama2-7b-seed42-step500-v2',
    [ValidatePattern('^[A-Za-z0-9_-]{1,80}$')][string]$RunId = '',
    [switch]$IncludeBase
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$python = (Get-Command python -ErrorAction Stop).Source
$env:PYTHONIOENCODING = 'utf-8'
$seed = 42

function Invoke-Modal([string[]]$Arguments) {
    $output = & $python -X utf8 -m modal @Arguments 2>&1 | Out-String
    return [pscustomobject]@{ Ok = ($LASTEXITCODE -eq 0); Text = $output }
}

# The latest preflight run ID is remembered so later steps need no -SddRunId.
$lastIdFile = Join-Path $repoRoot "results\sdd-last-run-id-$ModalProfile.txt"
if ($Step -eq 'Preflight') {
    $output = & (Join-Path $PSScriptRoot 'Run-SDDPreflight.ps1') -Seed $seed -AnswerPool AlpacaOnly `
        -ModalProfile $ModalProfile 6>&1 | ForEach-Object { Write-Host $_; "$_" }
    $match = $output | Select-String -Pattern 'SDD CPU preflight run ID: ([A-Za-z0-9_-]+)' | Select-Object -First 1
    if (-not $match) { throw 'Could not find the preflight run ID in the output above.' }
    $newId = $match.Matches[0].Groups[1].Value
    New-Item -ItemType Directory -Force (Split-Path $lastIdFile) | Out-Null
    Set-Content -LiteralPath $lastIdFile -Value $newId -Encoding ASCII
    Write-Host "Saved SddRunId $newId; later steps use it automatically."
    return
}
if (-not $SddRunId -and (Test-Path -LiteralPath $lastIdFile)) {
    $SddRunId = (Get-Content -LiteralPath $lastIdFile -TotalCount 1).Trim()
    Write-Host "Using saved SddRunId: $SddRunId"
}
if ($SddRunId -notmatch '^[A-Za-z0-9_-]{1,64}$') { throw "-Step $Step requires -SddRunId from the Preflight step." }
$runDir = "sdd-seed$seed-$SddRunId"

switch ($Step) {
    'TrainStatus' {
        $local = Join-Path $env:TEMP "sdd-status-$SddRunId.json"
        $result = Invoke-Modal @('volume', 'get', 'defense-compare-results', "$runDir/status.json",
                                 $local, '--force', '--profile', $ModalProfile)
        if (-not $result.Ok) { throw "Cannot read $runDir/status.json yet.`n$($result.Text)" }
        Get-Content -LiteralPath $local -Raw -Encoding UTF8
        return
    }
    'Train' {
        & (Join-Path $PSScriptRoot 'Run-SDDTrainPushHF.ps1') -Seed $seed -AnswerPool AlpacaOnly `
            -PreparedRunId $SddRunId -HfRepo $HfRepo -ModalProfile $ModalProfile
        return
    }
    { $_ -in 'Status', 'Download' } {
        if (-not $RunId) { throw "-Step $Step requires -RunId of the eval run." }
        $getArgs = @{ RunId = $RunId; ModalProfile = $ModalProfile }
        if ($Step -eq 'Download') { $getArgs.Download = $true }
        & (Join-Path $PSScriptRoot 'Get-DefenseEval.ps1') @getArgs
        return
    }
}

# Check / Eval: the trained model and SDD pairs must exist on this profile's volumes.
$model = Invoke-Modal @('volume', 'ls', 'defense-compare-results', "$runDir/defense/model", '--profile', $ModalProfile)
if (-not $model.Ok -or $model.Text -notmatch 'config\.json' -or $model.Text -notmatch '\.safetensors') {
    throw "SDD model not ready at defense-compare-results/$runDir/defense/model. Check -Step TrainStatus.`n$($model.Text)"
}
$pairs = Invoke-Modal @('volume', 'ls', 'defense-compare-data', $runDir, '--profile', $ModalProfile)
if (-not $pairs.Ok -or $pairs.Text -notmatch 'sdd_pairs\.jsonl') {
    throw "Missing defense-compare-data/$runDir/sdd_pairs.jsonl.`n$($pairs.Text)"
}
Write-Host "OK  SDD model and sdd_pairs.jsonl ($runDir)"

# Same protocol/data as config\defense_eval_oanh.yaml, plus this SDD run.
$template = Get-Content -LiteralPath (Join-Path $repoRoot 'config\defense_eval_oanh.yaml') -Raw -Encoding UTF8
$sddBlock = @"
models:
  sdd:
    source:
      kind: volume
      volume: defense-compare-results
      path: $runDir/defense/model
    defense_data:
      - kind: volume
        volume: defense-compare-data
        path: $runDir/sdd_pairs.jsonl
"@
$config = Join-Path $repoRoot 'config\defense_eval_sdd.yaml'
$text = ([regex]'(?m)^models:\r?\n').Replace($template, ($sddBlock -replace '\r', '') + "`n", 1)
[IO.File]::WriteAllText($config, $text, (New-Object Text.UTF8Encoding $false))

$evalArgs = @{ ConfigPath = $config; Suite = 'Robustness'; ModalProfile = $ModalProfile
               Models = $(if ($IncludeBase) { 'base,sdd' } else { 'sdd' }) }
if ($RunId) { $evalArgs.RunId = $RunId; $evalArgs.Resume = $true }
if ($Step -eq 'Check') { $evalArgs.DryRun = $true }
& (Join-Path $PSScriptRoot 'Run-DefenseEval.ps1') @evalArgs
