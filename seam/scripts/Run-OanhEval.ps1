<#
Eval Base + SEAM + CTRAP cho Table 1 + Table 6 (suite Robustness: clean + harmful
5 LR) trên Modal profile oanhtranbachkhoa, trong MỘT run. Không train, không push HF.
Job chạy --detach + spawn: Ctrl+C / tắt máy không dừng job.

  .\scripts\Run-OanhEval.ps1 -Step CopyData              # chép defense data từ chichichanhchanh222 (1 lần)
  .\scripts\Run-OanhEval.ps1 -Step Check                 # kiểm tra, không tốn tiền
  .\scripts\Run-OanhEval.ps1 -Step Run                   # gửi job, in Run ID
  .\scripts\Run-OanhEval.ps1 -Step Status -RunId <id>
  .\scripts\Run-OanhEval.ps1 -Step Download -RunId <id>  # -> results\<id>\comparison.md
  .\scripts\Run-OanhEval.ps1 -Step Run -RunId <id>       # chạy tiếp, bỏ qua case đã xong
  -Models ctrap  hoặc  -Models base,ctrap                 # chỉ eval một phần (giữ nguyên ở mọi bước)
#>
[CmdletBinding()]
param(
    [ValidateSet('CopyData', 'Check', 'Run', 'Status', 'Download')][string]$Step = 'Check',
    [string]$ModalProfile = 'oanhtranbachkhoa',
    [string]$SourceProfile = 'chichichanhchanh222',
    [ValidatePattern('^[A-Za-z0-9_-]{1,80}$')][string]$RunId = '',
    [string]$Models = 'base,seam,ctrap'
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$config = Join-Path $repoRoot 'config\defense_eval_oanh.yaml'
$python = (Get-Command python -ErrorAction Stop).Source
$env:PYTHONIOENCODING = 'utf-8'

# Volume, file path inside it; must match config\defense_eval_oanh.yaml.
$allDefenseFiles = @{
    seam  = @('seam-full-data', 'seam-full-seed42-20260930T104315Z-a64e5914/defense.json')
    ctrap = @('defense-compare-data', 'ctrap-l40s-seed42-20261001T030609Z-306acd8b/ctrap_defense.json')
}
$selected = $Models.Split(',') | ForEach-Object { $_.Trim().ToLowerInvariant() } | Where-Object { $_ }
$unknown = $selected | Where-Object { $_ -notin @('base', 'seam', 'ctrap') }
if ($unknown) { throw "Unsupported model(s) for this config: $($unknown -join ', ')" }
$defenseFiles = @($selected | Where-Object { $allDefenseFiles.ContainsKey($_) } | ForEach-Object { ,$allDefenseFiles[$_] })

function Invoke-Modal([string[]]$Arguments) {
    $output = & $python -X utf8 -m modal @Arguments 2>&1 | Out-String
    return [pscustomobject]@{ Ok = ($LASTEXITCODE -eq 0); Text = $output }
}

switch ($Step) {
    'CopyData' {
        $temp = Join-Path $env:TEMP 'seam-oanh-defense'
        New-Item -ItemType Directory -Force $temp | Out-Null
        foreach ($item in $defenseFiles) {
            $volume, $path = $item
            $local = Join-Path $temp ($path -replace '/', '__')
            $result = Invoke-Modal @('volume', 'get', $volume, $path, $local, '--force', '--profile', $SourceProfile)
            if (-not $result.Ok -or -not (Test-Path -LiteralPath $local -PathType Leaf)) {
                throw "Download failed: ${SourceProfile}:$volume/$path`n$($result.Text)"
            }
            # create fails harmlessly when the volume already exists.
            Invoke-Modal @('volume', 'create', $volume, '--profile', $ModalProfile) | Out-Null
            $result = Invoke-Modal @('volume', 'put', $volume, $local, $path, '--force', '--profile', $ModalProfile)
            if (-not $result.Ok) { throw "Upload failed: ${ModalProfile}:$volume/$path`n$($result.Text)" }
            Write-Host "Copied $volume/$path -> $ModalProfile"
        }
        return
    }
    { $_ -in 'Status', 'Download' } {
        if (-not $RunId) { throw "-Step $Step requires -RunId." }
        $getArgs = @{ RunId = $RunId; ModalProfile = $ModalProfile }
        if ($Step -eq 'Download') { $getArgs.Download = $true }
        & (Join-Path $PSScriptRoot 'Get-DefenseEval.ps1') @getArgs
        return
    }
}

# Missing defense data would silently weaken prompt-overlap exclusion.
foreach ($item in $defenseFiles) {
    $volume, $path = $item
    $parent, $name = $path -split '/', 2
    $result = Invoke-Modal @('volume', 'ls', $volume, $parent, '--profile', $ModalProfile)
    if (-not $result.Ok -or $result.Text -notmatch [regex]::Escape($name)) {
        throw "Missing ${ModalProfile}:$volume/$path. Run -Step CopyData first.`n$($result.Text)"
    }
    Write-Host "OK  $volume/$path"
}

$evalArgs = @{ ConfigPath = $config; Suite = 'Robustness'; Models = $Models; ModalProfile = $ModalProfile }
if ($RunId) { $evalArgs.RunId = $RunId; $evalArgs.Resume = $true }
if ($Step -eq 'Check') { $evalArgs.DryRun = $true }
& (Join-Path $PSScriptRoot 'Run-DefenseEval.ps1') @evalArgs
