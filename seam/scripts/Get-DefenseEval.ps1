<# Read remote status, or download results. -Force refreshes existing downloaded files. #>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][ValidatePattern('^[A-Za-z0-9_-]{1,80}$')][string]$RunId,
    [string]$ModalProfile = '',
    [switch]$Download,
    [string]$Destination = '',
    [switch]$Force
)
$ErrorActionPreference = 'Stop'
$python = (Get-Command python -ErrorAction Stop).Source
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if ($Download) {
    if (-not $Destination) { $Destination = Join-Path $repoRoot "results\$RunId" }
    $target = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Destination)
    if (Test-Path -LiteralPath $target) {
        if (-not (Test-Path -LiteralPath $target -PathType Container)) {
            throw "Download destination is a file; choose a directory: $target"
        }
        # Older downloads may have kept the remote run directory inside target.
        # Copy its contents up without deleting the original download or replacing files.
        $nested = Join-Path $target $RunId
        if ((Test-Path -LiteralPath (Join-Path $nested 'comparison.md') -PathType Leaf) -and
            (Test-Path -LiteralPath (Join-Path $nested 'summary.json') -PathType Leaf) -and
            -not (Test-Path -LiteralPath (Join-Path $target 'comparison.md'))) {
            $entries = @(Get-ChildItem -LiteralPath $nested -Force)
            $collisions = @($entries | Where-Object { Test-Path -LiteralPath (Join-Path $target $_.Name) })
            if ($collisions.Count -eq 0) {
                foreach ($entry in $entries) {
                    Copy-Item -LiteralPath $entry.FullName -Destination $target -Recurse
                }
                Write-Host "Recovered results from nested download: $nested"
            }
        }
        $hasContent = @(Get-ChildItem -LiteralPath $target -Force).Count -gt 0
        if ($hasContent -and -not $Force) {
            if ((Test-Path -LiteralPath (Join-Path $target 'comparison.md') -PathType Leaf) -and
                (Test-Path -LiteralPath (Join-Path $target 'summary.json') -PathType Leaf)) {
                Write-Host "Results already downloaded: $target"
                Write-Host "Comparison: $(Join-Path $target 'comparison.md')"
                Write-Host 'Use -Download -Force to refresh files from Modal.'
                return
            }
            throw "Destination contains partial results. Use -Download -Force to retry or choose -Destination: $target"
        }
    }
    $parent = Split-Path -Path $target -Parent
    if (-not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    if (-not (Test-Path -LiteralPath $target)) {
        New-Item -ItemType Directory -Path $target | Out-Null
    }
    # Modal versions differ in whether a recursive download retains its root
    # directory. Download into a fresh staging directory, then copy the contents.
    $stageName = '.defense-eval-download-' + [guid]::NewGuid().ToString('N')
    $staging = Join-Path $parent $stageName
    New-Item -ItemType Directory -Path $staging | Out-Null
    try {
        $commandArgs = @('-X', 'utf8', '-m', 'modal', 'volume', 'get', 'defense-eval-results', $RunId, $staging)
        if ($ModalProfile) { $commandArgs += @('--profile', $ModalProfile) }
        & $python @commandArgs
        if ($LASTEXITCODE -ne 0) { throw 'Result download failed.' }
        $downloaded = Join-Path $staging $RunId
        if (-not (Test-Path -LiteralPath $downloaded -PathType Container)) { $downloaded = $staging }
        foreach ($entry in Get-ChildItem -LiteralPath $downloaded -Force) {
            Copy-Item -LiteralPath $entry.FullName -Destination $target -Recurse -Force:$Force
        }
    }
    finally {
        # Only remove the unique staging directory we just created, after checking
        # its resolved absolute path is the expected child of the target parent.
        if (Test-Path -LiteralPath $staging -PathType Container) {
            $resolvedParent = (Resolve-Path -LiteralPath $parent).Path
            $resolvedStage = (Resolve-Path -LiteralPath $staging).Path
            $expectedStage = Join-Path $resolvedParent $stageName
            if ($resolvedStage -eq $expectedStage) {
                Remove-Item -LiteralPath $resolvedStage -Recurse -Force
            }
        }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $target 'comparison.md') -PathType Leaf)) {
        Write-Host 'comparison.md is not available yet; read the run status and download again when complete.'
    }
    Write-Host "Downloaded: $target"
    Write-Host "Comparison: $(Join-Path $target 'comparison.md')"
    return
}
$temporary = Join-Path ([System.IO.Path]::GetTempPath()) ('defense-eval-status-' + [guid]::NewGuid().ToString('N') + '.json')
try {
    $commandArgs = @('-X', 'utf8', '-m', 'modal', 'volume', 'get',
        'defense-eval-results', "$RunId/status.json", $temporary)
    if ($ModalProfile) { $commandArgs += @('--profile', $ModalProfile) }
    & $python @commandArgs
    if ($LASTEXITCODE -ne 0) { throw 'Could not read remote status.' }
    Get-Content -LiteralPath $temporary -Raw -Encoding UTF8
}
finally {
    if (Test-Path -LiteralPath $temporary -PathType Leaf) { Remove-Item -LiteralPath $temporary -Force }
}
