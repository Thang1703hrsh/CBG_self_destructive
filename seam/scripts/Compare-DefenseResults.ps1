<#
Build a Markdown comparison from three downloaded Modal result directories.
No network or GPU job is started by this script.

Example:
  powershell -ExecutionPolicy Bypass -File .\scripts\Compare-DefenseResults.ps1 `
    -SeamRunDir .\results\seam -SddRunDir .\results\sdd `
    -CtrapRunDir .\results\ctrap -OutputPath .\results\comparison.md
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$SeamRunDir,
    [Parameter(Mandatory)][string]$SddRunDir,
    [Parameter(Mandatory)][string]$CtrapRunDir,
    [string]$OutputPath = '.\defense-comparison-result.md'
)

$ErrorActionPreference = 'Stop'

function Read-Run([string]$Directory) {
    $resolved = (Resolve-Path -LiteralPath $Directory -ErrorAction Stop).Path
    $summaryPath = Join-Path $resolved 'summary.json'
    $manifestPath = Join-Path $resolved 'manifest.json'
    if (-not (Test-Path -LiteralPath $summaryPath -PathType Leaf)) {
        throw "Missing summary.json in $resolved"
    }
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        throw "Missing manifest.json in $resolved"
    }
    return [pscustomobject]@{
        summary = Get-Content -LiteralPath $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
        manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    }
}

function Get-Metric([object]$Run, [string]$Key) {
    $property = $Run.summary.metrics.PSObject.Properties[$Key]
    if ($null -eq $property) {
        throw "Missing metric key: $Key"
    }
    return $property.Value
}

function Format-Score([object]$Metric, [string]$Field) {
    $value = $null
    if ($Field -eq 'harmfulness') {
        $value = $Metric.harmfulness
    }
    else {
        $task = $Metric.post_utility.PSObject.Properties[$Field]
        if ($null -ne $task) {
            $value = $task.Value.'acc,none'
        }
    }
    if ($null -eq $value) { return 'n/a' }
    return ([double]$value).ToString('F4', [System.Globalization.CultureInfo]::InvariantCulture)
}

$seam = Read-Run $SeamRunDir
$sdd = Read-Run $SddRunDir
$ctrap = Read-Run $CtrapRunDir
if ($ctrap.manifest.attack_type -eq 'LoRA') {
    throw 'CTRAP used a LoRA attack; this comparison requires the same full-model attack protocol as SEAM and SDD.'
}
$seed = [int]$seam.manifest.seed
$sha = [string]$seam.manifest.source_sha256
$testSize = [int]$seam.manifest.test_size
if ($seed -ne [int]$sdd.manifest.seed -or $seed -ne [int]$ctrap.manifest.seed) {
    throw 'Cannot compare runs with different seeds.'
}
if ($sha -ne [string]$sdd.manifest.split.source_sha256 -or
    $sha -ne [string]$ctrap.manifest.split.source_sha256) {
    throw 'Cannot compare runs based on different RepNoise source hashes.'
}
if ($testSize -ne [int]$sdd.manifest.test_size -or
    $testSize -ne [int]$ctrap.manifest.test_size) {
    throw 'Cannot compare runs with different test sizes.'
}

$rates = @('pre', '2e-5', '5e-5', '8e-5', '1e-4', '2e-4')
$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add('# SEAM / SDD / CTRAP comparison')
$lines.Add('')
$lines.Add("Seed: $seed. BeaverTails test size: $testSize. Shared attack split: RepNoise indices 4000:5000.")
$lines.Add('Lower harmfulness is better; higher utility accuracy is better. This is a common comparison protocol, not any paper''s original table.')
$ctrapBudget = if ([int]$ctrap.manifest.defense_max_steps -gt 0) {
    "$($ctrap.manifest.defense_max_steps) optimizer steps"
}
else {
    "$($ctrap.manifest.defense_epochs) epochs"
}
$lines.Add("Training budgets: SEAM $($seam.manifest.train_steps) steps; SDD $($sdd.manifest.defense_steps) steps; CTRAP $ctrapBudget.")
$lines.Add('Equal optimizer-step counts do not imply equal training examples or GPU compute.')

foreach ($field in @('harmfulness', 'mmlu', 'arc_easy', 'truthfulqa_mc2', 'hellaswag')) {
    $lines.Add('')
    $lines.Add("## $field")
    $lines.Add('')
    $lines.Add('| Attack LR | Base | SEAM | SDD | CTRAP |')
    $lines.Add('| --- | ---: | ---: | ---: | ---: |')
    foreach ($rate in $rates) {
        $baseKey = if ($rate -eq 'pre') { 'base-pre' } else { "base-attack-lr$rate" }
        $seamKey = if ($rate -eq 'pre') { 'seam-pre' } else { "seam-attack-lr$rate" }
        $sddKey = if ($rate -eq 'pre') { 'sdd-pre' } else { "sdd-attack-lr$rate" }
        $ctrapKey = if ($rate -eq 'pre') { 'ctrap-pre' } else { "ctrap-attack-lr$rate" }
        $baseValue = Format-Score (Get-Metric $seam $baseKey) $field
        $seamValue = Format-Score (Get-Metric $seam $seamKey) $field
        $sddValue = Format-Score (Get-Metric $sdd $sddKey) $field
        $ctrapValue = Format-Score (Get-Metric $ctrap $ctrapKey) $field
        $lines.Add("| $rate | $baseValue | $seamValue | $sddValue | $ctrapValue |")
    }
}

$target = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputPath)
if (Test-Path -LiteralPath $target) {
    throw "Refusing to overwrite existing comparison file: $target"
}
$parent = Split-Path -Path $target -Parent
if (-not (Test-Path -LiteralPath $parent)) {
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
}
$lines | Set-Content -LiteralPath $target -Encoding UTF8
Write-Host "Comparison saved: $target"
