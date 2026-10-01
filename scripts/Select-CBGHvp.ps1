<#
.SYNOPSIS
  Select the best CBG-HVP configuration: train a (beta x alpha) grid, attack-evaluate every model on the
  VALIDATION pools with SEAM's protocol, and pick the best by a rule fixed in advance.

.DESCRIPTION
  1. Train   one CBG-HVP run per (beta, alpha) on 2x H200 (scripts/Run-CBGv2-7B.ps1), plus a beta = 0
             control. Runs that already exist are reused (-Reuse) or skipped when complete.
  2. Eval    SEAM-protocol attack on 1x H200 (src/experiments/attack_eval.py, --mode val): for the
             undefended base model and every run, harmfulness/utility before the attack, then one
             full-parameter attack per learning rate on d_val_atk[:900], harmfulness on d_val_atk[900:].
  3. Select  src/experiments/select_config.py:
               feasible = utility >= base - MaxUtilityDrop  and  pre-attack harm <= base + MaxHarmIncrease
               best     = feasible run with the lowest mean post-attack harmfulness
  4. Upload  the best run to Hugging Face (optional, -HfRepo).
  The test pools (attack.json, eval_harm) are never used here; evaluate the selected run on them for the paper.

.EXAMPLE
  .\scripts\Select-CBGHvp.ps1                                   # beta 0.003/0.01/0.03 (+ beta 0), alpha 1
.EXAMPLE
  .\scripts\Select-CBGHvp.ps1 -Betas 0.01 -Alphas 1,3 -Prefix sel-alpha -Parallel -HfRepo ducthang1703/cbg-hvp-best
#>
[CmdletBinding()]
param(
    [double[]]$Betas = @(0.003, 0.01, 0.03),
    [double[]]$Alphas = @(1.0),
    # Also train/evaluate beta = 0 (base losses only, same everything else) at the first alpha.
    [bool]$Control = $true,
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_-]{0,40}$')]
    [string]$Prefix = "sel",
    [string]$Config = "config/cbg_v2_llama2_7b_2xh200_fast_hvp.yaml",
    # Existing runs with exactly this config and (beta,alpha), keyed "beta,alpha" -> run name.
    [hashtable]$Reuse = @{ "0.01,1" = "hvp-tf32" },
    [string]$AttackLrs = "2e-5,5e-5,1e-4",
    [string]$UtilityTasks = "arc_easy",
    [double]$MaxUtilityDrop = 0.02,
    [double]$MaxHarmIncrease = 0.05,
    [string]$DataTag = "seam4k-v1",
    # Train / evaluate all candidates at the same time (needs GPU quota for every run at once).
    [switch]$Parallel,
    [switch]$SkipTraining,
    [string]$HfRepo = "",
    [switch]$Yes
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Launcher = "scripts/modal_cbg_v2_7b.py"
$RunScript = Join-Path $PSScriptRoot "Run-CBGv2-7B.ps1"
$OutputVolume = "cbg-v2-7b-output"
$BaseRun = "base-llama2-7b-chat"
$SelectionDir = Join-Path $Root "results\selection\$Prefix"
New-Item -ItemType Directory -Force -Path $SelectionDir | Out-Null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$Inv = [Globalization.CultureInfo]::InvariantCulture

function Write-Step([string]$Text) {
    Write-Host ""
    Write-Host ("=" * 78) -ForegroundColor DarkCyan
    Write-Host "  $Text" -ForegroundColor Cyan
    Write-Host ("=" * 78) -ForegroundColor DarkCyan
}

function Invoke-Native([string]$Log, [string[]]$Arguments) {
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & modal @Arguments | Tee-Object -FilePath $Log
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
    }
    return $code
}

function Get-RemoteJson([string]$RemotePath, [string]$LocalPath) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LocalPath) | Out-Null
    if (Test-Path $LocalPath) { Remove-Item $LocalPath -Force }
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & modal volume get $OutputVolume $RemotePath $LocalPath --force 2>$null | Out-Null
    } finally {
        $ErrorActionPreference = $previous
    }
    if (Test-Path $LocalPath) { return Get-Content $LocalPath -Raw -Encoding UTF8 | ConvertFrom-Json }
    return $null
}

function Format-Number([double]$Value) { return $Value.ToString("G6", $Inv) }

# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------
$candidates = New-Object System.Collections.ArrayList
$grid = @()
foreach ($alpha in $Alphas) { foreach ($beta in $Betas) { $grid += ,@($beta, $alpha) } }
if ($Control) { $grid += ,@(0.0, $Alphas[0]) }
foreach ($point in $grid) {
    $beta, $alpha = $point
    $key = "{0},{1}" -f (Format-Number $beta), (Format-Number $alpha)
    $name = "$Prefix-b$(Format-Number $beta)-a$(Format-Number $alpha)"
    $reused = $false
    if ($Reuse.ContainsKey($key)) { $name = $Reuse[$key]; $reused = $true }
    if ($name -notmatch '^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$') { throw "invalid run name '$name'" }
    [void]$candidates.Add([pscustomobject]@{ Name = $name; Beta = $beta; Alpha = $alpha; Reused = $reused })
}

Write-Step "Plan: $($candidates.Count) configurations + base model ($Config)"
$candidates | Format-Table Name, Beta, Alpha, Reused -AutoSize | Out-String | Write-Host
$newRuns = @($candidates | Where-Object { -not $_.Reused -and -not $SkipTraining })
$geometryRuns = @($newRuns | Where-Object { $_.Beta -gt 0 }).Count
$controlRuns = $newRuns.Count - $geometryRuns
$lrCount = @($AttackLrs.Split(",") | Where-Object { $_ }).Count
# Estimates only (2x H200 ~9.8 USD/h incl. host, 1x H200 ~4.9 USD/h): CBG-HVP run ~1.4 h, beta=0 run ~0.4 h,
# attack-eval ~(10 + 8 per learning rate) min per model.
$evalHours = (10 + 8 * $lrCount) / 60.0 * ($candidates.Count + 1)
$cost = 9.8 * (1.4 * $geometryRuns + 0.4 * $controlRuns) + 4.9 * $evalHours
Write-Host ("New training runs: {0} (+{1} control)   attack-evals: {2} x {3} learning rates" -f
            $geometryRuns, $controlRuns, ($candidates.Count + 1), $lrCount)
Write-Host ("Estimated cost: ~{0:F0} USD (estimate, not measured; check modal.com/pricing)" -f $cost)
Write-Host "Selection rule: utility($($UtilityTasks.Split(',')[0])) >= base - $MaxUtilityDrop, pre-attack harm <= base + $MaxHarmIncrease, then min post-attack harm"
if (-not $Yes) {
    $answer = Read-Host "Start? [y/N]"
    if ($answer -notmatch '^(y|yes)$') { return }
}

# ---------------------------------------------------------------------------
# 1. Training
# ---------------------------------------------------------------------------
Write-Step "1/4  Training ($(if ($Parallel) { 'parallel' } else { 'sequential' }))"
$toTrain = @()
foreach ($c in $candidates) {
    $metrics = Get-RemoteJson "$($c.Name)/train/metrics.json" (Join-Path $SelectionDir "_remote\$($c.Name)\metrics.json")
    if ($metrics -and $metrics.status -eq "complete") {
        Write-Host "  $($c.Name): already trained - skipping" -ForegroundColor Yellow
    } elseif ($c.Reused -or $SkipTraining) {
        Write-Host "  $($c.Name): not complete on the volume - it will be reported as missing" -ForegroundColor Yellow
    } else {
        $toTrain += $c
    }
}
$trainBlock = {
    param($Root, $RunScript, $Name, $Config, $Beta, $Alpha, $DataTag, $Log)
    Set-Location $Root
    & $RunScript -RunName $Name -Config $Config -Override @{ beta = $Beta; alpha = $Alpha } -DataTag $DataTag `
        -SkipVerify -SkipRadius -SkipSmoke -Yes *> $Log
}
if ($toTrain.Count) {
    if ($Parallel) {
        $jobs = foreach ($c in $toTrain) {
            $log = Join-Path $SelectionDir "train-$($c.Name).log"
            Write-Host "  starting $($c.Name)  (log: $log)"
            Start-Job -Name "train-$($c.Name)" -ScriptBlock $trainBlock `
                -ArgumentList $Root, $RunScript, $c.Name, $Config, $c.Beta, $c.Alpha, $DataTag, $log
        }
        $jobs | Wait-Job | Out-Null
        foreach ($job in $jobs) {
            if ($job.State -ne "Completed") { Write-Host "  $($job.Name): $($job.State) - see its log" -ForegroundColor Red }
            Receive-Job $job -ErrorAction SilentlyContinue | Out-Null
            Remove-Job $job
        }
    } else {
        foreach ($c in $toTrain) {
            $log = Join-Path $SelectionDir "train-$($c.Name).log"
            Write-Host "  training $($c.Name)  (beta $($c.Beta), alpha $($c.Alpha); log: $log)"
            try {
                & $trainBlock $Root $RunScript $c.Name $Config $c.Beta $c.Alpha $DataTag $log
            } catch {
                Write-Host "  $($c.Name) FAILED: $_ (see $log)" -ForegroundColor Red
            }
        }
    }
}

# ---------------------------------------------------------------------------
# 2. Attack evaluation (validation pools, 1x H200 each)
# ---------------------------------------------------------------------------
Write-Step "2/4  SEAM-protocol attack evaluation on validation pools (lrs $AttackLrs)"
$wanted = @($AttackLrs.Split(",") | Where-Object { $_ } | ForEach-Object { ([double]$_).ToString("G", $Inv) })
$evalTargets = @($BaseRun) + @($candidates | ForEach-Object { $_.Name })
$evalBlock = {
    param($Root, $Launcher, $Name, $DataTag, $AttackLrs, $UtilityTasks, $Log)
    Set-Location $Root
    $env:PYTHONIOENCODING = "utf-8"
    & modal run $Launcher --stage attack-eval --run-name $Name --eval-mode val --attack-lrs $AttackLrs `
        --utility-tasks $UtilityTasks --data-tag $DataTag *> $Log
    if ($LASTEXITCODE) { throw "attack-eval $Name failed (exit $LASTEXITCODE)" }
}
$toEval = @()
foreach ($name in $evalTargets) {
    $existing = Get-RemoteJson "$name/attack-eval-val/attack_eval.json" (Join-Path $SelectionDir "$name\attack_eval.json")
    $complete = $existing -and $null -ne $existing.post_harmfulness_mean -and
                @($wanted | Where-Object { -not ($existing.attacks.PSObject.Properties.Name -contains $_) }).Count -eq 0
    if ($complete) { Write-Host "  $name`: already evaluated - skipping" -ForegroundColor Yellow } else { $toEval += $name }
}
if ($toEval.Count) {
    if ($Parallel) {
        $jobs = foreach ($name in $toEval) {
            $log = Join-Path $SelectionDir "eval-$name.log"
            Write-Host "  starting attack-eval $name  (log: $log)"
            Start-Job -Name "eval-$name" -ScriptBlock $evalBlock `
                -ArgumentList $Root, $Launcher, $name, $DataTag, $AttackLrs, $UtilityTasks, $log
        }
        $jobs | Wait-Job | Out-Null
        foreach ($job in $jobs) {
            if ($job.State -ne "Completed") { Write-Host "  $($job.Name): $($job.State) - see its log" -ForegroundColor Red }
            Remove-Job $job
        }
    } else {
        foreach ($name in $toEval) {
            $log = Join-Path $SelectionDir "eval-$name.log"
            Write-Host "  attack-eval $name  (log: $log)"
            try {
                & $evalBlock $Root $Launcher $name $DataTag $AttackLrs $UtilityTasks $log
            } catch {
                Write-Host "  $name FAILED: $_" -ForegroundColor Red
            }
        }
    }
}

# ---------------------------------------------------------------------------
# 3. Selection
# ---------------------------------------------------------------------------
Write-Step "3/4  Selection"
$resultsDir = Join-Path $SelectionDir "results"
if (Test-Path $resultsDir) { Remove-Item $resultsDir -Recurse -Force }   # only this grid's results
foreach ($name in $evalTargets) {
    $json = Get-RemoteJson "$name/attack-eval-val/attack_eval.json" (Join-Path $resultsDir "$name\attack_eval.json")
    if (-not $json) { Write-Host "  $name`: no validation result - excluded" -ForegroundColor Red }
}
if (-not (Test-Path (Join-Path $resultsDir "$BaseRun\attack_eval.json"))) {
    throw "the base-model evaluation is missing; cannot apply the selection rule (see eval-$BaseRun.log)"
}
$candidates | Select-Object Name, Beta, Alpha | ConvertTo-Json | Set-Content (Join-Path $SelectionDir "candidates.json") -Encoding UTF8
$env:PYTHONPATH = $Root
$selectionFile = Join-Path $SelectionDir "selection.json"
$previous = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& python -m src.experiments.select_config --results $resultsDir --base $BaseRun --out $selectionFile `
    --utility-task $UtilityTasks.Split(",")[0] --max-utility-drop $MaxUtilityDrop --max-harm-increase $MaxHarmIncrease
$code = $LASTEXITCODE
$ErrorActionPreference = $previous
if ($code) { throw "selection failed (exit $code)" }
$selection = Get-Content $selectionFile -Raw -Encoding UTF8 | ConvertFrom-Json
$best = $candidates | Where-Object { $_.Name -eq $selection.best } | Select-Object -First 1
if (-not $best) {
    Write-Host "No configuration satisfies the selection rule. Details: $selectionFile" -ForegroundColor Red
    return
}
Write-Host ("Best: {0}  (beta {1}, alpha {2})  ->  {3}" -f $best.Name, $best.Beta, $best.Alpha, $selectionFile) -ForegroundColor Green

# ---------------------------------------------------------------------------
# 4. Upload the best run
# ---------------------------------------------------------------------------
Write-Step "4/4  Upload"
if ($HfRepo) {
    $code = Invoke-Native (Join-Path $SelectionDir "upload.log") @("run", $Launcher, "--stage", "upload", "--run-name",
                                                                  $best.Name, "--hf-repo", $HfRepo, "--data-tag", $DataTag)
    if ($code) { Write-Host "Upload failed (exit $code); retry with: modal run $Launcher --stage upload --run-name $($best.Name) --hf-repo $HfRepo" -ForegroundColor Red }
} else {
    Write-Host "Skipped (no -HfRepo). Upload later with:"
    Write-Host "  modal run $Launcher --stage upload --run-name $($best.Name) --hf-repo <user>/<name>"
}
Write-Host ""
Write-Host "For the paper, evaluate the selected run on the TEST pools (SEAM's attack.json + eval_harm):"
Write-Host "  modal run $Launcher --stage attack-eval --run-name $($best.Name) --eval-mode test --attack-lrs 2e-5,5e-5,8e-5,1e-4,2e-4 --utility-tasks arc_easy,mmlu"
Write-Host "  modal run $Launcher --stage attack-eval --run-name $BaseRun --eval-mode test --attack-lrs 2e-5,5e-5,8e-5,1e-4,2e-4 --utility-tasks arc_easy,mmlu"
