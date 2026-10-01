<#
.SYNOPSIS
  CBG v2 on Llama-2-7B-Chat (2x H200, Modal) from scratch: prepare -> verify -> radius -> smoke -> train (+ HF upload).

.DESCRIPTION
  Runs every stage of scripts/modal_cbg_v2_7b.py in order and stops at the first failure.
  Gates between stages:
    * prepare is skipped when the data tag is already prepared on the volume;
    * radius must report a plateau (otherwise method.radius has to be changed first);
    * after the smoke test the projected time and peak memory are shown and you confirm the
      full training run (the GPU stages bill your Modal account).
  The same -RunName can be launched again: finished stages are re-run cheaply, and `train`
  resumes from its last checkpoint (a finished run is never retrained).

.EXAMPLE
  .\scripts\Run-CBGv2-7B.ps1 -RunName fast-s42 -HfRepo myuser/cbg-v2-llama2-7b
.EXAMPLE
  .\scripts\Run-CBGv2-7B.ps1 -RunName fast-s42 -SmokeOnly
.EXAMPLE
  .\scripts\Run-CBGv2-7B.ps1 -RunName full-s42 -Config config/cbg_v2_llama2_7b_2xh200.yaml   # S0.6 reference
#>
[CmdletBinding()]
param(
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$')]
    [string]$RunName = "fast-s42",
    [string]$Config = "config/cbg_v2_llama2_7b_2xh200_fast.yaml",
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$')]
    [string]$DataTag = "seam4k-v1",
    [int]$SmokeSteps = 6,
    # Override method.radius (writes config/generated/<RunName>.yaml from -Config; comments are kept).
    [double]$Radius = 0,
    # Override execution.hvp_precision (HVP configs only), same generated-config mechanism.
    [ValidateSet("", "fp32", "tf32", "bf16")]
    [string]$HvpPrecision = "",
    # Any other numeric config keys, e.g. @{ beta = 0.03; alpha = 2 } (first "key:" line in -Config).
    [hashtable]$Override = @{},
    # Skip the smoke test before training (only for configs already smoke-tested).
    [switch]$SkipSmoke,
    # Hugging Face upload after training (empty = no upload). The Modal secret HF_TOKEN needs write access.
    [string]$HfRepo = "",
    [switch]$HfPublic,
    [ValidateSet("float32", "bfloat16", "float16")]
    [string]$HfDtype = "float32",
    [switch]$SkipVerify,
    [switch]$SkipRadius,
    # Exact HVP vs finite differences on 7B before smoke (use with an HVP config, e.g.
    # config/cbg_v2_llama2_7b_2xh200_fast_hvp.yaml). Stops when the check fails.
    [switch]$HvpCheck,
    [switch]$SmokeOnly,
    # Start training without the confirmation prompt after the smoke test.
    [switch]$Yes,
    # Keep the local terminal attached to the training logs (otherwise the run is detached).
    [switch]$Attach
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Launcher = "scripts/modal_cbg_v2_7b.py"
$OutputVolume = "cbg-v2-7b-output"
$DataVolume = "cbg-v2-7b-data"
$LogDir = Join-Path $Root "results\modal_runs\$RunName"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# The Modal CLI prints Unicode; keep Windows consoles from failing on it.
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Write-Step([string]$Text) {
    Write-Host ""
    Write-Host ("=" * 78) -ForegroundColor DarkCyan
    Write-Host "  $Text" -ForegroundColor Cyan
    Write-Host ("=" * 78) -ForegroundColor DarkCyan
}

function Invoke-Modal([string]$Name, [string[]]$Arguments) {
    # Native command: stderr (Modal progress) goes straight to the console; stdout is also logged.
    $log = Join-Path $LogDir "$Name.log"
    Write-Host "> modal $($Arguments -join ' ')" -ForegroundColor DarkGray
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & modal @Arguments | Tee-Object -FilePath $log
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
    }
    if ($code -ne 0) {
        throw "stage '$Name' failed (exit code $code); stdout saved to $log"
    }
}

function Get-VolumeJson([string]$RemotePath) {
    # Download one JSON file from the output volume and parse it.
    $target = Join-Path $LogDir ([IO.Path]::GetFileName($RemotePath))
    if (Test-Path $target) { Remove-Item $target -Force }
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & modal volume get $OutputVolume $RemotePath $target --force | Out-Null
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
    }
    if ($code -ne 0 -or -not (Test-Path $target)) {
        throw "could not download $RemotePath from volume $OutputVolume"
    }
    return Get-Content $target -Raw -Encoding UTF8 | ConvertFrom-Json
}

function Test-DataPrepared {
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $listing = & modal volume ls $DataVolume $DataTag 2>$null | Out-String
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
    }
    return ($code -eq 0 -and $listing -match "manifest\.json")
}

# ---------------------------------------------------------------------------
# 0. Pre-flight
# ---------------------------------------------------------------------------
Write-Step "0/5  Pre-flight checks"
if (-not (Get-Command modal -ErrorAction SilentlyContinue)) {
    throw "Modal CLI not found. Install it with 'pip install modal' and run 'modal setup'."
}
if (-not (Test-Path (Join-Path $Root $Config))) {
    throw "config not found: $Config"
}
if ($Radius -gt 0 -or $HvpPrecision -or $Override.Count) {
    # Overrides go into config/generated/<RunName>.yaml; the rest of -Config (and its comments) is kept.
    $text = Get-Content (Join-Path $Root $Config) -Raw -Encoding UTF8
    if ($Radius -gt 0) {
        $pattern = '(?m)^(\s+radius:\s*)[0-9.eE+-]+'
        if ($text -notmatch $pattern) { throw "method.radius not found in $Config" }
        $value = $Radius.ToString("0.0##e+0", [Globalization.CultureInfo]::InvariantCulture)
        $text = [regex]::Replace($text, $pattern, ('${1}' + $value), 1)
        Write-Host "Override: method.radius = $value" -ForegroundColor Yellow
    }
    if ($HvpPrecision) {
        $pattern = '(?m)^(\s+hvp_precision:\s*)[A-Za-z0-9]+'
        if ($text -notmatch $pattern) { throw "execution.hvp_precision not found in $Config (not an HVP config?)" }
        $text = [regex]::Replace($text, $pattern, ('${1}' + $HvpPrecision), 1)
        Write-Host "Override: execution.hvp_precision = $HvpPrecision" -ForegroundColor Yellow
    }
    foreach ($key in $Override.Keys) {
        if ($key -notmatch '^[A-Za-z_][A-Za-z0-9_]*$') { throw "invalid override key '$key'" }
        $pattern = '(?m)^(\s+' + $key + ':\s*)([0-9.eE+-]+)'
        $match = [regex]::Match($text, $pattern)
        if (-not $match.Success) { throw "numeric key '$key' not found in $Config" }
        $number = [double]$Override[$key]
        if ($match.Groups[2].Value -match '[.eE]') {
            # float key (beta, alpha, ...): keep it a YAML float
            $value = $number.ToString("0.0#####e+0", [Globalization.CultureInfo]::InvariantCulture)
            if ($number -ge 0.01 -and $number -lt 1000) {
                $value = $number.ToString("0.0#####", [Globalization.CultureInfo]::InvariantCulture)
            }
        } else {
            # integer key (k, lazy_period, steps, ...)
            if ($number -ne [math]::Floor($number)) { throw "'$key' is an integer in $Config, got $number" }
            $value = ([long]$number).ToString([Globalization.CultureInfo]::InvariantCulture)
        }
        $text = $text.Substring(0, $match.Groups[2].Index) + $value + $text.Substring($match.Groups[2].Index + $match.Groups[2].Length)
        Write-Host "Override: $key = $value" -ForegroundColor Yellow
    }
    $generated = "config/generated/$RunName.yaml"
    New-Item -ItemType Directory -Force -Path (Join-Path $Root "config/generated") | Out-Null
    [IO.File]::WriteAllText((Join-Path $Root $generated), $text, (New-Object System.Text.UTF8Encoding($false)))
    Write-Host "Generated config: $generated" -ForegroundColor Yellow
    $Config = $generated
}
if ($HfRepo -and $HfRepo -notmatch '^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$') {
    throw "-HfRepo must be <user or org>/<name>, got '$HfRepo'"
}
$previous = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$profileName = (& modal profile current 2>$null | Out-String).Trim()
$secrets = & modal secret list 2>$null | Out-String
$ErrorActionPreference = $previous
if (-not $profileName) {
    throw "Modal is not authenticated. Run 'modal setup' first."
}
if ($secrets -notmatch "HF_TOKEN") {
    throw ("Modal secret 'HF_TOKEN' is missing. Create it with:`n" +
           "  modal secret create HF_TOKEN HF_TOKEN=hf_xxx`n" +
           "(the token must have access to meta-llama/Llama-2-7b-chat-hf, and write access if -HfRepo is used)")
}
$gpuSpec = "H200:2"
if ($env:CBG_V2_MODAL_GPU) { $gpuSpec = $env:CBG_V2_MODAL_GPU }
if ($env:CBG_MODAL_GPU) {
    Write-Host "Note: CBG_MODAL_GPU='$env:CBG_MODAL_GPU' (S0 scripts) is ignored by the v2 launcher." -ForegroundColor Yellow
}
Write-Host "Modal profile : $profileName"
Write-Host "GPU per stage : $gpuSpec"
Write-Host "Run name      : $RunName"
Write-Host "Config        : $Config"
Write-Host "Data tag      : $DataTag"
if ($HfRepo) {
    $visibility = "private"
    if ($HfPublic) { $visibility = "public" }
    Write-Host "HF upload     : $HfRepo ($visibility, $HfDtype)"
} else {
    Write-Host "HF upload     : off (pass -HfRepo <user>/<name> to enable)"
}
Write-Host "Local logs    : $LogDir"

$common = @("--config", $Config, "--data-tag", $DataTag)

# ---------------------------------------------------------------------------
# 1. Prepare data + model (CPU)
# ---------------------------------------------------------------------------
Write-Step "1/5  Prepare SEAM data pools and download the model (CPU)"
if (Test-DataPrepared) {
    Write-Host "Data tag '$DataTag' is already prepared on volume $DataVolume - skipping." -ForegroundColor Yellow
} else {
    Invoke-Modal "prepare" (@("run", $Launcher, "--stage", "prepare") + $common)
}

# ---------------------------------------------------------------------------
# 2. Verify (CPU): reference implementation + unit tests
# ---------------------------------------------------------------------------
Write-Step "2/5  Verify the implementation (CPU)"
if ($SkipVerify) {
    Write-Host "Skipped (-SkipVerify)." -ForegroundColor Yellow
} else {
    Invoke-Modal "verify" (@("run", $Launcher, "--stage", "verify") + $common)
}

# ---------------------------------------------------------------------------
# 3. Radius check (2x H200, ~10 min)
# ---------------------------------------------------------------------------
Write-Step "3/5  Finite-difference radius check (2x H200)"
if ($SkipRadius) {
    Write-Host "Skipped (-SkipRadius)." -ForegroundColor Yellow
} else {
    Invoke-Modal "radius" (@("run", $Launcher, "--stage", "radius", "--run-name", $RunName) + $common)
    # Not $radius: PowerShell names are case-insensitive and -Radius is a [double] parameter.
    $check = Get-VolumeJson "$RunName/radius/radius_check.json"
    Write-Host ("  determinism: same r twice differs by {0:G3} (must be 0)" -f $check.repeat_relative_difference)
    Write-Host "  candidate r     worst diff x3   worst diff x1.73   cosine     plateau (x3 / x1.73)"
    foreach ($candidate in $check.candidates) {
        $mark = ""
        if ([math]::Abs($candidate.radius - $check.radius) -le 1e-9 * $check.radius) { $mark = "  <- current" }
        if ([math]::Abs($candidate.radius - $check.recommended_radius) -le 1e-9 * $candidate.radius) { $mark += "  <- best" }
        Write-Host ("  {0,-14:E2}  {1,13:F4}   {2,16:F4}   {3,8:F5}   {4} / {5}{6}" -f $candidate.radius,
                    $candidate.worst_relative_difference, $candidate.worst_relative_difference_narrow,
                    $candidate.worst_cosine, $candidate.plateau, $candidate.plateau_narrow, $mark)
    }
    Write-Host "  diagnosis: $($check.diagnosis)" -ForegroundColor Yellow
    if (-not ($check.plateau -or $check.plateau_narrow)) {
        $hint = ""
        if ($check.recommended_on_plateau) {
            $hint = ("Re-run with:  .\scripts\Run-CBGv2-7B.ps1 -RunName <new-name> -Radius {0:G3} -SkipVerify -SkipRadius -SmokeOnly" -f
                     $check.recommended_radius)
        }
        throw ("radius r = $($check.radius) is NOT on a plateau. $($check.diagnosis). $hint " +
               "Details: $LogDir\radius_check.json")
    }
    if ($check.plateau) {
        Write-Host "Radius plateau OK (r = $($check.radius))." -ForegroundColor Green
    } else {
        Write-Host "Radius OK on the narrow (x1.73) plateau (r = $($check.radius))." -ForegroundColor Green
    }
}

# ---------------------------------------------------------------------------
# 3b. Exact HVP vs finite differences (2x H200, optional)
# ---------------------------------------------------------------------------
if ($HvpCheck) {
    Write-Step "3b/5  Exact HVP vs finite differences (2x H200)"
    Invoke-Modal "hvp-check" (@("run", $Launcher, "--stage", "hvp-check", "--run-name", $RunName) + $common)
    $hvp = Get-VolumeJson "$RunName/hvp-check/hvp_check.json"
    Write-Host "  estimator        seconds   peak GiB (0/1)    diff to hvp_fp32   cos        diff to hvp_$($hvp.hvp_precision)"
    foreach ($row in $hvp.rows) {
        $names = $row.PSObject.Properties.Name
        $seconds = "OOM"
        if ($row.status -eq "ok") { $seconds = "{0:F1}" -f $row.seconds }
        $memory = "-"
        if ($names -contains "peak_memory_gib_0") { $memory = "{0:F0}/{1:F0}" -f $row.peak_memory_gib_0, $row.peak_memory_gib_1 }
        $toExact = "-"; $cosExact = "-"; $toTrain = "-"
        if ($names -contains "relative_difference_to_hvp_fp32") {
            $toExact = "{0:F4}" -f $row.relative_difference_to_hvp_fp32
            $cosExact = "{0:F5}" -f $row.cosine_to_hvp_fp32
        }
        $trainKey = "relative_difference_to_hvp_$($hvp.hvp_precision)"
        if ($names -contains $trainKey) { $toTrain = "{0:F4}" -f $row.$trainKey }
        Write-Host ("  {0,-15} {1,8}   {2,-15}   {3,16}   {4,-9}  {5}" -f $row.label, $seconds, $memory, $toExact, $cosExact, $toTrain)
    }
    Write-Host "  diagnosis: $($hvp.diagnosis)" -ForegroundColor Yellow
    if (-not $hvp.passed) {
        $hint = ""
        if ($hvp.recommended_hvp_precision) {
            $hint = (" Re-run with:  .\scripts\Run-CBGv2-7B.ps1 -RunName <new-name> -Config $Config " +
                     "-HvpPrecision $($hvp.recommended_hvp_precision) -SkipVerify -SkipRadius -SmokeOnly")
        }
        throw "HVP check failed: $($hvp.diagnosis).$hint Details: $LogDir\hvp_check.json"
    }
    Write-Host "HVP check passed." -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# 4. Smoke test (2x H200, a few steps)
# ---------------------------------------------------------------------------
Write-Step "4/5  Smoke test: $SmokeSteps steps (2x H200)"
if ($SkipSmoke -and -not $SmokeOnly) {
    Write-Host "Skipped (-SkipSmoke)." -ForegroundColor Yellow
} else {
    Invoke-Modal "smoke" (@("run", $Launcher, "--stage", "smoke", "--run-name", $RunName,
                            "--smoke-steps", "$SmokeSteps") + $common)
    $smoke = Get-VolumeJson "$RunName/smoke/smoke_summary.json"
    Write-Host ("  CBG step          : {0:F1} s" -f $smoke.seconds_cbg_step)
    if ($null -ne $smoke.seconds_plain_step) {
        Write-Host ("  plain step        : {0:F1} s" -f $smoke.seconds_plain_step)
    }
    Write-Host ("  average per step  : {0:F1} s   (lazy_period {1})" -f $smoke.seconds_per_step_average, $smoke.lazy_period)
    Write-Host ("  projected training: {0:F2} h (without geometry evals / checkpoints)" -f
                $smoke.projected_hours_without_diagnostics)
    foreach ($property in $smoke.peak_memory_gib.PSObject.Properties) {
        Write-Host ("  {0,-22}: {1:F1} GiB" -f $property.Name, $property.Value)
        if ($property.Value -gt 135) {
            Write-Host "    WARNING: close to the 141 GB of an H200; lower execution.microbatch_tokens." -ForegroundColor Yellow
        }
    }
}

if ($SmokeOnly) {
    Write-Step "Done (-SmokeOnly). Summary: $LogDir\smoke_summary.json"
    return
}

# ---------------------------------------------------------------------------
# 5. Training (+ HF upload)
# ---------------------------------------------------------------------------
Write-Step "5/5  Training"
if (-not $Yes) {
    $answer = Read-Host "Start the full training run '$RunName' on 2x H200 now? [y/N]"
    if ($answer -notmatch '^(y|yes)$') {
        Write-Host "Training not started. Re-run with the same -RunName (add -SkipVerify -SkipRadius) when ready."
        return
    }
}
$train = @("run")
if (-not $Attach) { $train += "--detach" }
$train += @($Launcher, "--stage", "train", "--run-name", $RunName) + $common
if ($HfRepo) {
    $train += @("--hf-repo", $HfRepo, "--hf-dtype", $HfDtype)
    if ($HfPublic) { $train += "--hf-public" }
}
Invoke-Modal "train" $train

Write-Step "Training submitted / finished"
Write-Host "Logs on the volume : modal volume get $OutputVolume $RunName/train/train_log.jsonl ."
Write-Host "Live container logs: https://modal.com/apps  (app 'cbg-v2-7b')"
Write-Host "Resume / retry     : run this script again with -RunName $RunName -SkipVerify -SkipRadius -Yes"
if ($HfRepo) {
    Write-Host "HF upload          : happens automatically at the end; retry with"
    Write-Host "  modal run $Launcher --stage upload --run-name $RunName --hf-repo $HfRepo --hf-dtype $HfDtype"
}
