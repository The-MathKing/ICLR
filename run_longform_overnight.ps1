# Long-context whole-graph run for the reviewer's long-form item.
#
#   powershell -ExecutionPolicy Bypass -File .\run_longform_overnight.ps1
#
# Writes progress to logs\longform_<timestamp>.log as it goes (no pipe buffering, so the
# file is readable while the run is in flight), and a one-line status to
# logs\longform_status.txt that says where it got to and whether it finished.
#
# Safe to stop and restart: extract.py skips a setting whose output already exists, so
# delete sinktda_out\ragtruth_qwen1.5b_long first if you want a clean run.

param(
    [int]$Rows    = 500,
    [int]$MaxLen  = 2048,
    [int]$Workers = 2,
    [string]$Model = "qwen1.5b"
)

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

New-Item -ItemType Directory -Force "logs" | Out-Null
$stamp  = Get-Date -Format "yyyyMMdd-HHmmss"
$log    = "logs\longform-$stamp.log"
$status = "logs\longform_status.txt"

$env:HF_HUB_OFFLINE          = "0"
$env:PYTHONUNBUFFERED        = "1"
$env:OPENBLAS_NUM_THREADS    = "1"
$env:OMP_NUM_THREADS         = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

"started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  rows=$Rows maxlen=$MaxLen model=$Model" |
    Tee-Object -FilePath $status | Write-Output

$t0 = Get-Date

# Redirect rather than pipe: a pipeline into Select-Object/Tee buffers until the process
# exits, which is what made the last attempt impossible to monitor.
$p = Start-Process -FilePath "python" `
        -ArgumentList @("-m", "sinktda.extract",
                        "--bench", "ragtruth",
                        "--model", $Model,
                        "--n", $Rows,
                        "--tag", "long",
                        "--no-perhead",
                        "--max-len", $MaxLen,
                        "--maxdim", "0",
                        "--workers", $Workers) `
        -RedirectStandardOutput $log `
        -RedirectStandardError "$log.err" `
        -NoNewWindow -PassThru

# Touching .Handle before waiting is what makes .ExitCode readable afterwards: without
# it PowerShell disposes the underlying handle and ExitCode comes back empty, which is
# why the first overnight run reported FAILED after finishing perfectly well.
$null = $p.Handle
$p.WaitForExit()
$mins = ((Get-Date) - $t0).TotalMinutes

$last = if (Test-Path $log) { (Get-Content $log -Tail 1) } else { "" }
$done = Test-Path "sinktda_out\ragtruth_${Model}_long\layers.parquet"

# The parquet existing is the real evidence the run worked; the exit code is a
# secondary check, and an unreadable one must not by itself condemn a good run.
$code = $p.ExitCode
$verdict = if ($done -and ($code -eq 0 -or $null -eq $code)) { "OK" } else { "FAILED" }
$line = "$verdict exit=$code after {0:N0} min | last: $last" -f $mins
$line | Tee-Object -FilePath $status | Write-Output

if ($verdict -ne "OK" -and (Test-Path "$log.err")) {
    "--- stderr tail ---"
    Get-Content "$log.err" -Tail 15
}
