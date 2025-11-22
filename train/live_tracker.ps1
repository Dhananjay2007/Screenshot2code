param(
    [int]$TrainPID
)

if (-not $TrainPID) {
    Write-Host "USAGE:  powershell -file live_tracker.ps1 -TrainPID <python_pid>"
    exit
}

function get-proc {
    try { return Get-Process -Id $TrainPID -ErrorAction Stop }
    catch { return $null }
}

$proc = get-proc
if ($null -eq $proc) {
    Write-Host "Process not found."
    exit
}

Write-Host "Tracking Python PID $TrainPID ..."
Write-Host "Using CPU-time-based step estimation."

# CONFIG
$sec_per_step = 14.0     # adjust if needed
$total_steps  = 2695
$last_cpu = $proc.CPU
$last_change = Get-Date

while ($true) {

    $proc = get-proc
    if ($null -eq $proc) {
        Write-Host "Process ended."
        break
    }

    $cpu = $proc.CPU
    $delta = $cpu - $last_cpu

    if ($delta -gt 0.1) {
        $last_change = Get-Date
        $last_cpu = $cpu
    }

    $steps_est = [math]::Floor($cpu / $sec_per_step)
    if ($steps_est -lt 0) { $steps_est = 0 }
    if ($steps_est -gt $total_steps) { $steps_est = $total_steps }

    $pct = [math]::Round(($steps_est / $total_steps) * 100, 2)

    $remaining = $total_steps - $steps_est
    $eta_secs = $remaining * $sec_per_step
    $eta_time = (Get-Date).AddSeconds($eta_secs).ToString("HH:mm:ss")

    $silent_secs = (New-TimeSpan -Start $last_change -End (Get-Date)).TotalSeconds
    $freeze = ""
    if ($silent_secs -gt 180) {
        $freeze = "WARNING: No CPU change for $silent_secs seconds (possible stall)"
    }

    Clear-Host
    Write-Host "PID: $TrainPID"
    Write-Host "CPU Time: $cpu sec  (+$([math]::Round($delta, 3)))"
    Write-Host "Estimated Step: $steps_est / $total_steps   ($pct`%)"
    Write-Host "ETA: $eta_time"
    Write-Host "Silent seconds: $silent_secs"
    if ($freeze -ne "") { Write-Host $freeze }

    Start-Sleep -Seconds 2
}
