<#
.SYNOPSIS
    Start (or stop) the call-analysis-pipeline API server on port 8010.

.DESCRIPTION
    Runs `uvicorn api:app --host 0.0.0.0 --port 8010` in the foreground of the
    current window, from the repo folder, with PYTHONIOENCODING=utf-8.

    Pre-checks before starting:
      - Refuses to start if anything already listens on port 8010 (prints PID and name).
      - Prints free memory and warns when commit headroom is below 6 GB.
      - Checks the Python interpreter can import uvicorn and fastapi.

.PARAMETER LogFile
    Also append the server output to output\server_8010.log (UTF-8). The output
    still appears live in the window.

.PARAMETER Stop
    Stop the server listening on port 8010, but only if its command line shows it
    is this project's uvicorn (`uvicorn ... api:app`). Anything else is refused.

.EXAMPLE
    .\start_server.ps1
    .\start_server.ps1 -LogFile
    .\start_server.ps1 -Stop
#>
[CmdletBinding()]
param(
    [switch]$LogFile,
    [switch]$Stop
)

$ErrorActionPreference = "Stop"

$Port = 8010
$RepoRoot = $PSScriptRoot

# Commit headroom (RAM + page file) below which a full job is at risk.
# Measured in KNOWN-ISSUES.md: a job's peak process commit is ~5.4 GB (Stage 2,
# plus ~100 MB per hour of audio), and Stage 3 fails to load Whisper medium below
# ~3 GB free. Windows fails allocations on commit, not on physical RAM, so commit
# is what gets the threshold; physical RAM is printed for information only (this
# machine routinely runs jobs fine at ~1 GB available physical).
$MinFreeCommitGB = 6.0

# Global Python 3.12 is the interpreter this project's packages live in; a
# Python 3.11 on PATH can shadow it (see CLAUDE.md).
$Python312 = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"

function Get-PortOwners {
    $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    foreach ($procId in @($conns | Select-Object -ExpandProperty OwningProcess -Unique)) {
        $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$procId" -ErrorAction SilentlyContinue
        [pscustomobject]@{
            Id          = $procId
            Name        = if ($proc) { $proc.Name } else { "<unknown>" }
            CommandLine = if ($proc) { $proc.CommandLine } else { $null }
        }
    }
}

function Test-IsThisProjectUvicorn([string]$CommandLine) {
    if (-not $CommandLine) { return $false }
    return ($CommandLine -match '(?i)\buvicorn(\.exe)?\b') -and
           ($CommandLine -match '(^|\s|")api:app(\s|"|$)') -and
           ($CommandLine -match "--port[\s=]+$Port\b")
}

# ---------------------------------------------------------------- -Stop
if ($Stop) {
    $owners = @(Get-PortOwners)
    if ($owners.Count -eq 0) {
        Write-Host "Nothing is listening on port $Port. Nothing to stop."
        exit 0
    }
    $refused = $false
    foreach ($o in $owners) {
        if (Test-IsThisProjectUvicorn $o.CommandLine) {
            Write-Host "Stopping PID $($o.Id) ($($o.Name)): $($o.CommandLine)"
            Stop-Process -Id $o.Id -Force
        } else {
            $cmd = if ($o.CommandLine) { $o.CommandLine } else { "<command line not readable>" }
            Write-Host "REFUSED: PID $($o.Id) ($($o.Name)) holds port $Port but is not this project's uvicorn." -ForegroundColor Red
            Write-Host "  Command line: $cmd"
            $refused = $true
        }
    }
    if ($refused) { exit 1 }

    # Confirm the port is actually released.
    $deadline = (Get-Date).AddSeconds(10)
    while ((Get-Date) -lt $deadline -and @(Get-PortOwners).Count -gt 0) {
        Start-Sleep -Milliseconds 250
    }
    if (@(Get-PortOwners).Count -gt 0) {
        Write-Host "Port $Port is still in use after stopping. Check Task Manager." -ForegroundColor Red
        exit 1
    }
    Write-Host "Server stopped. Port $Port is free."
    exit 0
}

# ---------------------------------------------------------------- pre-checks
Set-Location -LiteralPath $RepoRoot
Write-Host "Repo:   $RepoRoot"

# 1. Port must be free.
$owners = @(Get-PortOwners)
if ($owners.Count -gt 0) {
    Write-Host "Port $Port is already in use. Not starting a second server." -ForegroundColor Red
    foreach ($o in $owners) {
        Write-Host "  PID $($o.Id) ($($o.Name))"
        if ($o.CommandLine) { Write-Host "  Command line: $($o.CommandLine)" }
    }
    Write-Host "If that is this project's server, stop it with: .\start_server.ps1 -Stop"
    exit 1
}

# 2. Memory.
$os = Get-CimInstance Win32_OperatingSystem
$freePhysGB   = $os.FreePhysicalMemory / 1MB    # values are in KB
$totalPhysGB  = $os.TotalVisibleMemorySize / 1MB
$freeCommitGB = $os.FreeVirtualMemory / 1MB
Write-Host ("Memory: {0:N1} GB of {1:N1} GB RAM available; {2:N1} GB commit headroom (RAM + page file)" -f `
    $freePhysGB, $totalPhysGB, $freeCommitGB)
if ($freeCommitGB -lt $MinFreeCommitGB) {
    Write-Host ("WARNING: commit headroom is below {0:N0} GB. A full job peaks at ~5.4 GB; Stage 2 or 3 may fail" -f $MinFreeCommitGB) -ForegroundColor Yellow
    Write-Host "         with MemoryError, CUDA out of memory or WinError 1455. Close other apps before running a job." -ForegroundColor Yellow
}

# 3. Interpreter.
$Python = if (Test-Path -LiteralPath $Python312) { $Python312 } else { "python" }
& $Python -c "import uvicorn, fastapi" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python at '$Python' cannot import uvicorn/fastapi." -ForegroundColor Red
    Write-Host "Install with: pip install fastapi uvicorn python-multipart (see CLAUDE.md for the full install order)."
    exit 1
}
Write-Host "Python: $Python"

# ---------------------------------------------------------------- start
$env:PYTHONIOENCODING = "utf-8"
$uvicornArgs = @("-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "$Port")

if (-not $LogFile) {
    Write-Host "Starting uvicorn on port $Port. Press Ctrl+C to stop."
    Write-Host ""
    & $Python @uvicornArgs
    exit $LASTEXITCODE
}

$logDir  = Join-Path $RepoRoot "output"
$logPath = Join-Path $logDir "server_$Port.log"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# Piped stdout is block-buffered in Python; unbuffer it so the window and the log
# stay live. Decode the child's UTF-8 output as UTF-8 rather than the OEM codepage.
$env:PYTHONUNBUFFERED = "1"
$prevOutputEncoding = [Console]::OutputEncoding
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false

$writer = New-Object System.IO.StreamWriter($logPath, $true, (New-Object System.Text.UTF8Encoding $false))
$writer.AutoFlush = $true
$writer.WriteLine("===== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') start_server.ps1: uvicorn api:app --port $Port =====")

Write-Host "Starting uvicorn on port $Port, logging to $logPath. Press Ctrl+C to stop."
Write-Host ""
try {
    # uvicorn logs to stderr; with "Stop" the first stderr line would abort the pipeline.
    $ErrorActionPreference = "Continue"
    & $Python @uvicornArgs 2>&1 | ForEach-Object {
        $line = if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.Exception.Message } else { "$_" }
        Write-Host $line
        $writer.WriteLine($line)
    }
    $exitCode = $LASTEXITCODE
} finally {
    $writer.WriteLine("===== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') server exited =====")
    $writer.Dispose()
    [Console]::OutputEncoding = $prevOutputEncoding
}
exit $exitCode
