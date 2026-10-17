# =====================================================================
#  AARIS demo rehearsal launcher (Windows / PowerShell)
#
#  Usage:
#    powershell -ExecutionPolicy Bypass -File demo\run_demo.ps1
#    powershell -ExecutionPolicy Bypass -File demo\run_demo.ps1 -Reset
#    powershell -ExecutionPolicy Bypass -File demo\run_demo.ps1 -NoBrowser
#
#  -Reset  : deletes ground/fleet.db, starts fresh services, seeds the
#            demo fleet with ground/seed_demo.py, then POST /schedule
#            once so the plan exists (54 scheduled / 36 deferred over a
#            30-day horizon for the 90-aircraft demo roster).
#
#  Stop everything afterwards with stop.bat in the repo root.
#  Verifies readiness of all three services before returning.
# =====================================================================

param(
    [switch]$Reset,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Py       = Join-Path $RepoRoot '.venv\Scripts\python.exe'
$env:PYTHONIOENCODING = 'utf-8'

if (-not (Test-Path $Py)) {
    Write-Host "[demo] ERROR: repo venv not found at $Py" -ForegroundColor Red
    exit 1
}

function Stop-OnPort([int]$Port) {
    $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    foreach ($c in $conns) {
        Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
    }
}

function Wait-Http([string]$Url, [string]$Name, [int]$Seconds = 40) {
    $deadline = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $r = Invoke-RestMethod -Uri $Url -TimeoutSec 3
            Write-Host ("[demo] {0,-16} ready  ({1})" -f $Name, $Url) -ForegroundColor Green
            return $true
        } catch { Start-Sleep -Milliseconds 700 }
    }
    Write-Host "[demo] ERROR: $Name did not come up at $Url within $Seconds s" -ForegroundColor Red
    return $false
}

# --- optional reset ---------------------------------------------------
if ($Reset) {
    Write-Host "[demo] Reset mode: stopping any old instances and deleting ground\fleet.db"
    Stop-OnPort 8000; Stop-OnPort 8001; Stop-OnPort 8501
    Start-Sleep -Seconds 1
    $db = Join-Path $RepoRoot 'ground\fleet.db'
    if (Test-Path $db) { Remove-Item $db -Force }
    Remove-Item "$db-journal","$db-wal","$db-shm" -Force -ErrorAction SilentlyContinue
}

# --- start the three services ------------------------------------------
Write-Host "[demo] Starting AARIS stack (minimized windows)..."

Start-Process -FilePath $Py `
    -ArgumentList '-m','uvicorn','edge_node:app','--host','127.0.0.1','--port','8000' `
    -WorkingDirectory (Join-Path $RepoRoot 'edge') -WindowStyle Minimized

Start-Process -FilePath $Py `
    -ArgumentList '-m','uvicorn','ground_station:app','--host','127.0.0.1','--port','8001' `
    -WorkingDirectory (Join-Path $RepoRoot 'ground') -WindowStyle Minimized

Start-Process -FilePath $Py `
    -ArgumentList '-m','streamlit','run','dashboard\app.py','--server.headless','true','--server.port','8501' `
    -WorkingDirectory $RepoRoot -WindowStyle Minimized

$okEdge   = Wait-Http 'http://127.0.0.1:8000/health' 'Edge node'
$okGround = Wait-Http 'http://127.0.0.1:8001/health' 'Ground station'
$okDash   = Wait-Http 'http://127.0.0.1:8501/healthz' 'Dashboard'
if (-not ($okEdge -and $okGround -and $okDash)) { exit 1 }

# --- reset mode: seed the demo fleet and generate the plan --------------
if ($Reset) {
    Write-Host "[demo] Seeding demo fleet (ground\seed_demo.py)..."
    Push-Location (Join-Path $RepoRoot 'ground')
    & $Py seed_demo.py
    Pop-Location

    Write-Host "[demo] Logging in as maintainer (POST /auth/login)..."
    $tok = Invoke-RestMethod -Uri 'http://127.0.0.1:8001/auth/login' -Method Post `
            -ContentType 'application/x-www-form-urlencoded' `
            -Body 'username=maintainer&password=maintainer123' -TimeoutSec 30

    Write-Host "[demo] Generating CP-SAT plan (POST /schedule)..."
    $plan = Invoke-RestMethod -Uri 'http://127.0.0.1:8001/schedule' -Method Post `
             -ContentType 'application/json' -Body '{}' -TimeoutSec 60 `
             -Headers @{ Authorization = "Bearer $($tok.access_token)" }
    Write-Host ("[demo] Plan: solver={0} scheduled={1} deferred={2}" -f `
                $plan.solver_status, $plan.scheduled_count, $plan.deferred_count) -ForegroundColor Green
}

# --- URLs ---------------------------------------------------------------
Write-Host ""
Write-Host "  AARIS is up:" -ForegroundColor Cyan
Write-Host "    Edge node       http://127.0.0.1:8000  (docs: /docs, POST /predict)"
Write-Host "    Ground station  http://127.0.0.1:8001  (/fleet, /schedule/latest)"
Write-Host "                    protected: POST /auth/login first"
Write-Host "                    (demo users: maintainer / commander / admin)"
Write-Host "    Dashboard       http://localhost:8501"
Write-Host ""

if (-not $NoBrowser) { Start-Process 'http://localhost:8501' }

Write-Host "[demo] Rehearsal check:" -ForegroundColor Cyan
Write-Host "    & '$Py' '$RepoRoot\demo\verify_pipeline.py'"
Write-Host "[demo] Stop the stack with: stop.bat (repo root)."
