@echo off
REM ============================================================
REM  AARIS - start all three services (Edge, Ground, Dashboard)
REM  Usage:  start.bat                   (opens dashboard in browser too)
REM          start.bat --no-browser      (services only, no browser)
REM          start.bat --live-feed       (also run the AC-017 live reporter)
REM          start.bat --fleet-feed      (also run one live feed per aircraft)
REM  Flags combine:  start.bat --live-feed --no-browser
REM  Stop everything with: stop.bat
REM ============================================================

REM Short (no-spaces) form of this folder so nested cmd /c "..." is safe.
set SR=%~sdp0
set PY=%SR%.venv\Scripts\python.exe
set PYTHONIOENCODING=utf-8

REM Scan every argument so flag order does not matter.
set LIVE_FEED=0
set FLEET_FEED=0
set NO_BROWSER=0
:parseargs
if "%~1"=="" goto :argsdone
if /i "%~1"=="--live-feed"   set LIVE_FEED=1
if /i "%~1"=="--fleet-feed"  set FLEET_FEED=1
if /i "%~1"=="--no-browser"  set NO_BROWSER=1
shift
goto :parseargs
:argsdone

if not exist "%~dp0.venv\Scripts\python.exe" (
    echo [AARIS] ERROR: venv python not found in "%~dp0.venv"
    echo             Create the venv first - see dashboard\README.md
    exit /b 1
)

echo [AARIS] Starting Edge node      ^> http://127.0.0.1:8000
start "AARIS Edge (8000)" /min cmd /c "cd /d %SR%edge && %PY% -m uvicorn edge_node:app --host 127.0.0.1 --port 8000"

echo [AARIS] Starting Ground station ^> http://127.0.0.1:8001
start "AARIS Ground (8001)" /min cmd /c "cd /d %SR%ground && %PY% -m uvicorn ground_station:app --host 127.0.0.1 --port 8001"

echo [AARIS] Starting Dashboard      ^> http://localhost:8501
start "AARIS Dashboard (8501)" /min cmd /c "cd /d %SR% && %PY% -m streamlit run dashboard\app.py --server.headless true --server.port 8501"

echo [AARIS] Waiting for services to come up...
ping /n 7 127.0.0.1 >nul

if "%FLEET_FEED%"=="1" goto :fleetfeed
if "%LIVE_FEED%"=="1" goto :livefeed
goto :skipfeed

:livefeed
REM Rehearsal feed: AC-017 streams SYNTHETIC readings to the local ground station
REM (20 s routine tick, 5 s while degraded/critical) so the live profile updates
REM and a CRITICAL alert fires within ~2-3 minutes. No real operational data.
echo [AARIS] Starting live feed        ^> AC-017 reporter (synthetic, --fast --ramp)
start "AARIS Live Feed (AC-017)" /min cmd /c "cd /d %SR%edge && %PY% -u live_reporter.py --ground-url http://127.0.0.1:8001 --fast --ramp"
goto :skipfeed

:fleetfeed
REM Every aircraft on the roster gets its own feed, each authenticating with its
REM own device token, staggered in time and started at a different point of the
REM synthetic timeline - so all 90 profiles update live and critical alerts arrive
REM one at a time instead of all at once. OVERWRITES the canonical AC-017 demo
REM reading (112.76 / LOW) and moves the plan/KPI numbers; restore with
REM "powershell -File demo\run_demo.ps1 -Reset -NoBrowser" when the demo is over.
echo [AARIS] Starting fleet feed       ^> AC-001..AC-090 reporters (synthetic, staggered)
start "AARIS Fleet Feed" /min cmd /c "cd /d %SR%edge && %PY% -u live_reporter.py --ground-url http://127.0.0.1:8001 --fast --ramp --fleet"
:skipfeed

if "%NO_BROWSER%"=="1" goto :skipbrowser
start "" http://localhost:8501
:skipbrowser

echo.
echo [AARIS] Services launched in minimized windows:
echo     Edge node       :8000   (POST /predict)
echo     Ground station  :8001   (protected: POST /auth/login first, then
echo                              /fleet, /models, /schedule, /schedule/latest, /alerts)
echo     Dashboard       :8501   (Streamlit - DEMO MODE works with no credentials;
echo                              LIVE mode: sign in with ID + password, e.g.
echo                              commander / commander123)
if "%FLEET_FEED%"=="1" goto :sumfleet
if "%LIVE_FEED%"=="1" goto :sumsingle
goto :sumnone

:sumsingle
echo     Live feed       AC-017 synthetic reports - watch the dashboard
echo                     Aircraft Detail update and the critical banner.
goto :sumdone

:sumfleet
echo     Fleet feed      AC-001..AC-090 synthetic reports, each aircraft using
echo                     its own device token - watch every Aircraft Detail
echo                     update live and critical alerts arrive staggered.
echo                     NOTE: this replaces the canonical AC-017 demo reading
echo                     (112.76 / LOW) until you reseed with run_demo.ps1 -Reset.
goto :sumdone

:sumnone
echo.
echo     Live feed not running. Re-run as "start.bat --live-feed" to make
echo     AC-017 report every 15 min (rehearsal: compressed to 20 s ticks)
echo     and push urgent CRITICAL alerts to the dashboard banner, or
echo     "start.bat --fleet-feed" to feed all 90 aircraft at once.

:sumdone
echo.
echo [AARIS] Run stop.bat to shut everything down (live feed included).
