@echo off
REM ============================================================
REM  AARIS - stop Edge (8000), Ground (8001), Dashboard (8501)
REM  Kills whatever listens on those ports, then closes any
REM  leftover "AARIS *" console windows - including the live
REM  reporter, which owns no port and so needs its own cleanup.
REM ============================================================

echo [AARIS] Stopping services...

for /f "tokens=5" %%a in ('netstat -aon ^| findstr "LISTENING" ^| findstr /C:":8000 "') do taskkill /F /PID %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon ^| findstr "LISTENING" ^| findstr /C:":8001 "') do taskkill /F /PID %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon ^| findstr "LISTENING" ^| findstr /C:":8501 "') do taskkill /F /PID %%a >nul 2>&1

REM The live reporter (single-aircraft or --fleet) is a python child of an
REM "AARIS ... Feed" cmd window.
REM Killing the window alone leaves the python orphaned (it keeps pushing
REM reports), so match the command line too - but ONLY on python.exe. A broad
REM CommandLine match is self-destructive: shells or tools whose own command
REM line merely mentions live_reporter.py would be killed as well.
taskkill /F /FI "WINDOWTITLE eq AARIS Edge*"        >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq AARIS Ground*"      >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq AARIS Dashboard*"   >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq AARIS Live Feed*"   >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq AARIS Fleet Feed*"  >nul 2>&1
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*live_reporter.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }" >nul 2>&1

echo [AARIS] Edge, Ground, Dashboard and live feed stopped (ports 8000/8001/8501 freed).
