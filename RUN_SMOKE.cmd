@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "scripts\setup_windows.ps1"
if errorlevel 1 (echo Setup failed.& pause& exit /b 1)
".venv\Scripts\python.exe" scripts\run_suite.py --preset smoke --device cuda --output "work_dirs\smoke"
if errorlevel 1 (echo Smoke failed. See work_dirs\smoke\logs.& pause& exit /b 1)
echo Smoke completed on SYNTHETIC test data. NOT research results.
pause
