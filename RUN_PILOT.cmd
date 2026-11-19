@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "scripts\setup_windows.ps1"
if errorlevel 1 (echo Setup failed.& pause& exit /b 1)
".venv\Scripts\python.exe" scripts\run_suite.py --preset pilot --device cuda --output "work_dirs\pilot_runs"
if errorlevel 1 (echo Pilot interrupted; double-click again to resume.& pause& exit /b 1)
echo Pilot completed. Partial validation is diagnostic, NOT publishable final AP.
pause
