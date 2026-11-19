@echo off
setlocal
cd /d "%~dp0"
echo FULL PAPER MATRIX: 5%%/10%% labels, three seeds, 15 ablation variants.
echo This may use the GPU for MANY DAYS. Runs are sequential and resumable.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "scripts\setup_windows.ps1"
if errorlevel 1 (echo Setup failed.& pause& exit /b 1)
".venv\Scripts\python.exe" scripts\run_suite.py --preset paper --device cuda --output "work_dirs\paper_runs"
if errorlevel 1 (echo Interrupted: review logs then double-click to resume.& pause& exit /b 1)
echo Paper matrix complete. Inspect work_dirs\paper_runs\results.csv
pause
