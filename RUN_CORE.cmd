@echo off
setlocal
cd /d "%~dp0"
echo PAPER CORE: 5%%/10%%, 3 seeds, main baselines + single-seed ablations.
echo Still lengthy. All stages run sequentially and resume from checkpoints.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "scripts\setup_windows.ps1"
if errorlevel 1 (echo Setup failed.& pause& exit /b 1)
".venv\Scripts\python.exe" scripts\run_suite.py --preset core --device cuda --output "work_dirs\core_runs"
if errorlevel 1 (echo Interrupted. Check logs; double-click to resume.& pause& exit /b 1)
echo Paper core complete. See work_dirs\core_runs\PAPER_RESULTS.md
pause
