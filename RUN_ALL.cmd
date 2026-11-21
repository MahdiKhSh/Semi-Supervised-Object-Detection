@echo off
setlocal
cd /d "%~dp0"
echo FULL PAPER MATRIX: 5%%/10%% labels, three seeds, 15 ablation variants.
echo Automatically detects and resumes unfinished work from SQLite + verified checkpoints.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "scripts\setup_windows.ps1"
if errorlevel 1 (echo Setup failed.& pause& exit /b 1)
".venv\Scripts\python.exe" scripts\run_suite.py --preset paper --device cuda --amp auto --resume --output "work_dirs\paper_runs" %*
if errorlevel 1 (echo Interrupted or failed. Run RUN_STATUS.cmd to inspect or RUN_RESUME.cmd to continue.& pause& exit /b 1)
echo Paper matrix complete. See work_dirs\paper_runs\PAPER_RESULTS.md and docs\PROJECT_PROGRESS.md
pause
