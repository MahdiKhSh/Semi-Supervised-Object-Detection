@echo off
setlocal
cd /d "%~dp0"
echo RESUME PREVIOUS EXPERIMENT SUITE (crash-safe, checkpoint-verified).
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "scripts\setup_windows.ps1"
if errorlevel 1 (echo Setup failed.& pause& exit /b 1)
if exist "work_dirs\last_suite.json" (
  ".venv\Scripts\python.exe" scripts\run_suite.py --resume-last --resume %*
) else if exist "work_dirs\experiment_state.sqlite" (
  ".venv\Scripts\python.exe" scripts\run_suite.py --resume-last --resume %*
) else (
  echo No prior suite recorded in work_dirs\last_suite.json; defaulting to --preset core --resume.
  ".venv\Scripts\python.exe" scripts\run_suite.py --preset core --device cuda --amp auto --resume --output "work_dirs\core_runs" %*
)
if errorlevel 1 (echo Interrupted or failed. Inspect docs\PROJECT_PROGRESS.md or re-run RUN_RESUME.cmd.& pause& exit /b 1)
echo Resumed suite complete. See docs\PROJECT_PROGRESS.md
pause
