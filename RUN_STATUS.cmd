@echo off
setlocal
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
if exist "work_dirs\last_suite.json" (
  "%PY%" scripts\run_suite.py --resume-last --status %*
) else if exist "work_dirs\experiment_state.sqlite" (
  "%PY%" scripts\run_suite.py --resume-last --status %*
) else (
  "%PY%" scripts\run_suite.py --preset core --status --output "work_dirs\core_runs" %*
)
echo.
echo Detailed report: docs\PROJECT_PROGRESS.md
pause
