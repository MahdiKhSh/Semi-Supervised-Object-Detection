@echo off
setlocal
cd /d "%~dp0"
echo GPU BENCHMARK ONLY (resumable, no training launched).
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "scripts\setup_windows.ps1"
if errorlevel 1 (echo Setup failed.& pause& exit /b 1)
".venv\Scripts\python.exe" scripts\benchmark_gpu.py --device auto --amp auto --max-side 512 --batch-size 1 --steps 5 --db-path "work_dirs\experiment_state.sqlite" --output "work_dirs" %*
if errorlevel 1 (echo Benchmark failed.& pause& exit /b 1)
echo Saved resumable benchmark results to work_dirs\runtime_estimates.json and work_dirs\experiment_state.sqlite
pause
