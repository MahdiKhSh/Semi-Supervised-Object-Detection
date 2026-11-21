@echo off
setlocal
cd /d "%~dp0"
echo Q1 PAPER MATRIX (CoRe-SSOD vs Supervised, Mean Teacher, Soft Teacher, Consistent-Teacher).
echo Resolution: 800px | Detector: FasterRCNN-FPN-v2 | Effective batch: 8 (4x2) | Warmup-Cosine LR.
echo Crash-safe and fully resumable (identical effective batch to Kaggle 2x4!).
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "scripts\setup_windows.ps1"
if errorlevel 1 (echo Setup failed.& pause& exit /b 1)
".venv\Scripts\python.exe" scripts\run_suite.py --preset q1_paper --device cuda --amp auto --batch-size 4 --grad-accum-steps 2 --resume --output "work_dirs\q1_paper_runs" %*
if errorlevel 1 (echo Interrupted or failed. Run RUN_STATUS.cmd to inspect or RUN_RESUME.cmd to continue.& pause& exit /b 1)
echo Q1 paper matrix complete. See work_dirs\q1_paper_runs\PAPER_RESULTS.md and docs\PROJECT_PROGRESS.md
pause
