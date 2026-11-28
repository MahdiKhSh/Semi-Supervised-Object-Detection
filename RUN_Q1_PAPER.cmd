@echo off
setlocal
cd /d "%~dp0"
echo Q1 PAPER MATRIX (CoRe-SSOD v2 vs Supervised, Mean Teacher, Soft Teacher, Consistent-Teacher).
echo Resolution: 800px | Detector: FasterRCNN-FPN-v2 (FrozenBN+GN+EMAN) | Effective batch: 8 (4x2).
echo Crash-safe and fully resumable (identical effective batch & GroupNorm math to Kaggle 2x4!).
if not exist ".venv\Scripts\python.exe" (
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "scripts\setup_windows.ps1"
    if errorlevel 1 (echo Setup failed.& pause& exit /b 1)
)
set "DATA_ARGS="
if exist "..\Dataset\converted\labels_train.json" (
    set "DATA_ARGS=--train-labels ..\Dataset\converted\labels_train.json --val-labels ..\Dataset\converted\labels_val.json --train-images ..\Dataset\100k\train --val-images ..\Dataset\100k\val"
)
".venv\Scripts\python.exe" scripts\run_suite.py --preset q1_paper --device cuda --amp auto --batch-size 4 --grad-accum-steps 2 --resume --output "work_dirs\q1_paper_runs" %DATA_ARGS% %*
if errorlevel 1 (echo Interrupted or failed. Run RUN_STATUS.cmd to inspect or RUN_RESUME.cmd to continue.& pause& exit /b 1)
echo Q1 paper matrix complete. See work_dirs\q1_paper_runs\PAPER_RESULTS.md and docs\PROJECT_PROGRESS.md
pause
