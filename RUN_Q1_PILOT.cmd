@echo off
setlocal
cd /d "%~dp0"
echo Q1 PILOT v2 (CoRe-SSOD v2 vs Supervised, Mean Teacher, Soft Teacher, Consistent-Teacher).
echo Detector: FasterRCNN-FPN-v2 (FrozenBN + GroupNorm + EMAN) | 800px | Effective batch: 8 (4x2).
echo Warmup: 1500 steps (warmup_constant) | Train: 3000 steps (warmup_cosine, EMA=0.996, unsup ramp=300).
echo Output: work_dirs\q1_pilot_v2_runs (preserves prior work_dirs\q1_pilot_runs for comparison).
if not exist ".venv\Scripts\python.exe" (
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "scripts\setup_windows.ps1"
    if errorlevel 1 (echo Setup failed.& pause& exit /b 1)
)
set "DATA_ARGS="
if exist "..\Dataset\converted\labels_train.json" (
    set "DATA_ARGS=--train-labels ..\Dataset\converted\labels_train.json --val-labels ..\Dataset\converted\labels_val.json --train-images ..\Dataset\100k\train --val-images ..\Dataset\100k\val"
)
".venv\Scripts\python.exe" scripts\run_suite.py --preset q1_pilot --device cuda --amp auto --batch-size 4 --grad-accum-steps 2 --resume --output "work_dirs\q1_pilot_v2_runs" %DATA_ARGS% %*
if errorlevel 1 (echo Interrupted or failed. Re-run RUN_Q1_PILOT.cmd to resume cleanly.& pause& exit /b 1)
echo Q1 pilot v2 complete. See work_dirs\q1_pilot_v2_runs\PAPER_RESULTS.md and work_dirs\q1_pilot_v2_runs\figures\
pause
