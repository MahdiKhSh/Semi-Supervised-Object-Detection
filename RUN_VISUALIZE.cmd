@echo off
setlocal
cd /d "%~dp0"
set "CKPT=work_dirs\core_runs\runs\10pct_s42\full\last.pth"
if not exist "%CKPT%" (
  echo First complete RUN_CORE.cmd; checkpoint not found:
  echo %CKPT%
  pause
  exit /b 1
)
".venv\Scripts\python.exe" scripts\visualize_predictions.py --checkpoint "%CKPT%" --images "data\bdd100k\images\val" --output "work_dirs\qualitative" --max-images 50 --max-side 512 --threshold 0.5 --device cuda
if errorlevel 1 (echo Prediction visualization failed.& pause& exit /b 1)
echo Saved real prediction overlays in work_dirs\qualitative
pause
