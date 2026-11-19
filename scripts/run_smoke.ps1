# PowerShell 5.1-safe checks; does not install anything or start training.
$ErrorActionPreference = "Stop"
Write-Host "== GPU =="
nvidia-smi
Write-Host "== Python and dependency health =="
python --version
python scripts/check_environment.py
Write-Host "== CPU tests =="
python -m pytest -q
Write-Host "== Candidate MMDetection configs =="
python -c "from mmengine.config import Config; [print(p, Config.fromfile(p).get('model', {}).get('type')) for p in ['configs/bdd_softteacher.py', 'configs/bdd_reliability.py', 'configs/bdd_supervised.py']]"
Write-Host "No GPU training was launched."
