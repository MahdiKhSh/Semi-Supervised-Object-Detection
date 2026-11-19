#requires -Version 5.1
# Windows-native installation into local .venv. No network routing/RDP changes.
param([switch]$SkipTorchInstall, [switch]$AllowCpuOnly)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root
$venvPython = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "Creating Python environment..."
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) {
        & py -3.11 -c "import sys; print(sys.version)"
        if ($LASTEXITCODE -eq 0) { & py -3.11 -m venv .venv }
        else { & py -3.10 -m venv .venv }
    } else { & python -m venv .venv }
    if ($LASTEXITCODE -ne 0) { throw "Unable to create Python virtual environment (requires Python >=3.10)" }
}
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
$installed = $false
& $venvPython -c "import torch, torchvision; assert torch.cuda.is_available(); print(torch.__version__, torchvision.__version__, torch.version.cuda)"
if ($LASTEXITCODE -eq 0) { $installed = $true }
if (-not $installed -and -not $SkipTorchInstall) {
    Write-Host "Installing official CUDA 12.8 PyTorch 2.10 and torchvision 0.25 wheels..."
    & $venvPython -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) { throw "CUDA torch installation failed. Check driver and official PyTorch selector." }
}
& $venvPython -m pip install -e ".[dev,native]"
if ($LASTEXITCODE -ne 0) { throw "Research dependencies could not be installed" }
& $venvPython scripts/check_environment.py
if ($LASTEXITCODE -ne 0) { throw "Environment diagnostic failed" }
& $venvPython -c "import torch; assert torch.cuda.is_available(); assert torch.cuda.get_device_capability(0)[0] >= 12; x=torch.randn((128,128),device='cuda'); assert (x@x).isfinite().all().item(); print('CUDA Blackwell runtime OK')"
if ($LASTEXITCODE -ne 0 -and -not $AllowCpuOnly) { throw "CUDA/RTX 5090 verification failed. No training started." }
& $venvPython -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "Unit tests failed. No training started." }
Write-Host "Environment verified. Setup did not start model training."
