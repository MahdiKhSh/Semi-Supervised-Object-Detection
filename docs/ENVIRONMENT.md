# RTX 5090 deployment checklist (Windows / Linux)

The remote hardware cannot be accessed from this chat.

1. Identify OS, driver, GPU and CUDA capability:
   `nvidia-smi`
2. Install a current official NVIDIA Blackwell-supported PyTorch build
   suitable for this driver and Python environment. Do **not** pin an
   old CUDA 11 or torch 1.x stack. Use the official install selector:
   https://pytorch.org/get-started/locally/
3. Verify:
   `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_capability() if torch.cuda.is_available() else 'no CUDA')"`
   RTX 5090 should expose compute capability 12.0.
4. Clone `open-mmlab/mmdetection` at a **recorded tested revision** under
   `third_party/mmdetection`, then install compatible MMEngine and MMCV.
   OpenMMLab wheels and custom CUDA ops must match torch/CUDA.
5. Check `python -c "import mmcv, mmengine, mmdet; print(mmcv.__version__, mmengine.__version__, mmdet.__version__)"`
6. Run CPU tests, then a tiny training smoke test with 2 iterations and a
   low batch size. Check peak VRAM, NaNs, EMA, and checkpoint loading.
7. Do not run GPU-heavy jobs concurrently; close other training processes
   voluntarily but never kill unrelated jobs or alter RDP routes.

## Research and licensing
MMDetection is Apache-2.0. Preserve notices when modifying/reusing upstream
code. Dataset weights and downloadable pretraining checkpoints may have
distinct terms; check them separately. Maintain source commit hashes and
the exact Python package freeze in the final paper appendix.

## Known blocker
Neither GPU execution nor MMCV compatibility on this particular Windows
machine can be verified remotely from the present environment.
