# Project Progress

_Initialized before GPU experiment launch. This file is automatically updated by `scripts/run_suite.py` (including `--status` and after every completed stage)._

## Overview

| Metric | Value |
|---|---|
| Active Preset | `core` (not yet started on real BDD100K) |
| Total Stages / Experiments | **0 recorded in `work_dirs/experiment_state.sqlite` yet** |
| Completed | **0** |
| Running | **0** |
| Interrupted (Resumable) | **0** |
| Failed (Retryable) | **0** |
| Pending | **0** |
| Completed Training Steps | **0 / 0** (0.00%) |
| Accumulated Runtime | `0.0s` |
| Measured Step Speed | `None s/step` (unmeasured — run `RUN_BENCHMARK.cmd` or `python scripts/benchmark_gpu.py`) |
| Estimated Remaining Runtime | `unmeasured (run RUN_BENCHMARK.cmd or complete steps)` |
| Last Successful Checkpoint | None yet |

## Infrastructure & Recovery Readiness

- [x] Transactional SQLite experiment registry (`ssod_research/registry.py`)
- [x] Atomic rolling checkpoints with SHA-256 verification & corruption fallback (`ssod_research/recovery.py`)
- [x] Full state checkpointing (student, teacher/EMA, optimizer, LR scheduler, GradScaler, RNG, sampler, algorithm state) in `ssod_research/native_training.py`
- [x] Dependency-aware 15-stage pipeline DAG & selective downstream invalidation (`ssod_research/pipeline.py`, `scripts/run_suite.py`)
- [x] Resumable evaluation & atomic preprocessing (`ssod_research/native_training.py`, `scripts/prepare_bdd100k.py`)
- [x] Cross-GPU Kaggle / Colab / L40s / RTX 5090 bundle transfer & lightweight GitHub results sync (`ssod_research/cloud_sync.py`, `scripts/sync_workspace.py`, `scripts/run_cloud.sh`, `notebooks/run_ssod_cloud.ipynb`)
- [x] Windows launchers (`RUN_CORE.cmd`, `RUN_ALL.cmd`, `RUN_RESUME.cmd`, `RUN_STATUS.cmd`, `RUN_BENCHMARK.cmd`)
- [ ] Real BDD100K GPU benchmark and `core` / `paper` training runs

## Resume Command

```powershell
python scripts/run_suite.py --preset core --resume
```
