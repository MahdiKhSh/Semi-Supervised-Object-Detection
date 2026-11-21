# Codex Session Handoff & Continuity Protocol

**Target environments**: Windows 11/10 (RTX 5090), Linux GPU servers (L40s / A100 / 5090), Kaggle (T4 / P100), and Google Colab.

## Mandatory Startup Checklist (Every New Session)

At the start of every new Codex / agent session, execute these steps BEFORE modifying code or launching jobs:

1. **Inspect Git status & branch**:
   ```powershell
   git status
   git branch --show-current
   git log --oneline -5
   ```
   Expected active branch: `feat/research-baseline`.
2. **Read the three continuity documents**:
   - `docs/CODEX_HANDOFF.md` (this file)
   - `docs/PROJECT_PROGRESS.md` (live stage counts, checkpoints, ETA, resume command)
   - `docs/DECISIONS.md` (recovery & architecture invariants)
3. **Inspect the persistent experiment registry** without launching training:
   ```powershell
   python scripts/run_suite.py --preset core --status
   ```
   (or double-click `RUN_STATUS.cmd`).
4. **Identify completed, incomplete, and blocked tasks**:
   - Never rerun verified `COMPLETED` stages unless `--clean-restart` is explicitly requested by the user.
   - Stale `RUNNING` entries from crashes/reboots are automatically recovered to `INTERRUPTED`.
5. **Continue from the last unfinished task**:
   ```powershell
   python scripts/run_suite.py --preset core --resume
   ```
   (or `RUN_RESUME.cmd`).
6. **At every meaningful milestone**:
   - Update `docs/PROJECT_PROGRESS.md`, `docs/CODEX_HANDOFF.md`, and `docs/DECISIONS.md`.
   - Commit relevant source changes and lightweight results (`results/<preset>/`, `docs/`), NEVER checkpoints or BDD data.

## Cross-Machine Workflow (Kaggle Tonight -> L40s / RTX 5090 Tomorrow)

### 1. Running on Kaggle / Colab / L40s
- Open `notebooks/run_ssod_cloud.ipynb` or run:
  ```bash
  PRESET=core SYNC_GITHUB=1 bash scripts/run_cloud.sh
  ```
- Before the cloud session ends, export a SHA-256 verified portable bundle:
  ```bash
  python scripts/sync_workspace.py export --output work_dirs/core_runs --bundle /kaggle/working/ssod_bundle.tar.gz --archive
  ```

### 2. Continuing Tomorrow on Windows RTX 5090 (or Linux L40s)
- Pull latest code/results from GitHub:
  ```powershell
  git pull origin feat/research-baseline
  ```
- Verify & import the checkpoint bundle onto the new GPU:
  ```powershell
  python scripts/sync_workspace.py import --bundle path\to\ssod_bundle.tar.gz --output work_dirs\core_runs
  ```
- Check status and continue immediately from the exact last saved checkpoint:
  ```powershell
  python scripts/run_suite.py --preset core --status --output work_dirs\core_runs
  python scripts/run_suite.py --preset core --resume --output work_dirs\core_runs --sync-github
  ```

## Current Status Summary

- **Completed**:
  - Full crash-safe, resumable, idempotent, dependency-aware 15-stage execution architecture (`recovery.py`, `registry.py`, `pipeline.py`, `cloud_sync.py`, `run_suite.py`, `benchmark_gpu.py`, `sync_workspace.py`, Windows `.cmd` launchers, Linux/Cloud runner, Colab/Kaggle notebook).
  - Fixed critical TorchVision class-10 (`traffic sign`) crash in `consistency.py`.
  - Implemented Q1-ready proposed method **`CoRe-SSOD`** (`conformal.py` + `care_ssod.py`: Weighted Conformal Risk Control under zero-metadata photometric visibility shift + scale-aware NWD/JS + Tri-Zone ambiguity background suppression + single-pass FPN KL/GIoU).
  - Implemented strong matched baselines (`baselines.py`: faithful **Soft Teacher ICCV'21** with Box Jitter + Soft-Negative BG Weighting, and **Consistent-Teacher CVPR'23** dynamic per-class 1D GMM).
  - Upgraded training recipe (`fasterrcnn_v2`, `max_side=800`, `warmup_cosine` LR, cross-GPU portable `--grad-accum-steps` with constant `effective_batch_size=8`, `--preset q1_pilot` & `--preset q1_paper`, `RUN_Q1_PAPER.cmd`, `scripts/eval_external.py`, conformal/condition figures in `plot_results.py`, and `docs/Q1_PAPER_PROTOCOL.md`).
- **Next Unfinished Task**: Obtain official BDD100K (or attach Kaggle BDD100K dataset), run `--preset q1_pilot` (~15 min on 5090 / ~45 min on Kaggle T4) to verify AP climb and empirical conformal FDR calibration curves, then launch `--preset q1_paper --resume --sync-github`.
- **Blocked**: Waiting on BDD100K dataset download/attachment (see `docs/Q1_PAPER_PROTOCOL.md`).

## Non-Negotiable Restrictions

- **NEVER** restart completed work from scratch unless the user explicitly passes `--clean-restart`.
- **NEVER** fabricate AP or report synthetic/partial-validation AP as paper metrics.
- **NEVER** commit BDD images, raw label files, `.pth` checkpoints, SQLite DBs, or API tokens to GitHub.
- **NEVER** alter networking, proxies, or RDP routes, or kill unrelated processes.
- Preserve quarantine files and failure logs for forensic debugging.
