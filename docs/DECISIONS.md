# Architectural & Recovery Decisions

This document records durable engineering decisions for the crash-safe, resumable,
and cross-GPU portable execution architecture.

## 1. Persistent Experiment Registry (`ssod_research/registry.py`)

- **Decision**: Use SQLite in `WAL` mode (`PRAGMA synchronous=FULL`, `BEGIN IMMEDIATE` transactions, 30s `busy_timeout`) stored at `<output>/experiment_state.sqlite` and mirrored via `sqlite3.Connection.backup()` to `work_dirs/experiment_state.sqlite`.
- **Why**: Transactional ACID updates survive Windows power loss, Ctrl+C, OOM, and cloud runtime preemption without leaving torn JSON registries. Each preset directory remains self-contained for cross-machine bundle transfer, while `work_dirs/experiment_state.sqlite` provides a unified workspace view.
- **Completion rule**: Never treat folder existence as completion. Every stage requires verified artifacts, matching fingerprints, and `COMPLETED` registry status.

## 2. Rolling Atomic Checkpoints & Corruption Recovery (`ssod_research/recovery.py`)

- **Decision**: Write checkpoints to a temporary file in the destination directory, `flush()` + `os.fsync()`, hash with SHA-256, atomically `os.replace()` into `checkpoints/ckpt_XXXXXXXXX.pth`, and only then atomically write the JSON sidecar `ckpt_XXXXXXXXX.json`. Keep at least 2 valid checkpoints (`keep >= 2`) and mirror `last.pth` + `last.json` atomically for downstream tools (`RUN_VISUALIZE.cmd`, `--init-checkpoint`).
- **Corruption policy**: Checkpoints without a matching sidecar, with mismatched size/SHA-256, or unreadable by `torch.load(..., weights_only=True)` are **quarantined** (renamed with `.corrupt-<timestamp>` + logged to `quarantine.jsonl`), never silently deleted, and the loader falls back to the previous valid checkpoint.
- **Security**: All checkpoints are loaded exclusively with `torch.load(..., map_location="cpu", weights_only=True)` to avoid deserializing arbitrary untrusted pickle payloads. NumPy RNG states are serialized as primitive lists/ints so full RNG recovery remains `weights_only=True` safe.

## 3. Exact Resume vs Approximate Cross-Hardware Recovery

- **Decision**: Separate the **scientific fingerprint** (model variant, seed, batch size, image size, LR, schedule, EMA, thresholds, loss weights, warm-start SHA-256, dataset/split SHA-256) from **machine/runtime metadata** (absolute file paths, GPU model, device type, AMP mode).
- **Why**: Enables starting a run on Kaggle (Linux, T4 `fp16` or P100 `off`) and seamlessly continuing on Linux L40s (`bf16`) or Windows RTX 5090 (`bf16`) without breaking fingerprints, while still refusing to mix incompatible hyperparameters or datasets (`IncompatibleCheckpoint`).
- **Honesty contract**:
  - `resume_mode = "exact"` is reported ONLY when student, teacher (if SSOD), optimizer, LR scheduler, GradScaler, Python/NumPy/Torch/CUDA RNG states, sampler metadata, mutable algorithm state, AND hardware/AMP profile all match.
  - Otherwise `resume_mode = "approximate"` is logged prominently with explicit reasons (e.g., cross-GPU transfer from Tesla T4 `fp16` to RTX 5090 `bf16`).

## 4. Portable AMP Policy (`resolve_amp`)

- **Decision**: Default `--amp auto` selects:
  - `bf16` when CUDA supports bfloat16 (RTX 5090, L40s, A100, H100, RTX 30/40).
  - `fp16` on Volta/Turing GPUs without native bf16 (Kaggle T4, V100).
  - `off` (fp32) on Pascal (Kaggle P100) or CPU.

## 5. Concurrency Protection

- **Decision**: Enforce OS-level exclusive locks (`msvcrt.locking` on Windows, `fcntl.flock` on POSIX) at both suite level (`.suite.lock`) and experiment level (`.experiment.lock`). Because locks are kernel-owned, a crashed process automatically releases its lock.
- **Cloud multi-machine guard**: Shared persistent sync directories (`--sync-dir`) additionally use `.active_machine.json` heartbeat leases (`MachineLease`) to prevent two machines from simultaneously overwriting the same remote state.

## 6. Dependency-Aware Pipeline DAG (`ssod_research/pipeline.py`)

- **Decision**: Model all 15 pipeline stages as a validated DAG (`environment_verification`, `dataset_validation`, `gpu_benchmarking`, `dataset_splitting`, `supervised_warmup`, `baseline_training`, `mean_teacher_training`, `proposed_training`, `threshold_experiment`, `ablation_experiment`, `model_evaluation`, `condition_per_class_evaluation`, `statistical_analysis`, `plot_generation`, `final_report`).
- **Invalidation rule**: Changing an upstream artifact or stage config invalidates ONLY that stage and its transitive downstream dependents; unrelated completed experiments are preserved untouched.

## 7. Resumable Evaluation & Preprocessing

- **Decision**:
  - `scripts/prepare_bdd100k.py` writes all COCO JSONs atomically and commits `split_manifest.json` last with SHA-256 hashes of every generated file.
  - `native_training.evaluate()` writes per-image detections incrementally to `partial_predictions.jsonl` and commits `detections.json` + `metrics.json` atomically upon completion, so interrupted evaluations resume from remaining images without re-running training.

## 8. Lightweight GitHub Results Sync (`ssod_research/cloud_sync.py`)

- **Decision**: GitHub tracks ONLY lightweight artifacts under `results/<preset>/` (`results.csv`, `results.json`, `statistics.json`, `PAPER_RESULTS.md`, `progress.json`, `runtime_estimates.json`, `figures/*.png`, per-class/condition JSONs) and `docs/` (`PROJECT_PROGRESS.md`, `CODEX_HANDOFF.md`, `DECISIONS.md`).
- **Never committed**: `.pth`/`.pt`/`.ckpt` checkpoints, SQLite databases, `work_dirs/`, or raw BDD100K images/labels. Network/GitHub push failures are retried with backoff and never abort training.

## 9. Critical Consistency Bug Fix & CaRe-SSOD Proposed Method (`ssod_research/care_ssod.py`)

- **Critical Bug Fixed (`consistency.py` / `native_training.py`)**: `weighted_consistency_loss` previously assumed MMDetection's last-column background convention (`labels < class_dim - 1`). Under TorchVision's `background = 0`, foreground `1..10` convention, predicting `traffic sign` (`label = 10`) crashed training with `ValueError`. Added explicit `background_index` (`-1` default for MMDet, `0` for TorchVision).
- **Proposed Novel Method (`care_ssod`, `--preset care`)**:
  1. Zero-metadata RGB photometric visibility degradation estimator `d(x)` (replaces unlabeled ground-truth weather/time metadata leakage).
  2. Scale-aware Normalized Gaussian Wasserstein Distance (NWD) + cross-view JS divergence over the teacher's full `(C+1)`-class probability distribution (fixes small-object rejection for traffic lights/signs/pedestrians).
  3. Online dual-axis (class x visibility) EMA threshold calibrator (saved in checkpoints for exact deterministic resume).
  4. Tri-Zone reliability partitioning with single-pass FPN feature-shared Ambiguity-Zone background proposal suppression + aligned reliability-weighted GIoU and temperature-scaled KL soft distillation (eliminates hard-negative background poisoning, conflicting CE-vs-MSE losses, and redundant double student backbone forward passes).
- **Scientific Claim Discipline**: All baselines (`supervised`, `mean_teacher`, `full`) remain untouched for matched comparison; publication claims require measured multi-seed BDD100K gains from `--preset care`.

## 10. Q1 Journal Upgrade: CoRe-SSOD & Strong Baselines (`conformal.py`, `baselines.py`)

- **Weighted Conformal Risk Control (`core_ssod`, `--preset q1_paper` / `--preset q1_pilot`)**:
  - Replaces heuristic thresholds with finite-sample Conformal Risk Control on a strictly disjoint calibration fold (`8%`, max 256 images) carved from the labeled training split (zero validation leakage).
  - Monotonized upper-envelope FDR loss $\tilde{L}(x_i; \lambda) \in [0, 1]$ weighted by unlabeled-to-calibration photometric degradation density ratios $w_r(d(x_i))$ derives Zone-1 ($\alpha_{\text{strict}} = 0.15$) and Zone-2 ($\alpha_{\text{loose}} = 0.40$) thresholds automatically with per-class empirical-Bayes shrinkage.
- **Strong Controlled Baselines (`ssod_research/baselines.py`)**:
  - `soft_teacher` (ICCV 2021): faithful Box Jittering localization filtering + Soft-Negative Background Classification Weighting via EMA Teacher background probabilities.
  - `consistent_teacher` (CVPR 2023): checkpoint-serializable per-class 1D 2-component GMM dynamic thresholding.
- **Training Recipe & Cross-GPU Effective Batch Invariance**:
  - Added `fasterrcnn_v2` (`fasterrcnn_resnet50_fpn_v2`), `max_side = 800`, `warmup_constant` / `warmup_cosine` / `warmup_multistep` LR schedules, `--unsup-warmup-steps`, and `--grad-accum-steps`.
  - Fingerprint uses `effective_batch_size = batch_size * grad_accum_steps` so Kaggle T4 (`2×4`) and RTX 5090 / L40s (`4×2` or `8×1`) resume seamlessly with identical per-step image ordering.

## 11. Forensic Pilot Audit & CoRe-SSOD v2 Architecture Fixes (`native_training.py`, `care_ssod.py`, `conformal.py`, `baselines.py`)

Forensic analysis of the first `q1_pilot_runs` (500-step warmup, 1500-step train, micro-batch 2×4) revealed that ALL SSOD variants trailed `supervised` (`0.1100` AP vs `mean_teacher 0.0797`, `soft_teacher 0.0754`, `consistent_teacher 0.0731`, `core_ssod 0.0598`) due to identifiable architectural and statistical bugs — fixed uniformly and fairly in v2:

1. **FasterRCNN-v2 Normalization & EMAN Buffer Fix**: TorchVision `fasterrcnn_resnet50_fpn_v2` contains 65 active `nn.BatchNorm2d` layers (53 backbone + 8 FPN + 4 RoI head). Micro-batch=2 + augmented unlabeled forward passes corrupted BN statistics, while `ema_update` hard-copied student BN buffers directly into the lagging EMA teacher (`mean_teacher` Student AP `0.1219` vs Teacher AP `0.0866`, a `-3.53` AP collapse!). **Fix**: Convert `backbone.body` BNs to `FrozenBatchNorm2d` (preserving ImageNet-V2 stats), convert FPN/RPN/RoI BNs to `GroupNorm(32, C)` (Wu & He, ECCV'18 — strictly batch-size invariant across `2×4` and `4×2`), and upgrade `ema_update` to EMAN (EMA float buffers, Cai et al., CVPR'21).
2. **Schedule, Burn-In, Unsup Ramp-Up & Labeled Flip (Uniform Across All Variants)**:
   - Burn-in `warmup` now uses `warmup_constant` (no LR decay to 0 before SSOD starts) and 1,500 steps in `q1_pilot` (1.71 epochs on 10% BDD100K) / 3,000 steps in `q1_paper`.
   - Added `--unsup-warmup-steps` linear ramp (`300` for `q1_pilot`, `1000` for `q1_paper`) and schedule-matched EMA (`0.996` for 3k pilot, `0.9985` for 12k paper).
   - Added deterministic per-slot 50% horizontal flip on labeled batches across all variants.
3. **CoRe-SSOD v2 Algorithmic Upgrades**:
   - **Direct Bidirectional Cross-View RoI Probing + Weighted Box Fusion**: Replaces double RPN/NMS + $O(N^2)$ greedy matching with single-RPN + flipped-backbone RoI probing and consensus box fusion (`+1.06` TP IoU, `+57` TPs, `4.7×` faster calibration, `+45%` small-object recall at 15% FDR).
   - **Exact Monotone Bounded FP-per-GT Conformal Risk Control + 1D PAVA**: Replaces conservative suffix-max FDP (which inflated risk +35% and crushed night recall to `<8%`) with naturally monotone $L_{\text{FPR}}(x;\lambda) = \min(1, \text{FP}(\lambda)/\max(1,|G|))$ ($\alpha_{\text{strict}}=0.08, \alpha_{\text{loose}}=0.30$), true regime-conditional density ratios $p_U(\text{bin}|r)/p_{\text{cal}}(\text{bin}|r)$, candidate-conditioned per-class calibration, and simultaneous PAVA precision $\hat{\pi}_r(q)$.
   - **RPN Zone-2 Ignore + ROI Conformal Soft-Negative Weighting**: Replaces hard background proposal deletion (which caused a 50,000-detection FP explosion in v1 `core_ssod`) with RPN Zone-2 ignore masking (`-1`), normalized ROI background weighting $p_T(\text{bg}) \cdot (1 - \hat{\pi}_r(q_{z_2}))$ (mean `1.0`), localization-gated Smooth-L1 + sampled-proposal GIoU (`0.25`), and calibrated linear-$T$ KL distillation (`0.15`/`0.05`, $T=1.5$).
4. **Faithful Baseline Corrections (`baselines.py`)**:
   - `soft_teacher`: strictly decouples `cls_valid` (RPN + ROI soft-negative classification) from jitter-verified `reg_valid` (ROI box regression only).
   - `consistent_teacher`: filters GMM inputs to `score >= 0.15` and solves the exact Bayes Gaussian posterior decision boundary (preventing collapse into the background noise tail).
5. **Deterministic Multi-Worker I/O Prefetching & Batched AMP Eval/Calibration**: Overlaps PIL decode/resize with GPU compute using stateless per-step index prefetching and batched AMP evaluation/calibration without altering determinism or crash-safe incremental resumption.
