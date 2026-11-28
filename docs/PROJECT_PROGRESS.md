# Project Progress

_Last updated (UTC): `2026-11-24T14:02:23+00:00`_

## Overview

| Metric | Value |
|---|---|
| Active Preset | `q1_pilot` |
| Total Stages / Experiments | **18** |
| Completed | **18** |
| Running | **0** |
| Interrupted (Resumable) | **0** |
| Failed (Retryable) | **0** |
| Pending | **0** |
| Completed Training Steps | **16500 / 16500** (100.00%) |
| Accumulated Runtime | `9h 46m 22s` (35182.2s) |
| Measured Step Speed | `2.1314 s/step` (observed_training_steps) |
| Estimated Remaining Runtime | `0.0s` |
| Current / Next Stage | All stages complete |
| Last Successful Checkpoint | `10pct_s42/core_ssod` step 3000 (`D:\project\Abrishami\Semi-Supervised-Object-Detection\work_dirs\q1_pilot_v2_runs\runs\10pct_s42\core_ssod\checkpoints\ckpt_000003000.pth`, sha256=`7e86e6319c36`) |

## Resume Command

```powershell
python scripts/run_suite.py --preset q1_pilot --resume --output work_dirs\q1_pilot_v2_runs --device cuda
```

## Stage Status

| # | Stage | Type | Status | Step | Runtime | Resume Mode | Retries | Notes |
|---:|---|---|---|---:|---:|---|---:|---|
| 1 | `environment_verification` | `environment_verification` | **COMPLETED** | - | 2.4s | `fresh` | 0 |  |
| 2 | `dataset_validation` | `dataset_validation` | **COMPLETED** | - | 0.1s | `fresh` | 0 |  |
| 3 | `10pct_s42/prepare` | `dataset_splitting` | **COMPLETED** | - | 10.8s | `fresh` | 1 |  |
| 4 | `10pct_s42/warmup` | `supervised_warmup` | **COMPLETED** | 1500/1500 | 18m 55s | `fresh` | 0 |  |
| 5 | `10pct_s42/warmup/eval` | `model_evaluation` | **COMPLETED** | - | 0.0s | `fresh` | 0 |  |
| 6 | `10pct_s42/supervised` | `baseline_training` | **COMPLETED** | 3000/3000 | 37m 34s | `fresh` | 0 |  |
| 7 | `10pct_s42/supervised/eval` | `model_evaluation` | **COMPLETED** | - | 0.0s | `fresh` | 0 |  |
| 8 | `10pct_s42/mean_teacher` | `mean_teacher_training` | **COMPLETED** | 3000/3000 | 2h 5m 54s | `fresh` | 0 |  |
| 9 | `10pct_s42/mean_teacher/eval` | `model_evaluation` | **COMPLETED** | - | 0.0s | `fresh` | 0 |  |
| 10 | `10pct_s42/soft_teacher` | `proposed_training` | **COMPLETED** | 3000/3000 | 2h 31m 42s | `fresh` | 0 |  |
| 11 | `10pct_s42/soft_teacher/eval` | `model_evaluation` | **COMPLETED** | - | 0.0s | `fresh` | 0 |  |
| 12 | `10pct_s42/consistent_teacher` | `proposed_training` | **COMPLETED** | 3000/3000 | 2h 6m 48s | `fresh` | 0 |  |
| 13 | `10pct_s42/consistent_teacher/eval` | `model_evaluation` | **COMPLETED** | - | 0.0s | `fresh` | 0 |  |
| 14 | `10pct_s42/core_ssod` | `proposed_training` | **COMPLETED** | 3000/3000 | 2h 5m 14s | `fresh` | 0 |  |
| 15 | `10pct_s42/core_ssod/eval` | `model_evaluation` | **COMPLETED** | - | 0.0s | `fresh` | 0 |  |
| 16 | `statistical_analysis` | `statistical_analysis` | **COMPLETED** | - | 0.2s | `fresh` | 0 |  |
| 17 | `plot_generation` | `plot_generation` | **COMPLETED** | - | 1.2s | `fresh` | 0 |  |
| 18 | `final_report` | `final_report` | **COMPLETED** | - | 0.0s | `fresh` | 0 |  |
