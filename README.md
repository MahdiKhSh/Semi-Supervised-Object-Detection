# Road-SSOD: Reliability-Aware Semi-Supervised Road Object Detection

Research code for **BDD100K RGB 2D detection**. Target: **native Windows +
NVIDIA RTX 5090 (32GB)**, Python 3.11 / official CUDA-enabled PyTorch.
Automatic runs produce real measurements; no performance is claimed before
training on BDD100K. Scientific novelty is not yet established.

## One-click Windows entry points

1. Clone or download the **feat/research-baseline** branch of this repository.
2. Double-click **RUN_SMOKE.cmd** to verify the native training pipeline
   on synthetic data. NO synthetic AP is reportable as science.
3. Place official BDD100K images and labels into the directories specified
   in [Windows guide](docs/WINDOWS_ONE_CLICK.md).
4. Double-click **RUN_PILOT.cmd** for 10% labeled pilot and exploratory ablations.
5. Optionally run **RUN_CORE.cmd** for the compute-aware scientific core.
6. Double-click **RUN_ALL.cmd** for the complete **5%/10% × 3-seed**
   training matrix: 102 sequential stages including supervised warmups,
   teacher/student, reliability, condition/class adaptation, three-level
   consistency, threshold sweeps and on/off ablations.

The paper preset can be extremely time-consuming on one GPU. Runs save
checkpoints, logs, class/condition AP diagnostics and a cumulative results
CSV/JSON report. Re-running the same launcher resumes. Nothing changes RDP,
firewall, VPN or system GPU driver.

## Key modules

| Component | File |
|---|---|
| Primary native PyTorch/TorchVision trainer | ssod_research/native_training.py |
| Resize, image/box transform, COCO data loading | ssod_research/native_data.py |
| Geometric mappings and class-matched IoU | ssod_research/geometry.py |
| Differentiable box/class/confidence stability loss | ssod_research/consistency.py |
| BDD100K labeled/unlabeled train split | scripts/prepare_bdd100k.py |
| End-to-end resumable experiment orchestrator | scripts/run_suite.py |
| Synthetic smoke data fixture | scripts/make_synthetic_bdd.py |
| COCO scene/time/weather AP | scripts/eval_conditions.py |
| Native Windows setup and launchers | scripts/setup_windows.ps1, RUN_*.cmd |
| Paper protocol (zero to publication) | docs/PAPER_ROADMAP_0_TO_100.md |
| Prior-art matrix | docs/LITERATURE.md |
| Optional original MMDetection SoftTeacher extension | configs/, ssod_research/mmdet_extension.py |

## Scientific design and attribution

The **primary Windows algorithm** is a native TorchVision Faster R-CNN
supervised baseline, followed by a **Mean Teacher EMA** detector baseline
and ablations. It is **not an exact reproduction of the Soft Teacher paper**.
Our candidate method combines box localization agreement, confidence-gated
pseudo labels, class-frequency-dependent thresholds, scene-condition
thresholds, and differentiable auxiliary student ROI consistency.

The **optional MMDetection 3.x integration** uses its original SoftTeacher
implementation (Apache-2.0) at pinned upstream commit
cfd5d3a985b0249de009b67d04f37263e11cdf3d.
We do not duplicate its complete framework; it remains a separate research
comparison and has not been validated for native CUDA 5090 on this machine.

Closest papers: Soft Teacher (ICCV 2021), Consistent-Teacher (CVPR 2023),
Sparse Semi-DETR (CVPR 2024), Building Blocks (TMLR 2025),
STEP-DETR (ICCV 2025), SSRT-DETR (Sensors 2026).
These contain substantial prior art. The code is a **testable hypothesis**,
not yet proof of methodological novelty.

## What has and has not been verified

Automated GitHub Actions cover CPU utility tests, MMEngine configuration
parsing, PyTorch gradient checks and a synthetic native end-to-end smoke
attempt. Read [Actions](../../actions) for live status.
RTX 5090 training, real BDD100K AP, runtime reliability and final novelty
comparisons require running the Windows launchers on the user's machine.

Read [Windows one-click guide](docs/WINDOWS_ONE_CLICK.md),
[full experiment roadmap](docs/PAPER_ROADMAP_0_TO_100.md),
[Persian execution roadmap](docs/ROADMAP_FA.md), and [experiment protocol](docs/RESEARCH_PLAN.md).
