# Reliability-Aware Semi-Supervised Road Object Detection

Research scaffold for **BDD100K RGB 2D object detection** using OpenMMLab MMDetection's **SoftTeacher**. This project is a research extension, not a claimed reproduction of any paper.

## Provenance and baseline

- Framework: [MMDetection](https://github.com/open-mmlab/mmdetection), Apache-2.0. We **do not vendor/copy the full framework**; clone and pin it separately.
- Baseline: [Soft Teacher](https://arxiv.org/abs/2106.09018), as implemented in MMDetection 3.x.
- Additional prior art: [Consistent-Teacher](https://github.com/Adamdad/ConsistentTeacher), [Sparse Semi-DETR](https://arxiv.org/abs/2404.01819), [STEP-DETR](https://openaccess.thecvf.com/content/ICCV2025/html/Shehzadi_STEP-DETR_Advancing_DETR-based_Semi-Supervised_Object_Detection_with_Super_Teacher_and_ICCV_2025_paper.html), and SSRT-DETR (Sensors, 2026).
- The optional `ReliabilitySoftTeacher` extends existing pseudo-label filtering with a *prototype* confidence/uncertainty gate. **It is not established novelty, nor a finished three-term consistency loss.** Compare it against the original SoftTeacher with identical settings.

## Repository contents

- `ssod_research/reliability.py`: dependency-light selection, weights and matching logic.
- `ssod_research/mmdet_extension.py`: MMDetection-registered subclass that adjusts pseudo scores using localization uncertainty. Optional MMDetection dependency.
- `ssod_research/geometry.py`: geometric projection and IoU matching utilities.
- `scripts/prepare_bdd100k.py`: convert BDD100K detection JSON to COCO; create deterministic disjoint labeled/unlabeled splits.
- `configs/bdd_softteacher.py`: starter training config referencing externally cloned MMDetection.
- `configs/bdd_reliability.py`: identical config with the experimental selector.
- `tests/`: CPU-only unit tests; no training claims.

## Quick start

1. Install Python 3.11 and a Blackwell-compatible **official PyTorch/CUDA build** for your RTX 5090. Verify `torch.cuda.is_available()` and `torch.cuda.get_device_capability()` before installing MMCV. Do not reuse old pre-Blackwell wheels blindly.
2. `git clone https://github.com/open-mmlab/mmdetection.git third_party/mmdetection`; checkout pinned commit `cfd5d3a985b0249de009b67d04f37263e11cdf3d` and record any subsequent upgrade. Follow that version's installation instructions for `mmengine`, `mmcv` and `mmdet`. Run `git -C third_party/mmdetection checkout cfd5d3a985b0249de009b67d04f37263e11cdf3d`. The `third_party/` directory is gitignored.
3. `python -m pip install -e ".[dev]"`; run `python -m pytest -q`.
4. Download BDD100K detection images and labels yourself and review dataset license.
5. `python scripts/prepare_bdd100k.py --train-labels DATA/bdd100k_labels_images_train.json --val-labels DATA/bdd100k_labels_images_val.json --output data/bdd100k --fraction 0.10 --seed 42`
6. Arrange image directories as `data/bdd100k/images/train/` and `data/bdd100k/images/val/`.
7. Run a 2-iteration training smoke test **only after** the MMDetection + MMCV installation and dataset config are validated:
   `python third_party/mmdetection/tools/train.py configs/bdd_softteacher.py --cfg-options train_cfg.max_iters=2 train_cfg.val_interval=1000`
8. Then run full baselines with explicit iteration budgets and seeds. Never use the validation split for fitting the method and never evaluate on a hidden test split without the authorized evaluator.

**Important:** No GPU training, performance benchmark, third-party build or real BDD100K data validation has been executed in this repository yet. The `configs/` files are candidate MMDetection integrations until verified on the target machine.

See [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md), [docs/LITERATURE.md](docs/LITERATURE.md), and [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md).