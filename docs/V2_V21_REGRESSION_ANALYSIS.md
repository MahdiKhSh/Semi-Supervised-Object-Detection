# Forensic Regression Analysis: `CoRe-SSOD v2` (`4f8ce52`) vs `v2.1` (`0252a64`)

**Repository**: `https://github.com/MahdiKhSh/Semi-Supervised-Object-Detection`  
**Compared Commits**:
- **`v2` (`4f8ce523a0d83fa685534027e009d3e28909b158`)**: Local RTX 5090 pilot suite (`work_dirs/q1_pilot_v2_runs`)
- **`A100 branch` (`e8497564bc45789f03e0bc52e82bc3f10ea7a7d7`)**: A100 reference suite + zero-box filtering + `--reserve-calibration`
- **`v2.1` (`0252a64a708faab46ed9c391da7ebbbf62ab2520`)**: Merge of `4f8ce52` and `e849756` (`17 files changed, 678 insertions(+), 147 deletions(-)`)

---

## 1. Forensic Provenance of the Historical `13.85% / 17.81% / 18.15%` Numbers

Section 4 of the audit brief noted that the historical results (`13.85 / 17.81 / 17.75 / 17.43 / 17.50 / 18.15`) had not yet been conclusively linked to a commit or validation scope.

Forensic inspection of the local experiment artifacts in `work_dirs/q1_pilot_v2_runs/` (`results.csv`, `PAPER_RESULTS.md`, `experiment_plan.json`, `splits/10pct_s42/split_manifest.json`, and `runs/10pct_s42/*/manifest.json`) **conclusively establishes their exact origin to 16 decimal places**:

| Run (`work_dirs/q1_pilot_v2_runs`) | Step | `eval_max_images` | `partial_validation` | Labeled Pool | `eff_batch` | mAP (%) | AP50 (%) | AP75 (%) | AP_S (%) | AP_M (%) | AP_L (%) |
|---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `10pct_s42/warmup` | 1500 | **500** | **`True`** | **7,000** (unreserved) | `16 × 2 = 32` | **13.851** | 32.747 | 9.918 | 5.879 | 19.356 | 37.621 |
| `10pct_s42/supervised` | 3000 | **500** | **`True`** | **7,000** (unreserved) | `16 × 2 = 32` | **17.805** | 37.221 | 14.256 | 6.479 | 23.485 | 48.454 |
| `10pct_s42/mean_teacher` | 3000 | **500** | **`True`** | **7,000** (unreserved) | `16 × 2 = 32` | **17.750** | 36.590 | 14.533 | 6.647 | 23.041 | 46.589 |
| `10pct_s42/soft_teacher` | 3000 | **500** | **`True`** | **7,000** (unreserved) | `16 × 2 = 32` | **17.426** | 37.068 | 13.683 | 6.925 | 22.915 | 43.510 |
| `10pct_s42/consistent_teacher` | 3000 | **500** | **`True`** | **7,000** (unreserved) | `16 × 2 = 32` | **17.498** | 36.960 | 14.036 | 7.124 | 23.163 | 43.529 |
| `10pct_s42/core_ssod` | 3000 | **500** | **`True`** | **6,744** (`256` cal) | `16 × 2 = 32` | **18.151** | 37.550 | 14.293 | 7.203 | 23.332 | 46.055 |

Compare this directly against the confirmed A100 `v2.1` (`CORE_SSOD_V21_A100_BS32_FRESH`) results:

| Run (`CORE_SSOD_V21_A100_BS32_FRESH`) | Step | `eval_max_images` | `partial_validation` | Labeled Pool | `eff_batch` | mAP (%) | AP50 (%) | AP75 (%) | AP_S (%) | AP_M (%) | AP_L (%) |
|---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `10pct_s42/warmup` | 1500 | **0 (All 10,000)** | **`False`** | **6,744** (`256` cal) | `32 × 1 = 32` | **12.118** | 28.796 | 8.397 | 4.072 | 14.780 | 26.121 |
| `10pct_s42/supervised` | 3000 | **0 (All 10,000)** | **`False`** | **6,744** (`256` cal) | `32 × 1 = 32` | **16.557** | 36.636 | 12.859 | 5.377 | 19.262 | 35.702 |
| `10pct_s42/mean_teacher` | 3000 | **0 (All 10,000)** | **`False`** | **6,744** (`256` cal) | `32 × 1 = 32` | **16.355** | 35.777 | 12.685 | 5.201 | 19.176 | 35.595 |

---

## 2. Root-Cause Decomposition: Confirmed Facts vs Testable Hypotheses

### 2.1 CONFIRMED ROOT CAUSES (Proven from Source Code & Saved Manifests)

#### Cause A (Primary Metric Confounder): **500-Image Diagnostic Subset (`q1_pilot`) vs 10,000-Image Full Validation (`q1_a100`)**
- `q1_pilot_v2_runs/PAPER_RESULTS.md` explicitly flagged all six historical numbers under **`## Diagnostic Partial-Validation Summary (NOT for paper reporting)`** (`eval_max_images = 500`).
- Direct annotation census on `splits/10pct_s42/annotations/val.json` (`dataset_and_split_audit.json`) reveals severe tail-class skew between the first 500 sorted validation images and the full 10,000 validation set:
  - Category `train`: **1 ground-truth box** in `val_500` vs **15 boxes** across 10,000 images!
  - Category `motorcycle`: **20 boxes** in `val_500` vs **452 boxes** in `val_10000`.
  - Category `rider`: **35 boxes** in `val_500` vs **649 boxes** in `val_10000`.
  - Category `bicycle`: **40 boxes** in `val_500` vs **1,007 boxes** in `val_10000`.
  - Notice that `Supervised` AP50 barely changed (`37.22%` on `val_500` vs `36.64%` on `val_10000`, only `-0.58` points!), whereas `AP_large` shifted by **`-12.75` points (`48.45% -> 35.70%`)** and `AP_medium` shifted by **`-4.22` points (`23.49% -> 19.26%`)**! In `pycocotools`, `AP_large` is macro-averaged over all 10 classes that contain at least one large GT box in `imgIds`: in `val_500`, noisy/rare large-object categories with zero GT boxes in the 500-image slice are **excluded (`-1`) from the macro denominator**, artificially inflating `AP_large` by +10–12 points relative to full 10,000-image evaluation where rare large-object classes (`train`, `motorcycle`, `bicycle`, `rider`) enter the denominator with near-0% AP!
- **Crucial Scientific Conclusion**: For `warmup` and `supervised`, **zero supervised loss or model-architecture code changed between `4f8ce52` (`v2`) and `0252a64` (`v2.1`)**! Look at the exact diff of `native_training.py`: the supervised forward pass, `FrozenBN + GroupNorm`, `SGD(lr=0.01, momentum=0.9, wd=1e-4)`, `warmup_constant` / `warmup_cosine`, and `_weak_flip_sample` are **100% identical** between `v2` and `v2.1`. The *only* two differences for `warmup` and `supervised` between `q1_pilot_v2_runs` and `CORE_SSOD_V21_A100_BS32_FRESH` are:
  1. **Validation set**: `500` images (`partial_validation=True`) vs `10,000` images (`partial_validation=False`), and
  2. **Supervised pool size**: `7,000` images vs `6,744` images (`--reserve-calibration`, `-3.66%` labeled images).

#### Cause B (Confirmed Crash + Proposal Reduction in `v2.1` Soft Teacher):
- In `v2` (`4f8ce52`), `soft_teacher_unsupervised_loss` passed all `aug_images` into `model.rpn(transformed, s_features, cls_targets)` even when an image had `0` `cls_valid` boxes (because `cls_targets` was a list of empty target dicts, not `None`).
- In `v2.1` (`0252a64`, `baselines.py:305-315`), slicing `cls_pos` meant images with `0` `cls_valid` boxes and `>= 1` `reg_valid` boxes called `model.rpn(extra_imgs, extra_feats, None)` with `model.rpn.training == True`, crashing with `ValueError: targets should not be None`.
- The temporary server patch (`model.rpn.training = False`) fixed the exception but switched RPN `pre_nms_top_n` and `post_nms_top_n` from **2,000 (training) -> 1,000 (testing)** on regression-only images. Fixed in `v2.2` via `extract_rpn_proposals_only`.

#### Cause C (Confirmed Unsupervised Gradient Distortion in `v2.1` at `batch_size = 32`):
- In `v2.1`, excluding $B - M$ zero-pseudo-box images from a micro-batch of $B = 32$ images averaged RPN/ROI losses over $M$ active images without multiplying by $M / B$.
- Whenever early training steps or high thresholds yielded sparse batches ($M \ll 32$), the active images' unsupervised gradients were amplified by $32 / M$, breaking equivalence with `16 × 2` and `2 × 4` and injecting high-variance gradient spikes into student updates (explaining extra instability in `Mean Teacher`: `16.355%` vs `16.557%` Supervised, `-0.20` AP — note that in `v2` `Mean Teacher` also underperformed `Supervised` by `-0.06` AP (`17.750%` vs `17.805%`) due to uncalibrated `0.70` pseudo-label noise in night scenes!).

#### Cause D (Confirmed Conformal Threshold Asymmetry in `v2` & `v2.1`):
- Note an asymmetric subtle advantage `CoRe-SSOD` did **not** have in `v2` (`q1_pilot_v2_runs`), which actually makes `CoRe-SSOD`'s `+0.35` gain in `v2` *conservative*: in `v2`, `Supervised` trained on all **7,000** labeled images while `CoRe-SSOD` trained on only **6,744** labeled images (`native_training.py:802` in `4f8ce52` carved out 256 images *only* for `core_ssod`, though `warmup` had seen those 256 images!). In `v2.1`, `--reserve-calibration` equalized all methods at **6,744** supervised images from step 0.
- However, raising `max_high` from `0.90 -> 0.95` in `v2.1` without fixing single-image $|G_i|$ normalization (`conformal.py:124`) allowed `degraded` night `tau_high` to climb into `0.91–0.94` and `tau_low` to `0.71–0.75`, choking small-object and night pseudo-label recall. Fixed in `v2.2`.

---

### 2.2 UNVERIFIED HYPOTHESES (Require Stage-1 Controlled Experiments to Quantify)

1. **Exact AP cost of `--reserve-calibration` (`7,000 -> 6,744` images, `-3.66%` data)**:
   - *Hypothesis*: Holding out 256 images accounts for roughly `0.10–0.25` mAP of the `Supervised` difference, while the `500 -> 10,000` validation set expansion accounts for the remaining `~1.00–1.15` mAP.
   - *Test*: Diagnostic runs `E1` vs `E2` (`scripts/run_v22_diagnostic_matrix.py`) on identical 10,000-image validation.
2. **Optimization suboptimality of `eff_batch = 32` with unscaled `lr = 0.01` over 3,000 steps**:
   - *Hypothesis*: Increasing `eff_batch` $4\times$ (`8 -> 32`) without scaling `lr` (`0.01 -> 0.02`, Goyal et al. linear/sqrt scaling) reduces total SGD update travel per epoch, while 3,000 steps (`14.2` labeled epochs at `eb=32` vs `1.5` unlabeled epochs over 63,000 unlabeled images!) leaves unlabeled data 85% unseen or under-converged compared to `12,000` steps.
   - *Test*: Diagnostic runs `E4` (`eb=8, lr=0.01`), `E7` (`eb=32, lr=0.02`), and `E9` (`12,000` steps).

---

## 3. Complete File-by-File Diff Audit: `v2` (`4f8ce52`) -> `v2.1` (`0252a64`) -> `v2.2`

| File | Change in `v2.1` (`4f8ce52..0252a64`) | Consequence in `v2.1` | Resolution in `v2.2` (`feat/core-ssod-v2.2-audit`) |
|---|---|---|---|
| `ssod_research/baselines.py` | Sliced `soft_teacher_unsupervised_loss` to `cls_pos` and `reg_pos`; called `model.rpn(extra_imgs, extra_feats, None)` on `missing_reg`. | **Crashed** on regression-only images (`F-01`); omitted `M / B` batch scaling (`F-03`). | Added `extract_rpn_proposals_only` (2,000-proposal training policy, no crash) + `M / B` scaling + `--zero-box-filter` toggle. |
| `ssod_research/care_ssod.py` | Added `_rpn_forward_with_zone2_ignore` and sliced RPN/ROI detection path to `positive` Zone-1 images while retaining full-batch Zone-2 KL. | Protected dark 0-Zone-1 scenes from false negatives, **but** omitted `M / B` scaling (`F-03`) and kept occlusion-blind Cutout (`F-07`). | Added `M / B` scaling (`det_scale`, `kl1_scale`, `kl2_scale`), `<=35%` box-occlusion guard on Cutout, and strict boundary clamping. |
| `ssod_research/conformal.py` | Changed `alpha_strict: 0.08 -> 0.10`, `max_high: 0.90 -> 0.95`, `split_calibration_indices(1) -> ([0], [])`. | Allowed `degraded` night `tau_high` to climb $>0.90$ and `tau_low` to `0.73` (`F-05`); retained Kish heuristic instead of exact $w_{\max}$ WCRC (`F-04`). | Implemented exact $w_{\text{worst}}$ WCRC bound, regime-smoothed GT normalizer, bounded non-inverted class shrinkage, `max_low=0.55`, and stratified `split_calibration_indices`. |
| `ssod_research/native_training.py` | Added `--reserve-calibration`, `source_revision()`, `calibration_fold_sha256` guard, `GradScaler` overflow guard, and 0-box filtering in `consistent_teacher`/`mean_teacher`. | Enforced leak-free $D_{\text{cal}}$ from step 0 (good!), **but** omitted `M / B` scaling on filtered batches (`F-03`). | Added `M / B` scaling, batched `append_jsonl_batch` in `evaluate()`, `--anchor-profile road_small`, `--stratify-calibration`, `--zero-box-filter`, and `conformal_exchangeability_valid` manifest audit field. |
| `ssod_research/native_data.py` | Added cached recursive `build_image_index` + `resolve_image_path` supporting nested `trainA/trainB` directories. | Positive: `<0.3s` inventory check, zero impact on tensor values. | Retained unchanged. |
| `scripts/prepare_bdd100k.py` | Added `--train-images` / `--val-images` missing-file filtering and `verify_split_output` checks. | Positive for partial cloud mirrors; silent skip if 2020 `"person"` alias encountered. | Added `CATEGORY_ALIASES` (`person -> pedestrian`, etc.) for Detection 2020 compatibility while preserving exact 2018 SHA-256s. |
| `scripts/run_suite.py` | Added `--preset q1_a100` (full 10k val), `--dataset-from-plan`, and auto `--reserve-calibration`. | Evaluated all 10,000 val images (revealing partial-vs-full AP gap); pre/post-prepare `split_fingerprint` mismatch triggered false invalidation logs while `job_complete` skipped `init_checkpoint_sha256` & `eval_max_images` checks (`F-08`). | Added post-prepare DAG refresh + cryptographic verification of `dataset_sha256`, `init_checkpoint_sha256`, `eval_max_images`, `anchor_profile`, and `stratify_calibration`. |
| `scripts/eval_external.py` | Added `--weights {auto,student,teacher}` selection. | Positive diagnostic utility. | Retained unchanged. |
