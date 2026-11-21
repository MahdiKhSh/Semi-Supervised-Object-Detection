# Q1 Journal Paper Protocol: CoRe-SSOD on BDD100K

Target journals: **IEEE Transactions on Intelligent Transportation Systems (T-ITS)**, **IEEE Transactions on Intelligent Vehicles (T-IV)**, **Pattern Recognition**, or **Neural Networks**.

---

## 1. Core Contribution: `CoRe-SSOD`

**Full Name**: *Weighted Conformal Risk-Controlled Semi-Supervised Road Object Detection under Photometric Visibility Shift (`CoRe-SSOD`)*

### Why existing SSOD methods fail on adverse-weather road scenes
1. **Uncalibrated heuristic thresholds** (Soft Teacher ICCV'21, Consistent-Teacher CVPR'23, ARSL CVPR'23, SSRT-DETR 2026) offer **zero finite-sample guarantee** on pseudo-label False Discovery Rate (FDR), especially under covariate shift from daytime clear scenes to night, rain, and fog.
2. **Metadata leakage**: Condition-aware methods that consume ground-truth weather/time-of-day labels on unlabeled images violate true semi-supervised assumptions.
3. **Small-object IoU collapse**: Fixed IoU localization gating rejects tiny BDD100K `traffic light`, `traffic sign`, and `pedestrian` instances under 1–2 px localization jitter.
4. **False-negative background poisoning**: Hard binary filtering treats rejected uncertain objects as confident background, suppressing recall in night/rain scenes and rare classes.

### Mathematical Formulation (Section 3 Ready)

1. **Zero-Metadata Photometric Degradation Risk $d(x) \in [0, 1]$** (`ssod_research/care_ssod.py`):
   Computed directly from unlabeled RGB pixels via luminance darkness, contrast deficit, and dark-background specular glare, partitioning scenes into `clear` ($d < 0.35$) and `degraded` ($d \ge 0.35$) without external weather labels.
2. **Scale-Adaptive NWD + Cross-View JS Reliability** (`ssod_research/care_ssod.py`):
   $$q_i = s_i \cdot \exp\big(-\alpha_{\text{loc}}\,u_{\text{nwd}}(b_i, \tilde{b}_i) - \alpha_{\text{cls}}\,\text{JS}(p_T^{(1)} \| p_T^{(2)})\big)$$
3. **Weighted Conformal Risk Control under Photometric Shift** (`ssod_research/conformal.py`):
   Using a strictly disjoint calibration fold $\mathcal{D}_{\text{cal}} \subset \mathcal{D}_L$ (8% of labeled training images, capped at 256, zero validation leakage), define the bounded, monotonically non-increasing upper-envelope FDR loss:
   $$\tilde{L}(x_i; \lambda) = \sup_{\lambda' \ge \lambda} \frac{\sum_{j: q_j \ge \lambda'} \mathbb{1}[y_j = 0]}{\max(1, |\{j: q_j \ge \lambda'\}|)} \in [0, 1]$$
   Weighted by the unlabeled-to-calibration photometric density ratio $w_r(d(x_i))$ with Kish effective sample size $n_{\text{eff}}$:
   $$\hat{\lambda}_r(\alpha) = \inf \left\{ \lambda \in \Lambda : \frac{n_{\text{eff}}}{n_{\text{eff}} + 1}\hat{R}_r(\lambda) + \frac{1}{n_{\text{eff}} + 1} \le \alpha \right\}$$
   Per-class thresholds apply empirical-Bayes shrinkage toward $\hat{\lambda}_r(\alpha)$ for rare classes.
4. **Conformal Tri-Zone Partitioning + Single-Pass Ambiguity Suppression**:
   - **Zone 1** ($q_i \ge \hat{\lambda}(\alpha_{\text{strict}})$): Positive RPN/ROI targets + reliability-weighted GIoU + temperature-scaled KL distillation.
   - **Zone 2** ($\hat{\lambda}(\alpha_{\text{loose}}) \le q_i < \hat{\lambda}(\alpha_{\text{strict}})$): Excluded from background negative ROI sampling + soft KL distillation only (even when an image has zero Zone-1 boxes).
   - **Zone 3** ($q_i < \hat{\lambda}(\alpha_{\text{loose}})$): Background.
   Single student FPN backbone forward pass per unlabeled batch; zero extra inference cost at test time.

---

## 2. Controlled Experimental Matrix (`--preset q1_paper`)

- **Detector**: `Faster R-CNN ResNet50-FPN-v2` (`fasterrcnn_v2`), `max_side = 800` (`800×450`)
- **Optimization**: SGD (`lr = 0.01`, momentum 0.9, wd `1e-4`), **Linear Warmup (500 steps) + Cosine Decay**, **Effective Batch Size = 8**
- **Budgets**: Warmup `3,000` steps (24,000 views) + SSOD `12,000` steps (96,000 labeled + 96,000 unlabeled views)
- **Matched Main Baselines** (5% & 10% × 3 seeds: 42, 123, 2026):
  1. `supervised` (Supervised-only continuation)
  2. `mean_teacher` (Fixed-threshold EMA Mean Teacher)
  3. `soft_teacher` (Faithful Soft Teacher, ICCV'21: Soft-Negative BG Weighting + Box Jitter)
  4. `consistent_teacher` (Consistent-Teacher-style Dynamic Per-Class 1D GMM, CVPR'23)
  5. `core_ssod` (**Proposed**: Weighted Conformal Risk-Controlled Tri-Zone NWD/JS)
- **Targeted Ablations** (seed 42, 5% & 10%):
  - `care_ssod` (Heuristic EMA threshold vs Conformal Risk Control)
  - `core_no_vis_weight` (Unweighted conformal vs photometric-shift weighted conformal)
  - `core_no_trizone` (2-zone hard filtering vs Tri-Zone ambiguity suppression + soft KL)
  - `core_no_nwd` (IoU-only vs scale-aware NWD + JS)
  - `core_alpha005` ($\alpha_{\text{strict}} = 0.05$) & `core_alpha025` ($\alpha_{\text{strict}} = 0.25$)

---

## 3. Obtaining BDD100K

- **Official source**: https://bdd-data.berkeley.edu/ (or https://dl.cv.ethz.ch/bdd100k/data/) — requires free academic registration. Download `100K Images` (`train` + `val`) and `Detection 2020 Labels` (`bdd100k_labels_images_train.json`, `bdd100k_labels_images_val.json`).
- **On Kaggle**: Attach any verified public BDD100K dataset (e.g., search `bdd100k` in Kaggle Datasets) and point `TRAIN_LABELS`, `VAL_LABELS`, `TRAIN_IMAGES`, `VAL_IMAGES` to `/kaggle/input/<dataset>/...`.

---

## 4. Seamless Kaggle (Tonight) -> RTX 5090 / L40s (Tomorrow) Execution

### Step 1: Fast Validation Pilot First (~15 min on 5090 / ~45 min on Kaggle T4)
```bash
# Kaggle T4 (16 GB): micro-batch 2 x grad-accum 4 = effective batch 8
python scripts/run_suite.py --preset q1_pilot --device cuda --amp auto \
  --batch-size 2 --grad-accum-steps 4 --resume --benchmark --sync-github \
  --train-labels /kaggle/input/bdd100k/labels_train.json \
  --val-labels /kaggle/input/bdd100k/labels_val.json \
  --train-images /kaggle/input/bdd100k/images/train \
  --val-images /kaggle/input/bdd100k/images/val
```
Inspect `work_dirs/q1_pilot_runs/PAPER_RESULTS.md` and `figures/conformal_*.png` to confirm AP climb and empirical FDR $\le \alpha_{\text{strict}}$ before launching the full matrix.

### Step 2: Start Full Q1 Matrix on Kaggle Tonight
```bash
python scripts/run_suite.py --preset q1_paper --device cuda --amp auto \
  --batch-size 2 --grad-accum-steps 4 --resume --sync-github \
  --train-labels /kaggle/input/bdd100k/labels_train.json \
  --val-labels /kaggle/input/bdd100k/labels_val.json \
  --train-images /kaggle/input/bdd100k/images/train \
  --val-images /kaggle/input/bdd100k/images/val
```
Before Kaggle session ends, export verified bundle:
```bash
python scripts/sync_workspace.py export --output work_dirs/q1_paper_runs --bundle /kaggle/working/q1_bundle.tar.gz --archive
```

### Step 3: Continue Tomorrow on RTX 5090 / L40s (2x–4x Faster, Identical Effective Batch = 8!)
```powershell
git pull origin feat/research-baseline
python scripts/sync_workspace.py import --bundle q1_bundle.tar.gz --output work_dirs\q1_paper_runs
python scripts/run_suite.py --preset q1_paper --device cuda --amp auto --batch-size 4 --grad-accum-steps 2 --resume --sync-github --output work_dirs\q1_paper_runs
```
Or simply double-click **`RUN_Q1_PAPER.cmd`** on Windows!

### Step 4: Optional Zero-Shot External Adverse-Weather Test (DAWN / ACDC)
```powershell
python scripts/eval_external.py --checkpoint work_dirs\q1_paper_runs\runs\10pct_s42\core_ssod\last.pth --coco-ann data\dawn\annotations.json --images data\dawn\images --output work_dirs\external_dawn\core_ssod --max-side 800
```
