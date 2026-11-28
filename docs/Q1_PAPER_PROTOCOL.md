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

### Mathematical Formulation (Section 3 Ready — CoRe-SSOD v2)

1. **Zero-Metadata Multi-Cue Photometric Degradation Risk $d(x) \in [0, 1]$** (`ssod_research/care_ssod.py`):
   Computed directly from unlabeled RGB pixels via luminance darkness, contrast deficit, dark-background specular glare, and dark-channel + low-chroma atmospheric haze (He et al., CVPR'09), partitioning scenes into `clear` ($d < 0.28$) and `degraded` ($d \ge 0.28$) without external weather labels.
2. **Direct Bidirectional Cross-View RoI Probing + Weighted Box Fusion** (`ssod_research/care_ssod.py`):
   Runs teacher detection once on the weak view ($F_1 \to b^{(1)}$) and runs **only the teacher backbone** on the horizontally flipped view ($F_2$, eliminating redundant second RPN/NMS and $O(N^2)$ matching cliffs). Probing $F_2$ at $\text{flip}(b^{(1)})$ directly yields paired cross-view probabilities $(p^{(1)}, p^{(2)})$ and refined boxes $b^{(2)}$, giving:
   - Weighted bidirectional box fusion $\bar{b}_i = \frac{s_i^{(1)} b_i^{(1)} + s_i^{(2)} b_i^{(2)}}{s_i^{(1)} + s_i^{(2)}}$ (higher localization precision),
   - Bidirectional probability consensus $\bar{p}_i = \frac{1}{2}(p_i^{(1)} + p_i^{(2)})$ and consensus score $\bar{s}_i = \frac{1}{2}(s_i^{(1)} + s_i^{(2)})$,
   - Vectorized $O(N)$ scale-adaptive NWD+IoU localization uncertainty $u_{\text{loc}}$ and Jensen-Shannon divergence $u_{\text{cls}}$,
   - Calibrated reliability score $q_i = \bar{s}_i \cdot \exp(-\alpha_{\text{loc}}\,u_{\text{loc}} - \alpha_{\text{cls}}\,u_{\text{cls}})$ with $(\alpha_{\text{loc}}, \alpha_{\text{cls}}) = (0.35, 0.20)$, plus cached $F_1$ features reused at zero extra cost for ROI soft-negative background weighting.
3. **Exact Monotone Weighted Conformal Risk Control + PAVA Precision Calibration** (`ssod_research/conformal.py`):
   Using a strictly disjoint calibration fold $\mathcal{D}_{\text{cal}} \subset \mathcal{D}_L$ (8% of labeled training images, capped at 256, zero validation leakage), we control the **naturally monotone, bounded False-Discovery-per-Object risk** (eliminating suffix-max envelope inflation while satisfying Angelopoulos et al., ICLR'24 Theorem 1 directly):
   $$L_{\text{FPR}}(x_i; \lambda) = \min\!\left(1,\; \frac{\sum_{j} \mathbb{1}[q_{ij} \ge \lambda \;\wedge\; y_{ij} = 0]}{\max(1, |G_i|)}\right) \in [0, 1]$$
   Weighted by the regime-conditional unlabeled-to-calibration photometric density ratio $w_r(d(x_i)) = p_U(\text{bin} \mid r) / p_{\text{cal}}(\text{bin} \mid r)$ (Tibshirani et al., NeurIPS'19) with Kish effective sample size $n_{\text{eff}}$:
   $$\hat{\lambda}_r(\alpha) = \inf \left\{ \lambda \in \Lambda : \frac{n_{\text{eff}}}{n_{\text{eff}} + 1}\hat{R}_r(\lambda) + \frac{1}{n_{\text{eff}} + 1} \le \alpha \right\}$$
   with $\alpha_{\text{strict}} = 0.10$ (Zone 1, $\tau_{\max} = 0.95$) and $\alpha_{\text{loose}} = 0.30$ (Zone 2), plus per-class Empirical-Bayes shrinkage computed only over calibration images containing candidate predictions of class $c$. Simultaneously (zero extra forward passes), 1D Pool-Adjacent-Violators (PAVA) isotonic regression fits $\hat{\pi}_r(q) \approx P(\text{TP} \mid q, r) \in [0, 1]$.
4. **Conformal Tri-Zone Loss with RPN Ignore Masking & ROI Soft-Negative Weighting** (`ssod_research/care_ssod.py`):
   - **RPN Zone-2 Ignore Masking**: Negative RPN anchors overlapping Zone-2 boxes ($\text{IoU} \ge 0.30$) are marked ignore ($-1$) rather than hard background.
   - **ROI Conformal Soft-Negative Background Weighting**: Background ROI proposals are weighted by $r_j = p_T(\text{bg} \mid b_j) \cdot (1 - \hat{\pi}_r(q_{z_2}))$, normalized to mean $1.0$ across background proposals so background calibration is strictly preserved (preventing false-positive explosion).
   - **Task-Decoupled Foreground Supervision**: Zone-1 positives drive reliability-weighted CE, localization-gated ($q_{\text{loc}} \ge 0.55$) Smooth-L1 + sampled-proposal GIoU ($w_{\text{giou}} = 0.25$), and mild full-distribution KL distillation ($w_{\text{kl1}} = 0.15, w_{\text{kl2}} = 0.05, T = 1.5$, linear $T$ scaling).
   Single student FPN backbone forward pass per unlabeled batch; zero extra inference cost at test time.

---

## 2. Controlled Experimental Matrix (`--preset q1_paper` & `--preset q1_pilot`)

- **Detector**: `Faster R-CNN ResNet50-FPN-v2` (`fasterrcnn_v2`) upgraded with **`FrozenBatchNorm2d`** in the pretrained ImageNet-V2 backbone body and **`GroupNorm(32, C)`** in the FPN & RoI heads (Wu & He, ECCV'18) + **EMAN** buffer EMA (Cai et al., CVPR'21), eliminating micro-batch BN corruption and guaranteeing exact mathematical invariance between Kaggle (`2×4`) and RTX 5090 (`4×2`).
- **Resolution & Augmentation**: `max_side = 800` (`800×450`), uniform 50% horizontal flip on labeled batches across all variants (including `supervised` and `warmup`), deterministic multi-worker I/O prefetching.
- **Optimization**: SGD (`lr = 0.01`, momentum 0.9, wd `1e-4`), **Effective Batch Size = 8**.
  - `q1_pilot`: Warmup `1,500` steps (`warmup_constant`, 200-step LR warmup) + SSOD `3,000` steps (`warmup_cosine`, 250-step LR warmup, `ema = 0.996`, `unsup_warmup_steps = 300`, `conformal_every = 300`, diagnostic 500-img val at steps 1500 & 3000).
  - `q1_paper`: Warmup `3,000` steps (`warmup_constant`, 300-step LR warmup) + SSOD `12,000` steps (`warmup_cosine`, 500-step LR warmup, `ema = 0.9985`, `unsup_warmup_steps = 1000`, `conformal_every = 500`, full 10,000-img val).
- **Matched Main Baselines** (5% & 10% × 3 seeds: 42, 123, 2026):
  1. `supervised` (Supervised-only continuation)
  2. `mean_teacher` (Fixed-threshold EMAN Mean Teacher)
  3. `soft_teacher` (Faithful Soft Teacher, ICCV'21: strictly decoupled `cls_valid` vs jitter-verified `reg_valid` + normalized Soft-Negative BG Weighting)
  4. `consistent_teacher` (Consistent-Teacher-style Dynamic Per-Class 1D GMM with Bayes posterior boundary, CVPR'23)
  5. `core_ssod` (**Proposed**: CoRe-SSOD v2)
- **Targeted Ablations** (seed 42, 5% & 10%):
  - `care_ssod` (Heuristic EMA threshold vs Conformal Risk Control)
  - `core_no_vis_weight` (Unweighted conformal vs photometric-shift weighted conformal)
  - `core_no_trizone` (2-zone hard filtering vs Tri-Zone RPN-ignore + Conformal Soft-Negative + soft KL)
  - `core_no_nwd` (IoU-only vs scale-aware NWD + JS)
  - `core_alpha005` ($\alpha_{\text{strict}} = 0.05$) & `core_alpha025` ($\alpha_{\text{strict}} = 0.25$)

---

## 3. Obtaining BDD100K

- **Official source**: https://bdd-data.berkeley.edu/ (or https://dl.cv.ethz.ch/bdd100k/data/) — requires free academic registration. Download `100K Images` (`train` + `val`) and `Detection 2020 Labels` (`bdd100k_labels_images_train.json`, `bdd100k_labels_images_val.json`).
- **On Kaggle**: Attach any verified public BDD100K dataset (e.g., search `bdd100k` in Kaggle Datasets) and point `TRAIN_LABELS`, `VAL_LABELS`, `TRAIN_IMAGES`, `VAL_IMAGES` to `/kaggle/input/<dataset>/...`.

---

## A100 reference run

From the repository root on the server, activate the existing PyTorch environment and update the research branch. The input plan supplies the four BDD paths used in the earlier pilot. The new output directory keeps its checkpoints separate.

```bash
git switch feat/research-baseline
git pull --ff-only origin feat/research-baseline
source /data/venvs/ssod/bin/activate

# Option A: Single-seed 10% reference on full 10,000-image validation (preset q1_a100)
python scripts/run_suite.py --preset q1_a100 --device cuda --amp bf16 \
  --dataset-from-plan /data/outputs/q1_pilot_legacy_bs40/experiment_plan.json \
  --output /data/outputs/q1_a100_reference_v1

# Option B: Full Q1 multi-seed + ablations matrix on A100 40GB/80GB (or bash RUN_Q1_A100.sh)
python scripts/run_suite.py --preset q1_paper --device cuda --amp bf16 \
  --batch-size 8 --grad-accum-steps 1 --resume --sync-github \
  --dataset-from-plan /data/outputs/q1_pilot_legacy_bs40/experiment_plan.json \
  --output /data/outputs/q1_paper_a100
```

If the old plan is unavailable, supply `--train-labels`, `--val-labels`, `--train-images` and `--val-images` directly. Repeat the same command to resume. To run just preparation, warm-up and the supervised control first, add `--run-limit 3` on the first invocation.

The reference uses the 10% labeled split, seed 42, Faster R-CNN v2 with an ImageNet-pretrained ResNet-50, 800-pixel long side, micro-batch 2 and four accumulation steps. The shared supervised warm-up runs for 1,500 training steps. Its checkpoint initializes each of five 3,000-step runs. All branches reserve the same calibration fold before warm-up. Evaluation uses all images in the prepared validation split at steps 1,500 and 3,000, with final per-class and condition reports. The experiment plan records the generated commands. Each run manifest includes the source revision, dataset hashes, effective configuration and measured metrics in the adjacent `evaluations.jsonl`.

Historical reference values on the 0–100 AP scale are 13.85 for warm-up, 17.81 for supervised and 18.15 for CoRe. The old validation IDs, effective batch and raw output are unavailable. Compare supervised and CoRe at the same step in this suite and record their difference in AP points. The historical difference was +0.34. A single seed measures this run only.

For a checkpoint-level comparison, run `scripts/eval_external.py` twice with the same `--checkpoint`, validation `--coco-ann`, `--images` and `--max-side 800`. Select `--weights teacher` and `--weights student` with distinct `--output` directories. Use the same image limit for both.

The supervised burn-in followed by EMA teacher updates follows the training sequence described in [Unbiased Teacher v2](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Unbiased_Teacher_v2_Semi-Supervised_Object_Detection_for_Anchor-Free_and_Anchor-Based_CVPR_2022_paper.html). The reference code omits images with zero accepted boxes from hard pseudo-label detection losses. CoRe can still use ambiguous boxes for soft distillation. This implementation choice targets the false-negative supervision described in [Data-Uncertainty Guided Multi-Phase Learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Data-Uncertainty_Guided_Multi-Phase_Learning_for_Semi-Supervised_Object_Detection_CVPR_2021_paper.pdf).

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
