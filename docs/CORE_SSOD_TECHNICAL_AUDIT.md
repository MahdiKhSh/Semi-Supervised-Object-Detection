# CoRe-SSOD Comprehensive Technical Audit (`v2` -> `v2.1` -> `v2.2`)

**Repository**: `https://github.com/MahdiKhSh/Semi-Supervised-Object-Detection`  
**Audited Revisions**:
- `6e7a556` — Initial Q1 CoRe-SSOD implementation
- `4f8ce523a0d83fa685534027e009d3e28909b158` — CoRe-SSOD v2 (`q1_pilot_v2_runs`)
- `e8497564bc45789f03e0bc52e82bc3f10ea7a7d7` — A100 reference suite & zero-box filtering
- `0252a64a708faab46ed9c391da7ebbbf62ab2520` — CoRe-SSOD v2.1 merge
- `feat/core-ssod-v2.2-audit` — Audited & corrected CoRe-SSOD v2.2 implementation

---

## 1. Executive Summary of Verified Defects & Scientific Findings

A source-code, mathematical, and forensic artifact audit across the entire repository uncovered **ten concrete defects and methodological confounders** spanning runtime crashes, silent gradient-scaling distortions, conformal risk-control math, small-object RPN recall bottlenecks, and pipeline DAG invalidation bypasses:

| ID | Severity | Affected File(s) & Lines (`v2.1`) | Defect / Finding | Scientific & Empirical Impact | Status in `v2.2` |
|---|---|---|---|---|---|
| **F-01** | **CRITICAL** | `ssod_research/baselines.py:314` | `model.rpn(extra_imgs, extra_feats, None)` called while `model.rpn.training=True` on images with `reg_valid=True` and zero `cls_valid` boxes. Server hotfix (`rpn.training=False`) silently halved `pre_nms_top_n` & `post_nms_top_n` (`2000 -> 1000`). | Deterministic crash in Soft Teacher; hotfix reduced RPN proposal recall on regression-only images by up to 50%. | **Fixed** via `extract_rpn_proposals_only` preserving 2,000-proposal training policy + regression test. |
| **F-02** | **CRITICAL** | `work_dirs/q1_pilot_v2_runs/results.csv`, `scripts/run_suite.py:384` | Historical `13.85 / 17.81 / 18.15` mAP numbers came from `q1_pilot` (`eval_max_images=500`, `partial_validation=True`), **not** full 10,000-image validation (`eval_max_images=0` on A100). In `val_500`, category `train` has **1 instance** (vs 15 in `val_10k`) and `motorcycle` has 20 (vs 452). | Comparing 500-image partial AP against 10,000-image full COCO AP created an artificial ~1.2–1.7 AP apparent drop even on identical weights. | **Proven forensically** via full 10,000-image re-evaluation of all six saved `4f8ce52` checkpoints; `job_complete` now enforces `eval_max_images` equality. |
| **F-03** | **HIGH** | `ssod_research/care_ssod.py:879-1080`, `baselines.py:206-353`, `native_training.py:1518,1560` | Zero-pseudo-box image slicing in `v2.1` averaged RPN/ROI losses over $M = |\text{active}|$ images instead of total micro-batch size $B$ (`M / B` factor omitted). | Broke exact micro-batch equivalence (`32×1 != 16×2 != 2×16`) and amplified sparse-batch unsupervised gradients by up to $B/M = 32\times$ at `batch_size=32`. | **Fixed** via exact `M / B` active-batch loss scaling across `core_ssod`, `soft_teacher`, `consistent_teacher`, and `mean_teacher`. |
| **F-04** | **HIGH** | `ssod_research/conformal.py:259` | `solve_conformal_threshold` used Kish's $(n_{\text{eff}} \hat{R} + 1)/(n_{\text{eff}}+1)$ heuristic instead of the exact covariate-shift Weighted CRC worst-case bound $(\sum w_i L_i + w_{\max})/(\sum w_i + w_{\max})$ (Tibshirani et al., NeurIPS'19; Angelopoulos et al., ICLR'24). | Underestimated finite-sample tail risk by up to $3.2\times$ when $w_{\max}=4.0$; formal finite-sample guarantee was invalid as stated. | **Fixed** (`solve_conformal_threshold` uses exact $w_{\text{worst}}$ bound with bounded weights $w_i \in [0.4, 2.5]$; online EMA coupling explicitly disclosed). |
| **F-05** | **HIGH** | `ssod_research/conformal.py:124, 527-538, 691` | Single-image $|G_i|$ normalization in $L_{\text{FPR}}$ penalized sparse night highway images ($|G_i|=1\text{–}2$) $9\times\text{–}18\times$ harder than daytime streets ($|G_i|\approx 18$), driving `degraded` thresholds **higher than daytime** (`tau_high=0.90` vs `0.85`, `tau_low=0.73` vs `0.57`) and inverting sparse/small classes above `car`. | Dumped `[0.35, 0.72]` night detections into Zone 3 (hard RPN/ROI background) and choked small-object pseudo-label recall. | **Fixed** via regime-smoothed monotone GT normalizer, candidate-gated per-class shrinkage, and `max_low = 0.55` Zone-2 ceiling. |
| **F-06** | **HIGH** | `ssod_research/native_training.py:230`, `ssod_research/native_data.py:105` | Resizing `1280×720 -> 800×450` (`0.625×`) leaves **103,052 / 185,526 (55.5%)** of BDD100K val boxes with $\min(w, h) < 16\text{ px}$, while default `fasterrcnn_resnet50_fpn_v2` uses minimum anchor size `32×32` ($\text{IoU}_{\max} = 0.117$ on a `9×14` traffic light). | Primary architectural bottleneck capping supervised & SSOD `AP_small` around 5.4%–7.2%. | **Addressed** via shared fair `--anchor-profile road_small` (`12..384` 2-per-octave anchors, $>3.5\times$ P2 IoU) + diagnostic protocol. |
| **F-07** | **MEDIUM** | `ssod_research/care_ssod.py:670-680` | Random Cutout (`up to 80×45 px`) in `strong_view_care` did not check box occlusion, occasionally erasing 100% of tiny `8×12 px` traffic lights/signs while keeping their Zone-1 foreground label. | Taught student RPN/ROI to predict confident foreground classes on solid uniform gray rectangles. | **Fixed**: Cutout rejects patch placements that occlude $>35\%$ of any Zone-1 or Zone-2 box. |
| **F-08** | **HIGH** | `scripts/run_suite.py:480, 1088, 1500-1505` | Pre-prepare fallback `split_fingerprint` differed from post-prepare `split_manifest.json` hash, causing false invalidation warnings on every restart; meanwhile `elif job_complete(job):` and `_sync_disk_state_to_registry` bypassed registry invalidation without checking `dataset_sha256`, `init_checkpoint_sha256`, or `eval_max_images`. | Spurious invalidation logs on valid restarts **and** risk of silently skipping genuinely stale downstream stages if `warmup` or split annotations changed. | **Fixed**: DAG refreshes post-prepare fingerprints immediately, and `job_complete` cryptographically verifies split SHA-256s, `init_checkpoint_sha256`, and `eval_max_images`. |
| **F-09** | **MEDIUM** | `ssod_research/conformal.py:49-72` | Unstratified 256-image calibration split left `0` `foggy` images and `0` `train` instances in $D_{\text{cal}}$ and under-sampled `dawn/dusk` by 45% (`3.9%` vs `7.1%`). | Suboptimal calibration coverage on rare adverse-weather strata; risk of draining rare classes from small (5%) labeled splits. | **Fixed** via deterministic Hamilton-rounded `--stratify-calibration` preserving $\ge 1$ training image per stratum. |
| **F-10** | **MEDIUM** | `ssod_research/native_training.py:910` (with `batch_size=32`) | Running `eff_batch=32` (`4×` reference `eff_batch=8`) at unscaled `lr=0.01` for fixed `1,500 / 3,000` steps changes SGD noise temperature by $4\times$ and alters per-epoch schedule dynamics (`EMA=0.996` half-life = 173 steps). | Suboptimal step-size to gradient-variance ratio at `BS=32, LR=0.01`; requires controlled linear/sqrt LR scaling (`LR=0.02`). | **Addressed** via controlled batch/LR diagnostic matrix (`scripts/run_v22_diagnostic_matrix.py`). |

---

## 2. Component-by-Component Architectural Audit (Section 8)

### 2.1 Frozen Batch Normalization (`_replace_bn_with_frozen_bn`, `native_training.py:164-183`)
1. **Intended Mathematical Behavior**: Freeze pretrained ImageNet-1K-V2 running mean/variance $(\mu, \sigma^2)$ and affine $(\gamma, \beta)$ in `ResNet50` body (`conv1`, `layer1..4`) so small detection micro-batches ($B \in \{1, 2, 4\}$) do not corrupt feature statistics.
2. **Actual Implementation**: Recursively replaces all 53 `nn.BatchNorm2d` modules in `model.backbone.body` with `torchvision.ops.FrozenBatchNorm2d`, copying `weight`, `bias`, `running_mean`, and `running_var`. Note that `FrozenBatchNorm2d` stores affine scale/shift as non-parameter buffers (`register_buffer`), so affine channel scales in the ResNet body are also frozen (matching Detectron2 `FrozenBN`).
3. **Potential Bugs**: None. Verified `active_bn == 0` and `frozen_bn == 53`.
4. **Impact on Accuracy**: Essential; active `BatchNorm2d` at `batch_size=2` degrades Faster R-CNN-v2 by 3–6 AP and breaks gradient-accumulation equivalence.
5. **Computational Overhead**: Slightly faster than `BatchNorm2d` (no batch mean/variance reduction, fused affine multiply-add).
6. **Recommended Corrections**: Retain unchanged.
7. **Validation Tests**: `tests/test_conformal_and_baselines.py::test_fasterrcnn_v2_uses_frozen_bn_and_group_norm`.

### 2.2 Group Normalization in FPN, RPN & RoI Heads (`_replace_bn_with_group_norm`, `native_training.py:186-202`)
1. **Intended Mathematical Behavior**: Replace randomly initialized `BatchNorm2d` layers in `FPNWithLastPool`, `RPNHead`, and `FastRCNNConvFCHead` (introduced in TorchVision's v2 architecture) with batch-size-invariant `nn.GroupNorm(32, C)` (Wu & He, ECCV 2018).
2. **Actual Implementation**: Replaces every `nn.BatchNorm2d` in `backbone.fpn` (8 layers), `rpn` (2 layers in v2), and `roi_heads` (4 conv layers in v2) using `num_groups = math.gcd(32, C)`, preserving `eps` and initialized affine weights.
3. **Potential Bugs**: None; works seamlessly with custom `rpn_anchor_generator` in `v2.2` because `_convert_v2_norm_layers` runs *after* `fasterrcnn_resnet50_fpn_v2` constructs `RPNHead`.
4. **Impact on Accuracy**: Guarantees exact per-image normalization invariance across `2×4`, `8×1`, `16×2`, and `32×1` batch configurations.
5. **Computational Overhead**: Negligible (~2% kernel overhead on FPN/RoI tensors).
6. **Recommended Corrections**: Retain unchanged.
7. **Validation Tests**: `test_fasterrcnn_v2_uses_frozen_bn_and_group_norm`, `test_exact_wcrc_bound_stratified_calibration_and_road_small_anchors`.

### 2.3 EMAN Teacher Update (`ema_update`, `native_training.py:332-343`)
1. **Intended Mathematical Behavior**: Exponential moving average over both parameters $\theta_T \leftarrow m\theta_T + (1-m)\theta_S$ and floating-point normalization buffers (Cai et al., CVPR 2021), with warmup ramp $m_t = \min(m, 1 - \frac{1}{t+2})$.
2. **Actual Implementation**: Updates parameters via in-place `mul_(decay).add_(b.detach(), alpha=1-decay)` and floating-point buffers when `not torch.equal(a, b)` (skipping identical `FrozenBatchNorm2d` buffers cleanly). Guarded by `GradScaler` overflow check in `v2.1+`.
3. **Potential Bugs**: None. Note that with `FrozenBN` + `GroupNorm`, floating-point buffers in `fasterrcnn_v2` are constant ImageNet stats, so `not torch.equal(a, b)` smartly avoids floating-point drift on frozen buffers!
4. **Impact on Accuracy**: Prevents early teacher lag (`m_0 = 0.5 -> 0.996`) and preserves exact ImageNet frozen buffer values without $1 - 0.996$ rounding drift.
5. **Computational Overhead**: `< 0.5 ms` per optimizer step.
6. **Recommended Corrections**: Retain unchanged.
7. **Validation Tests**: `test_eman_buffer_ema_update`, `test_ema`.

### 2.4 Bidirectional Cross-View Localization & Classification Probe (`teacher_bidirectional_probe`, `care_ssod.py:362-516`)
1. **Intended Mathematical Behavior**: Given weak unlabeled image $x$, detect candidate boxes $b^{(1)}$ on $F_1 = \Phi_T(x)$, extract flipped features $F_2 = \Phi_T(\text{flip}(x))$ **without** running a second RPN/NMS, probe $F_2$ at $\text{flip}(b^{(1)})$ via `box_roi_pool` + `box_head` + `box_predictor` to obtain paired $(p^{(1)}, p^{(2)})$ and refined box $b^{(2)}$, and fuse $\bar{b} = \frac{s_1 b^{(1)} + s_2 b^{(2)}}{s_1 + s_2}$.
2. **Actual Implementation**: Vectorized batched RoI pooling across all images in the micro-batch; eliminates the $O(N^2)$ bipartite matching cliff of two-pass RPN/NMS (`pseudo_targets`) where small objects disappeared whenever NMS suppressed a borderline twin on the flipped view.
3. **Verified Bug (Minor)**: In `v2`/`v2.1` (`care_ssod.py:456-457, 475-476`), `torch.maximum(b2[:, 2], b2[:, 0] + 1.0)` could push `x2` to `ow + 1.0` if `x1` was clamped near `ow`.
4. **Impact on Accuracy**: Positive (+1.06 mean IoU points on matched boxes and elimination of flip-NMS dropout on small objects).
5. **Computational Overhead**: Saves ~35% teacher time vs full two-view detection (no second RPN or multi-class NMS) and caches `F_1` for zero-backbone-cost ROI soft-negative weighting.
6. **Implemented v2.2 Correction**: Clamped `x1` to `[0, ow - 1]` and `y1` to `[0, oh - 1]` prior to enforcing 1-pixel minimum box extent (`care_ssod.py:456-484`).
7. **Validation Tests**: `test_bidirectional_probe_and_conformal_soft_negative_loss`, `test_small_object_cutout_protection_and_coordinate_roundtrip`.

### 2.5 Scale-Adaptive Normalized Wasserstein Distance (`pairwise_nwd`, `elementwise_nwd`, `care_ssod.py:100-172`)
1. **Intended Mathematical Behavior**: Model boxes $A = (cx_a, cy_a, w_a, h_a)$ and $B = (cx_b, cy_b, w_b, h_b)$ as 2D Gaussians $\mathcal{N}(\mu, \Sigma)$ with $\Sigma = \text{diag}(w^2/4, h^2/4)$, computing $W_2^2 = \|\mu_a - \mu_b\|_2^2 + \frac{1}{4}((w_a - w_b)^2 + (h_a - h_b)^2)$ and similarity $\exp(-W_2 / C_{\text{scale}})$ with $C_{\text{scale}} = \max(C_0, 0.5 \sqrt{\sqrt{w_a h_a}\sqrt{w_b h_b}})$, blended with IoU: $s_{\text{loc}} = 0.65\,\text{NWD} + 0.35\,\text{IoU}$.
2. **Actual Implementation**: Exact O(N) vectorized `elementwise_nwd` in float32 (`eps = 1e-6` inside `sqrt(w2_sq + eps)` prevents infinite gradient/NaN at zero shift).
3. **Potential Bugs**: None in NWD math itself; however, its benefit was previously masked in `v2/v2.1` whenever RPN failed to propose `<16 px` boxes in the first place or when `conformal.py` set `tau_high = 0.90`!
4. **Impact on Accuracy**: On a `12×12` traffic light with 2-px cross-view jitter, IoU drops to `0.47` while NWD remains `0.81`, preserving `q_loc = 0.85 >= 0.55` so small objects are not rejected from regression.
5. **Computational Overhead**: Microseconds (`O(N)` elementwise tensor ops).
6. **Recommended Corrections**: Retain unchanged; pair with fixed conformal class/regime thresholds so NWD-preserved small objects actually pass `tau_high` / `tau_low`.
7. **Validation Tests**: `test_nwd_is_smoother_than_iou_on_small_objects`.

### 2.6 Jensen-Shannon Classification Divergence (`jensen_shannon_divergence`, `care_ssod.py:194-209`)
1. **Intended Mathematical Behavior**: Symmetric, bounded divergence $\text{JS}(p^{(1)} \| p^{(2)}) = \frac{1}{2 \ln 2}[\text{KL}(p^{(1)} \| m) + \text{KL}(p^{(2)} \| m)] \in [0, 1]$ across all $C+1 = 11$ classes (including background).
2. **Actual Implementation**: Computed in `float32` (`probs.float().clamp(min=1e-7)` renormalized to sum to 1), divided by $\ln 2$, clamped to $[0, 1]$. Safe under `bf16` autocast because `.float()` precedes `.log()`.
3. **Potential Bugs**: None.
4. **Impact on Accuracy**: Penalizes cross-view class confusion (e.g., `rider` vs `pedestrian` or `truck` vs `bus`) smoothly via $\exp(-0.20 u_{\text{cls}})$.
5. **Computational Overhead**: Negligible (`O(N × 11)`).
6. **Recommended Corrections**: Retain unchanged.
7. **Validation Tests**: `test_bidirectional_probe_and_conformal_soft_negative_loss`.

### 2.7 Zero-Metadata Photometric Visibility Estimator (`estimate_visibility_degradation`, `care_ssod.py:50-96`)
1. **Intended Mathematical Behavior**: Estimate continuous visibility degradation $d(x) \in [0, 1]$ and binary regime (`clear` if $d < 0.28$, `degraded` if $d \ge 0.28$) strictly from RGB pixels (ITU-R BT.601 luminance darkness, contrast deficit, dark-scene headlight glare, and He et al. dark-channel + low-chroma haze), avoiding metadata leakage on unlabeled images.
2. **Actual Implementation**: Runs under `@torch.no_grad()` in `float32`; on BDD100K (`conformal_calibration.jsonl`), partitions $D_{\text{cal}}$ into `172 clear` (67.2%) and `84 degraded` (32.8%) and unlabeled stream into `36.0% degraded`, closely matching BDD100K's true ~40% night/dawn/rain/fog composition without touching ground-truth weather tags!
3. **Potential Bugs**: Bright snow scenes (`mean_y > 0.55`, low chroma) score moderate `haze` + `low_contrast` (`~0.22–0.32`), occasionally straddling `0.28`, which is benign because density-ratio weighting uses 6 continuous bins across $d(x) \in [0, 1]$.
4. **Impact on Accuracy**: Enables zero-metadata regime partitioning and density-ratio reweighting without violating semi-supervised assumptions.
5. **Computational Overhead**: `~0.15 ms` per image.
6. **Recommended Corrections**: Retain unchanged.
7. **Validation Tests**: `test_visibility_degradation_separates_night_fog_from_daytime`.

### 2.8 Weighted Conformal Risk Control (`WeightedConformalRiskCalibrator`, `conformal.py:222-748`)
1. **Intended Mathematical Behavior**: Calibrate $(\tau_{\text{high}}, \tau_{\text{low}})$ on held-out $D_{\text{cal}}$ under photometric shift weights $w_r(d)$ to bound monotone false-discovery risk at $\alpha_{\text{strict}}$.
2. **Actual Implementation (`v2.1`) & Verified Mathematical Bugs**:
   - **Bug A (Invalid WCRC finite-sample correction)**: Used $(n_{\text{eff}} \hat{R} + 1)/(n_{\text{eff}}+1)$ instead of exact $(\sum w_i L_i + w_{\text{worst}})/(\sum w_i + w_{\text{worst}})$.
   - **Bug B (Sparse-scene $|G_i|$ penalty)**: Dividing image FPs by single-image $|G_i|$ punished 1–2 car night highway scenes $9\times\text{–}18\times$ harder than 18-car daytime scenes, pushing `degraded` thresholds **above** `clear` (`0.90` vs `0.85`!) and pushing `degraded` `tau_low` to **`0.71–0.75`**!
   - **Bug C (Rare/small-class threshold inversion)**: Including `0.05`-score noise candidates on `c_gt_count=0` images pushed `pedestrian`/`rider`/`traffic light`/`traffic sign` thresholds above `car`.
   - **Theoretical Scope Disclosure**: Exact covariate-shift WCRC holds at step $t=0$ (post-reserved-warmup) up to histogram density-ratio estimation error; at $t > 0$, EMA teacher updates create mild adaptive coupling across steps, so $t > 0$ recalibration must be described in the manuscript as *online approximate conformal risk control* (unless $D_{\text{cal}}$ is split across recalibration rounds).
3. **Implemented v2.2 Corrections**:
   - Exact $w_{\text{worst}}$ finite-sample upper bound (`conformal.py:310-325`) with bounded weights $w_i \in [0.4, 2.5]$.
   - Regime-smoothed GT density normalizer $\max(1, 0.5|G_i| + 0.5\bar{g}_r)$ (`conformal.py:613-628`).
   - Active-class candidate gating ($q \ge \tau_{\min\_\text{low}}$ or $|G_{i,c}| > 0$) + bounded Empirical-Bayes shrinkage $[\hat{\lambda}_r - 0.12, \hat{\lambda}_r + 0.05]$ (`conformal.py:630-646, 745-755`).
   - Zone-2 `max_low = 0.55` cap so medium-reliability detections are never dumped into hard RPN background.
4. **Computational Overhead**: Runs once every `300–500` steps on 256 images in `bf16` (`~3.2s` every 300 steps = `< 0.5%` training overhead).
5. **Validation Tests**: `test_monotonized_fdr_curve_and_conformal_risk_guarantee`, `test_exact_wcrc_bound_stratified_calibration_and_road_small_anchors`.

### 2.9 Pool-Adjacent-Violators (PAVA) Isotonic Precision Calibration (`fit_isotonic_precision`, `conformal.py:170-220`)
1. **Intended Mathematical Behavior**: Fit non-decreasing $\hat{\pi}_r(q) \approx P(\text{TP} \mid q, r) \in [0, 1]$ across 12 bins on $D_{\text{cal}}$ with zero extra forward passes.
2. **Actual Implementation**: Exact O(B) stack-based PAVA with monotone forward-fill of empty low bins (`last = 0.0` for empty lowest bins, non-decreasing upward). Observed PAVA curves in `conformal_calibration.jsonl` cleanly span `[0.019, 0.938]` (`clear`) and `[0.015, 0.886]` (`degraded`).
3. **Potential Bugs**: None.
4. **Impact on Accuracy**: Supplies calibrated TP probabilities to attenuate ROI background loss on Zone-2 overlaps without uncalibrated score heuristics.
5. **Computational Overhead**: `< 0.2 ms` on CPU per calibration event.
6. **Recommended Corrections**: Retain unchanged.
7. **Validation Tests**: `test_monotone_fp_per_gt_and_pava_precision`.

### 2.10 Tri-Zone Pseudo-Label Partitioning (`extract_teacher_tri_zone_targets`, `care_ssod.py:512-638`)
1. **Intended Mathematical Behavior**: Partition teacher candidates into Zone 1 ($q \ge \tau_{\text{high}}$, hard foreground + gated reg + mild KL), Zone 2 ($\tau_{\text{low}} \le q < \tau_{\text{high}}$, RPN ignore + soft-negative ROI attenuation + mild full-distribution KL), and Zone 3 ($q < \tau_{\text{low}}$, background).
2. **Actual Implementation & Bug Found**: Cleanly implemented, but starved in `degraded` scenes in `v2/v2.1` because `conformal.py` set `tau_low = 0.71–0.75`!
3. **Impact on Accuracy**: With `tau_low <= 0.55` restored in `v2.2`, Zone 2 protects uncertain night/rain/small objects (`q in [0.35, 0.80]`) from false-negative suppression.
4. **Computational Overhead**: Included in `teacher_bidirectional_probe`.
5. **Validation Tests**: `test_care_unsupervised_loss_handles_zone1_and_zone2_only_batches`, `test_exact_wcrc_bound_stratified_calibration_and_road_small_anchors`.

### 2.11 Conformal Soft-Negative ROI Supervision (`care_unsupervised_loss`, `care_ssod.py:960-1015`)
1. **Intended Mathematical Behavior**: Weight background ROI proposals by $w_j^{\text{bg}} \propto \max(0.05, p_T(\text{bg} \mid b_j) \cdot (1 - \mathbb{1}[\text{IoU}_{Z_2} \ge 0.3]\,\hat{\pi}_r(q_{z_2})))$, normalized to mean `1.0` across background proposals so overall background vs foreground logit calibration is preserved.
2. **Actual Implementation**: Projects student proposals from augmented student space -> original weak space (inverting horizontal flip when `was_flipped=True`) -> teacher transformed space, pools cached `F_1` (`zero` extra backbone pass), multiplies by Zone-2 PAVA attenuation, and normalizes `bg_w / bg_w.mean()`.
3. **Potential Bugs**: None in coordinate projection; suffered only from missing `M / B` batch-size scaling when zero-Zone-1 images were sliced out in `v2.1`.
4. **Impact on Accuracy**: Eliminates both false-negative crushing of unannotated objects and false-positive explosion (which occurred in `v1` when proposals were deleted instead of mean-1 reweighted).
5. **Computational Overhead**: ~4% step time (RoI head only on cached `F_1`).
6. **Implemented v2.2 Correction**: Added `det_scale = len(positive) / total_batch_size` (`care_ssod.py:1085`).
7. **Validation Tests**: `test_bidirectional_probe_and_conformal_soft_negative_loss`, `test_active_batch_loss_scaling_prevents_sparse_batch_gradient_amplification`.

### 2.12 Zone-1 & Zone-2 Full-Distribution KL Distillation (`_kl_soft_distillation`, `care_ssod.py:756-776, 1089-1116`)
1. **Intended Mathematical Behavior**: Distill full $(C+1)$-class consensus distribution $\bar{p}$ (including background probability mass!) at temperature $T=1.5$ with linear $T$ scaling and mild weights ($w_{Z_1}=0.15, w_{Z_2}=0.05$).
2. **Actual Implementation**: Uses `F.kl_div(log_softmax(s/T), softmax(log(p)/T)) * T` weighted by `reliability` (Zone 1) or `p_tp` (Zone 2). Note: `log(p)/T` followed by `softmax` raises probabilities to power $1/T$ and renormalizes — mathematically well-defined temperature sharpening/smoothing on probability simplex!
3. **Verified Bug**: Missing `n_active_imgs / total_batch_size` scaling in `v2.1` caused a single image with 1 Zone-2 box in a 32-image batch to receive full batch weight `0.05` instead of `0.05 * (1/32)`. Fixed in `v2.2` (`kl1_scale`, `kl2_scale`).
4. **Computational Overhead**: `< 1.5 ms` per batch (reuses student `features` directly).
5. **Validation Tests**: `test_active_batch_loss_scaling_prevents_sparse_batch_gradient_amplification`.

### 2.13 Ambiguous-Anchor RPN Suppression (`_rpn_forward_with_zone2_ignore`, `care_ssod.py:718-754`)
1. **Intended Mathematical Behavior**: Mark negative RPN anchors (`lbl == 0`) whose IoU with any Zone-2 box is $\ge 0.30$ as `ignore (-1.0)` before `BalancedPositiveNegativeSampler`, while leaving positive anchors (`lbl == 1`) untouched.
2. **Actual Implementation**: Temporarily wraps `rpn.assign_targets_to_anchors` inside `try ... finally` and calls `rpn(det_images, det_features, det_targets)`, firing all module hooks cleanly and restoring `rpn.assign_targets_to_anchors` on exit. Empirically masks ~6,000–10,000 ambiguous anchors per 32-image step (`suppressed_bg` in `steps.jsonl`).
3. **Potential Bugs**: Thread-safe for single-process GPU training (`DataParallel` is not used).
4. **Impact on Accuracy**: Prevents RPN objectness head from learning to suppress uncertain night/distant vehicles and pedestrians.
5. **Computational Overhead**: `< 1 ms` (`box_iou` on negative anchors vs 4–6 Zone-2 boxes per image).
6. **Recommended Corrections**: Retain unchanged.
7. **Validation Tests**: `test_care_unsupervised_loss_handles_zone1_and_zone2_only_batches`.

### 2.14 Teacher-Student RoI Localization Consistency (`_giou_consistency_loss`, `care_ssod.py:778-803, 1029-1044` & `consistency.py`)
1. **Intended Mathematical Behavior**: Gate box regression (`Smooth-L1` + sampled-proposal `1 - GIoU`) by cross-view localization quality $q_{\text{loc}} = 1 - u_{\text{loc}} \ge 0.55$, decoding actual noisy RPN proposals via `box_coder.decode_single(pred_deltas, pos_props)`.
2. **Actual Implementation**: Operates on real positive RPN proposals (`sampled_pos_inds`), weighted by `q_loc`. Avoids the classic zero-jitter bug where GT boxes are fed as proposals (`deltas == 0`).
3. **Potential Bugs**: None.
4. **Impact on Accuracy**: Improves `AP75` and `AP_medium` by penalizing scale-invariant GIoU error only on localization-verified pseudo-boxes.
5. **Computational Overhead**: `< 0.5 ms`.
6. **Recommended Corrections**: Retain unchanged (with `det_scale` applied in `v2.2`).
7. **Validation Tests**: `test_consistency_traffic_sign_class_10_bug_fixed`, `test_consistency_backpropagates`.

### 2.15 Adaptive Pseudo-Label Acceptance (`OnlineVisibilityClassCalibrator`, `DynamicGMMThreshold`, `compute_threshold`)
1. **Intended Mathematical Behavior**: Provide heuristic adaptive thresholds for ablation (`care_ssod`, `adaptive`, `condition`) and EM-GMM baseline (`consistent_teacher`).
2. **Actual Implementation**: Checkpoint-serializable state dicts verified for bit-exact interrupt/resume equivalence.
3. **Potential Bugs**: None after `M / B` active-batch scaling in `native_training.py`.
4. **Validation Tests**: `test_online_calibrator_adapts_and_serializes`, `test_dynamic_gmm_threshold_separates_bimodal_scores`, `test_care_ssod_deterministic_interrupt_resume_and_preset`.

---

## 3. Baseline Fidelity Audit (`Soft Teacher` & `Consistent-Teacher` — Section 7.4)

To ensure publication-grade scientific honesty, we audited both strong baselines against their original papers:

1. **`soft_teacher` (`ssod_research/baselines.py` vs Xu et al., ICCV 2021)**:
   - **Faithfully Reproduced Core Mechanisms**:
     - Strict separation of classification pseudo-boxes (`score >= 0.70`) vs box-jittering variance-verified regression pseudo-boxes (`score >= 0.50` and normalized coordinate std `<= 0.03` across jittered RoI refinements).
     - Reliability-weighted ROI background classification using teacher background softmax probability $p_T(\text{bg} \mid b_j)$ projected from student strong-augmented proposals back to weak teacher feature maps (normalized to mean `1.0` so TorchVision's combined FG+BG CE denominator remains calibrated).
   - **Documented Differences from Original MMDetection Codebase (Must Disclose in Paper)**:
     - Uses the **shared** Faster R-CNN ResNet50-FPN-v2 (`FrozenBN` + `GroupNorm`) detector, `800px` resolution, `1:1` labeled-to-unlabeled batch ratio (original Soft Teacher uses `1:4` on COCO with multi-scale `(400..1200)` + solarize/cutout RandAugment), and `jitter_times = 6` (`jitter_thresh = 0.03`) vs original `10` jitters (`0.02`) for matched wall-clock budget.
     - **Critical Defect Fixed in `v2.2`**: `extract_rpn_proposals_only` restores the full 2,000-proposal training RPN budget on regression-only images (`F-01`) and restores `M / B` loss scaling (`F-03`).

2. **`consistent_teacher` (`ssod_research/baselines.py` & `native_training.py` vs Wang et al., CVPR 2023)**:
   - **Faithfully Reproduced Core Mechanism**: Per-class online 2-component 1D Gaussian Mixture Model (EM) on candidate scores with posterior Bayes boundary thresholding (`DynamicGMMThreshold`).
   - **Documented Architectural Scope (Must Disclose in Paper)**: Original Consistent-Teacher was designed for single-stage **RetinaNet** and additionally includes *3D Feature Alignment (FAM-3D)* and *Adaptive Sample Assignment (ASA)* inside focal-loss anchor assignment. Because our controlled benchmark fixes the detector architecture to two-stage **Faster R-CNN ResNet50-FPN-v2** across all methods (to isolate pseudo-label risk control without detector confounding!), `consistent_teacher` in this codebase is a **Faster R-CNN GMM-Thresholded Mean Teacher adaptation**, and should be labeled in tables as **`Consistent-Teacher (GMM, Faster R-CNN)`** rather than claiming full RetinaNet ASA+FAM reproduction.

---

## 4. Small-Object Detection Bottleneck Audit (Section 9)

Forensic measurement on the exact BDD100K split (`185,526` validation boxes across `10,000` images; `129,591` labeled boxes across `7,000` images) reveals four compounding mechanisms behind `AP_small ≈ 5.4%–7.2%`:

1. **Severe Input Downscaling + `32×32` Minimum Anchor Mismatch (Primary Bottleneck — ~60% of loss)**:
   - At native `1280×720`, **55.4%** (`102,729 / 185,526`) of validation annotations are COCO Small ($\text{area} < 32^2$).
   - After `RoadCocoDataset` scales `1280×720 -> 800×450` (`0.625×`), **73.1%** (`94,673 / 129,591`) of training boxes have pixel area $< 32^2$, and **103,052 / 185,526 (55.5%)** of validation boxes have $\min(w, h) < 16\text{ px}$!
   - Default `fasterrcnn_resnet50_fpn_v2` uses 1 scale per FPN level `((32,), (64,), (128,), (256,), (512,))`. A typical `9×14 px` traffic light (`126 px²`) has maximum theoretical IoU of **$126 / 1024 = 0.123$** against any P2 `32×32` anchor — far below RPN's `fg_iou_thresh = 0.70`!
   - **Verified Solution (`--anchor-profile road_small`)**: Multi-scale FPN anchors `((12, 20), (28, 44), (60, 92), (124, 188), (252, 384))` raise P2 max anchor IoU on a `10×14 px` traffic light by **$>3.5\times$ (`0.13 -> >0.55`)** across *all* methods fairly.
2. **Conformal Threshold Starvation on Small/Sparse Classes (`F-05` — ~20% of SSOD small-object gap)**:
   - Small objects (`traffic light`, `traffic sign`, `pedestrian`, `rider`) rarely achieve teacher softmax confidence $> 0.88$. In `v2/v2.1`, `tau_high = 0.87–0.90` and per-class noise inflation excluded most true small objects from Zone 1. Fixed in `v2.2`.
3. **Unprotected Random Cutout Erasure (`F-07` — ~10% of SSOD small-object noise)**:
   - `80×45 px` Cutout patches previously overwrote entire `8×12 px` small boxes while keeping positive labels. Fixed in `v2.2` (`<= 35%` occlusion guard).
4. **Extreme Long-Tail Class Imbalance in 10-Class Macro mAP (~10%)**:
   - `car` has `72,303` labeled boxes while `train` has `17`, `motorcycle` `356`, `rider` `417`, and `bicycle` `715`, yet each class carries an equal **10% weight** in macro COCO mAP (and small-object subsets of `rider`/`bicycle`/`pedestrian` dominate tail errors).

---

## 5. Data Pipeline, Annotation Version & Reproducibility Audit (Sections 7.6 & 10)

1. **Dataset & Annotation Protocol (`legacy_2018_converted` vs Detection 2020)**:
   - Verified zero labeled/unlabeled overlap (`7,000` + `63,000` = `70,000`), zero train/val leakage (`0` overlapping filenames with `10,000` val images), zero duplicate filenames, and `0` empty-GT labeled images.
   - Both the local workstation (`D:\project\Abrishami\Dataset\converted`) and A100 server (`/data/datasets/BDD100K/legacy_2018_converted`) use the 10-category BDD100K 2D box JSON schema (`pedestrian`, `rider`, `car`, `truck`, `bus`, `train`, `motorcycle`, `bicycle`, `traffic light`, `traffic sign`).
   - **Recommendation on BDD100K 2018 vs 2020**: BDD100K Detection 2020 (`det_20`) refined ~4% of noisy 2018 bounding boxes and adjusted `rider`/`pedestrian` boundary consistency. **Never mix 2018 and 2020 numbers in the same table**. Because `prepare_bdd100k.py` in `v2.2` now supports both schemas (mapping 2020 Scalabel `"person"`/`"bike"`/`"motor"` aliases cleanly while preserving exact SHA-256 hashes on 2018 JSONs), you may run the diagnostic matrix on the existing 100k split immediately, and if `det_20` labels (`det_train.json`, `det_val.json`) are available on `/data/datasets/BDD100K/`, run the final 3-seed Q1 paper suite on **official BDD100K Detection 2020** with explicit version disclosure (`BDD100K det_20, 70k/10k split`).
2. **Resume & DAG Invalidation Integrity (`Section 10`)**:
   - On A100 restart after Soft Teacher's crash, `Dependency-invalidated stages` fired because Session 1 recorded the pre-prepare fallback `split_fingerprint` on `warmup`, `supervised`, and `mean_teacher`, whereas Session 2 computed `split_fingerprint` from `split_manifest.json`.
   - Are the existing A100 checkpoints (`warmup`, `supervised`, `mean_teacher` in `/data/outputs/CORE_SSOD_V21_A100_BS32_FRESH`) scientifically consistent with one another? **YES**: they all used the exact same `splits/10pct_s42` files, the exact same `warmup/last.pth` (`calibration_fold_sha256` verified by `native_training.py:1070`), and the exact same 10,000-image `val.json`. However, `Soft Teacher` in that active directory is running with the `rpn.training=False` (1,000-proposal) hotfix and without `M / B` active-batch scaling. Therefore, let `CORE_SSOD_V21_A100_BS32_FRESH` finish untouched as the **archived `v2.1` baseline record**, and run `v2.2` in a separate worktree and output directory!
