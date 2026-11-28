# CoRe-SSOD v2.2 Scientific Improvement Plan & Experimental Roadmap

**Branch**: `feat/core-ssod-v2.2-audit`  
**Target Venues**: *IEEE Transactions on Intelligent Transportation Systems (T-ITS)*, *IEEE Transactions on Intelligent Vehicles (T-IV)*, *Pattern Recognition*, *Neural Networks*.

---

## 1. Design Principles for CoRe-SSOD v2.2

Every change in **CoRe-SSOD v2.2** obeys four strict scientific constraints:
1. **Zero baseline weakening**: Strong baselines (`Soft Teacher`, `Consistent-Teacher`, `Mean Teacher`, `Supervised`) receive the exact same detector, normalization (`FrozenBN` + `GroupNorm`), RPN anchor profile, resolution, optimizer, LR schedule, warm-start checkpoint, and `M / B` micro-batch loss normalization. In fact, `v2.2` *strengthens* `Soft Teacher` by fixing its regression-only RPN crash without reducing its proposal budget from 2,000 to 1,000.
2. **Mathematical honesty in Conformal Risk Control**: Finite-sample weighted bounds use the exact worst-case weight formula $U_{\text{WCRC}}(\lambda) = \frac{\sum_{i=1}^n w_i L_i(\lambda) + w_{\max}}{\sum_{i=1}^n w_i + w_{\max}}$, calibration images are strictly excluded from step 0 (`--reserve-calibration`), and sequential EMA recalibration at $t > 0$ is explicitly framed as *online approximate conformal risk control* rather than overclaiming unconditional multi-step exchangeability.
3. **Elimination of sparse-scene & small-object threshold inversion**: Nighttime highway scenes and sparse/small classes (`pedestrian`, `rider`, `bicycle`, `motorcycle`, `traffic light`, `traffic sign`) are no longer penalized $9\times\text{–}18\times$ per false positive relative to `car`.
4. **Cryptographic pipeline reproducibility**: Every stage transition verifies split annotation SHA-256s, `init_checkpoint_sha256`, `calibration_fold_sha256`, and `eval_max_images`, with zero false invalidations across restarts.

---

## 2. Summary of Verified Corrections Implemented in `v2.2`

### 2.1 Exact Proposal-Only RPN Extraction for Task-Decoupled SSOD (`ssod_research/baselines.py`)
- Implemented `extract_rpn_proposals_only(rpn, images, features)`, invoking `rpn.head`, `rpn.anchor_generator`, `rpn.box_coder.decode`, and `rpn.filter_proposals` under `@torch.no_grad()` **without toggling `rpn.training = False`**.
- Preserves the full **2,000 pre-NMS / 2,000 post-NMS** training proposal distribution on regression-only images (`cls_valid = False`, `reg_valid = True`) while maintaining gradient flow through student FPN + backbone + RoI box regression heads.

### 2.2 Micro-Batch-Invariant Unsupervised Loss Normalization (`M / B` Scaling)
- In `v2.1`, excluding $B - M$ zero-pseudo-box images from a micro-batch of size $B$ averaged RPN/ROI losses over $M$ images instead of $B$, amplifying gradients by $B / M$ (up to $32\times$ at `batch_size = 32`) and breaking micro-batch equivalence (`32×1 != 2×16`).
- In `v2.2`, active-subset losses in `care_unsupervised_loss`, `soft_teacher_unsupervised_loss`, `consistent_teacher`, and `mean_teacher` are scaled by $M / B$, restoring **exact algebraic gradient invariance** across any `batch_size × grad_accum_steps = E` factorization.

### 2.3 Density-Stabilized Weighted Conformal Risk Control (`ssod_research/conformal.py`)
- **Exact Covariate-Shift Bound**: Replaced heuristic Kish $1/(n_{\text{eff}}+1)$ with exact worst-case likelihood-ratio bound $(\sum w_i L_i(\lambda) + w_{\max}) / (\sum w_i + w_{\max})$ with $w_i \in [0.4, 2.5]$.
- **Regime-Smoothed Monotone Loss**: Normalized per-image false positives by $d_{i,r} = \max(1, 0.5|G_i| + 0.5\bar{g}_r)$, which remains strictly constant in $\lambda$ (preserving exact monotonicity $L_i(\lambda_1) \ge L_i(\lambda_2)$ for $\lambda_1 < \lambda_2$ and boundedness in $[0, 1]$) while preventing 1-car night highway scenes from forcing $\hat{\lambda}_{\text{degraded}} = 0.90 > \hat{\lambda}_{\text{clear}} = 0.85$.
- **Non-Inverted Per-Class Shrinkage & Zone-2 Ceiling**: Restricted per-class calibration to images with GT or candidate reliability $q \ge \tau_{\min\_\text{low}}$ (`0.25`), bounded shrunk class thresholds to $[\hat{\lambda}_r - 0.12, \hat{\lambda}_r + 0.05]$, and capped $\tau_{\text{low}} \le 0.55$ so genuine ambiguous night/small objects (`[0.35, 0.55+]`) are protected in Zone 2 instead of crushed as Zone 3 hard background.
- **Stratified Calibration Fold (`--stratify-calibration`)**: Proportional Hamilton-rounded stratified sampling across `timeofday × weather × rare_class` ensuring every singleton rare stratum remains in supervised training while $D_{\text{cal}}$ mirrors true night/dawn/rain/snow/fog proportions.

### 2.4 Small-Object Recall & Occlusion Remedies (`care_ssod.py`, `native_training.py`)
- **Occlusion-Safe Cutout**: `strong_view_care` verifies that random Cutout patches occlude at most `35%` of any Zone-1 or Zone-2 bounding box, preventing tiny `8×12 px` traffic lights/signs from being completely overwritten by uniform gray patches while retaining positive targets.
- **Shared Multi-Scale Sub-32px Anchor Profile (`--anchor-profile road_small`)**: Optional fair detector configuration replacing single-scale `((32,), (64,), (128,), (256,), (512,))` anchors with 2-per-octave `((12, 20), (28, 44), (60, 92), (124, 188), (252, 384))` across **all** variants, increasing P2 anchor IoU on `10×14 px` BDD100K traffic lights from `0.12` to `>0.55`.

---

## 3. Prioritized Experimental Roadmap & Compute Estimates (A100-SXM4 80GB, BF16)

Measured A100-SXM4 80GB throughput in BF16 at `max_side = 800`:
- Supervised step (`eff_batch = 8`): `~0.16 s/step` (`1,500` warmup steps $\approx$ **4 min**; `3,000` steps $\approx$ **8 min**).
- SSOD step (`eff_batch = 8`): `~0.42 s/step` (`3,000` steps $\approx$ **21 min**; `12,000` steps $\approx$ **84 min**).
- Full 10,000-image COCO validation (`batch_size = 32`, BF16, batched JSONL): **~1.5 min** per evaluation.

### Stage 0 — Let Current Active Run Finish Untouched (`0 extra GPU-hours` beyond current job)
- Allow `/data/outputs/CORE_SSOD_V21_A100_BS32_FRESH` to finish naturally so you have a complete, unmodified `v2.1` (`eff_batch = 32, lr = 0.01`) full-10k-validation record on disk.
- Meanwhile, create an isolated git worktree `/data/projects/SSOD-A100-v2_2` on `feat/core-ssod-v2.2-audit`.

### Stage 1 — Controlled 9-Run Diagnostic Matrix (`~3.8 A100 GPU-hours` at 1,000-image triage, then full 10k eval on winners)
Run `MODE=diagnostics bash RUN_Q1_A100_V22.sh` (`scripts/run_v22_diagnostic_matrix.py`):

| ID | Experiment | Hypothesis Tested | Est. A100 Time |
|---|---|---|---:|
| `E1` | Shared Warmup (`--reserve-calibration --stratify-calibration`, `eb=8, lr=0.01, 1500 steps`) | Clean leak-free initialization for `E3–E6, E9` | 6 min |
| `E2` | Unreserved Warmup (`7,000` imgs, `eb=8, lr=0.01, 1500 steps`) | Quantify exact AP impact of holding out 256 calibration images (`3.66%` data) | 6 min |
| `E3` | Supervised (`eb=8, lr=0.01, 3000 steps`, from `E1`) | Matched supervised baseline under documented reference optimization | 11 min |
| `E4` | **CoRe-SSOD v2.2** (`eb=8, lr=0.01, alpha=0.10`, from `E1`) | Core v2.2 gain over matched supervised (`E4 - E3`) with fixed WCRC + `M/B` scaling | 24 min |
| `E5` | **CoRe-SSOD v2.2** (`alpha=0.08`, from `E1`) | Risk/coverage sensitivity (`alpha=0.08` vs `0.10`) | 24 min |
| `E6` | **CoRe-SSOD v2.2** (`--zero-box-filter 0`, from `E1`) | Isolated contribution of zero-pseudo-box RPN/ROI filtering | 24 min |
| `E7` | **CoRe-SSOD v2.2** (`eb=32, lr=0.02, 3000 steps`) | Linear/sqrt LR scaling (`0.01 -> 0.02`) for `eff_batch=32` vs `CORE_SSOD_V21_A100_BS32_FRESH` | 48 min |
| `E8` | **Small-Object Diagnostic** (`--anchor-profile road_small --max-side 1024`, `eb=8`) | Quantify `AP_small` jump (`traffic light`, `traffic sign`, `pedestrian`) from sub-32px anchors + `1024px` | 34 min |
| `E9` | **Extended Horizon** (`eb=8, 12000 steps, ema=0.9985`, from `E1`) | Verify whether 3,000 steps under-converges vs 12,000-step paper horizon | 88 min |

### Stage 2 — Head-to-Head Version Reproduction (`v2` vs `v2.1` vs `v2.2` on Shared 10k Split, `~4.5 A100 GPU-hours`)
- Run `MODE=compare_versions bash RUN_Q1_A100_V22.sh` (`eff_batch = 8, lr = 0.01`, full 10,000-image validation) across detached worktrees of `4f8ce52` (`v2`, evaluated on full 10k), `0252a64` (`v2.1`), and `HEAD` (`v2.2`) to lock down an airtight version progression table.

### Stage 3 — Final Multi-Seed Q1 Journal Suite (`~52 A100 GPU-hours` = ~2.2 days on 1× A100-80GB)
- Only **after** Stage 1 & Stage 2 confirm statistically clean gains (`CoRe-SSOD v2.2` > `Supervised`, `Mean Teacher`, `Soft Teacher`, `Consistent-Teacher` on full 10,000-image validation, including `AP_small` and `Night AP`):
  - Promote the winning shared anchor/resolution setting (e.g., `eb=16, lr=0.015` or `eb=8, lr=0.01`, `12,000` steps, applied **identically** to all 5 methods and 6 ablations!) to `MODE=paper bash RUN_Q1_A100_V22.sh` (`5%` & `10%` × seeds `42, 123, 2026` + seed-42 ablations = 48 runs).

---

## 4. Final Architectural Recommendation (Section 14.10)

**Recommendation: Adopt `CoRe-SSOD v2.2` (`feat/core-ssod-v2.2-audit`) — do NOT revert to raw `v2` (`4f8ce52`) and do NOT ship `v2.1` (`0252a64`) as final.**

- **Why not revert to `v2` (`4f8ce52`)?**
  1. `v2` trained `warmup` on the 256 calibration images, violating Conformal Risk Control exchangeability (`D_cal` was in-sample for the burn-in teacher).
  2. `v2` passed dark night images with 0 pseudo-boxes as 100% hard RPN/ROI negatives in all baselines and `core_ssod`.
  3. `v2`'s historical `18.15%` figure was measured on a 500-image partial subset (`partial_validation=True`), not full 10,000-image validation, and still suffered from sparse-scene night threshold inflation (`tau_high=0.90`, `tau_low=0.73`).
- **Why not keep `v2.1` (`0252a64`)?**
  1. `v2.1` crashes deterministically in `Soft Teacher` on regression-only images (`F-01`), and the `rpn.training=False` hotfix halves RPN proposals (`2000 -> 1000`).
  2. `v2.1` omits `M / B` active-batch loss scaling (`F-03`), breaking micro-batch equivalence and spiking gradients on sparse batches at `batch_size = 32`.
  3. `v2.1` raised `max_high` to `0.95` without fixing single-image $|G_i|$ normalization (`F-05`) or the Kish vs $w_{\max}$ WCRC formula (`F-04`).
  4. `v2.1` emits false `split_fingerprint` invalidation warnings on restart while allowing `job_complete` to skip runs without checking `init_checkpoint_sha256` or `eval_max_images` (`F-08`).
- **Why `v2.2` is the scientifically sound path forward**:
  `v2.2` preserves `v2.1`'s strict calibration reservation (`--reserve-calibration`) and zero-box false-negative protection while fixing **all** verified mathematical, gradient-normalization, RPN proposal-policy, small-object cutout/anchor, and DAG reproducibility defects, backed by comprehensive automated regression tests.
