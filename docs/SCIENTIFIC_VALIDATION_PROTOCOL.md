# Scientific Validation & Experimental Fairness Protocol (`CoRe-SSOD`)

**Document Scope**: Mandatory experimental, statistical, annotation, and reporting protocol for all single-seed diagnostic runs and multi-seed Q1 journal benchmarks on BDD100K and external adverse-weather datasets.

---

## 1. Strict Experimental Fairness Rules

Every comparison between `CoRe-SSOD` and baseline methods (`Supervised`, `Mean Teacher`, `Soft Teacher`, `Consistent-Teacher`) **must** hold the following invariants identical across all branches:

1. **Identical Detector & Normalization**:
   - `Faster R-CNN ResNet50-FPN-v2` with `FrozenBatchNorm2d` in the ResNet-50 body (53 layers) and `GroupNorm(32, C)` in FPN and RoI heads (12+ layers).
   - If `--anchor-profile road_small` or `--max-side 1024` is adopted after Stage-1 diagnostics, it **must** be applied to `warmup` and **every** baseline (`supervised`, `mean_teacher`, `soft_teacher`, `consistent_teacher`) — never exclusively to `core_ssod`.
2. **Identical Labeled Training Pool & Calibration Exclusion**:
   - When `--reserve-calibration` is active (default for `q1_a100` and `q1_paper`), the 256-image calibration fold $\mathcal{D}_{\text{cal}}$ (`calibration_fold_sha256`) is excluded from **all** methods starting from `warmup` step 0 (`6,744` supervised training images at 10%; `3,244` at 5%).
   - No method is allowed to train on more labeled images than another in the same table, and $\mathcal{D}_{\text{cal}}$ is never drawn from `val.json`.
3. **Identical Burn-In Checkpoint**:
   - All five main branches (`supervised`, `mean_teacher`, `soft_teacher`, `consistent_teacher`, `core_ssod`) and all seed-matched ablations initialize from the exact same `warmup/last.pth` (`init_checkpoint_sha256` verified cryptographically by `scripts/run_suite.py`).
4. **Zero Unlabeled Metadata Leakage**:
   - Neither `core_ssod`, `soft_teacher`, `consistent_teacher`, nor `mean_teacher` may access ground-truth `timeofday`, `weather`, or `scene` tags from `train_attributes.json` on unlabeled images. (`train_attributes.json` is restricted to the legacy oracle-attribute baseline `condition` and optional labeled-only `--stratify-calibration`).
5. **Honest Baseline Naming**:
   - Because all methods share two-stage Faster R-CNN ResNet50-FPN-v2 to isolate pseudo-label selection/weighting without detector confounding, report `consistent_teacher` in manuscript tables as **`Consistent-Teacher (GMM, Faster R-CNN)`** and document the shared `1:1` labeled:unlabeled batch ratio for **`Soft Teacher`**.

---

## 2. Theoretical Claims & Allowed Conformal Statements

To withstand rigorous statistical review at *Pattern Recognition*, *IEEE T-ITS*, or *Neural Networks*:

1. **What IS Mathematically Guaranteed**:
   - Conditional on the burn-in teacher $\theta_T^{(0)}$ trained with `--reserve-calibration`, $\mathcal{D}_{\text{cal}}$ has zero training overlap with $\theta_T^{(0)}$.
   - For each visibility regime $r \in \{\text{clear}, \text{degraded}\}$, the per-image loss
     $$L_{i,r}(\lambda) = \min\!\left(1,\; \frac{\sum_j \mathbb{1}[q_{ij} \ge \lambda \wedge y_{ij} = 0]}{\max(1,\; 0.5|G_i| + 0.5\bar{g}_r)}\right) \in [0, 1]$$
     is **deterministic in $\lambda$, bounded in $[0, 1]$, and monotonically non-increasing in $\lambda$** with zero suffix-max envelope inflation.
   - Under bounded likelihood-ratio weights $w_i \in [w_{\min}, w_{\max}]$ ($w_{\max} = 2.5$), `solve_conformal_threshold` selects the smallest grid threshold $\hat{\lambda}_r$ satisfying the exact worst-case weighted bound:
     $$U_{\text{WCRC}}(\lambda) = \frac{\sum_{i=1}^n w_i L_{i,r}(\lambda) + w_{\max}}{\sum_{i=1}^n w_i + w_{\max}} \le \alpha_{\text{strict}}.$$
2. **What Must Be Explicitly Qualified (Do NOT Overclaim)**:
   - **Density-ratio estimation gap**: Because $w_r(d)$ is estimated via a 6-bin photometric histogram rather than known oracle densities, finite-sample risk under covariate shift carries an additional total-variation term $\le \frac{1}{2}\|p_U(\cdot \mid r) - \hat{w}_r p_{\text{cal}}(\cdot \mid r)\|_1$ (Barber et al., *Annals of Statistics*, 2023; Tibshirani et al., *NeurIPS*, 2019).
   - **Sequential online recalibration ($t > 0$)**: After step 0, $\theta_T^{(t)}$ evolves via EMA on student updates influenced by prior pseudo-labels calibrated on $\mathcal{D}_{\text{cal}}$. Therefore, recalibrations at steps $300, 600, \dots$ constitute **online adaptive conformal risk control** and must be validated empirically via `conformal_calibration.jsonl` plots (`empirical_weighted_risk` and `pooled_empirical_fdr` vs step) rather than claimed as independent i.i.d. draws.

---

## 3. Validation, Subset & Annotation Hygiene

1. **Partial vs Full Validation Separation**:
   - Partial validation subsets (`--eval-max-images 500` in `q1_pilot` or `1000` in Stage-1 triage) carry `"partial_validation": true` in `evaluations.jsonl` and `results.csv`.
   - `scripts/report_results.py` segregates partial-validation rows into a diagnostic appendix and **never** merges them into main paper tables.
   - Never compare a `partial_validation=true` run (`500` images) against a `partial_validation=false` run (`10,000` images): rare classes (`train`: 1 box in `val_500` vs 15 in `val_10k`; `motorcycle`: 20 vs 452) cause systematic 1.2–1.8 mAP shifts between the 500-image prefix and full 10,000-image `val`.
2. **Annotation Version Discipline**:
   - Disclose the exact BDD100K annotation SHA-256 (`train_source_sha256`, `val_source_sha256` from `split_manifest.json`) and schema (`legacy_2018_converted` vs `det_20`). Never compare numbers across 2018 and 2020 ground-truth files.
3. **Teacher vs Student Checkpoint Evaluation**:
   - Standard SSOD evaluation (`native_training.py:1255`) evaluates the **EMA Teacher** (`teacher` weights) for all teacher-student methods and `student` weights for `supervised`.
   - For diagnostic parity checks (`scripts/eval_external.py`), run `--weights teacher` and `--weights student` into separate `--output` directories on the identical `val.json`.

---

## 4. Batch-Size vs Step-Budget Comparison Protocol

When comparing different effective batch sizes ($E = B \times G$, e.g., $E = 8$ vs $E = 32$), always report **both**:
1. **Fixed Optimizer-Step Comparison** (e.g., `3,000` optimizer updates at `E=8` vs `3,000` updates at `E=32` with linear/sqrt LR scaling $\eta_{32} \in \{0.015, 0.02\}$), measuring hardware wall-clock efficiency when data I/O + GPU compute saturates A100-80GB.
2. **Matched Image-Exposure Comparison** (`12,000` steps $\times E=8 = 96,000$ unlabeled views vs `3,000` steps $\times E=32 = 96,000$ unlabeled views, with matched EMA half-life in image exposures: $m_{32} = m_8^{4} \approx 0.9985^4 = 0.994$).

---

## 5. Pre-Submission Verification Checklist

Before declaring any benchmark table publication-ready:
- [ ] Full `pytest` suite (`72+` unit, regression, resume, fingerprint, and CUDA BF16 tests) passes with `0` failures.
- [ ] `partial_validation` is `False` (`images == 10000`) on every row in `results.csv`.
- [ ] All 3 paired seeds (`42`, `123`, `2026`) completed for both `5%` and `10%` splits across all 5 main methods (`30` main runs + `12` seed-42 ablations + `6` warmups = `48` runs).
- [ ] `manifest.json` records `"conformal_exchangeability_valid": true` and matching `calibration_fold_sha256` on every `core_ssod` run.
- [ ] Report mean $\pm$ sample standard deviation, paired seed-wise $\Delta\text{AP}$, per-class AP (all 10 classes), and condition-stratified AP (`daytime`, `night`, `dawn/dusk`, `rainy`, `snowy`, `foggy`) plus zero-shot external adverse-weather AP (`ACDC` / `DAWN`) if available.
