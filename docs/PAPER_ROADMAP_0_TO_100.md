# Complete research roadmap (0–100)

Scope: Semi-supervised **2D RGB** object detection for autonomous driving,
not LiDAR 3D detection or sensor fusion. Our candidate approach combines
EMA, view agreement, localization reliability, class-/condition-aware
thresholds, and matched-ROI box/class/confidence consistency.

No publication acceptance or claimed novelty is guaranteed by this plan.

## 00–10: Pre-register the scientific questions

- RQ1: Does reliability-aware pseudo-target filtering beat a matched fixed
  threshold EMA teacher at 5% and 10% labeled data?
- RQ2: Which component (localization, class frequency, scene condition,
  consistency box, class, confidence) actually helps?
- RQ3: How stable are gains under rainy/night conditions and rare classes?
- RQ4: Is the method robust to threshold and random seed changes?

Literature: Soft Teacher ICCV 2021, Consistent-Teacher CVPR 2023,
Sparse Semi-DETR CVPR 2024, Building Blocks TMLR 2025,
STEP-DETR ICCV 2025, SSRT-DETR Sensors 2026. These already contain
important overlap; compare equations and experimental protocols exactly.

A claim about *joint condition/reliability* requires a distinct algorithm
and evidence, not the mere coexistence of previously known modules.

## 10–20: Data validity and splitting

1. Obtain official BDD100K JPEGs and 2D box label JSON; retain source files.
2. Verify all 10 class IDs, images, box coordinates, ignored annotations,
   empty images, duplicate filenames, and source hashes.
3. Create 5% and 10% labeled fractions of training only, with three seeds
   (42, 123, 2026). Use the remainder as unlabeled.
4. Validation annotations must never influence class-frequency thresholds,
   pseudo-label training or loss hyperparameter data.
5. **Temporal leakage risk:** check BDD source video or drive IDs. Current
   conversion only ensures filename-disjoint images; group by drive when
   source sequence IDs are available. Document any remaining correlation.
6. Preserve scene/time/weather metadata of unlabeled images separately
   from object labels; identify it as weak auxiliary metadata.
7. Produce class counts and condition counts for every split.

## 20–30: Native Windows readiness and smoke tests

1. Run RUN_SMOKE.cmd on synthetic data; never report synthetic AP.
2. Check torch CUDA 12.8 build, RTX 5090 compute capability, VRAM.
3. Verify supervised gradients, teacher update, paired horizontal flip,
   pseudo-target validity, stable matched RoI, empty pseudo batches,
   three-term loss gradients, full COCO evaluator and checkpoint resume.
4. Confirm run logs record exact torch/torchvision versions, data hashes,
   settings, elapsed time and peak GPU reserved memory.
5. Avoid installing WSL or changing RDP/network routes.
6. Fail closed when image paths or annotation JSON are missing.

## 30–50: Baseline protocol

For each labeled fraction and seed, train one supervised **warmup** model
from ImageNet backbone weights (the detector head starts randomly).
All variants then start with identical warmup weights and optimization
schedule for fair comparison; clearly count warmup compute.

B0 supervised-only continuation.
B1 Mean Teacher: EMA teacher with fixed pseudo-score threshold.
B2 B1 + agreement-based localization uncertainty gate.
B3 B2 + class-frequency-adaptive thresholds.
B4 B2 + scene-condition thresholds.
B5 B2 + B3 + B4 + differentiable consistency loss on matched ROI box,
class and confidence predictions.

Important: B2–B5 add a second teacher forward pass and B5 adds an
additional student forward pass. Iteration budget is matched, FLOPs and
runtime are **not**; report peak memory and wall-clock.

The default unsupervised loss does not use the original published
SoftTeacher soft-negative classification weighting or jitter mechanism.
Call B1 a native **Mean Teacher**, not a SoftTeacher reproduction.

## 50–70: Ablations and sensitivity

- Fixed confidence thresholds: 0.50 / 0.70 / 0.90.
- Localization agreement gating: enabled vs disabled (remove second-view localization filtering).
- Class adaptation: strength 0 vs 0.12.
- Condition offset: 0 vs +0.05.
- Consistency class term on/off.
- Consistency box term on/off.
- Consistency confidence term on/off.
- Entire added consistency loss on/off.
- Full method vs individual and additive components.
- Examine unmatched teacher boxes, zero-pseudo-label steps, false
  positives of rare classes, small objects and nighttime scenes.

All ablation variants and exact commands are in experiment_plan.json.
Keep a record of failed runs, not only successful ones.

## 70–85: Evaluation and statistics

Primary: COCO AP@[.50:.95] on entire BDD validation, secondary AP50,
AP75 and per-class AP. Derive truly stratified AP by BDD weather/time/scene
using original metadata and per-image detections, with subgroup sample
counts. Report speed, wall time, memory, pseudo-label acceptance ratios,
and sample seed variance.

Report mean ± sample standard deviation for 3 paired seeds;
paired differences B5−B0 and B5−B1 at each label ratio;
bootstrap intervals on paired seed differences, noting low power with
three seeds. Do not infer significance from non-overlapping single-run
boxes; evaluate multiple comparisons transparently.

Do not use validation as hidden test. If an authorized test server exists,
freeze the model/threshold selection first and evaluate once.

## 85–95: Write the paper

1. Abstract: question, method, measured effects only.
2. Introduction: road detection robustness and semi-supervised motivation.
3. Related work: target selection, localization uncertainty, adaptive
   assignment, consistency, long-tail and domain adaptation.
4. Method: pseudo boxes, inverse view projection, matching, condition
   gating, loss and stop-gradients, background handling, edge cases.
5. Experiment setup: hardware, Windows CUDA, exact code commit and
   dataset hashes, train schedule, seed, comparison fairness.
6. Results: B0–B5 and table of methods, ablations and thresholds.
7. Robustness: weather/time/rare-class/per-size AP and confidence.
8. Compute efficiency: training seconds/step, VRAM, pseudo matching costs.
9. Visualization: genuine detections, false positives/negatives,
   failure cases, no fabricated images.
10. Limitations, ethics, reproducibility, conclusions.

Suggested plots: pseudo acceptance over training, loss per stage,
AP by class, AP by condition with sample size, paired-seed deltas,
threshold response, runtime vs AP Pareto.

## 95–100: Final review gates

- Fresh Windows system installs and runs smoke, real-data 2-step
  train, evaluation, checkpoint resume.
- Source, packages and BDD raw/split hashes recorded.
- Full validation evaluation with no tuning leakage.
- Every ablation and its seed matched to same warmup and data.
- Failures and negative results retained, not omitted.
- Novelty checked critically against Consistent-Teacher and SSRT-DETR.
- Draft read by independent reviewer before journal submission.

## Resource-aware fallback

Full paper preset is 102 stages and may exceed a 14-day timeline
on a single RTX 5090. Run pilot first to measure true seconds/step.
If time is inadequate, pre-register a smaller core design (5% and 10%,
at least 3 seeds, B0/B1/B2/B5, threshold and key loss ablations)
before seeing final AP. Never cherry-pick the fastest or strongest
completed runs as the claimed full matrix.
