# Research design and 14-day execution plan

## Question
Can reliability-calibrated pseudo-target selection improve per-condition
BDD100K road-object detection under limited labels, while avoiding losses
in clean-condition AP?

## Protocol
1. Fix taxonomy, source hashes and split manifest; partition **training**
   images only into labeled and unlabeled sets. Run 5% and 10% ratios.
2. Ensure no same-frame leakage. The basic script enforces filename
   disjointness, but BDD images may be temporally correlated: upgrade to
   group/sequence-aware splitting when identifiers exist.
3. Keep validation labels only for evaluation and model selection.
   Reserve final test for final reporting if authorized labels/evaluator exist.
4. Compare supervised-only, original SoftTeacher and
   ReliabilitySoftTeacher under identical detector, initialization,
   data, augmentation, optimizer, schedule and seed.
5. Disable each experimental component in controlled ablations:
   localization gate, quality floor, per-class thresholds.
6. Evaluate COCO AP@[.50:.95], AP50, AP75, per-class AP, day/night
   condition subsets, class frequency, throughput, peak VRAM and wall time.
   The supplied converter does not yet preserve scene attributes in COCO:
   implement a condition manifest before claiming day/night comparisons.
7. Repeat with seeds 42, 123 and 2026 if compute permits; report
   mean/std and confidence intervals rather than a single lucky run.
8. Record all failed runs and changes transparently.

## Milestones
- Days 1–2: toolchain tests, dataset integrity, exact upstream revision.
- Days 3–4: supervised-only and SoftTeacher baseline.
- Days 5–7: reliability selector, unit/integration tests, smoke training.
- Days 8–10: 5/10% experiments and ablations.
- Days 11–12: class/condition breakdown, robustness diagnostics.
- Days 13–14: tables, figures, failure analysis, reproducibility appendix.

## Stop/go thresholds
- Do not proceed to long training until CUDA, MMDetection forward/backward,
  data pipeline, bbox homographies, empty pseudo-label batches,
  EMA update and checkpoint resume all work.
- Do not report an improvement unless both runs share a controlled protocol.
- **No training performance measurements have been generated yet.**

## Known implementation limits
- Optional extension now includes a differentiable ROI-level three-term
  loss, **not yet integration-tested on the full MMDetection training loop**.
  This is not proof of mathematical novelty or trained performance.
- COCO-style annotations alone do not fully encode BDD weather/time metadata.
- CUDA/C++ MMCV wheel availability for RTX 5090 has not been tested.
- Some MMDetection versions may require minor config adaptation.

## Warm-start fairness

A random detection head produces few useful initial pseudo-labels. The native
suite first trains a supervised warmup for each fraction/seed, then initializes
ALL comparison variants (including supervised-only continuation) from the
same warmup checkpoint. Results must state warmup + continuation compute
and never confuse ImageNet backbone initialization with a COCO pretrained
full detector. Threshold sweeps reuse the identical supervised warm start.
