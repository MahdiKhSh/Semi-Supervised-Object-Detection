# Codex execution handoff: first GPU milestone

This is a *research prototype*, not a certified reproduction or a finished
journal article. Start from branch `feat/research-baseline`.

## Required first steps (do not skip)
1. Run `python -m pytest -q` and `python scripts/check_environment.py`.
2. Record the exact NVIDIA driver, Python, torch CUDA build, MMCV, MMEngine,
   and MMDetection commit. The code assumes the current MMDet 3.x APIs.
3. Review package incompatibilities and fix with minimal changes; prefer
   the official OpenMMLab code rather than rewriting SoftTeacher.
4. Review the configs using `mmengine.Config.fromfile`, then build dataset,
   sampler, detector, and optimizer before training.
5. Prepare a miniature **synthetic COCO dataset** and run a CPU/GPU forward,
   backward, and checkpoint/resume smoke test.
6. Confirm custom `ReliabilitySoftTeacher` is registered and used only
   by the proposed-method configuration. Compare nonzero pseudo-label
   distributions against original SoftTeacher.
7. Use real BDD100K files only after explicit validation of license, split
   integrity, image paths and annotation taxonomy.
8. Save a reproducibility report in `docs/IMPLEMENTATION_REPORT.md` with
   command outputs, exact version pins, test logs and unresolved issues.
9. Build and run the supervised-only baseline; then SoftTeacher baseline;
   then prototype reliability selector. Use matching seeds and compute.
10. Before claiming a new multi-level consistency loss, implement a
    separate differentiable loss and gradient tests, add required assignment
    and valid-visibility masks, and show it changes optimization. Current
    code does **not** implement that complete loss.

## Restrictions
- Do not run multi-hour training without first giving the full exact command.
- Do not replace or hide unsuccessful experiments.
- Do not change networking, proxies or RDP routes.
- Do not claim equivalence of COCO and BDD protocols.
- Never commit BDD images, customer data, API keys or large checkpoints.
- Use a separate PR for subsequent experimental changes.
