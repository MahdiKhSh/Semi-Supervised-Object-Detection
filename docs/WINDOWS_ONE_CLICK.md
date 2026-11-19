# Native Windows RTX 5090: one-click experiments

**No Linux, WSL, MMCV, network routing changes or RDP disruption required by the primary execution path.**

## Prerequisites

Windows x64, NVIDIA RTX 5090 with a compatible driver, Python 3.11 recommended
(Python 3.10 fallback), disk space for BDD100K and model checkpoints.
Double-click scripts create and verify a local Python virtual environment.
They use official PyTorch 2.10.0 / TorchVision 0.25.0 CUDA 12.8 wheels:
https://pytorch.org/get-started/previous-versions/

MMDetection/SoftTeacher remains an optional *separate* baseline, not the
primary Windows execution path. The native TorchVision variants are not
numerically equivalent to the original SoftTeacher ICCV implementation.

## Place BDD100K data

Download the official **image detection** dataset yourself and accept the
dataset license. The scripts cannot redistribute the data or guess paths.

~~~text
data/bdd100k/raw/labels_train.json
data/bdd100k/raw/labels_val.json
data/bdd100k/images/train/    # JPEG images from official train
data/bdd100k/images/val/      # JPEG images from official val
~~~

The official label JSON may have other filenames. Rename a local copy or
provide explicit CLI --train-labels and --val-labels arguments. The images
are never copied into each experimental split. The converter preserves
object-free scene metadata separately from unlabeled detection boxes.

Inspect required paths from PowerShell:

~~~powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\prepare_data_windows.ps1
~~~

## Double-click launchers

| File | What it does | Important |
|---|---|---|
| RUN_SMOKE.cmd | Synthetic data, supervised warmup, EMA baseline, full variant | Not paper data |
| RUN_PILOT.cmd | Real 10% labeled, one seed, main variants + thresholds + ablations | Partial validation only; diagnostic |
| RUN_CORE.cmd | 5%/10%, three seeds, key baselines and one-seed ablations | Compute-aware core paper matrix (52 stages) |
| RUN_ALL.cmd | Real 5%/10% labeled, three seeds, all components and thresholds | Exhaustive 102-stage study, significant GPU time |
| RUN_VISUALIZE.cmd | Real-image overlays from trained CORE checkpoint | Qualitative evidence, not an AP metric |

Each launcher prepares the isolated virtual environment, runs tests,
verifies the CUDA compute capability, then runs its matrix sequentially.
The launcher never kills unrelated GPU processes. On failure, inspect
work_dirs/<preset>/logs and double-click again. Completed stages are
skipped, interrupted stages resume from saved checkpoint when possible.

For commands, output locations and the full protocol in Persian see [ROADMAP_FA.md](ROADMAP_FA.md).

**Important compute warning:** the full paper matrix includes 102 stages
(6 fraction/seed combinations × [preparation + supervised warmup +
15 continued experiments]). At 8,000 continuation iterations per
experiment, a single 5090 may require many days or weeks. Run SMOKE and
PILOT first; measure seconds per iteration, memory and AP.

## Manual commands in Windows PowerShell

~~~powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
.\.venv\Scripts\python.exe scripts\run_suite.py --preset smoke
.\.venv\Scripts\python.exe scripts\run_suite.py --preset pilot
.\.venv\Scripts\python.exe scripts\run_suite.py --preset paper --dry-run
~~~

To use D: data without copying images:

~~~powershell
.\.venv\Scripts\python.exe scripts\run_suite.py --preset pilot --train-labels "D:\BDD\labels_train.json" --val-labels "D:\BDD\labels_val.json" --train-images "D:\BDD\images\train" --val-images "D:\BDD\images\val"
~~~

## Scientific limitations

- The native method has not yet been validated on the user's RTX 5090.
- The official BDD images and true metrics are unavailable in this chat.
- Pseudo labels generated from an insufficiently trained teacher may be
  sparse; log accepted and candidate counts, and examine warmup quality.
- Random image splits prevent literal filename overlap but can include
  temporally correlated frames. Audit drive/sequence IDs before publication.
- The weather/time attributes of unlabeled images are additional **scene
  metadata** used as weak supervision; disclose this in the manuscript.
- The PILOT subset AP is not full-validation paper AP.
- The target results remain hypotheses until completed runs, honest
  statistical comparisons and external novelty analysis.
