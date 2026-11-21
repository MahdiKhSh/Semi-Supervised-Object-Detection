#!/usr/bin/env bash
# Portable resumable SSOD runner for Kaggle (T4/P100), Google Colab, and Linux GPUs (L40s / 5090 / A100).
# Usage:
#   PRESET=q1_pilot BATCH_SIZE=2 GRAD_ACCUM_STEPS=4 SYNC_GITHUB=1 bash scripts/run_cloud.sh
#   PRESET=q1_paper BATCH_SIZE=4 GRAD_ACCUM_STEPS=2 SYNC_GITHUB=1 bash scripts/run_cloud.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PRESET="${PRESET:-q1_paper}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/work_dirs/${PRESET}_runs}"
SYNC_DIR="${SYNC_DIR:-}"
SYNC_GITHUB="${SYNC_GITHUB:-0}"
BENCHMARK="${BENCHMARK:-1}"
DEVICE="${DEVICE:-auto}"
AMP="${AMP:-auto}"
BATCH_SIZE="${BATCH_SIZE:-}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

echo "=== SSOD Portable Cloud/Linux Runner ==="
echo "Host       : $(hostname)"
echo "Preset     : $PRESET"
echo "Output Dir : $OUTPUT_DIR"
echo "Sync Dir   : ${SYNC_DIR:-<none>}"
echo "GitHub Sync: $SYNC_GITHUB"

# Install lightweight project dependencies if missing (uses existing PyTorch on Kaggle/Colab/L40s).
"$PYTHON_BIN" -m pip install -q --no-build-isolation -e ".[dev,native]" || "$PYTHON_BIN" -m pip install -q -e ".[dev,native]"

"$PYTHON_BIN" scripts/check_environment.py
"$PYTHON_BIN" scripts/sync_workspace.py env --output "$OUTPUT_DIR"

ARGS=(
  "scripts/run_suite.py"
  "--preset" "$PRESET"
  "--device" "$DEVICE"
  "--amp" "$AMP"
  "--resume"
  "--output" "$OUTPUT_DIR"
)

if [[ -n "$BATCH_SIZE" ]]; then ARGS+=("--batch-size" "$BATCH_SIZE"); fi
if [[ -n "$GRAD_ACCUM_STEPS" ]]; then ARGS+=("--grad-accum-steps" "$GRAD_ACCUM_STEPS"); fi
if [[ -n "${TRAIN_LABELS:-}" ]]; then ARGS+=("--train-labels" "$TRAIN_LABELS"); fi
if [[ -n "${VAL_LABELS:-}" ]]; then ARGS+=("--val-labels" "$VAL_LABELS"); fi
if [[ -n "${TRAIN_IMAGES:-}" ]]; then ARGS+=("--train-images" "$TRAIN_IMAGES"); fi
if [[ -n "${VAL_IMAGES:-}" ]]; then ARGS+=("--val-images" "$VAL_IMAGES"); fi
if [[ -n "$SYNC_DIR" ]]; then ARGS+=("--sync-dir" "$SYNC_DIR"); fi
if [[ "$SYNC_GITHUB" == "1" ]]; then ARGS+=("--sync-github"); fi
if [[ "$BENCHMARK" == "1" ]]; then ARGS+=("--benchmark"); fi

# shellcheck disable=SC2206
if [[ -n "$EXTRA_ARGS" ]]; then ARGS+=($EXTRA_ARGS); fi

"$PYTHON_BIN" "${ARGS[@]}"
