#!/usr/bin/env bash
# Crash-safe Linux NVIDIA A100 (40GB / 80GB) launcher for CoRe-SSOD Q1 experiments.
# Supports both:
#   PRESET=q1_a100  -> 10% seed-42 full 10,000-image validation reference suite (6 runs)
#   PRESET=q1_paper -> Full Q1 paper matrix (5% & 10% x 3 seeds + 6 ablations, 48 runs)
#
# Re-running the exact same command safely resumes from the last verified checkpoint!

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

VENV_ACTIVATE="${VENV_ACTIVATE:-/data/venvs/ssod/bin/activate}"
if [[ -f "${VENV_ACTIVATE}" ]]; then
    # shellcheck disable=SC1090
    source "${VENV_ACTIVATE}"
elif [[ -f "${SCRIPT_DIR}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/.venv/bin/activate"
fi

PRESET="${PRESET:-q1_paper}"
AMP="${AMP:-bf16}"
# Default effective batch = 8 (8x1 on A100 40GB/80GB, ~14 GiB VRAM).
# For high-throughput effective batch = 32 on A100 40GB/80GB (~26 GiB VRAM), set BATCH_SIZE=16 GRAD_ACCUM=2.
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/work_dirs/${PRESET}_runs}"

DATASET_ARGS=()
if [[ -n "${DATASET_PLAN:-}" && -f "${DATASET_PLAN}" ]]; then
    DATASET_ARGS+=(--dataset-from-plan "${DATASET_PLAN}")
elif [[ -f "/data/outputs/q1_pilot_legacy_bs40/experiment_plan.json" && -z "${TRAIN_LABELS:-}" ]]; then
    DATASET_ARGS+=(--dataset-from-plan "/data/outputs/q1_pilot_legacy_bs40/experiment_plan.json")
else
    [[ -n "${TRAIN_LABELS:-}" ]] && DATASET_ARGS+=(--train-labels "${TRAIN_LABELS}")
    [[ -n "${VAL_LABELS:-}" ]]   && DATASET_ARGS+=(--val-labels "${VAL_LABELS}")
    [[ -n "${TRAIN_IMAGES:-}" ]] && DATASET_ARGS+=(--train-images "${TRAIN_IMAGES}")
    [[ -n "${VAL_IMAGES:-}" ]]   && DATASET_ARGS+=(--val-images "${VAL_IMAGES}")
fi

echo "=================================================================="
echo " Launching CoRe-SSOD v2.1 on NVIDIA A100"
echo " Preset     : ${PRESET}"
echo " Precision  : ${AMP}"
echo " Batch      : ${BATCH_SIZE} x ${GRAD_ACCUM} (effective=$((BATCH_SIZE * GRAD_ACCUM)))"
echo " Output Dir : ${OUTPUT_DIR}"
echo "=================================================================="

python -u scripts/run_suite.py \
    --preset "${PRESET}" \
    --device cuda \
    --amp "${AMP}" \
    --batch-size "${BATCH_SIZE}" \
    --grad-accum-steps "${GRAD_ACCUM}" \
    --output "${OUTPUT_DIR}" \
    --resume \
    "${DATASET_ARGS[@]}" \
    "$@"
