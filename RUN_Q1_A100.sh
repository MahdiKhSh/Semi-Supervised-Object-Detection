#!/usr/bin/env bash
# Crash-safe Linux NVIDIA A100-SXM4 80GB launcher for CoRe-SSOD Q1 experiments.
# Focuses exclusively on the latest high-accuracy CoRe-SSOD suite (zero legacy steps).
#
# Presets:
#   PRESET=q1_a100  -> (Default) 10% labeled, seed 42, full 10,000-image validation
#                      across Supervised, Mean Teacher, Soft Teacher, Consistent-Teacher,
#                      and proposed CoRe-SSOD (6 stages total: warmup + 5 methods).
#   PRESET=q1_paper -> Full multi-seed Q1 journal matrix (5% & 10% x 3 seeds + ablations).
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

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

PRESET="${PRESET:-q1_a100}"
AMP="${AMP:-bf16}"
# Tuned for NVIDIA A100-SXM4 80GB:
# - MAX_SIDE=1024 gives +64% pixel area for tiny traffic lights/signs & distant riders (~28 GiB VRAM at BS=16).
# - ANCHOR_PROFILE=road_small (12..384 multi-scale anchors) raises small-object RPN recall >3x across all methods.
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_SIDE="${MAX_SIDE:-1024}"
LR="${LR:-0.015}"
ANCHOR_PROFILE="${ANCHOR_PROFILE:-road_small}"

if [[ "${PRESET}" == "q1_paper" ]]; then
    WARMUP_STEPS="${WARMUP_STEPS:-3000}"
    TRAIN_STEPS="${TRAIN_STEPS:-12000}"
    DEFAULT_OUT="/data/outputs/CORE_SSOD_Q1_PAPER"
else
    WARMUP_STEPS="${WARMUP_STEPS:-2000}"
    TRAIN_STEPS="${TRAIN_STEPS:-6000}"
    DEFAULT_OUT="/data/outputs/CORE_SSOD_Q1_A100_BEST"
fi

if [[ -d "/data/outputs" ]]; then
    OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUT}}"
else
    OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/work_dirs/${PRESET}_runs}"
fi

DATASET_ARGS=()
if [[ -n "${DATASET_PLAN:-}" && -f "${DATASET_PLAN}" ]]; then
    DATASET_ARGS+=(--dataset-from-plan "${DATASET_PLAN}")
elif [[ -n "${TRAIN_LABELS:-}" || -n "${VAL_LABELS:-}" || -n "${TRAIN_IMAGES:-}" || -n "${VAL_IMAGES:-}" ]]; then
    [[ -n "${TRAIN_LABELS:-}" ]] && DATASET_ARGS+=(--train-labels "${TRAIN_LABELS}")
    [[ -n "${VAL_LABELS:-}" ]]   && DATASET_ARGS+=(--val-labels "${VAL_LABELS}")
    [[ -n "${TRAIN_IMAGES:-}" ]] && DATASET_ARGS+=(--train-images "${TRAIN_IMAGES}")
    [[ -n "${VAL_IMAGES:-}" ]]   && DATASET_ARGS+=(--val-images "${VAL_IMAGES}")
elif [[ -f "/data/datasets/BDD100K/legacy_2018_converted/labels_train.json" && -d "/data/datasets/BDD100K/100k/train" ]]; then
    DATASET_ARGS+=(
        --train-labels "/data/datasets/BDD100K/legacy_2018_converted/labels_train.json"
        --val-labels   "/data/datasets/BDD100K/legacy_2018_converted/labels_val.json"
        --train-images "/data/datasets/BDD100K/100k/train"
        --val-images   "/data/datasets/BDD100K/100k/val"
    )
elif [[ -f "/data/outputs/CORE_SSOD_V21_A100_BS32_FRESH/experiment_plan.json" ]]; then
    DATASET_ARGS+=(--dataset-from-plan "/data/outputs/CORE_SSOD_V21_A100_BS32_FRESH/experiment_plan.json")
elif [[ -f "/data/outputs/q1_pilot_legacy_bs40/experiment_plan.json" ]]; then
    DATASET_ARGS+=(--dataset-from-plan "/data/outputs/q1_pilot_legacy_bs40/experiment_plan.json")
fi

echo "=================================================================="
echo " Launching Latest CoRe-SSOD High-Accuracy Suite on NVIDIA A100"
echo " Preset       : ${PRESET}"
echo " Precision    : ${AMP}"
echo " Resolution   : max_side=${MAX_SIDE} | anchors=${ANCHOR_PROFILE}"
echo " Batch & LR   : ${BATCH_SIZE} x ${GRAD_ACCUM} (effective=$((BATCH_SIZE * GRAD_ACCUM))) | lr=${LR}"
echo " Schedule     : warmup=${WARMUP_STEPS} steps -> main=${TRAIN_STEPS} steps (full 10k val)"
echo " Output Dir   : ${OUTPUT_DIR}"
echo "=================================================================="

python -u scripts/run_suite.py \
    --preset "${PRESET}" \
    --device cuda \
    --amp "${AMP}" \
    --batch-size "${BATCH_SIZE}" \
    --grad-accum-steps "${GRAD_ACCUM}" \
    --max-side "${MAX_SIDE}" \
    --lr "${LR}" \
    --warmup-steps "${WARMUP_STEPS}" \
    --train-steps "${TRAIN_STEPS}" \
    --anchor-profile "${ANCHOR_PROFILE}" \
    --stratify-calibration \
    --output "${OUTPUT_DIR}" \
    --resume \
    "${DATASET_ARGS[@]}" \
    "$@"
