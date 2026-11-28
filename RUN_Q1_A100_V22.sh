#!/usr/bin/env bash
# Safe, Isolated-Worktree A100-SXM4 80GB Execution Pipeline for CoRe-SSOD v2.2 & Version Benchmarks.
#
# SAFETY GUARANTEES:
# 1. NEVER modifies /data/projects/SSOD-A100-v2_1 or /data/outputs/CORE_SSOD_V21_A100_BS32_FRESH.
# 2. Supports side-by-side reproducible git-worktree benchmarking of v2 (4f8ce52), v2.1 (0252a64),
#    and v2.2 (feat/core-ssod-v2.2-audit) on the exact same BDD100K split.
# 3. Crash-safe & idempotent: re-running resumes from the latest SHA-256 verified checkpoint.
#
# Usage examples on A100 server:
#   # A. Provision isolated worktree (leaves active v2.1 run 100% untouched!):
#   git -C /data/projects/SSOD-A100-v2_1 fetch origin feat/core-ssod-v2.2-audit
#   git -C /data/projects/SSOD-A100-v2_1 worktree add /data/projects/SSOD-A100-v2_2 origin/feat/core-ssod-v2.2-audit
#
#   # B. Launch CoRe-SSOD v2.2 10% full-10k-val reference suite (effective batch = 16, LR = 0.015):
#   cd /data/projects/SSOD-A100-v2_2
#   MODE=reference bash RUN_Q1_A100_V22.sh
#
#   # C. Launch controlled v2 vs v2.1 vs v2.2 head-to-head benchmark in isolated worktrees:
#   MODE=compare_versions bash RUN_Q1_A100_V22.sh
#
#   # D. Launch full multi-seed Q1 paper matrix (5% & 10% x 3 seeds + ablations):
#   MODE=paper bash RUN_Q1_A100_V22.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Guardrail: refuse to write into the active v2.1 output directory
ACTIVE_V21_OUT="/data/outputs/CORE_SSOD_V21_A100_BS32_FRESH"

VENV_ACTIVATE="${VENV_ACTIVATE:-/data/venvs/ssod/bin/activate}"
if [[ -f "${VENV_ACTIVATE}" ]]; then
    # shellcheck disable=SC1090
    source "${VENV_ACTIVATE}"
elif [[ -f "${SCRIPT_DIR}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/.venv/bin/activate"
fi

TRAIN_LABELS="${TRAIN_LABELS:-/data/datasets/BDD100K/legacy_2018_converted/labels_train.json}"
VAL_LABELS="${VAL_LABELS:-/data/datasets/BDD100K/legacy_2018_converted/labels_val.json}"
TRAIN_IMAGES="${TRAIN_IMAGES:-/data/datasets/BDD100K/100k/train}"
VAL_IMAGES="${VAL_IMAGES:-/data/datasets/BDD100K/100k/val}"

MODE="${MODE:-reference}"
AMP="${AMP:-bf16}"
# Recommended A100 80GB setting: batch_size=16 x grad_accum=1 (eff_batch=16, ~19 GiB VRAM) with LR=0.015,
# or documented protocol batch_size=8 x grad_accum=1 (eff_batch=8, ~12 GiB VRAM) with LR=0.01.
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-0.01}"
ANCHOR_PROFILE="${ANCHOR_PROFILE:-default}"

check_not_active_v21() {
    local target="$1"
    if [[ "$(realpath -m "${target}")" == "$(realpath -m "${ACTIVE_V21_OUT}")"* ]]; then
        echo "ERROR: Refusing to touch active v2.1 output directory ${ACTIVE_V21_OUT}" >&2
        exit 1
    fi
}

if [[ "${MODE}" == "compare_versions" ]]; then
    BENCH_ROOT="${OUTPUT_ROOT:-/data/outputs/CORE_SSOD_VERSION_BENCHMARK}"
    check_not_active_v21 "${BENCH_ROOT}"
    WT_ROOT="${WORKTREE_ROOT:-/data/projects/ssod_version_worktrees}"
    mkdir -p "${BENCH_ROOT}" "${WT_ROOT}"

    for spec in "v2:4f8ce523a0d83fa685534027e009d3e28909b158" "v2_1:0252a64a708faab46ed9c391da7ebbbf62ab2520" "v2_2:HEAD"; do
        ver="${spec%%:*}"
        rev="${spec##*:}"
        wt="${WT_ROOT}/${ver}"
        out="${BENCH_ROOT}/${ver}"
        if [[ ! -d "${wt}" ]]; then
            git worktree add --detach "${wt}" "${rev}"
        fi
        echo "=== Benchmarking ${ver} (${rev}) -> ${out} ==="
        if [[ "${ver}" == "v2" ]]; then
            # v2 predates q1_a100; run q1_pilot with full 10,000-image validation via native_training / suite
            (cd "${wt}" && python -u scripts/run_suite.py \
                --preset q1_pilot \
                --device cuda --amp "${AMP}" \
                --batch-size "${BATCH_SIZE}" --grad-accum-steps "${GRAD_ACCUM}" \
                --lr "${LR}" \
                --train-labels "${TRAIN_LABELS}" --val-labels "${VAL_LABELS}" \
                --train-images "${TRAIN_IMAGES}" --val-images "${VAL_IMAGES}" \
                --output "${out}" --resume "$@")
        elif [[ "${ver}" == "v2_1" ]]; then
            (cd "${wt}" && python -u scripts/run_suite.py \
                --preset q1_a100 \
                --device cuda --amp "${AMP}" \
                --batch-size "${BATCH_SIZE}" --grad-accum-steps "${GRAD_ACCUM}" \
                --lr "${LR}" \
                --train-labels "${TRAIN_LABELS}" --val-labels "${VAL_LABELS}" \
                --train-images "${TRAIN_IMAGES}" --val-images "${VAL_IMAGES}" \
                --output "${out}" --resume "$@")
        else
            (cd "${wt}" && python -u scripts/run_suite.py \
                --preset q1_a100 \
                --device cuda --amp "${AMP}" \
                --batch-size "${BATCH_SIZE}" --grad-accum-steps "${GRAD_ACCUM}" \
                --lr "${LR}" --anchor-profile "${ANCHOR_PROFILE}" --stratify-calibration \
                --train-labels "${TRAIN_LABELS}" --val-labels "${VAL_LABELS}" \
                --train-images "${TRAIN_IMAGES}" --val-images "${VAL_IMAGES}" \
                --output "${out}" --resume "$@")
        fi
    done
    exit 0
fi

if [[ "${MODE}" == "diagnostics" ]]; then
    OUT_DIR="${OUTPUT_DIR:-/data/outputs/CORE_SSOD_V22_DIAGNOSTICS}"
    check_not_active_v21 "${OUT_DIR}"
    SPLIT_DIR="${SPLIT_DIR:-${OUT_DIR}/splits/10pct_s42}"
    if [[ ! -f "${SPLIT_DIR}/split_manifest.json" ]]; then
        python -u scripts/prepare_bdd100k.py \
            --train-labels "${TRAIN_LABELS}" --val-labels "${VAL_LABELS}" \
            --train-images "${TRAIN_IMAGES}" --val-images "${VAL_IMAGES}" \
            --output "${SPLIT_DIR}" --fraction 0.10 --seed 42
    fi
    python -u scripts/run_v22_diagnostic_matrix.py \
        --split-dir "${SPLIT_DIR}" \
        --train-images "${TRAIN_IMAGES}" --val-images "${VAL_IMAGES}" \
        --output "${OUT_DIR}" --amp "${AMP}" "$@"
    exit 0
fi

PRESET="q1_a100"
if [[ "${MODE}" == "paper" ]]; then
    PRESET="q1_paper"
fi
OUT_DIR="${OUTPUT_DIR:-/data/outputs/CORE_SSOD_V22_${PRESET^^}}"
check_not_active_v21 "${OUT_DIR}"

echo "=================================================================="
echo " Launching CoRe-SSOD v2.2 on NVIDIA A100-SXM4 80GB"
echo " Mode / Preset  : ${MODE} (${PRESET})"
echo " Precision      : ${AMP}"
echo " Batch x Accum  : ${BATCH_SIZE} x ${GRAD_ACCUM} (effective=$((BATCH_SIZE * GRAD_ACCUM))) | LR=${LR}"
echo " Anchor Profile : ${ANCHOR_PROFILE} | Stratified D_cal: enabled"
echo " Output Dir     : ${OUT_DIR}"
echo "=================================================================="

python -u scripts/run_suite.py \
    --preset "${PRESET}" \
    --device cuda \
    --amp "${AMP}" \
    --batch-size "${BATCH_SIZE}" \
    --grad-accum-steps "${GRAD_ACCUM}" \
    --lr "${LR}" \
    --anchor-profile "${ANCHOR_PROFILE}" \
    --stratify-calibration \
    --train-labels "${TRAIN_LABELS}" \
    --val-labels "${VAL_LABELS}" \
    --train-images "${TRAIN_IMAGES}" \
    --val-images "${VAL_IMAGES}" \
    --output "${OUT_DIR}" \
    --resume \
    "$@"
