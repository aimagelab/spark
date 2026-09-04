#!/usr/bin/env bash
# Train the AdaLN predictor with the paper configuration (main-table setting).
#
#   source scripts/env.sh
#   bash scripts/train_predictor.sh [run_name]
#
# Phase 1 (online channel selection, Algorithm 1) and Phase 2 (predictor
# training) both run inside this single command.
#
# One 48GB GPU (L40S). Phase 1 stops early on its own -- typically after ~400
# images / ~25 mini-batches, about 8 minutes. Phase 2 then takes ~2.5 hours.

set -euo pipefail

RUN_NAME="${1:-topk8_paper}"
[[ $# -gt 0 ]] && shift   # anything left over is forwarded to the python script
RUN_DIR="${OUTPUT_ROOT:-outputs}/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

# ---- paper configuration --------------------------------------------------
TOPK="${TOPK:-8}"                       # number of modulated AdaLN channels
SELECTION_MODE="${SELECTION_MODE:-topk}" # topk | bottomk | random (ablations)
IMPORTANCE_MODE="${IMPORTANCE_MODE:-mean_abs}"   # mean_abs | std (supp. C.2)
# Phase-1 online EMA selection (Algorithm 1)
PHASE1_BATCH_SIZE="${PHASE1_BATCH_SIZE:-16}"
EMA_DECAY="${EMA_DECAY:-0.95}"          # lambda
WINDOW="${WINDOW:-5}"                   # W
WINDOW_TRANSITIONS="${WINDOW_TRANSITIONS:-4}"  # P
STABILITY_TAU="${STABILITY_TAU:-0.9}"   # tau
LR="${LR:-1e-4}"
ALPHA_LPIPS="${ALPHA_LPIPS:-1.0}"
ALPHA_IQA="${ALPHA_IQA:-0.1}"           # LIQE term
ALPHA_TV="${ALPHA_TV:-0.0001}"
STEPS="${STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
SEED="${SEED:-42}"
# Upper budget only: the stability criterion normally terminates well before it.
PHASE1_SAMPLES="${PHASE1_SAMPLES:-2000}"

python -u train_adaln_predictor.py \
  --pretrained_model_name_or_path "${SD3_MODEL_PATH}" \
  --lora_dir "${TSDSR_LORA_DIR}" \
  --embedding_dir "${TSDSR_EMBEDDING_DIR}" \
  --input_dir "${DATA_ROOT}/DRealSR/test_LR" \
  --gt_dir "${DATA_ROOT}/DRealSR/test_HR" \
  --opt_input_dir "${DATA_ROOT}/DIV2K_train/LR" \
  --opt_gt_dir "${DATA_ROOT}/DIV2K_train/HR" \
  --output_dir "${RUN_DIR}" \
  --adaln_predictor_train \
  --adaln_predictor_steps "${STEPS}" \
  --adaln_predictor_max_num_steps "${STEPS}" \
  --adaln_predictor_max_num_epochs 1 \
  --adaln_predictor_batch_size "${BATCH_SIZE}" \
  --adaln_predictor_grad_accum_steps "${GRAD_ACCUM}" \
  --adaln_predictor_lr "${LR}" \
  --adaln_predictor_alpha_lpips "${ALPHA_LPIPS}" \
  --adaln_predictor_alpha_liqe "${ALPHA_IQA}" \
  --adaln_predictor_alpha_tv "${ALPHA_TV}" \
  --adaln_predictor_tv_norm l1 \
  --adaln_predictor_val_every 100 \
  --adaln_predictor_early_stop_patience 5 \
  --selection_mode "${SELECTION_MODE}" \
  --topk "${TOPK}" \
  --adaln_per_channel \
  --stream_weight_mode manual \
  --phase1_selector online_ema \
  --phase1_importance_mode "${IMPORTANCE_MODE}" \
  --phase1_batch_size "${PHASE1_BATCH_SIZE}" \
  --phase1_ema_decay "${EMA_DECAY}" \
  --phase1_window "${WINDOW}" \
  --phase1_window_transitions "${WINDOW_TRANSITIONS}" \
  --phase1_stability_tau "${STABILITY_TAU}" \
  --phase1_max_samples "${PHASE1_SAMPLES}" \
  --phase1_shuffle_samples \
  --seed "${SEED}" \
  "$@"

echo "[done] run directory: ${RUN_DIR}"
