#!/usr/bin/env bash
# TSD-SR + our AdaLN modulation on a benchmark -- the "+Ours" rows of the main
# table. Takes a run directory produced by scripts/train_predictor.sh.
#
#   source scripts/env.sh
#   bash scripts/eval_predictor.sh outputs/topk8_paper DRealSR
#
# The run directory must contain the trained predictor checkpoint and the
# predictor config written by training; the script reads both from there, so no
# hyper-parameter needs to be repeated here.

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/eval_predictor.sh <run_dir> <DRealSR|RealSR|DIV2K> [output_dir]"
  exit 2
fi

RUN_DIR="$1"
DATASET="$2"
OUT_DIR="${3:-${RUN_DIR}/predictor_eval_${DATASET}}"

python -u infer_adaln_predictor_dataset.py \
  --experiment_dir "${RUN_DIR}" \
  --dataset "${DATASET}" \
  --output_dir "${OUT_DIR}" \
  --pretrained_model_name_or_path "${SD3_MODEL_PATH}" \
  --lora_dir "${TSDSR_LORA_DIR}" \
  --embedding_dir "${TSDSR_EMBEDDING_DIR}" \
  --mixed_precision bf16 \
  --align_method wavelet

echo "[done] ${OUT_DIR}/metrics_summary.json"
