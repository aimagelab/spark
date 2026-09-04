#!/usr/bin/env bash
# Vanilla TSD-SR on a benchmark -- the "baseline" rows of the main table.
#
#   source scripts/env.sh
#   bash scripts/eval_baseline.sh DRealSR
#   bash scripts/eval_baseline.sh RealSR
#   bash scripts/eval_baseline.sh DIV2K
#
# Writes SR images, per-image metrics.csv and metrics_summary.json into
# $OUTPUT_ROOT/baseline/<dataset>. Re-running resumes where it left off.

set -euo pipefail

DATASET="${1:-DRealSR}"
OUT_DIR="${2:-${OUTPUT_ROOT:-outputs}/baseline/${DATASET}}"

python -u infer_tsdsr_baseline_dataset.py \
  --dataset "${DATASET}" \
  --output_dir "${OUT_DIR}" \
  --pretrained_model_name_or_path "${SD3_MODEL_PATH}" \
  --lora_dir "${TSDSR_LORA_DIR}" \
  --embedding_dir "${TSDSR_EMBEDDING_DIR}" \
  --mixed_precision bf16 \
  --align_method wavelet \
  --metrics lpips ssim maniqa-pipal clipiqa musiq liqe \
  --resume

echo "[done] ${OUT_DIR}/metrics_summary.json"
