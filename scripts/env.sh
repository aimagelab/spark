#!/usr/bin/env bash
# Shared configuration. Edit the four paths below (or export them yourself),
# then `source scripts/env.sh` before running any of the other scripts.
#
# Every one of these is also accepted as a command-line flag, so nothing here
# is mandatory -- it just keeps the example commands short.

# Stable Diffusion 3 Medium (diffusers layout).
export SD3_MODEL_PATH="${SD3_MODEL_PATH:-$PWD/checkpoints/stable-diffusion-3-medium-diffusers}"

# Pretrained TSD-SR LoRA weights and the precomputed prompt embeddings that
# ship with them.
export TSDSR_LORA_DIR="${TSDSR_LORA_DIR:-$PWD/checkpoints/tsdsr/lora}"
export TSDSR_EMBEDDING_DIR="${TSDSR_EMBEDDING_DIR:-$PWD/checkpoints/tsdsr/embeddings}"

# Root holding DRealSR/, RealSR/, DIV2K/ and DIV2K_train/ (see README).
export DATA_ROOT="${DATA_ROOT:-$PWD/datasets}"

# Where runs and evaluations are written.
export OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/outputs}"

# uv's cache can grow to several GB; keep it off small home quotas.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"

echo "[env] SD3_MODEL_PATH      = $SD3_MODEL_PATH"
echo "[env] TSDSR_LORA_DIR      = $TSDSR_LORA_DIR"
echo "[env] TSDSR_EMBEDDING_DIR = $TSDSR_EMBEDDING_DIR"
echo "[env] DATA_ROOT           = $DATA_ROOT"
echo "[env] OUTPUT_ROOT         = $OUTPUT_ROOT"
