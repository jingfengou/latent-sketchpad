#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/local_env_qwen35.sh"

MODEL_PATH="${1:-$LATENT_SKETCHPAD_ROOT/outputs/spatialviz_merged_stage2_qwen35_full}"
OUTPUT_DIR="${2:-$LATENT_SKETCHPAD_ROOT/outputs/spatialviz_merged_stage2_infer_qwen35}"
SAMPLE_INDEX="${3:-0}"

mkdir -p "$OUTPUT_DIR"

"$LATENT_SKETCHPAD_PYTHON_BIN" \
  "$LATENT_SKETCHPAD_ROOT/inference_qwen35_stage2.py" \
  --model_path "$MODEL_PATH" \
  --decoder_path "$LATENT_SKETCHPAD_DECODER_PATH" \
  --data_path "$LATENT_SKETCHPAD_SPATIALVIZ_MERGED_TEST" \
  --output_dir "$OUTPUT_DIR" \
  --sample_index "$SAMPLE_INDEX" \
  --use_perceiver
