#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/local_env_qwen35.sh"

MODEL_PATH="${1:-$LATENT_SKETCHPAD_ROOT/outputs/spatialviz_merged_stage1_qwen35_full}"
OUTPUT_DIR="${2:-$LATENT_SKETCHPAD_ROOT/outputs/spatialviz_merged_stage1_qwen35_full/test_eval_merged}"

mkdir -p "$OUTPUT_DIR"

"/workspace/home/miniconda3/envs/${LATENT_SKETCHPAD_CONDA_ENV}/bin/python" \
  "$LATENT_SKETCHPAD_ROOT/scripts/eval_3dproject_stage1_qwen35.py" \
  --model-path "$MODEL_PATH" \
  --data-path "$LATENT_SKETCHPAD_SPATIALVIZ_MERGED_TEST" \
  --image-dir "/workspace/home/oujingfeng/project" \
  --output-dir "$OUTPUT_DIR" \
  --max-new-tokens 2500

"/workspace/home/miniconda3/envs/${LATENT_SKETCHPAD_CONDA_ENV}/bin/python" \
  "$LATENT_SKETCHPAD_ROOT/scripts/summarize_eval_by_task.py" \
  --predictions "$OUTPUT_DIR/predictions.json" \
  --data-path "$LATENT_SKETCHPAD_SPATIALVIZ_MERGED_TEST" \
  --output-path "$OUTPUT_DIR/task_summary.json"
