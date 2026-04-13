#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/local_env_qwen35.sh"

PYTHON_BIN="/workspace/home/miniconda3/envs/${LATENT_SKETCHPAD_CONDA_ENV}/bin/python"

WAIT_MODE=0
if [[ "${1:-}" == "--wait" ]]; then
  WAIT_MODE=1
  shift
fi

MODEL_PATH="${1:-$LATENT_SKETCHPAD_ROOT/outputs/3dproject_stage2_qwen35_full_lfalign_perceiver_run2/checkpoint-3330}"
OUTPUT_ROOT="${2:-$LATENT_SKETCHPAD_ROOT/outputs/3dproject_stage2_qwen35_full_lfalign_perceiver_run2/test_eval_final}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
NUM_SHARDS="${NUM_SHARDS:-4}"

mkdir -p "$OUTPUT_ROOT"

if [[ "$NUM_SHARDS" != "4" ]]; then
  printf 'This wrapper currently expects NUM_SHARDS=4 for 2 GPUs x 2 processes per GPU.\n' >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES=0 nohup "$PYTHON_BIN" \
  "$LATENT_SKETCHPAD_ROOT/scripts/eval_3dproject_stage2_qwen35.py" \
  --model-path "$MODEL_PATH" \
  --data-path "$LATENT_SKETCHPAD_3DPROJECT_TEST" \
  --image-dir "$LATENT_SKETCHPAD_IMAGE_ROOT" \
  --output-dir "$OUTPUT_ROOT/shard_0" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --num-shards "$NUM_SHARDS" \
  --shard-id 0 \
  --use-perceiver > "$OUTPUT_ROOT/shard_0.log" 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup "$PYTHON_BIN" \
  "$LATENT_SKETCHPAD_ROOT/scripts/eval_3dproject_stage2_qwen35.py" \
  --model-path "$MODEL_PATH" \
  --data-path "$LATENT_SKETCHPAD_3DPROJECT_TEST" \
  --image-dir "$LATENT_SKETCHPAD_IMAGE_ROOT" \
  --output-dir "$OUTPUT_ROOT/shard_1" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --num-shards "$NUM_SHARDS" \
  --shard-id 1 \
  --use-perceiver > "$OUTPUT_ROOT/shard_1.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup "$PYTHON_BIN" \
  "$LATENT_SKETCHPAD_ROOT/scripts/eval_3dproject_stage2_qwen35.py" \
  --model-path "$MODEL_PATH" \
  --data-path "$LATENT_SKETCHPAD_3DPROJECT_TEST" \
  --image-dir "$LATENT_SKETCHPAD_IMAGE_ROOT" \
  --output-dir "$OUTPUT_ROOT/shard_2" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --num-shards "$NUM_SHARDS" \
  --shard-id 2 \
  --use-perceiver > "$OUTPUT_ROOT/shard_2.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup "$PYTHON_BIN" \
  "$LATENT_SKETCHPAD_ROOT/scripts/eval_3dproject_stage2_qwen35.py" \
  --model-path "$MODEL_PATH" \
  --data-path "$LATENT_SKETCHPAD_3DPROJECT_TEST" \
  --image-dir "$LATENT_SKETCHPAD_IMAGE_ROOT" \
  --output-dir "$OUTPUT_ROOT/shard_3" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --num-shards "$NUM_SHARDS" \
  --shard-id 3 \
  --use-perceiver > "$OUTPUT_ROOT/shard_3.log" 2>&1 &

printf 'Started stage2 shard evaluations in %s\n' "$OUTPUT_ROOT"

if [[ "$WAIT_MODE" == "1" ]]; then
  while true; do
    ready=1
    for shard_id in 0 1 2 3; do
      if [[ ! -f "$OUTPUT_ROOT/shard_${shard_id}/summary.json" ]]; then
        ready=0
        break
      fi
    done

    if [[ "$ready" == "1" ]]; then
      break
    fi
    sleep 30
  done

  "$PYTHON_BIN" \
    "$LATENT_SKETCHPAD_ROOT/scripts/merge_3dproject_eval_shards.py" \
    --input-root "$OUTPUT_ROOT"
fi
