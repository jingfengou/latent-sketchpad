#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/local_env.sh"

WAIT_MODE=0
if [[ "${1:-}" == "--wait" ]]; then
  WAIT_MODE=1
  shift
fi

CHECKPOINT_PATH="${1:-$LATENT_SKETCHPAD_ROOT/outputs/3dproject_stage1_full_lora/checkpoint-200}"
OUTPUT_ROOT="${2:-$LATENT_SKETCHPAD_ROOT/outputs/3dproject_stage1_full_lora/test_eval_checkpoint200}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
NUM_SHARDS="${NUM_SHARDS:-4}"

mkdir -p "$OUTPUT_ROOT"

if [[ "$NUM_SHARDS" != "4" ]]; then
  echo "This wrapper currently expects NUM_SHARDS=4 for 2 GPUs x 2 processes per GPU." >&2
  echo "Got NUM_SHARDS=$NUM_SHARDS" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES=0 nohup conda run -n "$LATENT_SKETCHPAD_CONDA_ENV" python \
  "$LATENT_SKETCHPAD_ROOT/scripts/eval_3dproject_stage1_lora.py" \
  --base-model "$LATENT_SKETCHPAD_QWEN_PATH" \
  --adapter-path "$CHECKPOINT_PATH" \
  --data-path "$LATENT_SKETCHPAD_3DPROJECT_TEST" \
  --image-dir "$LATENT_SKETCHPAD_IMAGE_ROOT" \
  --output-dir "$OUTPUT_ROOT/shard_0" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --num-shards "$NUM_SHARDS" \
  --shard-id 0 > "$OUTPUT_ROOT/shard_0.log" 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup conda run -n "$LATENT_SKETCHPAD_CONDA_ENV" python \
  "$LATENT_SKETCHPAD_ROOT/scripts/eval_3dproject_stage1_lora.py" \
  --base-model "$LATENT_SKETCHPAD_QWEN_PATH" \
  --adapter-path "$CHECKPOINT_PATH" \
  --data-path "$LATENT_SKETCHPAD_3DPROJECT_TEST" \
  --image-dir "$LATENT_SKETCHPAD_IMAGE_ROOT" \
  --output-dir "$OUTPUT_ROOT/shard_1" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --num-shards "$NUM_SHARDS" \
  --shard-id 1 > "$OUTPUT_ROOT/shard_1.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup conda run -n "$LATENT_SKETCHPAD_CONDA_ENV" python \
  "$LATENT_SKETCHPAD_ROOT/scripts/eval_3dproject_stage1_lora.py" \
  --base-model "$LATENT_SKETCHPAD_QWEN_PATH" \
  --adapter-path "$CHECKPOINT_PATH" \
  --data-path "$LATENT_SKETCHPAD_3DPROJECT_TEST" \
  --image-dir "$LATENT_SKETCHPAD_IMAGE_ROOT" \
  --output-dir "$OUTPUT_ROOT/shard_2" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --num-shards "$NUM_SHARDS" \
  --shard-id 2 > "$OUTPUT_ROOT/shard_2.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup conda run -n "$LATENT_SKETCHPAD_CONDA_ENV" python \
  "$LATENT_SKETCHPAD_ROOT/scripts/eval_3dproject_stage1_lora.py" \
  --base-model "$LATENT_SKETCHPAD_QWEN_PATH" \
  --adapter-path "$CHECKPOINT_PATH" \
  --data-path "$LATENT_SKETCHPAD_3DPROJECT_TEST" \
  --image-dir "$LATENT_SKETCHPAD_IMAGE_ROOT" \
  --output-dir "$OUTPUT_ROOT/shard_3" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --num-shards "$NUM_SHARDS" \
  --shard-id 3 > "$OUTPUT_ROOT/shard_3.log" 2>&1 &

printf 'Started shard evaluations in %s\n' "$OUTPUT_ROOT"
printf 'Logs:\n%s\n%s\n%s\n%s\n' \
  "$OUTPUT_ROOT/shard_0.log" \
  "$OUTPUT_ROOT/shard_1.log" \
  "$OUTPUT_ROOT/shard_2.log" \
  "$OUTPUT_ROOT/shard_3.log"

if [[ "$WAIT_MODE" == "1" ]]; then
  echo "Waiting for shard summaries to appear..."
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

  conda run -n "$LATENT_SKETCHPAD_CONDA_ENV" python \
    "$LATENT_SKETCHPAD_ROOT/scripts/merge_3dproject_eval_shards.py" \
    --input-root "$OUTPUT_ROOT"
fi
