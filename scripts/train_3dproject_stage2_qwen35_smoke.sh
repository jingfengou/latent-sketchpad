#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/local_env_qwen35.sh"

OUTPUT_DIR="${1:-$LATENT_SKETCHPAD_ROOT/outputs/qwen35_stage2_smoke1_stable_from_stage1}"
MASTER_PORT="${MASTER_PORT:-29636}"

"$LATENT_SKETCHPAD_DEEPSPEED_BIN" \
  --num_gpus=2 \
  --master_port="$MASTER_PORT" \
  "$LATENT_SKETCHPAD_ROOT/train_stage2_qwen35.py" \
  --data_path "$LATENT_SKETCHPAD_3DPROJECT_TRAIN_SMOKE32" \
  --image_dir "$LATENT_SKETCHPAD_IMAGE_ROOT" \
  --decoder_path "$LATENT_SKETCHPAD_DECODER_PATH" \
  --model_path "${LATENT_SKETCHPAD_QWEN35_STAGE1_PATH:-$LATENT_SKETCHPAD_QWEN35_PATH}" \
  --learning_rate 1e-4 \
  --num_train_epochs 1 \
  --max_steps 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --per_device_eval_batch_size 1 \
  --save_steps 1 \
  --logging_steps 1 \
  --output_dir "$OUTPUT_DIR" \
  --wandb_project latent-sketchpad-qwen35 \
  --ds_config "$LATENT_SKETCHPAD_DS_Z2_CONFIG" \
  --dataloader_num_workers 8 \
  --dataloader_pin_memory \
  --disable_eval \
  --freeze-backbone \
  --use_perceiver \
  --text_loss_weight 0.0 \
  --image_loss_weight 1.0 \
  --sum-loss \
  --loss_type l1
