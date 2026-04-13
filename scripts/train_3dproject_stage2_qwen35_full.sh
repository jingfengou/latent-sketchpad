#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/local_env_qwen35.sh"

MASTER_PORT="${MASTER_PORT:-29635}"
OUTPUT_DIR="${1:-$LATENT_SKETCHPAD_ROOT/outputs/3dproject_stage2_qwen35_full}"

mkdir -p "$WANDB_DIR"
printf '[wandb] mode=%s disabled=%s dir=%s\n' "$WANDB_MODE" "$WANDB_DISABLED" "$WANDB_DIR"

"$LATENT_SKETCHPAD_DEEPSPEED_BIN" \
  --num_gpus=2 \
  --master_port="$MASTER_PORT" \
  "$LATENT_SKETCHPAD_ROOT/train_stage2_qwen35.py" \
  --data_path "$LATENT_SKETCHPAD_3DPROJECT_TRAIN" \
  --image_dir "$LATENT_SKETCHPAD_IMAGE_ROOT" \
  --decoder_path "$LATENT_SKETCHPAD_DECODER_PATH" \
  --model_path "${LATENT_SKETCHPAD_QWEN35_STAGE1_PATH:-$LATENT_SKETCHPAD_QWEN35_PATH}" \
  --learning_rate 1e-5 \
  --max_grad_norm 1.0 \
  --num_train_epochs 5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --per_device_eval_batch_size 1 \
  --weight_decay 0.01 \
  --save_steps 1000 \
  --eval_steps 100 \
  --logging_steps 10 \
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
