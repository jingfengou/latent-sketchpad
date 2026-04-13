#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/local_env.sh"

MASTER_PORT="${MASTER_PORT:-29624}"
OUTPUT_DIR="${1:-$LATENT_SKETCHPAD_ROOT/outputs/3dproject_stage1_full_lora_e3_b16_lr5e5_noval}"

"$LATENT_SKETCHPAD_PYTHON_BIN" -m torch.distributed.run \
  --nproc_per_node=2 \
  --master_port="$MASTER_PORT" \
  "$LATENT_SKETCHPAD_ROOT/train.py" \
  --data_path "$LATENT_SKETCHPAD_3DPROJECT_TRAIN" \
  --decoder_path "$LATENT_SKETCHPAD_DECODER_PATH" \
  --image_dir "$LATENT_SKETCHPAD_IMAGE_ROOT" \
  --model_path "$LATENT_SKETCHPAD_QWEN_PATH" \
  --learning_rate 5e-5 \
  --max_grad_norm 1.0 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --per_device_eval_batch_size 1 \
  --weight_decay 0.01 \
  --ds_config "$LATENT_SKETCHPAD_ROOT/ds_cfg.json" \
  --save_steps 167 \
  --eval_steps 167 \
  --logging_steps 1 \
  --resume_from_checkpoint False \
  --output_dir "$OUTPUT_DIR" \
  --wandb_project latent-sketchpad-3dproject \
  --disable_eval \
  --augment \
  --image_loss_weight 0.0 \
  --unfreeze-connector \
  --stage1 \
  --sum-loss \
  --loss_type l1 \
  --use-lora
