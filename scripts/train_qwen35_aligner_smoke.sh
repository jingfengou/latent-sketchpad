#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/local_env_qwen35.sh"

SMOKE_DIR="${1:-$LATENT_SKETCHPAD_ROOT/outputs/qwen35_aligner_smoke}"
CONFIG_PATH="$SMOKE_DIR/dense-config-qwen35-smoke.json"

mkdir -p "$SMOKE_DIR"
mkdir -p "$WANDB_DIR"
printf '[wandb] mode=%s disabled=%s dir=%s\n' "$WANDB_MODE" "$WANDB_DISABLED" "$WANDB_DIR"

cat > "$CONFIG_PATH" <<'EOF'
{
  "project_name": "sketch-decoder-qwen35-aligner-smoke",
  "learning_rate": 0.0001,
  "epochs": 1,
  "batch_size": 2,
  "non_background_weight": 1,
  "setting": "qwen35_dense_aligner_smoke",
  "checkpoints_dir": "checkpoints",
  "restore_optimizer": false,
  "dataset": "quickdraw-344-classes",
  "cate_num": 1,
  "number_per_class": 8,
  "train_split": 0.8,
  "val_split": 0.1,
  "dense_align": true,
  "gray_image": true,
  "eval_every_n_steps": 1,
  "vision_model_name": "../models/Qwen3.5-4B",
  "image_size": 224,
  "layer": 2,
  "input_dim": 1024,
  "accumulate_grad_batches": 1,
  "causal_mask": false
}
EOF

CONFIG_FILE_PATH="$CONFIG_PATH" \
MOUNT_DIR="$SMOKE_DIR" \
DATA_DIR="$LATENT_SKETCHPAD_ROOT/decoder" \
"$LATENT_SKETCHPAD_PYTHON_BIN" \
  "$LATENT_SKETCHPAD_ROOT/decoder/train.py" \
  --gpus 1
