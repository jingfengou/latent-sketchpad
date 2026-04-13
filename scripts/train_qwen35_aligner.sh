#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/local_env_qwen35.sh"

export CONFIG_FILE_PATH="${CONFIG_FILE_PATH:-$LATENT_SKETCHPAD_ROOT/decoder/configs/dense-config-qwen35.json}"
export MOUNT_DIR="${MOUNT_DIR:-$LATENT_SKETCHPAD_ROOT/outputs/qwen35_aligner}"
export DATA_DIR="${DATA_DIR:-$LATENT_SKETCHPAD_ROOT/decoder}"

mkdir -p "$MOUNT_DIR"
mkdir -p "$WANDB_DIR"
printf '[wandb] mode=%s disabled=%s dir=%s\n' "$WANDB_MODE" "$WANDB_DISABLED" "$WANDB_DIR"

"/workspace/home/miniconda3/envs/${LATENT_SKETCHPAD_CONDA_ENV}/bin/python" \
  "$LATENT_SKETCHPAD_ROOT/decoder/train.py" \
  --gpus 2
