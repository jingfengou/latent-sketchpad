#!/usr/bin/env bash

export LATENT_SKETCHPAD_ROOT="/workspace/home/oujingfeng/project/Latent-Sketchpad"
export LATENT_SKETCHPAD_CONDA_ENV="sketchpad-qwen35-cu128"

export CUDA_HOME="/workspace/home/miniconda3"
export LD_LIBRARY_PATH="/workspace/home/miniconda3/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="/workspace/home/miniconda3/envs/${LATENT_SKETCHPAD_CONDA_ENV}/bin:/workspace/home/miniconda3/bin:$PATH"

export LATENT_SKETCHPAD_QWEN35_PATH="/workspace/home/oujingfeng/project/models/Qwen3.5-4B"
export LATENT_SKETCHPAD_QWEN35_STAGE1_PATH="/workspace/home/oujingfeng/project/Latent-Sketchpad/outputs/spatialviz_merged_stage1_qwen35_full"
export LATENT_SKETCHPAD_DS_Z2_CONFIG="/workspace/home/oujingfeng/project/LLaMA-Factory/examples/deepspeed/ds_z2_config.json"
export LATENT_SKETCHPAD_DECODER_PATH="/workspace/home/oujingfeng/project/models/Latent-Sketchpad.Sketch_Decoder/sketch_decoder_qwen25_vl.ckpt"
export LATENT_SKETCHPAD_IMAGE_ROOT="/workspace/home/oujingfeng/project/unimrg/datasets/spatialviz"

export LATENT_SKETCHPAD_3DPROJECT_TRAIN="/workspace/home/oujingfeng/project/unimrg/datasets/spatialviz/latent_sketchpad_3dproject_currentstyle_3329/train.json"
export LATENT_SKETCHPAD_3DPROJECT_VAL="/workspace/home/oujingfeng/project/unimrg/datasets/spatialviz/latent_sketchpad_3dproject_currentstyle_3329/val.json"
export LATENT_SKETCHPAD_3DPROJECT_TEST="/workspace/home/oujingfeng/project/unimrg/datasets/spatialviz/latent_sketchpad_3dproject_currentstyle_3329/test.json"

export LATENT_SKETCHPAD_SPATIALVIZ_MERGED_ROOT="/workspace/home/oujingfeng/project/unimrg/datasets/spatialviz/latent_sketchpad_spatialviz_merged_qwen35"
export LATENT_SKETCHPAD_SPATIALVIZ_MERGED_TRAIN="$LATENT_SKETCHPAD_SPATIALVIZ_MERGED_ROOT/train.json"
export LATENT_SKETCHPAD_SPATIALVIZ_MERGED_VAL="$LATENT_SKETCHPAD_SPATIALVIZ_MERGED_ROOT/val.json"
export LATENT_SKETCHPAD_SPATIALVIZ_MERGED_TEST="$LATENT_SKETCHPAD_SPATIALVIZ_MERGED_ROOT/test.json"

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DISABLED="${WANDB_DISABLED:-false}"
export WANDB_DIR="${WANDB_DIR:-$LATENT_SKETCHPAD_ROOT/wandb}"
export TRITON_CACHE_DIR="/workspace/home/.triton/latent-sketchpad-qwen35"
