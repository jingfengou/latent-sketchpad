#!/usr/bin/env bash

export LATENT_SKETCHPAD_ROOT="/workspace/home/oujingfeng/project/Latent-Sketchpad"
export LATENT_SKETCHPAD_CONDA_ENV="sketchpad-clean"

export CUDA_HOME="/workspace/home/miniconda3"
export LD_LIBRARY_PATH="/workspace/home/miniconda3/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="/workspace/home/miniconda3/envs/${LATENT_SKETCHPAD_CONDA_ENV}/bin:/workspace/home/miniconda3/bin:$PATH"

export LATENT_SKETCHPAD_QWEN_PATH="/workspace/home/oujingfeng/project/models/Qwen2.5-VL-7B-Instruct"
export LATENT_SKETCHPAD_DECODER_PATH="/workspace/home/oujingfeng/project/models/Latent-Sketchpad.Sketch_Decoder/sketch_decoder_qwen25_vl.ckpt"
export LATENT_SKETCHPAD_IMAGE_ROOT="/workspace/home/oujingfeng/project/unimrg/datasets/spatialviz"

export LATENT_SKETCHPAD_3DPROJECT_TRAIN="/workspace/home/oujingfeng/project/unimrg/datasets/spatialviz/latent_sketchpad_3dproject_currentstyle_3329/train.json"
export LATENT_SKETCHPAD_3DPROJECT_VAL="/workspace/home/oujingfeng/project/unimrg/datasets/spatialviz/latent_sketchpad_3dproject_currentstyle_3329/val.json"
export LATENT_SKETCHPAD_3DPROJECT_TEST="/workspace/home/oujingfeng/project/unimrg/datasets/spatialviz/latent_sketchpad_3dproject_currentstyle_3329/test.json"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export TRITON_CACHE_DIR="/workspace/home/.triton/latent-sketchpad"
