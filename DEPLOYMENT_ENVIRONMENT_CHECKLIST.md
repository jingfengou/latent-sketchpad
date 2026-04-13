# Deployment Environment Checklist

This document records the environment requirements and deployment steps needed to run the current `Latent-Sketchpad` Qwen3.5 workflow on another server.

## Scope

This checklist is intended for the current working setup used for:

- Qwen3.5 stage1 training
- Qwen3.5 stage2 training
- stage1 evaluation scripts
- decoder-aligner training

It is based on the currently active environment in this workspace and should be treated as the source of truth when reproducing the setup elsewhere.

## Current Runtime Environment

- Conda env name: `sketchpad-qwen35-cu128`
- Python: `3.11`
- GPU: `NVIDIA H100 80GB HBM3`
- Torch CUDA runtime: `12.8`
- cuDNN: `91900`

## If The Target Server Has No CUDA

If the remote server does not have a CUDA-capable GPU available, then it should not be treated as a full training server for the current workflow.

What will not be practical on a no-CUDA server:

- Qwen3.5 stage1 training
- Qwen3.5 stage2 training
- decoder-aligner training
- multi-image Qwen evaluation at normal speed
- installing and using CUDA-dependent extensions such as `flash-attention`

What is still reasonable on a no-CUDA server:

- cloning and versioning the repository
- editing code and scripts
- preparing JSON datasets and path mappings
- documentation work
- lightweight sanity checks that do not require model execution on GPU
- downloading or organizing non-model assets

Recommended split if you have both GPU and non-GPU machines:

- Use the non-CUDA server as a code/data preparation machine.
- Use a CUDA server as the actual training/inference machine.
- Sync code through GitHub.
- Sync large datasets/models separately with `rsync`, object storage, or shared mounts.

If the target machine has no CUDA and no NVIDIA GPU, you should skip CUDA toolkit and `flash-attention` setup entirely for that machine.

## Exact Package Versions

The following versions were read from the current training environment.

```text
torch==2.7.1+cu128
torchvision==0.22.1+cu128
torchaudio==2.7.1+cu128
transformers==5.2.0
accelerate==1.11.0
deepspeed==0.18.4
peft==0.18.0
datasets==3.2.0
diffusers==0.38.0.dev0
lightning==2.6.1
wandb==0.25.1
safetensors==0.7.0
einops==0.8.2
PIL==11.3.0
requests==2.33.0
torchdata==0.11.0+cpu
open_clip==3.3.0
timm==1.0.26
numpy==1.26.4
scipy==1.17.1
matplotlib==3.10.8
tqdm==4.67.3
jsonlines==MISSING
xformers==MISSING
```

Notes:

- `jsonlines` is currently missing in the main Qwen3.5 environment.
- `xformers` is currently missing in the main Qwen3.5 environment.
- If a new server needs decoder-side utilities that require these packages, install them explicitly.

## Recommended Base Setup On A New Server

Install basic OS packages:

```bash
sudo apt-get update
sudo apt-get install -y git tmux wget curl build-essential
```

If Miniconda is not installed:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

Clone the repository:

```bash
git clone git@github.com:jingfengou/latent-sketchpad.git
cd latent-sketchpad
```

Create the conda environment:

```bash
conda create -n sketchpad-qwen35-cu128 python=3.11 -y
conda activate sketchpad-qwen35-cu128
```

Install the currently used package versions:

```bash
pip install --upgrade pip
pip install \
  torch==2.7.1+cu128 \
  torchvision==0.22.1+cu128 \
  torchaudio==2.7.1+cu128

pip install \
  transformers==5.2.0 \
  accelerate==1.11.0 \
  deepspeed==0.18.4 \
  peft==0.18.0 \
  datasets==3.2.0 \
  diffusers==0.38.0.dev0 \
  lightning==2.6.1 \
  wandb==0.25.1 \
  safetensors==0.7.0 \
  einops==0.8.2 \
  pillow==11.3.0 \
  requests==2.33.0 \
  torchdata==0.11.0 \
  open_clip_torch==3.3.0 \
  timm==1.0.26 \
  numpy==1.26.4 \
  scipy==1.17.1 \
  matplotlib==3.10.8 \
  tqdm==4.67.3
```

Optional packages if needed by your workflow:

```bash
pip install jsonlines
pip install xformers
```

## Required External Assets

At minimum, the following assets must exist on the new server:

### Model paths

- Qwen3.5 base model
- Sketch decoder checkpoint
- Stage1 output checkpoint directory if running stage2

Current code expects paths similar to:

```text
/workspace/home/oujingfeng/project/models/Qwen3.5-4B
/workspace/home/oujingfeng/project/models/Latent-Sketchpad.Sketch_Decoder/sketch_decoder_qwen25_vl.ckpt
/workspace/home/oujingfeng/project/Latent-Sketchpad/outputs/spatialviz_merged_stage1_qwen35_full
```

### Dataset paths

Current scripts expect:

```text
unimrg/datasets/spatialviz/latent_sketchpad_spatialviz_merged_qwen35/
unimrg/datasets/spatialviz/latent_sketchpad_3dproject_currentstyle_3329/
datasets/spatialviz-ours/
```

You must ensure the JSON files and referenced images are both available.

## QuickDraw Setup

QuickDraw data is not committed in the repository and should be downloaded on the target server.

The repair/download script is:

```text
scripts/download_quickdraw_npz.py
```

Run:

```bash
python scripts/download_quickdraw_npz.py --workers 16
```

Verify only:

```bash
python scripts/download_quickdraw_npz.py --verify-only
```

Default QuickDraw root:

```text
decoder/QuickDraw/
```

## Environment Variables And Paths

Before running training on a new server, review and update:

```text
scripts/local_env_qwen35.sh
```

This file contains machine-specific absolute paths for:

- `LATENT_SKETCHPAD_ROOT`
- `LATENT_SKETCHPAD_QWEN35_STAGE1_PATH`
- merged dataset root
- `WANDB_DIR`
- `TRITON_CACHE_DIR`

These should be adjusted to match the new machine.

## Current Training/Eval Entry Scripts

Stage1 train:

```bash
bash scripts/train_spatialviz_merged_stage1_qwen35_full.sh
```

Stage2 train:

```bash
bash scripts/train_spatialviz_merged_stage2_qwen35_full.sh
```

Stage1 eval shard launcher:

```bash
scripts/run_eval_stage1_qwen35_shard.sh MODEL_PATH OUTPUT_DIR SHARD_ID NUM_SHARDS
```

Decoder-aligner train:

```bash
bash scripts/train_qwen35_aligner.sh
```

## Recommended Validation Order On A New Server

1. Verify CUDA and Torch:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

2. Verify imports:

```bash
python -c "import transformers, deepspeed, diffusers, lightning"
```

3. Verify local paths in `scripts/local_env_qwen35.sh`

4. Run a stage1 training launch

5. Run one stage1 eval shard

6. Run stage2 training

7. Run decoder-aligner training if needed

## Publishing Notes

The repository currently ignores local artifacts such as:

- `outputs/`
- `tmp/`
- `.gradio/`
- `decoder/QuickDraw/`
- `wandb/`
- `__pycache__/`

This is intentional so the GitHub repository remains code-only and can be cloned on other servers without large local artifacts.
