# Local 3DProject Setup

This note records the local assets and dataset conversion prepared for trying `Latent-Sketchpad` with the existing `3dproject` interleaved reasoning data.

## Downloaded checkpoints

Qwen2.5-VL base model:

- `/workspace/home/oujingfeng/project/models/Qwen2.5-VL-7B-Instruct`

Sketch Decoder checkpoint for Qwen2.5-VL:

- `/workspace/home/oujingfeng/project/models/Latent-Sketchpad.Sketch_Decoder/sketch_decoder_qwen25_vl.ckpt`

## Source dataset used

The converted data comes from the existing interleaved `3dproject` dataset:

- `/workspace/home/oujingfeng/project/unimrg/datasets/spatialviz/showo2_3dproject_currentstyle_interleaved_3329/`

This source already contains:

- the original composite question image
- interleaved reasoning text
- intermediate process images such as `target_progress_1.png`
- final answer tags like `<answer>B</answer>`

## Converted dataset for Latent-Sketchpad

Converted output directory:

- `/workspace/home/oujingfeng/project/unimrg/datasets/spatialviz/latent_sketchpad_3dproject_currentstyle_3329/`

Generated files:

- `train.json` with `2663` samples
- `val.json` with `333` samples
- `test.json` with `333` samples
- `train_smoke32.json`
- `val_smoke32.json`
- `test_smoke32.json`
- `summary.json`

Image root to use with these JSON files:

- `/workspace/home/oujingfeng/project/unimrg/datasets/spatialviz`

## Conversion script

The local conversion script is stored in:

- `/workspace/home/oujingfeng/project/unimrg/scripts/convert_3dproject_to_latent_sketchpad.py`

It converts each sample into the format expected by `Latent-Sketchpad/data/dataset.py`:

- `input_text`
- `input_img`
- `label_text`
- `label_img`

The conversion logic is:

- `input_text`: user question and prompt, ending with one `<image>` placeholder
- `input_img`: the original `composite.png`
- `label_text`: interleaved assistant reasoning text with `<image>` placeholders inserted between reasoning steps, ending with the final `<answer>...</answer>`
- `label_img`: intermediate process images in order

## Environment used

Verified conda environment:

- `sketchpad-clean`

Important package versions in the working environment:

- `python==3.10`
- `torch==2.6.0+cu124`
- `torchvision==0.21.0+cu124`
- `transformers==4.54.0`
- `flash_attn==2.7.4.post1`
- `deepspeed`
- `xformers==0.0.29.post3`

## Required environment variables

The training and inference commands worked with these environment variables set:

- `CUDA_HOME=/workspace/home/miniconda3`
- `LD_LIBRARY_PATH=/workspace/home/miniconda3/lib`
- `PATH=/workspace/home/miniconda3/envs/sketchpad-clean/bin:/workspace/home/miniconda3/bin:$PATH`
- `LATENT_SKETCHPAD_QWEN_PATH=/workspace/home/oujingfeng/project/models/Qwen2.5-VL-7B-Instruct`
- `LATENT_SKETCHPAD_IMAGE_ROOT=/workspace/home/oujingfeng/project/unimrg/datasets/spatialviz`
- `WANDB_MODE=offline`
- `WANDB_DISABLED=true`

## Local code changes applied

The following local fixes were needed to make the project runnable in this workspace:

1. `train.py`, `inference.py`, and `evaluate.py` were patched to use local environment-variable paths instead of checked-in placeholders like `/path/to/...`.
2. `train.py` was patched so `WANDB_DISABLED=true` does not still force the `wandb` callback.
3. `train.py` was patched so an empty `--ds_config` can disable DeepSpeed for local debugging.
4. `train.py` LoRA flow was patched to stop calling `merge_and_unload()` immediately after `get_peft_model(...)`.

## Runtime issues encountered

These were the main issues found while bringing up training:

1. Official `setup_qwen.sh` was not sufficient by itself.
   The environment needed explicit handling for `flash-attn` and CUDA toolchain visibility.

2. `flash_attn` failed when installed with default build isolation.
   Working fix:
   - install with `--no-build-isolation`
   - ensure `CUDA_HOME` points at the local CUDA-capable conda root

3. `torchrun` must come from the `sketchpad-clean` environment.
   Using system `/usr/local/bin/torchrun` launched a Python that did not have `flash_attn` installed.

4. Two-GPU training cannot use the accidental `DataParallel` fallback.
   `Qwen2.5-VL` uses non-standard visual input packing (`pixel_values` and `image_grid_thw`), and `DataParallel` split the visual tensor incorrectly, causing shape/index failures.

5. Single-process DeepSpeed startup tried MPI discovery.
   That path required `mpi4py` and a system MPI runtime, so it was not the right path for quick smoke testing.

6. Full-parameter stage1 training OOMed even on 2x H100.
   The failure happened during optimizer state initialization because the default stage1 path still had about `7.38B` trainable parameters.

7. LoRA stage1 was required to make 2-GPU training fit.
   After the LoRA fix above, trainable parameters dropped to about `594.6M`, and training succeeded.

8. Directly running local helper scripts from outside the project root initially failed with:
   - `ModuleNotFoundError: No module named 'data'`
   Working fix:
   - add the project root to `sys.path` inside `scripts/eval_3dproject_stage1_lora.py`

9. The long shell prefixes were needed because the environment was not yet fully self-describing.
   Concretely, successful runs depended on all of the following being true at the same time:
   - the correct conda env was used: `sketchpad-clean`
   - the correct `torchrun`/`python` from that env was used instead of system `/usr/local/bin/torchrun`
   - CUDA toolchain paths were visible through `CUDA_HOME`, `LD_LIBRARY_PATH`, and `PATH`
   - local model/data paths were injected through `LATENT_SKETCHPAD_*` environment variables

## Why the wrapper scripts are needed

The wrapper scripts are not just convenience helpers. They exist because this repository currently depends on local machine-specific paths and CUDA visibility that are not fully encoded in the Python package itself.

Without the wrapper or equivalent environment setup, common failures included:

- `flash_attn` not found because the wrong Python interpreter launched
- `torchrun` resolving to the system executable instead of the conda env executable
- DeepSpeed/CUDA extension startup failures when `CUDA_HOME` was missing
- local model and image paths falling back to placeholder `/path/to/...` defaults

For this workspace, the supported way to run training/evaluation is:

1. `conda activate sketchpad-clean`
2. use the wrapper scripts in `scripts/`

## Verified working commands

### 1. Official MazePlanning loader smoke test

This confirmed that the public MazePlanning test data and Qwen processor path work:

```bash
CUDA_HOME=/workspace/home/miniconda3 \
LD_LIBRARY_PATH=/workspace/home/miniconda3/lib \
PATH=/workspace/home/miniconda3/envs/sketchpad-clean/bin:/workspace/home/miniconda3/bin:$PATH \
conda run -n sketchpad-clean python -c "from transformers import AutoTokenizer, AutoProcessor; from data.dataset import MultimodalEvalDataset; tok=AutoTokenizer.from_pretrained('/workspace/home/oujingfeng/project/models/Qwen2.5-VL-7B-Instruct'); proc=AutoProcessor.from_pretrained('/workspace/home/oujingfeng/project/models/Qwen2.5-VL-7B-Instruct'); ds=MultimodalEvalDataset(tok, proc, '/workspace/home/oujingfeng/project/datasets/MazePlanning-Test/updated_data.json', image_dir='/workspace/home/oujingfeng/project/datasets/MazePlanning-Test', model_name='qwen'); item=ds[0]; print('input_ids', item['input_ids'].shape); print('pixel_values', item['pixel_values'].shape); print('image_grid_thw', item['image_grid_thw'].shape)"
```

### 2. Single-sample 3dproject forward smoke test

This confirmed that one converted `3dproject` sample can pass through the patched Qwen model:

```bash
CUDA_HOME=/workspace/home/miniconda3 \
LD_LIBRARY_PATH=/workspace/home/miniconda3/lib \
PATH=/workspace/home/miniconda3/envs/sketchpad-clean/bin:/workspace/home/miniconda3/bin:$PATH \
conda run -n sketchpad-clean python -c "import torch; from transformers import AutoTokenizer, AutoProcessor; from data.dataset import MultimodalDataset; from model.uni_qwen import UniQwenForConditionalGeneration; mp='/workspace/home/oujingfeng/project/models/Qwen2.5-VL-7B-Instruct'; dp='/workspace/home/oujingfeng/project/unimrg/datasets/spatialviz/latent_sketchpad_3dproject_currentstyle_3329/train_smoke32.json'; idir='/workspace/home/oujingfeng/project/unimrg/datasets/spatialviz'; dec='/workspace/home/oujingfeng/project/models/Latent-Sketchpad.Sketch_Decoder/sketch_decoder_qwen25_vl.ckpt'; model=UniQwenForConditionalGeneration.from_pretrained(mp, torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2').to('cuda').eval(); tok=AutoTokenizer.from_pretrained(mp); proc=AutoProcessor.from_pretrained(mp); ds=MultimodalDataset(tok, proc, model, checkpoint_path=dec, feature_dim=model.config.vision_config.hidden_size, json_file=[dp], image_dir=idir, augment=False, stage1=True, model_name='qwen', image_token_index=torch.tensor(model.image_token_index), boi_id=model.config.vision_start_token_id, eoi_id=model.config.vision_end_token_id, ignore_image=True); item=ds[0]; out=model(input_ids=item['input_ids'].unsqueeze(0).to('cuda'), attention_mask=item['attention_mask'].unsqueeze(0).to('cuda'), pixel_values=item['pixel_values'].to('cuda'), image_grid_thw=item['image_grid_thw'].to('cuda'), labels=item['labels'].unsqueeze(0).to('cuda'), output_hidden_states=True, loss_type='l1'); print(out.logits.shape)"
```

### 3. Successful 2-GPU 3dproject stage1 smoke training

This is the first verified end-to-end training command that completed successfully:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
CUDA_HOME=/workspace/home/miniconda3 \
LD_LIBRARY_PATH=/workspace/home/miniconda3/lib \
PATH=/workspace/home/miniconda3/envs/sketchpad-clean/bin:/workspace/home/miniconda3/bin:$PATH \
WANDB_MODE=offline \
WANDB_DISABLED=true \
LATENT_SKETCHPAD_IMAGE_ROOT=/workspace/home/oujingfeng/project/unimrg/datasets/spatialviz \
MASTER_PORT=29623 \
/workspace/home/miniconda3/envs/sketchpad-clean/bin/python -m torch.distributed.run --nproc_per_node=2 --master_port=29623 \
  /workspace/home/oujingfeng/project/Latent-Sketchpad/train.py \
  --data_path /workspace/home/oujingfeng/project/unimrg/datasets/spatialviz/latent_sketchpad_3dproject_currentstyle_3329/train_smoke32.json \
  --decoder_path /workspace/home/oujingfeng/project/models/Latent-Sketchpad.Sketch_Decoder/sketch_decoder_qwen25_vl.ckpt \
  --image_dir /workspace/home/oujingfeng/project/unimrg/datasets/spatialviz \
  --model_path /workspace/home/oujingfeng/project/models/Qwen2.5-VL-7B-Instruct \
  --learning_rate 1e-4 --max_grad_norm 1.0 --num_train_epochs 1 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --per_device_eval_batch_size 1 \
  --weight_decay 0.01 --ds_config /workspace/home/oujingfeng/project/Latent-Sketchpad/ds_cfg.json \
  --save_steps 20 --eval_steps 10 --logging_steps 1 --resume_from_checkpoint False \
  --output_dir /workspace/home/oujingfeng/project/Latent-Sketchpad/tmp/3dproject_stage1_smoke_torchrun2_lora \
  --wandb_project latent-sketchpad-3dproject-smoke --validation_split 0.125 --augment \
  --image_loss_weight 0.0 --unfreeze-connector --stage1 --sum-loss --loss_type l1 --use-lora
```

Successful output directory from that run:

- `/workspace/home/oujingfeng/project/Latent-Sketchpad/tmp/3dproject_stage1_smoke_torchrun2_lora`

## Current status

The project is now runnable locally for `3dproject` stage1 training, with the verified path being:

- local `Qwen2.5-VL-7B-Instruct`
- local `sketch_decoder_qwen25_vl.ckpt`
- converted `3dproject` interleaved dataset
- 2-GPU `torchrun`
- DeepSpeed
- LoRA-enabled stage1 training

## Simplified local usage

After activating the conda environment:

```bash
conda activate sketchpad-clean
```

you can use the local wrapper scripts instead of retyping the full command lines.

Environment wrapper:

- `scripts/local_env.sh`

Full stage1 training wrapper:

- `scripts/train_3dproject_stage1.sh`

Example:

```bash
bash scripts/train_3dproject_stage1.sh
```

You can optionally pass a custom output directory as the first argument:

```bash
bash scripts/train_3dproject_stage1.sh /path/to/output_dir
```

Two-GPU stage1 evaluation wrapper:

- `scripts/eval_3dproject_stage1_lora.sh`
- merged-summary helper: `scripts/merge_3dproject_eval_shards.py`

## Qwen3.5 local usage

Dedicated environment wrapper:

- `scripts/local_env_qwen35.sh`

Important environment values for the Qwen3.5 route:

- conda env: `sketchpad-qwen35-cu128`
- base model: `/workspace/home/oujingfeng/project/models/Qwen3.5-4B`
- stage1 init / reuse path: `/workspace/home/oujingfeng/project/Latent-Sketchpad/outputs/3dproject_stage1_qwen35_full`
- DeepSpeed config: `/workspace/home/oujingfeng/project/LLaMA-Factory/examples/deepspeed/ds_z2_config.json`

Current Qwen3.5 wrappers:

- stage1 training: `scripts/train_3dproject_stage1_qwen35_full.sh`
- stage1 evaluation: `scripts/eval_3dproject_stage1_qwen35.sh`
- stage2 training: `scripts/train_3dproject_stage2_qwen35_full.sh`
- stage2 smoke training: `scripts/train_3dproject_stage2_qwen35_smoke.sh`
- aligner training: `scripts/train_qwen35_aligner.sh`
- aligner smoke training: `scripts/train_qwen35_aligner_smoke.sh`
- stage2 inference: `inference_qwen35_stage2.py`

Verified current launch style:

- `stage1` now runs with `DeepSpeed ZeRO-2`, not raw `torchrun`
- `stage2` wrappers have also been switched to `DeepSpeed ZeRO-2`
- long-running jobs should be launched from `tmux`, not from a foreground tool session, otherwise the launcher can receive `SIGTERM`

### Current stage1 training command (Qwen3.5, DS-ZeRO2)

```bash
tmux new-session -d -s stage1_qwen35_lfalign 'cd "/workspace/home/oujingfeng/project/Latent-Sketchpad" && env WANDB_DISABLED=false WANDB_MODE=online bash "/workspace/home/oujingfeng/project/Latent-Sketchpad/scripts/train_3dproject_stage1_qwen35_full.sh" "/workspace/home/oujingfeng/project/Latent-Sketchpad/outputs/3dproject_stage1_qwen35_full_lfalign" >> "/workspace/home/oujingfeng/project/Latent-Sketchpad/outputs/3dproject_stage1_qwen35_full_lfalign/train.log" 2>&1'
```

This run is aligned to the observed `LLaMA-Factory` full-SFT freezing strategy:

- freeze `model.visual`
- freeze `model.visual.merger`
- train the language model including `embed_tokens`

It also uses the `LLaMA-Factory` full-SFT learning rate and scheduler settings that were copied over:

- `learning_rate=5e-6`
- `gradient_accumulation_steps=8`
- `num_train_epochs=3`
- `save_steps=200`
- `logging_steps=10`
- `warmup_ratio=0.1`

### Current stage1 evaluation command (Qwen3.5)

```bash
bash /workspace/home/oujingfeng/project/Latent-Sketchpad/scripts/eval_3dproject_stage1_qwen35.sh --wait \
  /workspace/home/oujingfeng/project/Latent-Sketchpad/outputs/3dproject_stage1_qwen35_full \
  /workspace/home/oujingfeng/project/Latent-Sketchpad/outputs/3dproject_stage1_qwen35_full/test_eval_final
```

### Current stage2 training command (Qwen3.5, DS-ZeRO2)

```bash
tmux new-session -d -s stage2_qwen35 'cd "/workspace/home/oujingfeng/project/Latent-Sketchpad" && bash "/workspace/home/oujingfeng/project/Latent-Sketchpad/scripts/train_3dproject_stage2_qwen35_full.sh" >> "/workspace/home/oujingfeng/project/Latent-Sketchpad/outputs/3dproject_stage2_qwen35_full/train.log" 2>&1'
```

### Current stage2 inference command (Qwen3.5 final checkpoint)

```bash
conda run -n sketchpad-qwen35-cu128 python /workspace/home/oujingfeng/project/Latent-Sketchpad/inference_qwen35_stage2.py \
  --model_path /workspace/home/oujingfeng/project/Latent-Sketchpad/outputs/3dproject_stage2_qwen35_full \
  --decoder_path /workspace/home/oujingfeng/project/Latent-Sketchpad/outputs/qwen35_aligner_smoke/-sketch-decoder-qwen35-aligner-smoke/checkpoints/last.ckpt \
  --data_path /workspace/home/oujingfeng/project/unimrg/datasets/spatialviz/latent_sketchpad_3dproject_currentstyle_3329/test.json \
  --output_dir /workspace/home/oujingfeng/project/Latent-Sketchpad/outputs/qwen35_stage2_infer_final_1024_debug \
  --max_new_tokens 1024 \
  --sample_index 0
```

Default evaluation mode in this wrapper:

- `4` shards total
- `GPU 0` runs `shard_0` and `shard_1`
- `GPU 1` runs `shard_2` and `shard_3`
- each process keeps `batch_size=1`

Example:

```bash
bash scripts/eval_3dproject_stage1_lora.sh
```

Optional arguments:

- first arg: checkpoint path, defaults to `outputs/3dproject_stage1_full_lora/checkpoint-200`
- second arg: output root for shard results

Example:

```bash
bash scripts/eval_3dproject_stage1_lora.sh \
  /workspace/home/oujingfeng/project/Latent-Sketchpad/outputs/3dproject_stage1_full_lora/checkpoint-200 \
  /workspace/home/oujingfeng/project/Latent-Sketchpad/outputs/3dproject_stage1_full_lora/test_eval_checkpoint200
```
