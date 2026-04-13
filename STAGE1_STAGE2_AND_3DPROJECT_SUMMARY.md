# Stage1 / Stage2 and 3DProject Summary

This note summarizes:

- what `stage1` and `stage2` actually train in `Latent-Sketchpad`
- where the supervision targets come from
- what local adaptation work was completed for the `3dproject` dataset

## 1. High-level distinction

`Latent-Sketchpad` is not a plain VLM SFT setup.

It modifies the base VLM so that, in addition to normal text generation, the model can also predict intermediate visual features through a `regression_head` and then decode them with the pretrained Sketch Decoder.

In practice, the two training stages are split as follows:

- `stage1`: task/text alignment stage
- `stage2`: visual latent / sketch prediction stage

## 2. What stage1 trains

### Goal

`stage1` mainly trains the model to:

- follow the task format
- produce the long reasoning text in the expected interleaved format
- produce the final answer in the required output pattern

It is the part that makes the model behave like a task-specific multimodal reasoner at the text level.

### How the code behaves

In `train.py`:

- `--stage1` marks the run as stage1
- if the output path does **not** contain `stage2`, then `ignore_image = True`
- in the dataset, stage1 sometimes removes intermediate image supervision from the text sequence and keeps a more text-oriented version of the sample

In `data/dataset.py`:

- when `self.stage1` is enabled, some samples are converted into a more text-only form
- extra `<image>` placeholders in the label text can be removed
- `label_img` can be emptied for those stage1 text-only cases

This means stage1 can still see the task image context, but it does **not** focus on learning to generate intermediate images correctly.

### What supervision it uses

Stage1 mainly uses:

- `label_text`
- the final answer tokens inside `label_text`

For the current local runs, stage1 was launched with:

- `--image_loss_weight 0.0`
- `--stage1`
- `--use-lora`

So the practical role of stage1 in this workflow is:

- adapt the base model to the `3dproject` reasoning/output style
- learn the answering format
- prepare for later visual-latent learning in stage2

## 3. What stage2 trains

### Goal

`stage2` is the stage that trains the model to predict intermediate visual states.

More specifically, it trains the model to:

- use the previous text+image context
- identify positions where an image should be produced
- regress the visual representation of the target intermediate image

### How the code behaves

In `train.py`:

- if `"stage2" in args.output_dir`, then `ignore_image = False`
- the stage2 convention is to use settings such as:
  - `--freeze-backbone`
  - `--text_loss_weight 0.0`

This means stage2 is intended to:

- stop focusing on text learning
- keep most of the backbone fixed
- train the visual regression pathway

### Where the GT comes from

The GT for stage2 does **not** come from an external latent file.

It comes from the dataset's real intermediate process images:

- `label_img`

For each training sample, `data/dataset.py` loads:

- `input_img`: input images
- `label_img`: target intermediate images

Then in `model/uni_qwen.py`:

- `label_pixel_values` is built from those `label_img` images
- `target_vit = self.get_vit_features(...)`
- `target_features = self.visual(...)`

So the supervision target is:

- the visual features extracted from the **real intermediate GT images**

This is the key point:

- GT images come from `label_img`
- GT visual targets are obtained by passing those GT images through the model's own visual encoder

### What loss is optimized

In `uni_qwen.py`, image loss is computed by comparing:

- predicted visual features from `regression_head`
- target visual features from GT intermediate images

Supported loss types include:

- `mse`
- `l1`
- `cosine`

The local training runs used:

- `loss_type: l1`

## 4. How this maps to 3dproject

For the converted `3dproject` dataset:

- `input_img` is the original composite problem image
- `label_text` is the interleaved reasoning text
- `label_img` is the sequence of intermediate process images such as `target_progress_*.png`

This makes `3dproject` a valid fit for `Latent-Sketchpad` because it already contains:

- task input image
- reasoning text
- intermediate process images
- final answer

In other words:

- stage1 uses the `3dproject` reasoning/output format to align the model to the task
- stage2 can use the `3dproject` intermediate process images as direct visual supervision

## 5. Work completed in this local adaptation

### Models and checkpoints

Downloaded and prepared:

- base model: `/workspace/home/oujingfeng/project/models/Qwen2.5-VL-7B-Instruct`
- sketch decoder: `/workspace/home/oujingfeng/project/models/Latent-Sketchpad.Sketch_Decoder/sketch_decoder_qwen25_vl.ckpt`

### Data conversion

Converted the existing `3dproject` interleaved dataset into the format expected by `Latent-Sketchpad`.

Created conversion script:

- `unimrg/scripts/convert_3dproject_to_latent_sketchpad.py`

Converted dataset output:

- `train.json` (`2663` samples)
- `val.json` (`333` samples)
- `test.json` (`333` samples)

Location:

- `/workspace/home/oujingfeng/project/unimrg/datasets/spatialviz/latent_sketchpad_3dproject_currentstyle_3329/`

### Local code fixes

Patched local execution so the project is runnable in this workspace:

- `train.py`, `inference.py`, `evaluate.py` now use local env-backed paths instead of `/path/to/...`
- `train.py` now respects `WANDB_DISABLED=true`
- `train.py` supports an optional dedicated `--eval_data_path`
- LoRA flow was fixed so it no longer immediately calls `merge_and_unload()` after `get_peft_model(...)`
- helper scripts were added for local environment, training, and evaluation

### Environment fixes

Created and used conda environment:

- `sketchpad-clean`

Resolved practical environment issues involving:

- `flash-attn`
- `torchrun`
- `CUDA_HOME`
- DeepSpeed startup
- local path injection

### Training validation work

Verified:

- official MazePlanning public inference path works locally
- `3dproject` single-sample forward pass works
- `3dproject` stage1 two-GPU training works with `torchrun + DeepSpeed + LoRA`

### Stage1 evaluation status

An evaluated stage1 run at `checkpoint-200` produced test accuracy around:

- `125 / 333 = 37.54%`

This should be interpreted as a stage1 text/task-alignment result, not as the final expected result of the full `Latent-Sketchpad` pipeline.

## 6. Current training direction

The current active direction is:

- rerun `stage1`
- use the dedicated `val.json` instead of splitting train
- use effective batch size `16`
- keep `wandb` online
- after training finishes, evaluate the final checkpoint on the test set first
- if the final checkpoint is not best, evaluate earlier checkpoints

## 7. Practical takeaway

The most important conceptual summary is:

- `stage1` is mainly for task/text adaptation
- `stage2` is the stage that actually learns intermediate visual prediction
- stage2 GT comes from the dataset's real intermediate images in `label_img`

So if the goal is to test whether `Latent-Sketchpad` helps by introducing visual intermediate reasoning, the real comparison should not stop at stage1 alone. The meaningful comparison is:

- base / prior VLM SFT baseline
- stage1 result
- stage2 result
