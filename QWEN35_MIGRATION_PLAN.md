# Qwen3.5-4B Migration Plan

## Goal

Bring up a separate `Qwen3.5-4B` codepath for `Latent-Sketchpad` without modifying the existing `Qwen2.5-VL` route.

## Scope

1. Create a dedicated `sketchpad-qwen35-cu128` environment aligned with `llamafactory` (`Python 3.11`, `torch 2.7.1+cu128`).
2. Add a new `Qwen3.5` model wrapper instead of editing `model/uni_qwen.py`.
3. Add a new dataset/collator that preserves `mm_token_type_ids`, which `Qwen3.5` requires for multimodal RoPE.
4. Add new `train/inference/evaluate` entrypoints for the `Qwen3.5` route.
5. Add new wrapper scripts for `3dproject` stage1 training and evaluation.
6. Smoke-test each step in the `qwen35` environment.

## Known Differences vs Qwen2.5

1. The model type is `qwen3_5`, so the old `transformers==4.54.0` environment cannot load it.
2. The model structure keeps the vision stack under `model.visual`, while embeddings remain at `model.language_model.embed_tokens`.
3. `Qwen3.5` requires `mm_token_type_ids` whenever multimodal inputs are passed.
4. The visual merger structure is `model.visual.merger.{norm,linear_fc1,linear_fc2}` instead of the `Qwen2.5` `merger.ln_q/mlp` layout.
5. The compatible FlashAttention route is `flash-attn==2.8.3` with the `torch 2.7 + cp311 + cxx11abiTRUE` release wheel, not `flash-attn-4`.
6. The `flash_attention_2` path requires a local patch in `transformers/modeling_flash_attention_utils.py` so 3D `position_ids` from Qwen3.5 are not misinterpreted as packed sequences.
7. `stage2` should initialize from the completed `stage1` checkpoint/output directory rather than the raw base model.

## Bring-up Strategy

1. First target `stage1` text alignment and full-finetune evaluation.
2. Keep the `Sketch Decoder` path available in the new code, but do not assume `stage2` is ready until the new visual regression path is validated.
3. Fail clearly for unsupported `Qwen3.5` multimodal generation paths instead of silently reusing `Qwen2.5` assumptions.

## Current Status

### Completed

1. A separate `Qwen3.5-4B` codepath exists for stage1 and stage2.
2. The runtime environment is aligned with `llamafactory` on `Python 3.11`, `torch 2.7.1+cu128`, and `flash-attn 2.8.3`.
3. `transformers` was locally patched so `flash_attention_2` does not mis-handle `Qwen3.5` 3D `position_ids`.
4. `stage1` test evaluation succeeded once and produced `191 / 333 = 57.36%` on `outputs/3dproject_stage1_qwen35_full/test_eval_final`.
5. `stage2` stable training completed once end-to-end at `outputs/3dproject_stage2_qwen35_full`.
6. A smoke `Qwen3.5` aligner was trained successfully and saved to `outputs/qwen35_aligner_smoke/-sketch-decoder-qwen35-aligner-smoke/checkpoints/last.ckpt`.

### Main Problems Encountered and Fixes

1. `flash_attention_2` crashed with `illegal memory access`.
   Fix: patch `transformers/modeling_flash_attention_utils.py` to return `False` for packed-sequence checks when `position_ids.dim() > 2`.

2. `stage1` and `stage2` failed under DDP with unused parameters.
   Fix: freeze unused parts of `regression_head` when that branch is not active.

3. Foreground launcher processes were killed by the tool/session timeout.
   Fix: long-running jobs should be launched from `tmux`.

4. Official decoder training expected assets that were not present, including `vae-encoder.pth`.
   Fix: initialize `VaeEncoder` from the public `sdxl-vae` encoder when the file is missing.

5. Official decoder training assumed installed `torchscale` and optional `xformers`.
   Fix: add local vendored `torchscale` paths and fallback code so decoder smoke training can run without `xformers` changing the torch stack.

6. `Qwen2.5` decoder checkpoint could not be loaded as a `Qwen3.5` aligner.
   Fix: decouple stage2 training from aligner loading and train a smoke `Qwen3.5` aligner separately.

### What Still Needs To Be Done

1. Re-run `stage1` with the updated settings:
   - `DeepSpeed ZeRO-2`
   - `LLaMA-Factory`-aligned trainable parameter set
   - `vision_start` supervision kept in stage1 labels

2. Check whether the new `stage1` model starts generating `vision_start_token_id` during inference.

3. Extend the stable `stage2` route so the model not only learns image regression loss, but also reliably emits the `vision_start` special token that switches generation into image mode.

4. Train a non-smoke `Qwen3.5` aligner and use it in stage2 inference.

5. After the stable route is verified, continue with the full perceiver-based stage2 implementation.
