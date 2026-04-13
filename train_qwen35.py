import argparse
import os
import uuid

import torch
import torch.distributed as dist
import wandb
from torch.utils.data import random_split
from transformers import AutoProcessor, AutoTokenizer, TrainingArguments, set_seed

from data.dataset_qwen35 import DataCollatorForQwen35Inputs, MultimodalQwen35Dataset
from gen_utils import get_last_checkpoint
from model.uni_qwen35 import UniQwen35ForConditionalGeneration
from multimodal_trainer import MultimodalTrainer
from qwen35_utils import resolve_qwen35_attn_implementation


LOCAL_QWEN35_PATH = os.environ.get(
    "LATENT_SKETCHPAD_QWEN35_PATH",
    "../models/Qwen3.5-4B",
)


def count_trainable_params(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3.5-4B on Latent-Sketchpad data")
    parser.add_argument("--model_path", type=str, default=LOCAL_QWEN35_PATH)
    parser.add_argument("--data_path", type=str, required=True, nargs="+")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--eval_data_path", type=str, nargs="+", default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs/qwen35_stage1")
    parser.add_argument("--ds_config", type=str, default="")
    parser.add_argument("--validation_split", type=float, default=0.02)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--resume_from_checkpoint", type=bool, default=False)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--dataloader_pin_memory", action="store_true", default=False)
    parser.add_argument("--no_gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--ddp_timeout", type=int, default=180000000)
    parser.add_argument("--wandb_project", type=str, default="latent-sketchpad-qwen35")
    parser.add_argument("--text_embed", action="store_true", default=False)
    parser.add_argument("--image_loss_weight", type=float, default=0.0)
    parser.add_argument("--text_loss_weight", type=float, default=1.0)
    parser.add_argument("--sum-loss", action="store_true", default=False)
    parser.add_argument("--disable_eval", action="store_true", default=False)
    parser.add_argument("--use-lora", action="store_true", default=False)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--loss_type", type=str, default="mse")
    parser.add_argument("--freeze-backbone", action="store_true", default=False)
    parser.add_argument("--unfreeze-connector", action="store_true", default=False)
    parser.add_argument("--stage1", action="store_true", default=False)
    args, extra_args = parser.parse_known_args()
    if extra_args:
        print(f"Extra arguments provided: {extra_args}")

    torch.manual_seed(args.seed)
    set_seed(args.seed)
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    global_rank = None
    if args.local_rank is not None or local_rank != -1:
        if args.local_rank is None:
            args.local_rank = local_rank
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")
        global_rank = dist.get_rank()
        print(f"[Rank {global_rank}] Distributed training initialized. World size: {dist.get_world_size()}")
    else:
        print("Running in single-process mode.")

    if global_rank in {None, 0} and os.environ.get("WANDB_DISABLED", "").lower() not in {"true", "1", "yes"}:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        run_name = f"qwen35-{uuid.uuid4().hex[:6]}"
        wandb.init(project=args.wandb_project, config=vars(args), name=run_name)
        print(f"Initialized wandb run with name: {run_name}")

    model = UniQwen35ForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=resolve_qwen35_attn_implementation(),
    ).to("cuda")

    for parameter in model.parameters():
        parameter.requires_grad = True

    if not args.text_embed and not args.stage1:
        for parameter in model.model.language_model.embed_tokens.parameters():
            parameter.requires_grad = False

    for parameter in model.model.visual.parameters():
        parameter.requires_grad = False

    if (not args.unfreeze_connector) or args.stage1:
        for parameter in model.model.visual.merger.parameters():
            parameter.requires_grad = False

    if args.freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    for parameter in model.regression_head.parameters():
        parameter.requires_grad = True

    if args.use_lora:
        from peft import LoraConfig, get_peft_model

        target_keys = {"q_proj", "k_proj", "v_proj", "o_proj"}
        lora_target_modules = []
        for name, _module in model.named_modules():
            if name.split(".")[-1] in target_keys and "model.visual" not in name and "regression_head" not in name:
                lora_target_modules.append(name)

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            modules_to_save=["lm_head", "model.visual.merger", "regression_head"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        print("LoRA fine-tuning is enabled.")

    if args.image_loss_weight == 0.0:
        for parameter in model.regression_head.parameters():
            parameter.requires_grad = False

    trainable = count_trainable_params(model)
    print(f"Total trainable parameters: {trainable:,}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    processor = AutoProcessor.from_pretrained(args.model_path)
    ignore_image = args.image_loss_weight == 0.0
    train_dataset = MultimodalQwen35Dataset(
        tokenizer=tokenizer,
        processor=processor,
        json_file=args.data_path,
        image_dir=args.image_dir,
        stage1=args.stage1,
        ignore_image=ignore_image,
    )

    if args.eval_data_path:
        eval_dataset = MultimodalQwen35Dataset(
            tokenizer=tokenizer,
            processor=processor,
            json_file=args.eval_data_path,
            image_dir=args.image_dir,
            stage1=args.stage1,
            ignore_image=ignore_image,
        )
    else:
        eval_size = int(len(train_dataset) * args.validation_split)
        train_size = len(train_dataset) - eval_size
        if args.disable_eval:
            eval_dataset = None
        else:
            if train_size <= 0 or eval_size <= 0:
                raise ValueError("Invalid split: adjust --validation_split or provide --eval_data_path")
            train_dataset, eval_dataset = random_split(train_dataset, [train_size, eval_size])

    report_to = [] if os.environ.get("WANDB_DISABLED", "").lower() in {"true", "1", "yes"} else ["wandb"]
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=10,
        bf16=True,
        eval_strategy="no" if args.disable_eval else "steps",
        eval_steps=None if args.disable_eval else args.eval_steps,
        local_rank=args.local_rank,
        report_to=report_to,
        deepspeed=args.ds_config or None,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        max_grad_norm=args.max_grad_norm,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        load_best_model_at_end=not args.disable_eval,
        greater_is_better=False,
        metric_for_best_model=None if args.disable_eval else "eval_loss",
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        batch_eval_metrics=True,
        seed=args.seed,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=args.dataloader_pin_memory,
        ddp_timeout=args.ddp_timeout,
    )

    last_checkpoint = get_last_checkpoint(args.output_dir) if os.path.isdir(args.output_dir) else None
    data_collator = DataCollatorForQwen35Inputs(tokenizer.pad_token_id)
    trainer = MultimodalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None if args.disable_eval else eval_dataset,
        processing_class=processor,
        data_collator=data_collator,
        image_loss_weight=args.image_loss_weight,
        text_loss_weight=args.text_loss_weight,
        sum_loss=args.sum_loss,
        image_token_index=model.config.image_token_id,
        loss_type=args.loss_type,
        boi_id=model.config.vision_start_token_id,
        eoi_id=model.config.vision_end_token_id,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint and last_checkpoint)
    trainer.save_model()
    trainer.save_state()
    if global_rank in {None, 0} and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
