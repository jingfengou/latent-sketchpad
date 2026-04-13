import argparse
import json
import os
from pathlib import Path

import torch
import torchvision.transforms as transforms
from transformers import AutoProcessor, AutoTokenizer, set_seed

from data.dataset import MultimodalEvalDataset, decode_img, load_models
from model.uni_qwen35 import UniQwen35ForConditionalGeneration
from qwen35_utils import resolve_qwen35_attn_implementation


LOCAL_IMAGE_ROOT = os.environ.get(
    "LATENT_SKETCHPAD_IMAGE_ROOT",
    "/workspace/home/oujingfeng/project/unimrg/datasets/spatialviz",
)


def main():
    parser = argparse.ArgumentParser(description="Qwen3.5 stage2 inference")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--decoder_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", default="evaluation_results_qwen35_stage2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=2500)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--use_perceiver", action="store_true", default=False)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    set_seed(args.seed)
    os.environ["GENERATION_TYPE"] = "multimodal"
    os.environ["LATENT_SKETCHPAD_QWEN35_USE_PERCEIVER"] = "1" if args.use_perceiver else "0"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "generated_images"
    images_dir.mkdir(parents=True, exist_ok=True)

    model = UniQwen35ForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=resolve_qwen35_attn_implementation(),
    ).to(args.device)
    model.eval()
    aligner_net, vae_ref = load_models(
        model.device,
        args.decoder_path,
        feature_dim=model.config.vision_config.hidden_size,
        strict=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    processor = AutoProcessor.from_pretrained(args.model_path)
    dataset = MultimodalEvalDataset(tokenizer, processor, args.data_path, image_dir=LOCAL_IMAGE_ROOT, is_original_model=False, model_name="qwen")
    batch = dataset[args.sample_index]

    with torch.inference_mode():
        generated_ids, image_embeds, image_vit_feats = model.generate(
            input_ids=batch["input_ids"].unsqueeze(0).to(args.device),
            attention_mask=batch["attention_mask"].unsqueeze(0).to(args.device),
            pixel_values=batch["pixel_values"].unsqueeze(0).to(args.device),
            image_grid_thw=batch["image_grid_thw"].unsqueeze(0).to(args.device),
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
        )

    sequence = generated_ids[0].detach().cpu().numpy()
    input_length = batch["input_ids"].shape[0]
    generated_suffix = sequence[input_length:].tolist()
    response = tokenizer.decode(generated_suffix, skip_special_tokens=True)
    raw_response = tokenizer.decode(generated_suffix, skip_special_tokens=False)
    image_token_id = model.config.image_token_id
    vision_start_id = model.config.vision_start_token_id
    vision_end_id = model.config.vision_end_token_id
    result = {
        "sample_index": args.sample_index,
        "response": response,
        "raw_response": raw_response,
        "generated_image_count": 0 if image_vit_feats is None else int(image_vit_feats.shape[0]),
        "generated_image_token_count": int(sum(token == image_token_id for token in generated_suffix)),
        "generated_vision_start_count": int(sum(token == vision_start_id for token in generated_suffix)),
        "generated_vision_end_count": int(sum(token == vision_end_id for token in generated_suffix)),
    }

    if image_vit_feats is not None:
        for index in range(image_vit_feats.size(0)):
            decoded = decode_img(image_vit_feats[index].to(args.device), aligner_net, vae_ref, args.device)
            transforms.ToPILImage()(decoded).save(images_dir / f"output_img-{index}.png")

    (output_dir / "response.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
