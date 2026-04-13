import argparse
import json
import re
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.dataset_qwen35 import MultimodalQwen35EvalDataset
from model.uni_qwen35 import UniQwen35ForConditionalGeneration
from qwen35_utils import resolve_qwen35_attn_implementation


ANSWER_RE = re.compile(r"<answer>\s*([A-D])\s*</answer>", re.IGNORECASE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 3dproject stage1 LoRA checkpoint with Qwen3.5")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = json.loads(Path(args.data_path).read_text(encoding="utf-8"))
    if args.num_shards > 1:
        shard_size = (len(rows) + args.num_shards - 1) // args.num_shards
        start = args.shard_id * shard_size
        end = min((args.shard_id + 1) * shard_size, len(rows))
        rows = rows[start:end]
        sharded_data_path = out_dir / f"data_shard_{args.shard_id}_of_{args.num_shards}.json"
        sharded_data_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        data_path = str(sharded_data_path)
    else:
        start = 0
        data_path = args.data_path

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    processor = AutoProcessor.from_pretrained(args.base_model)
    dataset = MultimodalQwen35EvalDataset(tokenizer, processor, data_path, image_dir=args.image_dir)

    model = UniQwen35ForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation=resolve_qwen35_attn_implementation(),
    ).to("cuda")
    model = PeftModel.from_pretrained(model, args.adapter_path).to("cuda").eval()

    predictions = []
    num_correct = 0
    pred_with_answer_tag = 0

    for i in range(len(dataset)):
        item = dataset[i]
        with torch.inference_mode():
            generate_kwargs = dict(
                input_ids=item["input_ids"].unsqueeze(0).to("cuda"),
                attention_mask=item["attention_mask"].unsqueeze(0).to("cuda"),
                pixel_values=item["pixel_values"].to("cuda"),
                image_grid_thw=item["image_grid_thw"].to("cuda"),
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
            )
            if "mm_token_type_ids" in item:
                generate_kwargs["mm_token_type_ids"] = item["mm_token_type_ids"].unsqueeze(0).to("cuda")
            generated = model.generate(**generate_kwargs)

        sequence = generated[0].detach().cpu()
        response = tokenizer.decode(sequence[item["input_ids"].shape[0] :].tolist(), skip_special_tokens=True)
        gold = rows[i].get("answer")
        match = ANSWER_RE.search(response)
        pred = match.group(1).upper() if match else None
        if pred is not None:
            pred_with_answer_tag += 1
        correct = pred == gold
        if correct:
            num_correct += 1

        predictions.append(
            {
                "sample_idx": i,
                "global_sample_idx": start + i,
                "sample_id": rows[i].get("sample_id"),
                "gold": gold,
                "pred": pred,
                "correct": correct,
                "output_text": response,
            }
        )

    summary = {
        "base_model": args.base_model,
        "adapter_path": args.adapter_path,
        "num_samples": len(dataset),
        "num_correct": num_correct,
        "accuracy": num_correct / len(dataset),
        "pred_with_answer_tag": pred_with_answer_tag,
        "max_new_tokens": args.max_new_tokens,
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "start_index": start,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "predictions.json").write_text(json.dumps(predictions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
