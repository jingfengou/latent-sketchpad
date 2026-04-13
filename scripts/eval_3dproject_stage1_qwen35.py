import argparse
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.dataset_qwen35 import MultimodalQwen35EvalDataset
from model.uni_qwen35 import UniQwen35ForConditionalGeneration
from qwen35_utils import resolve_qwen35_attn_implementation


def _compile_patterns(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(pattern, re.IGNORECASE | re.DOTALL) for pattern in patterns]


OPTION_PATTERNS = _compile_patterns([
    r"<answer>\s*(?P<value>.*?)\s*</answer>",
    r"<answer>\s*option\s+(?P<value>[A-D])(?=answer>)",
    r"</answer>\s*(?P<value>[A-D])\b",
    r"(?:final|correct\s+)?answer\s*(?:is|:)\s*(?:option\s*)?(?P<value>[A-D])\b",
    r"correct\s+path\s*(?:is|:)?\s*(?:option\s*)?(?P<value>[A-D])\b",
    r"correct\s+choice\s*(?:is|:)?\s*(?:option\s*)?(?P<value>[A-D])\b",
    r"option\s+(?P<value>[A-D])\b",
    r"choose\s+(?P<value>[A-D])\b",
    r"\\{1,2}boxed\{(?:\\text\{)?(?P<value>[A-D])",
])

STOP_PATTERNS = _compile_patterns([
    r"<answer>\s*(?P<value>.*?)\s*</answer>",
    r"<answer>\s*option\s+(?P<value>[A-D])(?=answer>)",
    r"</answer>\s*(?P<value>[A-D])\b",
    r"(?:final|correct\s+)?answer\s*(?:is|:)\s*(?:option\s*)?(?P<value>[A-D])\b",
    r"correct\s+path\s*(?:is|:)?\s*(?:option\s*)?(?P<value>[A-D])\b",
    r"correct\s+choice\s*(?:is|:)?\s*(?:option\s*)?(?P<value>[A-D])\b",
])

STRICT_ANSWER_RE = re.compile(r"<answer>\s*([A-D])\s*</answer>", re.IGNORECASE | re.DOTALL)


def extract_prediction(response: str) -> tuple[str | None, bool, str | None]:
    for pattern in OPTION_PATTERNS:
        match = pattern.search(response)
        if not match:
            continue
        raw_value = match.group("value").strip()
        letter_match = re.search(r"\b([A-D])\b", raw_value, re.IGNORECASE)
        if letter_match:
            return letter_match.group(1).upper(), True, pattern.pattern
    return None, False, None


def matches_stop_pattern(response: str) -> bool:
    for pattern in STOP_PATTERNS:
        match = pattern.search(response)
        if not match:
            continue
        raw_value = match.group("value").strip()
        if re.search(r"\b([A-D])\b", raw_value, re.IGNORECASE):
            return True
    return False


class AnswerPatternStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, prompt_length: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        response_ids = input_ids[0, self.prompt_length :].tolist()
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=False)
        return matches_stop_pattern(response_text)


def write_outputs(out_dir: Path, predictions, summary) -> None:
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "predictions.json").write_text(
        json.dumps(predictions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 3dproject stage1 full checkpoint with Qwen3.5")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--allow-image-tokens", action="store_true")
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    processor = AutoProcessor.from_pretrained(args.model_path)
    dataset = MultimodalQwen35EvalDataset(tokenizer, processor, data_path, image_dir=args.image_dir)

    model = UniQwen35ForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=resolve_qwen35_attn_implementation(),
        ignore_mismatched_sizes=True,
    ).to("cuda").eval()

    suppress_tokens = None
    if not args.allow_image_tokens:
        suppress_tokens = [
            model.config.vision_start_token_id,
            model.config.image_token_id,
            model.config.vision_end_token_id,
        ]
        if getattr(model.config, "video_token_id", None) is not None:
            suppress_tokens.append(model.config.video_token_id)

    predictions = []
    num_correct = 0
    pred_with_answer_tag = 0

    for i in range(len(dataset)):
        item = dataset[i]
        with torch.inference_mode():
            stopping_criteria = StoppingCriteriaList(
                [AnswerPatternStoppingCriteria(tokenizer, prompt_length=item["input_ids"].shape[0])]
            )
            extra_generate_kwargs = {}
            if suppress_tokens is not None:
                extra_generate_kwargs["suppress_tokens"] = suppress_tokens
            if "mm_token_type_ids" in item:
                extra_generate_kwargs["mm_token_type_ids"] = item["mm_token_type_ids"].unsqueeze(0).to("cuda")

            generated = model.generate(
                input_ids=item["input_ids"].unsqueeze(0).to("cuda"),
                attention_mask=item["attention_mask"].unsqueeze(0).to("cuda"),
                pixel_values=item["pixel_values"].to("cuda"),
                image_grid_thw=item["image_grid_thw"].to("cuda"),
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                stopping_criteria=stopping_criteria,
                **extra_generate_kwargs,
            )

        sequence = generated[0].detach().cpu()
        response = tokenizer.decode(sequence[item["input_ids"].shape[0] :].tolist(), skip_special_tokens=True)
        gold = rows[i].get("answer")
        strict_match = STRICT_ANSWER_RE.search(response)
        strict_pred = strict_match.group(1).upper() if strict_match else None
        pred, has_answer_pattern, matched_pattern = extract_prediction(response)
        if strict_pred is not None:
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
                "strict_pred": strict_pred,
                "matched_pattern": matched_pattern,
                "correct": correct,
                "has_answer_pattern": has_answer_pattern,
                "output_text": response,
            }
        )

        summary = {
            "model_path": args.model_path,
            "num_samples": len(dataset),
            "num_completed": len(predictions),
            "num_correct": num_correct,
            "accuracy": num_correct / len(predictions) if predictions else 0.0,
            "pred_with_answer_tag": pred_with_answer_tag,
            "max_new_tokens": args.max_new_tokens,
            "num_shards": args.num_shards,
            "shard_id": args.shard_id,
            "start_index": start,
            "status": "running",
        }
        write_outputs(out_dir, predictions, summary)

    summary = {
        "model_path": args.model_path,
        "num_samples": len(dataset),
        "num_completed": len(predictions),
        "num_correct": num_correct,
        "accuracy": num_correct / len(dataset) if len(dataset) else 0.0,
        "pred_with_answer_tag": pred_with_answer_tag,
        "max_new_tokens": args.max_new_tokens,
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "start_index": start,
        "status": "completed",
    }

    write_outputs(out_dir, predictions, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
