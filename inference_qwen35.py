import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen3.5 Latent-Sketchpad stage1 inference")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", default="evaluation_results_qwen35")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--allow_image_tokens", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    processor = AutoProcessor.from_pretrained(args.model_path)
    dataset = MultimodalQwen35EvalDataset(tokenizer, processor, args.data_path, args.image_dir)
    item = dataset[args.sample_index]

    model = UniQwen35ForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=resolve_qwen35_attn_implementation(),
    ).to("cuda")
    model.eval()

    suppress_tokens = None
    if not args.allow_image_tokens:
        suppress_tokens = [
            model.config.vision_start_token_id,
            model.config.image_token_id,
            model.config.vision_end_token_id,
        ]
        if getattr(model.config, "video_token_id", None) is not None:
            suppress_tokens.append(model.config.video_token_id)

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
    result = {
        "sample_index": args.sample_index,
        "response": response,
    }
    (output_dir / "response.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
