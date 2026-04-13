import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge sharded evaluation outputs.")
    parser.add_argument("--shards-root", required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    shards_root = Path(args.shards_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_predictions = []
    total_samples = 0
    total_correct = 0
    pred_with_answer_tag = 0
    model_path = None
    max_new_tokens = None

    for shard_id in range(args.num_shards):
        shard_dir = shards_root / f"shard_{shard_id}_of_{args.num_shards}"
        summary = json.loads((shard_dir / "summary.json").read_text(encoding="utf-8"))
        predictions = json.loads((shard_dir / "predictions.json").read_text(encoding="utf-8"))
        all_predictions.extend(predictions)
        total_samples += summary["num_samples"]
        total_correct += summary["num_correct"]
        pred_with_answer_tag += summary.get("pred_with_answer_tag", 0)
        model_path = summary.get("model_path", model_path)
        max_new_tokens = summary.get("max_new_tokens", max_new_tokens)

    all_predictions.sort(key=lambda item: item.get("global_sample_idx", item.get("sample_idx", -1)))
    merged_summary = {
        "model_path": model_path,
        "num_samples": total_samples,
        "num_correct": total_correct,
        "accuracy": total_correct / total_samples if total_samples else 0.0,
        "pred_with_answer_tag": pred_with_answer_tag,
        "max_new_tokens": max_new_tokens,
        "num_shards": args.num_shards,
    }

    (output_dir / "predictions.json").write_text(
        json.dumps(all_predictions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(merged_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(merged_summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
