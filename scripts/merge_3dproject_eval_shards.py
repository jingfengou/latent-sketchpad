import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge sharded 3dproject evaluation outputs")
    parser.add_argument("--input-root", required=True, help="Directory containing shard_* outputs")
    args = parser.parse_args()

    root = Path(args.input_root)
    shard_dirs = sorted([path for path in root.iterdir() if path.is_dir() and path.name.startswith("shard_")])
    if not shard_dirs:
        raise FileNotFoundError(f"No shard_* directories found under {root}")

    shard_summaries = []
    merged_predictions = []
    for shard_dir in shard_dirs:
        summary_path = shard_dir / "summary.json"
        predictions_path = shard_dir / "predictions.json"
        if not summary_path.exists() or not predictions_path.exists():
            raise FileNotFoundError(f"Missing summary or predictions in {shard_dir}")

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
        shard_summaries.append(summary)
        merged_predictions.extend(predictions)

    merged_predictions.sort(key=lambda row: row.get("global_sample_idx", row.get("sample_idx", -1)))

    total_samples = sum(item["num_samples"] for item in shard_summaries)
    total_correct = sum(item["num_correct"] for item in shard_summaries)
    total_with_answer = sum(item["pred_with_answer_tag"] for item in shard_summaries)

    merged_summary = {
        "base_model": shard_summaries[0].get("base_model", shard_summaries[0].get("model_path")),
        "adapter_path": shard_summaries[0].get("adapter_path"),
        "num_samples": total_samples,
        "num_correct": total_correct,
        "accuracy": total_correct / total_samples,
        "pred_with_answer_tag": total_with_answer,
        "max_new_tokens": shard_summaries[0]["max_new_tokens"],
        "num_shards": len(shard_summaries),
        "shards": shard_summaries,
    }

    (root / "summary.json").write_text(json.dumps(merged_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (root / "predictions.json").write_text(json.dumps(merged_predictions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(merged_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
