import argparse
import json
from pathlib import Path

from transformers import AutoProcessor, AutoTokenizer


def percentile(values, q):
    if not values:
        return 0
    values = sorted(values)
    idx = int(round((len(values) - 1) * q))
    return values[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description='Analyze token lengths for Qwen3.5 merged dataset.')
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--output-path', required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    processor = AutoProcessor.from_pretrained(args.model_path)

    rows = json.loads(Path(args.data_path).read_text(encoding='utf-8'))
    lengths = []
    by_task = {}
    for row in rows:
        text = (row['input_text'] + row['label_text']).replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
        encoded = processor.tokenizer(text, return_tensors=None)
        length = len(encoded['input_ids'])
        lengths.append(length)
        stats = by_task.setdefault(row.get('task_name', 'unknown'), [])
        stats.append(length)

    summary = {
        'num_samples': len(lengths),
        'min': min(lengths) if lengths else 0,
        'max': max(lengths) if lengths else 0,
        'p50': percentile(lengths, 0.50),
        'p90': percentile(lengths, 0.90),
        'p95': percentile(lengths, 0.95),
        'p99': percentile(lengths, 0.99),
        'by_task': {
            task: {
                'num_samples': len(vals),
                'min': min(vals),
                'max': max(vals),
                'p50': percentile(vals, 0.50),
                'p90': percentile(vals, 0.90),
                'p95': percentile(vals, 0.95),
                'p99': percentile(vals, 0.99),
            }
            for task, vals in sorted(by_task.items())
        },
    }

    Path(args.output_path).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
