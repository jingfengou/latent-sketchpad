import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description='Summarize prediction accuracy by task_name.')
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--output-path', required=True)
    args = parser.parse_args()

    predictions = json.loads(Path(args.predictions).read_text(encoding='utf-8'))
    rows = json.loads(Path(args.data_path).read_text(encoding='utf-8'))

    counts = defaultdict(lambda: {'num_samples': 0, 'num_correct': 0})
    for pred in predictions:
        idx = pred['global_sample_idx'] if 'global_sample_idx' in pred else pred['sample_idx']
        task_name = rows[idx].get('task_name', 'unknown')
        counts[task_name]['num_samples'] += 1
        counts[task_name]['num_correct'] += int(bool(pred.get('correct')))

    summary = {}
    total = {'num_samples': 0, 'num_correct': 0}
    for task_name, stats in sorted(counts.items()):
        total['num_samples'] += stats['num_samples']
        total['num_correct'] += stats['num_correct']
        summary[task_name] = {
            **stats,
            'accuracy': stats['num_correct'] / stats['num_samples'] if stats['num_samples'] else 0.0,
        }
    summary['__overall__'] = {
        **total,
        'accuracy': total['num_correct'] / total['num_samples'] if total['num_samples'] else 0.0,
    }

    Path(args.output_path).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
