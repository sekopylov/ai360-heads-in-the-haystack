"""Build head-score histories from saved detection cases; no GPU required."""
import argparse
import json
import math
from pathlib import Path


def aggregate(run_root, *, output_dir=None, success_threshold=50.0, all_cases=False,
              case_ids=None, lengths=None, depths=None, allow_incomplete=False):
    detection = Path(run_root) / 'detection'
    source = json.loads((detection / 'run.json').read_text())
    if not source['complete'] and not allow_incomplete:
        raise ValueError('Detection is incomplete; use --allow-incomplete explicitly')
    names = source['retrieval_metrics']
    histories = {name: {} for name in names}
    selected = []
    seen = set()
    for path in sorted((detection / 'results').glob('*_results.json')):
        row = json.loads(path.read_text())
        experiment = row['experiment']
        if source.get('run_id') and experiment.get('run_id') != source['run_id']:
            continue  # Ignore leftovers from another run in the same output folder.
        if (row['model'] != source['model'] or row['context_length'] not in source['lengths']
                or row['depth_percent'] not in source['depths']
                or experiment['attention_scope'] != source['attention_scope']):
            raise ValueError(f'Result does not match detection manifest: {path}')
        key = (row['case_id'], row['context_length'], row['depth_percent'])
        if key in seen:
            raise ValueError(f'Duplicate case: {key}')
        seen.add(key)
        if case_ids is not None and row['case_id'] not in case_ids:
            continue
        if lengths is not None and row['context_length'] not in lengths:
            continue
        if depths is not None and row['depth_percent'] not in depths:
            continue
        if not all_cases and not row['score'] > success_threshold:
            continue
        scores = experiment['retrieval_scores']
        for name in names:
            values = scores[name]
            if not values or any(not math.isfinite(float(v)) for v in values.values()):
                raise ValueError(f'Invalid head scores: {path}, {name}')
            history = histories[name]
            if history and set(history) != set(values):
                raise ValueError(f'Head sets differ: {path}, {name}')
            for head, value in values.items():
                history.setdefault(head, []).append(float(value))
        selected.append({'file': str(path.resolve()), 'case_id': row['case_id'],
                         'length': row['context_length'], 'depth': row['depth_percent'],
                         'score': row['score'], 'prompt_sha256': row['prompt_sha256']})
    if source['complete'] and len(seen) != source['completed_cases']:
        raise ValueError('Saved case count differs from completed_cases')
    if not selected:
        raise ValueError('No cases pass aggregation filters; no ranking written')
    output = Path(output_dir) if output_dir is not None else detection / 'aggregation'
    output.mkdir(parents=True, exist_ok=True)
    files = {name: f'head_scores_{name}.json' for name in names}
    for name, history in histories.items():
        (output / files[name]).write_text(json.dumps(history, ensure_ascii=False))
    manifest = {'kind': 'head_score_aggregation', 'complete': source['complete'],
                'source_run': str((detection / 'run.json').resolve()),
                'source_run_id': source.get('run_id'), 'retrieval_metrics': names,
                'head_score_files': files, 'selected_cases': selected,
                'selected_case_count': len(selected), 'available_case_count': len(seen),
                'filters': {'all_cases': all_cases, 'success_threshold': success_threshold,
                            'case_ids': case_ids, 'lengths': lengths, 'depths': depths}}
    (output / 'run.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--success-threshold', type=float, default=50.0)
    parser.add_argument('--all-cases', action='store_true')
    parser.add_argument('--case-ids', help='comma-separated case IDs')
    parser.add_argument('--lengths', help='comma-separated context lengths')
    parser.add_argument('--depths', help='comma-separated needle depths')
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    result = aggregate(args.run, output_dir=args.output_dir,
                       success_threshold=args.success_threshold, all_cases=args.all_cases,
                       case_ids=args.case_ids.split(',') if args.case_ids else None,
                       lengths=[int(v) for v in args.lengths.split(',')] if args.lengths else None,
                       depths=[float(v) for v in args.depths.split(',')] if args.depths else None,
                       allow_incomplete=args.allow_incomplete)
    print(f"Aggregated {result['selected_case_count']}/{result['available_case_count']} cases")


if __name__ == '__main__':
    main()
