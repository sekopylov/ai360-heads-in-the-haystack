"""Validate completed masking blocks without model loading or prompt tokenization."""
import json
from pathlib import Path
from retrieval_heads.experiment.storage import result_stem


def resume_plan(config, output, model, ranking, ranked):
    output = Path(output)
    context_root = output / f"evaluation/context-{config['context_seed']}"
    grid_size = len(config['lengths']) * len(config['depths'])
    conditions = [('baseline', 0, 'top')] + [
        (f'{selection}{k}', k, selection)
        for k in config['topks'] for selection in config['mask_selections']]
    plan = []
    for name, k, selection in conditions:
        directory = context_root / 'evaluation' / name
        path = directory / 'run.json'
        files = {p.name for p in (directory / 'results').glob('*_results.json')}
        if not path.exists():
            if files:
                raise ValueError(f'Results exist without a manifest: {name}')
            plan.append(dict(condition=name, k=k, selection=selection, saved=0,
                             remaining=grid_size, complete=False))
            continue
        run = json.loads(path.read_text())
        blocked = [] if not k else sorted(head for head, _ in
                                          (ranked[:k] if selection == 'top' else ranked[-k:]))
        expected = dict(kind='head_masking', condition=name, adapter=config['adapter'],
                        model=str(model), lengths=config['lengths'], depths=config['depths'],
                        context_seed=config['context_seed'], max_new_tokens=config['max_new_tokens'],
                        prefill_attention=config['prefill_attention'], decode_attention=config['decode_attention'],
                        mask_mode='legacy_uniform', randomized_per_case=False,
                        context_window='seeded_contiguous_offset', case_ids=[config['mask_case_id']],
                        head_selection=selection if k else 'baseline',
                        blocked_heads=[list(head) for head in blocked],
                        head_scores_file=str(ranking) if k else None)
        for key, value in expected.items():
            if run.get(key) != value:
                raise ValueError(f'Resume manifest mismatch: {name}: {key}')
        expected_files = {result_stem(run['model_version'], length, depth,
                                     case_id=config['mask_case_id']) + '_results.json'
                          for length in config['lengths'] for depth in config['depths']}
        if files - expected_files:
            raise ValueError(f'Unexpected result files: {name}')
        complete = bool(run['complete'])
        if complete and (files != expected_files or run.get('completed_cases') != grid_size):
            raise ValueError(f'Completed manifest has missing/extra results: {name}')
        plan.append(dict(condition=name, k=k, selection=selection, saved=len(files),
                         remaining=grid_size-len(files), complete=complete))
    return plan


def condition_command(config, root, model, ranking, output, item):
    """Keep the original masking CLI and full grid for deterministic corpus windows."""
    import sys
    command = [sys.executable, '-u', str(Path(root) / 'needle_in_haystack_with_mask.py'),
               '--model', str(model), '--head-scores', str(ranking),
               '--output-root', str(Path(output) / f"evaluation/context-{config['context_seed']}"),
               '--validation-root', str(Path(root) / config['mask_data']),
               '--case-id', config['mask_case_id'], '--mask-mode', 'legacy_uniform',
               '--mask-topk', str(item['k']), '--head-selection', item['selection'], '--resume']
    for key in ('adapter', 'dtype', 'device_map', 'prefill_attention', 'decode_attention',
                'attention_scope', 'max_new_tokens', 'context_seed'):
        command += ['--' + key.replace('_', '-'), str(config[key])]
    for key in ('lengths', 'depths'):
        command += ['--' + key, ','.join(map(str, config[key]))]
    # Original job does NOT forward --seed to the masking subprocess; preserve that.
    return command
