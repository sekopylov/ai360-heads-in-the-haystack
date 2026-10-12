"""Colab bridge to the unchanged paired masking job; frozen QRScore ranking."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NAME = 'qwen3-8b-no-thinking-paul-graham-qrscore-mask'
PARENT = Path('/content/drive/MyDrive/retrieval-heads-colab')
OUTPUT = PARENT / NAME
CONFIG_NAMES = (NAME, 'glm4-6v-flash-no-thinking-paul-graham-qrscore-top-mask',
                'ministral3-8b-no-thinking-paul-graham-qrscore-top-mask',
                'glm4-6v-flash-no-thinking-paul-graham-full-needle-mass-top-mask')


def expected_head_keys(adapter):
    layers, heads = {'qwen3_8b_no_thinking': (36, 32),
                     'glm4_6v_flash_no_thinking': (40, 32),
                     'ministral3_8b_no_thinking': (34, 32)}[adapter]
    return {f'{l}-{h}' for l in range(layers) for h in range(heads)}


def model_directory(config):
    return '/content/models/' + config['model'].split('/')[-1]


def generation_count(config):
    return (len(config['lengths']) * len(config['depths']) * config['context_count'] *
            (1 + len(config['topks']) * len(config['mask_selections'])))


def status(stage, **details):
    payload = dict(stage=stage, utc=datetime.now(timezone.utc).isoformat(), **details)
    (PARENT / f'{NAME}-status.json').write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload), flush=True)


def command(config, model, ranking, output):
    args = [sys.executable, '-u', str(ROOT / 'datasphere/job.py'), '--profile', 'mask',
            '--model', str(model), '--output-root', str(output),
            '--mask-data', str(ROOT / config['mask_data']), '--head-scores', str(ranking)]
    for key in ('adapter', 'dtype', 'device_map', 'prefill_attention', 'decode_attention',
                'attention_scope', 'max_new_tokens', 'context_count', 'context_seed', 'seed',
                'mask_case_id'):
        args += ['--' + key.replace('_', '-'), str(config[key])]
    for key in ('lengths', 'depths', 'topks', 'mask_selections'):
        args += ['--' + key.replace('_', '-'), ','.join(map(str, config[key]))]
    return args


def main(resume=False):
    if OUTPUT.exists() and not resume:
        raise RuntimeError('Refusing to overwrite/reuse existing mask output')
    if resume and not OUTPUT.is_dir():
        raise RuntimeError('Resume requires the original output directory')
    status('installing')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r',
                    str(ROOT / 'colab/requirements.txt')], check=True)
    import torch
    import yaml
    from huggingface_hub import snapshot_download
    from aggregate_head_scores import aggregate
    from retrieval_heads.experiment.scoring import rank_heads
    config = yaml.safe_load((ROOT / 'colab' / f'{NAME}.yaml').read_text())
    if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Expected A100 GPU')
    source = PARENT / config['ranking_source']
    manifest = json.loads((source / 'detection/run.json').read_text())
    if (not manifest['complete'] or manifest['adapter'] != config['adapter'] or
            manifest['run_id'] != config['ranking_source_run_id'] or
            manifest['retrieval_metrics'] != config.get('ranking_source_metrics', [config['ranking_metric']]) or
            manifest['completed_cases'] != 27):
        raise ValueError('Wrong or incomplete no-thinking ranking source')
    if resume:
        old_runtime = json.loads((OUTPUT / 'colab_runtime.json').read_text())
        if old_runtime['config'] != config:
            raise ValueError('Resume YAML differs from original run')
        aggregation = json.loads((OUTPUT / 'ranking/run.json').read_text())
        provenance = json.loads((OUTPUT / 'ranking/provenance.json').read_text())
        if (aggregation['source_run_id'] != manifest['run_id'] or
                provenance['source_run_id'] != manifest['run_id'] or
                not aggregation['filters']['all_cases']):
            raise ValueError('Wrong frozen QRScore ranking provenance')
    else:
        OUTPUT.mkdir()
        status('aggregating_ranking')
        aggregation = aggregate(source, output_dir=OUTPUT / 'ranking', all_cases=True)
    ranking = OUTPUT / 'ranking' / aggregation['head_score_files'][config['ranking_metric']]
    history = json.loads(ranking.read_text())
    if resume and hashlib.sha256(ranking.read_bytes()).hexdigest() != provenance['sha256']:
        raise ValueError('Frozen ranking changed since original run')
    expected_heads = expected_head_keys(config['adapter'])
    if (set(history) != expected_heads or aggregation['selected_case_count'] != 27 or
            any(len(values) != 27 or any(not math.isfinite(v) or not 0 <= v <= 1.000001
                                        for v in values) for values in history.values())):
        raise ValueError(f'QRScore ranking must contain {len(expected_heads)} heads x 27 scores')
    ranked = rank_heads(history)
    if not resume:
        (OUTPUT / 'ranking/mean_scores.json').write_text(json.dumps(
            {f'{l}-{h}': score for (l, h), score in ranked}, indent=2))
        (OUTPUT / 'ranking/provenance.json').write_text(json.dumps({
            'source_run_id': manifest['run_id'], 'adapter': manifest['adapter'],
            'metric': config['ranking_metric'], 'all_cases': True, 'case_count': 27,
            'head_count': len(history), 'sha256': hashlib.sha256(ranking.read_bytes()).hexdigest(),
            'top10': [[list(head), score] for head, score in ranked[:10]],
        }, indent=2))
    print(f"Frozen {config['ranking_metric']} ranking: {len(expected_heads)} heads, 27 cases; top10:", ranked[:10], flush=True)
    plan = None
    if resume:
        from colab.mask_resume_plan import resume_plan, condition_command
        plan = resume_plan(config, OUTPUT, model_directory(config), ranking, ranked)
        print('Resume plan (NO model loading for completed blocks):', json.dumps(plan), flush=True)
        status('resume_planned', saved=sum(item['saved'] for item in plan),
               remaining=sum(item['remaining'] for item in plan))
    status('downloading_weights')
    model = snapshot_download(config['model'], revision=config['revision'],
                              local_dir=model_directory(config),
                              allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', '*.model'])
    runtime_file = f"resume_runtime_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}.json" if resume else 'colab_runtime.json'
    (OUTPUT / runtime_file).write_text(json.dumps({
        'config': config, 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0), 'model_path': model}, indent=2))
    checkpoint = json.loads((Path(model) / 'config.json').read_text())
    text_config = checkpoint.get('text_config', checkpoint)
    actual_heads = {f'{l}-{h}' for l in range(text_config['num_hidden_layers'])
                    for h in range(text_config['num_attention_heads'])}
    if actual_heads != expected_heads:
        raise ValueError('Checkpoint head geometry differs from frozen ranking')
    total = generation_count(config)
    status('masking', total_generations=total)
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
    if resume:
        for item in plan:
            if item['complete']:
                print(f"[resume-block] {item['condition']}: {item['saved']} saved; NO model load", flush=True)
                continue
            status('masking', condition=item['condition'], saved=item['saved'],
                   remaining=item['remaining'], total_generations=total)
            subprocess.run(condition_command(config, ROOT, model, ranking, OUTPUT, item),
                           cwd=ROOT, env=env, check=True)
    else:
        subprocess.run(command(config, model, ranking, OUTPUT), cwd=ROOT, env=env, check=True)
    conditions = ['baseline'] + [f'{s}{k}' for k in config['topks'] for s in config['mask_selections']]
    for condition in conditions:
        folder = OUTPUT / f"evaluation/context-{config['context_seed']}/evaluation/{condition}"
        run = json.loads((folder / 'run.json').read_text())
        rows = [json.loads(p.read_text()) for p in (folder / 'results').glob('*_results.json')]
        grid = {(r['context_length'], r['depth_percent']) for r in rows}
        expected_grid = {(l, d) for l in config['lengths'] for d in config['depths']}
        if (not run['complete'] or len(rows) != len(expected_grid) or grid != expected_grid or
                run['adapter'] != config['adapter'] or run['mask_mode'] != 'legacy_uniform'):
            raise ValueError(f'Incomplete or mismatched condition: {condition}')
    status('exporting', output=str(OUTPUT), total_generations=total)
    from colab.archive_mask_results import archive_results
    archive = archive_results(OUTPUT, config)
    status('complete', output=str(OUTPUT), total_generations=total, **archive)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--config-name', choices=CONFIG_NAMES, default=NAME)
    args = parser.parse_args()
    NAME = args.config_name
    OUTPUT = PARENT / NAME
    try:
        main(resume=args.resume)
    except Exception as error:
        status('failed', error=str(error))
        traceback.print_exc()
        sys.exit(1)
