"""Colab entry point; uses the existing Qwen adapter and detection CLI unchanged."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'colab/qwen3-8b-no-thinking-smoke.yaml'


def command(config, model, output):
    return [sys.executable, '-u', str(ROOT / 'retrieval_head_detection.py'),
            '--adapter', config['adapter'], '--model', str(model),
            '--dtype', config['dtype'], '--device-map', config['device_map'],
            '--prefill-attention', config['prefill_attention'],
            '--attention-scope', config['attention_scope'],
            '--lengths', ','.join(map(str, config['lengths'])),
            '--depths', ','.join(map(str, config['depths'])),
            '--max-new-tokens', str(config['max_new_tokens']),
            '--context-seed', str(config['context_seed']),
            '--retrieval-metrics', ','.join(config['retrieval_metrics']),
            '--haystack-root', str(ROOT / config['haystack_root']),
            '--output-root', str(output)]


def validate(output, config):
    manifest = json.loads((output / 'detection/run.json').read_text())
    if not manifest['complete'] or manifest['adapter'] != 'qwen3_8b_no_thinking':
        raise ValueError('Expected complete no-thinking detection')
    expected = {(case, length, depth) for case in ('detect-1', 'detect-2', 'detect-3')
                for length in config['lengths'] for depth in config['depths']}
    rows = [json.loads(path.read_text()) for path in (output / 'detection/results').glob('*_results.json')]
    keys = {(r['case_id'], r['context_length'], r['depth_percent']) for r in rows}
    if keys != expected or len(rows) != len(expected) or manifest['completed_cases'] != len(expected):
        raise ValueError('Smoke grid is incomplete or duplicated')
    heads = {f'{layer}-{head}' for layer in range(36) for head in range(32)}
    for row in rows:
        if row['experiment']['run_id'] != manifest['run_id']:
            raise ValueError('Result belongs to another detection run')
        if not row['generated_token_ids'] or len(row['generated_token_ids']) > config['max_new_tokens']:
            raise ValueError('Invalid generation length')
        if not row['model_response'].strip() or not row['final_answer_started']:
            raise ValueError('No final answer')
        if '<think>' in row['raw_model_response'] or '</think>' in row['raw_model_response']:
            raise ValueError('Unexpected generated thinking markers')
        for metric in config['retrieval_metrics']:
            scores = row['experiment']['retrieval_scores'][metric]
            if (set(scores) != heads or any(not math.isfinite(v) or v < 0 for v in scores.values())
                    or not any(v > 0 for v in scores.values())):
                raise ValueError(f'Invalid or empty attention scores: {metric}')
    return {'passed': True, 'kind': 'functional_smoke_not_scientific_ranking',
            'cases': len(rows), 'head_count': len(heads),
            'metrics': config['retrieval_metrics'],
            'answers': [{'case_id': r['case_id'], 'score': r['score'],
                         'finish_reason': r['finish_reason'],
                         'tokens': len(r['generated_token_ids']), 'answer': r['model_response']}
                        for r in sorted(rows, key=lambda r: r['case_id'])]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--model-path', type=Path, required=True, help='downloaded Qwen/Qwen3-8B directory')
    parser.add_argument('--output', type=Path, required=True, help='new, empty result directory')
    parser.add_argument('--dry-run', action='store_true', help='print commands without GPU or downloads')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config['adapter'] != 'qwen3_8b_no_thinking' or config['model'] != 'Qwen/Qwen3-8B':
        raise ValueError('This smoke is explicitly for Qwen3-8B without thinking')
    output = args.output.resolve()
    commands = [command(config, args.model_path.resolve(), output),
                [sys.executable, '-u', str(ROOT / 'aggregate_head_scores.py'),
                 '--run', str(output), '--all-cases']]
    for cmd in commands:
        print(shlex.join(cmd), flush=True)
    if args.dry_run:
        return
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output already contains files: choose a new run directory')
    if not (args.model_path / 'config.json').is_file():
        raise FileNotFoundError('Download the checkpoint before running the smoke')
    import torch
    from packaging.version import Version
    if Version(torch.__version__.split('+')[0]) < Version('2.6'):
        raise RuntimeError('torch >=2.6 is required; do not install the DataSphere cu118 requirements')
    if not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime in Colab first')
    props = torch.cuda.get_device_properties(0)
    if torch.cuda.get_device_capability(0)[0] < 8:
        raise RuntimeError('This unchanged Flash smoke requires Ampere or newer; select A100, not T4')
    output.mkdir(parents=True, exist_ok=True)
    runtime = {'torch': torch.__version__, 'cuda': torch.version.cuda,
               'gpu': props.name, 'vram_gib': props.total_memory / 2**30,
               'python': sys.version, 'config': config,
               'packages': {p: importlib.metadata.version(p) for p in
                            ('transformers', 'accelerate', 'huggingface-hub', 'rouge-score')},
               'model_path': str(args.model_path.resolve())}
    (output / 'colab_runtime.json').write_text(json.dumps(runtime, indent=2))
    print(json.dumps(runtime, indent=2), flush=True)
    # Local checkpoint only; never silently download another model during inference.
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
    with (output / 'smoke.log').open('w') as log:
        for cmd in commands:
            with subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True) as process:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                if process.wait():
                    raise RuntimeError(f'Command failed; see {output / "smoke.log"}')
    validation = validate(output, config)
    (output / 'smoke_validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2))
    print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
