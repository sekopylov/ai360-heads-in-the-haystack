"""Colab exec: exact full-needle detection, then baseline/top5/10/15/20/40 masking."""
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

SURVEY = 'glm4-6v-flash-no-thinking-full-needle-mass-survey'
MASK = 'glm4-6v-flash-no-thinking-paul-graham-full-needle-mass-top-mask'


def bind_ranking(mask_config, survey_config, run, validation, runtime):
    """Freeze only this newly completed survey, never an unrelated Drive ranking."""
    if (not run.get('complete') or run.get('completed_cases') != 27
            or not run.get('run_id') or not validation.get('passed')
            or validation.get('cases') != 27 or validation.get('head_count') != 1280
            or runtime.get('config') != survey_config):
        raise ValueError('Survey validation/runtime mismatch')
    for field in ('adapter', 'context_seed', 'lengths', 'depths', 'max_new_tokens',
                  'attention_scope', 'retrieval_metrics'):
        if run.get(field) != survey_config[field]:
            raise ValueError(f'Survey manifest mismatch: {field}')
    for field in ('adapter', 'model', 'revision', 'dtype', 'prefill_attention', 'context_seed'):
        if mask_config[field] != survey_config[field]:
            raise ValueError(f'Mask/survey config mismatch: {field}')
    if (mask_config['ranking_source_metrics'] != survey_config['retrieval_metrics']
            or mask_config['ranking_metric'] != 'full_needle_attention_mass_v1'
            or mask_config['ranking_source'] != SURVEY
            or mask_config['ranking_source_run_id'] is not None):
        raise ValueError('Unexpected full-needle ranking template')
    return {**mask_config, 'ranking_source_run_id': run['run_id']}


def main():
    parent = Path('/content/drive/MyDrive/retrieval-heads-colab')
    if not parent.parent.is_dir():
        raise RuntimeError('Mount Google Drive before launching')
    parent.mkdir(exist_ok=True)
    log = parent / f'{MASK}-workflow.log'
    if log.exists() or (parent / SURVEY).exists() or (parent / MASK).exists():
        raise RuntimeError('Already launched; refusing overwrite or duplicate execution')
    with ZipFile('/content/retrieval-heads-full-needle.zip') as bundle:
        bundle.extractall('/content/retrieval-heads-full-needle')
    root = Path('/content/retrieval-heads-full-needle/updated')
    with log.open('x', buffering=1) as output:
        def execute(script, name):
            with subprocess.Popen([sys.executable, '-u', str(root / 'colab' / script),
                                   '--config-name', name], cwd=root,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
                for line in process.stdout:
                    output.write(line)
                    print(line, end='', flush=True)
                code = process.wait()
            if code:
                raise RuntimeError(f'{name} failed ({code}); see {log}')

        print('Step 1: 27-case full-needle survey', flush=True)
        execute('survey_worker.py', SURVEY)
        import yaml  # installed by survey_worker
        survey_config = yaml.safe_load((root / 'colab' / f'{SURVEY}.yaml').read_text())
        mask_path = root / 'colab' / f'{MASK}.yaml'
        mask_config = yaml.safe_load(mask_path.read_text())
        source = parent / SURVEY
        run = json.loads((source / 'detection/run.json').read_text())
        validation = json.loads((source / 'survey_validation.json').read_text())
        runtime = json.loads((source / 'colab_runtime.json').read_text())
        resolved = bind_ranking(mask_config, survey_config, run, validation, runtime)
        mask_path.write_text(yaml.safe_dump(resolved, sort_keys=False))
        print(f"Step 2: frozen full-needle ranking from {run['run_id']}; 660 masking cases", flush=True)
        execute('mask_worker.py', MASK)
    print('Complete: survey and masking results/ZIP are saved on Drive', flush=True)


if __name__ == '__main__':
    main()
