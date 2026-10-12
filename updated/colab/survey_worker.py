"""One explicitly selected QRScore survey; no resume or reuse of old results."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PARENT = Path('/content/drive/MyDrive/retrieval-heads-colab')
NAME = 'qwen3-8b-no-thinking-qrscore-survey'
OUTPUT = PARENT / NAME
STATUS = PARENT / f'{NAME}-status.json'


def status(stage, **details):
    payload = {'stage': stage, 'utc': datetime.now(timezone.utc).isoformat(), **details}
    STATUS.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload), flush=True)


def main():
    if OUTPUT.exists():
        raise RuntimeError('Survey output already exists; no overwrite/resume')
    status('installing')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r',
                    str(ROOT / 'colab/requirements.txt')], check=True)
    import torch
    import yaml
    from huggingface_hub import snapshot_download
    from colab.run_smoke import command
    from colab.survey_validation import validate_survey
    config = yaml.safe_load((ROOT / 'colab' / f'{NAME}.yaml').read_text())
    if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Survey requires the selected A100 runtime')
    status('downloading_weights', revision=config['revision'])
    model = snapshot_download(config['model'], revision=config['revision'],
                              local_dir=f"/content/models/{config['model'].split('/')[-1]}",
                              allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', '*.model'])
    OUTPUT.mkdir()
    (OUTPUT / 'colab_runtime.json').write_text(json.dumps({
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0), 'python': sys.version,
        'config': config, 'model_path': model,
        'packages': {p: importlib.metadata.version(p) for p in
                     ('transformers', 'accelerate', 'huggingface-hub', 'rouge-score')},
    }, indent=2))
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
    status('detecting', total_cases=27)
    subprocess.run(command(config, Path(model), OUTPUT), cwd=ROOT, env=env, check=True)
    status('aggregating')
    subprocess.run([sys.executable, '-u', str(ROOT / 'aggregate_head_scores.py'),
                    '--run', str(OUTPUT), '--all-cases'], cwd=ROOT, env=env, check=True)
    checkpoint_config = json.loads((Path(model) / 'config.json').read_text())
    validation = validate_survey(OUTPUT, config, checkpoint_config)
    (OUTPUT / 'survey_validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2))
    status('complete', output=str(OUTPUT), validation=validation)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-name', choices=(NAME, 'ministral3-8b-no-thinking-qrscore-survey',
                                                 'glm4-6v-flash-no-thinking-qrscore-survey'), default=NAME)
    NAME = parser.parse_args().config_name
    OUTPUT = PARENT / NAME
    STATUS = PARENT / f'{NAME}-status.json'
    try:
        main()
    except Exception as error:
        status('failed', error=str(error))
        traceback.print_exc()
        sys.exit(1)
