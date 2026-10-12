"""Install Colab-only dependencies, download weights locally, run one smoke."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path('/content/drive/MyDrive/retrieval-heads-colab/qwen3-smoke')
STATUS = OUTPUT.parent / 'qwen3-smoke-status.json'


def status(stage, **details):
    value = {'stage': stage, 'utc': datetime.now(timezone.utc).isoformat(), **details}
    STATUS.write_text(json.dumps(value, indent=2))
    print(json.dumps(value), flush=True)


def main():
    if OUTPUT.exists():
        raise RuntimeError('Refusing to reuse an existing output directory')
    status('installing')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r',
                    str(ROOT / 'colab/requirements.txt')], check=True)
    from huggingface_hub import HfApi, snapshot_download
    status('downloading_weights')
    revision = HfApi().model_info('Qwen/Qwen3-8B').sha
    model = snapshot_download('Qwen/Qwen3-8B', revision=revision,
                              local_dir='/content/models/Qwen3-8B',
                              allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja'])
    (OUTPUT.parent / 'qwen3-smoke-checkpoint.json').write_text(json.dumps(
        {'repo': 'Qwen/Qwen3-8B', 'revision': revision, 'path': model}, indent=2))
    status('detecting', revision=revision)
    subprocess.run([sys.executable, '-u', str(ROOT / 'colab/run_smoke.py'),
                    '--model-path', model, '--output', str(OUTPUT)], check=True)
    status('complete', output=str(OUTPUT))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed', error=str(error))
        traceback.print_exc()
        sys.exit(1)
