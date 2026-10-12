"""Execute via colab exec after uploading /content/retrieval-heads-smoke.zip."""
from pathlib import Path
import json
import subprocess
import sys
from zipfile import ZipFile

drive = Path('/content/drive/MyDrive')
if not drive.is_dir():
    raise RuntimeError('Mount Google Drive before starting the smoke')
import torch
print({'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
       'drive': str(drive)}, flush=True)
state = Path('/content/retrieval-heads-smoke-state.json')
if state.exists():
    raise RuntimeError('Smoke already launched; inspect state rather than duplicate it')
with ZipFile('/content/retrieval-heads-smoke.zip') as bundle:
    bundle.extractall('/content/retrieval-heads-smoke')
root = Path('/content/retrieval-heads-smoke/updated')
log_path = drive / 'retrieval-heads-colab/qwen3-smoke-bootstrap.log'
log_path.parent.mkdir(parents=True, exist_ok=True)
with log_path.open('x') as log:
    process = subprocess.Popen([sys.executable, '-u', str(root / 'colab/worker.py')],
                               cwd=root, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
state.write_text(json.dumps({'pid': process.pid, 'log': str(log_path)}))
print({'pid': process.pid, 'log': str(log_path)}, flush=True)
