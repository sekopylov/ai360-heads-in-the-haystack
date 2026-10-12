"""Foreground Colab entrypoint for the paired Paul Graham QRScore mask run."""
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile
from datetime import datetime, timezone
import os

parent = Path('/content/drive/MyDrive/retrieval-heads-colab')
name = globals().get('MASK_NAME', 'qwen3-8b-no-thinking-paul-graham-qrscore-mask')
if name not in ('qwen3-8b-no-thinking-paul-graham-qrscore-mask',
                'glm4-6v-flash-no-thinking-paul-graham-qrscore-top-mask',
                'ministral3-8b-no-thinking-paul-graham-qrscore-top-mask'):
    raise ValueError('Unknown masking config')
if not parent.parent.is_dir():
    raise RuntimeError('Mount Google Drive before launching')
resume = globals().get('RESUME', False)
suffix = '-resume-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f') if resume else ''
log = parent / f'{name}{suffix}-bootstrap.log'
if not resume and (log.exists() or (parent / name).exists()):
    raise RuntimeError('Already launched; refusing duplicate execution or overwrite')
if resume and not (parent / name / 'colab_runtime.json').is_file():
    raise RuntimeError('Cannot resume: original runtime record missing')
with ZipFile('/content/retrieval-heads-mask.zip') as bundle:
    bundle.extractall('/content/retrieval-heads-mask')
root = Path('/content/retrieval-heads-mask/updated')
max_seconds = globals().get('MAX_RUNTIME_SECONDS')
if max_seconds is not None:
    if not isinstance(max_seconds, int) or max_seconds <= 0 or not os.environ.get('TBE_RUNTIME_ADDR'):
        raise RuntimeError('Cannot arm the runtime-local auto-stop timer')
    timer_log = parent / f'{name}{suffix}-autostop.log'
    with timer_log.open('x', buffering=1) as timer_output:
        timer = subprocess.Popen([sys.executable, '-u', str(root / 'colab/runtime_deadline.py'),
            '--status', str(parent / f'{name}-status.json'),
            '--record', str(parent / f'{name}{suffix}-autostop.json'),
            '--seconds', str(max_seconds)], stdout=timer_output, stderr=subprocess.STDOUT,
            start_new_session=True)
    print(f'Auto-stop armed: {max_seconds}s maximum; earlier after ZIP is saved (watchdog PID {timer.pid})', flush=True)
print(f'Starting {name}', flush=True)
with log.open('x', buffering=1) as output:
    with subprocess.Popen([sys.executable, '-u', str(root / 'colab/mask_worker.py'), '--config-name', name] +
                          (['--resume'] if resume else []),
                          cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True) as process:
        for line in process.stdout:
            output.write(line)
            print(line, end='', flush=True)
        code = process.wait()
if code:
    raise RuntimeError(f'Masking failed ({code}); see {log}')
print('Masking complete; results are on Drive', flush=True)
