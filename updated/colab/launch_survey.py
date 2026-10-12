"""Run via colab exec; keep the kernel executing while the survey runs."""
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

parent = Path('/content/drive/MyDrive/retrieval-heads-colab')
if not parent.parent.is_dir():
    raise RuntimeError('Google Drive is not mounted; mount it before launching')
parent.mkdir(exist_ok=True)
name = globals().get('SURVEY_NAME', 'qwen3-8b-no-thinking-qrscore-survey')
if name not in ('qwen3-8b-no-thinking-qrscore-survey', 'ministral3-8b-no-thinking-qrscore-survey',
                'glm4-6v-flash-no-thinking-qrscore-survey'):
    raise ValueError('Unknown survey name')
log = parent / f'{name}-bootstrap.log'
if log.exists() or (parent / name).exists():
    raise RuntimeError('Survey already launched; refusing duplicate execution')
with ZipFile('/content/retrieval-heads-survey.zip') as bundle:
    bundle.extractall('/content/retrieval-heads-survey')
root = Path('/content/retrieval-heads-survey/updated')
print(f'Starting 27-case {name}', flush=True)
with log.open('x', buffering=1) as output:
    with subprocess.Popen([sys.executable, '-u', str(root / 'colab/survey_worker.py'), '--config-name', name],
                          cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True) as process:
        for line in process.stdout:
            output.write(line)
            print(line, end='', flush=True)
        code = process.wait()
if code:
    raise RuntimeError(f'Survey failed ({code}); see {log}')
print('Survey and validation complete; results are on Google Drive', flush=True)
