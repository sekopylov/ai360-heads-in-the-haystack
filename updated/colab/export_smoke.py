"""Archive this smoke's small result files for local verification, never weights."""
from pathlib import Path
import json
from zipfile import ZipFile, ZIP_DEFLATED

root = Path('/content/drive/MyDrive/retrieval-heads-colab')
status = json.loads((root / 'qwen3-smoke-status.json').read_text())
if status['stage'] not in ('complete', 'failed'):
    raise RuntimeError('Smoke still running; export only after it finishes')
files = sorted(p for p in root.glob('qwen3-smoke*') if p.is_file())
output = root / 'qwen3-smoke'
if output.exists():
    files.extend(sorted(p for p in output.rglob('*') if p.is_file()))
with ZipFile('/content/qwen3-smoke-results.zip', 'w', ZIP_DEFLATED) as archive:
    for path in files:
        archive.write(path, str(path.relative_to(root)))
print(json.dumps(status))
validation = output / 'smoke_validation.json'
if validation.exists():
    print(validation.read_text())
print('Archived files:', len(files))
