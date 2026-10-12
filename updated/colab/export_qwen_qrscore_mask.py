"""Export one complete logical mask run on CPU; never load model weights."""
from pathlib import Path
import hashlib
import json
from zipfile import ZipFile, ZIP_DEFLATED
from concurrent.futures import ThreadPoolExecutor

parent = Path('/content/drive/MyDrive/retrieval-heads-colab')
name = 'qwen3-8b-no-thinking-paul-graham-qrscore-mask'
root = parent / name
if not parent.parent.is_dir():
    raise RuntimeError('Mount Google Drive before exporting')
conditions = ['baseline'] + [f'{s}{k}' for k in (5, 10, 15, 20, 40, 60, 80, 100)
                           for s in ('top', 'bottom')]
files = []
for condition in conditions:
    folder = root / 'evaluation/context-42/evaluation' / condition
    run = json.loads((folder / 'run.json').read_text())
    results = sorted((folder / 'results').glob('*_results.json'))
    if not run['complete'] or run['completed_cases'] != 110 or len(results) != 110:
        raise ValueError(f'Incomplete condition: {condition}')
    files += [folder / 'run.json', *results]
    print(f'Validated {condition}: 110 cases', flush=True)
files += sorted(p for p in (root / 'ranking').iterdir() if p.is_file())
files += sorted(p for p in root.glob('*runtime*.json') if p.is_file())
status = parent / f'{name}-status.json'
if json.loads(status.read_text())['stage'] != 'complete':
    raise ValueError('Expected final complete status')
files.append(status)
archive_path = Path('/content') / f'{name}-results.zip'
with ThreadPoolExecutor(max_workers=12) as pool, ZipFile(archive_path, 'w', ZIP_DEFLATED) as archive:
    # Overlap slow Drive reads; ZIP writing remains single-threaded and deterministic.
    for i, (path, data) in enumerate(zip(files, pool.map(lambda p: p.read_bytes(), files)), 1):
        archive.writestr(str(path.relative_to(parent)), data)
        if i % 110 == 0:
            print(f'Archived {i}/{len(files)} files', flush=True)
print(json.dumps({'archive': str(archive_path), 'files': len(files), 'cases': 1870,
                  'bytes': archive_path.stat().st_size,
                  'sha256': hashlib.sha256(archive_path.read_bytes()).hexdigest()}), flush=True)
