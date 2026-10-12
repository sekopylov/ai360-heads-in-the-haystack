"""Persist a complete results-only ZIP before a Colab runtime can disappear."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def archive_results(root, config):
    root = Path(root)
    conditions = ['baseline'] + [f'{s}{k}' for k in config['topks'] for s in config['mask_selections']]
    count = len(config['lengths']) * len(config['depths'])
    files = []
    for condition in conditions:
        folder = root / f"evaluation/context-{config['context_seed']}/evaluation/{condition}"
        run = json.loads((folder / 'run.json').read_text())
        rows = sorted((folder / 'results').glob('*_results.json'))
        if not run['complete'] or run['completed_cases'] != count or len(rows) != count:
            raise ValueError(f'Cannot export incomplete block {condition}')
        files += [folder / 'run.json', *rows]
    files += sorted(p for p in (root / 'ranking').iterdir() if p.is_file())
    files += sorted(p for p in root.glob('*runtime*.json') if p.is_file())
    target = root.parent / f'{root.name}-results.zip'
    temporary = target.with_suffix('.zip.tmp')
    with ThreadPoolExecutor(max_workers=12) as pool, ZipFile(temporary, 'w', ZIP_DEFLATED) as archive:
        for path, data in zip(files, pool.map(lambda p: p.read_bytes(), files)):
            archive.writestr(str(path.relative_to(root.parent)), data)
    temporary.replace(target)
    return dict(archive=str(target), archive_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                archive_bytes=target.stat().st_size, archive_results=len(conditions)*count)
