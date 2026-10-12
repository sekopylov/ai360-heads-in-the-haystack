"""Package current workspace code + smoke data, without weights, credentials or old results."""
from pathlib import Path
import argparse
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]


def bundle(output):
    files = {ROOT / name for name in ('retrieval_head_detection.py', 'aggregate_head_scores.py',
                                    'needle_in_haystack_with_mask.py', 'datasphere/job.py')}
    for folder in ('retrieval_heads', 'data/haystack_for_detect', 'data/validation_haystacks', 'colab'):
        files.update(p for p in (ROOT / folder).rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and p.suffix != '.pyc')
    if output.resolve() in {p.resolve() for p in files}:
        raise ValueError('Place the ZIP outside the colab/code/data directories')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, 'w', compression=ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, str(Path('updated') / path.relative_to(ROOT)))
    return len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/retrieval-heads-colab-smoke.zip'))
    args = parser.parse_args()
    print(f'{args.output}: {bundle(args.output)} files; {args.output.stat().st_size / 2**20:.2f} MiB')


if __name__ == '__main__':
    main()
