import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile
from colab.archive_mask_results import archive_results


class MaskArchiveTests(unittest.TestCase):
    def test_top_only_results_archive_and_incomplete_block_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / 'run'
            config = dict(lengths=[1000], depths=[50], topks=[1], mask_selections=['top'], context_seed=42)
            for condition in ('baseline', 'top1'):
                folder = root / f'evaluation/context-42/evaluation/{condition}'
                (folder / 'results').mkdir(parents=True)
                (folder / 'run.json').write_text(json.dumps(dict(complete=True, completed_cases=1)))
                (folder / 'results/case_results.json').write_text('{"score":100}')
            (root / 'ranking').mkdir()
            (root / 'ranking/provenance.json').write_text('{}')
            (root / 'colab_runtime.json').write_text('{}')
            result = archive_results(root, config)
            self.assertEqual(result['archive_results'], 2)
            with ZipFile(result['archive']) as archive:
                self.assertIsNone(archive.testzip())
                self.assertEqual(sum(n.endswith('_results.json') for n in archive.namelist()), 2)
                self.assertFalse(any('bottom' in n for n in archive.namelist()))
            previous = Path(result['archive']).read_bytes()
            (root / 'evaluation/context-42/evaluation/top1/run.json').write_text('{"complete":false}')
            with self.assertRaises(ValueError):
                archive_results(root, config)
            self.assertEqual(Path(result['archive']).read_bytes(), previous)
