import json
import tempfile
import unittest
from pathlib import Path

from aggregate_head_scores import aggregate


class AggregationTests(unittest.TestCase):
    def fixture(self, root, complete=True):
        detection = root / 'detection'
        results = detection / 'results'
        results.mkdir(parents=True)
        source = {'complete': complete, 'completed_cases': 3,
                  'retrieval_metrics': ['legacy', 'needle_attention_mass_v1'],
                  'model': 'fixture', 'lengths': [1000, 2000], 'depths': [45],
                  'attention_scope': 'all_decode_tokens', 'run_id': 'current'}
        (detection / 'run.json').write_text(json.dumps(source))
        for i, score in enumerate([50, 85, 0]):
            row = {'model': 'fixture', 'case_id': f'detect-{i+1}',
                   'context_length': 1000 if i < 2 else 2000,
                   'depth_percent': 45, 'score': score, 'prompt_sha256': str(i),
                   'experiment': {'run_id': 'current', 'attention_scope': 'all_decode_tokens',
                                  'retrieval_scores': {name: {'0-0': i / 3, '0-1': .2}
                                                      for name in source['retrieval_metrics']}}}
            (results / f'{i}_results.json').write_text(json.dumps(row))
        return detection

    def test_threshold_all_and_filters_do_not_change_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            detection = self.fixture(root)
            original = (detection / 'run.json').read_text()
            a = aggregate(root)
            self.assertEqual(a['selected_case_count'], 1)  # strict >50
            b = aggregate(root, all_cases=True, output_dir=root / 'all')
            self.assertEqual(b['selected_case_count'], 3)
            c = aggregate(root, all_cases=True, lengths=[2000], case_ids=['detect-3'],
                          output_dir=root / 'filtered')
            self.assertEqual(c['selected_case_count'], 1)
            self.assertEqual((detection / 'run.json').read_text(), original)
            scores = json.loads((root / 'all/head_scores_legacy.json').read_text())
            self.assertEqual(scores['0-0'], [0, 1/3, 2/3])
            with self.assertRaisesRegex(ValueError, 'No cases'):
                aggregate(root, success_threshold=100)

    def test_partial_and_stale_results(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            detection = self.fixture(root, complete=False)
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                aggregate(root)
            stale = json.loads((detection / 'results/0_results.json').read_text())
            stale['experiment']['run_id'] = 'old'
            (detection / 'results/old_results.json').write_text(json.dumps(stale))
            result = aggregate(root, all_cases=True, allow_incomplete=True)
            self.assertFalse(result['complete'])
            self.assertEqual(result['available_case_count'], 3)

    def test_mismatched_heads_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            detection = self.fixture(root)
            path = detection / 'results/2_results.json'
            row = json.loads(path.read_text())
            row['experiment']['retrieval_scores']['legacy'] = {'9-9': .2}
            path.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, 'Head sets differ'):
                aggregate(root, all_cases=True)
