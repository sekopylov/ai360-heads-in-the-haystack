import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from needle_in_haystack_with_mask import load_ranked_heads


class MaskRankingSelectionTests(unittest.TestCase):
    def test_explicit_file_selects_metric_and_no_implicit_primary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root/'detection'
            folder.mkdir()
            (folder/'run.json').write_text(json.dumps({'complete': True,
                'retrieval_metrics': ['legacy', 'needle_attention_mass_v1']}))
            (folder/'head_scores.json').write_text(json.dumps({'0-0': [999]}))
            a, b = folder/'head_scores_legacy.json', folder/'head_scores_needle_attention_mass_v1.json'
            a.write_text(json.dumps({'0-0': [1], '0-1': [0]}))
            b.write_text(json.dumps({'0-0': [0], '0-1': [.9]}))
            args = SimpleNamespace(output_root=root, head_scores=None)
            model = SimpleNamespace(eligible_heads=((0, 0), (0, 1)))
            with self.assertRaisesRegex(ValueError, '--head-scores'):
                load_ranked_heads(args, model)
            args.head_scores = a
            self.assertEqual(load_ranked_heads(args, model)[0], (0, 0))
            args.head_scores = b
            self.assertEqual(load_ranked_heads(args, model)[0], (0, 1))
