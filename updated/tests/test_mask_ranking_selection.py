import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from needle_in_haystack_with_mask import load_ranked_heads, choose_blocked_heads


class MaskRankingSelectionTests(unittest.TestCase):
    def test_bottom_uses_full_ranking_and_lowest_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'scores.json'
            path.write_text(json.dumps({f'0-{h}': [float(120-h)] for h in range(120)}))
            model = SimpleNamespace(eligible_heads=tuple((0, h) for h in range(120)))
            args = SimpleNamespace(output_root=root, head_scores=path, mask_topk=2,
                                   head_selection='bottom', random_exclusion_top=None)
            ranked = load_ranked_heads(args, model)
            self.assertEqual(len(ranked), 120)
            self.assertEqual(choose_blocked_heads(args, model, ranked),
                             frozenset({(0, 118), (0, 119)}))
            args.head_selection = 'top'
            self.assertEqual(choose_blocked_heads(args, model, ranked),
                             frozenset({(0, 0), (0, 1)}))
            args.mask_topk = 0
            self.assertFalse(choose_blocked_heads(args, model, ranked))

    def test_single_metric_uses_manifest_without_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'detection' / 'aggregation'
            folder.mkdir(parents=True)
            filename = 'head_scores_needle_attention_mass_v1.json'
            (folder / 'run.json').write_text(json.dumps({'complete': True,
                'head_score_files': {'needle_attention_mass_v1': filename}}))
            (folder / filename).write_text(json.dumps({'0-0': [.2], '0-1': [.9]}))
            args = SimpleNamespace(output_root=root, head_scores=None)
            model = SimpleNamespace(eligible_heads=((0, 0), (0, 1)))
            self.assertEqual(load_ranked_heads(args, model)[0], (0, 1))
            (folder / 'run.json').write_text(json.dumps({'complete': True}))
            (folder / 'head_scores.json').write_text(json.dumps({'0-0': [999]}))
            with self.assertRaisesRegex(ValueError, '--head-scores'):
                load_ranked_heads(args, model)

    def test_explicit_file_selects_metric_and_no_implicit_primary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root/'detection'/'aggregation'
            folder.mkdir(parents=True)
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
