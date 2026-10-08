import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import retrieval_head_detection as detection
from retrieval_heads.attention import AttentionController
from retrieval_heads.models.base import GenerationResult, Prompt
from retrieval_heads.experiment.types import ExperimentCase, NeedleSpan, PreparedExample, RunResult


class MultiMetricDetectionTests(unittest.TestCase):
    def exercise(self, multiple, score=100, case_id=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = ['detect', '--lengths', '1000', '--depths', '45', '--output-root', directory]
            if multiple:
                argv += ['--retrieval-metrics', multiple]
            if case_id is not None:
                argv += ['--case-id', case_id]
            model = SimpleNamespace(model_id='fixture', model_version='fixture',
                tokenizer=object(), period_tokens=[4], eligible_heads=((0, 0),),
                attention_scope='all_decode_tokens')
            case = ExperimentCase('detect-1', 'needle', 'question', 'answer', root)
            prepared = PreparedExample(case, 'context', Prompt(None, [99, 11, 22]),
                                       NeedleSpan(1, 3), 1000, 45)
            calls = []
            def run(example, attention, observer):
                calls.append(attention)
                controller = AttentionController((0,))
                controller.start(attention, observer)
                for token, probs in [(11, [.1, .6, .3]), (22, [.8, .1, .1])]:
                    controller.begin_step()
                    controller.record(0, torch.tensor(probs).reshape(1, 1, 1, 3))
                    controller.end_step(token)
                controller.finish()
                return RunResult(example, GenerationResult([11, 22], 'answer'), score, .1)
            runner = SimpleNamespace(prepare=lambda *a, **kw: prepared, run=run)
            with patch.object(sys, 'argv', argv), patch.object(detection, 'create_model', return_value=model), \
                 patch.object(detection, 'ContextBuilder'), patch.object(detection, 'LegacyOverlapLocator'), \
                 patch.object(detection, 'ExperimentRunner', return_value=runner), \
                 patch.object(detection, 'load_detection_cases', return_value=[case] if case_id is None else [
                     case, ExperimentCase('detect-2', 'needle', 'question', 'answer', root)]):
                detection.main()
            self.assertEqual(len(calls), 1)
            manifest = json.loads((root/'detection/run.json').read_text())
            self.assertTrue(manifest['complete'])
            payload = json.loads(next((root/'detection/results').glob('*.json')).read_text())
            self.assertFalse((root/'detection/attention').exists())
            alias_path = root/'detection/head_scores.json'
            self.assertFalse(alias_path.exists())
            self.assertNotIn('head_score_files', manifest)
            self.assertFalse(list((root/'detection').glob('head_scores_*.json')))
            return manifest, payload, calls[0]

    def test_single_case_filter(self):
        manifest, _, _ = self.exercise(None, case_id='detect-1')
        self.assertEqual(manifest['total_cases'], 1)

    def test_unknown_case_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown detection case'):
            self.exercise(None, case_id='detect-missing')

    def test_three_metrics_one_generation_without_primary(self):
        manifest, payload, request = self.exercise('needle_attention_mass_v1,needle_token_multiset_v1,legacy')
        self.assertEqual(request.capture, 'top1')
        self.assertEqual(request.needle_span, (1, 3))
        self.assertIsNone(manifest['retrieval_metric'])
        scores = payload['experiment']['retrieval_scores']
        self.assertAlmostEqual(scores['needle_attention_mass_v1']['0-0'], .55, places=6)
        self.assertEqual(scores['legacy']['0-0'], .5)
        self.assertEqual(scores['needle_token_multiset_v1']['0-0'], .5)

    def test_mass_only_capture_is_compact_and_failed_case_still_saved(self):
        manifest, payload, request = self.exercise('needle_attention_mass_v1', score=0)
        self.assertEqual(request.capture, 'none')
        self.assertEqual(manifest['successful_cases'], 0)
        self.assertAlmostEqual(payload['experiment']['retrieval_scores']['needle_attention_mass_v1']['0-0'], .55, places=6)

    def test_default_old_metric_and_capture_unchanged(self):
        manifest, _, request = self.exercise(None)
        self.assertEqual(manifest['retrieval_metric'], 'needle_token_multiset_v1')
        self.assertEqual(request.capture, 'top1')
        self.assertIsNone(request.needle_span)
