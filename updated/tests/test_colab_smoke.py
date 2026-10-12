import copy
import json
from pathlib import Path
import tempfile
import unittest

from colab.run_smoke import command, validate


class ColabSmokeTests(unittest.TestCase):
    def setUp(self):
        self.config = {'adapter': 'qwen3_8b_no_thinking', 'dtype': 'float16',
                       'device_map': 'cuda:0', 'prefill_attention': 'sdpa_flash',
                       'attention_scope': 'all_decode_tokens', 'lengths': [1000],
                       'depths': [50], 'max_new_tokens': 64, 'context_seed': 42,
                       'retrieval_metrics': ['needle_token_multiset_v1',
                                             'needle_attention_mass_v1'],
                       'haystack_root': 'data/haystack_for_detect'}
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'detection/results').mkdir(parents=True)
        (self.root / 'detection/run.json').write_text(json.dumps({
            'complete': True, 'adapter': 'qwen3_8b_no_thinking',
            'completed_cases': 3, 'run_id': 'smoke'}))
        self.scores = {f'{layer}-{head}': 0.1 for layer in range(36) for head in range(32)}
        self.rows = [{'case_id': f'detect-{case}', 'context_length': 1000,
                      'depth_percent': 50, 'generated_token_ids': [1, 2],
                      'model_response': 'answer', 'raw_model_response': 'answer',
                      'final_answer_started': True, 'finish_reason': 'eos', 'score': 100,
                      'experiment': {'run_id': 'smoke', 'retrieval_scores': {
                          metric: copy.deepcopy(self.scores)
                          for metric in self.config['retrieval_metrics']}}}
                     for case in range(1, 4)]

    def write_rows(self):
        for row in self.rows:
            (self.root / 'detection/results' / (row['case_id'] + '_results.json')).write_text(
                json.dumps(row))

    def test_valid_smoke(self):
        self.write_rows()
        result = validate(self.root, self.config)
        self.assertTrue(result['passed'])
        self.assertEqual(result['head_count'], 1152)
        self.assertEqual(result['cases'], 3)

    def test_thinking_is_rejected(self):
        self.rows[0]['raw_model_response'] = '<think>thought</think>answer'
        self.write_rows()
        with self.assertRaisesRegex(ValueError, 'thinking'):
            validate(self.root, self.config)

    def test_missing_head_is_rejected(self):
        del self.rows[0]['experiment']['retrieval_scores'][self.config['retrieval_metrics'][0]]['0-0']
        self.write_rows()
        with self.assertRaisesRegex(ValueError, 'attention scores'):
            validate(self.root, self.config)

    def test_incomplete_grid_is_rejected(self):
        self.rows.pop()
        self.write_rows()
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            validate(self.root, self.config)

    def test_preserves_qwen_generation_flags(self):
        cmd = command(self.config, Path('/model'), self.root)
        self.assertEqual(cmd[cmd.index('--adapter') + 1], 'qwen3_8b_no_thinking')
        self.assertEqual(cmd[cmd.index('--dtype') + 1], 'float16')
        self.assertEqual(cmd[cmd.index('--max-new-tokens') + 1], '64')
