import copy
import hashlib
import json
from types import SimpleNamespace
import unittest

from needle_in_haystack_with_mask import validate_saved_case


class MaskResumeTests(unittest.TestCase):
    def fixture(self):
        prepared = SimpleNamespace(case=SimpleNamespace(case_id='case', needle='needle', expected_answer='answer'),
                                   context_length=1000, depth_percent=45,
                                   prompt=SimpleNamespace(token_ids=[1, 2, 3]))
        config = dict(condition='top80', mask_mode='legacy_uniform', context_seed=42,
                      context_window='seeded_contiguous_offset', prefill_attention='sdpa_flash',
                      decode_attention='sdpa_flash', seed=None, max_new_tokens=256)
        row = dict(case_id='case', model='model', context_length=1000, depth_percent=45,
                   needle='needle', expected_answer='answer',
                   prompt_sha256=hashlib.sha256(json.dumps([1, 2, 3]).encode()).hexdigest(),
                   experiment={k: v for k, v in config.items() if k != 'max_new_tokens'},
                   generated_token_ids=[7, 8], finish_reason='eos', score=100)
        row['experiment']['blocked_heads'] = [[1, 2]]
        return row, prepared, config

    def test_saved_result_is_accepted_without_mutation(self):
        row, prepared, config = self.fixture()
        original = copy.deepcopy(row)
        validate_saved_case(row, prepared, 'model', config, {(1, 2)})
        self.assertEqual(row, original)

    def test_changed_prompt_or_heads_or_parameters_rejected(self):
        for mutation in ('prompt', 'heads', 'backend', 'incomplete'):
            row, prepared, config = self.fixture()
            if mutation == 'prompt':
                prepared.prompt.token_ids.append(4)
            elif mutation == 'heads':
                row['experiment']['blocked_heads'] = [[1, 3]]
            elif mutation == 'backend':
                row['experiment']['decode_attention'] = 'eager'
            else:
                row['generated_token_ids'] = []
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_saved_case(row, prepared, 'model', config, {(1, 2)})

    def test_native_token_limit_result_is_accepted_without_mutation(self):
        row, prepared, config = self.fixture()
        row['finish_reason'] = 'max_new_tokens'
        row['generated_token_ids'] = [7] * config['max_new_tokens']
        original = copy.deepcopy(row)
        validate_saved_case(row, prepared, 'model', config, {(1, 2)})
        self.assertEqual(row, original)

    def test_token_limit_result_with_short_token_list_is_rejected(self):
        row, prepared, config = self.fixture()
        row['finish_reason'] = 'max_new_tokens'
        with self.assertRaisesRegex(ValueError, 'inconsistent token count'):
            validate_saved_case(row, prepared, 'model', config, {(1, 2)})
