import json
from pathlib import Path
import tempfile
import unittest

from colab.survey_validation import validate_survey


class SurveyValidationTests(unittest.TestCase):
    def test_glm_real_tiny_qr_forward(self):
        from tests.test_glm4v_adapter import Glm4vAdapterTests
        from retrieval_heads.models.base import Prompt
        import torch
        adapter = Glm4vAdapterTests().adapter()
        ids = torch.tensor([[1, 5, 6, 7, 8, 9]])
        scores = adapter.score_query_retrieval(Prompt(ids, ids[0].tolist()),
                                              query_span=(4, 6), gold_span=(1, 3))
        self.assertEqual(set(scores), {f'{l}-{h}' for l, h in adapter.eligible_heads})
        self.assertTrue(all(0 <= v <= 1 for v in scores.values()))

    def test_ministral_head_count_is_not_qwen_head_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / 'detection/results'
            results.mkdir(parents=True)
            aggregation = root / 'detection/aggregation'
            aggregation.mkdir()
            config = {'adapter': 'ministral3_8b_no_thinking', 'lengths': [8000],
                      'depths': [15], 'max_new_tokens': 2048,
                      'retrieval_metrics': ['query_retrieval_score_v1']}
            (root / 'detection/run.json').write_text(json.dumps({
                'complete': True, 'adapter': config['adapter'], 'run_id': 'test', 'completed_cases': 3}))
            (aggregation / 'run.json').write_text(json.dumps({
                'source_run_id': 'test', 'selected_case_count': 3, 'filters': {'all_cases': True}}))
            scores = {f'{l}-{h}': .1 for l in range(34) for h in range(32)}
            for case in range(1, 4):
                (results / f'{case}_results.json').write_text(json.dumps({
                    'case_id': f'detect-{case}', 'context_length': 8000, 'depth_percent': 15,
                    'score': 0, 'finish_reason': 'eos', 'generated_token_ids': [1],
                    'raw_model_response': 'answer', 'experiment': {
                        'run_id': 'test', 'qrscore_query_span': [5, 7], 'qrscore_gold_span': [1, 3],
                        'retrieval_scores': {'query_retrieval_score_v1': scores}}}))
            checkpoint = {'text_config': {'model_type': 'ministral3',
                                         'num_hidden_layers': 34, 'num_attention_heads': 32}}
            checked = validate_survey(root, config, checkpoint)
            self.assertEqual(checked['head_count'], 1088)
            self.assertEqual(checked['mean_answer_score'], 0)
            self.assertTrue(checked['passed'])

    def test_ministral_real_tiny_qr_forward(self):
        from tests.test_ministral_adapter import MinistralAdapterTests
        from retrieval_heads.models.base import Prompt
        import torch
        adapter = MinistralAdapterTests().make_adapter()
        ids = torch.tensor([[1, 5, 6, 7, 8, 9]])
        scores = adapter.score_query_retrieval(Prompt(ids, ids[0].tolist()),
                                              query_span=(4, 6), gold_span=(1, 3))
        self.assertEqual(set(scores), {f'{l}-{h}' for l, h in adapter.eligible_heads})
        self.assertTrue(all(0 <= v <= 1 for v in scores.values()))
