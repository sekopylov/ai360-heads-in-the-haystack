from types import SimpleNamespace
import json
from pathlib import Path
import tempfile
import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM, AttentionInterface

from retrieval_heads.attention.qrscore import question_gold_mass, score_prompt, BACKEND
from retrieval_heads.experiment.query_spans import locate_qr_spans
from retrieval_heads.experiment.scoring import create_retrieval_collector, retrieval_capture_requirements
from retrieval_heads.models.base import Prompt
from aggregate_head_scores import aggregate


class QRScoreTests(unittest.TestCase):
    def test_uniform_causal_mass_averages_question_not_gold(self):
        q = torch.zeros(1, 4, 8, 2)
        k = torch.zeros(1, 2, 8, 2)
        score = question_gold_mass(q, k, None, query_span=(4, 6), gold_span=(1, 3))
        torch.testing.assert_close(score, torch.full((4,), (2 / 5 + 2 / 6) / 2))

    def test_matches_full_softmax_with_gqa_and_chunking(self):
        torch.manual_seed(17)
        q, k = torch.randn(1, 4, 12, 8), torch.randn(1, 2, 12, 8)
        logits = q @ k.repeat_interleave(2, dim=1).transpose(-2, -1) * .3
        causal = torch.ones(12, 12, dtype=torch.bool).tril()
        logits.masked_fill_(~causal, float('-inf'))
        reference = logits.softmax(-1)[0, :, 8:11, 2:5].sum(-1).mean(-1)
        score = question_gold_mass(q, k, None, query_span=(8, 11), gold_span=(2, 5),
                                   scaling=.3, chunk_size=1)
        torch.testing.assert_close(score, reference)

    def test_boolean_and_additive_masks(self):
        q, k = torch.zeros(1, 2, 6, 2), torch.zeros(1, 1, 6, 2)
        mask = torch.ones(1, 1, 6, 6, dtype=torch.bool)
        mask[..., 0] = False
        additive = torch.zeros_like(mask, dtype=torch.float32).masked_fill(~mask, float('-inf'))
        for value in (mask, additive):
            scores = question_gold_mass(q, k, value, query_span=(4, 6), gold_span=(1, 3))
            torch.testing.assert_close(scores, torch.full((2,), (2/4 + 2/5)/2))

    def test_invalid_spans_and_empty_question(self):
        q = torch.zeros(1, 1, 5, 2)
        for query, gold in [((2, 2), (0, 1)), ((1, 3), (0, 2)), ((4, 6), (0, 1))]:
            with self.assertRaises(ValueError):
                question_gold_mass(q, q, None, query_span=query, gold_span=gold)

    def test_question_and_gold_locator_ignores_question_in_context(self):
        tokenizer = SimpleNamespace(decode=lambda ids, **kw: ''.join(map(chr, ids)))
        text = 'Q? old context GOLD. Question: Q? Answer:'
        query, gold = locate_qr_spans(tokenizer, list(map(ord, text)), question='Q?', needle='GOLD.')
        self.assertEqual(text[query.start:query.end], 'Q?')
        self.assertEqual(query.start, text.rfind('Q?'))
        self.assertEqual(text[gold.start:gold.end], 'GOLD.')
        with self.assertRaises(ValueError):
            locate_qr_spans(tokenizer, list(map(ord, text)), question='Q?', needle='missing')
        with self.assertRaises(ValueError):
            duplicate = 'GOLD GOLD Question: Q?'
            locate_qr_spans(tokenizer, list(map(ord, duplicate)), question='Q?', needle='GOLD')

    def test_locator_handles_space_merged_token(self):
        pieces = ['intro', ' GOLD', '.', ' Question:', ' what', '?', ' end']
        tokenizer = SimpleNamespace(decode=lambda ids, **kw: ''.join(pieces[i] for i in ids))
        query, gold = locate_qr_spans(tokenizer, list(range(7)), question='what?', needle='GOLD.')
        self.assertEqual((query.start, query.end), (4, 6))
        self.assertEqual((gold.start, gold.end), (1, 3))

    def test_decode_cannot_modify_qrscore_and_no_decode_capture_needed(self):
        collector = create_retrieval_collector('query_retrieval_score_v1',
            eligible_heads=((0, 0),), prompt_token_ids=[1, 2],
            needle_span=SimpleNamespace(start=0, end=1))
        collector.set_prefill_scores({'0-0': .7})
        collector.on_step(None)
        self.assertEqual(collector.scores, {'0-0': .7})
        self.assertEqual(retrieval_capture_requirements(['query_retrieval_score_v1']), (False, False))
        with self.assertRaises(ValueError):
            collector.set_prefill_scores({'0-0': float('nan')})

    def adapter(self):
        torch.manual_seed(3)
        config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                            num_hidden_layers=2, num_attention_heads=4,
                            num_key_value_heads=2, head_dim=4, max_position_embeddings=32)
        model = Qwen3ForCausalLM(config).eval()
        model.set_attn_implementation('sdpa')
        return SimpleNamespace(model=model, eligible_heads=tuple((l, h) for l in range(2) for h in range(4)),
                               input_device=torch.device('cpu'))

    def test_real_tiny_qwen_forward_and_backend_restoration(self):
        adapter = self.adapter()
        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        with torch.inference_mode():
            before = adapter.model(ids, use_cache=False).logits
        scores = score_prompt(adapter, Prompt(ids, ids[0].tolist()),
                              query_span=(5, 7), gold_span=(1, 3))
        self.assertEqual(set(scores), {f'{l}-{h}' for l in range(2) for h in range(4)})
        self.assertTrue(all(0 < v < 1 for v in scores.values()))
        self.assertEqual(adapter.model.config._attn_implementation, 'sdpa')
        self.assertNotIn(BACKEND, AttentionInterface())
        with torch.inference_mode():
            after = adapter.model(ids, use_cache=False).logits
        torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_backend_restores_on_invalid_span(self):
        adapter = self.adapter()
        ids = torch.tensor([[1, 2, 3, 4]])
        with self.assertRaises(ValueError):
            score_prompt(adapter, Prompt(ids, ids[0].tolist()),
                         query_span=(3, 5), gold_span=(0, 1))
        self.assertEqual(adapter.model.config._attn_implementation, 'sdpa')
        self.assertNotIn(BACKEND, AttentionInterface())

    def test_aggregation_requires_all_cases_and_includes_failed_answer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / 'detection/results'
            results.mkdir(parents=True)
            metric = 'query_retrieval_score_v1'
            (root / 'detection/run.json').write_text(json.dumps({
                'complete': True, 'run_id': 'test', 'model': 'fixture',
                'retrieval_metrics': [metric], 'completed_cases': 2,
                'lengths': [1000], 'depths': [50], 'attention_scope': 'answer_only'}))
            for number, (answer_score, qr_score) in enumerate([(100, .2), (0, .8)]):
                (results / f'{number}_results.json').write_text(json.dumps({
                    'case_id': str(number), 'context_length': 1000, 'depth_percent': 50,
                    'model': 'fixture', 'score': answer_score, 'prompt_sha256': 'test',
                    'experiment': {'run_id': 'test', 'attention_scope': 'answer_only',
                                   'retrieval_scores': {metric: {'0-0': qr_score}}}}))
            with self.assertRaisesRegex(ValueError, 'all-cases'):
                aggregate(root)
            manifest = aggregate(root, all_cases=True)
            self.assertEqual(manifest['selected_case_count'], 2)
            history = json.loads((root / 'detection/aggregation' / f'head_scores_{metric}.json').read_text())
            self.assertEqual(history, {'0-0': [.2, .8]})

    def test_qr_pass_preserves_cached_qwen_generation_and_decode_attention(self):
        from tests.test_qwen_adapters import QwenAdapterTests
        from retrieval_heads.models import Qwen3EightBNoThinkingAdapter
        from retrieval_heads.attention import AttentionRequest, FullTraceCollector
        adapter = QwenAdapterTests().make_adapter('all_decode_tokens', Qwen3EightBNoThinkingAdapter)
        ids = torch.tensor([[2, 11, 22, 3, 4, 5, 6]])
        prompt = Prompt(ids, ids[0].tolist())
        request = AttentionRequest(capture='full', needle_span=(1, 3))
        old_trace, new_trace = FullTraceCollector(prompt.token_ids), FullTraceCollector(prompt.token_ids)
        before = adapter.generate(prompt, max_new_tokens=4, attention=request, observer=old_trace)
        adapter.score_query_retrieval(prompt, query_span=(4, 6), gold_span=(1, 3))
        after = adapter.generate(prompt, max_new_tokens=4, attention=request, observer=new_trace)
        self.assertEqual(before, after)
        self.assertEqual(len(old_trace.steps), len(new_trace.steps))
        for old, new in zip(old_trace.steps, new_trace.steps):
            for layer in old.layers:
                torch.testing.assert_close(old.layers[layer], new.layers[layer], rtol=0, atol=0)
                torch.testing.assert_close(old.needle_attention_mass[layer],
                                           new.needle_attention_mass[layer], rtol=0, atol=0)
