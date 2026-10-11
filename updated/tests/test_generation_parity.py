"""Compare the refactored runtime against the frozen pre-refactor Qwen path."""
from contextlib import nullcontext
from types import MethodType, SimpleNamespace
from unittest import TestCase
from unittest.mock import patch, MagicMock

import torch
from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM

from retrieval_heads.attention import AttentionRequest, FullTraceCollector, flash
from retrieval_heads.attention.prefill import resolve_prefill_backend
from retrieval_heads.experiment.scoring import AnswerScorer, create_retrieval_collector
from retrieval_heads.experiment.types import NeedleSpan
from retrieval_heads.experiment.storage import result_payload
from retrieval_heads.models import (
    Qwen3Adapter, Qwen3EightBAdapter, Qwen3EightBNoThinkingAdapter,
    Qwen3EightBNoYarn64KAdapter, Qwen3EightBYarnAdapter,
    Qwen3EightBNoYarn64KNoThinkingAdapter, Qwen3EightBYarnNoThinkingAdapter, Qwen35Adapter,
)
from retrieval_heads.models.base import Prompt
from tests.reference_qwen_decode import historical_generate, historical_filter, historical_result, historical_encode_prompt
from tests import test_qwen_adapters as qwen_fixtures


class GenerationParityTests(TestCase):
    def test_qwen_prompt_tokenization_is_unchanged(self):
        for cls in (Qwen35Adapter, Qwen3Adapter, Qwen3EightBAdapter,
                    Qwen3EightBNoThinkingAdapter, Qwen3EightBNoYarn64KAdapter,
                    Qwen3EightBYarnAdapter, Qwen3EightBNoYarn64KNoThinkingAdapter,
                    Qwen3EightBYarnNoThinkingAdapter):
            with self.subTest(adapter=cls.__name__):
                adapter = cls.__new__(cls)
                adapter._tokenizer = MagicMock()
                adapter._tokenizer.apply_chat_template.return_value = "rendered prompt"
                adapter._tokenizer.return_value = {"input_ids": torch.tensor([[2, 3, 11]])}
                old = historical_encode_prompt(adapter, "context\nwith unicode: иголка", "question?")
                old_calls = list(adapter._tokenizer.mock_calls)
                adapter._tokenizer.reset_mock()
                new = adapter.encode_prompt("context\nwith unicode: иголка", "question?")
                self.assertEqual(adapter._tokenizer.mock_calls, old_calls)
                self.assertEqual(old.token_ids, new.token_ids)
                torch.testing.assert_close(old.input_ids, new.input_ids, rtol=0, atol=0)

    def test_storage_keeps_uniform_metadata_for_qwen35(self):
        adapter = Qwen35Adapter.__new__(Qwen35Adapter)
        adapter._tokenizer = qwen_fixtures.ToyTokenizer()
        generation = adapter._generation_result(Prompt(None, [2]), [9, 62], "newline")
        prepared = SimpleNamespace(case=SimpleNamespace(case_id="test", needle="needle", expected_answer="word9"),
            context_length=1000, depth_percent=50, prompt=Prompt(None, [2]))
        row = result_payload(SimpleNamespace(prepared=prepared, generation=generation,
                            score=100.0, duration_seconds=1.0), "fixture")
        self.assertEqual(row["model_response"], "word9")
        self.assertEqual(row["score"], 100.0)
        self.assertEqual(row["generated_token_ids"], [9, 62])
        self.assertEqual(row["raw_model_response"], "word9 \n")
        self.assertEqual(row["finish_reason"], "newline")
        self.assertTrue(row["final_answer_started"])

    def test_qwen35_hybrid_decode_preserves_tokens_and_attention(self):
        torch.manual_seed(42)
        config = Qwen3_5TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            layer_types=["linear_attention", "full_attention"], linear_num_key_heads=2,
            linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
            max_position_embeddings=128, eos_token_id=63,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5})
        model = Qwen3_5ForCausalLM(config).eval()
        with patch("retrieval_heads.models.transformers_common.resolve_model_source", return_value=("fixture", True)), \
             patch("retrieval_heads.models.transformers_common.AutoTokenizer.from_pretrained", return_value=qwen_fixtures.ToyTokenizer()), \
             patch.object(Qwen3_5ForCausalLM, "from_pretrained", return_value=model):
            adapter = Qwen35Adapter(device_map="cpu", dtype="float32")
        prompt = Prompt(torch.tensor([[2, 11, 22]]), [2, 11, 22])
        for blocked in (frozenset(), frozenset({(1, 2)})):
            traces = [FullTraceCollector(prompt.token_ids), FullTraceCollector(prompt.token_ids)]
            request = AttentionRequest(capture="full", blocked_heads=blocked)
            before = historical_generate(adapter, prompt, max_new_tokens=4, attention=request, observer=traces[0])
            after = adapter.generate(prompt, max_new_tokens=4, attention=request, observer=traces[1])
            self.assertEqual(before, after)
            self.assertEqual(len(traces[0].steps), len(traces[1].steps))
            for old_step, new_step in zip(traces[0].steps, traces[1].steps):
                torch.testing.assert_close(old_step.layers[1], new_step.layers[1], rtol=0, atol=0)

    def test_qwen_tokens_attention_scores_and_output_match_historical_path(self):
        classes = (
            Qwen3Adapter, Qwen3EightBAdapter, Qwen3EightBNoThinkingAdapter,
            Qwen3EightBNoYarn64KAdapter, Qwen3EightBYarnAdapter,
            Qwen3EightBNoYarn64KNoThinkingAdapter, Qwen3EightBYarnNoThinkingAdapter,
        )
        for cls in classes:
            for scope in ("answer_only", "all_decode_tokens"):
                for mode, blocked in (
                    ("legacy_uniform", frozenset()),
                    ("legacy_uniform", frozenset({(0, 1)})),
                    ("zero_output", frozenset({(1, 2)})),
                ):
                    with self.subTest(adapter=cls.__name__, scope=scope, mode=mode, blocked=blocked):
                        adapter = qwen_fixtures.QwenAdapterTests().make_adapter(scope, cls)
                        prompt = Prompt(torch.tensor([[2, 11, 22]]), [2, 11, 22])
                        request = AttentionRequest(capture="full", needle_span=(1, 3),
                                                   blocked_heads=blocked, mask_mode=mode)
                        old_trace = FullTraceCollector(prompt.token_ids)
                        with patch.object(adapter, "_attention_filter", MethodType(historical_filter, adapter)), \
                             patch.object(adapter, "_generation_result", MethodType(historical_result, adapter)):
                            old = historical_generate(adapter, prompt, max_new_tokens=4,
                                                      attention=request, observer=old_trace)
                        new_trace = FullTraceCollector(prompt.token_ids)
                        new = adapter.generate(prompt, max_new_tokens=4,
                                               attention=request, observer=new_trace)
                        self.assertEqual(old, new)
                        self.assertEqual(len(old_trace.steps), len(new_trace.steps))
                        for old_step, new_step in zip(old_trace.steps, new_trace.steps):
                            self.assertEqual((old_step.index, old_step.token_id),
                                             (new_step.index, new_step.token_id))
                            for layer in old_step.layers:
                                torch.testing.assert_close(old_step.layers[layer], new_step.layers[layer], rtol=0, atol=0)
                                torch.testing.assert_close(old_step.needle_attention_mass[layer],
                                                           new_step.needle_attention_mass[layer], rtol=0, atol=0)
                        for metric in ("legacy", "needle_token_multiset_v1", "needle_attention_mass_v1"):
                            collectors = []
                            for trace in (old_trace, new_trace):
                                collector = create_retrieval_collector(metric,
                                    eligible_heads=adapter.eligible_heads,
                                    prompt_token_ids=prompt.token_ids, needle_span=NeedleSpan(1, 3))
                                for step in trace.steps:
                                    collector.on_step(step)
                                collectors.append(collector)
                            self.assertEqual(collectors[0].scores, collectors[1].scores)
                        scorer = AnswerScorer()
                        self.assertEqual(scorer.score("word11 word22", old.text),
                                         scorer.score("word11 word22", new.text))

    def test_qwen_flash_outputs_match_historical_loop(self):
        adapter = qwen_fixtures.QwenAdapterTests().make_adapter("all_decode_tokens", Qwen3EightBNoThinkingAdapter)
        adapter.decode_attention = "sdpa_flash"
        adapter.model.set_attn_implementation(resolve_prefill_backend("sdpa_flash"))
        prompt = Prompt(torch.tensor([[2, 11, 22]]), [2, 11, 22])
        with patch.object(flash, "flash_kernel", return_value=nullcontext()):
            for blocked in (frozenset(), frozenset({(0, 1)})):
                request = AttentionRequest(blocked_heads=blocked)
                old = historical_generate(adapter, prompt, max_new_tokens=4, attention=request)
                new = adapter.generate(prompt, max_new_tokens=4, attention=request)
                self.assertEqual(old, new)

    def test_qwen_thinking_and_prompt_markers_match_historical_policy(self):
        adapter = qwen_fixtures.QwenAdapterTests().make_adapter()
        for prompt_ids in ([2, 60], [2, 60, 61, 62], [2]):
            for tokens in ([7, 62, 8], [7, 61, 9, 63], [60, 7, 61, 9, 63]):
                prompt = Prompt(None, prompt_ids)
                self.assertEqual(historical_result(adapter, prompt, tokens, "max_new_tokens"),
                                 adapter._generation_result(prompt, tokens, "max_new_tokens"))
                before = historical_filter(adapter, prompt)
                after = adapter._attention_filter(prompt)
                self.assertEqual([before(t) for t in tokens], [after(t) for t in tokens])
