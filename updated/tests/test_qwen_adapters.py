"""Exercise real Qwen3 cached forwards with tiny random weights, no downloads."""
import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from retrieval_heads.attention import AttentionRequest, FullTraceCollector
from retrieval_heads.models import Qwen3Adapter, Qwen3EightBAdapter, Qwen3EightBYarnAdapter, Qwen35Adapter, available_models
from retrieval_heads.models.base import Prompt
from retrieval_heads.experiment.scoring import create_retrieval_collector
from retrieval_heads.experiment.types import NeedleSpan


class ToyTokenizer:
    eos_token_id = 63

    def encode(self, text, **kwargs):
        return {"<think>": [60], "</think>": [61]}.get(text, [4])

    def decode(self, ids, skip_special_tokens=False):
        tokens = {60: "<think>\n", 61: "</think>", 62: "\n", 63: "<eos>"}
        return "".join(tokens.get(i, f"word{i} ") for i in ids
                       if not (skip_special_tokens and i == 63))


class QwenAdapterTests(unittest.TestCase):
    def test_yarn_real_checkpoint_load_and_rope(self):
        self.assertIn('qwen3_8b_yarn', available_models())
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=2, num_attention_heads=4,
                            num_key_value_heads=2, head_dim=8,
                            max_position_embeddings=40960, eos_token_id=63,
                            rope_parameters={'rope_type': 'default', 'rope_theta': 1000000.0})
        with tempfile.TemporaryDirectory() as folder:
            Qwen3ForCausalLM(config).save_pretrained(folder)
            with patch('retrieval_heads.models.qwen_common.AutoTokenizer.from_pretrained',
                       return_value=ToyTokenizer()):
                native = Qwen3EightBAdapter(model_id=folder, device_map='cpu', dtype='float32')
                yarn = Qwen3EightBYarnAdapter(model_id=folder, device_map='cpu', dtype='float32')
            self.assertEqual(native.model.config.max_position_embeddings, 40960)
            self.assertEqual(native.model.config.rope_parameters['rope_type'], 'default')
            self.assertEqual(yarn.model.config.max_position_embeddings, 131072)
            self.assertEqual(yarn.model.config.rope_parameters['rope_type'], 'yarn')
            self.assertEqual(yarn.model.config.rope_parameters['factor'], 4.0)
            self.assertEqual(yarn.eligible_heads, native.eligible_heads)
            for key, value in native.model.state_dict().items():
                torch.testing.assert_close(value, yarn.model.state_dict()[key])
            self.assertFalse(torch.equal(native.model.model.rotary_emb.inv_freq,
                                         yarn.model.model.rotary_emb.inv_freq))
            with torch.no_grad():
                result = yarn.model(input_ids=torch.tensor([[2, 3]]),
                                    position_ids=torch.tensor([[48000, 48001]]))
            self.assertTrue(torch.isfinite(result.logits).all())
            # Native guard is replaced by the YaRN budget, using the same loop.
            long_prompt = Prompt(torch.zeros((1, 131070), dtype=torch.long), [])
            with patch.object(Qwen3Adapter, 'generate', return_value=None) as generate:
                yarn.generate(long_prompt, max_new_tokens=2, attention=AttentionRequest())
                with self.assertRaisesRegex(ValueError, 'YaRN context budget exceeded'):
                    yarn.generate(long_prompt, max_new_tokens=3, attention=AttentionRequest())
                self.assertEqual(generate.call_count, 1)

    def make_adapter(self, attention_scope=None, adapter_class=Qwen3Adapter):
        torch.manual_seed(42)
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=2, num_attention_heads=4,
                            num_key_value_heads=2, head_dim=8, max_position_embeddings=128,
                            eos_token_id=63, attention_dropout=0.0)
        config._attn_implementation = "sdpa"
        model = Qwen3ForCausalLM(config).eval()
        with patch("retrieval_heads.models.qwen_common.resolve_model_source", return_value=("fixture", True)), \
             patch("retrieval_heads.models.qwen_common.AutoTokenizer.from_pretrained", return_value=ToyTokenizer()) as tokenizer_load, \
             patch.object(Qwen3ForCausalLM, "from_pretrained", return_value=model) as model_load:
            adapter = adapter_class(device_map="cpu", dtype="float32", attention_scope=attention_scope)
        self.assertTrue(tokenizer_load.call_args.kwargs["local_files_only"])
        self.assertTrue(model_load.call_args.kwargs["local_files_only"])
        return adapter

    def test_qwen3_8b_load_and_real_decode(self):
        self.assertIn("qwen3_8b", available_models())
        adapter = self.make_adapter(adapter_class=Qwen3EightBAdapter)
        self.assertEqual(adapter.model_id, "Qwen/Qwen3-8B")
        self.assertEqual(adapter.attention_scope, "answer_only")
        self.assertEqual(len(adapter.eligible_heads), 8)  # tiny fixture, config-driven
        result = adapter.generate(Prompt(torch.tensor([[2, 3, 5]]), [2, 3, 5]),
                                  max_new_tokens=2, attention=AttentionRequest())
        self.assertTrue(result.token_ids)
        self.assertEqual(adapter.model.config._attn_implementation, "sdpa")

    def test_qwen3_8b_native_budget_and_shared_thinking(self):
        adapter = Qwen3EightBAdapter.__new__(Qwen3EightBAdapter)
        adapter._tokenizer = ToyTokenizer()
        adapter._eos_token_ids = {63}
        self.assertIsNone(adapter._stop_reason(62))  # newline must not stop thinking
        self.assertEqual(adapter._stop_reason(63), 'eos')
        answer = adapter._generation_result(Prompt(None, [60]), [7, 61, 9, 63], 'eos')
        self.assertEqual(answer.text, 'word9')
        prompt = Prompt(torch.zeros((1, 30720), dtype=torch.long), [])
        with patch.object(Qwen3Adapter, 'generate', return_value=answer) as generate:
            self.assertIs(adapter.generate(prompt, max_new_tokens=2048,
                                           attention=AttentionRequest()), answer)
            with self.assertRaisesRegex(ValueError, 'native context budget exceeded'):
                adapter.generate(prompt, max_new_tokens=2049, attention=AttentionRequest())
            self.assertEqual(generate.call_count, 1)

    def test_registry_and_real_decode_capture_mask_restore(self):
        self.assertIn("qwen3", available_models())
        adapter = self.make_adapter()
        self.assertEqual(adapter.eligible_heads, tuple((l, h) for l in range(2) for h in range(4)))
        prompt = Prompt(torch.tensor([[2, 3, 5]]), [2, 3, 5])
        trace = FullTraceCollector(prompt.token_ids)
        result = adapter.generate(prompt, max_new_tokens=3,
                                  attention=AttentionRequest(capture="full",
                                      blocked_heads=frozenset({(0, 1)})), observer=trace)
        self.assertTrue(result.token_ids)
        self.assertEqual(len(trace.steps), len(result.token_ids))
        for step in trace.steps:
            self.assertEqual(set(step.layers), {0, 1})
            probabilities = step.layers[0][1]
            torch.testing.assert_close(probabilities, torch.full_like(probabilities, 1 / len(probabilities)))
        self.assertEqual(adapter.model.config._attn_implementation, "sdpa")
        self.assertFalse(adapter.attention._active)
        # Exercise the additional ablation and a fresh cache after the first run.
        adapter.generate(prompt, max_new_tokens=2,
                         attention=AttentionRequest(blocked_heads=frozenset({(1, 2)}), mask_mode="zero_output"))
        self.assertEqual(adapter.model.config._attn_implementation, "sdpa")

    def test_observer_failure_restores_backend_and_controller(self):
        adapter = self.make_adapter()
        observer = SimpleNamespace(on_step=lambda step: (_ for _ in ()).throw(RuntimeError("observer failed")))
        with self.assertRaisesRegex(RuntimeError, "observer failed"):
            adapter.generate(Prompt(torch.tensor([[2, 3]]), [2, 3]), max_new_tokens=2,
                             attention=AttentionRequest(capture="top1"), observer=observer)
        self.assertEqual(adapter.model.config._attn_implementation, "sdpa")
        self.assertFalse(adapter.attention._active)

    def test_qwen3_final_answer_and_unfinished_thought(self):
        adapter = Qwen3Adapter.__new__(Qwen3Adapter)
        adapter._tokenizer = ToyTokenizer()
        prompt = Prompt(None, [2, 60])
        complete = adapter._generation_result(prompt, [7, 62, 8, 61, 9, 63], "eos")
        self.assertEqual(complete.text, "word9")
        self.assertIn("word7", complete.raw_text)
        self.assertEqual(complete.finish_reason, "eos")
        incomplete = adapter._generation_result(prompt, [7, 62, 8], "max_new_tokens")
        self.assertEqual(incomplete.text, "")
        self.assertTrue(incomplete.raw_text)
        plain = adapter._generation_result(Prompt(None, [2]), [9, 63], "eos")
        self.assertEqual(plain.text, "word9")

    def test_thinking_gate_and_fresh_generation(self):
        adapter = Qwen3Adapter.__new__(Qwen3Adapter)
        adapter._tokenizer = ToyTokenizer()
        predicate = adapter._attention_filter(Prompt(None, [2, 60]))
        self.assertEqual([predicate(t) for t in [7, 62, 8, 61, 9, 63]],
                         [False, False, False, False, True, True])
        # A new generation must not inherit the completed thought's state.
        fresh = adapter._attention_filter(Prompt(None, [2, 60]))
        self.assertFalse(fresh(9))
        plain = adapter._attention_filter(Prompt(None, [2]))
        self.assertEqual([plain(t) for t in [60, 7, 61, 9]], [False, False, False, True])

    def test_only_answer_steps_reach_collector_with_real_forwards(self):
        adapter = self.make_adapter()
        prompt = Prompt(torch.tensor([[2, 60]]), [2, 60])
        trace = FullTraceCollector(prompt.token_ids)
        original = adapter.model.forward
        tokens = iter([7, 62, 8, 61, 9, 63])

        def scripted_forward(*args, **kwargs):
            output = original(*args, **kwargs)
            if kwargs.get("past_key_values") is not None:
                output.logits.fill_(-1000)
                output.logits[:, -1, next(tokens)] = 1000
            return output

        with patch.object(adapter.model, "forward", side_effect=scripted_forward):
            result = adapter.generate(prompt, max_new_tokens=8,
                                      attention=AttentionRequest(capture="full"), observer=trace)
        self.assertEqual(result.token_ids, [7, 62, 8, 61, 9, 63])
        self.assertEqual([step.token_id for step in trace.steps], [9, 63])
        self.assertEqual([step.index for step in trace.steps], [4, 5])
        self.assertEqual(result.text, "word9")

    def test_unfinished_thinking_produces_no_observations(self):
        adapter = self.make_adapter()
        prompt = Prompt(torch.tensor([[2, 60]]), [2, 60])
        trace = FullTraceCollector(prompt.token_ids)
        original = adapter.model.forward

        def scripted_forward(*args, **kwargs):
            output = original(*args, **kwargs)
            output.logits.fill_(-1000)
            output.logits[:, -1, 7] = 1000
            return output

        with patch.object(adapter.model, "forward", side_effect=scripted_forward):
            result = adapter.generate(prompt, max_new_tokens=3,
                                      attention=AttentionRequest(capture="top1"), observer=trace)
        self.assertEqual(trace.steps, [])
        self.assertEqual(result.text, "")
        self.assertEqual(result.finish_reason, "max_new_tokens")

    def test_all_decode_includes_thinking_and_keeps_final_answer(self):
        adapter = self.make_adapter(attention_scope="all_decode_tokens")
        self.assertEqual(adapter.attention_scope, "all_decode_tokens")
        prompt = Prompt(torch.tensor([[2, 60]]), [2, 60])
        original = adapter.model.forward
        for sequence, expected_answer in [([7, 62, 8, 61, 9, 63], "word9"), ([7, 8], "")]:
            trace = FullTraceCollector(prompt.token_ids)
            tokens = iter(sequence)
            def scripted_forward(*args, **kwargs):
                output = original(*args, **kwargs)
                if kwargs.get("past_key_values") is not None:
                    output.logits.fill_(-1000)
                    output.logits[:, -1, next(tokens)] = 1000
                return output
            with patch.object(adapter.model, "forward", side_effect=scripted_forward):
                result = adapter.generate(prompt, max_new_tokens=len(sequence),
                                          attention=AttentionRequest(capture="full"), observer=trace)
            self.assertEqual([s.token_id for s in trace.steps], sequence)
            self.assertEqual([s.index for s in trace.steps], list(range(len(sequence))))
            self.assertEqual(result.text, expected_answer)

    def test_invalid_scope_rejected_before_loading(self):
        with self.assertRaisesRegex(ValueError, "Unknown attention scope"):
            Qwen3Adapter(attention_scope="unknown")

    def test_real_decode_online_attention_mass_without_vectors(self):
        adapter = self.make_adapter(attention_scope="all_decode_tokens")
        prompt = Prompt(torch.tensor([[2, 11, 22]]), [2, 11, 22])
        collector = create_retrieval_collector('needle_attention_mass_v1',
            eligible_heads=adapter.eligible_heads, prompt_token_ids=prompt.token_ids,
            needle_span=NeedleSpan(1, 3))
        original = adapter.model.forward
        tokens = iter([11, 22, 63])
        def scripted_forward(*args, **kwargs):
            output = original(*args, **kwargs)
            if kwargs.get('past_key_values') is not None:
                output.logits.fill_(-1000)
                output.logits[:, -1, next(tokens)] = 1000
            return output
        def observed(step):
            self.assertEqual(step.layers, {})
            self.assertEqual(set(step.needle_attention_mass), {0, 1})
            collector.on_step(step)
        with patch.object(adapter.model, 'forward', side_effect=scripted_forward):
            result = adapter.generate(prompt, max_new_tokens=3,
                attention=AttentionRequest(needle_span=(1, 3)),
                observer=SimpleNamespace(on_step=observed))
        self.assertEqual(result.token_ids, [11, 22, 63])
        self.assertEqual(collector.qualifying_steps, 2)
        self.assertTrue(all(0 <= score <= 1 for score in collector.scores.values()))
        self.assertTrue(all(n == 2 for n in collector._counts.values()))

    def test_family_stopping_rules_and_qwen35_layer_discovery(self):
        for cls in (Qwen3Adapter, Qwen35Adapter):
            adapter = cls.__new__(cls)
            adapter._tokenizer = ToyTokenizer()
            adapter._eos_token_ids = {63}
            self.assertEqual(adapter._stop_reason(63), "eos")
            self.assertEqual(adapter._stop_reason(62), None if cls is Qwen3Adapter else "newline")
        legacy = Qwen35Adapter.__new__(Qwen35Adapter)
        legacy._tokenizer = ToyTokenizer()
        legacy.model = SimpleNamespace(model=SimpleNamespace(layers=[
            SimpleNamespace(block_type="linear_attention"), SimpleNamespace(block_type="full_attention")]))
        self.assertEqual(legacy._discover_attention_layers(), (1,))
        result = legacy._generation_result(Prompt(None, [2]), [9, 62], "newline")
        self.assertEqual(result.text, "word9")
        self.assertIsNone(result.raw_text)


if __name__ == "__main__":
    unittest.main()
