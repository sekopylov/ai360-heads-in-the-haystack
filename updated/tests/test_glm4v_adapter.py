"""Native tiny GLM-4.6V-Flash inference; no weights, GPU or downloads."""
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import Glm4vConfig, Glm4vForConditionalGeneration
from transformers.utils.chat_template_utils import _compile_jinja_template

from retrieval_heads.attention import AttentionRequest, FullTraceCollector
from retrieval_heads.attention import flash
from retrieval_heads.attention.prefill import resolve_prefill_backend
from retrieval_heads.models import Glm46VFlashNoThinkingAdapter, available_models
from retrieval_heads.experiment.prompting import context_only_messages, ministral_context_only_messages


class TokenizerFixture:
    eos_token_id = 2  # Model config also declares user/observation EOS IDs.

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(tokenize=True, add_generation_prompt=True,
                              return_tensors="pt", return_dict=True, enable_thinking=False)
        assert messages == context_only_messages("context", "question?")
        return {"input_ids": torch.tensor([[1, 5, 6, 10, 11]])}

    def encode(self, text, **kwargs):
        markers = {"<think>": 10, "</think>": 11, "<answer>": 12, "</answer>": 13}
        if text in markers:
            return [markers[text]]
        return [7]

    def decode(self, ids, skip_special_tokens=False):
        markers = {10: "<think>", 11: "</think>", 12: "<answer>", 13: "</answer>"}
        return " ".join(markers.get(i, str(i)) for i in ids
                        if not skip_special_tokens or i not in [2, 3, 4])


def tiny_glm4v():
    config = Glm4vConfig(text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=1, head_dim=8,
                        max_position_embeddings=128, attention_bias=True,
                        attention_dropout=0.0, eos_token_id=[2, 3, 4], pad_token_id=2,
                        rope_parameters={"rope_type": "default", "rope_theta": 500000.0,
                                         "partial_rotary_factor": 0.5,
                                         "mrope_section": [1, 1, 0]}),
                       vision_config=dict(depth=1, hidden_size=16, num_heads=2,
                                          intermediate_size=32, out_hidden_size=32,
                                          image_size=8, patch_size=2))
    return Glm4vForConditionalGeneration(config).eval()


class Glm4vAdapterTests(unittest.TestCase):
    def adapter(self, **kwargs):
        torch.manual_seed(42)
        model = tiny_glm4v()
        with patch("retrieval_heads.models.transformers_common.resolve_model_source",
                   return_value=("fixture", True)), patch(
            "retrieval_heads.models.transformers_common.AutoTokenizer.from_pretrained",
            return_value=TokenizerFixture(),
        ) as tokenizer_load, patch.object(Glm4vForConditionalGeneration, "from_pretrained", return_value=model) as model_load:
            adapter = Glm46VFlashNoThinkingAdapter(device_map="cpu", dtype="float32", **kwargs)
        self.assertTrue(tokenizer_load.call_args.kwargs["local_files_only"])
        self.assertFalse(tokenizer_load.call_args.kwargs["trust_remote_code"])
        self.assertTrue(model_load.call_args.kwargs["local_files_only"])
        model.set_attn_implementation(resolve_prefill_backend(adapter.prefill_attention))
        return adapter

    def test_registry_prompt_and_configuration(self):
        self.assertIn("glm4_6v_flash_no_thinking", available_models())
        self.assertNotIn("glm4_1v_9b", available_models())
        self.assertNotIn("glm4_9b_0414", available_models())
        adapter = self.adapter()
        self.assertEqual(adapter.model_id, "zai-org/GLM-4.6V-Flash")
        self.assertEqual(adapter.eligible_heads, tuple((l, h) for l in range(2) for h in range(4)))
        self.assertEqual(adapter.attention_scope, "all_decode_tokens")
        self.assertEqual(adapter.encode_prompt("context", "question?").token_ids, [1, 5, 6, 10, 11])
        self.assertEqual(adapter.input_device.type, "cpu")
        self.assertEqual(adapter._language_config().rope_parameters["partial_rotary_factor"], 0.5)
        self.assertEqual(adapter._language_config().rope_parameters["rope_theta"], 500000.0)
        self.assertEqual(context_only_messages("context", "question?"),
                         ministral_context_only_messages("context", "question?"))

    def test_all_checkpoint_eos_ids_and_unified_result(self):
        adapter = self.adapter()
        for token in [2, 3, 4]:
            self.assertEqual(adapter._stop_reason(token), "eos")
        self.assertIsNone(adapter._stop_reason(9))
        self.assertIsNone(adapter._stop_reason(13))  # Unlike 4.1V, </answer> is not EOS.
        result = adapter._generation_result(adapter.encode_prompt("context", "question?"), [9, 3], "eos")
        self.assertEqual((result.text, result.raw_text, result.finish_reason), ("9", "9 3", "eos"))
        self.assertTrue(result.final_answer_started)

    def test_thinking_output_and_truncated_answer(self):
        adapter = self.adapter()
        prompt = adapter.encode_prompt("context", "question?")
        for tokens, text, started in [([10, 8], "", False),
                                     ([10, 8, 11, 12, 9, 13], "9", True),
                                     ([10, 8, 11, 12, 9], "9", True),
                                     ([10, 8, 11, 9], "9", True),
                                     ([9], "9", True)]:
            result = adapter._generation_result(prompt, tokens, "length")
            self.assertEqual((result.text, result.final_answer_started), (text, started))
            self.assertEqual(result.token_ids, tokens)
        all_tokens = adapter._attention_filter(prompt)
        self.assertTrue(all_tokens(10))
        answer_adapter = self.adapter(attention_scope="answer_only")
        answer_tokens = answer_adapter._attention_filter(prompt)
        self.assertEqual([answer_tokens(t) for t in [10, 8, 11, 12, 9, 13]],
                         [False, False, False, False, True, False])

    def test_native_greedy_parity_and_capture(self):
        adapter = self.adapter()
        prompt = adapter.encode_prompt("context", "question?")
        adapter.model.set_attn_implementation("eager")
        with torch.inference_mode():
            expected = adapter.model.generate(prompt.input_ids, max_new_tokens=4, do_sample=False,
                                              attention_mask=torch.ones_like(prompt.input_ids))
        trace = FullTraceCollector(prompt.token_ids)
        result = adapter.generate(prompt, max_new_tokens=4,
            attention=AttentionRequest(capture="full", needle_span=(1, 3)), observer=trace)
        self.assertEqual(result.token_ids, expected[0, len(prompt.token_ids):].tolist())
        self.assertEqual(len(trace.steps), len(result.token_ids))
        for step in trace.steps:
            self.assertEqual(set(step.layers), {0, 1})
            for probabilities in step.layers.values():
                self.assertEqual(probabilities.shape[0], 4)
                torch.testing.assert_close(probabilities.sum(-1), torch.ones(4))
            self.assertEqual(set(step.needle_attention_mass), {0, 1})
        self.assertEqual(adapter.model.config._attn_implementation, "eager")
        self.assertFalse(adapter.attention._active)

    def test_uniform_and_zero_output_masks_restore_backend(self):
        for mode in ["legacy_uniform", "zero_output"]:
            adapter = self.adapter()
            prompt = adapter.encode_prompt("context", "question?")
            trace = FullTraceCollector(prompt.token_ids)
            adapter.generate(prompt, max_new_tokens=2, attention=AttentionRequest(
                capture="full", blocked_heads=frozenset({(0, 1)}), mask_mode=mode), observer=trace)
            if mode == "legacy_uniform":
                for step in trace.steps:
                    probs = step.layers[0][1]
                    torch.testing.assert_close(probs, torch.full_like(probs, 1 / probs.numel()))
            self.assertEqual(adapter.model.config._attn_implementation, "sdpa")
            self.assertFalse(adapter.attention._active)

    def test_flash_masking_cpu_emulation(self):
        adapter = self.adapter(prefill_attention="sdpa_flash", decode_attention="sdpa_flash")
        prompt = adapter.encode_prompt("context", "question?")
        with patch.object(flash, "flash_kernel", return_value=nullcontext()) as kernel:
            result = adapter.generate(prompt, max_new_tokens=2,
                attention=AttentionRequest(blocked_heads=frozenset({(0, 1)})))
        self.assertTrue(result.token_ids)
        self.assertGreaterEqual(kernel.call_count, 3)
        self.assertEqual(adapter.model.config._attn_implementation, flash.PREFILL_NAME)

    def test_wrong_architecture_and_context_limit(self):
        adapter = self.adapter()
        with self.assertRaisesRegex(ValueError, "context limit"):
            adapter.generate(adapter.encode_prompt("context", "question?"), max_new_tokens=126,
                             attention=AttentionRequest())
        adapter.model.config.model_type = "glm4_moe"
        with self.assertRaisesRegex(ValueError, "native glm4v"):
            adapter._discover_attention_layers()

    def test_template_ignoring_no_thinking_is_rejected(self):
        adapter = self.adapter()
        with patch.object(adapter.tokenizer, "apply_chat_template", return_value={
                "input_ids": torch.tensor([[1, 5, 6]])}):
            with self.assertRaisesRegex(ValueError, "enable_thinking=False support"):
                adapter.encode_prompt("context", "question?")

    def test_official_template_no_thinking_switch(self):
        # Exact official template snapshot fetched 2026-10-10 from:
        # https://huggingface.co/zai-org/GLM-4.6V-Flash/raw/main/chat_template.jinja
        template = _compile_jinja_template((Path(__file__).parent / "fixtures" /
                                           "glm46v_flash_chat_template.jinja").read_text())
        messages = context_only_messages("context", "question?")
        enabled = template.render(messages=messages, add_generation_prompt=True,
                                  enable_thinking=True)
        disabled = template.render(messages=messages, add_generation_prompt=True,
                                   enable_thinking=False)
        self.assertTrue(disabled.rstrip().endswith("<|assistant|>\n<think></think>"))
        self.assertEqual(disabled.count("/nothink"), 1)
        self.assertTrue(enabled.rstrip().endswith("<|assistant|>"))
        self.assertNotIn("/nothink", enabled)
        self.assertNotIn("<answer>", disabled)
        # The entire context-only instruction and supplied data are preserved.
        self.assertIn(messages[0]["content"], disabled)
