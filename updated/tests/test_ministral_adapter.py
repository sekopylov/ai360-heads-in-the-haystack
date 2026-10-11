"""Exercise the Ministral adapter with tiny random weights and no downloads."""
from __future__ import annotations

import unittest
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import torch
from transformers import (
    Ministral3Config,
    Mistral3Config,
    Mistral3ForConditionalGeneration,
)

from retrieval_heads.attention import AttentionRequest, FullTraceCollector
from retrieval_heads.attention import flash
from retrieval_heads.attention.prefill import resolve_prefill_backend
from retrieval_heads.models import (
    Ministral3EightBNoThinkingAdapter,
    available_models,
)
from retrieval_heads.models.base import Prompt
from retrieval_heads.experiment.prompting import (
    experiment_messages,
    ministral_context_only_messages,
)


class MinistralTokenizerFixture:
    eos_token_id = [2, 63]

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
            "return_dict": True,
        }
        assert "<context>\ncontext\n</context>" in messages[0]["content"]
        assert "sole source of truth" in messages[0]["content"]
        assert "<book>" not in messages[0]["content"]
        return {"input_ids": torch.tensor([[1, 5, 6]])}

    def encode(self, text, **kwargs):
        return [7]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(
            "" if skip_special_tokens and token in self.eos_token_id else f"token{token} "
            for token in ids
        ).strip()


def tiny_model() -> Mistral3ForConditionalGeneration:
    text = Ministral3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        eos_token_id=[2, 63],
        attention_dropout=0.0,
        rope_parameters={
            "rope_type": "yarn",
            "rope_theta": 1_000_000.0,
            "factor": 4.0,
            "original_max_position_embeddings": 32,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "llama_4_scaling_beta": 0.1,
        },
    )
    vision = {
        "model_type": "pixtral",
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "head_dim": 8,
        "image_size": 28,
        "patch_size": 14,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10_000.0},
    }
    config = Mistral3Config(
        text_config=text.to_dict(),
        vision_config=vision,
        spatial_merge_size=1,
        image_token_index=10,
    )
    return Mistral3ForConditionalGeneration(config).eval()


class MinistralAdapterTests(unittest.TestCase):
    def test_context_only_prompt_preserves_text_and_question(self):
        context = "Facts with\nnewlines and punctuation.Mr Green is mean."
        question = "Why is Mr Green disliked?"
        messages = ministral_context_only_messages(context, question)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        content = messages[0]["content"]
        self.assertIn(f"<context>\n{context}\n</context>", content)
        self.assertIn(f"Question: {question}\n", content)
        self.assertIn("Do not identify its source", content)
        self.assertIn("one short sentence or phrase", content)
        self.assertNotIn("<book>", content)
        self.assertNotIn("Based on the content of the book", content)

    def test_qwen_prompt_remains_historical(self):
        self.assertEqual(experiment_messages("context", "question?"), [
            {"role": "user", "content":
             "<book>context</book>\nBased on the content of the book, Question: question?\nAnswer:"}
        ])

    def make_adapter(self, **kwargs):
        torch.manual_seed(42)
        model = tiny_model()
        tokenizer = MinistralTokenizerFixture()
        with patch(
            "retrieval_heads.models.transformers_common.resolve_model_source",
            return_value=("fixture", True),
        ), patch(
            "retrieval_heads.models.transformers_common.AutoTokenizer.from_pretrained",
            return_value=tokenizer,
        ) as tokenizer_load, patch.object(
            Mistral3ForConditionalGeneration,
            "from_pretrained",
            return_value=model,
        ) as model_load:
            adapter = Ministral3EightBNoThinkingAdapter(
                device_map="cpu", dtype="float32", **kwargs
            )
        self.assertTrue(tokenizer_load.call_args.kwargs["local_files_only"])
        self.assertTrue(model_load.call_args.kwargs["local_files_only"])
        return adapter

    def test_registry_prompt_and_non_reasoning_policy(self):
        self.assertIn("ministral3_8b_no_thinking", available_models())
        adapter = self.make_adapter()
        self.assertEqual(
            adapter.model_id,
            "mistralai/Ministral-3-8B-Instruct-2512-BF16",
        )
        self.assertEqual(adapter.attention_scope, "all_decode_tokens")
        self.assertEqual(adapter._chat_template_kwargs(), {})
        self.assertEqual(adapter.eligible_heads, tuple((layer, head) for layer in range(2) for head in range(4)))
        prompt = adapter.encode_prompt("context", "question?")
        self.assertEqual(prompt.token_ids, [1, 5, 6])
        self.assertEqual(adapter.input_device.type, "cpu")

    def test_real_cached_decode_capture_mask_and_restore(self):
        adapter = self.make_adapter()
        prompt = Prompt(torch.tensor([[1, 5, 6]]), [1, 5, 6])
        trace = FullTraceCollector(prompt.token_ids)
        result = adapter.generate(
            prompt,
            max_new_tokens=3,
            attention=AttentionRequest(
                capture="full", blocked_heads=frozenset({(0, 1)})
            ),
            observer=trace,
        )
        self.assertTrue(result.token_ids)
        self.assertEqual(len(trace.steps), len(result.token_ids))
        self.assertEqual(adapter.model.config._attn_implementation, "sdpa")
        self.assertEqual(adapter.model.config.text_config._attn_implementation, "sdpa")
        self.assertFalse(adapter.attention._active)
        for step in trace.steps:
            probabilities = step.layers[0][1]
            torch.testing.assert_close(
                probabilities,
                torch.full_like(probabilities, 1 / len(probabilities)),
            )

    def test_eos_only_stopping_and_saved_raw_output(self):
        adapter = self.make_adapter()
        self.assertIsNone(adapter._stop_reason(9))
        self.assertEqual(adapter._stop_reason(63), "eos")
        result = adapter._generation_result(
            Prompt(None, [1]), [9, 63], "eos"
        )
        self.assertEqual(result.text, "token9")
        self.assertEqual(result.raw_text, "token9 token63")
        self.assertEqual(result.finish_reason, "eos")

    def test_flash_masking_path_uses_nested_language_decoder(self):
        adapter = self.make_adapter(
            prefill_attention="sdpa_flash", decode_attention="sdpa_flash"
        )
        adapter.model.set_attn_implementation(resolve_prefill_backend("sdpa_flash"))
        prompt = Prompt(torch.tensor([[1, 5, 6]]), [1, 5, 6])
        with patch.object(flash, "flash_kernel", return_value=nullcontext()) as kernel:
            result = adapter.generate(
                prompt,
                max_new_tokens=2,
                attention=AttentionRequest(blocked_heads=frozenset({(0, 1)})),
            )
        self.assertTrue(result.token_ids)
        self.assertGreater(kernel.call_count, 2)
        self.assertEqual(adapter.model.config._attn_implementation, flash.PREFILL_NAME)
        self.assertEqual(
            adapter.model.config.text_config._attn_implementation,
            flash.PREFILL_NAME,
        )

    def test_rejects_wrong_outer_architecture(self):
        adapter = Ministral3EightBNoThinkingAdapter.__new__(
            Ministral3EightBNoThinkingAdapter
        )
        adapter.model = MagicMock()
        adapter.model.config.model_type = "ministral3"
        with self.assertRaisesRegex(ValueError, "Mistral3 multimodal"):
            adapter._discover_attention_layers()


if __name__ == "__main__":
    unittest.main()
