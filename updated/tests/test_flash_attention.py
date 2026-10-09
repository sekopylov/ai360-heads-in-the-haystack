from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from retrieval_heads.attention import AttentionRequest
from retrieval_heads.attention import flash
from retrieval_heads.attention.backend import observable_eager_attention
from retrieval_heads.attention.controller import AttentionController
from retrieval_heads.attention.prefill import resolve_prefill_backend
from retrieval_heads.models.qwen3_8b import Qwen3EightBAdapter
from retrieval_heads.models.base import Prompt
from tests.test_qwen_adapters import ToyTokenizer


class FlashAttentionTests(unittest.TestCase):
    def test_compact_gqa_baseline_and_uniform_match_eager(self):
        torch.manual_seed(42)
        query = torch.randn(1, 8, 1, 16)
        key = torch.randn(1, 2, 37, 16)
        value = torch.randn_like(key)
        for blocked in (frozenset(), frozenset({(0, 1), (0, 6)})):
            controller = AttentionController((0,))
            controller.start(AttentionRequest(blocked_heads=blocked))
            controller.begin_step()
            args = (SimpleNamespace(layer_idx=0, num_key_value_groups=4, training=False),
                    query, key, value, None)
            eager, _ = observable_eager_attention(*args, retrieval_attention_controller=controller)
            with patch.object(flash, "flash_kernel", return_value=nullcontext()):
                result, probabilities = flash.flash_decode(*args, retrieval_attention_controller=controller)
            torch.testing.assert_close(result, eager, rtol=1e-5, atol=1e-6)
            self.assertIsNone(probabilities)
            controller.end_step(0)
            controller.finish()

    def test_rejects_unsupported_requests_and_cpu(self):
        for request in (AttentionRequest(capture="top1"),
                        AttentionRequest(needle_span=(1, 2)),
                        AttentionRequest(mask_mode="zero_output")):
            with self.assertRaises(ValueError):
                flash.validate_flash_request(request)
        with self.assertRaises(ValueError):
            flash.validate_flash_request(AttentionRequest(), observer=object())
        with self.assertRaisesRegex(ValueError, "Ampere"):
            flash.flash_kernel(torch.device("cpu"))

    def test_real_adapter_prefill_decode_and_restore(self):
        torch.manual_seed(42)
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=2, num_attention_heads=4,
                            num_key_value_heads=2, head_dim=8, max_position_embeddings=128)
        model = Qwen3ForCausalLM(config).eval()
        with patch('retrieval_heads.models.qwen_common.AutoTokenizer.from_pretrained',
                   return_value=ToyTokenizer()), \
             patch.object(Qwen3ForCausalLM, 'from_pretrained', return_value=model):
            adapter = Qwen3EightBAdapter(device_map='cpu', dtype='float32',
                                        prefill_attention='sdpa_flash', decode_attention='sdpa_flash')
        model.set_attn_implementation(resolve_prefill_backend('sdpa_flash'))
        prompt = Prompt(input_ids=torch.tensor([[1, 2, 3, 4]]), token_ids=[1, 2, 3, 4])
        request = AttentionRequest(blocked_heads=frozenset({(0, 1)}))
        with patch.object(flash, 'flash_kernel', return_value=nullcontext()) as kernel:
            output = adapter.generate(prompt, max_new_tokens=2, attention=request)
        self.assertGreaterEqual(len(output.token_ids), 1)
        self.assertGreater(kernel.call_count, 2)
        self.assertEqual(model.config._attn_implementation, flash.PREFILL_NAME)
        with patch.object(model, 'forward', side_effect=AssertionError('must fail before prefill')):
            with self.assertRaisesRegex(ValueError, 'masking-only'):
                adapter.generate(prompt, max_new_tokens=1, attention=AttentionRequest(capture='full'))
