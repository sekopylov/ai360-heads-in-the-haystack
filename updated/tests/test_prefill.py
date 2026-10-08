from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import torch
from torch.nn.attention import SDPBackend
from transformers import Qwen3Config, Qwen3ForCausalLM

from retrieval_heads.attention.prefill import (
    _efficient_kernel, memory_efficient_prefill, resolve_prefill_backend,
    MEMORY_EFFICIENT_NAME,
)


class PrefillTests(unittest.TestCase):
    def test_expanded_kv_matches_gqa_and_preserves_inputs(self):
        torch.manual_seed(42)
        q = torch.randn(1, 4, 6, 8)
        k, v = torch.randn(1, 2, 6, 8), torch.randn(1, 2, 6, 8)
        originals = k.clone(), v.clone()
        for mask in (None, torch.ones(6, 6, dtype=torch.bool).tril()):
            with patch('retrieval_heads.attention.prefill._efficient_kernel', return_value=nullcontext()), \
                 patch('retrieval_heads.attention.prefill.F.scaled_dot_product_attention',
                       wraps=torch.nn.functional.scaled_dot_product_attention) as call:
                out, weights = memory_efficient_prefill(SimpleNamespace(is_causal=True), q, k, v, mask)
            self.assertFalse(call.call_args.kwargs['enable_gqa'])
            self.assertEqual(call.call_args.args[1].shape[1], 4)
            expected = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, is_causal=mask is None, enable_gqa=True)
            torch.testing.assert_close(out, expected.transpose(1, 2))
            self.assertIsNone(weights)
            torch.testing.assert_close(k, originals[0])
            torch.testing.assert_close(v, originals[1])

    def test_only_efficient_kernel_enabled(self):
        with patch('retrieval_heads.attention.prefill.sdpa_kernel') as kernel:
            _efficient_kernel(torch.device('cuda'))
        kernel.assert_called_once_with(backends=[SDPBackend.EFFICIENT_ATTENTION])
        with self.assertRaisesRegex(ValueError, 'requires CUDA'):
            _efficient_kernel(torch.device('cpu'))

    def test_explicit_selection_and_standard_backends_unchanged(self):
        self.assertEqual(resolve_prefill_backend('sdpa_memory_efficient'), MEMORY_EFFICIENT_NAME)
        for name in ('sdpa', 'eager', 'flash_attention_2'):
            self.assertEqual(resolve_prefill_backend(name), name)

    def test_real_model_prefill_logits_and_compact_cache(self):
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=2, num_attention_heads=4,
                            num_key_value_heads=2, head_dim=8)
        config._attn_implementation = 'sdpa'
        model = Qwen3ForCausalLM(config).eval()
        ids = torch.tensor([[2, 3, 4, 5, 6]])
        with torch.inference_mode():
            standard = model(input_ids=ids, use_cache=True, logits_to_keep=1)
            model.set_attn_implementation(resolve_prefill_backend('sdpa_memory_efficient'))
            with patch('retrieval_heads.attention.prefill._efficient_kernel', return_value=nullcontext()):
                efficient = model(input_ids=ids, use_cache=True, logits_to_keep=1)
        torch.testing.assert_close(standard.logits, efficient.logits)
        for layer in efficient.past_key_values.layers:
            self.assertEqual(layer.keys.shape[1], 2)
