import importlib.util
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from retrieval_heads.attention import AttentionRequest
from retrieval_heads.attention.backend import observable_eager_attention
from retrieval_heads.attention.controller import AttentionController

spec = importlib.util.spec_from_file_location(
    "a100_smoke", Path(__file__).resolve().parents[1] / "datasphere/a100_mask_smoke.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class FlashSmokeTests(unittest.TestCase):
    def test_backend_names_do_not_trigger_external_flash_import(self):
        from transformers import AttentionInterface, AttentionMaskInterface, Qwen3Config, Qwen3ForCausalLM
        from transformers.masking_utils import sdpa_mask
        model = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=32, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=1, head_dim=16))
        for name, function in ((smoke.FLASH_PREFILL, smoke.flash_prefill),
                               (smoke.FLASH_DECODE, smoke.flash_decode)):
            AttentionInterface.register(name, function)
            AttentionMaskInterface.register(name, sdpa_mask)
            model.set_attn_implementation(name)
            self.assertEqual(model.config._attn_implementation, name)

    def test_gqa_uniform_matches_eager(self):
        torch.manual_seed(42)
        query = torch.randn(1, 8, 1, 16)
        key = torch.randn(1, 2, 37, 16)
        value = torch.randn_like(key)
        controller = AttentionController((0,))
        controller.start(AttentionRequest(blocked_heads=frozenset({(0, 1), (0, 6)})))
        controller.begin_step()
        module = SimpleNamespace(layer_idx=0, num_key_value_groups=4, training=False)
        args = (module, query, key, value, None)
        eager, _ = observable_eager_attention(*args, retrieval_attention_controller=controller)
        with patch.object(smoke, "sdpa_kernel", return_value=nullcontext()):
            flash, _ = smoke.flash_decode(*args, retrieval_attention_controller=controller)
        torch.testing.assert_close(flash, eager, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(flash[0, 0, 1], value[0, 0].mean(0))
        torch.testing.assert_close(flash[0, 0, 6], value[0, 1].mean(0))
        self.assertFalse(torch.equal(query[:, 1], torch.zeros_like(query[:, 1])))

    def test_rejects_capture(self):
        controller = AttentionController((0,))
        controller.start(AttentionRequest(capture="top1"))
        query = torch.randn(1, 2, 1, 16)
        with self.assertRaisesRegex(ValueError, "masking-only"):
            smoke.flash_decode(SimpleNamespace(layer_idx=0), query, query, query, None,
                               retrieval_attention_controller=controller)

    def test_rejects_explicit_mask(self):
        query = torch.randn(1, 2, 1, 16)
        with self.assertRaisesRegex(ValueError, "single-token"):
            smoke.flash_decode(None, query, query, query, torch.zeros(1, 1, 1, 1))
