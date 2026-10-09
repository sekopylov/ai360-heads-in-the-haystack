import importlib.util
from pathlib import Path
import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

spec = importlib.util.spec_from_file_location(
    "manual_model_example",
    Path(__file__).resolve().parents[1] / "examples/run_qwen3_by_hand.py")
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


class ManualModelExampleTests(unittest.TestCase):
    def test_manual_loop_matches_transformers_greedy(self):
        torch.manual_seed(42)
        config = Qwen3Config(
            vocab_size=32, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=8, max_position_embeddings=64,
            bos_token_id=1, eos_token_id=None, pad_token_id=0,
        )
        model = Qwen3ForCausalLM(config).eval()
        model.set_attn_implementation("eager")
        input_ids = torch.tensor([[1, 2, 3, 4]])
        lengths = []
        def before(module, args, kwargs):
            lengths.append(kwargs["input_ids"].shape[1])
        handle = model.register_forward_pre_hook(before, with_kwargs=True)
        try:
            manual = example.greedy_forward(model, input_ids, 8, set())
        finally:
            handle.remove()
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                do_sample=False, max_new_tokens=8,
                use_cache=True, pad_token_id=0,
            )
        self.assertEqual(manual, generated[0, input_ids.shape[1]:].tolist())
        self.assertEqual(lengths, [4] + [1] * 7)

    def test_manual_loop_stops_at_eos(self):
        torch.manual_seed(42)
        model = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=16, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=1, head_dim=8, max_position_embeddings=32,
        )).eval()
        input_ids = torch.tensor([[1, 2, 3]])
        with torch.inference_mode():
            first = int(model(input_ids=input_ids, logits_to_keep=1).logits[0, -1].argmax())
        generated = example.greedy_forward(model, input_ids, 8, {first})
        self.assertEqual(generated, [first])
