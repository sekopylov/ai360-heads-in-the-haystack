"""Fast tests against a tiny locally-built hybrid model -- no checkpoint download.

The architecture-specific behaviour (discovery on a real hybrid config, both
attention-capture paths, masking a real ``o_proj``, chunked-prefill equivalence)
otherwise lives only in the `integration` suite, which CI never runs.  A four-layer
Qwen3.5-style model with ~50k parameters closes that gap in a couple of seconds.
"""

from __future__ import annotations

import shutil

import pytest
import torch

from retrieval_heads.attention import HeadMasker, masked_heads
from retrieval_heads.generation import greedy_ids, prefill_cache
from retrieval_heads.scoring import decode_with_attention
from retrieval_heads.utils import HeadRef
from tests.conftest import MODEL_DIRS


def _ids(rows: int = 1, length: int = 24, vocab: int = 64) -> torch.LongTensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randint(1, vocab, (rows, length), generator=generator)


def test_discovery_sees_a_real_hybrid_layout(tiny_hybrid):
    model, info = tiny_hybrid
    assert info.scoreable_layers == [2]
    assert info.linear_layers == [0, 1, 3]
    assert info.is_hybrid
    assert info.num_heads == {2: 4}
    assert info.head_dim == 8
    assert info.model_class == "Qwen3_5ForCausalLM"
    assert info.n_all_heads == 4 + 3 * 4


def test_both_capture_methods_agree_on_the_tiny_model(tiny_hybrid):
    model, info = tiny_hybrid
    ids = _ids()

    def capture(method: str):
        trace, _ = decode_with_attention(model, info, ids, max_new_tokens=3,
                                         capture_method=method, stop_on_eos=False)
        return trace

    public = capture("output_attentions")
    patched = capture("patch")

    assert len(public.steps) == len(patched.steps)
    for a, b in zip(public.steps, patched.steps):
        assert a.positions()[2].tolist() == b.positions()[2].tolist()


def test_head_masker_changes_logits_on_a_real_oproj_and_restores(tiny_hybrid):
    model, info = tiny_hybrid
    ids = _ids()

    base = model(input_ids=ids).logits
    with masked_heads(model, info, [HeadRef(2, 0)]):
        masked = model(input_ids=ids).logits
        assert not torch.allclose(base, masked), "masking head L2H0 changed nothing"
    assert torch.equal(model(input_ids=ids).logits, base), "hooks were not removed"


def test_head_masker_rejects_a_module_from_another_model(tiny_hybrid):
    model, info = tiny_hybrid
    from retrieval_heads.models import build_model_info

    other = type(model)(model.config)
    other.eval()
    # `other` has the same architecture but different modules; masking through the
    # first model's info must not silently touch the wrong tensors.
    try:
        HeadMasker(other, info, [HeadRef(2, 0)])
    except KeyError as exc:
        assert "does not belong" in str(exc)
    else:  # pragma: no cover - the guard must fire
        raise AssertionError("HeadMasker accepted modules from another model instance")


def test_chunked_prefill_matches_one_shot_on_the_tiny_model(tiny_hybrid):
    model, info = tiny_hybrid
    ids = _ids(length=40)

    full_cache, full_logits = prefill_cache(model, ids)
    chunk_cache, chunk_logits = prefill_cache(model, ids, prefill_chunk=8)

    assert torch.allclose(full_logits, chunk_logits, atol=1e-4), \
        (full_logits - chunk_logits).abs().max().item()
    assert greedy_ids(model, ids, max_new_tokens=4, eos=set()) == \
        greedy_ids(model, ids, max_new_tokens=4, eos=set(), prefill_chunk=8)


def test_truncated_run_scopes_the_last_step_to_same_step(tiny_hybrid):
    model, info = tiny_hybrid
    trace, generated = decode_with_attention(model, info, _ids(), max_new_tokens=3,
                                             stop_on_eos=False)
    assert len(generated) == 3
    assert trace.stopped_on_eos is False
    scopes = [step.applies_to for step in trace.steps]
    assert scopes[0] == ("next_step",), "the prefill row must stay with next_step"
    assert scopes[-1] == ("same_step",), "the never-emitted prediction must be dropped"
    assert trace.kv_len == 24 + 3


def test_load_model_records_class_and_dtype(tiny_hybrid, tmp_path):
    """End-to-end load: the class that actually ran and the dtype are recorded."""
    from retrieval_heads.models import load_model

    model, _info = tiny_hybrid
    tokenizer_dir = MODEL_DIRS["qwen3-0.6b"]
    if not (tokenizer_dir / "tokenizer.json").exists():
        pytest.skip("Qwen3-0.6B tokenizer files are not available")

    model.save_pretrained(tmp_path)
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                 "generation_config.json"):
        source = tokenizer_dir / name
        if source.exists():
            shutil.copy(source, tmp_path / name)

    _model, _tokenizer, info = load_model(str(tmp_path), dtype="float32",
                                          attn_implementation="eager")
    assert info.model_class == "Qwen3_5ForCausalLM"
    assert info.dtype == "float32"
    assert info.scoreable_layers == [2]


def test_zeroing_the_attention_row_equals_zeroing_the_oproj_slice(tiny_hybrid):
    """The masking shortcut must be exact head pruning, not an approximation.

    Same check as the integration test on Qwen3.5, but on the tiny local hybrid so
    the fast CI suite covers it too.
    """
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q35

    from retrieval_heads.attention import restore_attn_implementation, set_attn_implementation

    model, info = tiny_hybrid
    ids = _ids(length=16)
    head = HeadRef(2, 1)
    previous = set_attn_implementation(model, "eager")
    try:
        base = model(input_ids=ids).logits
        with HeadMasker(model, info, [head]):
            via_oproj = model(input_ids=ids).logits

        original = q35.eager_attention_forward

        def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
            out, weights = original(module, query, key, value, attention_mask,
                                    scaling, dropout, **kwargs)
            if getattr(module, "layer_idx", None) == head.layer:
                weights = weights.clone()
                weights[:, head.head] = 0.0
                value_states = q35.repeat_kv(value, module.num_key_value_groups)
                out = torch.matmul(weights, value_states).transpose(1, 2).contiguous()
            return out, weights

        q35.eager_attention_forward = patched
        try:
            via_row = model(input_ids=ids).logits
        finally:
            q35.eager_attention_forward = original
    finally:
        restore_attn_implementation(model, previous)

    assert not torch.allclose(base, via_oproj), "the head mask changed nothing"
    assert torch.allclose(via_oproj, via_row, atol=1e-5), \
        (via_oproj - via_row).abs().max().item()
