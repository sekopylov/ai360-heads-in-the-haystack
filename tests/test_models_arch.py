"""Architecture-aware integration tests against the real checkpoints.

The first group is the reason this codebase looks the way it does: Qwen3.5-0.8B
is a *hybrid* model whose linear layers have no attention map at all, and the
census must reflect that instead of pretending every layer is scoreable.
"""

from __future__ import annotations

import pytest
import torch

from retrieval_heads.attention import (
    AttentionRecorder,
    HeadMasker,
    TokenMixerMasker,
    masked_token_mixers,
)
from retrieval_heads.haystack import HaystackBuilder, build_needle_sample
from retrieval_heads.scoring import score_instance
from retrieval_heads.utils import HeadRef

pytestmark = pytest.mark.integration

NEEDLE = "The best thing to do in San Francisco is to eat a sandwich in Dolores Park on a sunny day."
QUESTION = "What is the best thing to do in San Francisco?"


# --------------------------------------------------------------------------- census
def test_qwen35_is_hybrid_and_only_six_layers_are_scoreable(qwen35):
    _, _, info = qwen35
    assert info.is_hybrid
    assert info.num_layers == 24
    assert info.scoreable_layers == [3, 7, 11, 15, 19, 23]
    assert info.linear_layers == [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22]
    assert len(info.scoreable_layers) + len(info.linear_layers) == info.num_layers
    assert info.n_scoreable_heads == 48
    assert set(info.num_heads.values()) == {8}
    assert set(info.num_kv_heads.values()) == {2}
    assert info.head_dim == 256


def test_qwen35_layer_types_come_from_config(qwen35):
    _, _, info = qwen35
    assert info.layer_type(0) == "linear_attention"
    assert info.layer_type(3) == "full_attention"
    assert info.layer_type(23) == "full_attention"


def test_qwen3_is_dense_and_fully_scoreable(qwen3):
    _, _, info = qwen3
    assert not info.is_hybrid
    assert info.num_layers == 28
    assert info.scoreable_layers == list(range(28))
    assert info.linear_layers == []
    assert info.n_scoreable_heads == 448
    assert info.head_dim == 128


# --------------------------------------------------------------------------- capture
@pytest.mark.parametrize("method", ["output_attentions", "patch"])
def test_recorder_returns_one_map_per_scoreable_layer(qwen35, method):
    model, tokenizer, info = qwen35
    ids = tokenizer("The quick brown fox jumps over the lazy dog.", return_tensors="pt").input_ids
    out = model(input_ids=ids, use_cache=True)
    nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)

    recorder = AttentionRecorder(model, info, method=method)
    _, attn = recorder.forward(input_ids=nxt, past_key_values=out.past_key_values)

    assert sorted(attn) == info.scoreable_layers
    expected_kv = ids.shape[1] + 1
    for layer, tensor in attn.items():
        assert tensor.shape == (1, info.num_heads[layer], 1, expected_kv)
        rows = tensor[0, :, 0, :]
        assert torch.allclose(rows.sum(-1), torch.ones(info.num_heads[layer]), atol=1e-4)
        assert (rows >= 0).all()


def test_both_capture_methods_agree(qwen35):
    """The public API and the monkeypatched path must produce identical maps.

    Each capture needs its *own* prefill: a decode step appends to the KV cache it
    is handed, so reusing one cache for both captures silently compares a 12-token
    row against a 13-token one.
    """
    model, tokenizer, info = qwen35
    ids = tokenizer("Attention maps should not depend on how we capture them.",
                    return_tensors="pt").input_ids

    def capture(method: str) -> dict[int, torch.Tensor]:
        out = model(input_ids=ids, use_cache=True)
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        _, attn = AttentionRecorder(model, info, method=method).forward(
            input_ids=nxt, past_key_values=out.past_key_values)
        return attn

    a = capture("output_attentions")
    b = capture("patch")

    assert sorted(a) == sorted(b)
    for layer in a:
        assert a[layer].shape == b[layer].shape
        assert torch.allclose(a[layer], b[layer], atol=1e-6)


def test_recorder_on_dense_model_covers_every_layer(qwen3):
    model, tokenizer, info = qwen3
    ids = tokenizer("A dense model exposes every layer.", return_tensors="pt").input_ids
    out = model(input_ids=ids, use_cache=True)
    nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
    _, attn = AttentionRecorder(model, info, method="output_attentions").forward(
        input_ids=nxt, past_key_values=out.past_key_values)
    assert sorted(attn) == list(range(info.num_layers))


# --------------------------------------------------------------------------- masking
def test_head_masker_changes_logits_and_restores(qwen35):
    model, tokenizer, info = qwen35
    ids = tokenizer("Masking a head must change the logits.", return_tensors="pt").input_ids
    base = model(input_ids=ids).logits

    with HeadMasker(model, info, [HeadRef(3, 0)]):
        masked = model(input_ids=ids).logits
    assert not torch.allclose(base, masked)

    assert torch.equal(base, model(input_ids=ids).logits), "hooks were not removed"


def test_masking_more_heads_moves_logits_further(qwen35):
    model, tokenizer, info = qwen35
    ids = tokenizer("More heads masked should perturb more.", return_tensors="pt").input_ids
    base = model(input_ids=ids).logits
    deltas = []
    for k in (1, 4, 16):
        heads = info.scoreable_heads[:k]
        with HeadMasker(model, info, heads):
            deltas.append((base - model(input_ids=ids).logits).abs().max().item())
    # Every mask must actually change the logits, and masking the widest set must
    # perturb at least as much as masking one head.  Requiring strict monotonicity
    # at every step was flaky on a single sentence: a head can be redundant with
    # one already zeroed, which does not make the masking wrong.
    assert all(delta > 0 for delta in deltas), deltas
    assert deltas[-1] > deltas[0], deltas


def test_head_masker_rejects_non_scoreable_layer(qwen35):
    model, _, info = qwen35
    with pytest.raises(KeyError, match="no scoreable attention module"):
        HeadMasker(model, info, [HeadRef(0, 0)])   # layer 0 is a Gated DeltaNet


def test_token_mixer_masker_works_on_linear_and_attention_layers(qwen35):
    model, tokenizer, info = qwen35
    ids = tokenizer("Ablating whole layers should also matter.", return_tensors="pt").input_ids
    base = model(input_ids=ids).logits

    with TokenMixerMasker(model, info, [0]):          # linear layer
        linear_masked = model(input_ids=ids).logits
    with TokenMixerMasker(model, info, [3]):          # full-attention layer
        attn_masked = model(input_ids=ids).logits
    assert not torch.allclose(base, linear_masked)
    assert not torch.allclose(base, attn_masked)
    assert torch.equal(base, model(input_ids=ids).logits)


def test_masked_token_mixers_helper_restores(qwen35):
    model, tokenizer, info = qwen35
    ids = tokenizer("Context manager cleanup.", return_tensors="pt").input_ids
    base = model(input_ids=ids).logits
    with masked_token_mixers(model, info, info.linear_layers[:2]):
        assert not torch.allclose(base, model(input_ids=ids).logits)
    assert torch.equal(base, model(input_ids=ids).logits)


def test_zeroing_the_attention_row_equals_zeroing_the_oproj_slice(qwen35):
    """The masking shortcut must be *exactly* head pruning, not an approximation.

    ``HeadMasker`` never touches the attention kernel; it zeroes the head's slice
    of the tensor entering ``o_proj``.  Here we instead zero the head's softmax
    row inside a patched attention function -- the literal reading of "mask out
    the head" -- and require bit-comparable logits.
    """
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q35

    model, tokenizer, info = qwen35
    ids = tokenizer("Two ways to remove a head must agree.", return_tensors="pt").input_ids
    head = HeadRef(3, 1)

    with HeadMasker(model, info, [head]):
        via_oproj = model(input_ids=ids).logits

    original = q35.eager_attention_forward

    def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        out, weights = original(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
        if getattr(module, "layer_idx", None) == head.layer:
            weights = weights.clone()
            weights[:, head.head] = 0.0
            value_states = q35.repeat_kv(value, module.num_key_value_groups)
            out = torch.matmul(weights, value_states).transpose(1, 2).contiguous()
        return out, weights

    q35.eager_attention_forward = patched
    try:
        via_attention_row = model(input_ids=ids).logits
    finally:
        q35.eager_attention_forward = original

    assert torch.allclose(via_oproj, via_attention_row, atol=1e-5), (
        (via_oproj - via_attention_row).abs().max().item()
    )


# --------------------------------------------------------------------------- end to end
def test_chunked_prefill_matches_single_shot(qwen35):
    """Chunked prefill must be numerically equivalent, not merely plausible.

    SDPA on float32 can fall back to the math backend and materialise the whole
    ``(heads, seq, seq)`` matrix -- a 16K fp32 prefill asked for 20.6 GiB on an L4
    and OOM'd.  Feeding the prompt in chunks through the KV cache fixes the memory,
    but only if positions and the causal mask stay correct, so this test compares
    both the resulting logits and the greedy continuation.
    """
    from retrieval_heads.scoring import decode_with_attention

    model, tokenizer, info = qwen35
    text = ("The quick brown fox jumps over the lazy dog while the mirror "
            "reflects a candle near the window. ") * 12
    ids = tokenizer(text, return_tensors="pt").input_ids
    assert ids.shape[1] > 100, ids.shape

    full_trace, full_gen = decode_with_attention(
        model, info, ids, max_new_tokens=4, tokenizer=tokenizer, prefill_chunk=None)
    chunk_trace, chunk_gen = decode_with_attention(
        model, info, ids, max_new_tokens=4, tokenizer=tokenizer, prefill_chunk=16)

    assert full_gen == chunk_gen, f"greedy continuation diverged: {full_gen} vs {chunk_gen}"
    assert torch.allclose(full_trace.prefill_logits, chunk_trace.prefill_logits, atol=1e-3), \
        (full_trace.prefill_logits - chunk_trace.prefill_logits).abs().max().item()


def test_end_to_end_retrieval_detection_on_qwen35(qwen35):
    """The paper's whole pipeline on one instance: recite the needle, score heads."""
    model, tokenizer, info = qwen35
    sample = build_needle_sample(
        tokenizer, needle=NEEDLE, question=QUESTION, target_tokens=512, depth=0.5,
        builder=HaystackBuilder(seed=4),
    )
    result = score_instance(model, info, sample, tokenizer, max_new_tokens=24)

    assert result.n_steps > 0
    assert result.needle_recall > 0.5, f"model failed to recite the needle: {result.generated_text!r}"

    # Both pairings are scored from the one pass, and the strict in-order variant
    # is stored alongside the paper's rule.
    assert set(result.scores) == {"next_step", "same_step"}
    assert set(result.aligned_scores) == {"next_step", "same_step"}
    assert all(
        h in result.aligned_scores["next_step"] for h in map(str, info.scoreable_heads)
    )

    scores = torch.tensor([result.scores["next_step"][str(h)] for h in info.scoreable_heads])
    assert scores.max() > 0.1, "no head behaved like a retrieval head"
    # Sparsity on a *single* 512-token instance is noisy -- use scores are averaged
    # over ~18 instances for the real claim.  What one instance can establish is
    # that the distribution is structured rather than uniform: a few heads carry
    # the copy, most do not.
    assert float((scores > 0.1).float().mean()) < 0.9, "every head scoring is not structure"
    assert float((scores > 0.5).float().mean()) < 0.3, "strong retrieval heads should be sparse"


def test_end_to_end_retrieval_detection_on_qwen3(qwen3):
    model, tokenizer, info = qwen3
    sample = build_needle_sample(
        tokenizer, needle=NEEDLE, question=QUESTION, target_tokens=512, depth=0.5,
        builder=HaystackBuilder(seed=4),
    )
    result = score_instance(model, info, sample, tokenizer, max_new_tokens=24)

    assert result.needle_recall > 0.5, f"model failed to recite the needle: {result.generated_text!r}"
    scores = torch.tensor([result.scores["next_step"][str(h)] for h in info.scoreable_heads])
    assert scores.max() > 0.1


def test_haystack_domain_is_a_lower_bound_of_the_prompt_domain(qwen3):
    """The one relation the domain change must satisfy, on a real instance.

    Restricting the argmax to the haystack can only *add* credit -- every haystack
    position is a prompt position, so a head that earned credit under `prompt` still
    does, and a head whose prompt argmax was a template token can now earn more.  It
    is the direct check that the span is the context (not the whole prompt) and that
    the scoring actually reads it.
    """
    model, tokenizer, info = qwen3
    sample = build_needle_sample(
        tokenizer, needle=NEEDLE, question=QUESTION, target_tokens=512, depth=0.5,
        builder=HaystackBuilder(seed=4),
    )
    haystack = score_instance(model, info, sample, tokenizer, max_new_tokens=24,
                              argmax_domain="haystack")
    prompt = score_instance(model, info, sample, tokenizer, max_new_tokens=24,
                            argmax_domain="prompt")

    assert haystack.meta["argmax_domain"] == "haystack"
    assert haystack.meta["argmax_span"] == list(sample.haystack_span)
    shift = haystack.meta["argmax_domain_shift"]
    assert shift["positions"] > 0 and 0.0 <= shift["share"] <= 1.0
    for head in info.scoreable_heads:
        key = str(head)
        assert haystack.scores["next_step"][key] >= prompt.scores["next_step"][key] - 1e-9, (
            f"{head}: the haystack domain lost credit ({haystack.scores['next_step'][key]} < "
            f"{prompt.scores['next_step'][key]}); the span is not a subset of the prompt"
        )
    # The sink diagnostic is domain-independent: it is the same prompt argmax.
    assert haystack.sink_rate["next_step"]["__overall__"] == pytest.approx(
        prompt.sink_rate["next_step"]["__overall__"])


def test_masking_the_top_head_hurts(qwen35):
    """The paper's claim on a single instance: the strongest head matters.

    This is deliberately *not* the full top-K vs random-K experiment -- a single
    instance cannot measure the random arm (that needs the ~18-instance curve in
    ``masking_curve``).  The random-pool selection itself is pinned by
    ``tests/test_masking_regressions.py``.
    """
    model, tokenizer, info = qwen35
    sample = build_needle_sample(
        tokenizer, needle=NEEDLE, question=QUESTION, target_tokens=512, depth=0.5,
        builder=HaystackBuilder(seed=4),
    )
    baseline = score_instance(model, info, sample, tokenizer, max_new_tokens=24)
    ranked = sorted(info.scoreable_heads,
                    key=lambda h: -baseline.scores["next_step"][str(h)])
    top = ranked[0]

    from retrieval_heads.masking import evaluate_samples

    unmasked = evaluate_samples(model, tokenizer, info, [sample], max_new_tokens=24)
    masked_top = evaluate_samples(model, tokenizer, info, [sample], masked_heads=[top],
                                  max_new_tokens=24)
    assert unmasked.f1 >= masked_top.f1, "masking the strongest retrieval head should not help"

    # Self-contained guard against a no-op masker: `>=` alone is satisfied by an
    # inert hook.  Masking must at least change the logits on this same instance.
    ids = sample.input_ids
    base_logits = model(input_ids=ids).logits
    with HeadMasker(model, info, [top]):
        masked_logits = model(input_ids=ids).logits
    assert not torch.allclose(base_logits, masked_logits), "the mask did nothing"
