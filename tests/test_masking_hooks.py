"""Fast, checkpoint-free tests for the masking hooks and the decode trace.

The equivalent assertions in ``tests/test_models_arch.py`` are marked
``integration`` and never run in CI, so a regression in hook install/removal or in
the prefill-row plumbing would leave CI green.  These use toy ``nn.Module``\\ s and
a stub model instead of the real checkpoints.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from retrieval_heads.attention import (
    HeadMasker,
    TokenMixerMasker,
    masked_heads,
    masked_token_mixers,
)
from retrieval_heads.models import ModelInfo
from retrieval_heads.scoring import credits_from_trace, decode_with_attention
from retrieval_heads.utils import HeadRef


# --------------------------------------------------------------------------- helpers
class ToyAttention(nn.Module):
    """Minimal attention block: an ``o_proj`` whose input slices are the heads."""

    def __init__(self, heads: int = 2, head_dim: int = 3) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = heads
        self.o_proj = nn.Linear(heads * head_dim, heads * head_dim, bias=False)
        with torch.no_grad():
            self.o_proj.weight.copy_(torch.eye(heads * head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.o_proj(x)


class ToyKwAttention(nn.Module):
    """Same, but calls ``o_proj`` by keyword (the pre-hook must not IndexError)."""

    def __init__(self, heads: int = 2, head_dim: int = 3) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = heads
        self.o_proj = nn.Linear(heads * head_dim, heads * head_dim, bias=False)
        with torch.no_grad():
            self.o_proj.weight.copy_(torch.eye(heads * head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.o_proj(input=x)


def test_head_masker_handles_a_keyword_o_proj_call():
    module = ToyKwAttention()
    info = attention_info(module)
    x = torch.ones(1, 6)
    base = module(x)

    with masked_heads(object(), info, [HeadRef(0, 0)]):
        masked = module(x)

    assert torch.equal(masked[:, :3], torch.zeros(1, 3))
    assert torch.equal(masked[:, 3:], base[:, 3:])
    assert torch.equal(module(x), base)


class ToyMixer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1.0


def attention_info(module: nn.Module, *, heads: int = 2, head_dim: int = 3) -> ModelInfo:
    return ModelInfo(
        name="toy", path="toy", model_type="toy", num_layers=1,
        layer_types=["full_attention"], num_heads={0: heads}, num_kv_heads={0: heads},
        head_dim=head_dim, hidden_size=heads * head_dim, max_position_embeddings=64,
        attention_modules={0: module}, scoreable_layers_=[0],
    )


# --------------------------------------------------------------------------- hooks
def test_head_masker_zeroes_only_that_heads_slice_and_restores():
    module = ToyAttention()
    info = attention_info(module)
    x = torch.ones(1, 6)
    base = module(x)

    with masked_heads(object(), info, [HeadRef(0, 0)]):
        masked = module(x)

    assert torch.equal(masked[:, :3], torch.zeros(1, 3)), "head 0's slice was not zeroed"
    assert torch.equal(masked[:, 3:], base[:, 3:]), "head 1 was touched"
    assert torch.equal(module(x), base), "the hook was not removed"


def test_head_masker_rejects_a_layer_with_no_attention_module():
    module = ToyAttention()
    info = attention_info(module)
    try:
        HeadMasker(object(), info, [HeadRef(5, 0)])
    except KeyError as exc:
        assert "no scoreable attention module" in str(exc)
    else:  # pragma: no cover - the guard must fire
        raise AssertionError("HeadMasker accepted a layer with no attention module")


def test_token_mixer_masker_zeroes_a_layer_output_and_restores():
    module = ToyMixer()
    info = ModelInfo(
        name="toy", path="toy", model_type="toy", num_layers=1,
        layer_types=["linear_attention"], num_heads={}, num_kv_heads={},
        head_dim=0, hidden_size=3, max_position_embeddings=64,
        num_linear_heads={0: 2}, linear_modules={0: module}, linear_layers_=[0],
    )
    x = torch.ones(1, 3)

    with masked_token_mixers(object(), info, [0]):
        assert torch.equal(module(x), torch.zeros(1, 3))

    assert torch.equal(module(x), x + 1.0), "the hook was not removed"


# --------------------------------------------------------------------------- decode
def test_head_masker_rejects_a_mismatched_o_proj_width():
    """A gated/MLA block whose q_proj is wider than heads x head_dim must be caught."""
    class Odd(nn.Module):
        def __init__(self):
            super().__init__()
            self.head_dim = 3
            self.num_heads = 2
            self.o_proj = nn.Linear(12, 12, bias=False)   # four heads' worth of width

        def forward(self, x):
            return self.o_proj(x)

    module = Odd()
    info = attention_info(module)                          # expects 2 x 3 = 6
    with pytest.raises(RuntimeError, match="num_heads"):
        with masked_heads(object(), info, [HeadRef(0, 0)]):
            module(torch.ones(1, 12))


def test_token_mixer_masker_registers_nothing_when_a_layer_is_missing():
    """Validation happens before any hook, so a bad layer cannot leave hooks live."""
    module = ToyMixer()
    info = ModelInfo(
        name="toy", path="toy", model_type="toy", num_layers=1,
        layer_types=["linear_attention"], num_heads={}, num_kv_heads={},
        head_dim=0, hidden_size=3, max_position_embeddings=64,
        num_linear_heads={0: 2}, linear_modules={0: module}, linear_layers_=[0],
    )
    try:
        TokenMixerMasker(object(), info, [0, 5])       # layer 5 does not exist
    except KeyError:
        pass
    else:  # pragma: no cover - the guard must fire
        raise AssertionError("TokenMixerMasker accepted a missing layer")
    assert not module._forward_hooks, "layer 0's hook was left installed"


def test_evaluate_samples_does_not_leak_the_first_masker(monkeypatch):
    """If the second masker fails to build, the first must be uninstalled."""
    import retrieval_heads.masking as masking

    module = ToyAttention()
    info = attention_info(module)

    class Boom:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("cannot build the mixer")

    monkeypatch.setattr(masking, "TokenMixerMasker", Boom)
    try:
        masking.evaluate_samples(object(), None, info, [], masked_heads=[HeadRef(0, 0)],
                                 masked_layers=[0])
    except RuntimeError:
        pass
    else:  # pragma: no cover - the mixer constructor must be reached
        raise AssertionError("the failing mixer constructor was not called")
    assert not module.o_proj._forward_pre_hooks, "HeadMasker hooks leaked"


def test_discover_modules_skips_vision_towers():
    from retrieval_heads.models import discover_modules

    class Attention(nn.Module):
        def __init__(self, layer_idx, heads=2, head_dim=3):
            super().__init__()
            self.layer_idx = layer_idx
            self.head_dim = head_dim
            self.num_heads = heads
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                setattr(self, name, nn.Linear(heads * head_dim, heads * head_dim, bias=False))

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Attention(0), Attention(1)])
            # a vision tower whose layer_idx collides with the text stack
            self.vision_tower = nn.ModuleList([Attention(0), Attention(1)])

    attention, _linear, names = discover_modules(Tiny())
    assert sorted(attention) == [0, 1]
    assert all("vision" not in names[layer] for layer in attention)


class _FakeSample:
    def __init__(self, prompt_ids, needle_span):
        self.input_ids = torch.tensor([prompt_ids])
        self.needle_span = needle_span
        self.needle_ids = prompt_ids[needle_span[0]:needle_span[1]]
        self.needle_text = "needle"

    def as_dict(self):
        return {"prompt_tokens": len(self.needle_ids)}


def test_decode_with_attention_emits_the_next_step_prefill_row(monkeypatch):
    """The row that produces the first token must come out of `decode_with_attention`.

    The credits tests build that StepTrace by hand; this exercises the production
    path that is supposed to create it (and that used to skip the first token).
    """
    import retrieval_heads.scoring as scoring

    class Output:
        def __init__(self, logits, cache):
            self.logits = logits
            self.past_key_values = cache

    class StubModel(nn.Module):
        """Always predicts token 3; the cache is just the number of cached tokens."""

        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(eos_token_id=None)
            self.generation_config = SimpleNamespace(eos_token_id=None)
            self.dummy = nn.Parameter(torch.zeros(1))

        def forward(self, input_ids=None, past_key_values=None, use_cache=True, **kwargs):
            logits = torch.zeros(1, input_ids.shape[1], 5)
            logits[..., 3] = 1.0
            return Output(logits, int(past_key_values or 0) + int(input_ids.shape[1]))

    model = StubModel()
    info = attention_info(None)  # modules are never touched: the recorder is stubbed

    def make_recorder(m, i, method="output_attentions"):
        class Recorder:
            def forward(self, *, input_ids, past_key_values=None, use_cache=True,
                        attention_mask=None):
                kv_len = int(input_ids.shape[1]) + int(past_key_values or 0)
                row = torch.zeros(1, 1, 1, kv_len)
                row[..., 1] = 1.0                      # mass on prompt position 1
                out = m(input_ids=input_ids, past_key_values=past_key_values,
                        use_cache=use_cache)
                return out, {0: row}

        return Recorder()

    monkeypatch.setattr(scoring, "AttentionRecorder", make_recorder)
    monkeypatch.setattr(scoring, "set_attn_implementation", lambda *a, **k: {})
    monkeypatch.setattr(scoring, "restore_attn_implementation", lambda *a, **k: None)

    ids = torch.tensor([[0, 3, 0, 0, 0]])
    trace, generated = decode_with_attention(model, info, ids, max_new_tokens=2,
                                             tokenizer=None)

    assert generated == [3, 3]
    # The last step ran out of budget, so its never-emitted prediction is scoped to
    # same_step; next_step must not credit it.
    assert [s.applies_to for s in trace.steps] == [("next_step",), None, ("same_step",)]
    assert trace.steps[0].predicted_token == 3
    assert trace.steps[1].fed_token == 3
    # Default: only the per-head argmax is kept, not the (heads, kv_len) rows.
    assert trace.steps[0].attn == {}
    assert trace.steps[0].argmax[0].tolist() == [1]
    assert trace.kv_len == 5 + 2

    sample = _FakeSample([0, 3, 0, 0, 0], (1, 2))
    next_credits, _, _ = credits_from_trace(trace, sample, info, pairing="next_step")
    assert next_credits[HeadRef(0, 0)] == {3}, "the first generated token was not credited"
    same_credits, _, _ = credits_from_trace(trace, sample, info, pairing="same_step")
    assert same_credits[HeadRef(0, 0)] == {3}, "same_step must skip the prefill row"

    # store_rows=True keeps the full rows for the case-study figure.
    rows_trace, _ = decode_with_attention(model, info, ids, max_new_tokens=1,
                                          tokenizer=None, store_rows=True)
    assert rows_trace.steps[0].attn[0].shape == (1, 5)

    # A real EOS stop is reported, not guessed from the generated length: here the
    # first predicted token is EOS, so nothing is generated but the run did stop
    # on EOS rather than running out of budget.
    model.config.eos_token_id = 3
    eos_trace, eos_generated = decode_with_attention(model, info, ids, max_new_tokens=2,
                                                     tokenizer=None)
    assert eos_generated == []
    assert eos_trace.stopped_on_eos is True
    model.config.eos_token_id = None


def test_shared_prefill_cache_chunks_in_order():
    from retrieval_heads.generation import prefill_cache

    class CountingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(1))
            self.chunks: list[int] = []

        def forward(self, input_ids=None, past_key_values=None, use_cache=True, **kwargs):
            self.chunks.append(int(input_ids.shape[1]))
            logits = torch.zeros(1, input_ids.shape[1], 5)
            logits[..., 3] = 1.0
            return SimpleNamespace(
                logits=logits,
                past_key_values=int(past_key_values or 0) + int(input_ids.shape[1]),
            )

    ids = torch.zeros(1, 5, dtype=torch.long)

    model = CountingModel()
    cache, logits = prefill_cache(model, ids)
    assert model.chunks == [5] and cache == 5
    assert logits.argmax(-1).item() == 3

    model = CountingModel()
    cache, _ = prefill_cache(model, ids, prefill_chunk=2)
    assert model.chunks == [2, 2, 1], model.chunks
    assert cache == 5


def test_shared_greedy_ids_stops_at_eos():
    from retrieval_heads.generation import greedy_ids

    class StubModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(1))

        def forward(self, input_ids=None, past_key_values=0, use_cache=True, **kwargs):
            logits = torch.zeros(1, input_ids.shape[1], 5)
            logits[..., 3] = 1.0
            return SimpleNamespace(
                logits=logits,
                past_key_values=int(past_key_values or 0) + int(input_ids.shape[1]),
            )

    ids = torch.zeros(1, 5, dtype=torch.long)
    assert greedy_ids(StubModel(), ids, max_new_tokens=3, eos=set()) == [3, 3, 3]
    assert greedy_ids(StubModel(), ids, max_new_tokens=3, eos={3}) == []


class NonContigAttention(nn.Module):
    """Feeds ``o_proj`` a stride-2 view, so a naive `reshape` would copy."""

    def __init__(self, heads: int = 2, head_dim: int = 2) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = heads
        self.o_proj = nn.Linear(heads * head_dim, heads * head_dim, bias=False)
        with torch.no_grad():
            self.o_proj.weight.copy_(torch.eye(heads * head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        view = x.view(x.shape[0], 4, 2)[..., 0]   # (1, 4), stride (8, 2)
        assert view.stride(-1) != 1
        return self.o_proj(view)


def test_head_masker_works_on_a_non_contiguous_input():
    module = NonContigAttention()
    info = attention_info(module, heads=2, head_dim=2)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
    base = module(x)
    assert base.tolist() == [[1.0, 3.0, 5.0, 7.0]]

    with masked_heads(object(), info, [HeadRef(0, 0)]):
        masked = module(x)

    # Head 0 owns columns 0..1 of the o_proj input; without the contiguity fix the
    # zeroing lands in a copy and `masked` comes back unchanged.
    assert masked.tolist() == [[0.0, 0.0, 5.0, 7.0]], masked
    assert torch.equal(module(x), base)


def test_patch_capture_keys_on_the_scored_module_not_layer_idx(monkeypatch):
    """A foreign block sharing a layer_idx must not overwrite the scored row."""
    import sys
    import types

    from retrieval_heads.attention import AttentionRecorder

    fake = types.ModuleType("fake_modeling_for_patch_test")

    def eager(module_, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        weights = torch.ones(1, getattr(module_, "num_heads", 1), 1, 2)
        return torch.zeros(1, 1, 1, 1), weights

    fake.eager_attention_forward = eager
    monkeypatch.setitem(sys.modules, fake.__name__, fake)

    def make(name):
        return type(name, (nn.Module,), {
            "__module__": fake.__name__,
            "layer_idx": 3,
            "num_heads": 1,
            "head_dim": 1,
        })()

    scoreable, foreign = make("Scoreable"), make("Foreign")
    info = ModelInfo(
        name="toy", path="toy", model_type="toy", num_layers=4,
        layer_types=["full_attention"] * 4, num_heads={3: 1}, num_kv_heads={3: 1},
        head_dim=1, hidden_size=1, max_position_embeddings=64,
        attention_modules={3: scoreable}, scoreable_layers_=[3],
    )
    recorder = AttentionRecorder(model=None, info=info, method="patch")
    store: dict[int, torch.Tensor] = {}
    with recorder._patched_eager(store):
        fake.eager_attention_forward(foreign, None, None, None, None, 1.0)
        assert store == {}, "a foreign module wrote into the capture store"
        fake.eager_attention_forward(scoreable, None, None, None, None, 1.0)
        assert 3 in store


def test_output_attentions_head_count_mismatch_is_rejected():
    from types import SimpleNamespace

    from retrieval_heads.attention import AttentionRecorder

    class Stub(nn.Module):
        def __init__(self, heads_returned: int) -> None:
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(1))
            self.attn = (torch.zeros(1, heads_returned, 1, 2),)

        def forward(self, **kwargs):
            return SimpleNamespace(attentions=self.attn, logits=torch.zeros(1, 1, 2),
                                   past_key_values=None)

    info = attention_info(None)                       # expects 2 heads on layer 0
    recorder = AttentionRecorder(model=Stub(3), info=info, method="output_attentions")
    with pytest.raises(RuntimeError, match="heads"):
        recorder.forward(input_ids=torch.zeros(1, 1, dtype=torch.long))


class _CacheAware(nn.Module):
    """Logits depend on the cache length, so a frozen cache changes the output."""

    def __init__(self, vocab: int = 32) -> None:
        super().__init__()
        self.vocab = vocab
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids=None, past_key_values=0, use_cache=True, **kwargs):
        kv_len = int(input_ids.shape[1]) + int(past_key_values or 0)
        logits = torch.full((1, input_ids.shape[1], self.vocab), -1.0)
        logits[..., kv_len % self.vocab] = 1.0
        return SimpleNamespace(logits=logits, past_key_values=kv_len)


def test_greedy_ids_advances_the_cache_each_step():
    from retrieval_heads.generation import greedy_ids

    ids = torch.zeros(1, 4, dtype=torch.long)
    # prompt 4 -> predicts 4; with the cache growing, then 5, then 6.  If the loop
    # forgot `cache = out.past_key_values`, every step would predict 5.
    assert greedy_ids(_CacheAware(), ids, max_new_tokens=3, eos=set()) == [4, 5, 6]


def test_decode_with_attention_advances_the_cache_each_step(monkeypatch):
    import retrieval_heads.scoring as scoring

    model = _CacheAware()
    info = attention_info(None)

    def make_recorder(m, i, method="patch"):
        class Recorder:
            def forward(self, *, input_ids, past_key_values=None, use_cache=True):
                out = m(input_ids=input_ids, past_key_values=past_key_values,
                        use_cache=use_cache)
                kv_len = int(out.past_key_values)
                row = torch.zeros(1, 1, 1, kv_len)
                row[..., 0] = 1.0
                return out, {0: row}

        return Recorder()

    monkeypatch.setattr(scoring, "AttentionRecorder", make_recorder)
    monkeypatch.setattr(scoring, "set_attn_implementation", lambda *a, **k: {})
    monkeypatch.setattr(scoring, "restore_attn_implementation", lambda *a, **k: None)

    ids = torch.zeros(1, 4, dtype=torch.long)
    trace, generated = decode_with_attention(model, info, ids, max_new_tokens=3,
                                             tokenizer=None, stop_on_eos=False)
    assert generated == [4, 5, 6], generated
