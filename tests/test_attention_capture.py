"""Attention capture: how the eager kernel is found and what it writes.

`AttentionRecorder.method="patch"` wraps a *module global* named
``eager_attention_forward``.  Which module that is depends on how the modeling code
resolved the name, and getting it wrong is silent until the recorder reports that it
captured nothing -- after the whole prefill has run.  These tests pin both aliasing
forms that the original single-namespace patch missed.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch

from retrieval_heads.attention import AttentionRecorder
from retrieval_heads.models import ModelInfo


def _info(module) -> ModelInfo:
    return ModelInfo(
        name="toy", path="toy", model_type="toy", num_layers=1,
        layer_types=["full_attention"], num_heads={0: 1}, num_kv_heads={0: 1},
        head_dim=2, hidden_size=2, max_position_embeddings=16,
        attention_modules={0: module},
    )


def _fake_modeling(name: str, source: str) -> types.ModuleType:
    module = types.ModuleType(name)
    exec(source, module.__dict__)  # noqa: S102 - the test's own source
    sys.modules[name] = module
    return module


_EAGER = """
import torch

def eager_attention_forward(module, query, key, value, attention_mask, scaling,
                            dropout=0.0, **kwargs):
    weights = torch.full((1, 3), 0.5)
    return "out", weights
"""


def test_patch_capture_reaches_a_forward_defined_in_another_module():
    """A subclass whose `forward` lives elsewhere must still be captured.

    ``type(module).__module__`` is the subclass's module, which has no
    ``eager_attention_forward`` at all; the name is looked up in the module the
    method was *defined* in, i.e. ``forward.__globals__``.
    """
    base = _fake_modeling("rh_fake_base_module", _EAGER + """
class Base:
    layer_idx = 0
    def forward(self, *args, **kwargs):
        return eager_attention_forward(*args, **kwargs)
""")
    subclass = _fake_modeling("rh_fake_subclass_module", """
from rh_fake_base_module import Base

class Block(Base):
    pass
""")
    block = subclass.Block()
    assert not hasattr(subclass, "eager_attention_forward"), "the test is not testing aliasing"
    assert base.eager_attention_forward.__name__ == "eager_attention_forward"

    recorder = AttentionRecorder(model=None, info=_info(block), method="patch")
    store: dict[int, torch.Tensor] = {}
    with recorder._patched_eager(store):
        # The first argument is the attention module itself, exactly as
        # `transformers` calls `eager_attention_forward(module, query, ...)`.
        block.forward(block, "q", "k", "v", None, 1.0)
    assert 0 in store and store[0].shape == (1, 3)
    # Restored exactly, even for the aliased namespace.
    assert base.eager_attention_forward.__name__ == "eager_attention_forward"


def test_patch_capture_reaches_a_module_object_call():
    """`import x` + `x.eager_attention_forward(...)` must be intercepted too.

    The name is not in the calling module's globals at all, so a patch keyed on the
    class's module found nothing and the recorder raised "could not locate a
    module-level eager_attention_forward".
    """
    base = _fake_modeling("rh_fake_callee_module", _EAGER)
    caller = _fake_modeling("rh_fake_caller_module", """
import rh_fake_callee_module as callee

class Block:
    layer_idx = 0
    def forward(self, *args, **kwargs):
        return callee.eager_attention_forward(*args, **kwargs)
""")
    block = caller.Block()
    recorder = AttentionRecorder(model=None, info=_info(block), method="patch")
    store: dict[int, torch.Tensor] = {}
    with recorder._patched_eager(store):
        block.forward(block, "q", "k", "v", None, 1.0)
    assert 0 in store and store[0].shape == (1, 3)
    assert base.eager_attention_forward.__name__ == "eager_attention_forward"


def test_patch_capture_ignores_a_foreign_block_with_the_same_layer_idx():
    """Only the model's own attention modules may write into the store."""
    base = _fake_modeling("rh_fake_known_module", _EAGER + """
class Block:
    layer_idx = 0
    def forward(self, *args, **kwargs):
        return eager_attention_forward(*args, **kwargs)
""")
    known = base.Block()
    foreign = base.Block()          # same class, same layer_idx, not in `info`
    recorder = AttentionRecorder(model=None, info=_info(known), method="patch")
    store: dict[int, torch.Tensor] = {}
    with recorder._patched_eager(store):
        foreign.forward(foreign, "q", "k", "v", None, 1.0)
    assert store == {}, "a foreign attention block overwrote the scored layer"


def test_patch_capture_fails_loudly_when_there_is_nothing_to_patch():
    """A model with no reachable eager function must say so, not capture nothing."""
    plain = _fake_modeling("rh_fake_plain_module", "class Block:\n    pass\n")
    recorder = AttentionRecorder(model=None, info=_info(plain.Block()), method="patch")
    with pytest.raises(RuntimeError, match="could not locate"):
        with recorder._patched_eager({}):
            pass
