"""Regression tests: io (split out of test_regressions.py).

Each test pins a specific failure mode the old code had, so the fix cannot
quietly regress.  They are all fast (no checkpoints).
"""

from __future__ import annotations


from tests._helpers import (  # noqa: F401
    _Args, _well_formed_curve, attention_info, scores_with,
)


def test_chat_helper_respects_no_chat_template():
    from retrieval_heads.downstream import _chat

    class Tok:
        def apply_chat_template(self, messages, **kwargs):
            return "TEMPLATED"

    assert _chat(Tok(), "hello", enable_thinking=False) == "TEMPLATED"
    assert _chat(Tok(), "hello", enable_thinking=False, chat_template=False) == "hello"


def test_render_chat_falls_back_on_a_non_typeerror():
    from retrieval_heads.haystack import render_chat

    class PickyTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            if "enable_thinking" in kwargs:
                raise ValueError("unexpected keyword argument 'enable_thinking'")
            return "ok"

    assert render_chat(PickyTokenizer(), [{"role": "user", "content": "x"}],
                       enable_thinking=False) == "ok"

