"""`scripts/case_study.py` — the one script with no coverage.

The defect this pins: `--argmax-domain` reached `find_copy_step` (the figure) but not
`decode_with_attention` (the capture the credits are computed from), so with
`--argmax-domain full` the artifact's `top_heads` described the prompt domain while
the JSON claimed `full`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def case_study():
    spec = importlib.util.spec_from_file_location(
        "case_study", REPO_ROOT / "scripts" / "case_study.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["case_study"] = module
    spec.loader.exec_module(module)
    return module


def test_argmax_domain_reaches_the_capture(case_study, monkeypatch, tmp_path):
    """The domain must be given to the capture, not only to the display path."""
    import torch

    from tests.test_regressions import attention_info

    info = attention_info(2, 2)
    seen: dict[str, object] = {}

    def fake_decode(model, info, input_ids, **kwargs):
        seen.update(kwargs)
        return type("T", (), {"steps": []})(), [1, 2, 3]

    class FakeStep:
        # One row per scoreable layer; `main` reads `.cpu().numpy()` and `.max()`.
        attn = {layer: torch.zeros(2, 4) for layer in info.scoreable_layers}
        step = 0

    class FakeSample:
        input_ids = [[1, 2, 3]]
        needle_span = (1, 2)
        length = 3
        needle_text_ids = [2]
        n_unique_needle_text_tokens = 1
        target_tokens = 3
        depth = 0.5

    monkeypatch.setattr(case_study, "decode_with_attention", fake_decode)
    monkeypatch.setattr(case_study, "build_needle_sample", lambda *a, **k: FakeSample())
    monkeypatch.setattr(case_study, "credits_from_trace",
                        lambda *a, **k: ({h: set() for h in info.scoreable_heads}, {},
                                {h: 1 for h in info.scoreable_heads}))
    monkeypatch.setattr(case_study, "find_copy_step",
                        lambda *a, **k: (FakeStep(), 2, 1))
    monkeypatch.setattr(case_study, "save_json", lambda payload, path: seen.update(json=payload))
    monkeypatch.setattr(case_study, "plot_attention_distribution", lambda d: None)
    monkeypatch.setattr(case_study, "save_fig", lambda fig, path: None)
    monkeypatch.setattr(case_study, "add_provenance", lambda payload, **k: payload)
    class FakeTokenizer:
        def decode(self, ids, **kwargs):
            return "answer"

    monkeypatch.setattr(case_study, "_load", lambda name: (None, FakeTokenizer(), info))
    monkeypatch.setattr(sys, "argv", ["case_study", "--model", "toy", "--length", "8",
                                      "--argmax-domain", "full",
                                      "--out", str(tmp_path)])

    assert case_study.main() == 0
    assert seen["argmax_domain"] == "full", "the capture ran in a different domain"
    assert seen["json"]["argmax_domain"] == "full"
