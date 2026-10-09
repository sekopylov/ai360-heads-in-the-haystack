"""`scripts/case_study.py` — the one script with no coverage.

Two defects are pinned here:
* `--argmax-domain` reached `find_copy_step` (the figure) but not
  `decode_with_attention` (the capture the credits come from), so with `full` the
  artifact's `top_heads` described the prompt domain while the JSON claimed `full`;
* the script ignored every other recorded condition (`chat_template`,
  `system_prompt`, `enable_thinking`, `capture_method`), so the figure could show a
  different prompt mode than the run whose heads it explains.  `--scores` now reuses
  the run's own config, as `resolve_detection_settings` does for the ablations.
"""

from __future__ import annotations

import importlib.util
import json
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


def _stub(case_study, monkeypatch, info, seen: dict):
    """Replace everything that needs a real model or a plot backend."""
    import torch

    def fake_decode(model, info, input_ids, **kwargs):
        seen["capture"] = kwargs
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

    class FakeTokenizer:
        def decode(self, ids, **kwargs):
            return "answer"

    monkeypatch.setattr(case_study, "decode_with_attention", fake_decode)
    monkeypatch.setattr(case_study, "build_needle_sample",
                        lambda *a, **k: seen.update(sample=k) or FakeSample())
    monkeypatch.setattr(case_study, "credits_from_trace",
                        lambda *a, **k: ({h: set() for h in info.scoreable_heads}, {},
                                         {h: 1 for h in info.scoreable_heads}))
    monkeypatch.setattr(case_study, "find_copy_step", lambda *a, **k: (FakeStep(), 2, 1))
    monkeypatch.setattr(case_study, "save_json",
                        lambda payload, path: seen.update(json=payload))
    monkeypatch.setattr(case_study, "plot_attention_distribution", lambda d: None)
    monkeypatch.setattr(case_study, "save_fig", lambda fig, path: None)
    monkeypatch.setattr(case_study, "add_provenance", lambda payload, **k: payload)
    monkeypatch.setattr(case_study, "_load", lambda name, **k: (None, FakeTokenizer(), info))


def _run(case_study, monkeypatch, tmp_path, argv, info):
    seen: dict = {}
    _stub(case_study, monkeypatch, info, seen)
    monkeypatch.setattr(sys, "argv", ["case_study", "--model", "toy", "--length", "8",
                                      "--out", str(tmp_path), *argv])
    assert case_study.main() == 0
    return seen


def test_argmax_domain_reaches_the_capture(case_study, monkeypatch, tmp_path):
    """The domain must be given to the capture, not only to the display path."""
    from tests.test_regressions import attention_info

    seen = _run(case_study, monkeypatch, tmp_path, ["--argmax-domain", "full"],
                attention_info(2, 2))
    assert seen["capture"]["argmax_domain"] == "full", "the capture ran in another domain"
    assert seen["json"]["argmax_domain"] == "full"


def test_scores_directory_supplies_the_recorded_conditions(case_study, monkeypatch, tmp_path):
    """`--scores` must make the figure use the illustrated run's own conditions."""
    from tests.test_regressions import attention_info

    run = tmp_path / "run"
    run.mkdir()
    (run / "scores_next_step.json").write_text(json.dumps({"meta": {"config": {
        "chat_template": False,
        "system_prompt": "SYS",
        "enable_thinking": True,
        "argmax_domain": "full",
        "capture_method": "output_attentions",
    }}}), encoding="utf-8")

    seen = _run(case_study, monkeypatch, tmp_path, ["--scores", str(run)],
                attention_info(2, 2))

    assert seen["sample"]["chat_template"] is False
    assert seen["sample"]["system_prompt"] == "SYS"
    assert seen["sample"]["enable_thinking"] is True
    assert seen["capture"]["argmax_domain"] == "full"
    assert seen["capture"]["capture_method"] == "output_attentions"
    payload = seen["json"]
    assert payload["capture_method"] == "output_attentions"
    assert payload["chat_template"] is False and payload["enable_thinking"] is True
