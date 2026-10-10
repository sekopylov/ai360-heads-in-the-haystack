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


def _stub(case_study, monkeypatch, info, seen: dict, row_dtype=None):
    """Replace everything that needs a real model or a plot backend."""
    import torch

    if row_dtype is None:
        row_dtype = torch.float32

    def fake_decode(model, info, input_ids, **kwargs):
        seen["capture"] = kwargs
        return type("T", (), {"steps": []})(), [1, 2, 3]

    class FakeStep:
        # One row per scoreable layer.  `row_dtype` is what the capture returns in a
        # real run: the query dtype, i.e. bfloat16 on a GPU job.
        attn = {layer: torch.zeros(2, 4, dtype=row_dtype) for layer in info.scoreable_layers}
        step = 0

    class FakeSample:
        input_ids = [[1, 2, 3]]
        needle_span = (1, 2)
        length = 3
        needle_text_ids = [2]
        n_unique_needle_text_tokens = 1
        target_tokens = 3
        depth = 0.5
        haystack_span = (0, 3)

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
    monkeypatch.setattr(case_study, "plot_attention_distribution",
                        lambda d: seen.update(distributions=d))
    monkeypatch.setattr(case_study, "save_fig", lambda fig, path: None)
    monkeypatch.setattr(case_study, "add_provenance",
                        lambda payload, **k: (seen.update(provenance_kwargs=k), payload)[1])
    monkeypatch.setattr(case_study, "_load", lambda name, **k: (None, FakeTokenizer(), info))


def _run(case_study, monkeypatch, tmp_path, argv, info, row_dtype=None):
    seen: dict = {}
    _stub(case_study, monkeypatch, info, seen, row_dtype=row_dtype)
    monkeypatch.setattr(sys, "argv", ["case_study", "--model", "toy", "--length", "8",
                                      "--out", str(tmp_path), *argv])
    assert case_study.main() == 0
    return seen


def test_argmax_domain_reaches_the_capture(case_study, monkeypatch, tmp_path):
    """The domain must be given to the capture, not only to the display path."""
    from tests._helpers import attention_info

    seen = _run(case_study, monkeypatch, tmp_path, ["--argmax-domain", "full"],
                attention_info(2, 2))
    assert seen["capture"]["argmax_domain"] == "full", "the capture ran in another domain"
    assert seen["json"]["argmax_domain"] == "full"
    # The prompt/full domains search no sub-span, so the capture must not be handed
    # one (a stray span would silently turn `full` into `haystack`).
    assert seen["capture"]["argmax_span"] is None


def test_haystack_domain_passes_the_span_to_the_capture(case_study, monkeypatch, tmp_path):
    """`haystack` needs the span at the capture, not just the domain name."""
    from tests._helpers import attention_info

    seen = _run(case_study, monkeypatch, tmp_path, ["--argmax-domain", "haystack"],
                attention_info(2, 2))
    assert seen["capture"]["argmax_domain"] == "haystack"
    assert seen["capture"]["argmax_span"] == (0, 3), "the capture searched another span"
    assert seen["json"]["haystack_span"] == [0, 3]


def test_scores_directory_supplies_the_recorded_conditions(case_study, monkeypatch, tmp_path):
    """`--scores` must make the figure use the illustrated run's own conditions."""
    from tests._helpers import attention_info

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


def test_the_case_study_records_its_dtype_and_filler_seed(case_study, monkeypatch, tmp_path):
    """Without these the figure cannot be rebuilt from its own artifact.

    `--dtype` defaulted to `None`, and `add_provenance(dtype=None)` omits the key
    entirely, so a run without the flag recorded no precision at all; the filler was
    hard-coded to `HaystackBuilder(seed=0)` and the seed was not written down.
    """
    from tests._helpers import attention_info

    info = attention_info(2, 2)
    info.dtype = "bfloat16"        # what `_load` records for a real checkpoint
    seen = _run(case_study, monkeypatch, tmp_path, ["--seed", "7"], info)
    payload = seen["json"]
    assert payload["filler_seed"] == 7
    assert seen["sample"]["builder"].seed == 7, "the seed did not reach the builder"
    assert seen["provenance_kwargs"]["dtype"] == "bfloat16", (
        "a run without --dtype must fall back to the checkpoint's own precision"
    )
    # The other conditions that used to be missing from the artifact.
    assert payload["length_requested"] == 8 and payload["max_new_tokens"] == 32


def test_bfloat16_attention_rows_reach_the_figure_as_float32(case_study, monkeypatch, tmp_path):
    """The first A100 `mask` job died here, after 75 minutes of masking.

    `StepTrace.attn` holds the softmax in the *query* dtype, so on a bf16 GPU run the
    rows are bfloat16 -- and numpy has no bfloat16, so `.cpu().numpy()` raises
    `TypeError: Got unsupported ScalarType BFloat16`.  Job `bt1u3ja8cb0it4klqehl`
    (`case_study.py:203`) died exactly three seconds into this stage, having already
    paid for the hybrid's whole masking curve.  The stage had only ever run on CPU
    fp32, so no test saw it: this one hands the figure bf16 rows, as a real GPU run
    does, and requires float32 arrays at the plot.
    """
    import numpy as np
    import torch

    from tests._helpers import attention_info

    seen = _run(case_study, monkeypatch, tmp_path, [], attention_info(2, 2),
                row_dtype=torch.bfloat16)
    rows = [row for row, _span in seen["distributions"].values()]
    assert len(rows) == 2, "the figure must get the strong and the weak head's row"
    for row in rows:
        assert isinstance(row, np.ndarray)
        assert row.dtype == np.float32, f"matplotlib got {row.dtype}, not float32"
        assert np.isfinite(row).all()
