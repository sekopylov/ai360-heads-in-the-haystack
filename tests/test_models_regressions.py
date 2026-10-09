"""Regression tests: models (split out of test_regressions.py).

Each test pins a specific failure mode the old code had, so the fix cannot
quietly regress.  They are all fast (no checkpoints).
"""

from __future__ import annotations

import pytest
import torch
from retrieval_heads.models import ModelInfo, require_scoreable

from tests._helpers import (  # noqa: F401
    _Args, _well_formed_curve, attention_info, scores_with,
)


def test_model_info_round_trip_and_fallback():
    from retrieval_heads.models import ModelInfo

    info = attention_info(2, 2)
    assert ModelInfo.from_dict(info.as_dict()).scoreable_layers == [0, 1]

    # A metadata dict without `scoreable_layers` derives them from num_heads.
    legacy = info.as_dict()
    del legacy["scoreable_layers"]
    assert ModelInfo.from_dict(legacy).scoreable_layers == [0, 1]

    # ... and one with neither is rejected instead of yielding an empty model.
    del legacy["num_heads"]
    with pytest.raises(ValueError, match="scoreable_layers"):
        ModelInfo.from_dict(legacy)


def test_attn_implementation_restores_each_config_individually():
    """One `previous` value for all configs pinned every config to the first one."""
    from torch import nn

    from retrieval_heads.attention import (
        restore_attn_implementation,
        set_attn_implementation,
    )

    class Cfg:
        def __init__(self, value):
            self._attn_implementation = value

    class Block(nn.Module):
        def __init__(self, value):
            super().__init__()
            self.config = Cfg(value)

    model = nn.Module()
    model.config = Cfg("root")
    model.add_module("a", Block("eager"))
    model.add_module("b", Block("sdpa"))
    # None is a valid "not set" value and must be restored too, not skipped.
    model.add_module("c", Block(None))

    previous = set_attn_implementation(model, "flash")
    assert model.config._attn_implementation == "flash"
    assert model.a.config._attn_implementation == "flash"
    assert model.b.config._attn_implementation == "flash"
    assert model.c.config._attn_implementation == "flash"

    restore_attn_implementation(model, previous)
    assert model.config._attn_implementation == "root"
    assert model.a.config._attn_implementation == "eager"
    assert model.b.config._attn_implementation == "sdpa"
    assert model.c.config._attn_implementation is None, "a previously-None config was not restored"


def test_text_config_keeps_the_outer_config_when_nested_is_none():
    from types import SimpleNamespace

    from retrieval_heads.models import text_config

    outer = SimpleNamespace(text_config=None, num_hidden_layers=4)
    assert text_config(outer) is outer


def test_verify_weights_rejects_a_partial_tree(tmp_path):
    import json

    from tests.test_cli_argv import load_job_driver

    registry = tmp_path / "models.json"
    registry.write_text(json.dumps({"models": {
        "m": {"path": "models/M", "files": {"config.json": "x"}},
        "other": {"path": "models/O", "files": {"config.json": "y"}},
    }}), encoding="utf-8")
    root = tmp_path / "models"
    (root / "M").mkdir(parents=True)
    driver = load_job_driver()

    with pytest.raises(SystemExit, match="incomplete"):
        driver.verify_weights(root, registry)

    (root / "M" / "config.json").write_text("{}", encoding="utf-8")
    # With --models only the selected keys are required, so a job for one model
    # does not fail because another registry entry is not on disk.
    driver.verify_weights(root, registry, keys=["m"])
    with pytest.raises(SystemExit, match="incomplete"):
        driver.verify_weights(root, registry)                 # `other` still missing

    (root / "O").mkdir()
    (root / "O" / "config.json").write_text("{}", encoding="utf-8")
    driver.verify_weights(root, registry)                     # complete now

    with pytest.raises(SystemExit, match="not in"):
        driver.verify_weights(root, registry, keys=["missing-key"])


def test_model_info_from_dict_requires_heads_for_every_scoreable_layer():
    from retrieval_heads.models import ModelInfo

    info = attention_info(2, 2)
    broken = info.as_dict()
    del broken["num_heads"]["1"]          # layer 1 is scoreable but has no head count
    with pytest.raises(ValueError, match="num_heads"):
        ModelInfo.from_dict(broken)


def test_discovery_requires_every_layer_to_be_classified():
    """An unknown mixer class must fail loudly, not make the model look dense."""
    from types import SimpleNamespace

    from torch import nn

    from retrieval_heads.models import build_model_info

    class UnknownMixer(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_idx = 1

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([UnknownMixer()])

    config = SimpleNamespace(num_hidden_layers=2, num_attention_heads=2,
                             num_key_value_heads=2, head_dim=4, hidden_size=8,
                             max_position_embeddings=64)
    with pytest.raises(RuntimeError, match="neither a scoreable attention"):
        build_model_info(Model(), config, path="toy")


def test_head_dim_disagreement_is_fatal():
    from types import SimpleNamespace

    from torch import nn

    from retrieval_heads.models import build_model_info

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_idx = 0
            self.head_dim = 4
            self.config = SimpleNamespace(num_attention_heads=2, num_key_value_heads=1,
                                          hidden_size=8, max_position_embeddings=64)
            for name in ("q_proj", "k_proj", "v_proj"):
                setattr(self, name, nn.Linear(8, 8, bias=False))
            self.o_proj = nn.Linear(16, 8, bias=False)      # 16 != 2 * 4

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Attention()])

    config = SimpleNamespace(num_hidden_layers=1, num_attention_heads=2,
                             num_key_value_heads=1, head_dim=4, hidden_size=8,
                             max_position_embeddings=64)
    with pytest.raises(RuntimeError, match="head geometry is inconsistent"):
        build_model_info(Model(), config, path="toy")


def test_inhomogeneous_records_none_not_zero(caplog):
    """Layers that disagree on a width record `null`, not a plausible-looking 0.

    `head_dim = 0` then flowed into geometry comparisons and plots as if it had been
    measured (and two broken models compared equal).
    """
    import logging

    from retrieval_heads.models import _inhomogeneous

    with caplog.at_level(logging.WARNING):
        assert _inhomogeneous("head_dim", {4, 8}) is None
    assert "recording null" in caplog.text

    # No disagreement at all (no scoreable module) is not a warning: require_scoreable
    # reports that case, and warning here only added a misleading first line.
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert _inhomogeneous("head_dim", set()) is None
    assert caplog.text == ""


def test_model_info_round_trips_an_unknown_width():
    from retrieval_heads.models import ModelInfo

    payload = attention_info(2, 2).as_dict()
    payload["head_dim"] = None
    payload["hidden_size"] = None
    restored = ModelInfo.from_dict(payload)
    assert restored.head_dim is None and restored.hidden_size is None
    assert ModelInfo.from_dict(restored.as_dict()).head_dim is None


def test_require_matching_scores_rejects_an_unknown_width():
    """`None == None` is False, so the guard has to test for it explicitly.

    Otherwise two inhomogeneous models would pass the geometry check and a head
    masker built on an unknown head_dim would cut the wrong slice.
    """
    from retrieval_heads.cli import require_matching_scores
    from retrieval_heads.scoring import RetrievalScores

    broken = attention_info(1, 2)
    broken.head_dim = None
    scores = RetrievalScores(info=broken, score=torch.zeros(1, 2),
                             activation_freq=torch.zeros(1, 2), n_instances=1)
    with pytest.raises(SystemExit, match="head_dim is unknown"):
        require_matching_scores(scores, attention_info(1, 2))

