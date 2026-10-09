"""Regression tests: plotting (split out of test_regressions.py).

Each test pins a specific failure mode the old code had, so the fix cannot
quietly regress.  They are all fast (no checkpoints).
"""

from __future__ import annotations

import pytest
import torch
from retrieval_heads.scoring import RetrievalScores

from tests._helpers import (  # noqa: F401
    _Args, _well_formed_curve, attention_info, scores_with,
)


def test_attention_distribution_fills_every_panel():
    """Fig. 1 draws one panel per distribution, not just the first row of axes."""
    import matplotlib.pyplot as plt
    import numpy as np

    from retrieval_heads.plotting import plot_attention_distribution

    distributions = {
        "strong head": (np.linspace(0, 1, 16), (2, 6)),
        "weak head": (np.linspace(1, 0, 16) * 0.1, (4, 8)),
    }
    fig = plot_attention_distribution(distributions)
    try:
        assert len(fig.axes) == len(distributions)
        for ax in fig.axes:
            assert ax.collections, "an axis was created but left empty"
            assert ax.get_title(loc="left"), "every panel should carry its label"
            assert ax.get_legend() is not None
    finally:
        plt.close(fig)


def test_plot_masking_curve_exact_match_requires_the_series():
    import matplotlib.pyplot as plt

    from retrieval_heads.plotting import plot_masking_curve

    old_artifact = {"m": {"k_values": [1], "k_effective": [1], "retrieval": [50.0],
                          "random_mean": [60.0], "random_std": [1.0], "baseline": 70.0}}
    with pytest.raises(ValueError, match="exact-match"):
        plot_masking_curve(old_artifact, metric="exact_match")
    plt.close("all")


def test_plot_mixer_ablation_requires_matching_k():
    import matplotlib.pyplot as plt

    from retrieval_heads.plotting import plot_mixer_ablation

    a = {"k_values": [1, 2], "full_attention": [90.0, 80.0], "linear_attention": [10.0, 5.0],
         "baseline": 95.0}
    b = {"k_values": [1, 4], "full_attention": [90.0, 70.0], "linear_attention": [10.0, 5.0],
         "baseline": 95.0}
    with pytest.raises(ValueError, match="different K"):
        plot_mixer_ablation({"a": a, "b": b})
    plt.close("all")
    fig = plot_mixer_ablation({"a": a})       # a single K set is fine
    plt.close(fig)


def test_layouts_match_and_grid_mode_warns(caplog):
    from retrieval_heads.properties import correlate, layouts_match

    a = scores_with([0.9, 0.8, 0.05])
    b = scores_with([0.9, 0.8, 0.05])
    assert layouts_match(a, b) is True

    wider = attention_info(1, 4)
    c = RetrievalScores(info=wider, score=torch.tensor([[0.9, 0.8, 0.05, 0.0]]),
                        activation_freq=torch.zeros(1, 4), n_instances=1)
    assert layouts_match(a, c) is False

    with caplog.at_level("WARNING"):
        value = correlate(a, c, mode="grid")
    assert "layouts" in caplog.text
    assert isinstance(value, float)
    assert layouts_match(a, c) is False


def test_plot_score_pie_uses_the_runs_own_threshold():
    import matplotlib.pyplot as plt

    from retrieval_heads.plotting import plot_score_pie

    info = attention_info(1, 3)
    scores = RetrievalScores(info=info, score=torch.tensor([[0.4, 0.2, 0.0]]),
                             activation_freq=torch.zeros(1, 3), n_instances=1,
                             threshold=0.35)
    fig = plot_score_pie({"m": scores})
    try:
        labels = [text.get_text() for text in fig.axes[0].texts]
        assert any(">0.35" in label for label in labels), labels
    finally:
        plt.close(fig)


def test_head_overlap_is_incomparable_across_layouts():
    from retrieval_heads.properties import head_overlap

    a = scores_with([0.9, 0.8, 0.05])
    wider = attention_info(1, 4)
    b = RetrievalScores(info=wider, score=torch.tensor([[0.9, 0.8, 0.05, 0.0]]),
                        activation_freq=torch.zeros(1, 4), n_instances=1)

    result = head_overlap(a, b, threshold=0.1, mode="sorted")
    assert result.comparable is False
    assert result.jaccard != result.jaccard          # NaN
    assert result.shared == [] and result.only_a == []
    assert result.as_dict()["comparable"] is False


def test_all_nan_scores_still_plot():
    """category_fractions must keep its bucket keys so the ring chart cannot KeyError."""
    import matplotlib.pyplot as plt

    from retrieval_heads.plotting import plot_score_pie
    from retrieval_heads.properties import category_fractions

    info = attention_info(1, 2)
    scores = RetrievalScores(info=info, score=torch.tensor([[float("nan")] * 2]),
                             activation_freq=torch.zeros(1, 2), n_instances=1)

    fractions = category_fractions(scores)
    assert fractions["n_heads"] == 0
    assert fractions["retrieval"]["frac"] == 0.0

    fig = plot_score_pie({"m": scores})
    plt.close(fig)


def test_plot_corr_map_accepts_a_json_artifact_with_nulls():
    """`save_json` writes non-finite values as null; the plot must not crash on it."""
    import matplotlib.pyplot as plt

    from retrieval_heads.plotting import plot_corr_map

    corr = {"labels": ["a", "b"], "values": [[1.0, None], [None, 1.0]], "mode": "sorted"}
    fig = plot_corr_map(corr)
    plt.close(fig)


def test_score_pie_title_does_not_say_mixed():
    import matplotlib.pyplot as plt

    from retrieval_heads.plotting import plot_score_pie

    info = attention_info(1, 2)
    scores = RetrievalScores(info=info, score=torch.tensor([[0.9, 0.05]]),
                             activation_freq=torch.zeros(1, 2), n_instances=1)
    fig = plot_score_pie({"m": scores})
    try:
        titles = " ".join(ax.get_title() for ax in fig.axes)
        assert "mixed" not in titles
    finally:
        plt.close(fig)


def test_grid_correlation_across_layouts_is_nan():
    from retrieval_heads.properties import correlate

    a = RetrievalScores(info=attention_info(2, 2), score=torch.rand(2, 2),
                        activation_freq=torch.zeros(2, 2), n_instances=1)
    b = RetrievalScores(info=attention_info(3, 1), score=torch.rand(3, 1),
                        activation_freq=torch.zeros(3, 1), n_instances=1)
    assert correlate(a, b, mode="grid") != correlate(a, b, mode="grid")   # NaN


def test_correlation_figure_carries_the_sorted_caveat():
    import matplotlib.pyplot as plt

    from retrieval_heads.plotting import plot_corr_map
    from retrieval_heads.properties import CorrelationMatrix

    matrix = CorrelationMatrix(labels=["a", "b"], values=[[1.0, 0.9], [0.9, 1.0]],
                               mode="sorted",
                               caveat="sorted mode compares ranks, not heads")
    fig = plot_corr_map(matrix)
    try:
        texts = [text.get_text() for text in fig.texts]
        assert any("ranks" in text for text in texts), texts
    finally:
        plt.close(fig)


def test_head_overlap_artifact_carries_its_mode():
    """The old version of this test only read the *default*, so it passed while
    `head_overlap` never passed `mode` to its own constructor: `overlap.json` said
    `mode: grid` next to a correlation computed in `sorted` (and next to a caveat
    about sorted), i.e. the artifact contradicted itself.
    """
    from retrieval_heads.properties import head_overlap

    a = RetrievalScores(info=attention_info(1, 2), score=torch.tensor([[0.9, 0.0]]),
                        activation_freq=torch.zeros(1, 2), n_instances=1)
    b = RetrievalScores(info=attention_info(1, 2), score=torch.tensor([[0.8, 0.0]]),
                        activation_freq=torch.zeros(1, 2), n_instances=1)

    grid = head_overlap(a, b, mode="grid").as_dict()
    assert grid["mode"] == "grid" and grid["caveat"] is None

    # Different layouts force the cross-family path, which is where `sorted` lives.
    c = RetrievalScores(info=attention_info(2, 2), score=torch.tensor([[0.9, 0.0], [0.1, 0.0]]),
                        activation_freq=torch.zeros(2, 2), n_instances=1)
    sorted_payload = head_overlap(a, c, mode="sorted").as_dict()
    assert sorted_payload["mode"] == "sorted", sorted_payload
    assert sorted_payload["comparable"] is False
    assert sorted_payload["caveat"], "a sorted comparison must carry its caveat"


def test_plot_masking_curve_default_metric_is_the_f1_series():
    """The default path must draw the F1 series.

    A guard inserted before its assignments made it raise UnboundLocalError, and no
    test used the default metric -- the figures stage died after 5 of 9 PDFs.
    """
    import matplotlib.pyplot as plt

    from retrieval_heads.plotting import plot_masking_curve

    fig = plot_masking_curve(_well_formed_curve())
    try:
        errbars = [c for c in fig.axes[0].containers
                   if type(c).__name__ == "ErrorbarContainer"]
        assert len(errbars) == 2, errbars
        assert any("no masking" in (line.get_label() or "") for line in fig.axes[0].lines)
    finally:
        plt.close(fig)


def test_plot_masking_curve_can_plot_recall():
    import matplotlib.pyplot as plt

    from retrieval_heads.plotting import plot_masking_curve

    fig = plot_masking_curve(_well_formed_curve(), metric="recall")
    plt.close(fig)


def test_plot_masking_curve_rejects_a_ragged_artifact():
    import matplotlib.pyplot as plt

    from retrieval_heads.plotting import plot_masking_curve

    curve = _well_formed_curve()
    curve["m"]["retrieval"] = [90.0]          # shorter than k_values
    with pytest.raises(ValueError, match="inconsistent lengths"):
        plot_masking_curve(curve)
    plt.close("all")


def test_cmd_figures_writes_every_figure(tmp_path):
    """End-to-end figures stage on a synthetic run tree.

    `test_every_stage_and_profile_builds_parseable_argv` only parses argv, so a
    figure that cannot be drawn at all went unnoticed.
    """
    import json

    from retrieval_heads.cli import main

    run = tmp_path / "m"
    run.mkdir()
    info = attention_info(1, 2)
    scores = RetrievalScores(info=info, score=torch.tensor([[0.9, 0.1]]),
                             activation_freq=torch.zeros(1, 2), n_instances=1)
    scores.save(run / "scores_next_step")
    # The artifact on disk is a single curve, not the {label: curve} mapping the
    # plotting function takes in memory.
    (run / "masking_curve.json").write_text(
        json.dumps(_well_formed_curve()["m"]), encoding="utf-8")
    (run / "task_qa.json").write_text(json.dumps({
        "baseline_f1": 50.0, "n_samples": 2, "n_scoreable_heads": 2,
        "by_k": {"1": {"k_effective": 1, "retrieval_f1": 10.0, "drop_retrieval": 40.0,
                       "random_f1_mean": 45.0, "drop_random": 5.0}}}), encoding="utf-8")
    (run / "task_cot.json").write_text(json.dumps({
        "k": 1, "n_samples": 2, "results": {"cot": {
            "baseline": 50.0, "retrieval_masked": 10.0, "random_masked_mean": 45.0,
            "random_masked_std": 5.0}}}), encoding="utf-8")
    (run / "mixer_ablation.json").write_text(json.dumps({
        "baseline": 50.0, "k_values": [1], "full_attention": [10.0],
        "linear_attention": [40.0], "n_full_layers": 1, "n_linear_layers": 1}),
        encoding="utf-8")

    out = tmp_path / "figs"
    assert main(["figures", "--runs", str(run), "--out", str(out)]) == 0
    produced = {path.name for path in out.glob("*.pdf")}
    for name in ("ring_graph.pdf", "masking_heads.pdf", "task_qa.pdf", "task_cot.pdf",
                 "mixer_ablation.pdf", "corr_map.pdf"):
        assert name in produced, (name, sorted(produced))


def test_plot_task_cot_gets_a_figure_title():
    import matplotlib.pyplot as plt

    from retrieval_heads.plotting import plot_task_cot

    result = {"results": {
        "answer_only": {"baseline": 75.0, "retrieval_masked": 50.0,
                        "random_masked_mean": 12.0, "random_masked_std": 0.0},
        "cot": {"baseline": 100.0, "retrieval_masked": 75.0,
                "random_masked_mean": 81.0, "random_masked_std": 0.0},
    }}
    fig = plot_task_cot(result)
    try:
        assert fig._suptitle is not None, "plot_task_cot is the only figure without finish()"
    finally:
        plt.close(fig)


def test_plot_masking_curve_exact_match_uses_the_stored_std():
    import matplotlib.pyplot as plt

    from retrieval_heads.plotting import plot_masking_curve

    curve = {"m": {"k_values": [1], "k_effective": [1], "retrieval": [50.0],
                   "retrieval_std": [5.0], "retrieval_exact_match": [40.0],
                   "retrieval_exact_std": [7.0], "random_mean": [60.0],
                   "random_std": [3.0], "random_exact_match_mean": [55.0],
                   "baseline": 70.0, "baseline_exact_match": 65.0}}
    fig = plot_masking_curve(curve, metric="exact_match")
    try:
        errbars = [c for c in fig.axes[0].containers
                   if type(c).__name__ == "ErrorbarContainer"]
        assert len(errbars) == 2, errbars
        assert len(errbars[0].lines[1]) > 0, "retrieval error bars missing"
    finally:
        plt.close(fig)


def test_hybrid_pie_title_states_both_head_bases():
    """The panel must not show only the scoreable head count for a hybrid.

    The README quotes "9 of 336 heads" (2.7%) and "68.8% of 48 scoreable heads" in one
    paragraph; a caption that says only "48 scoreable heads / 18 linear layers" invites
    reading the second share against the wrong denominator.
    """
    import matplotlib.pyplot as plt

    from retrieval_heads.models import ModelInfo
    from retrieval_heads.plotting import plot_score_pie

    info = ModelInfo(
        name="hybrid", path="h", model_type="toy", num_layers=2,
        layer_types=["full_attention", "linear_attention"],
        num_heads={0: 2}, num_kv_heads={0: 2}, head_dim=4, hidden_size=8,
        max_position_embeddings=128, scoreable_layers_=[0],
        linear_layers_=[1],
    )
    scores = RetrievalScores(info=info, score=torch.tensor([[0.9, 0.05], [float("nan")] * 2]),
                             activation_freq=torch.zeros(2, 2), n_instances=1)
    fig = plot_score_pie({"m": scores})
    try:
        title = fig.axes[0].get_title()
        assert "2 scoreable heads" in title, title
        assert "1 linear layers" in title, title
        assert str(info.n_all_heads) in title, title      # 2 + 1*4 = 6 heads in all
    finally:
        plt.close(fig)
