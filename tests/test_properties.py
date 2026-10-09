"""Statistics and metric helpers (no checkpoints involved)."""

from __future__ import annotations

import numpy as np
import pytest

from retrieval_heads.downstream import accuracy, final_answer, word_f1
from retrieval_heads.masking import normalized_contains, token_f1
from retrieval_heads.models import ModelInfo
from retrieval_heads.properties import (
    category_fractions,
    correlate,
    correlation_matrix,
    head_overlap,
    layer_profile,
)
from retrieval_heads.scoring import RetrievalScores


def make_info(num_layers=4, heads=2, scoreable=(0, 2)) -> ModelInfo:
    return ModelInfo(
        name="toy", path="toy", model_type="toy", num_layers=num_layers,
        layer_types=["full_attention" if i in scoreable else "linear_attention"
                     for i in range(num_layers)],
        num_heads={i: heads for i in scoreable},
        num_kv_heads={i: heads for i in scoreable},
        head_dim=4, hidden_size=8, max_position_embeddings=128,
        scoreable_layers_=list(scoreable),
        linear_layers_=[i for i in range(num_layers) if i not in scoreable],
    )


def make_scores(values: dict[tuple[int, int], float], info: ModelInfo | None = None,
                activation: dict[tuple[int, int], float] | None = None,
                name: str = "toy") -> RetrievalScores:
    info = info or make_info()
    score = info.empty_matrix()
    act = info.empty_matrix()
    for (layer, head), value in values.items():
        score[layer, head] = value
        act[layer, head] = (activation or {}).get((layer, head), 1.0)
    return RetrievalScores(info=info, score=score, activation_freq=act, n_instances=1,
                           meta={"name": name})


def test_category_fractions_splits_heads_into_three_buckets():
    info = make_info(num_layers=4, heads=4, scoreable=(0, 2))   # 8 scoreable heads
    values = {(0, 0): 0.0, (0, 1): 0.0, (0, 2): 0.05, (0, 3): 0.05,
              (2, 0): 0.5, (2, 1): 0.9, (2, 2): 0.0, (2, 3): 0.02}
    frac = category_fractions(make_scores(values, info), threshold=0.1)
    assert frac["n_heads"] == 8
    assert frac["zero"]["n"] == 3
    assert frac["low"]["n"] == 3
    assert frac["retrieval"]["n"] == 2
    assert frac["retrieval"]["frac"] == pytest.approx(0.25)
    assert frac["max_score"] == pytest.approx(0.9)


def test_non_scoreable_entries_stay_nan():
    info = make_info(num_layers=4, heads=2, scoreable=(0, 2))
    scores = make_scores({(0, 0): 0.4}, info)
    assert np.isnan(scores.score[1, 0])
    assert np.isnan(scores.score[3, 1])
    assert not np.isnan(scores.score[0, 0])


def test_ranked_heads_ignores_non_scoreable_layers():
    info = make_info(num_layers=4, heads=2, scoreable=(0, 2))
    scores = make_scores({(0, 0): 0.1, (0, 1): 0.9, (2, 0): 0.5, (2, 1): 0.2}, info)
    ranked = [str(h) for h in scores.ranked_heads()]
    assert ranked == ["L0H1", "L2H0", "L2H1", "L0H0"]
    assert len(ranked) == 4


def test_heads_above_threshold():
    info = make_info(num_layers=2, heads=3, scoreable=(0, 1))
    scores = make_scores({(0, 0): 0.05, (0, 1): 0.11, (0, 2): 0.5,
                          (1, 0): 0.0, (1, 1): 0.2, (1, 2): 0.099}, info)
    assert {str(h) for h in scores.heads_above(0.1)} == {"L0H1", "L0H2", "L1H1"}


def test_correlation_of_a_matrix_with_itself_is_one():
    info = make_info()
    values = {(0, 0): 0.1, (0, 1): 0.4, (2, 0): 0.7, (2, 1): 0.2}
    scores = make_scores(values, info)
    assert correlate(scores, scores) == pytest.approx(1.0)


def test_correlation_detects_an_inverted_pattern():
    info = make_info()
    a = make_scores({(0, 0): 0.1, (0, 1): 0.9, (2, 0): 0.1, (2, 1): 0.9}, info)
    b = make_scores({(0, 0): 0.9, (0, 1): 0.1, (2, 0): 0.9, (2, 1): 0.1}, info)
    assert correlate(a, b) == pytest.approx(-1.0)


def test_correlation_across_different_shapes_uses_sorted_scores():
    a = make_scores({(0, 0): 0.1, (0, 1): 0.9, (2, 0): 0.3, (2, 1): 0.6},
                    make_info(num_layers=4, heads=2, scoreable=(0, 2)))
    b = make_scores({(0, 0): 0.2, (0, 1): 0.9, (0, 2): 0.3, (1, 0): 0.6},
                    make_info(num_layers=2, heads=3, scoreable=(0, 1)))
    value = correlate(a, b, mode="sorted")
    assert np.isfinite(value)


def test_correlation_matrix_is_symmetric_with_unit_diagonal():
    info = make_info()
    a = make_scores({(0, 0): 0.1, (0, 1): 0.9, (2, 0): 0.1, (2, 1): 0.9}, info)
    b = make_scores({(0, 0): 0.9, (0, 1): 0.1, (2, 0): 0.1, (2, 1): 0.9}, info)
    corr = correlation_matrix([a, b], labels=["a", "b"])
    assert corr.values[0][0] == 1.0 and corr.values[1][1] == 1.0
    assert corr.values[0][1] == pytest.approx(corr.values[1][0])


def test_head_overlap_jaccard():
    info = make_info(num_layers=2, heads=2, scoreable=(0, 1))
    a = make_scores({(0, 0): 0.5, (0, 1): 0.5, (1, 0): 0.01, (1, 1): 0.01}, info)
    b = make_scores({(0, 0): 0.5, (0, 1): 0.01, (1, 0): 0.5, (1, 1): 0.01}, info)
    overlap = head_overlap(a, b, threshold=0.1)
    assert overlap.n_shared == 1
    assert overlap.jaccard == pytest.approx(1 / 3)


def test_sparsity_counts_only_scoreable_heads():
    info = make_info(num_layers=4, heads=2, scoreable=(0, 2))
    scores = make_scores({(0, 0): 0.0, (0, 1): 0.5, (2, 0): 0.0, (2, 1): 0.2}, info)
    stats = scores.sparsity()
    assert stats["n_heads"] == 4
    assert stats["thresholds"]["0.0"]["n"] == 2
    assert stats["thresholds"]["0.5"]["n"] == 0
    assert stats["thresholds"]["0.1"]["n"] == 2


def test_layer_profile_runs():
    info = make_info()
    scores = make_scores({(0, 0): 0.0, (0, 1): 0.5, (2, 0): 0.0, (2, 1): 0.2}, info)
    prof = layer_profile(scores)
    assert [row["layer"] for row in prof["layers"]] == [0, 2]


def test_normalized_contains_tolerates_markdown_and_case():
    needle = "The best thing to do in San Francisco is to eat a sandwich in Dolores Park on a sunny day."
    assert normalized_contains(
        "Based on the text provided, the best thing to do in San Francisco is to "
        "**eat a sandwich in Dolores Park on a sunny day**.", needle)
    assert normalized_contains("According to the text, " + needle.upper(), needle)
    assert not normalized_contains("nothing relevant here", needle)
    assert not normalized_contains("", needle)


def test_token_f1():
    assert token_f1([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert token_f1([1, 2], [3, 4]) == 0.0
    assert token_f1([1, 2, 3, 4], [1, 2]) == pytest.approx(2 * (2 / 4) * (2 / 2) / (2 / 4 + 1))
    assert token_f1([], [1]) == 0.0


def test_word_f1():
    assert word_f1("7241", "7241") == pytest.approx(1.0)
    assert word_f1("the northern depot", "northern depot") == pytest.approx(0.8)
    assert word_f1("banana", "7241") == 0.0


def test_final_answer_extraction():
    assert final_answer("blah\n#### 294") == "294"
    assert final_answer("reasoning...\nAnswer: 36") == "36"
    assert final_answer("just 12") == "just 12"


def test_accuracy_numeric_and_span():
    assert accuracy("#### 294", "294") == 1.0
    assert accuracy("The answer is 294 lanterns", "294") == 1.0
    assert accuracy("4.6 million crowns", "4.6 million crowns") == 1.0
    assert accuracy("Lupinus ferrugineus", "Lupinus ferrugineus") == 1.0
    assert accuracy("banana", "294") == 0.0


def test_matrix_shape_matches_model_census():
    info = make_info(num_layers=6, heads=3, scoreable=(1, 4))
    scores = make_scores({(1, 0): 0.2, (4, 2): 0.7}, info)
    assert scores.score.shape == (6, 3)
    assert scores.info.n_scoreable_heads == 6
    assert int(scores.scoreable_mask.sum()) == 6
