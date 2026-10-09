"""`qa_ablation`: the matched arms, the control pool and the recorded spread.

The QA/COT ablations had no test at all -- they are the Sec. 5 pipeline, and the
numbers they write (baseline, retrieval arm, random arm, drops) are exactly what a
reader quotes.  These run on the local tiny hybrid with a canned completion, so no
checkpoint is needed and the arithmetic is checkable by hand.
"""

from __future__ import annotations

import pytest
import torch

from retrieval_heads.downstream import QASample, qa_ablation
from retrieval_heads.scoring import RetrievalScores


def _scores(info, above=((2, 0),)):
    """One retrieval head (layer 2, head 0), everything else below the threshold."""
    score = torch.zeros(info.num_layers, info.max_heads)
    for layer, head in above:
        score[layer, head] = 0.5
    return RetrievalScores(info=info, score=score,
                           activation_freq=torch.zeros_like(score),
                           n_instances=1, threshold=0.1)


def _canned(monkeypatch, answers):
    """Make `_generate_text` return a fixed answer per prompt, in order."""
    calls = {"n": 0}

    def fake_generate(model, tokenizer, prompt, **kwargs):
        answer = answers[calls["n"] % len(answers)]
        calls["n"] += 1
        return answer

    monkeypatch.setattr("retrieval_heads.downstream._generate_text", fake_generate)
    return calls


def test_qa_ablation_records_the_retrieval_arms_per_sample_spread(tiny_hybrid, monkeypatch):
    """A mean of 50 is not the same measurement as a spread of 50.

    The artifact used to record only means, so "every item lost half its F1" and
    "one item collapsed" were indistinguishable -- the reason those error bars were
    absent from the QA table.
    """
    model, info = tiny_hybrid
    samples = [QASample("ctx one", "q1", "7241"), QASample("ctx two", "q2", "7241")]
    # The same two completions for every arm: item 1 right, item 2 wrong.
    _canned(monkeypatch, ["7241", "banana"])

    out = qa_ablation(model, None, info, _scores(info), samples, k_values=[2],
                      n_random_trials=2, chat_template=False)

    assert out["baseline_f1"] == 50.0
    assert out["baseline_f1s"] == [100.0, 0.0]
    assert out["baseline_f1_std"] == 50.0
    row = out["by_k"]["2"]
    assert row["k_effective"] == 2
    assert row["retrieval_f1"] == 50.0
    assert row["retrieval_f1s"] == [100.0, 0.0]
    assert row["retrieval_f1_std"] == 50.0
    assert row["drop_retrieval"] == 0.0
    # The control arm is still matched in size and drawn without repetition, and the
    # audit fields that say so are recorded rather than trusted.
    assert row["random_f1_std"] >= 0.0
    assert len(row["masked_heads"]) == 2
    assert len(row["random_picks"]) == 2
    assert len({tuple(p) for p in row["random_picks"]}) == 2
    # With only one head above the threshold, K=2 reaches into the control pool, so
    # the arms genuinely overlap -- which is exactly what this audit field exists to
    # expose (the earlier "random arm contains no retrieval head" claim is only true
    # while K stays inside the above-threshold set).
    masked = set(row["masked_heads"])
    assert row["random_retrieval_overlap"] == [
        len(masked & set(pick)) for pick in row["random_picks"]
    ]


def test_qa_ablation_reports_an_absent_control_instead_of_a_silent_zero(
        tiny_hybrid, monkeypatch):
    """Every head above the threshold means the random arm is contaminated."""
    model, info = tiny_hybrid
    _canned(monkeypatch, ["7241"])
    # Flag every scoreable head.
    above = tuple((layer, head) for layer in info.scoreable_layers
                  for head in range(info.num_heads[layer]))
    out = qa_ablation(model, None, info, _scores(info, above=above),
                      [QASample("c", "q", "7241")], k_values=[1], n_random_trials=1,
                      chat_template=False)
    assert out["random_control_contaminated"] is True
    assert out["n_non_retrieval_heads"] == len(above)
