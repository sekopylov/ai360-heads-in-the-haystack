"""Regression tests: detection (split out of test_regressions.py).

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


def test_detection_save_writes_the_secondary_pairing(tmp_path):
    """`same_step` comes from the same pass, so its aggregate must be saved too."""
    from retrieval_heads.detection import DetectionConfig, DetectionRun

    info = attention_info(2, 2)
    primary = RetrievalScores(
        info=info, score=torch.tensor([[0.9, 0.2], [0.3, 0.0]]),
        activation_freq=torch.zeros(2, 2), n_instances=1, pairing="next_step",
    )
    secondary = RetrievalScores(
        info=info, score=torch.tensor([[0.1, 0.0], [0.8, 0.4]]),
        activation_freq=torch.zeros(2, 2), n_instances=1, pairing="same_step",
    )
    run = DetectionRun(scores=primary, instances=[], config=DetectionConfig(),
                       model_info=info, secondary=secondary)
    run.save(tmp_path)

    for pairing in ("next_step", "same_step"):
        assert (tmp_path / f"summary_{pairing}.json").exists()
        assert (tmp_path / f"scores_{pairing}.npz").exists()
    assert (tmp_path / "instances_next_step.jsonl").exists()


def test_accuracy_does_not_credit_a_suffix():
    """`t in p` marked "163" correct for "63"; that inflates short numeric answers."""
    from retrieval_heads.downstream import accuracy

    assert accuracy("19", "9") == 0.0
    assert accuracy("163", "63") == 0.0
    assert accuracy("10.5", "0.5") == 0.0
    # '.' is not a word character, so a plain \\w boundary still let a decimal tail
    # through: "1.63" credited "63" and "6.5" credited "6".
    assert accuracy("1.63", "63") == 0.0
    assert accuracy("6.5", "6") == 0.0
    assert accuracy("16.5", "6.5") == 0.0
    # ... but a real answer wrapped in prose still counts
    assert accuracy("the answer is 63", "63") == 1.0
    assert accuracy("63 chairs", "63") == 1.0
    assert accuracy("63", "63") == 1.0


def test_instance_result_meta_cannot_clobber_reserved_keys():
    from retrieval_heads.scoring import InstanceResult

    result = InstanceResult(
        sample={"prompt_tokens": 1}, scores={"next_step": {"L1H1": 0.5}},
        activations={}, considered={}, sink_rate={}, generated_ids=[],
        generated_text="", needle_recall=1.0, n_steps=7,
        # Deliberately collide with two reserved keys: before the fix `**meta` was
        # spread last and overwrote them.
        meta={"scores": "clobbered", "n_steps": 999, "pairing": "next_step"},
    )
    payload = result.as_dict()
    assert payload["scores"] == {"next_step": {"L1H1": 0.5}}
    assert payload["n_steps"] == 7
    assert payload["pairing"] == "next_step"   # a non-reserved key still passes through


def test_non_finite_scores_are_dropped_from_descriptive_stats():
    from retrieval_heads.properties import category_fractions

    info = attention_info(1, 3)
    scores = RetrievalScores(
        info=info, score=torch.tensor([[0.5, float("nan"), 0.0]]),
        activation_freq=torch.zeros(1, 3), n_instances=1, threshold=0.1,
    )

    fractions = category_fractions(scores)
    assert fractions["n_heads"] == 2 and fractions["n_non_finite"] == 1
    total = sum(fractions[b]["frac"] for b in ("zero", "low", "retrieval"))
    assert abs(total - 1.0) < 1e-9, total


def test_provenance_is_stamped_and_old_artifacts_warn(caplog):
    import logging

    from retrieval_heads.provenance import SCHEMA_VERSION, add_provenance, warn_if_stale

    payload = add_provenance({"x": 1}, dtype="bfloat16")
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["provenance"]["dtype"] == "bfloat16"
    assert payload["provenance"]["schema_version"] == SCHEMA_VERSION

    with caplog.at_level(logging.WARNING):
        warn_if_stale({"x": 1}, "old.json", log=logging.getLogger("test"))
    assert "schema_version" in caplog.text


def test_saved_scores_record_schema_and_dtype(tmp_path):
    import json

    info = attention_info(1, 2)
    info.dtype = "bfloat16"
    RetrievalScores(info=info, score=torch.tensor([[0.5, 0.1]]),
                    activation_freq=torch.zeros(1, 2), n_instances=1).save(
        tmp_path / "scores_next_step")

    meta = json.loads((tmp_path / "scores_next_step.json").read_text(encoding="utf-8"))
    assert meta["schema_version"] >= 1
    assert meta["provenance"]["dtype"] == "bfloat16"
    assert meta["model"]["dtype"] == "bfloat16"
    assert RetrievalScores.load(tmp_path / "scores_next_step").info.dtype == "bfloat16"


def test_save_json_is_atomic(tmp_path):
    import json

    from retrieval_heads.utils import save_json

    target = tmp_path / "artifact.json"
    save_json({"a": 1}, target)
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
    assert list(tmp_path.glob("*.tmp")) == [], "a temp file was left behind"


def test_detection_plan_seeds_are_unique_and_stable():
    from retrieval_heads.detection import DetectionConfig

    first = DetectionConfig(lengths=[1024, 4096], depths_per_length=3)
    second = DetectionConfig(lengths=[1024, 4096], depths_per_length=3)
    seeds = [item["seed"] for item in first.plan()]
    assert len(seeds) == len(set(seeds)), "duplicate filler seeds in the grid"
    # Two independently constructed configs, not the same cached plan twice.
    assert seeds == [item["seed"] for item in second.plan()], "seeds are not deterministic"


def test_save_json_writes_null_instead_of_nan(tmp_path):
    import json

    from retrieval_heads.utils import save_json

    target = tmp_path / "corr.json"
    save_json({"values": [[1.0, float("nan")]], "jaccard": float("nan")}, target)
    text = target.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text) == {"values": [[1.0, None]], "jaccard": None}


def test_finite_json_recurses_into_as_dict_objects_and_numpy(tmp_path):
    """A NaN inside an `as_dict()` payload must become null, not raise mid-write.

    `finite_json` used to walk only dicts and lists, so a dataclass that exposed a
    non-finite float through `as_dict()` reached `json.dump(..., allow_nan=False)`
    unfiltered -- the exact failure the helper exists to prevent.
    """
    import json

    import numpy as np

    from retrieval_heads.utils import finite_json, save_json

    class Payload:
        def __init__(self, value):
            self.value = value

        def as_dict(self):
            return {"value": self.value}

    assert finite_json({"p": Payload(float("nan"))}) == {"p": {"value": None}}
    assert finite_json({"p": Payload(0.5)}) == {"p": {"value": 0.5}}
    assert finite_json({"a": np.float64("nan"), "b": np.array([1.0, np.inf])}) == {
        "a": None, "b": [1.0, None]}

    target = tmp_path / "nested.json"
    save_json({"p": Payload(float("nan"))}, target)
    text = target.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert json.loads(text) == {"p": {"value": None}}


def test_detection_run_can_skip_rewriting_the_streamed_jsonl(tmp_path):
    from retrieval_heads.detection import DetectionConfig, DetectionRun

    info = attention_info(1, 2)
    scores = RetrievalScores(info=info, score=torch.tensor([[0.5, 0.1]]),
                             activation_freq=torch.zeros(1, 2), n_instances=0)
    run = DetectionRun(scores=scores, instances=[], config=DetectionConfig(), model_info=info)

    run.save(tmp_path / "with", write_instances=True)
    assert (tmp_path / "with" / "instances_next_step.jsonl").exists()

    run.save(tmp_path / "without", write_instances=False)
    assert not (tmp_path / "without" / "instances_next_step.jsonl").exists()
    assert (tmp_path / "without" / "summary_next_step.json").exists()


def test_detection_and_eval_needles_do_not_overlap():
    """The causal ablation must not be scored on the selection needles (leak)."""
    from retrieval_heads.detection import (
        DETECTION_NEEDLES, EVAL_NEEDLES, assert_needles_disjoint,
    )

    detection = {needle for needle, _ in DETECTION_NEEDLES}
    evaluation = {needle for needle, _ in EVAL_NEEDLES}
    assert detection and evaluation
    assert not (detection & evaluation), "eval needle is also a detection needle"
    assert_needles_disjoint()          # must not raise


def test_duplicate_layer_idx_is_fatal():
    """Silently keeping the first match collapsed a model onto one layer."""
    from torch import nn

    from retrieval_heads.models import discover_modules

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_idx = 0
            self.head_dim = 2
            self.num_heads = 1
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                setattr(self, name, nn.Linear(2, 2, bias=False))

    class Doubled(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = Attention()
            self.b = Attention()

    with pytest.raises(RuntimeError, match="layer_idx"):
        discover_modules(Doubled())


def test_run_detection_aggregates_streams_and_writes_conditional(tmp_path, monkeypatch, caplog):
    import logging

    """Exercise the whole detect pipeline without a checkpoint.

    This path (aggregation, streaming JSONL, recited-only matrices, the needle
    statistics) had no coverage at all, which is how a missing numpy import got
    through the suite.
    """
    from types import SimpleNamespace

    import retrieval_heads.detection as detection
    from retrieval_heads.detection import DetectionConfig, run_detection
    from retrieval_heads.scoring import InstanceResult

    info = attention_info(1, 2)

    def fake_build(tokenizer, *, needle, question, target_tokens, depth, builder,
                   chat_template=True, enable_thinking=False, system_prompt=None, seed=0):
        return SimpleNamespace(
            needle_span=(1, 3), needle_text=needle, question=question, depth=depth,
            length=target_tokens, target_tokens=target_tokens, seed=seed,
            n_needle_tokens=3, n_unique_needle_tokens=2,
            input_ids=torch.zeros(1, target_tokens, dtype=torch.long),
            # NOTE: no `truncated` key here on purpose -- NeedleSample has none, and
            # the old fake added one, which conserved the dead-warning bug.
            as_dict=lambda: {"needle_text": needle, "n_needle_tokens": 3,
                             "n_unique_needle_tokens": 2},
        )

    calls = {"n": 0}

    def fake_score(model, info_, sample, tokenizer, **kwargs):
        calls["n"] += 1
        recited = calls["n"] == 1                     # first instance is solved
        return InstanceResult(
            sample={"n_needle_tokens": 3, "n_unique_needle_tokens": 2,
                    "n_unique_needle_text_tokens": 2},
            meta={"eos_reached": False, "truncated": True},
            scores={"next_step": {"L0H0": 0.9 if recited else 0.0, "L0H1": 0.2},
                    "same_step": {"L0H0": 0.8 if recited else 0.0, "L0H1": 0.1}},
            # The raw per-token denominator: 2 unique tokens of 3, so the same
            # numerators over 3 instead of 2.
            scores_raw={"next_step": {"L0H0": 0.6 if recited else 0.0, "L0H1": 0.2 / 3},
                        "same_step": {"L0H0": 0.8 * 2 / 3 if recited else 0.0,
                                      "L0H1": 0.1 * 2 / 3}},
            activations={"next_step": {"L0H0": 1.0, "L0H1": 1.0},
                         "same_step": {"L0H0": 1.0, "L0H1": 1.0}},
            considered={}, sink_rate={"next_step": {"__overall__": 0.0}},
            generated_ids=[1], generated_text="x",
            needle_recall=0.9 if recited else 0.0, n_steps=1,
            aligned_scores={"next_step": {"L0H0": 0.9, "L0H1": 0.2},
                            "same_step": {"L0H0": 0.8, "L0H1": 0.1}},
            copied_tokens={"next_step": {"L0H0": [11, 12]}, "same_step": {"L0H0": [11]}},
            # The alternative domains the same decode pass captured.  `scores` is the
            # primary (`haystack` by default), so this map holds the complement.
            scores_by_domain={
                "prompt": {"next_step": {"L0H0": 0.9 if recited else 0.4, "L0H1": 0.0},
                           "same_step": {"L0H0": 0.8 if recited else 0.3, "L0H1": 0.0}},
                "full": {"next_step": {"L0H0": 0.9 if recited else 0.4, "L0H1": 0.0},
                         "same_step": {"L0H0": 0.8 if recited else 0.3, "L0H1": 0.0}},
            },
        )

    monkeypatch.setattr(detection, "build_needle_sample", fake_build)
    monkeypatch.setattr(detection, "score_instance", fake_score)

    config = DetectionConfig(lengths=[64], depths_per_length=2, needles=[("n", "q")])
    with caplog.at_level(logging.WARNING):
        run = run_detection(None, None, info, config, out_dir=tmp_path, progress=False)

    # Every instance is truncated, so the budget warning must fire (it read the wrong
    # dict before and never could).
    assert "budget-limited" in caplog.text

    summary = run.summary()
    assert summary["n_instances"] == 2
    # Recited-only numbers are stored per pairing, not shared.
    assert set(run.conditional) == {"next_step", "same_step"}
    assert run.summary(run.secondary)["sparsity_recited"] == \
        run.conditional["same_step"].sparsity()
    assert summary["config"]["grid_size"] == 2
    assert summary["needle_stats"]["denominator_inflation"] > 1.0   # 3 vs 2 unique
    # All four needle statistics must actually be present (one was written under a
    # different name and silently dropped).
    for key in ("needle_tokens_mean", "unique_needle_tokens_mean",
                "denominator_inflation", "tokenization_attainable_mean"):
        assert key in summary["needle_stats"], (key, summary["needle_stats"])
    # The numerator |g_h ∩ k| is auditable per head now.
    instance = run.instances[0]
    assert instance.copied_tokens["next_step"]["L0H0"], instance.copied_tokens
    assert summary["sparsity_recited"] is not None, "conditional matrices missing"
    assert summary["top_heads_recited"]

    jsonl = (tmp_path / "instances_next_step.jsonl").read_text(encoding="utf-8")
    assert len(jsonl.strip().splitlines()) == 2, "the JSONL was not streamed"
    assert (tmp_path / "scores_next_step_recited.npz").exists()
    # The raw per-token denominator is a first-class artifact, not a rescaling:
    # both pairings get their own sidecar, and the summary reports its sparsity.
    assert (tmp_path / "scores_next_step_raw.npz").exists()
    assert (tmp_path / "scores_same_step_raw.npz").exists()
    assert summary["sparsity_raw"] is not None
    # 0.6 on the recited instance and 0.0 on the other -> 0.3, below the 0.45 the
    # unique-token denominator gives for the same numerators.
    assert run.raw["next_step"].score[0, 0] == pytest.approx(0.3)
    assert run.raw["next_step"].score[0, 0] < run.scores.score[0, 0]
    assert run.raw["next_step"].meta["denominator"] == "raw_token_count"

    # Every captured argmax domain is aggregated from the same per-instance map and
    # written beside the primary matrix, so the "sparse" headline can be read against
    # the position set instead of being pinned by the run's own flag.
    assert set(run.domains) == {("next_step", "prompt"), ("next_step", "full"),
                                ("same_step", "prompt"), ("same_step", "full")}
    assert (tmp_path / "scores_next_step_prompt.npz").exists()
    assert (tmp_path / "scores_same_step_full.npz").exists()
    by_domain = summary["sparsity_by_domain"]
    assert set(by_domain) == {"haystack", "prompt", "full"}, by_domain
    assert by_domain["haystack"] == run.scores.sparsity()
    # 0.9 on the recited instance and 0.4 on the other -> 0.65, above the primary's
    # 0.45: the prompt domain credits a head the haystack domain does not.
    assert run.domains[("next_step", "prompt")].score[0, 0] == pytest.approx(0.65)
    assert run.domains[("next_step", "prompt")].meta["argmax_domain"] == "prompt"
    assert run.domains[("next_step", "prompt")].meta["domain_is_primary"] is False
    assert summary["top_heads_by_domain"]["prompt"][0]["head"] == "L0H0"
    # The derived activation frequency is exact (`activation_freq` *is* P(score > 0)).
    assert run.domains[("next_step", "prompt")].activation_freq[0, 0] == pytest.approx(1.0)
    assert run.domains[("next_step", "prompt")].activation_freq[0, 1] == pytest.approx(0.0)


def test_detection_config_plan_is_cached(caplog):
    import logging

    from retrieval_heads.detection import DetectionConfig

    config = DetectionConfig(lengths=[64], depths_per_length=2, needles=[("n", "q")], limit=1)
    with caplog.at_level(logging.WARNING):
        first = config.plan()
        config.as_dict()
        config.as_dict()
    assert len(first) == 1
    assert caplog.text.count("--limit") == 1, "the plan warning was re-logged"

    # A limit equal to the grid is not a subsample: the preflights pass `--limit 60`
    # for a grid of exactly 60, and a "debugging sample" warning there would make the
    # one cheap geometry measurement look biased.
    caplog.clear()
    exact = DetectionConfig(lengths=[64, 128], depths_per_length=2, needles=[("n", "q")],
                            limit=4)
    with caplog.at_level(logging.WARNING):
        assert len(exact.plan()) == 4
    assert "--limit" not in caplog.text, caplog.text


def test_score_instance_always_records_both_pairings():
    """The removed `compute_second_pairing` flag could strip a key summary() reads."""
    import inspect

    from retrieval_heads.scoring import score_instance

    assert "compute_second_pairing" not in inspect.signature(score_instance).parameters


def test_aligned_credits_are_a_subset_of_the_loose_ones():
    """The strict variant must never credit a token the paper's rule does not."""
    from retrieval_heads.scoring import (
        DecodeTrace, StepTrace, credits_aligned, credits_from_trace,
    )
    from tests.test_scoring import FakeSample, make_info, spike

    sample = FakeSample([11, 7, 8, 12, 13], (1, 3))
    info = make_info(num_layers=1, heads=1)
    trace = DecodeTrace(prompt_len=5, steps=[
        StepTrace(step=0, fed_token=1, predicted_token=7, attn={0: spike(1, 5, {0: 2})}),
        StepTrace(step=1, fed_token=7, predicted_token=8, attn={0: spike(1, 6, {0: 2})}),
    ])
    loose, _, _ = credits_from_trace(trace, sample, info, pairing="next_step")
    strict = credits_aligned(trace, sample, info, pairing="next_step")
    for head in info.scoreable_heads:
        assert strict[head] <= loose[head], (head, strict[head], loose[head])


def test_credits_refuse_a_head_count_that_changes_mid_run():
    """The accumulator used to silently restart, dropping everything counted so far.

    `sink_t` then failed on a shape mismatch anyway, so the "tolerance" only ever
    turned a loud error into a quiet loss of the tally.
    """
    from retrieval_heads.scoring import DecodeTrace, StepTrace, credits_from_trace
    from tests.test_scoring import FakeSample, make_info, spike

    sample = FakeSample([11, 7, 8, 12, 13], (1, 3))
    info = make_info(num_layers=1, heads=2)
    trace = DecodeTrace(prompt_len=5, steps=[
        StepTrace(step=0, fed_token=1, predicted_token=7, attn={0: spike(2, 5, {0: 2})}),
        # The same layer now reports one head instead of two.
        StepTrace(step=1, fed_token=7, predicted_token=8, attn={0: spike(1, 6, {0: 2})}),
    ])
    with pytest.raises(ValueError, match="cannot change mid-run"):
        credits_from_trace(trace, sample, info, pairing="next_step")


def test_squad_f1_is_shared_between_ids_and_words():
    from retrieval_heads.downstream import word_f1
    from retrieval_heads.masking import token_f1

    assert abs(token_f1([1, 2, 3], [1, 2]) - 0.8) < 1e-9
    assert abs(word_f1("a b c", "a b") - 0.8) < 1e-9
    assert token_f1([], [1]) == 0.0 and word_f1("", "x") == 0.0


def test_detection_summary_surfaces_aligned_and_honours_the_pairing():
    from retrieval_heads.detection import DetectionConfig, DetectionRun
    from retrieval_heads.scoring import InstanceResult

    info = attention_info(1, 2)
    primary = RetrievalScores(info=info, score=torch.tensor([[0.9, 0.1]]),
                              activation_freq=torch.zeros(1, 2), n_instances=1,
                              pairing="next_step")
    secondary = RetrievalScores(info=info, score=torch.tensor([[0.2, 0.8]]),
                                activation_freq=torch.zeros(1, 2), n_instances=1,
                                pairing="same_step")
    instance = InstanceResult(
        sample={}, scores={"next_step": {"L0H0": 0.9, "L0H1": 0.1},
                           "same_step": {"L0H0": 0.2, "L0H1": 0.8}},
        activations={}, considered={}, generated_ids=[],
        sink_rate={"next_step": {"__overall__": 0.5}, "same_step": {"__overall__": 0.1}},
        generated_text="", needle_recall=1.0, n_steps=1,
        aligned_scores={"next_step": {"L0H0": 0.4, "L0H1": 0.1},
                        "same_step": {"L0H0": 0.1, "L0H1": 0.9}},
    )
    run = DetectionRun(scores=primary, instances=[instance], config=DetectionConfig(),
                       model_info=info, secondary=secondary, n_planned=5)

    same = run.summary(secondary)
    assert "same_step" in same["pairing_comparison"]["primary"], "primary pairing not honoured"
    # It must compare same_step against next_step, not against itself (which gave
    # overlap == top_k and jaccard == 1.0 in every sidecar).
    comparison = same["pairing_comparison"]
    assert set(comparison["primary"]) | set(comparison["secondary"]) == {
        "next_step", "same_step"}, comparison
    assert comparison["overlap"] < comparison["top_k"], comparison
    assert same["aligned_top_heads"][0]["head"] == "L0H1"      # same_step aligned ranking
    assert same["n_planned"] == 5
    # sink rate is per pairing now: the secondary summary must not show next_step's
    assert same["mean_sink_rate"] == 0.1
    assert run.summary()["mean_sink_rate"] == 0.5

    nxt = run.summary()
    assert nxt["aligned_top_heads"][0]["head"] == "L0H0"       # next_step aligned ranking

    # without a secondary, the passed pairing is still the primary
    run.secondary = None
    only = DetectionRun(scores=secondary, instances=[instance], config=DetectionConfig(),
                        model_info=info)
    assert list(only.summary(secondary)["pairing_comparison"]["primary"]) == ["same_step"]


def test_load_rejects_a_stored_mask_that_disagrees(tmp_path):
    """A mask that drifted from its metadata would mis-place every head: refuse."""
    import numpy as np

    info = attention_info(1, 2)
    RetrievalScores(info=info, score=torch.tensor([[0.5, 0.1]]),
                    activation_freq=torch.zeros(1, 2), n_instances=1).save(tmp_path / "s")

    stored = dict(np.load(tmp_path / "s.npz"))
    stored["scoreable_mask"] = ~stored["scoreable_mask"]
    np.savez_compressed(tmp_path / "s.npz", **stored)

    with pytest.raises(ValueError, match="scoreable_mask"):
        RetrievalScores.load(tmp_path / "s")


def test_needle_sample_meta_cannot_clobber_reserved_keys():
    from retrieval_heads.haystack import NeedleSample

    sample = NeedleSample(
        prompt_text="p", input_ids=torch.zeros(1, 3, dtype=torch.long), needle_span=(1, 2),
        needle_text="n", question="q", depth=0.5, target_tokens=3, prompt_tokens=3,
        seed=0, meta={"needle_text": "clobbered", "seed": 999},
    )
    payload = sample.as_dict()
    assert payload["needle_text"] == "n"
    assert payload["seed"] == 0


def test_cross_kind_layer_idx_collision_is_fatal():
    from torch import nn

    from retrieval_heads.models import discover_modules

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_idx = 0
            self.head_dim = 2
            self.num_heads = 1
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                setattr(self, name, nn.Linear(2, 2, bias=False))

    class LinearAttention(nn.Module):      # name matches _LINEAR_MARKERS
        def __init__(self):
            super().__init__()
            self.layer_idx = 0

    class Both(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = Attention()
            self.mixer = LinearAttention()

    with pytest.raises(RuntimeError, match="both"):
        discover_modules(Both())


def test_detection_saves_and_summarises_every_argmax_domain(tmp_path):
    """Which position set criterion (2) searches is a reporting decision.

    Every domain is captured in the same forward pass, so the run must write a matrix
    per domain and say how much the "sparse" headline moves with it -- otherwise the
    choice is baked into an expensive run that cannot be re-read.
    """
    from retrieval_heads.detection import DetectionConfig, DetectionRun

    info = attention_info(2, 2)
    primary = RetrievalScores(
        info=info, score=torch.tensor([[0.9, 0.2], [0.3, 0.0]]),
        activation_freq=torch.zeros(2, 2), n_instances=1, pairing="next_step",
        meta={"argmax_domain": "haystack"},
    )
    alt = RetrievalScores(
        # 0.05 rather than a value at the threshold: `head_score` compares a float32
        # score against the Python float, so a score that *is* float32(0.1) counts as
        # above 0.1 (it is 0.100000001...).  Both `heads_above` and `sparsity` do that
        # consistently; the test avoids depending on it.
        info=info, score=torch.tensor([[0.05, 0.9], [0.0, 0.0]]),
        activation_freq=torch.zeros(2, 2), n_instances=1, pairing="next_step",
        meta={"argmax_domain": "prompt"},
    )
    run = DetectionRun(
        scores=primary, instances=[], config=DetectionConfig(), model_info=info,
        domains={("next_step", "prompt"): alt},
    )
    run.save(tmp_path)
    assert (tmp_path / "scores_next_step.npz").exists()
    assert (tmp_path / "scores_next_step_prompt.npz").exists()
    assert (tmp_path / "scores_next_step_prompt.json").exists()

    summary = run.summary()
    by_domain = summary["sparsity_by_domain"]
    assert set(by_domain) == {"haystack", "prompt"}, by_domain
    assert by_domain["haystack"] == primary.sparsity()
    assert by_domain["prompt"] == alt.sparsity()
    # The masking arm ranks by the primary domain, so the summary says how much the
    # alternative domain would change the set it masks.  (The summary's own overlap
    # uses the top 10, which here is every head; the ranking difference is visible at
    # top_k=1, where the two domains pick different heads.)
    overlap = run.domain_ranking_overlap(top_k=1)
    assert overlap["primary_domain"] == "haystack"
    assert overlap["by_domain"]["prompt"]["top_heads"] == ["L0H1"]
    assert overlap["by_domain"]["prompt"]["overlap"] == 0
    assert summary["domain_ranking_overlap"]["primary_domain"] == "haystack"
    assert set(summary["top_heads_by_domain"]) == {"haystack", "prompt"}
    # The random arm's pool per domain: `mask` draws its control from these heads, so
    # the number is reported before the expensive stage runs (on the hybrid it can be
    # small enough that every large-K point collapses into one intervention).
    pools = summary["retrieval_pool_by_domain"]
    assert pools["haystack"] == {"threshold": 0.1, "n_heads": 4,
                                 "n_above_threshold": 3, "n_pool": 1}, pools
    assert pools["prompt"]["n_pool"] == 3, pools


def _preflight_harness(monkeypatch, *, span_for):
    """A fake build/score pair that records the order of the two calls."""
    import logging
    from types import SimpleNamespace

    from retrieval_heads import detection
    from retrieval_heads.scoring import InstanceResult

    info = attention_info(1, 2)
    order: list[str] = []

    def fake_build(tokenizer, *, needle, question, target_tokens, depth, builder,
                   chat_template=True, enable_thinking=False, system_prompt=None, seed=0):
        order.append("build")
        # 1-based build count, so a test can make a *specific* instance fail without
        # depending on the plan's seed arithmetic.
        index = sum(1 for entry in order if entry == "build")
        return SimpleNamespace(
            needle_span=(1, 3), needle_text=needle, question=question, depth=depth,
            length=target_tokens, target_tokens=target_tokens, seed=seed,
            n_needle_tokens=2, n_unique_needle_tokens=2,
            input_ids=torch.zeros(1, target_tokens, dtype=torch.long),
            haystack_span=span_for(index),
            as_dict=lambda: {"needle_text": needle, "n_needle_tokens": 2,
                             "n_unique_needle_tokens": 2},
        )

    def fake_score(model, info_, sample, tokenizer, **kwargs):
        order.append("score")
        return InstanceResult(
            sample={"n_needle_tokens": 2, "n_unique_needle_tokens": 2,
                    "n_unique_needle_text_tokens": 2},
            meta={"eos_reached": True, "truncated": False},
            scores={"next_step": {"L0H0": 0.5, "L0H1": 0.0},
                    "same_step": {"L0H0": 0.4, "L0H1": 0.0}},
            scores_raw={"next_step": {"L0H0": 0.5, "L0H1": 0.0},
                        "same_step": {"L0H0": 0.4, "L0H1": 0.0}},
            activations={"next_step": {"L0H0": 1.0, "L0H1": 0.0},
                         "same_step": {"L0H0": 1.0, "L0H1": 0.0}},
            considered={}, sink_rate={"next_step": {"__overall__": 0.0}},
            generated_ids=[1], generated_text="x", needle_recall=1.0, n_steps=1,
            aligned_scores={"next_step": {"L0H0": 0.5, "L0H1": 0.0},
                            "same_step": {"L0H0": 0.4, "L0H1": 0.0}},
            copied_tokens={"next_step": {"L0H0": [1]}, "same_step": {"L0H0": [1]}},
        )

    monkeypatch.setattr(detection, "build_needle_sample", fake_build)
    monkeypatch.setattr(detection, "score_instance", fake_score)
    return info, order, logging


def test_preflight_builds_every_prompt_before_the_first_forward(monkeypatch):
    """The whole point of the preflight: all CPU work, then the GPU work.

    A prompt whose haystack cannot be located used to abort `detect` mid-grid, after
    the GPU time already spent; building everything first turns that into a failure
    before the first forward pass.
    """
    from retrieval_heads.detection import DetectionConfig, run_detection

    info, order, _ = _preflight_harness(monkeypatch, span_for=lambda index: (0, 4))
    config = DetectionConfig(lengths=[64], depths_per_length=2, needles=[("n", "q")],
                             preflight=True)
    run = run_detection(None, None, info, config, progress=False)

    assert order == ["build", "build", "score", "score"], order
    assert run.summary()["config"]["preflight"] is True
    assert run.summary()["n_instances"] == 2


def test_preflight_fails_before_any_gpu_work(monkeypatch):
    """A missing haystack span must stop the run before the first forward pass."""
    from retrieval_heads.detection import DetectionConfig, run_detection

    # The second instance's prompt cannot be located verbatim, so `haystack` has no
    # span (the builder records that in `haystack_span_verbatim`).
    info, order, _ = _preflight_harness(
        monkeypatch, span_for=lambda index: None if index == 2 else (0, 4))
    config = DetectionConfig(lengths=[64], depths_per_length=2, needles=[("n", "q")],
                             preflight=True)
    with pytest.raises(ValueError, match="preflight failed on instance"):
        run_detection(None, None, info, config, progress=False)
    assert order == ["build", "build"], "a forward pass ran before the preflight ended"

    # Without the flag the two phases interleave: the first instance is scored before
    # the second prompt is even built, which is why one bad prompt costs the GPU time
    # already spent.  (The real `score_instance` is what refuses the missing span --
    # see `test_haystack_domain_runs_end_to_end_on_the_tiny_hybrid`; this fake scorer
    # has no span check of its own.)
    order.clear()
    run_detection(None, None, info, DetectionConfig(
        lengths=[64], depths_per_length=2, needles=[("n", "q")]), progress=False)
    assert order == ["build", "score", "build", "score"], order


def test_a_domain_missing_on_some_instances_is_averaged_over_the_subset(monkeypatch, caplog):
    """`haystack` is undefined for a prompt that does not contain the context verbatim.

    A `prompt`-domain run can therefore capture it on some instances and not others.
    That used to be a `KeyError` in the per-domain aggregation -- after the GPU time
    already spent -- so the matrix must instead be the mean over the instances that
    have the domain, with the missing count recorded in the artifact.
    """
    import logging
    from types import SimpleNamespace

    from retrieval_heads import detection
    from retrieval_heads.detection import DetectionConfig, run_detection
    from retrieval_heads.scoring import InstanceResult

    info = attention_info(1, 2)
    calls = {"n": 0}

    def fake_build(tokenizer, **kwargs):
        return SimpleNamespace(
            needle_span=(1, 3), needle_text="n", question="q", depth=0.5, length=24,
            target_tokens=24, seed=0, n_needle_tokens=2, n_unique_needle_tokens=2,
            input_ids=torch.zeros(1, 24, dtype=torch.long), haystack_span=(0, 4),
            as_dict=lambda: {"needle_text": "n", "n_needle_tokens": 2,
                             "n_unique_needle_tokens": 2},
        )

    def fake_score(model, info_, sample, tokenizer, **kwargs):
        calls["n"] += 1
        # The second instance's prompt had no verbatim haystack, so that domain is
        # absent from its map (the first instance has it).
        by_domain = {"full": {"next_step": {"L0H0": 0.5, "L0H1": 0.0},
                              "same_step": {"L0H0": 0.5, "L0H1": 0.0}}}
        if calls["n"] == 1:
            by_domain["haystack"] = {"next_step": {"L0H0": 0.25, "L0H1": 0.0},
                                     "same_step": {"L0H0": 0.25, "L0H1": 0.0}}
        return InstanceResult(
            sample={"n_needle_tokens": 2, "n_unique_needle_tokens": 2,
                    "n_unique_needle_text_tokens": 2},
            meta={"eos_reached": True, "truncated": False,
                  "argmax_domains": ["prompt", "full"] + (["haystack"] if calls["n"] == 1
                                                          else [])},
            scores={"next_step": {"L0H0": 0.6, "L0H1": 0.0},
                    "same_step": {"L0H0": 0.6, "L0H1": 0.0}},
            scores_raw={"next_step": {"L0H0": 0.6, "L0H1": 0.0},
                        "same_step": {"L0H0": 0.6, "L0H1": 0.0}},
            scores_by_domain=by_domain,
            activations={"next_step": {"L0H0": 1.0, "L0H1": 0.0},
                         "same_step": {"L0H0": 1.0, "L0H1": 0.0}},
            considered={}, sink_rate={"next_step": {"__overall__": 0.0}},
            generated_ids=[1], generated_text="x", needle_recall=1.0, n_steps=1,
            aligned_scores={"next_step": {"L0H0": 0.6, "L0H1": 0.0},
                            "same_step": {"L0H0": 0.6, "L0H1": 0.0}},
            copied_tokens={"next_step": {"L0H0": [1]}, "same_step": {"L0H0": [1]}},
        )

    monkeypatch.setattr(detection, "build_needle_sample", fake_build)
    monkeypatch.setattr(detection, "score_instance", fake_score)
    config = DetectionConfig(lengths=[24], depths_per_length=2, needles=[("n", "q")],
                             argmax_domain="prompt")
    with caplog.at_level(logging.WARNING):
        run = run_detection(None, None, info, config, progress=False)

    assert "haystack domain is missing on 1/2 instances" in caplog.text
    haystack = run.domains[("next_step", "haystack")]
    # The mean is over the single instance that has the domain, not over both (which
    # would silently treat the missing one as a zero).
    assert haystack.score[0, 0].item() == pytest.approx(0.25)
    assert haystack.n_instances == 1
    assert haystack.meta["n_instances_without_domain"] == 1
    full = run.domains[("next_step", "full")]
    assert full.n_instances == 2 and full.score[0, 0].item() == pytest.approx(0.5)
