"""Regression tests for the defects found in review.

Each test here pins a specific failure mode that the old code had, so the fix
cannot quietly regress.  They are all fast (no checkpoints): the only fixture
that touches disk is the Qwen3-0.6B tokenizer, which needs no weights.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import retrieval_heads.masking as masking
from retrieval_heads.cli import normalize_prefill_chunk, resolve_k
from retrieval_heads.haystack import HaystackBuilder, build_needle_sample
from retrieval_heads.masking import (
    NiahMetrics,
    control_pool,
    matched_k,
    token_mixer_ablation,
)
from retrieval_heads.models import ModelInfo, require_scoreable
from retrieval_heads.scoring import RetrievalScores
from tests.conftest import REPO_ROOT


# --------------------------------------------------------------------------- helpers
def attention_info(num_layers: int, heads: int, name: str = "toy") -> ModelInfo:
    return ModelInfo(
        name=name, path=name, model_type="toy", num_layers=num_layers,
        layer_types=["full_attention"] * num_layers,
        num_heads={i: heads for i in range(num_layers)},
        num_kv_heads={i: heads for i in range(num_layers)},
        head_dim=4, hidden_size=8, max_position_embeddings=128,
        scoreable_layers_=list(range(num_layers)),
    )


class _Args:
    def __init__(self, k=None, k_frac=None):
        self.k = k
        self.k_frac = k_frac


def scores_with(values: list[float], *, threshold: float = 0.1) -> RetrievalScores:
    """A single-layer score matrix, one value per head."""
    info = attention_info(1, len(values))
    score = torch.tensor([values], dtype=torch.float32)
    return RetrievalScores(
        info=info, score=score, activation_freq=torch.zeros_like(score),
        n_instances=1, threshold=threshold,
    )


# --------------------------------------------------------------------------- resolve_k
def test_explicit_k_is_not_unioned_with_the_default_fractions():
    """`mask --k 1 2 4 8 16` must select exactly those five K values."""
    info = attention_info(28, 16)  # 448 scoreable heads, like Qwen3-0.6B
    assert info.n_scoreable_heads == 448
    args = _Args(k=[1, 2, 4, 8, 16])
    assert resolve_k(args, info, default_fracs=(0.02, 0.04, 0.08, 0.17, 0.33)) == [1, 2, 4, 8, 16]


def test_default_fractions_apply_only_when_no_k_is_given():
    info = attention_info(28, 16)
    args = _Args()
    assert resolve_k(args, info, default_fracs=(0.02, 0.04, 0.08)) == [9, 18, 36]


def test_explicit_fractions_select_only_those():
    info = attention_info(28, 16)
    args = _Args(k_frac=[0.01, 0.5])
    assert resolve_k(args, info, default_fracs=(0.02,)) == [4, 224]


def test_k_and_fractions_are_unioned_only_when_both_are_explicit():
    info = attention_info(28, 16)
    assert resolve_k(_Args(k=[8], k_frac=[0.02]), info, default_fracs=(0.33,)) == [8, 9]


def test_normalize_prefill_chunk_maps_zero_to_one_shot():
    assert normalize_prefill_chunk(0) is None
    assert normalize_prefill_chunk(None) is None
    assert normalize_prefill_chunk(-4) is None
    assert normalize_prefill_chunk(4096) == 4096


def test_decode_rejects_a_non_positive_chunk_before_touching_the_model():
    """The guard runs before any model use, so it needs no checkpoint."""
    from retrieval_heads.scoring import decode_with_attention

    with pytest.raises(ValueError, match="prefill_chunk"):
        decode_with_attention(None, None, torch.zeros(1, 4, dtype=torch.long), prefill_chunk=0)


# --------------------------------------------------------------------------- controls
def test_random_pool_excludes_retrieval_heads():
    scores = scores_with([0.9, 0.5, 0.05, 0.0])
    pool, _ = control_pool(scores)
    assert [str(h) for h in pool] == ["L0H2", "L0H3"]


def test_matched_k_caps_both_arms_at_the_pool_size():
    assert matched_k(4, pool_size=18) == 4
    assert matched_k(64, pool_size=18) == 18
    assert matched_k(3, pool_size=0) == 3


def test_mixer_ablation_keeps_both_arms_aligned(monkeypatch):
    """K larger than a stack must truncate, not desynchronise the two lists."""
    info = ModelInfo(
        name="hybrid", path="h", model_type="toy", num_layers=24,
        layer_types=["linear_attention"] * 24,
        num_heads={l: 8 for l in (3, 7, 11, 15, 19, 23)},
        num_kv_heads={l: 2 for l in (3, 7, 11, 15, 19, 23)},
        head_dim=4, hidden_size=8, max_position_embeddings=128,
        num_linear_heads={l: 16 for l in range(24) if l not in (3, 7, 11, 15, 19, 23)},
        scoreable_layers_=[3, 7, 11, 15, 19, 23],
        linear_layers_=[l for l in range(24) if l not in (3, 7, 11, 15, 19, 23)],
    )
    seen: list[tuple[int, ...]] = []

    def fake_evaluate(model, tokenizer, info_, samples, *, masked_layers=(), masked_heads=(),
                      max_new_tokens=32, prefill_chunk=None, attn_impl="sdpa"):
        seen.append(tuple(masked_layers))
        return NiahMetrics(f1=50.0, exact_match=50.0, recall=50.0, n=0)

    monkeypatch.setattr(masking, "evaluate_samples", fake_evaluate)
    abl = token_mixer_ablation(None, None, info, [], k_values=(1, 2, 4, 6, 8, 20))

    # One entry per requested K for *both* arms: the old code appended a duplicate
    # full-attention value for K > n_full and dropped the linear arm entirely when
    # K > n_linear, so the two lists silently stopped corresponding.
    assert abl.k_values == [1, 2, 4, 6, 8, 20]
    assert len(abl.full_attention) == len(abl.linear_attention) == 6
    # The realized counts differ because the stacks differ (6 attention vs 18
    # linear); that is recorded rather than concealed.
    assert abl.full_attention_masked == [1, 2, 4, 6, 6, 6]
    assert abl.linear_attention_masked == [1, 2, 4, 6, 8, 18]
    assert all(f <= 1.0 for f in abl.fractions)
    assert abl.fractions[-1] == 1.0
    assert info.n_all_heads == 48 + 18 * 16


# --------------------------------------------------------------------------- validation
def test_require_scoreable_rejects_a_model_with_no_attention():
    info = ModelInfo(
        name="all-linear", path="x", model_type="toy", num_layers=2,
        layer_types=["linear_attention"] * 2, num_heads={}, num_kv_heads={},
        head_dim=0, hidden_size=8, max_position_embeddings=128,
        scoreable_layers_=[], linear_layers_=[0, 1],
    )
    with pytest.raises(RuntimeError, match="no scoreable softmax-attention layers"):
        require_scoreable(info)


def test_missing_scores_artifact_names_the_detect_command(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run `detect`"):
        RetrievalScores.load(tmp_path / "does-not-exist")


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


# --------------------------------------------------------------------------- accuracy
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


# --------------------------------------------------------------------------- controls
def test_control_pool_flags_contamination():
    from retrieval_heads.masking import control_pool

    clean = scores_with([0.9, 0.05, 0.0])          # two heads below the threshold
    pool, contaminated = control_pool(clean)
    assert [str(h) for h in pool] == ["L0H1", "L0H2"]
    assert contaminated is False

    dirty = scores_with([0.9, 0.5])                 # every head is a retrieval head
    pool, contaminated = control_pool(dirty)
    assert len(pool) == 2 and contaminated is True


def test_masking_curve_lists_stay_aligned_and_audit_the_control(monkeypatch):
    """Every parallel list must have one entry per evaluated K, cap included."""
    import retrieval_heads.masking as masking
    from retrieval_heads.masking import NiahMetrics, masking_curve

    info = attention_info(1, 4)
    scores = RetrievalScores(
        info=info, score=torch.tensor([[0.9, 0.8, 0.05, 0.0]]),
        activation_freq=torch.zeros(1, 4), n_instances=1, threshold=0.1,
    )

    def fake_evaluate(model, tokenizer, info_, samples, *, masked_heads=(), masked_layers=(),
                      max_new_tokens=32, prefill_chunk=None, attn_impl="sdpa"):
        # Two samples with different f1: the spread must survive into the artifact.
        return NiahMetrics(f1=50.0, exact_match=50.0, recall=50.0, n=2,
                           f1s=[40.0, 60.0], exact_matches=[100.0, 0.0],
                           recalls=[50.0, 50.0])

    monkeypatch.setattr(masking, "evaluate_samples", fake_evaluate)
    curve = masking_curve(None, None, info, scores, [], k_values=(1, 2, 5),
                          n_random_trials=2, seed=0, progress=False)

    n = len(curve.k_values)
    assert n == 3
    for series in (curve.k_effective, curve.retrieval, curve.retrieval_exact,
                   curve.random_mean, curve.random_std, curve.random_trials,
                   curve.random_retrieval_overlap, curve.retrieval_std,
                   curve.retrieval_exact_std):
        assert len(series) == n
    assert curve.k_effective == [1, 2, 2]           # capped at the 2-head pool
    assert curve.retrieval_std == [10.0, 10.0, 10.0], curve.retrieval_std
    assert all(overlap == [0, 0] for overlap in curve.random_retrieval_overlap)
    assert curve.meta["k_fraction_effective"][-1] == 2 / 4


def test_summarizer_reports_the_effective_k(tmp_path):
    """A capped K must not be reported as the requested fraction of heads."""
    import json

    from scripts.summarize_results import masking_section

    model_dir = tmp_path / "m"
    model_dir.mkdir()
    (model_dir / "masking_curve.json").write_text(json.dumps({
        "model": "m", "k_values": [1, 2, 5], "k_effective": [1, 2, 2],
        "n_scoreable_heads": 4, "retrieval": [90.0, 80.0, 70.0],
        "random_mean": [95.0, 95.0, 95.0], "random_std": [0.0, 0.0, 0.0],
        "retrieval_exact_match": [100.0, 100.0, 0.0],
        "random_exact_match_mean": [100.0, 100.0, 100.0],
        "baseline": 95.0, "baseline_exact_match": 100.0,
    }), encoding="utf-8")

    table = "\n".join(masking_section(tmp_path, ["m"]))
    # The capped point must be labelled and counted by its effective K, not by the
    # requested 5 (which would read as 125% of the 4 heads).
    assert "| 5 →2 | 50.0% |" in table, table
    assert "125.0%" not in table, table


def test_load_registry_honours_the_env_override(tmp_path, monkeypatch):
    import json

    from retrieval_heads.cli import load_registry

    registry = tmp_path / "runtime.json"
    registry.write_text(json.dumps({"models": {"only-here": {"path": "p"}}}), encoding="utf-8")
    monkeypatch.setenv("RETRIEVAL_HEADS_MODELS_JSON", str(registry))
    assert "only-here" in load_registry()
    monkeypatch.delenv("RETRIEVAL_HEADS_MODELS_JSON")
    assert "only-here" not in load_registry()


def test_override_dtype_leaves_the_tracked_registry_alone(tmp_path, monkeypatch):
    """The driver must write a runtime copy, not modify configs/models.json."""
    import json
    import os
    import shutil

    from tests.test_cli_argv import load_job_driver

    (tmp_path / "configs").mkdir()
    tracked_path = tmp_path / "configs" / "models.json"
    shutil.copy(REPO_ROOT / "configs" / "models.json", tracked_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RETRIEVAL_HEADS_MODELS_JSON", raising=False)
    before = tracked_path.read_bytes()

    load_job_driver().override_dtype(["qwen3-0.6b"], "bfloat16")

    # Compare bytes, not a literal dtype: asserting "float32" would fail for the
    # wrong reason the day the project changes its default precision.
    assert tracked_path.read_bytes() == before
    runtime = json.loads((tmp_path / "configs" / "models.runtime.json").read_text(encoding="utf-8"))
    assert runtime["models"]["qwen3-0.6b"]["dtype"] == "bfloat16"
    assert os.environ["RETRIEVAL_HEADS_MODELS_JSON"].endswith("models.runtime.json")


# --------------------------------------------------------------------------- metadata
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


# --------------------------------------------------------------------------- A1-A7
def test_require_matching_scores_rejects_a_foreign_artifact():
    from retrieval_heads.cli import require_matching_scores

    loaded = attention_info(2, 2)
    matching = RetrievalScores(info=attention_info(2, 2), score=torch.zeros(2, 2),
                               activation_freq=torch.zeros(2, 2), n_instances=1)
    require_matching_scores(matching, loaded)          # must not raise

    foreign = RetrievalScores(info=attention_info(3, 2), score=torch.zeros(3, 2),
                              activation_freq=torch.zeros(3, 2), n_instances=1)
    with pytest.raises(SystemExit, match="not"):
        require_matching_scores(foreign, loaded)


def test_within_context_limit_drops_extrapolated_lengths():
    from retrieval_heads.cli import within_context_limit

    from retrieval_heads.cli import PROMPT_OVERHEAD_TOKENS

    info = attention_info(1, 1)
    info.max_position_embeddings = 40960
    # A requested length equal to the window is dropped: the realized prompt is
    # filler + needle + question + template, i.e. ~64 tokens longer.
    kept, dropped = within_context_limit([1024, 49152, 40960], info)
    assert kept == [1024]
    assert dropped == [49152, 40960]
    assert within_context_limit([40960 - PROMPT_OVERHEAD_TOKENS], info)[0] == [
        40960 - PROMPT_OVERHEAD_TOKENS]
    info.max_position_embeddings = None
    assert within_context_limit([1024, 49152], info) == ([1024, 49152], [])


def test_eos_ids_collects_config_generation_and_tokenizer():
    from types import SimpleNamespace

    from retrieval_heads.utils import eos_ids

    model = SimpleNamespace(config=SimpleNamespace(eos_token_id=1),
                            generation_config=SimpleNamespace(eos_token_id=[2, 3]))
    assert eos_ids(model, SimpleNamespace(eos_token_id=4)) == {1, 2, 3, 4}
    assert eos_ids(model, None) == {1, 2, 3}
    assert eos_ids(SimpleNamespace(), None) == set()


def test_make_eval_samples_uses_the_corpus(tokenizer):
    from retrieval_heads.detection import DEFAULT_NEEDLES
    from retrieval_heads.masking import make_eval_samples

    needle, question = DEFAULT_NEEDLES[1]
    samples = make_eval_samples(
        tokenizer, lengths=(256,), depths=(0.5,), needle=needle, question=question,
        seed=0, corpus=["Zebra quark nebula sentence."],
    )
    assert samples and all("zebra" in s.prompt_text.lower() for s in samples)


def test_mask_parser_accepts_a_corpus():
    from retrieval_heads.cli import build_parser

    args = build_parser().parse_args(["mask", "--model", "m", "--corpus", "c.txt"])
    assert args.corpus == "c.txt"


def test_default_out_dir_keeps_path_models_out_of_their_checkpoint():
    from retrieval_heads.cli import REPO_ROOT, default_out_dir

    assert default_out_dir("/abs/checkpoints/Qwen3-0.6B", None) == \
        REPO_ROOT / "results" / "Qwen3-0.6B"
    assert default_out_dir("qwen3-0.6b", None) == REPO_ROOT / "results" / "qwen3-0.6b"
    assert default_out_dir("qwen3-0.6b", "/tmp/x") == Path("/tmp/x")


def test_plot_masking_curve_exact_match_requires_the_series():
    import matplotlib.pyplot as plt

    from retrieval_heads.plotting import plot_masking_curve

    old_artifact = {"m": {"k_values": [1], "k_effective": [1], "retrieval": [50.0],
                          "random_mean": [60.0], "random_std": [1.0], "baseline": 70.0}}
    with pytest.raises(ValueError, match="exact-match"):
        plot_masking_curve(old_artifact, metric="exact_match")
    plt.close("all")


def test_load_runs_disambiguates_duplicate_model_names(tmp_path):
    from retrieval_heads.cli import load_runs

    for sub in ("a", "b"):
        info = attention_info(1, 2, name="same-name")
        run_dir = tmp_path / sub / "run"
        run_dir.mkdir(parents=True)
        RetrievalScores(info=info, score=torch.tensor([[0.5, 0.1]]),
                        activation_freq=torch.zeros(1, 2), n_instances=1).save(
            run_dir / "scores_next_step")

    runs = load_runs([tmp_path / "a" / "run", tmp_path / "b" / "run"], "next_step")
    assert len(runs) == 2, f"labels collapsed: {list(runs)}"


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


# --------------------------------------------------------------------------- 6.10-6.13
def test_resolve_model_finds_registry_paths_outside_the_repo_root(tmp_path, monkeypatch):
    """Registry paths are repo-relative, not CWD-relative."""
    from retrieval_heads.cli import resolve_model

    if not (REPO_ROOT / "models" / "Qwen3-0.6B").exists():
        pytest.skip("Qwen3-0.6B checkpoints not downloaded")
    monkeypatch.chdir(tmp_path)
    path, settings = resolve_model("qwen3-0.6b")
    assert Path(path).is_absolute(), path
    assert Path(path).exists(), path
    assert settings["dtype"]


def test_cli_accepts_dtype_for_model_commands():
    from retrieval_heads.cli import build_parser

    args = build_parser().parse_args(["detect", "--model", "m", "--dtype", "bfloat16"])
    assert args.dtype == "bfloat16"
    # compare/figures never load a model, so they do not carry the flag
    assert build_parser().parse_args(["compare", "--runs", "a"]).__dict__.get("dtype") is None


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


def test_text_config_keeps_the_outer_config_when_nested_is_none():
    from types import SimpleNamespace

    from retrieval_heads.models import text_config

    outer = SimpleNamespace(text_config=None, num_hidden_layers=4)
    assert text_config(outer) is outer


# --------------------------------------------------------------------------- provenance
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


def test_resolve_k_rejects_explicitly_non_positive_values():
    info = attention_info(1, 2)
    with pytest.raises(SystemExit, match="positive"):
        resolve_k(_Args(k=[0]), info)
    with pytest.raises(SystemExit, match="positive"):
        resolve_k(_Args(k_frac=[0.0]), info)


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


def test_summarizer_tolerates_short_lists(tmp_path):
    """A partially written artifact must give a partial report, not IndexError."""
    import json

    from scripts.summarize_results import masking_section

    model_dir = tmp_path / "m"
    model_dir.mkdir()
    (model_dir / "masking_curve.json").write_text(json.dumps({
        "model": "m", "k_values": [1, 2, 4], "k_effective": [1],
        "retrieval": [90.0], "retrieval_std": [5.0], "random_mean": [], "random_std": [],
        "n_scoreable_heads": 4, "baseline": 95.0,
    }), encoding="utf-8")

    table = "\n".join(masking_section(tmp_path, ["m"]))
    assert "n/a" in table or "90.0" in table


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


def test_save_json_writes_null_instead_of_nan(tmp_path):
    import json

    from retrieval_heads.utils import save_json

    target = tmp_path / "corr.json"
    save_json({"values": [[1.0, float("nan")]], "jaccard": float("nan")}, target)
    text = target.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text) == {"values": [[1.0, None]], "jaccard": None}


def test_model_info_from_dict_requires_heads_for_every_scoreable_layer():
    from retrieval_heads.models import ModelInfo

    info = attention_info(2, 2)
    broken = info.as_dict()
    del broken["num_heads"]["1"]          # layer 1 is scoreable but has no head count
    with pytest.raises(ValueError, match="num_heads"):
        ModelInfo.from_dict(broken)


def test_k_fraction_above_one_is_clamped_with_a_warning(caplog):
    import logging

    info = attention_info(1, 4)
    with caplog.at_level(logging.WARNING):
        values = resolve_k(_Args(k_frac=[2.0]), info)
    assert values == [4], values
    assert "clamping" in caplog.text


def test_head_masker_rejects_an_out_of_range_head():
    from retrieval_heads.attention import HeadMasker
    from retrieval_heads.utils import HeadRef

    info = attention_info(1, 2)
    with pytest.raises(KeyError, match="out of range"):
        HeadMasker(object(), info, [HeadRef(0, 5)])


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


def test_decode_rejects_a_non_positive_max_new_tokens():
    from retrieval_heads.scoring import decode_with_attention

    with pytest.raises(ValueError, match="max_new_tokens must be positive"):
        decode_with_attention(None, None, torch.zeros(1, 4, dtype=torch.long),
                              max_new_tokens=0)


def test_evaluate_samples_signature_is_a_tripwire():
    """The test fakes accept a wide signature; this pins the real one."""
    import inspect

    from retrieval_heads.masking import evaluate_samples

    params = list(inspect.signature(evaluate_samples).parameters)
    assert params[:4] == ["model", "tokenizer", "info", "samples"]
    for name in ("masked_heads", "masked_layers", "max_new_tokens", "prefill_chunk",
                 "attn_impl"):
        assert name in params, f"{name} disappeared from evaluate_samples"


def test_summarizer_tolerates_a_summary_without_sparsity(tmp_path):
    import json

    from scripts.summarize_results import detection_section

    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "summary_next_step.json").write_text(
        json.dumps({"model": "m", "n_instances": 2, "top_heads": []}), encoding="utf-8")
    table = "\n".join(detection_section(tmp_path, ["m"]))
    assert "n/a" in table


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
            activations={"next_step": {"L0H0": 1.0, "L0H1": 1.0},
                         "same_step": {"L0H0": 1.0, "L0H1": 1.0}},
            considered={}, sink_rate={"next_step": {"__overall__": 0.0}},
            generated_ids=[1], generated_text="x",
            needle_recall=0.9 if recited else 0.0, n_steps=1,
            aligned_scores={"next_step": {"L0H0": 0.9, "L0H1": 0.2},
                            "same_step": {"L0H0": 0.8, "L0H1": 0.1}},
            copied_tokens={"next_step": {"L0H0": [11, 12]}, "same_step": {"L0H0": [11]}},
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


def test_plot_corr_map_accepts_a_json_artifact_with_nulls():
    """`save_json` writes non-finite values as null; the plot must not crash on it."""
    import matplotlib.pyplot as plt

    from retrieval_heads.plotting import plot_corr_map

    corr = {"labels": ["a", "b"], "values": [[1.0, None], [None, 1.0]], "mode": "sorted"}
    fig = plot_corr_map(corr)
    plt.close(fig)


def test_linear_mixer_class_names_are_not_scored_as_attention():
    from torch import nn

    from retrieval_heads.models import discover_modules

    class GatedDeltaNet(nn.Module):        # matches _LINEAR_MARKERS
        def __init__(self):
            super().__init__()
            self.layer_idx = 0
            self.head_dim = 2
            self.num_heads = 1
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                setattr(self, name, nn.Linear(2, 2, bias=False))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.mixer = GatedDeltaNet()

    attention, linear, names = discover_modules(Model())
    assert not attention, "a recurrent mixer was classified as softmax attention"
    assert list(linear) == [0]


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


def test_greedy_generate_derives_eos_by_default(monkeypatch):
    """`greedy_generate(eos=None)` must call the real function, not a stand-in.

    The previous version monkeypatched `masking.greedy_generate` itself, so it
    validated `evaluate_samples` and never touched the default path -- which is how
    a `NameError` on `tokenizer` survived it.
    """
    import retrieval_heads.masking as masking

    seen = {}

    def fake_ids(model, input_ids, *, max_new_tokens, eos, attn_impl="sdpa", prefill_chunk=None):
        seen["eos"] = set(eos)
        return [1]

    monkeypatch.setattr(masking, "greedy_ids", fake_ids)
    monkeypatch.setattr(masking, "eos_ids", lambda model, tokenizer=None: {7})

    class Tok:
        eos_token_id = None

    assert masking.greedy_generate(None, torch.zeros(1, 2, dtype=torch.long),
                                   max_new_tokens=2, tokenizer=Tok()) == [1]
    assert seen["eos"] == {7}, "the default did not derive EOS from the model"


def test_eos_ids_accepts_numpy_and_torch_integers():
    from types import SimpleNamespace

    import numpy as np

    from retrieval_heads.utils import eos_ids

    model = SimpleNamespace(
        config=SimpleNamespace(eos_token_id=np.int64(11)),
        generation_config=SimpleNamespace(eos_token_id=torch.tensor(12)),
    )
    assert eos_ids(model) == {11, 12}


def test_require_matching_scores_checks_head_dim():
    from retrieval_heads.cli import require_matching_scores

    info = attention_info(1, 2)
    scores = RetrievalScores(info=info, score=torch.zeros(1, 2),
                             activation_freq=torch.zeros(1, 2), n_instances=1)
    other = attention_info(1, 2)
    other.head_dim = info.head_dim + 8
    with pytest.raises(SystemExit, match="head_dim"):
        require_matching_scores(scores, other)


def test_mixer_ablation_reports_both_stack_fractions():
    from retrieval_heads.masking import MixerAblation

    ablation = MixerAblation(full_attention=[1.0], linear_attention=[1.0], baseline=90.0,
                             k_values=[6], n_full_layers=6, n_linear_layers=18)
    assert ablation.fractions == [1.0]
    assert ablation.fractions_linear == [pytest.approx(1 / 3)]


def test_grid_correlation_across_layouts_is_nan():
    from retrieval_heads.properties import correlate

    a = RetrievalScores(info=attention_info(2, 2), score=torch.rand(2, 2),
                        activation_freq=torch.zeros(2, 2), n_instances=1)
    b = RetrievalScores(info=attention_info(3, 1), score=torch.rand(3, 1),
                        activation_freq=torch.zeros(3, 1), n_instances=1)
    assert correlate(a, b, mode="grid") != correlate(a, b, mode="grid")   # NaN


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


def test_score_instance_always_records_both_pairings():
    """The removed `compute_second_pairing` flag could strip a key summary() reads."""
    import inspect

    from retrieval_heads.scoring import score_instance

    assert "compute_second_pairing" not in inspect.signature(score_instance).parameters


def test_head_overlap_artifact_carries_its_mode():
    from retrieval_heads.properties import head_overlap

    a = RetrievalScores(info=attention_info(1, 2), score=torch.tensor([[0.9, 0.0]]),
                        activation_freq=torch.zeros(1, 2), n_instances=1)
    b = RetrievalScores(info=attention_info(1, 2), score=torch.tensor([[0.8, 0.0]]),
                        activation_freq=torch.zeros(1, 2), n_instances=1)
    payload = head_overlap(a, b, mode="grid").as_dict()
    assert payload["mode"] == "grid", payload


def _well_formed_curve() -> dict:
    return {"m": {"k_values": [1, 2], "k_effective": [1, 2], "retrieval": [90.0, 80.0],
                  "retrieval_std": [3.0, 4.0], "random_mean": [95.0, 94.0],
                  "random_std": [1.0, 1.5], "baseline": 97.0,
                  "retrieval_exact_match": [80.0, 70.0], "retrieval_exact_std": [5.0, 6.0],
                  "random_exact_match_mean": [90.0, 89.0], "retrieval_recall": [70.0, 60.0],
                  "random_recall_mean": [88.0, 87.0], "baseline_recall": 92.0}}


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


def test_every_ablation_command_accepts_system_prompt():
    """detect/mask/qa/cot must expose the flag, or mask silently measures None."""
    from retrieval_heads.cli import build_parser

    parser = build_parser()
    for command in ("detect", "mask", "qa", "cot"):
        args = parser.parse_args([command, "--model", "m", "--system-prompt", "be terse"])
        assert args.system_prompt == "be terse", command


def test_ablation_reuses_every_condition_detect_recorded(caplog):
    """The whole recorded condition set, not just the system prompt.

    `detect --no-chat-template` followed by a default `mask` used to measure a
    different task, and a custom corpus was silently replaced by synthetic filler.
    """
    import logging
    from types import SimpleNamespace

    from retrieval_heads.cli import resolve_detection_settings

    scores = SimpleNamespace(meta={
        "config": {"system_prompt": "recorded", "chat_template": False,
                   "enable_thinking": None, "threshold": 0.2},
        "corpus_path": "essays.txt",
        "argmax_domain": "full",
    })
    defaults = SimpleNamespace(system_prompt=None, no_chat_template=False, thinking=False,
                              argmax_domain="prompt", threshold=0.1, corpus=None)

    settings = resolve_detection_settings(defaults, scores)
    assert settings.system_prompt == "recorded"
    assert settings.chat_template is False          # detect ran without a template
    assert settings.enable_thinking is None
    assert settings.threshold == 0.2
    assert settings.argmax_domain == "full"
    assert settings.corpus_path == "essays.txt"

    # An explicit disagreement is loud and wins.
    explicit = SimpleNamespace(system_prompt=None, no_chat_template=True, thinking=True,
                               argmax_domain="prompt", threshold=0.1, corpus="other.txt")
    with caplog.at_level(logging.WARNING):
        other = resolve_detection_settings(explicit, scores)
    assert other.chat_template is False and other.corpus_path == "other.txt"
    assert "differs from the value detect recorded" in caplog.text


def test_resolve_k_rejects_a_mixed_negative_list():
    from retrieval_heads.cli import resolve_k

    info = attention_info(1, 4)
    with pytest.raises(SystemExit, match="positive throughout"):
        resolve_k(_Args(k=[-1, 5]), info)


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


# --------------------------------------------------------------------------- review round 4
def test_chat_helper_respects_no_chat_template():
    from retrieval_heads.downstream import _chat

    class Tok:
        def apply_chat_template(self, messages, **kwargs):
            return "TEMPLATED"

    assert _chat(Tok(), "hello", enable_thinking=False) == "TEMPLATED"
    assert _chat(Tok(), "hello", enable_thinking=False, chat_template=False) == "hello"


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
        needle_text="n", question="q", depth=0.5, target_tokens=3, haystack_tokens=3,
        seed=0, meta={"needle_text": "clobbered", "seed": 999},
    )
    payload = sample.as_dict()
    assert payload["needle_text"] == "n"
    assert payload["seed"] == 0


# --------------------------------------------------------------------------- lengths
def test_realized_context_length_tracks_the_request(tokenizer):
    """The needle, question and chat template are added on top of the filler.

    Uncorrected that overhead made a requested 1024 come out at 1088 (+6%).  The
    builder now measures the rendered prompt and re-budgets, so the realized
    length is within 2% at both ends of the range.
    """
    builder = HaystackBuilder(seed=0)
    needle = "The best thing to do in San Francisco is to eat a sandwich in Dolores Park."
    question = "What is the best thing to do in San Francisco?"
    # depth matters: the cut is snapped to whitespace, so the extreme depths take a
    # different path through the re-budget loop.
    for depth in (0.0, 0.5, 1.0):
        for target in (1024, 4096):
            sample = build_needle_sample(
                tokenizer, needle=needle, question=question, target_tokens=target,
                depth=depth, builder=builder, seed=1234,
            )
            error = abs(sample.haystack_tokens - target) / target
            assert error <= 0.02, (
                f"depth {depth}: target {target} realized {sample.haystack_tokens} "
                f"({error:.1%})"
            )
            assert 0 <= sample.needle_span[0] < sample.needle_span[1] <= sample.haystack_tokens


def test_summarizer_load_warns_on_an_unstamped_artifact(tmp_path, caplog, monkeypatch):
    """The stale-schema guard used to raise TypeError into a bare except."""
    import json
    import logging

    from scripts.summarize_results import load

    target = tmp_path / "summary_next_step.json"
    target.write_text(json.dumps({"model": "old", "n_instances": 1}), encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        payload = load(target)
    assert payload is not None
    assert "schema_version" in caplog.text, "the stale guard stayed silent"


def test_head_masker_installs_nothing_when_a_later_layer_fails():
    from torch import nn

    from retrieval_heads.attention import HeadMasker
    from retrieval_heads.models import ModelInfo
    from retrieval_heads.utils import HeadRef

    class Good(nn.Module):
        def __init__(self):
            super().__init__()
            self.head_dim = 2
            self.num_heads = 1
            self.o_proj = nn.Linear(2, 2, bias=False)

    class Bad(nn.Module):
        """`_head_dim` cannot work here: no head_dim, no num_heads, no q_proj."""

        def __init__(self):
            super().__init__()
            self.o_proj = nn.Linear(2, 2, bias=False)

    good, bad = Good(), Bad()
    info = ModelInfo(
        name="toy", path="toy", model_type="toy", num_layers=2,
        layer_types=["full_attention", "full_attention"], num_heads={0: 1, 1: 1},
        num_kv_heads={0: 1, 1: 1}, head_dim=2, hidden_size=2, max_position_embeddings=64,
        attention_modules={0: good, 1: bad}, scoreable_layers_=[0, 1],
    )
    with pytest.raises(AttributeError):
        HeadMasker(object(), info, [HeadRef(0, 0), HeadRef(1, 0)])
    assert not good.o_proj._forward_pre_hooks, "layer 0's hook leaked from a failed install"


def test_token_mixer_masker_rejects_a_foreign_model():
    from torch import nn

    from retrieval_heads.attention import TokenMixerMasker
    from retrieval_heads.models import ModelInfo

    class Mixer(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_idx = 0

    module = Mixer()
    info = ModelInfo(
        name="toy", path="toy", model_type="toy", num_layers=1,
        layer_types=["linear_attention"], num_heads={}, num_kv_heads={},
        head_dim=0, hidden_size=1, max_position_embeddings=64,
        linear_modules={0: module}, linear_layers_=[0],
    )
    other = nn.Linear(1, 1)
    with pytest.raises(KeyError, match="does not belong"):
        TokenMixerMasker(other, info, [0])


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


def test_render_chat_falls_back_on_a_non_typeerror():
    from retrieval_heads.haystack import render_chat

    class PickyTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            if "enable_thinking" in kwargs:
                raise ValueError("unexpected keyword argument 'enable_thinking'")
            return "ok"

    assert render_chat(PickyTokenizer(), [{"role": "user", "content": "x"}],
                       enable_thinking=False) == "ok"


def test_detect_rejects_an_empty_grid_before_loading_a_model():
    from retrieval_heads.cli import main

    with pytest.raises(SystemExit, match="empty detection grid"):
        main(["detect", "--model", "not-a-real-model", "--depths", "0"])
