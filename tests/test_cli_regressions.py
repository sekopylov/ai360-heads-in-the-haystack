"""Regression tests: cli (split out of test_regressions.py).

Each test pins a specific failure mode the old code had, so the fix cannot
quietly regress.  They are all fast (no checkpoints).
"""

from __future__ import annotations

from pathlib import Path
import pytest
import torch
from retrieval_heads.cli import normalize_prefill_chunk, resolve_k
from retrieval_heads.scoring import RetrievalScores
from tests.conftest import REPO_ROOT

from tests._helpers import (  # noqa: F401
    _Args, _well_formed_curve, attention_info, scores_with,
)


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


def test_missing_scores_artifact_names_the_detect_command(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run `detect`"):
        RetrievalScores.load(tmp_path / "does-not-exist")


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


def test_require_matching_scores_separates_a_base_model_from_its_variant():
    """A base checkpoint and its chat/fine-tune share layers, heads and head_dim.

    That is exactly the Sec. 4.3 "intrinsic" comparison, and without hidden_size /
    model_class the ablation would silently mask heads chosen on the other one.
    """
    import dataclasses

    from retrieval_heads.cli import require_matching_scores

    base_info = attention_info(2, 2)
    loaded = dataclasses.replace(base_info, name="chat", hidden_size=base_info.hidden_size * 2)
    matching = RetrievalScores(info=base_info, score=torch.zeros(2, 2),
                               activation_freq=torch.zeros(2, 2), n_instances=1)
    with pytest.raises(SystemExit, match="hidden_size"):
        require_matching_scores(matching, loaded)

    variant = dataclasses.replace(base_info, name="base", model_class="OtherForCausalLM")
    matching = RetrievalScores(info=variant, score=torch.zeros(2, 2),
                               activation_freq=torch.zeros(2, 2), n_instances=1)
    with pytest.raises(SystemExit, match="model_class"):
        require_matching_scores(matching, loaded)


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


def test_resolve_k_rejects_explicitly_non_positive_values():
    info = attention_info(1, 2)
    with pytest.raises(SystemExit, match="positive"):
        resolve_k(_Args(k=[0]), info)
    with pytest.raises(SystemExit, match="positive"):
        resolve_k(_Args(k_frac=[0.0]), info)


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


def test_k_fraction_above_one_is_clamped_with_a_warning(caplog):
    import logging

    info = attention_info(1, 4)
    with caplog.at_level(logging.WARNING):
        values = resolve_k(_Args(k_frac=[2.0]), info)
    assert values == [4], values
    assert "clamping" in caplog.text


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


def test_require_matching_scores_checks_head_dim():
    from retrieval_heads.cli import require_matching_scores

    info = attention_info(1, 2)
    scores = RetrievalScores(info=info, score=torch.zeros(1, 2),
                             activation_freq=torch.zeros(1, 2), n_instances=1)
    other = attention_info(1, 2)
    other.head_dim = info.head_dim + 8
    with pytest.raises(SystemExit, match="head_dim"):
        require_matching_scores(scores, other)


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

    from retrieval_heads.cli import DEFAULT_ARGMAX_DOMAIN, resolve_detection_settings

    scores = SimpleNamespace(meta={
        "config": {"system_prompt": "recorded", "chat_template": False,
                   "enable_thinking": None, "threshold": 0.2},
        "corpus_path": "essays.txt",
        "argmax_domain": "full",
    })
    # `DEFAULT_ARGMAX_DOMAIN`, not a literal: this test is about "the flag was left at
    # its default", and a literal silently becomes an *explicit* request when the
    # default moves (which is exactly what happened when `haystack` became the default).
    defaults = SimpleNamespace(system_prompt=None, no_chat_template=False, thinking=False,
                               argmax_domain=DEFAULT_ARGMAX_DOMAIN, threshold=0.1,
                               corpus=None)

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


def test_detect_rejects_an_empty_grid_before_loading_a_model():
    from retrieval_heads.cli import main

    with pytest.raises(SystemExit, match="empty detection grid"):
        main(["detect", "--model", "not-a-real-model", "--depths", "0"])



def test_collapsing_k_fractions_warn_and_stay_visible(caplog):
    """Two fractions can resolve to the same K, and that must not be silent.

    On the hybrid's 48 heads `--k-frac 0.01 0.02` both give K=1, so a six-point curve
    has five points there and six on the dense model -- with the same flag.  The log
    says so, and `cmd_mask` records the requested values beside the resolved ones, so
    the artifact cannot imply that all six ran.
    """
    import logging

    info = attention_info(6, 8)          # 48 scoreable heads, like Qwen3.5-0.8B
    assert info.n_scoreable_heads == 48
    with caplog.at_level(logging.WARNING):
        values = resolve_k(_Args(k_frac=[0.01, 0.02, 0.04, 0.08, 0.17, 0.33]), info)
    assert values == [1, 2, 4, 8, 16], values
    assert "collapsed to 5 distinct K" in caplog.text, caplog.text

    # The dense model has no collision, and the warning must not fire there.
    caplog.clear()
    dense = attention_info(28, 16)
    with caplog.at_level(logging.WARNING):
        dense_values = resolve_k(_Args(k_frac=[0.01, 0.02, 0.04, 0.08, 0.17, 0.33]), dense)
    assert len(dense_values) == 6, dense_values
    assert "collapsed" not in caplog.text, caplog.text
