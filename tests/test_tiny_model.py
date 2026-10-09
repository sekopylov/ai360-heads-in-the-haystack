"""Fast tests against a tiny locally-built hybrid model -- no checkpoint download.

The architecture-specific behaviour (discovery on a real hybrid config, both
attention-capture paths, masking a real ``o_proj``, chunked-prefill equivalence)
otherwise lives only in the `integration` suite, which CI never runs.  A four-layer
Qwen3.5-style model with ~50k parameters closes that gap in a couple of seconds.
"""

from __future__ import annotations

import shutil

import pytest
import torch

from retrieval_heads.attention import HeadMasker, masked_heads
from retrieval_heads.generation import greedy_ids, prefill_cache
from retrieval_heads.scoring import decode_with_attention
from retrieval_heads.utils import HeadRef
from tests.conftest import MODEL_DIRS


def _ids(rows: int = 1, length: int = 24, vocab: int = 64) -> torch.LongTensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randint(1, vocab, (rows, length), generator=generator)


def test_discovery_sees_a_real_hybrid_layout(tiny_hybrid):
    model, info = tiny_hybrid
    assert info.scoreable_layers == [2]
    assert info.linear_layers == [0, 1, 3]
    assert info.is_hybrid
    assert info.num_heads == {2: 4}
    assert info.head_dim == 8
    assert info.model_class == "Qwen3_5ForCausalLM"
    assert info.n_all_heads == 4 + 3 * 4


def test_both_capture_methods_agree_on_the_tiny_model(tiny_hybrid):
    model, info = tiny_hybrid
    ids = _ids()

    def capture(method: str):
        trace, _ = decode_with_attention(model, info, ids, max_new_tokens=3,
                                         capture_method=method, stop_on_eos=False)
        return trace

    public = capture("output_attentions")
    patched = capture("patch")

    assert len(public.steps) == len(patched.steps)
    for a, b in zip(public.steps, patched.steps):
        assert a.positions()[2].tolist() == b.positions()[2].tolist()


def test_head_masker_changes_logits_on_a_real_oproj_and_restores(tiny_hybrid):
    model, info = tiny_hybrid
    ids = _ids()

    base = model(input_ids=ids).logits
    with masked_heads(model, info, [HeadRef(2, 0)]):
        masked = model(input_ids=ids).logits
        assert not torch.allclose(base, masked), "masking head L2H0 changed nothing"
    assert torch.equal(model(input_ids=ids).logits, base), "hooks were not removed"


def test_head_masker_rejects_a_module_from_another_model(tiny_hybrid):
    model, info = tiny_hybrid
    from retrieval_heads.models import build_model_info

    other = type(model)(model.config)
    other.eval()
    # `other` has the same architecture but different modules; masking through the
    # first model's info must not silently touch the wrong tensors.
    try:
        HeadMasker(other, info, [HeadRef(2, 0)])
    except KeyError as exc:
        assert "does not belong" in str(exc)
    else:  # pragma: no cover - the guard must fire
        raise AssertionError("HeadMasker accepted modules from another model instance")


def test_chunked_prefill_matches_one_shot_on_the_tiny_model(tiny_hybrid):
    model, info = tiny_hybrid
    ids = _ids(length=40)

    full_cache, full_logits = prefill_cache(model, ids)
    chunk_cache, chunk_logits = prefill_cache(model, ids, prefill_chunk=8)

    assert torch.allclose(full_logits, chunk_logits, atol=1e-4), \
        (full_logits - chunk_logits).abs().max().item()
    assert greedy_ids(model, ids, max_new_tokens=4, eos=set()) == \
        greedy_ids(model, ids, max_new_tokens=4, eos=set(), prefill_chunk=8)


def test_truncated_run_scopes_the_last_step_to_same_step(tiny_hybrid):
    model, info = tiny_hybrid
    trace, generated = decode_with_attention(model, info, _ids(), max_new_tokens=3,
                                             stop_on_eos=False)
    assert len(generated) == 3
    assert trace.stopped_on_eos is False
    scopes = [step.applies_to for step in trace.steps]
    assert scopes[0] == ("next_step",), "the prefill row must stay with next_step"
    assert scopes[-1] == ("same_step",), "the never-emitted prediction must be dropped"
    assert trace.kv_len == 24 + 3


def test_load_model_records_class_and_dtype(tiny_hybrid, tmp_path):
    """End-to-end load: the class that actually ran and the dtype are recorded."""
    from retrieval_heads.models import load_model

    model, _info = tiny_hybrid
    tokenizer_dir = MODEL_DIRS["qwen3-0.6b"]
    if not (tokenizer_dir / "tokenizer.json").exists():
        pytest.skip("Qwen3-0.6B tokenizer files are not available")

    model.save_pretrained(tmp_path)
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                 "generation_config.json"):
        source = tokenizer_dir / name
        if source.exists():
            shutil.copy(source, tmp_path / name)

    _model, _tokenizer, info = load_model(str(tmp_path), dtype="float32",
                                          attn_implementation="eager")
    assert info.model_class == "Qwen3_5ForCausalLM"
    assert info.dtype == "float32"
    assert info.scoreable_layers == [2]


def test_zeroing_the_attention_row_equals_zeroing_the_oproj_slice(tiny_hybrid):
    """The masking shortcut must be exact head pruning, not an approximation.

    Same check as the integration test on Qwen3.5, but on the tiny local hybrid so
    the fast CI suite covers it too.
    """
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q35

    from retrieval_heads.attention import restore_attn_implementation, set_attn_implementation

    model, info = tiny_hybrid
    ids = _ids(length=16)
    head = HeadRef(2, 1)
    previous = set_attn_implementation(model, "eager")
    try:
        base = model(input_ids=ids).logits
        with HeadMasker(model, info, [head]):
            via_oproj = model(input_ids=ids).logits

        original = q35.eager_attention_forward

        def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
            out, weights = original(module, query, key, value, attention_mask,
                                    scaling, dropout, **kwargs)
            if getattr(module, "layer_idx", None) == head.layer:
                weights = weights.clone()
                weights[:, head.head] = 0.0
                value_states = q35.repeat_kv(value, module.num_key_value_groups)
                out = torch.matmul(weights, value_states).transpose(1, 2).contiguous()
            return out, weights

        q35.eager_attention_forward = patched
        try:
            via_row = model(input_ids=ids).logits
        finally:
            q35.eager_attention_forward = original
    finally:
        restore_attn_implementation(model, previous)

    assert not torch.allclose(base, via_oproj), "the head mask changed nothing"
    assert torch.allclose(via_oproj, via_row, atol=1e-5), \
        (via_oproj - via_row).abs().max().item()


def test_eager_capture_and_sdpa_prefill_agree_on_positions(tiny_hybrid):
    """detect captures under eager but prefills under sdpa; the argmax must agree."""
    model, info = tiny_hybrid
    ids = _ids(length=24)

    def positions(prefill_impl: str):
        trace, _ = decode_with_attention(model, info, ids, max_new_tokens=3,
                                         capture_method="patch",
                                         prefill_impl=prefill_impl, stop_on_eos=False)
        return [step.positions()[2].tolist() for step in trace.steps]

    assert positions("eager") == positions("sdpa")


def test_prefill_and_capture_really_use_different_kernels(tiny_hybrid, monkeypatch):
    """Comparing positions is not enough: a no-op switch would agree too.

    Two things are checked, because they are two different claims:
    * `transformers` reads `config._attn_implementation` at forward time, so a spy
      records which implementation was in effect for the multi-token prefill body
      versus the single-token capture steps;
    * the capture-by-monkeypatch path only works while `"eager"` is *not* a
      registered key in `ALL_ATTENTION_FUNCTIONS`: the call site passes
      `eager_attention_forward` as the default, so an unregistered name resolves to
      that module-global -- which is what `AttentionRecorder` patches.  If upstream
      ever registers `"eager"`, this fails loudly instead of the recorder silently
      capturing nothing.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    sentinel = lambda *args, **kwargs: None  # noqa: E731 - identity only
    assert ALL_ATTENTION_FUNCTIONS.get_interface("eager", sentinel) is sentinel, (
        "`eager` is now a registered kernel, so the monkeypatch capture no longer "
        "intercepts the call"
    )
    assert ALL_ATTENTION_FUNCTIONS.get_interface("sdpa", sentinel) is not sentinel, (
        "`sdpa` must resolve to a real kernel for the prefill/capture split to differ"
    )

    model, info = tiny_hybrid
    ids = _ids(length=24)
    seen: list[tuple[int, str]] = []
    original = model.forward

    def spy(*args, **kwargs):
        seen.append((int(kwargs["input_ids"].shape[1]), model.config._attn_implementation))
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "forward", spy)
    decode_with_attention(model, info, ids, max_new_tokens=2, capture_method="patch",
                          prefill_impl="sdpa", capture_impl="eager", stop_on_eos=False)

    body = [impl for length, impl in seen if length > 1]
    steps = [impl for length, impl in seen if length == 1]
    assert body, seen
    assert all(impl == "sdpa" for impl in body), seen
    assert steps and all(impl == "eager" for impl in steps), seen


def test_argmax_domain_is_recorded_and_positions_stay_in_the_prompt(tiny_hybrid):
    model, info = tiny_hybrid
    ids = _ids(length=24)
    for domain in ("prompt", "full"):
        trace, _ = decode_with_attention(model, info, ids, max_new_tokens=3,
                                         argmax_domain=domain, stop_on_eos=False)
        assert trace.argmax_domain == domain
        if domain == "prompt":
            for step in trace.steps:
                for positions in step.positions().values():
                    assert int(positions.max()) < ids.shape[1]


def test_haystack_domain_runs_end_to_end_on_the_tiny_hybrid(tiny_hybrid):
    """The third domain must reach the capture and stay inside its span."""
    from retrieval_heads.haystack import NeedleSample
    from retrieval_heads.scoring import score_instance

    model, info = tiny_hybrid
    ids = _ids(length=24)          # ids in [1, 64): inside the tiny vocab
    sample = NeedleSample(
        prompt_text="", input_ids=ids, needle_span=(8, 12), needle_text="n",
        question="q", depth=0.5, target_tokens=24, prompt_tokens=24, seed=0,
        haystack_span=(3, 20), meta={"needle_text_ids": ids[0, 8:12].tolist()},
    )
    start, end = sample.haystack_span
    trace, _ = decode_with_attention(model, info, sample.input_ids, max_new_tokens=2,
                                     argmax_domain="haystack", argmax_span=sample.haystack_span,
                                     stop_on_eos=False)
    assert trace.argmax_domain == "haystack" and trace.argmax_span == (start, end)
    for step in trace.steps:
        for positions in step.positions().values():
            assert start <= int(positions.min()) and int(positions.max()) < end

    # `score_instance` derives the span from the sample, so a `haystack` run needs
    # no span argument -- and a sample without one is refused rather than silently
    # scored in the prompt domain.
    result = score_instance(model, info, sample, max_new_tokens=2, argmax_domain="haystack")
    assert result.meta["argmax_domain"] == "haystack"
    assert result.meta["argmax_span"] == [start, end]
    assert result.sample["haystack_span"] == [start, end]

    sample.haystack_span = None
    with pytest.raises(ValueError, match="no haystack_span"):
        score_instance(model, info, sample, max_new_tokens=2, argmax_domain="haystack")


def test_one_pass_captures_every_argmax_domain(tiny_hybrid):
    """Every domain comes from the same forward pass, so scoring all of them is free.

    The capture must hold an argmax per domain, each inside its own position set, and
    `score_instance` must record the alternatives beside the primary matrix instead of
    requiring a second run.
    """
    from retrieval_heads.haystack import NeedleSample
    from retrieval_heads.scoring import score_instance

    model, info = tiny_hybrid
    ids = _ids(length=24)
    sample = NeedleSample(
        prompt_text="", input_ids=ids, needle_span=(8, 12), needle_text="n",
        question="q", depth=0.5, target_tokens=24, prompt_tokens=24, seed=0,
        haystack_span=(3, 20), meta={"needle_text_ids": ids[0, 8:12].tolist()},
    )
    start, end = sample.haystack_span
    trace, _ = decode_with_attention(model, info, sample.input_ids, max_new_tokens=2,
                                     argmax_domain="haystack",
                                     argmax_span=sample.haystack_span, stop_on_eos=False)
    assert trace.domains == ("prompt", "full", "haystack")
    for step in trace.steps:
        assert set(step.argmax_by_domain) == {"prompt", "full", "haystack"}
        assert step.positions("haystack") is step.argmax_by_domain["haystack"]
        for positions in step.positions("haystack").values():
            assert start <= int(positions.min()) and int(positions.max()) < end
        for positions in step.positions("prompt").values():
            assert 0 <= int(positions.min()) and int(positions.max()) < ids.shape[1]
        # `positions()` with no argument is the scoring domain.
        assert step.positions() is step.argmax_by_domain["haystack"]

    result = score_instance(model, info, sample, max_new_tokens=2, argmax_domain="haystack")
    # The primary is `scores`; the alternatives are the complement, so nothing is
    # duplicated in the JSONL.
    assert set(result.scores_by_domain) == {"prompt", "full"}
    assert result.meta["argmax_domains"] == ["prompt", "full", "haystack"]
    for domain, per_pairing in result.scores_by_domain.items():
        assert set(per_pairing) == {"next_step", "same_step"}, domain
        assert set(per_pairing["next_step"]) == {str(h) for h in info.scoreable_heads}
    # A head's credit can only grow when the argmax searches a larger set: the prompt
    # domain contains the haystack span, so every haystack credit is also a prompt one.
    for head in info.scoreable_heads:
        key = str(head)
        assert (result.scores["next_step"][key]
                <= result.scores_by_domain["prompt"]["next_step"][key] + 1e-9), head


def _tiny_detect_tree(tiny_hybrid, tmp_path, monkeypatch):
    """A real `detect` run on the tiny hybrid, with only the prompt builder stubbed.

    The tiny model has no tokenizer, so `build_needle_sample` is replaced by a
    hand-built `NeedleSample`; everything downstream (forward pass, capture, credits,
    per-domain aggregation, artifact writing) is the real code.
    """
    from retrieval_heads.detection import DetectionConfig, run_detection
    from retrieval_heads.haystack import NeedleSample

    model, info = tiny_hybrid
    ids = _ids(length=24)

    def fake_build(tokenizer, **kwargs):
        return NeedleSample(
            prompt_text="", input_ids=ids, needle_span=(8, 12), needle_text="n",
            question="q", depth=kwargs["depth"], target_tokens=24, prompt_tokens=24,
            seed=kwargs["seed"], haystack_span=(3, 20),
            meta={"needle_text_ids": ids[0, 8:12].tolist(),
                  "haystack_span_verbatim": True},
        )

    monkeypatch.setattr("retrieval_heads.detection.build_needle_sample", fake_build)
    config = DetectionConfig(lengths=[24], depths_per_length=2, needles=[("n", "q")],
                             max_new_tokens=2)
    return run_detection(model, None, info, config, out_dir=tmp_path, progress=False), info


def test_detect_writes_a_matrix_per_argmax_domain(tiny_hybrid, tmp_path, monkeypatch):
    """End to end through `run_detection`, not through a fake scorer.

    The per-domain outputs are the round-eight change: every captured domain must be
    aggregated and written from *real* scores, the primary must stay the run's own
    domain, and the JSONL must carry the complement.
    """
    import json

    run, info = _tiny_detect_tree(tiny_hybrid, tmp_path, monkeypatch)

    summary = json.loads((tmp_path / "summary_next_step.json").read_text(encoding="utf-8"))
    assert summary["argmax_domain"] == "haystack"
    assert sorted(summary["argmax_domains_captured"]) == ["full", "haystack", "prompt"]
    assert set(summary["sparsity_by_domain"]) == {"haystack", "prompt", "full"}
    # The primary matrix is the run's own domain, and the sidecars exist beside it.
    assert (tmp_path / "scores_next_step.npz").exists()
    assert (tmp_path / "scores_next_step_prompt.npz").exists()
    assert (tmp_path / "scores_next_step_full.npz").exists()
    assert run.domains[("next_step", "prompt")].meta["argmax_domain"] == "prompt"
    # The one-way domain bound (prompt credit survives under `haystack`, not the
    # reverse) is a theorem, and it is asserted on quantised synthetic rows in
    # `test_scoring.py::test_prompt_credit_survives_the_haystack_domain`.  It is not
    # asserted here on purpose: this tiny model earns no credit at all, so *any*
    # direction would pass vacuously.  This assertion is what keeps that honest.
    assert float(torch.nan_to_num(run.scores.score).max()) == 0.0
    # The per-instance JSONL carries the alternative domains (the complement of the
    # primary), so a reader can re-derive a domain without a re-run.
    line = json.loads((tmp_path / "instances_next_step.jsonl").read_text(
        encoding="utf-8").splitlines()[0])
    assert set(line["scores_by_domain"]) == {"prompt", "full"}
    assert set(line["argmax_domain_shift"]) == {"next_step", "same_step"}


def test_figures_stage_reads_a_schema_8_tree(tiny_hybrid, tmp_path, monkeypatch):
    """The `figures` stage must consume what the new `detect` writes.

    The per-domain sidecars changed the artifact set, and no test ran the figure stage
    over it: `load_runs` picks `scores_<pairing>` and the plotters read the summary's
    new fields, so a mismatch would surface only on the GPU job's last stage.
    """
    from retrieval_heads.cli import main

    _tiny_detect_tree(tiny_hybrid, tmp_path, monkeypatch)
    assert main(["figures", "--runs", str(tmp_path), "--out", str(tmp_path / "figures")]) == 0

    figures = sorted(p.name for p in (tmp_path / "figures").glob("*.pdf"))
    assert "ring_graph.pdf" in figures, figures
    assert "heat_map.pdf" in figures, figures
