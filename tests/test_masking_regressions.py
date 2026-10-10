"""Regression tests: masking (split out of test_regressions.py).

Each test pins a specific failure mode the old code had, so the fix cannot
quietly regress.  They are all fast (no checkpoints).
"""

from __future__ import annotations

import pytest
import torch
import retrieval_heads.masking as masking
from retrieval_heads.haystack import HaystackBuilder, build_needle_sample
from retrieval_heads.masking import (
    NiahMetrics,
    control_pool,
    matched_k,
    token_mixer_ablation,
)
from retrieval_heads.models import ModelInfo, require_scoreable
from retrieval_heads.scoring import RetrievalScores

from tests._helpers import (  # noqa: F401
    _Args, _well_formed_curve, attention_info, scores_with,
)


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


def test_require_scoreable_rejects_a_model_with_no_attention():
    info = ModelInfo(
        name="all-linear", path="x", model_type="toy", num_layers=2,
        layer_types=["linear_attention"] * 2, num_heads={}, num_kv_heads={},
        head_dim=0, hidden_size=8, max_position_embeddings=128,
        scoreable_layers_=[], linear_layers_=[0, 1],
    )
    with pytest.raises(RuntimeError, match="no scoreable softmax-attention layers"):
        require_scoreable(info)


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


def test_head_masker_rejects_an_out_of_range_head():
    from retrieval_heads.attention import HeadMasker
    from retrieval_heads.utils import HeadRef

    info = attention_info(1, 2)
    with pytest.raises(KeyError, match="out of range"):
        HeadMasker(object(), info, [HeadRef(0, 5)])


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


def test_mixer_ablation_reports_both_stack_fractions():
    from retrieval_heads.masking import MixerAblation

    ablation = MixerAblation(full_attention=[1.0], linear_attention=[1.0], baseline=90.0,
                             k_values=[6], n_full_layers=6, n_linear_layers=18)
    assert ablation.fractions == [1.0]
    assert ablation.fractions_linear == [pytest.approx(1 / 3)]


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
            error = abs(sample.prompt_tokens - target) / target
            assert error <= 0.02, (
                f"depth {depth}: target {target} realized {sample.prompt_tokens} "
                f"({error:.1%})"
            )
            assert 0 <= sample.needle_span[0] < sample.needle_span[1] <= sample.prompt_tokens


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



def _readme_masking_rows() -> dict[tuple[str, int], str]:
    """The README's masking table as ``{(model, K label): the whole row}``.

    Anchoring on the row is the point: a substring search over the whole file catches
    a *stale* table but not a *shuffled* one -- if K=18's numbers end up in the K=36
    row, every substring is still present and the check passes while the table lies.
    The K column is the only one whose third cell looks like ``9 (2%)``, so it selects
    this table out of the README's dozen; a continuation row (empty model cell)
    belongs to the model named above it.
    """
    import re
    from pathlib import Path

    rows: dict[tuple[str, int], str] = {}
    model = None
    for line in (Path(__file__).resolve().parent.parent / "README.md").read_text(
            encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 5 or all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        if cells[0]:
            model = cells[0]
        match = re.match(r"(\d+) \(", cells[2])
        if model is not None and match:
            rows[(model, int(match.group(1)))] = line.replace("**", "")
    return rows


def test_the_readme_masking_table_matches_the_committed_artifacts():
    """Pin the README's masking table to the tree it claims to describe.

    It is one of the two headline results, and it was never checked against
    `ds-results/` -- so a stale table would have shipped.  (A reviewer read the
    top-level `random_trials` field, which is per-trial *f1*, as the table's per-trial
    *recall* column and reported every random cell as wrong; the table was right and
    the field name is the trap, which `MaskingCurve.random_trials` now says in prose.)
    The random cells are checked against `per_sample[k]["random"]`, the only place the
    per-trial recall and exact-match series exist.

    Every number is looked up **inside its own row**, so a transposed or re-attributed
    table fails here instead of looking plausible.
    """
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    rows = _readme_masking_rows()
    assert rows, "the masking table was not found in README.md"
    # The README labels the hybrid's last point by its *effective* K (15), because the
    # random arm caps both arms at the 15-head pool; `k_values` still says 16.
    cases = [("qwen3-0.6b", "Qwen3-0.6B", 0), ("qwen3-0.6b", "Qwen3-0.6B", 1),
             ("qwen3-0.6b", "Qwen3-0.6B", 2), ("qwen3.5-0.8b", "Qwen3.5-0.8B", 0),
             ("qwen3.5-0.8b", "Qwen3.5-0.8B", 3), ("qwen3.5-0.8b", "Qwen3.5-0.8B", 4)]
    for model, display, i in cases:
        curve = json.loads((root / "ds-results" / model / "masking_curve.json").read_text(
            encoding="utf-8"))
        k, label = curve["k_values"][i], curve["k_effective"][i]
        row = rows.get((display, label))
        assert row is not None, (
            f"README has no {display} K={label} row (the table has "
            f"{sorted(k for m, k in rows if m == display)})"
        )
        trials = curve["per_sample"][str(k)]["random"]
        recall = ", ".join(f"{t['recall']:.1f}" for t in trials)
        assert recall in row, (
            f"the {display} K={label} row does not contain its per-trial recall "
            f"{recall!r}; the table and the artifact have drifted apart (or the rows "
            f"were shuffled -- the numbers are present elsewhere in the file)"
        )
        assert f"{curve['retrieval'][i]:.1f} / " in row, (
            f"the {display} K={label} row does not carry retrieval "
            f"{curve['retrieval'][i]:.1f} in the retrieval column"
        )
        assert f"{label} (" in row, (display, label)
        if model == "qwen3-0.6b" and i == 0:
            # The README prints per-trial exact-match for this one row only: it is the
            # place where a control trial beats the baseline (10/10 against 9/10), and
            # the hybrid's exact column is 0% for a metric reason stated below it.
            exact = ", ".join(f"{t['exact_match'] / 10:.0f}/10" for t in trials)
            assert exact in row, f"the {display} K={label} row lacks exact {exact!r}"
        if i == 0:
            # The baseline is printed on the model's first row, so a model swap would
            # leave the two blocks carrying each other's baseline.
            assert f"{curve['baseline']:.1f} f1" in row, (
                f"the {display} baseline is not on the {display} K={label} row"
            )
