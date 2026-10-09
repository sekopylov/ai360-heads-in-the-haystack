"""Command line interface.

    python -m retrieval_heads.cli describe  --model qwen3.5-0.8b
    python -m retrieval_heads.cli detect    --model qwen3.5-0.8b --profile smoke
    python -m retrieval_heads.cli mask      --model qwen3.5-0.8b --k 1 2 4 8
    python -m retrieval_heads.cli qa        --model qwen3.5-0.8b
    python -m retrieval_heads.cli cot       --model qwen3.5-0.8b
    python -m retrieval_heads.cli compare   --runs results/qwen3.5-0.8b results/qwen3-0.6b
    python -m retrieval_heads.cli figures   --runs results/qwen3.5-0.8b results/qwen3-0.6b
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from retrieval_heads.provenance import add_provenance, warn_if_stale
from retrieval_heads.utils import (
    ensure_dir,
    finite_json,
    get_logger,
    load_json,
    save_json,
    set_seed,
)

log = get_logger("cli")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = REPO_ROOT / "configs" / "models.json"
#: Defaults used to tell "the user asked for this" from "argparse filled it in".
DEFAULT_THRESHOLD = 0.1
DEFAULT_ARGMAX_DOMAIN = "prompt"
#: Sentinel: "detect did not record this field" (None can be a real value).
_MISSING = object()
#: Set by the job driver when it needs a modified registry (e.g. a dtype
#: override).  Keeps `configs/models.json`, which is tracked, untouched.
REGISTRY_ENV = "RETRIEVAL_HEADS_MODELS_JSON"

#: Named grids, so a laptop run and a GPU run differ by one flag.
PROFILES: dict[str, dict[str, Any]] = {
    "smoke": {"lengths": [1024], "depths": 1, "needles": 1, "max_new_tokens": 16},
    "laptop": {"lengths": [1024, 2048, 4096], "depths": 3, "needles": 2, "max_new_tokens": 32},
    "paper": {"lengths": [1024, 2048, 4096, 8192, 16384, 32768, 49152], "depths": 10,
              "needles": 3, "max_new_tokens": 48},
}

#: Per-command default when neither ``--k`` nor ``--k-frac`` is given.  These are
#: applied by :func:`resolve_k`, *not* as argparse defaults: an argparse default
#: is indistinguishable from a value the user typed, so ``mask --k 1 2 4 8 16``
#: used to be silently unioned with the default fractions (10 K values instead of
#: 5 on a 448-head model).  Keeping the fallback here means an explicit ``--k``
#: selects exactly the requested K.
DEFAULT_K_FRACS: dict[str, tuple[float, ...]] = {
    "mask": (0.02, 0.04, 0.08, 0.17, 0.33),
    "qa": (0.04, 0.08, 0.17),
    "cot": (0.08,),
}


def normalize_prefill_chunk(value: int | None) -> int | None:
    """Translate the CLI's ``0 = one shot`` convention into ``None``.

    :func:`retrieval_heads.scoring.decode_with_attention` treats a non-positive
    ``prefill_chunk`` as a programming error and raises; the CLI is the layer that
    promises users ``0`` means "no chunking", so it converts here.
    """
    if value is None:
        return None
    return int(value) if int(value) > 0 else None


def require_matching_scores(scores, info) -> None:
    """Fail when the ablation's head indices come from a different model.

    ``mask``/``qa``/``cot`` load saved scores from ``--out`` and then load the
    model named by ``--model``; nothing tied the two together.  A typo in either
    (``--model qwen3-0.6b --out results/qwen3.5-0.8b``) silently masked heads that
    were chosen for another architecture.  The check is structural -- layer count,
    scoreable layers, head counts -- because names legitimately differ between a
    registry key and a raw path.
    """
    saved = scores.info
    problems: list[str] = []
    if saved.num_layers != info.num_layers:
        problems.append(f"layers {saved.num_layers} != {info.num_layers}")
    if saved.head_dim != info.head_dim:
        # Same layers/heads but a different o_proj geometry would slice the wrong
        # positions; a plain typo in --model reaches here.
        problems.append(f"head_dim {saved.head_dim} != {info.head_dim}")
    if sorted(saved.scoreable_layers) != sorted(info.scoreable_layers):
        problems.append(
            f"scoreable layers {saved.scoreable_layers} != {info.scoreable_layers}"
        )
    if saved.num_heads != info.num_heads:
        problems.append(f"heads per layer {saved.num_heads} != {info.num_heads}")
    if saved.num_kv_heads != info.num_kv_heads:
        problems.append(f"kv heads {saved.num_kv_heads} != {info.num_kv_heads}")
    if saved.hidden_size != info.hidden_size:
        # A base model and its chat/fine-tuned variant share every field above, so
        # without these two the Sec. 4.3-style comparison would silently mask heads
        # chosen on the other checkpoint.
        problems.append(f"hidden_size {saved.hidden_size} != {info.hidden_size}")
    if saved.model_class != info.model_class:
        problems.append(f"model_class {saved.model_class!r} != {info.model_class!r}")
    if problems:
        raise SystemExit(
            f"the saved scores in --out are from {saved.name!r}, not {info.name!r} "
            f"({'; '.join(problems)}). Re-run `detect` for this model, or point --out at "
            f"its own results directory."
        )
    if saved.name != info.name:
        # Names differ legitimately (a registry key vs a raw path), and in a job the
        # weights live under /job/models/..., so this is a warning, not a failure.
        log.warning("the saved scores are labelled %r while --model resolved to %r; "
                    "the geometry matches, so this is probably the same checkpoint "
                    "under two names", saved.name, info.name)


@dataclass
class DetectionSettings:
    """The conditions a detection run recorded, reused by the ablations."""

    system_prompt: str | None = None
    chat_template: bool = True
    enable_thinking: bool | None = False
    argmax_domain: str = "prompt"
    threshold: float = DEFAULT_THRESHOLD
    corpus_path: str | None = None
    #: How `detect` captured the attention rows.  `mask` has no flag for it, so
    #: reading it from the command line would label every artifact "patch" even
    #: after a `detect --capture-method output_attentions` run.
    capture_method: str = "patch"


def resolve_detection_settings(args: argparse.Namespace, scores: Any) -> DetectionSettings:
    """Reuse what detection measured instead of silently changing the conditions.

    ``require_matching_scores`` only checks head geometry, so `detect
    --no-chat-template` followed by a default `mask` used to measure a different
    task, and `detect --corpus essays.txt` followed by `mask` without `--corpus`
    selected heads on natural text but measured the causal effect on the synthetic
    filler.  Every field here prefers the value recorded in
    ``scores.meta['config']`` and warns when the command line explicitly disagrees.
    """
    config = ((getattr(scores, "meta", None) or {}).get("config") or {})
    meta = getattr(scores, "meta", None) or {}
    settings = DetectionSettings()

    def pick(name: str, recorded: Any, requested: Any, implied_by_default: bool) -> Any:
        # `_MISSING` distinguishes "not recorded" from "recorded as None": for
        # enable_thinking, None means "do not pass the kwarg at all" -- which is not
        # the same as True: templates that test `is defined` treat it as off.
        if recorded is _MISSING:
            return requested
        if implied_by_default:
            if recorded != requested:
                log.info("%s=%r taken from the detect run (the command line implies %r)",
                         name, recorded, requested)
            return recorded
        if requested != recorded:
            log.warning("%s=%r differs from the value detect recorded (%r); this ablation "
                        "measures different conditions", name, requested, recorded)
        return requested

    settings.system_prompt = pick("system-prompt", config.get("system_prompt", _MISSING),
                                  args.system_prompt, args.system_prompt is None)
    settings.chat_template = pick("chat-template", config.get("chat_template", _MISSING),
                                  not args.no_chat_template, not args.no_chat_template)
    # `True`, not `None`: omitting the kwarg leaves `enable_thinking` undefined, and
    # the Qwen3.5 template tests `is defined and is true` -- so `--thinking` used to
    # do nothing there (and the artifact recorded `null`, which reads as "left on").
    # `True` is correct for both shipped templates: Qwen3.5 turns thinking on, and
    # Qwen3-0.6B (whose test is `is defined and is false`) is not turned off.
    requested_thinking = True if args.thinking else False
    settings.enable_thinking = pick("enable-thinking",
                                    config.get("enable_thinking", _MISSING),
                                    requested_thinking, not args.thinking)
    recorded_domain = meta.get("argmax_domain", config.get("argmax_domain", _MISSING))
    # Only `detect` has the flag; the ablations still record the domain the heads
    # were selected under (it is not re-derived from the scores).
    requested_domain = getattr(args, "argmax_domain", DEFAULT_ARGMAX_DOMAIN)
    settings.argmax_domain = pick("argmax-domain", recorded_domain, requested_domain,
                                  requested_domain == DEFAULT_ARGMAX_DOMAIN)
    requested_threshold = getattr(args, "threshold", DEFAULT_THRESHOLD)
    settings.threshold = pick("threshold", config.get("threshold", _MISSING),
                              requested_threshold,
                              requested_threshold == DEFAULT_THRESHOLD)
    # Not a flag on the ablations: the only truthful source is what detect recorded.
    settings.capture_method = config.get("capture_method") or settings.capture_method

    recorded_corpus = meta.get("corpus_path")
    # `qa`/`cot` have no --corpus flag at all, so this must not assume one.
    requested_corpus = getattr(args, "corpus", None)
    if requested_corpus is None and recorded_corpus:
        log.info("corpus=%r taken from the detect run", recorded_corpus)
        settings.corpus_path = recorded_corpus
    else:
        if requested_corpus != recorded_corpus:
            log.warning("corpus=%r differs from the value detect recorded (%r); the "
                        "filler distribution changes", requested_corpus, recorded_corpus)
        settings.corpus_path = requested_corpus
    return settings


def default_out_dir(model_arg: str, out: str | None) -> Path:
    """Where a command writes when ``--out`` is omitted.

    ``resolve_model`` accepts a filesystem path, and ``REPO_ROOT / "results" /
    "/abs/path"`` collapses to the absolute path -- artifacts ended up inside the
    checkpoint directory.  Path-like arguments contribute only their basename.
    """
    if out:
        return Path(out)
    path_like = os.sep in model_arg or "/" in model_arg
    return REPO_ROOT / "results" / (Path(model_arg).name if path_like else model_arg)


#: Rough token overhead of the needle, question and chat template on top of the
#: filler (``build_needle_sample`` measures it exactly; this is the pre-flight guard).
PROMPT_OVERHEAD_TOKENS = 64


def within_context_limit(lengths, info) -> tuple[list[int], list[int]]:
    """Split lengths into ``(kept, dropped)`` by the model's context window.

    ``target_tokens`` is the realized prompt, but the check happens before the
    prompt exists, so a margin is subtracted: without it a requested length equal to
    ``max_position_embeddings`` passed the filter and the realized context (filler +
    needle + question + template) landed past the trained window.

    Positions past the trained range are extrapolation, not a measurement, and the
    paper profile asks for 49152 from Qwen3-0.6B, which was trained to 40960.  The
    dropped list is returned so the artifact can show the requested grid.
    """
    limit = getattr(info, "max_position_embeddings", None)
    if not limit:
        return list(lengths), []
    budget = limit - PROMPT_OVERHEAD_TOKENS
    kept = [length for length in lengths if length <= budget]
    dropped = [length for length in lengths if length > budget]
    if dropped:
        log.warning("%s: dropping lengths %s above max_position_embeddings=%d minus the "
                    "%d-token prompt overhead (budget %d)",
                    info.name, dropped, limit, PROMPT_OVERHEAD_TOKENS, budget)
    return kept, dropped


def load_runs(paths, pairing: str):
    """``{label: RetrievalScores}`` with collisions disambiguated by directory.

    The label used to be ``scores.info.name`` alone, so ``--runs a/qwen3-0.6b
    b/qwen3-0.6b`` collapsed to one entry and the correlation silently described
    a single model.
    """
    from retrieval_heads.scoring import RetrievalScores

    runs = {}
    for run in paths:
        path = Path(run)
        scores = RetrievalScores.load(path / f"scores_{pairing}")
        label = scores.info.name
        if label in runs:
            label = f"{label} ({path.parent})"
            log.warning("two runs share the model name %r; labelling %s as %r",
                        scores.info.name, path, label)
        while label in runs:
            label += " "
        runs[label] = scores
    return runs


def load_registry(path: str | Path | None = None) -> dict[str, Any]:
    path = path or os.environ.get(REGISTRY_ENV) or DEFAULT_REGISTRY
    return load_json(path)["models"]


def resolve_model(name: str | Path, registry: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
    """Accept a registry key or a raw path; return ``(path, settings)``.

    Registry paths are relative to the repository, not to the caller's working
    directory, so the installed ``retrieval-heads`` console script and any run
    from outside the repo root still find the checkpoints.
    """
    registry = registry if registry is not None else load_registry()
    if name in registry:
        settings = dict(registry[name])
        path = Path(settings["path"])
        if not path.is_absolute() and not path.exists():
            rooted = REPO_ROOT / path
            if rooted.exists():
                path = rooted
        if not path.exists():
            raise SystemExit(
                f"checkpoint for {name!r} is not on disk at {path} "
                f"(registry path {settings['path']!r}); run scripts/download_models.sh "
                f"or pass --model <existing path>"
            )
        settings["path"] = str(path)
        return str(path), settings
    candidate = Path(name)
    if candidate.exists():
        return str(candidate), {"path": str(candidate), "dtype": "float32"}
    raise SystemExit(f"unknown model {name!r}; known: {sorted(registry)} (or pass an existing path)")


def resolve_k(args: argparse.Namespace, info, *, default_fracs: Sequence[float] = ()) -> list[int]:
    """Turn ``--k`` (absolute) and ``--k-frac`` (fraction of heads) into one list.

    Absolute K is meaningless across models: K=8 is 17% of Qwen3.5-0.8B's 48
    scoreable heads but only 1.8% of Qwen3-0.6B's 448.  Fractions make the two
    directly comparable, which is what the paper's "about 5% of heads" framing
    actually requires.

    Precedence, in order:

    * ``--k`` and ``--k-frac`` are **both** given explicitly -> their union (the
      user asked for both).
    * only one of them -> exactly that one.
    * neither -> ``default_fracs`` for the command.

    This replaces the earlier behaviour where ``--k-frac`` carried a non-empty
    argparse default, so an explicit ``--k`` was always unioned with it.
    """
    raw_k = getattr(args, "k", None)
    raw_frac = getattr(args, "k_frac", None)
    if raw_k is not None and any(int(k) <= 0 for k in raw_k):
        # A mixed list like [-1, 5] used to drop the negative silently.
        raise SystemExit(f"--k must be positive throughout, got {list(raw_k)}")
    if raw_frac is not None and any(float(f) <= 0 for f in raw_frac):
        raise SystemExit(f"--k-frac must be positive throughout, got {list(raw_frac)}")
    ks = [int(k) for k in (raw_k or [])]
    fracs = [float(f) for f in (raw_frac or [])]
    if not ks and not fracs:
        fracs = [float(f) for f in default_fracs]

    values: set[int] = set(ks)
    for frac in fracs:
        if frac > 1.0:
            log.warning("--k-frac %g exceeds 1.0 (all scoreable heads); clamping to 1.0", frac)
            frac = 1.0
        # int(x + 0.5), not round(): round() is banker's rounding, so 0.5 heads
        # rounded down to the even neighbour.
        values.add(max(1, int(frac * info.n_scoreable_heads + 0.5)))
    if not values:
        values = {max(1, int(0.05 * info.n_scoreable_heads + 0.5))}
    return sorted(values)


#: One resident model, reused across the stages of the same model in one process.
#: `datasphere_job.py` runs a model's stages back to back for exactly this reason:
#: loading Qwen3.5-0.8B takes ~50 s on the job GPU, and a full run used to pay that
#: once per stage per model.  Size 1 on purpose -- two models at once would double
#: the resident memory for no benefit, since the stages are model-major.
_LOADED: dict[tuple, tuple] = {}


def _load(name: str, *, attn_implementation: str = "eager", dtype: str | None = None):
    import torch

    from retrieval_heads.models import describe_model, load_model

    path, settings = resolve_model(name)
    resolved_dtype = dtype or settings.get("dtype", "float32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = (path, str(resolved_dtype), attn_implementation, device)
    cached = _LOADED.get(key)
    if cached is not None:
        log.info("reusing the already-loaded %s (no second weight load)", name)
        return cached

    if _LOADED:
        # Evict *before* allocating the new weights.  Clearing after `load_model`
        # (as this did) kept both models resident through the load, contradicting
        # the "one resident model" guarantee the cache exists to provide.
        _LOADED.clear()
        if device == "cuda":  # pragma: no cover - GPU path
            torch.cuda.empty_cache()

    model, tokenizer, info = load_model(
        path, dtype=resolved_dtype,
        attn_implementation=attn_implementation, device=device,
    )
    if device == "cuda":  # pragma: no cover - GPU path
        log.info("loaded model on CUDA")
    print(describe_model(info))
    print()
    _LOADED[key] = (model, tokenizer, info)
    return model, tokenizer, info


# --------------------------------------------------------------------------- commands
def cmd_describe(args: argparse.Namespace) -> int:
    _, _, info = _load(args.model, dtype=args.dtype)
    if args.out:
        save_json(add_provenance(info.as_dict(), dtype=info.dtype),
                  Path(args.out) / "model_info.json")
    return 0


def cmd_detect(args: argparse.Namespace) -> int:
    from retrieval_heads.detection import DETECTION_NEEDLES, DetectionConfig, run_detection

    profile = PROFILES.get(args.profile, {}) if args.profile else {}
    # `is None`, not `or`: a user-supplied 0 or [] must be an error, not a silent
    # fallback to the profile (which is what `--depths 0` used to do).
    lengths = args.lengths if args.lengths is not None else profile.get(
        "lengths", [1024, 2048, 4096])
    depths = args.depths if args.depths is not None else profile.get("depths", 3)
    needles = args.needles if args.needles is not None else profile.get(
        "needles", len(DETECTION_NEEDLES))
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else profile.get(
        "max_new_tokens", 32)
    if int(needles) > len(DETECTION_NEEDLES):
        log.warning("--needles %s clamped to the %d shipped detection needles",
                    needles, len(DETECTION_NEEDLES))
    if not lengths or int(depths) < 1 or int(needles) < 1:
        raise SystemExit(
            f"empty detection grid: lengths={list(lengths)}, depths={depths}, "
            f"needles={needles}; each must be non-empty/positive"
        )
    out_dir = default_out_dir(args.model, args.out)

    model, tokenizer, info = _load(args.model, dtype=args.dtype)
    lengths, dropped_lengths = within_context_limit(lengths, info)
    if not lengths:
        raise SystemExit(
            f"every requested length exceeds {info.name}'s max_position_embeddings="
            f"{info.max_position_embeddings}; nothing to measure"
        )
    cfg = DetectionConfig(
        lengths=list(lengths),
        depths_per_length=int(depths),
        needles=list(DETECTION_NEEDLES[: int(needles)]),
        max_new_tokens=int(max_new_tokens),
        threshold=args.threshold,
        pairing=args.pairing,
        chat_template=not args.no_chat_template,
        # `True`, not `None`: see resolve_detection_settings -- omitting the kwarg
        # leaves the Qwen3.5 template's `enable_thinking` undefined, i.e. off.
        enable_thinking=True if args.thinking else False,
        system_prompt=args.system_prompt,
        capture_method=args.capture_method,
        argmax_domain=args.argmax_domain,
        capture_impl=args.capture_impl,
        prefill_impl=args.prefill_impl,
        prefill_chunk=normalize_prefill_chunk(args.prefill_chunk),
        seed=args.seed,
        limit=args.limit,
        dropped_lengths=dropped_lengths,
    )
    if args.corpus:
        from retrieval_heads.haystack import load_corpus

        corpus = load_corpus(args.corpus)
    else:
        corpus = None
    run_detection(model, tokenizer, info, cfg, corpus=corpus, out_dir=out_dir,
                  corpus_path=args.corpus)
    return 0


def require_ablation_args(args: argparse.Namespace) -> None:
    """Reject the ablation arguments that would otherwise produce silent zeros.

    `detect` already refuses `max_new_tokens <= 0` (via `decode_with_attention`), but
    `mask`/`qa`/`cot` accepted it and then reported all-zero metrics; and
    `--random-trials 0` left `np.mean([])` -> NaN in the artifact with an empty
    `random_trials` list, while `token_mixer_ablation` silently clamped its own trial
    count with `max(1, n_trials)`.  Fail like the rest of the CLI does.
    """
    # `getattr` everywhere, including inside the messages: this helper is called by
    # `mask`, `qa` and `cot`, and `--mixer-trials` exists only on `mask`.
    max_new_tokens = getattr(args, "max_new_tokens", 1)
    if max_new_tokens < 1:
        raise SystemExit(
            f"--max-new-tokens must be >= 1 (got {max_new_tokens}); 0 generates "
            f"nothing and every metric would be 0"
        )
    random_trials = getattr(args, "random_trials", 1)
    if random_trials < 1:
        raise SystemExit(
            f"--random-trials must be >= 1 (got {random_trials}); the control arm "
            f"is undefined without at least one trial"
        )
    mixer_trials = getattr(args, "mixer_trials", 1)
    if mixer_trials < 1:
        raise SystemExit(f"--mixer-trials must be >= 1 (got {mixer_trials})")


def cmd_mask(args: argparse.Namespace) -> int:
    require_ablation_args(args)
    import torch

    from retrieval_heads.detection import EVAL_NEEDLES
    from retrieval_heads.masking import (
        make_eval_samples, masking_curve, token_mixer_ablation,
    )
    from retrieval_heads.scoring import RetrievalScores

    from retrieval_heads.haystack import iter_depths, load_corpus

    out_dir = default_out_dir(args.model, args.out)
    scores = RetrievalScores.load(out_dir / f"scores_{args.pairing}")
    model, tokenizer, info = _load(args.model, dtype=args.dtype)
    require_matching_scores(scores, info)
    # Must match what detect measured: prefer the recorded conditions, warn on a clash.
    settings = resolve_detection_settings(args, scores)
    corpus = load_corpus(settings.corpus_path) if settings.corpus_path else None

    # The eval needle is held out of detection: selecting heads on the same text
    # they are then scored on inflates the retrieval arm (paper: "additional set").
    needle, question = EVAL_NEEDLES[0]
    # Ablations re-run the *prefill* for every masking configuration, so context
    # length dominates their cost.  One length x three depths keeps the laptop run
    # honest without turning it into an overnight job.
    # Five depths (not three): the curve is noisy at three samples, and the extra
    # points are what make the per-K spread meaningful.  A properly powered run is
    # still the paper-scale grid.
    # Same window guard as `detect`: a length past the trained window is
    # extrapolation, not a measurement, and `detect` already refuses it.
    # An explicit empty list is an error, not a request for the default; `detect`
    # already refuses an empty grid the same way.
    if args.lengths is not None and not args.lengths:
        raise SystemExit("--lengths was given but empty; pass at least one length")
    lengths, _dropped = within_context_limit(args.lengths or (1024,), info)
    if not lengths:
        raise SystemExit(
            f"every requested length exceeds {info.name}'s window; nothing to measure"
        )
    # More depths and more held-out needles widen the ablation sample set, which is
    # what `retrieval_std` and the random arm's spread are computed over.
    if args.depths < 1:
        raise SystemExit(f"--depths must be >= 1 (got {args.depths})")
    eval_needles = list(EVAL_NEEDLES[: max(1, args.needles)]
                        if args.needles <= len(EVAL_NEEDLES) else EVAL_NEEDLES)
    if args.needles > len(EVAL_NEEDLES):
        log.warning("--needles %d exceeds the %d held-out eval needle(s); using all of "
                    "them (add more to EVAL_NEEDLES to widen the ablation set)",
                    args.needles, len(EVAL_NEEDLES))
    samples = []
    for extra_index, (extra_needle, extra_question) in enumerate(eval_needles):
        samples.extend(make_eval_samples(
            tokenizer, lengths=lengths, depths=tuple(iter_depths(args.depths)),
            needle=extra_needle, question=extra_question,
            # A different seed block per needle, so two needles never share filler.
            seed=args.seed + 7 + 101 * extra_index,
            chat_template=settings.chat_template,
            enable_thinking=settings.enable_thinking,
            corpus=corpus,
            system_prompt=settings.system_prompt,
        ))
    k_values = resolve_k(args, info, default_fracs=DEFAULT_K_FRACS["mask"])
    log.info("masking K values %s (%.1f%%-%.1f%% of %d scoreable heads)",
             k_values, 100 * k_values[0] / info.n_scoreable_heads,
             100 * k_values[-1] / info.n_scoreable_heads, info.n_scoreable_heads)
    curve = masking_curve(
        model, tokenizer, info, scores, samples,
        k_values=k_values, n_random_trials=args.random_trials,
        max_new_tokens=args.max_new_tokens, seed=args.seed,
        prefill_chunk=normalize_prefill_chunk(args.prefill_chunk),
    )
    curve.meta["corpus"] = "custom" if corpus else "synthetic"
    curve.meta["corpus_path"] = settings.corpus_path
    # With `--needles > 1` the curve averages over several held-out needles, so the
    # artifact records all of them (the single `needle` key stays for older readers).
    curve.meta["needle"] = needle
    curve.meta["question"] = question
    curve.meta["needles"] = [{"needle": n, "question": q} for n, q in eval_needles]
    curve.meta["needle_source"] = "eval"       # never in DETECTION_NEEDLES
    curve.meta["depths_per_length"] = args.depths
    curve.meta["seed"] = args.seed
    # The conditions the *heads were chosen under* (detect) and the ones this
    # ablation ran under; if they differ the artifact says so instead of hiding it.
    curve.meta["chat_template"] = settings.chat_template
    curve.meta["system_prompt"] = settings.system_prompt
    curve.meta["enable_thinking"] = settings.enable_thinking
    curve.meta["threshold"] = settings.threshold
    curve.meta["detection_pairing"] = (scores.meta or {}).get("config", {}).get("pairing")
    curve.meta["capture_method"] = settings.capture_method
    save_json(add_provenance(curve.as_dict(), dtype=info.dtype),
              out_dir / "masking_curve.json")

    if info.linear_layers:
        mixer_k = tuple(k for k in k_values if k <= 4) or (1, 2)
        if mixer_k != tuple(k_values):
            log.warning("token-mixer ablation runs at K=%s, not the requested K=%s: the "
                        "ablation re-runs the prefill per layer, so it is capped at K<=4",
                        list(mixer_k), list(k_values))
        ablation = token_mixer_ablation(model, tokenizer, info, samples,
                                        k_values=mixer_k,
                                        max_new_tokens=args.max_new_tokens,
                                        prefill_chunk=normalize_prefill_chunk(args.prefill_chunk),
                                        n_trials=args.mixer_trials, seed=args.seed,
                                        # The masking curve already ran the unmasked
                                        # pass on these exact samples; on the hybrid
                                        # re-running it was the stage's costliest step.
                                        baseline=curve.baseline)
        payload = ablation.as_dict()
        payload.update({"needle": needle, "question": question,
                        "needle_source": "eval", "seed": args.seed,
                        "model": info.name, "max_new_tokens": args.max_new_tokens,
                        "prefill_chunk": normalize_prefill_chunk(args.prefill_chunk),
                        "chat_template": settings.chat_template,
                        "system_prompt": settings.system_prompt,
                        "corpus": "custom" if corpus else "synthetic",
                        "corpus_path": settings.corpus_path})
        save_json(add_provenance(payload, dtype=info.dtype),
                  out_dir / "mixer_ablation.json")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


def warn_oversized_samples(samples: Sequence[Any], info: Any, tokenizer: Any,
                           what: str) -> None:
    """Warn when a document is longer than the model's trained window.

    `--data` is user-supplied and bypasses every grid check, so the only guard is
    this measurement.
    """
    limit = getattr(info, "max_position_embeddings", None)
    if not limit:
        return
    budget = limit - PROMPT_OVERHEAD_TOKENS
    longest = 0
    for sample in samples:
        text = f"{getattr(sample, 'context', '')} {getattr(sample, 'question', '')}"
        longest = max(longest, len(tokenizer(text, add_special_tokens=False).input_ids))
    if longest > budget:
        log.warning("%s: the longest sample is %d tokens, above the %d-token window "
                    "minus overhead; those answers are extrapolation",
                    what, longest, limit)


def cmd_qa(args: argparse.Namespace) -> int:
    require_ablation_args(args)
    from retrieval_heads.downstream import builtin_qa_samples, load_qa_jsonl, qa_ablation
    from retrieval_heads.scoring import RetrievalScores

    out_dir = default_out_dir(args.model, args.out)
    scores = RetrievalScores.load(out_dir / f"scores_{args.pairing}")
    model, tokenizer, info = _load(args.model, dtype=args.dtype)
    require_matching_scores(scores, info)
    settings = resolve_detection_settings(args, scores)

    samples = load_qa_jsonl(args.data) if args.data else builtin_qa_samples()
    warn_oversized_samples(samples, info, tokenizer, "qa")
    result = qa_ablation(model, tokenizer, info, scores, samples,
                         k_values=resolve_k(args, info, default_fracs=DEFAULT_K_FRACS["qa"]),
                         n_random_trials=args.random_trials,
                         seed=args.seed, max_new_tokens=args.max_new_tokens,
                         prefill_chunk=normalize_prefill_chunk(args.prefill_chunk),
                         enable_thinking=settings.enable_thinking,
                         chat_template=settings.chat_template,
                         system_prompt=settings.system_prompt)
    save_json(add_provenance(result, dtype=info.dtype), out_dir / "task_qa.json")
    return 0


def cmd_cot(args: argparse.Namespace) -> int:
    require_ablation_args(args)
    from retrieval_heads.downstream import (
        builtin_reasoning_samples, cot_ablation, load_reasoning_jsonl,
    )
    from retrieval_heads.scoring import RetrievalScores

    out_dir = default_out_dir(args.model, args.out)
    scores = RetrievalScores.load(out_dir / f"scores_{args.pairing}")
    model, tokenizer, info = _load(args.model, dtype=args.dtype)
    require_matching_scores(scores, info)
    settings = resolve_detection_settings(args, scores)

    samples = load_reasoning_jsonl(args.data) if args.data else builtin_reasoning_samples()
    warn_oversized_samples(samples, info, tokenizer, "cot")
    ks = resolve_k(args, info, default_fracs=DEFAULT_K_FRACS["cot"])
    if len(ks) > 1:
        log.warning("cot evaluates a single K per run; using %d and ignoring %s",
                    ks[0], ks[1:])
    result = cot_ablation(model, tokenizer, info, scores, samples,
                          k=ks[0],
                          n_random_trials=args.random_trials,
                          seed=args.seed, max_new_tokens=args.max_new_tokens,
                          prefill_chunk=normalize_prefill_chunk(args.prefill_chunk),
                          enable_thinking=settings.enable_thinking,
                          chat_template=settings.chat_template,
                          system_prompt=settings.system_prompt)
    save_json(add_provenance(result, dtype=info.dtype), out_dir / "task_cot.json")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from retrieval_heads.properties import correlation_matrix, head_overlap

    runs = load_runs(args.runs, args.pairing)
    names = list(runs)
    auto = args.mode is None
    mode = args.mode or ("grid" if _same_layout(runs) else "sorted")
    if auto and mode == "sorted":
        log.warning("layouts differ, so `sorted` mode is used: it correlates *ranked* "
                    "score vectors and is high by construction for any two heavy-tailed "
                    "distributions.  It does NOT mean the models use the same heads.")
    # The caveat and the mode now travel together, set by `properties` itself, so the
    # file and stdout cannot disagree and neither can contradict its own `mode` field
    # (`head_overlap` used to leave `mode` at its default while the correlation inside
    # had been computed in `sorted`).
    corr_payload = correlation_matrix([runs[n] for n in names], mode=mode, labels=names).as_dict()
    out_dir = Path(args.out or REPO_ROOT / "results")
    save_json(add_provenance(corr_payload), out_dir / "correlation.json")
    # `finite_json` first: `json.dumps` would print bare `NaN` for a grid-mode
    # comparison across layouts, which is not valid JSON even though `save_json`
    # (allow_nan=False) writes null for the same value.
    print(json.dumps(finite_json(corr_payload), indent=2))
    if len(names) == 2:
        overlap = head_overlap(runs[names[0]], runs[names[1]], threshold=args.threshold,
                               mode=mode)
        overlap_payload = overlap.as_dict()
        save_json(add_provenance(overlap_payload), out_dir / "overlap.json")
        print(json.dumps(finite_json(overlap_payload), indent=2))
    return 0


def cmd_figures(args: argparse.Namespace) -> int:
    from retrieval_heads.plotting import (
        plot_corr_map, plot_heat_map, plot_layer_profile, plot_masking_curve,
        plot_mixer_ablation, plot_score_distribution, plot_score_pie, plot_task_cot,
        plot_task_qa, save_fig,
    )
    from retrieval_heads.properties import correlation_matrix

    fig_dir = ensure_dir(args.out or REPO_ROOT / "results" / "figures")
    runs = load_runs(args.runs, args.pairing)
    labels = list(runs)
    pairs = list(zip((Path(run) for run in args.runs), labels))
    curves, qa, cot, mixers = {}, {}, {}, {}
    for path, label in pairs:
        for name, store in (("masking_curve.json", curves), ("task_qa.json", qa),
                            ("task_cot.json", cot), ("mixer_ablation.json", mixers)):
            candidate = path / name
            if candidate.exists():
                payload = json.loads(candidate.read_text(encoding="utf-8"))
                warn_if_stale(payload, str(candidate), log=log)
                store[label] = payload

    def safe(name: str, factory) -> None:
        # One unplottable figure (e.g. mixer sweeps with different K sets) must not
        # abort the stage after half the PDFs were already written.  Catching only
        # ValueError still let a malformed artifact's KeyError kill the stage.
        try:
            save_fig(factory(), fig_dir / name)
        except Exception as exc:  # noqa: BLE001 - a figure must never kill the stage
            log.warning("skipping figure %s: %s", name, exc)

    if runs:
        safe("ring_graph.pdf", lambda: plot_score_pie(runs))
        safe("score_distribution.pdf", lambda: plot_score_distribution(runs))
        safe("heat_map.pdf", lambda: plot_heat_map(runs))
        safe("layer_profile.pdf", lambda: plot_layer_profile(runs))
        mode = "grid" if _same_layout(runs) else "sorted"
        corr = correlation_matrix(list(runs.values()), mode=mode, labels=list(runs))
        if mode == "sorted":
            corr.caveat = ("sorted mode correlates ranked score vectors; a high value "
                           "does not mean the models use the same heads")
        safe("corr_map.pdf", lambda: plot_corr_map(corr))
    if curves:
        safe("masking_heads.pdf", lambda: plot_masking_curve(curves))
        # F1/EM score against the whole needle while the question asks for a
        # sub-span, so the confounded series is not the only one shipped: the LCS
        # recall is plotted beside it.
        safe("masking_recall.pdf", lambda: plot_masking_curve(curves, metric="recall"))
    # QA/CoT are per-model figures; the old code plotted only the first run and
    # silently dropped the rest.  With one run the filenames stay as documented;
    # with several each gets a label suffix.
    for label, result in qa.items():
        name = "task_qa.pdf" if len(qa) == 1 else f"task_qa_{_slug(label)}.pdf"
        safe(name, lambda result=result: plot_task_qa(result))
    for label, result in cot.items():
        name = "task_cot.pdf" if len(cot) == 1 else f"task_cot_{_slug(label)}.pdf"
        safe(name, lambda result=result: plot_task_cot(result))
    if mixers:
        safe("mixer_ablation.pdf", lambda: plot_mixer_ablation(mixers))
    return 0

def _same_layout(runs: dict[str, Any]) -> bool:
    """True when every run addresses the same layer x head grid.

    The mode used to be chosen by matrix *shape*: two hybrids with the same
    ``(num_layers, max_heads)`` but different attention layers took the ``grid``
    path and compared unrelated positions.
    """
    from retrieval_heads.properties import layouts_match

    items = list(runs.values())
    return all(layouts_match(items[0], other) for other in items[1:])


def _slug(text: str) -> str:
    """Filesystem-safe figure suffix for a run label."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-") or "run"


# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval_heads", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model", required=True, help="registry key (see configs/models.json) or path")
        p.add_argument("--out", default=None, help="results directory")
        p.add_argument("--pairing", default="next_step", choices=["next_step", "same_step"])
        p.add_argument("--no-chat-template", action="store_true")
        p.add_argument("--thinking", action="store_true", help="leave the model's thinking mode on")
        p.add_argument("--system-prompt", default=None,
                       help="system message for the chat template; it must be the same "
                            "for detect and every ablation, or they measure different "
                            "prompts (detect records it in scores.meta['config'])")
        p.add_argument("--dtype", default=None, choices=["float32", "bfloat16"],
                       help="override the registry dtype (bfloat16 on GPU)")
        # Also accepted per-subcommand (not just before it): job drivers build
        # argv as `detect --model ... --seed 0 ...`, and argparse only allows a
        # parent-parser option *before* the subcommand.
        p.add_argument("--seed", type=int, default=argparse.SUPPRESS)

    p = sub.add_parser("describe", help="print the architecture and scoreable-head census")
    add_common(p)
    p.set_defaults(func=cmd_describe)

    p = sub.add_parser("detect", help="run the retrieval-head detection grid")
    add_common(p)
    p.add_argument("--profile", default="laptop", choices=sorted(PROFILES))
    p.add_argument("--lengths", type=int, nargs="*")
    p.add_argument("--depths", type=int)
    p.add_argument("--needles", type=int)
    p.add_argument("--max-new-tokens", type=int)
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--capture-method", default="patch",
                   choices=["output_attentions", "patch"],
                   help="patch (default) keys captured maps by layer_idx and cannot "
                        "be confused by attention-map ordering")
    p.add_argument("--argmax-domain", default="prompt", choices=["prompt", "full"],
                   help="positions the attention argmax may choose from; 'prompt' is "
                        "the paper's input-token criterion, 'full' also allows the "
                        "already-generated tokens")
    p.add_argument("--capture-impl", default="eager",
                   help="attention kernel used while capturing (eager required)")
    p.add_argument("--prefill-impl", default="sdpa")
    p.add_argument("--prefill-chunk", type=int, default=4096,
                   help="feed the prefill in chunks of this size (0 = one shot); bounds "
                        "memory when float32 SDPA falls back to the matmul kernel")
    p.add_argument("--corpus", default=None, help="text file of filler sentences")
    p.add_argument("--limit", type=int, default=None, help="cap the number of instances")
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser("mask", help="mask top-K retrieval heads vs K random heads")
    add_common(p)
    p.add_argument("--k", type=int, nargs="+", default=None,
                   help="absolute numbers of heads to mask (overrides the default "
                        "--k-frac; give both to union them)")
    p.add_argument("--k-frac", type=float, nargs="+", default=None,
                   help="K as a fraction of scoreable heads -- comparable across models "
                        f"(default: {' '.join(map(str, DEFAULT_K_FRACS['mask']))})")
    p.add_argument("--lengths", type=int, nargs="*", default=None)
    p.add_argument("--random-trials", type=int, default=3)
    # The ablation sample set used to be hard-coded (5 depths, one held-out needle),
    # and it is the weakest part of the causal measurement: `retrieval_std` is the
    # spread over exactly those samples.  A paper-scale machine can afford more.
    p.add_argument("--depths", type=int, default=5,
                   help="relative depths per length, endpoints inclusive "
                        "(default 5; the detection grid uses 10)")
    p.add_argument("--needles", type=int, default=1,
                   help="how many held-out eval needles to use (default 1)")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--corpus", default=None,
                   help="text file of filler sentences; keep it the same as the detect "
                        "run, or the causal experiment uses different filler")
    p.add_argument("--mixer-trials", type=int, default=3,
                   help="random layer subsets per K for the token-mixer ablation "
                        "(averaged; 1 restores the old deterministic first-K choice)")
    p.add_argument("--prefill-chunk", type=int, default=4096)
    p.set_defaults(func=cmd_mask)

    p = sub.add_parser("qa", help="extractive-QA ablation")
    add_common(p)
    p.add_argument("--data", default=None, help="JSONL with context/question/answer")
    p.add_argument("--k", type=int, nargs="+", default=None)
    p.add_argument("--k-frac", type=float, nargs="+", default=None,
                   help="K as a fraction of scoreable heads "
                        f"(default: {' '.join(map(str, DEFAULT_K_FRACS['qa']))})")
    p.add_argument("--random-trials", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--prefill-chunk", type=int, default=4096,
                   help="feed the prefill in chunks of this size (0 = one shot); bounds "
                        "memory on long --data documents")
    p.set_defaults(func=cmd_qa)

    p = sub.add_parser("cot", help="chain-of-thought ablation")
    add_common(p)
    p.add_argument("--data", default=None, help="JSONL with question/answer")
    p.add_argument("--k", type=int, nargs="+", default=None)
    p.add_argument("--k-frac", type=float, nargs="+", default=None,
                   help="K as a fraction of scoreable heads "
                        f"(default: {' '.join(map(str, DEFAULT_K_FRACS['cot']))})")
    p.add_argument("--random-trials", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--prefill-chunk", type=int, default=4096,
                   help="feed the prefill in chunks of this size (0 = one shot); bounds "
                        "memory on long --data documents")
    p.set_defaults(func=cmd_cot)

    p = sub.add_parser("compare", help="correlate / overlap models")
    p.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--pairing", default="next_step", choices=["next_step", "same_step"])
    p.add_argument("--mode", default=None, choices=["grid", "sorted"])
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("figures", help="regenerate every figure from saved results")
    p.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--pairing", default="next_step", choices=["next_step", "same_step"])
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_figures)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    set_seed(getattr(args, "seed", 0))
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
