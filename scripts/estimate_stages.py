#!/usr/bin/env python
"""Estimate the A100 wall clock of `detect` and `mask` for the registry's models.

    .venv/bin/python scripts/estimate_stages.py
    .venv/bin/python scripts/estimate_stages.py --check-anchors
    .venv/bin/python scripts/estimate_stages.py --random-trials 3 --price 542.88

Why this exists: the cost comments in `configs/datasphere/*.yaml` have been wrong
twice (once from extrapolating the prefill quadratically, once from using the wrong
pass unit), and a comment cannot be re-derived.  This script holds the model, the
measured anchors, and prints the table a config can quote.

The grids are read from the driver (`SCALES[scale]`) and the CLI (`PROFILES`), so
they cannot drift from what a job actually runs.

The model, and how honest it is
-------------------------------
Both stages are counted in **passes** (one prompt, one generation), not tokens:

* `detect` is one pass per instance.  Its cost is split between the **capture**
  (eager attention + a row argmax over the whole KV, at *every* decode step, which
  scales with `scoreable layers x Q-heads`) and prefill+decode (which scales with the
  **active** parameters -- a MoE reads only its active experts).
* `mask` is `samples x (1 baseline + k_points x (1 + random_trials))` passes, plus
  the token-mixer ablation for a hybrid (`len(mixer_k) x 2 kinds x --mixer-trials x
  samples`, where `mixer_k` is the CLI's `k <= 4` subset and is 2 values, not 3, on
  three of the four hybrids).

Two anchors (below) fix the constants, and that is also the model's weakness: with two
models and four measured numbers it has **no degrees of freedom left**, so it is a fit,
not a validated model.  `--check-anchors` prints the fit and says which rows are
independent (the dense `detect` row is the one `CAPTURE_S_PER_HEADLAYER` was solved from,
so its `-0%` is an identity, not evidence).

Known gaps, stated rather than hidden
-------------------------------------
1. **No attention term.**  `compute_seconds` is the linear (weight) part only.  The
   prefill's attention is quadratic in the length and the capture's row-`argmax` is
   linear in the KV, so both are missing.  This is the most likely reason the two mask
   anchors invert (the dense 0.6B's mask pass costs *more* than the hybrid's despite 0.7x
   the active parameters and 0.69x the length-dependent work): the anchors bracket the
   overhead instead of pinning it, and the residual is the missing term.  Direction of
   the error: the table is **optimistic**, most on the rows with the most scoreable
   head-layers and the longest contexts.  `a100-preflight-mask` exists to measure it.
2. **One price per mask pass, although there are two kinds.**  The mixer ablation's passes
   are ~2x a curve pass: with the token mixers zeroed the model stops emitting EOS and
   spends the whole 96-token budget, while curve passes usually stop early.  On the
   anchored hybrid the curve blocks cost 1.37-1.53 s/pass (measured, 270 passes each) and
   the ablation's ~3 s, so the single fitted `MASK_OVERHEAD_S` is an average that happens
   to land the total.  Changing the curve:ablation ratio (e.g. `--random-trials 3`) breaks
   that compensation.
3. **The capture is priced per instance, not per token.**  Real cost grows with the
   context the `argmax` scans; the grid holds the lengths fixed, so the fit absorbs it.
4. `MASK_OVERHEAD_S` is the midpoint of the bracket the two anchors leave, so the mask
   column is a plan, and the first hour of a real run is what confirms it.

Anchors (A100 80 GB, bf16, 45 samples/point, 96-token budget):
    qwen3-0.6b    detect 563 s / 210 instances    mask 3660 s = 1665 passes
    qwen3.5-0.8b  detect 610 s / 270 instances    mask 4500 s = 2205 passes
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: A100 bf16 peak is 312 TFLOPS; these shapes reach ~50%, and the prefill is chunked.
FLOPS_EFFECTIVE = 150e12
#: HBM bandwidth of an A100-80GB: the decode is bandwidth-bound at batch 1.
BANDWIDTH = 1.55e12
DECODE_TOKENS = 96

#: Fitted to the dense `detect` anchor, with the compute term from the FLOPs model and
#: a 0.5 s prompt-build/scoring overhead assumed (the two are not separable from one
#: equation): (563/210 - 0.5 - 0.179) / 448.
CAPTURE_S_PER_HEADLAYER = 0.00447
DETECT_OVERHEAD_S = 0.50
#: Fitted to the hybrid `detect` anchor: the Gated DeltaNet layers run the reference
#: conv1d path (the job image cannot build `causal_conv1d`), so its prefill costs this
#: much more per active parameter than a dense model's.
HYBRID_PREFILL_SLOWDOWN = 4.5
#: Fitted to the two mask anchors: the dense one implies 2.05 s of overhead per pass
#: and the hybrid's implies 1.82 s, so this is the value that lands both within ~7%.
MASK_OVERHEAD_S = 1.90

#: (total params, active params, layers, Q-heads, scoreable layers, window, hybrid).
#: Architectures as verified in docs/model-shortlist.md.
MODELS: dict[str, tuple[float, float, int, int, int, int, bool]] = {
    "qwen3-0.6b":      (0.60e9, 0.60e9, 28, 16, 28, 40960, False),
    "qwen3.5-0.8b":    (0.87e9, 0.87e9, 24,  8,  6, 262144, True),
    "qwen3-4b":        (4.02e9, 4.02e9, 36, 32, 36, 40960, False),
    "qwen3-8b":        (8.19e9, 8.19e9, 36, 32, 36, 40960, False),
    "qwen3-30b-a3b":  (30.53e9, 3.30e9, 48, 32, 48, 40960, False),
    "qwen3.5-4b":      (4.66e9, 4.66e9, 32, 16,  8, 262144, True),
    "qwen3.5-9b":      (9.65e9, 9.65e9, 32, 16,  8, 262144, True),
    "qwen3.5-35b-a3b": (35.95e9, 3.00e9, 40, 16, 10, 262144, True),
    "ministral-3-8b":  (8.92e9, 8.92e9, 34, 32, 34, 262144, False),
    "ministral-3-14b": (13.95e9, 13.95e9, 40, 32, 40, 262144, False),
}

ANCHORS = {"qwen3-0.6b": (563, 3660), "qwen3.5-0.8b": (610, 4500)}


def load_driver():
    spec = importlib.util.spec_from_file_location(
        "datasphere_job", REPO_ROOT / "scripts" / "datasphere_job.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["datasphere_job"] = module
    spec.loader.exec_module(module)
    return module


def flag_values(argv: list[str], name: str) -> list[str]:
    """Values of a `nargs="*"` flag: everything up to the next `--flag`."""
    if name not in argv:
        return []
    out: list[str] = []
    for token in argv[argv.index(name) + 1:]:
        if token.startswith("--"):
            break
        out.append(token)
    return out


def grid(scale: str) -> dict:
    """The detection and masking grids a job on `scale` actually runs.

    `mask_flags` is kept raw because the K points have to come from the CLI's own
    `resolve_k` (see :func:`resolved_k_values`), and that needs the `--k-frac` list as
    it will be passed, not as floats re-derived here.
    """
    driver = load_driver()
    from retrieval_heads.cli import PROFILES

    if scale not in driver.SCALES:
        raise SystemExit(f"unknown scale {scale!r}; known: {sorted(driver.SCALES)}")
    detect = driver.SCALES[scale]["detect"]
    mask = driver.SCALES[scale]["mask"]
    # A `--profile` as the last token would otherwise raise IndexError, and `mask`
    # legitimately sets neither `--depths` nor `--needles` on the t4 scale (the CLI
    # defaults apply there).
    profiles = flag_values(detect, "--profile")
    profile = PROFILES[profiles[0]] if profiles else PROFILES["laptop"]

    def flag_int(argv: list[str], name: str, fallback: int) -> int:
        values = flag_values(argv, name)
        return int(values[0]) if values else fallback

    return {
        "detect_lengths": [int(v) for v in flag_values(detect, "--lengths")]
                          or profile["lengths"],
        "detect_depths": flag_int(detect, "--depths", profile["depths"]),
        "detect_needles": flag_int(detect, "--needles", profile["needles"]),
        "mask_lengths": [int(v) for v in flag_values(mask, "--lengths")],
        "mask_depths": flag_int(mask, "--depths", 5),
        "mask_needles": flag_int(mask, "--needles", 1),
        "k_frac": [float(v) for v in flag_values(mask, "--k-frac")],
        "trials": flag_int(mask, "--random-trials", 3),
        "mixer_trials": flag_int(mask, "--mixer-trials", 3),
        "mask_flags": mask,
    }


def compute_seconds(active: float, tokens: float) -> float:
    """Prefill + decode of one pass on `tokens` prompt tokens."""
    return (2 * active * tokens / FLOPS_EFFECTIVE
            + DECODE_TOKENS * 2 * active / BANDWIDTH)


def kept_lengths(window: int, lengths: list[int]) -> list[int]:
    """`within_context_limit` drops anything above `window - 64`."""
    return [length for length in lengths if length <= window - 64]


def detect_seconds(key: str, g: dict) -> tuple[float, int]:
    _total, active, _layers, heads, score_layers, window, hybrid = MODELS[key]
    lengths = kept_lengths(window, g["detect_lengths"])
    instances = len(lengths) * g["detect_depths"] * g["detect_needles"]
    mean_tokens = sum(lengths) / len(lengths)
    per_instance = (DETECT_OVERHEAD_S + CAPTURE_S_PER_HEADLAYER * score_layers * heads
                    + compute_seconds(active, mean_tokens)
                    * (HYBRID_PREFILL_SLOWDOWN if hybrid else 1.0))
    return instances * per_instance, instances


def resolved_k_values(g: dict, n_heads: int) -> list[int]:
    """The K points `mask` will really run, from the CLI's own resolver.

    Re-deriving them here is how an earlier version of this script priced three mixer
    K values on models that run two: the ablation is capped at ``K <= 4`` and the
    Qwen3.5-4B/9B (128 heads) and 35B-A3B (160 heads) grids resolve to `[1, 3]` and
    `[2, 3]`.  Calling `resolve_k` removes the whole class of drift, including the
    ``int(x + 0.5)`` rounding that `round()` gets wrong.
    """
    from types import SimpleNamespace

    from retrieval_heads.cli import resolve_k

    args = SimpleNamespace(k=None, k_frac=flag_values(g["mask_flags"], "--k-frac"))
    info = SimpleNamespace(n_scoreable_heads=n_heads, name="estimate")
    # `cmd_mask` passes `DEFAULT_K_FRACS["mask"]` when neither `--k` nor `--k-frac` is
    # given, so a scale that sets no fractions must be priced the same way rather than
    # as a single 5%-of-heads point.
    from retrieval_heads.cli import DEFAULT_K_FRACS

    return resolve_k(args, info, default_fracs=DEFAULT_K_FRACS["mask"])


def mixer_k_values(k_values: list[int]) -> list[int]:
    """The subset the token-mixer ablation uses -- `cli.cmd_mask`'s own rule."""
    return [k for k in k_values if k <= 4] or [1, 2]


def mask_seconds(key: str, g: dict, *, random_trials: int | None = None) -> tuple[float, int]:
    _total, active, _layers, heads, score_layers, _window, hybrid = MODELS[key]
    trials = g["trials"] if random_trials is None else random_trials
    samples = len(g["mask_lengths"]) * g["mask_depths"] * g["mask_needles"]
    k_values = resolved_k_values(g, score_layers * heads)
    k_points = len(k_values)
    passes = samples * (1 + k_points * (1 + trials))
    if hybrid:
        # `token_mixer_ablation` runs `len(mixer_k) x 2 kinds x n_trials x samples`.
        # The multiplier is `--mixer-trials` (CLI default 3), not "3 subsets": the
        # preflight sets it to 1, and pricing that as 3 overstated the preflight by 16%.
        passes += len(mixer_k_values(k_values)) * 2 * g["mixer_trials"] * samples
    mean_tokens = sum(g["mask_lengths"]) / len(g["mask_lengths"])
    per_pass = MASK_OVERHEAD_S + compute_seconds(active, mean_tokens)
    return passes * per_pass, passes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scale", default="a100")
    parser.add_argument("--random-trials", type=int, default=None,
                        help="override the scale's --random-trials")
    parser.add_argument("--price", type=float, default=542.88, help="RUB/hour, incl. VAT")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--check-anchors", action="store_true",
                        help="print the measured runs against the model")
    parser.add_argument("--only", choices=["both", "detect", "mask"], default="both",
                        help="price one stage (a preflight scale runs only one)")
    args = parser.parse_args(argv)
    # `resolve_k` logs a K-collapse warning per call, and this script calls it once per
    # model: the collapse is already visible in the `passes` column, and "on estimate"
    # in a log line is noise.
    logging.getLogger("retrieval_heads").setLevel(logging.ERROR)
    g = grid(args.scale)

    if args.check_anchors:
        print(f"grid: detect {len(g['detect_lengths'])} lengths x {g['detect_depths']} "
              f"depths x {g['detect_needles']} needles; mask {g['mask_lengths']} x "
              f"{g['mask_depths']} x {g['mask_needles']}")
        print("anchor check (measured vs modelled):")
        for key, (measured_detect, measured_mask) in ANCHORS.items():
            detect, _ = detect_seconds(key, g)
            mask, _ = mask_seconds(key, g)
            # Which rows are evidence and which are arithmetic.  `CAPTURE_S_PER_HEADLAYER`
            # is solved from the dense detect row and `HYBRID_PREFILL_SLOWDOWN` from the
            # hybrid's, so those two print `-0%` by construction; `MASK_OVERHEAD_S` is the
            # midpoint of the two mask rows' implied overheads, so neither mask row is
            # independent either.  Printing four percentages as if they validated the
            # model was the misleading part.
            tags = "  (detect = fit, mask = half of the overhead bracket)"
            print(f"  {key:14s} detect {measured_detect:>5.0f}s vs {detect:>5.0f}s "
                  f"({detect / measured_detect - 1:+.0%})   "
                  f"mask {measured_mask:>5.0f}s vs {mask:>5.0f}s "
                  f"({mask / measured_mask - 1:+.0%}){tags}")
        print("  -> all four rows are fitted, so none of them is an independent check;")
        print("     they show the fit is consistent, not that the model is right.")
        return 0

    models = args.models or list(MODELS)
    want_detect = args.only in ("both", "detect")
    want_mask = args.only in ("both", "mask")
    # The counters belong to their own stage: printing detect instances under a
    # mask-only run (or mask passes under a detect-only one) was a number for a stage the
    # table is not pricing.
    head = f"{'model':17s} {'score':>5s}"
    if want_detect:
        head += f" {'inst':>5s}"
    if want_mask:
        head += f" {'passes':>7s}"
    if want_detect:
        head += f" {'detect':>8s}"
    if want_mask:
        head += f" {'mask':>8s}"
    head += f" {'total':>7s} {'RUB':>6s}"
    print(head)
    total = 0.0
    for key in models:
        detect, instances = detect_seconds(key, g)
        mask, passes = mask_seconds(key, g, random_trials=args.random_trials)
        # A preflight scale runs one stage, so pricing the other would put a number in
        # the table that no job will ever spend.
        cost = (detect if want_detect else 0.0) + (mask if want_mask else 0.0)
        total += cost
        scoreable = MODELS[key][4] * MODELS[key][3]
        row = f"{key:17s} {scoreable:>5d}"
        if want_detect:
            row += f" {instances:>5d}"
        if want_mask:
            row += f" {passes:>7d}"
        if want_detect:
            row += f" {detect / 3600:>7.1f}h"
        if want_mask:
            row += f" {mask / 3600:>7.1f}h"
        row += f" {cost / 3600:>6.1f}h {cost / 3600 * args.price:>6.0f}"
        print(row)
    print(f"\n{'all':17s} {total / 3600:>6.1f}h {total / 3600 * args.price:>6.0f}"
          f"  (at {args.price} RUB/h)"
          + (f"  [{args.only} only]" if args.only != "both" else ""))
    trials = g["trials"] if args.random_trials is None else args.random_trials
    print(f"grid: detect {len(g['detect_lengths'])} lengths 1K-49K x "
          f"{g['detect_depths']} depths x {g['detect_needles']} needles; mask "
          f"{len(g['mask_lengths'])}x{g['mask_depths']}x{g['mask_needles']} = "
          f"{len(g['mask_lengths']) * g['mask_depths'] * g['mask_needles']} samples/point, "
          f"{trials} random trials, {DECODE_TOKENS}-token budget")
    print(f"calibration: capture {CAPTURE_S_PER_HEADLAYER}s per head-layer per detect "
          f"instance + {DETECT_OVERHEAD_S}s overhead; mask {MASK_OVERHEAD_S}s overhead per "
          f"pass; hybrid prefill x{HYBRID_PREFILL_SLOWDOWN} (detect only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
