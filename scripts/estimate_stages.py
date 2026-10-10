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
  the token-mixer ablation for a hybrid (3 K x 2 kinds x 3 subsets).

Two anchors (below) fix the constants, and that is also the model's weakness: with two
small models, the split between per-pass *overhead* and per-pass *compute* is
**underdetermined** -- the hybrid's mask pass is cheaper than the dense model's despite
1.4x the parameters, so the two anchors bracket the overhead rather than pinning it.
`MASK_OVERHEAD_S` is therefore the midpoint of that bracket, and the table is a plan:
the first hour of a real run is what confirms it.  `--check-anchors` prints the fit.

Anchors (A100 80 GB, bf16, 45 samples/point, 96-token budget):
    qwen3-0.6b    detect 563 s / 210 instances    mask 3660 s = 1665 passes
    qwen3.5-0.8b  detect 610 s / 270 instances    mask 4500 s = 2205 passes
"""

from __future__ import annotations

import argparse
import importlib.util
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
    """The detection and masking grids a job on `scale` actually runs."""
    driver = load_driver()
    from retrieval_heads.cli import PROFILES

    detect = driver.SCALES[scale]["detect"]
    mask = driver.SCALES[scale]["mask"]
    profile = PROFILES[flag_values(detect, "--profile")[0] or "laptop"]
    return {
        "detect_lengths": [int(v) for v in flag_values(detect, "--lengths")]
                          or profile["lengths"],
        "detect_depths": int(flag_values(detect, "--depths")[0]) if flag_values(
            detect, "--depths") else profile["depths"],
        "detect_needles": int(flag_values(detect, "--needles")[0]) if flag_values(
            detect, "--needles") else profile["needles"],
        "mask_lengths": [int(v) for v in flag_values(mask, "--lengths")],
        "mask_depths": int(flag_values(mask, "--depths")[0]),
        "mask_needles": int(flag_values(mask, "--needles")[0]),
        "k_frac": [float(v) for v in flag_values(mask, "--k-frac")],
        "trials": int(flag_values(mask, "--random-trials")[0]),
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


def mask_seconds(key: str, g: dict, *, random_trials: int | None = None) -> tuple[float, int]:
    _total, active, _layers, heads, score_layers, _window, hybrid = MODELS[key]
    trials = g["trials"] if random_trials is None else random_trials
    samples = len(g["mask_lengths"]) * g["mask_depths"] * g["mask_needles"]
    k_points = len({max(1, round(frac * score_layers * heads)) for frac in g["k_frac"]})
    passes = samples * (1 + k_points * (1 + trials))
    if hybrid:
        passes += 3 * 2 * 3 * samples          # the K<=4 token-mixer ablation
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
    args = parser.parse_args(argv)
    g = grid(args.scale)

    if args.check_anchors:
        print(f"grid: detect {len(g['detect_lengths'])} lengths x {g['detect_depths']} "
              f"depths x {g['detect_needles']} needles; mask {g['mask_lengths']} x "
              f"{g['mask_depths']} x {g['mask_needles']}")
        print("anchor check (measured vs modelled):")
        for key, (measured_detect, measured_mask) in ANCHORS.items():
            detect, _ = detect_seconds(key, g)
            mask, _ = mask_seconds(key, g)
            print(f"  {key:14s} detect {measured_detect:>5.0f}s vs {detect:>5.0f}s "
                  f"({detect / measured_detect - 1:+.0%})   "
                  f"mask {measured_mask:>5.0f}s vs {mask:>5.0f}s "
                  f"({mask / measured_mask - 1:+.0%})")
        return 0

    models = args.models or list(MODELS)
    print(f"{'model':17s} {'inst':>4s} {'score':>5s} {'passes':>7s} {'detect':>8s} "
          f"{'mask':>8s} {'total':>7s} {'RUB':>6s}")
    total = 0.0
    for key in models:
        detect, instances = detect_seconds(key, g)
        mask, passes = mask_seconds(key, g, random_trials=args.random_trials)
        total += detect + mask
        scoreable = MODELS[key][4] * MODELS[key][3]
        print(f"{key:17s} {instances:>4d} {scoreable:>5d} {passes:>7d} "
              f"{detect / 3600:>7.1f}h {mask / 3600:>7.1f}h {(detect + mask) / 3600:>6.1f}h "
              f"{(detect + mask) / 3600 * args.price:>6.0f}")
    print(f"\n{'all':17s} {'':>4s} {'':>5s} {'':>7s} {'':>8s} {'':>8s} {total / 3600:>6.1f}h "
          f"{total / 3600 * args.price:>6.0f}  (at {args.price} RUB/h)")
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
