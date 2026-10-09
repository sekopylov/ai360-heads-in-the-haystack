"""Figures of the paper, regenerated from measured data.

Every function takes measured objects (or the JSON written by the scripts) and
writes a PDF/PNG pair.  Matplotlib runs headless (Agg), because these are batch
artefacts, not interactive plots.

Figure map (paper name -> function):
    fig_retrieval_head              -> plot_masking_curve
    fig_retrieval_attention_dist    -> plot_attention_distribution
    fig_score_pie / ring_graph      -> plot_score_pie
    fig_score_distribution          -> plot_score_distribution
    fig_heat_map                    -> plot_heat_map
    fig_corr_map_masking_heads      -> plot_corr_map, plot_masking_curve
    fig_extractqa_head_case         -> plot_task_qa
    fig_task_cot / fig_case_cot     -> plot_task_cot
    token-mixer ablation            -> plot_mixer_ablation
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# Matplotlib wants a writable config dir; default to a repo-local one rather than
# warning and rebuilding the font cache in /tmp on every run.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent.parent / ".cache" / "matplotlib"),
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from retrieval_heads.properties import (  # noqa: E402
    CorrelationMatrix,
    activation_gap,
    category_fractions,
    layer_profile,
)
from retrieval_heads.scoring import RetrievalScores  # noqa: E402
from retrieval_heads.utils import ensure_dir, get_logger  # noqa: E402

log = get_logger("plotting")

#: The colour-blind-safe qualitative palette used throughout.
PALETTE = ("#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860")
NA_COLOUR = "#d9d9d9"

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def finish(fig: plt.Figure, title: str) -> plt.Figure:
    """Place a figure-level title without letting it overlap the panel titles.

    ``bbox_inches="tight"`` at save time does not know about a ``suptitle`` that
    was drawn over the axes, so the axes area has to be shrunk explicitly.
    """
    fig.suptitle(title, fontsize=10, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def save_fig(fig: plt.Figure, path: str | Path, *, formats: Sequence[str] = ("pdf", "png")) -> list[Path]:
    """Write ``path`` with each extension in ``formats``."""
    path = Path(path)
    ensure_dir(path.parent)
    written = []
    for fmt in formats:
        target = path.with_suffix(f".{fmt}")
        fig.savefig(target, bbox_inches="tight")
        written.append(target)
    plt.close(fig)
    log.info("figure -> %s", ", ".join(str(p) for p in written))
    return written


# --------------------------------------------------------------------------- Fig. 2 / ring
def plot_score_pie(scores_by_model: Mapping[str, RetrievalScores],
                   *, threshold: float | None = None) -> plt.Figure:
    """Fraction of retrieval heads per model -- the paper's ring graph.

    ``threshold=None`` (the default) uses each run's own ``scores.threshold``, so
    Fig. 2 agrees with the masking/QA experiments instead of hard-coding 0.1.
    """
    names = list(scores_by_model)
    n = len(names)
    if n == 0:
        raise ValueError("plot_score_pie needs at least one model")
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))
    thr_label = ("the run's own" if threshold is None else f"{threshold:g}")
    fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.7 * rows), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        scores = scores_by_model[name]
        thr = scores.threshold if threshold is None else threshold
        frac = category_fractions(scores, thr)
        values = [
            frac["retrieval"]["frac"],
            frac["low"]["frac"],
            frac["zero"]["frac"],
        ]
        labels = [
            f"retrieval >{thr}\n{frac['retrieval']['n']} ({values[0] * 100:.1f}%)",
            f"weak (0, {thr}]\n{frac['low']['n']} ({values[1] * 100:.1f}%)",
            f"zero\n{frac['zero']['n']} ({values[2] * 100:.1f}%)",
        ]
        if sum(values) <= 0:
            # matplotlib rejects an all-zero pie; say so instead of raising.
            ax.pie([1.0], labels=["no finite scores"], colors=[PALETTE[0]],
                   startangle=90, counterclock=False,
                   wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 1.2},
                   textprops={"fontsize": 7})
        else:
            ax.pie(values, labels=labels, colors=(PALETTE[3], PALETTE[1], PALETTE[0]),
                   startangle=90, counterclock=False,
                   wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 1.2},
                   textprops={"fontsize": 7})
        title = f"{name}\n{scores.info.n_scoreable_heads} scoreable heads"
        if scores.info.is_hybrid:
            title += f" / {len(scores.info.linear_layers)} linear layers"
        ax.set_title(title, fontsize=8)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    # Descriptive: the hybrid run has 62.5% of its scoreable heads above 0.1, so
    # "universal and sparse" is exactly the claim the data can contradict.
    return finish(fig, f"Head score buckets per model (retrieval bucket >{thr_label})")


# --------------------------------------------------------------------------- Fig. 3
def plot_score_distribution(scores_by_model: Mapping[str, RetrievalScores],
                            *, top_k: int = 40) -> plt.Figure:
    """Retrieval score vs activation frequency for the strongest heads."""
    names = list(scores_by_model)
    if not names:
        raise ValueError("plot_score_distribution needs at least one model")
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.2), squeeze=False)
    for ax, name in zip(axes[0], names):
        gap = activation_gap(scores_by_model[name], top_k=top_k)
        x = np.arange(len(gap["score"]))
        ax.plot(x, gap["score"], "-o", ms=3, color=PALETTE[0], label="retrieval score")
        ax.plot(x, gap["activation_freq"], "-s", ms=3, color=PALETTE[2],
                label="activation frequency")
        ax.fill_between(x, gap["score"], gap["activation_freq"], color=PALETTE[2], alpha=0.12)
        n_always = len(gap["always_active"])
        ax.set_title(f"{name}\nalways-active heads: {n_always}")
        ax.set_xlabel("head rank")
        ax.set_ylabel("value")
        ax.set_ylim(-0.03, 1.03)
        ax.legend(loc="lower left")
    # Descriptive: activation frequency is E[1(per-instance score > 0)], i.e. the
    # same score reduced to an indicator, so the two curves are not independent.
    return finish(fig, "Score vs activation frequency (the same score as an indicator)")


# --------------------------------------------------------------------------- Fig. 5 heat map
def plot_heat_map(scores_by_model: Mapping[str, RetrievalScores],
                  *, labels: Sequence[str] | None = None) -> plt.Figure:
    """Layer x head retrieval-score heatmaps, NaN (non-scoreable) shown as grey."""
    if not scores_by_model:
        # `max()` over an empty generator is a ValueError; callers guard today, but
        # the function should not depend on that.
        raise ValueError("plot_heat_map needs at least one run")
    names = list(scores_by_model)
    labels = list(labels) if labels else names
    shape = (max(s.info.num_layers for s in scores_by_model.values()),
             max(s.info.max_heads for s in scores_by_model.values()))
    cols = min(len(names), 3)
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols + 1, 3.0 * rows), squeeze=False)
    for ax, name, label in zip(axes.ravel(), names, labels):
        info = scores_by_model[name].info
        grid = np.full(shape, np.nan)
        grid[: info.num_layers, : info.max_heads] = scores_by_model[name].score.numpy()
        masked = np.ma.masked_invalid(grid)
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(NA_COLOUR)
        im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0,
                       interpolation="nearest")
        ax.set_title(f"{label}\n{len(info.scoreable_layers)}/{info.num_layers} scoreable layers")
        ax.set_xlabel("head index")
        ax.set_ylabel("layer")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    return finish(fig, "Retrieval score by layer and head (grey = no attention map)")


def plot_layer_profile(scores_by_model: Mapping[str, RetrievalScores]) -> plt.Figure:
    """Retrieval mass per layer -- makes the hybrid layout visible."""
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    width = 0.8 / max(len(scores_by_model), 1)
    for i, (name, scores) in enumerate(scores_by_model.items()):
        prof = layer_profile(scores)["layers"]
        layers = [row["layer"] for row in prof]
        means = [row["mean_score"] for row in prof]
        ax.bar(np.array(layers) + i * width, means, width=width, label=name,
               color=PALETTE[i % len(PALETTE)])
    ax.set_xlabel("layer index")
    ax.set_ylabel("mean retrieval score")
    ax.set_title("Retrieval mass by layer")
    ax.legend()
    return finish(fig, "Retrieval mass by layer, per model")


# --------------------------------------------------------------------------- Fig. 5 corr map
def plot_corr_map(corr: CorrelationMatrix | Mapping[str, Any]) -> plt.Figure:
    """Pearson correlation between models' retrieval-score distributions."""
    caveat = None
    if isinstance(corr, CorrelationMatrix):
        labels, values, mode = corr.labels, np.array(corr.values), corr.mode
        caveat = getattr(corr, "caveat", None)
    else:
        # dtype=float: a JSON artifact writes non-finite entries as `null`, which
        # numpy otherwise reads as an object array and isfinite() then rejects.
        labels = corr["labels"]
        values = np.array(corr["values"], dtype=float)
        mode = corr.get("mode", "grid")
        caveat = corr.get("caveat")
    if not labels:
        raise ValueError("plot_corr_map needs at least one label")
    if not np.any(np.isfinite(values)):
        log.warning("plot_corr_map: every correlation is non-finite; the figure will be "
                    "blank (too few shared finite entries, or constant scores)")
    fig, ax = plt.subplots(figsize=(1.1 + 0.85 * len(labels), 1.1 + 0.75 * len(labels)))
    im = ax.imshow(values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = values[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(value) > 0.55 else "black")
    ax.set_title(f"Retrieval-score correlation ({mode})", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    if caveat:
        fig.text(0.01, 0.005, caveat, fontsize=6, color="grey", wrap=True)
    return finish(fig, "Retrieval-score correlation between models")


# --------------------------------------------------------------------------- Fig. 1 masking
def plot_masking_curve(curves: Mapping[str, Any],
                       *, metric: str = "f1") -> plt.Figure:
    """Needle-in-a-Haystack score as top-K retrieval heads / K random heads go away.

    ``metric`` picks which series is drawn: ``"f1"`` (the default) or
    ``"exact_match"``.  It used to change only the axis label while always
    plotting f1.
    """
    if metric not in {"f1", "exact_match", "recall"}:
        raise ValueError(
            f"metric must be 'f1', 'exact_match' or 'recall', got {metric!r}"
        )
    capped: list[str] = []
    fig, axes = plt.subplots(1, max(len(curves), 1),
                             figsize=(4.4 * max(len(curves), 1), 3.2), squeeze=False)
    for ax, (name, curve) in zip(axes[0], curves.items()):
        data = curve.as_dict() if hasattr(curve, "as_dict") else curve
        # Plot against the K that was actually masked when the artifact records it;
        # otherwise a capped K would be drawn at its requested position.
        k = data.get("k_effective") or data["k_values"]
        if metric == "recall":
            top, rand = data.get("retrieval_recall"), data.get("random_recall_mean")
            baseline = data.get("baseline_recall", 0.0)
            top_yerr = None
            rand_yerr = None
            if not top or not rand:
                raise ValueError(
                    f"{name}: this artifact has no recall series; re-run `mask`"
                )
        elif metric == "exact_match":
            top = data.get("retrieval_exact_match")
            rand = data.get("random_exact_match_mean")
            if not top or not rand:
                # `or data["retrieval"]` used to fall back to the F1 series while the
                # axis still said "exact_match".
                raise ValueError(
                    f"{name}: this artifact has no exact-match series; it predates the "
                    f"column, or the run was made before the metric was split. "
                    f"Re-run `mask` or plot metric='f1'."
                )
            baseline = data.get("baseline_exact_match", data.get("baseline", 0.0))
            top_yerr = data.get("retrieval_exact_std")
            rand_yerr = None  # no per-trial exact-match std is stored
        else:
            top, rand = data.get("retrieval"), data.get("random_mean")
            baseline = data.get("baseline", 0.0)
            top_yerr = data.get("retrieval_std")
            rand_yerr = data.get("random_std")
        # The length check must come *after* the assignments (an earlier edit put the
        # raise first, which turned them into dead code and broke the default path).
        if not top or not rand or len(top) != len(k) or len(rand) != len(k):
            raise ValueError(
                f"{name}: the artifact's series have inconsistent lengths "
                f"(k={len(k)}, retrieval={len(top or [])}, random={len(rand or [])})"
            )
        # The two error bars are different quantities: retrieval is the spread
        # across evaluation samples, random is the spread across random trials.
        ax.errorbar(k, top, yerr=top_yerr, fmt="-o", color=PALETTE[3], capsize=3,
                    label="top-K retrieval heads (bar: across eval samples)")
        ax.errorbar(k, rand, yerr=rand_yerr, fmt="-s",
                    color=PALETTE[0], capsize=3,
                    label="K random heads (bar: across random trials)")
        ax.axhline(baseline, ls=":", color="grey", lw=1, label="no masking")
        requested = data.get("k_values") or []
        if requested and list(requested) != list(k):
            capped.append(f"{name}: requested K={list(requested)} -> masked K_eff={list(k)}")
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlabel("heads masked (K)")
        ax.set_ylabel(f"NIAH {metric} (%)")
        ax.set_title(name)
        ax.set_ylim(-3, 103)
        ax.legend(loc="lower left")
    # Descriptive, not a claim: on the dense model the random arm also collapses
    # once K approaches the whole head budget, so a stronger title would
    # contradict the panel next to it.
    if capped:
        # The figure hid the K cap: a requested K=64 masked as 18 looked identical
        # to a real K=18 run.
        fig.text(0.01, 0.005, " | ".join(capped), fontsize=6, color="grey")

    return finish(fig, "Needle-in-a-Haystack after masking top-K retrieval heads "
                        "vs K random heads")


def plot_mixer_ablation(ablations: Mapping[str, Any]) -> plt.Figure:
    """Full-attention layers vs linear-attention layers silenced (hybrid models).

    Draws the last point of each sweep, and says which K that is: taking ``[-1]``
    without a label let two models be compared at different K values silently.
    """
    k_sets = {tuple(a.get("k_values") or []) for a in ablations.values()}
    if len(k_sets) > 1:
        raise ValueError(f"ablations were run at different K values: {sorted(k_sets)}")
    k_values = next(iter(k_sets), ())
    k = k_values[-1] if k_values else None

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    names = list(ablations)
    width = 0.35
    x = np.arange(len(names))
    full = [ablations[n]["full_attention"][-1] if ablations[n]["full_attention"] else np.nan
            for n in names]
    linear = [ablations[n]["linear_attention"][-1] if ablations[n]["linear_attention"] else np.nan
              for n in names]
    base = [ablations[n]["baseline"] for n in names]
    full_err = [ablations[n].get("full_attention_std") or [] for n in names]
    linear_err = [ablations[n].get("linear_attention_std") or [] for n in names]
    full_yerr = [err[-1] if err else 0.0 for err in full_err]
    linear_yerr = [err[-1] if err else 0.0 for err in linear_err]
    # Bars centred on the tick: the previous offsets put the label off the group.
    ax.bar(x - width, base, width, label="baseline", color=PALETTE[0])
    ax.bar(x, full, width, yerr=full_yerr, capsize=3,
           label="full-attention layers masked", color=PALETTE[3])
    if any(np.isfinite(linear)):
        ax.bar(x + width, linear, width, yerr=linear_yerr, capsize=3,
               label="linear layers masked", color=PALETTE[2])
    labels_with_counts = []
    for i, name in enumerate(names):
        full_n = (ablations[name].get("full_attention_masked") or [None])[-1]
        linear_n = (ablations[name].get("linear_attention_masked") or [None])[-1]
        if linear_n is not None:
            labels_with_counts.append(f"{name}\n({full_n} full / {linear_n} linear masked)")
        else:
            labels_with_counts.append(f"{name}\n({full_n} full masked)")
    ax.set_xticks(x, labels_with_counts, fontsize=7)
    ax.set_ylabel("NIAH F1 (%)")
    ax.set_title(f"Whole token-mixer layers silenced (K={k})" if k is not None
                 else "Whole token-mixer layers silenced")
    ax.legend()
    return finish(fig, "Does retrieval need full attention? (maximum masked K per run)")


# --------------------------------------------------------------------------- Fig. 1 attention
def plot_attention_distribution(
    distributions: Mapping[str, tuple[np.ndarray, tuple[int, int]]],
    *,
    top_n: int = 3,
) -> plt.Figure:
    """Attention rows of the strongest heads, with the needle span shaded.

    ``distributions`` maps a label to ``(attention_row, needle_span)`` where the
    row is a 1-D array over input positions.  This is the paper's argument in one
    picture: the head's mass sits exactly on the needle.
    """
    fig, axes = plt.subplots(len(distributions), 1,
                             figsize=(7.0, 1.7 * len(distributions)), squeeze=False)
    # `axes` is (n, 1) with squeeze=False, so the panels are `axes[:, 0]`.
    # Iterating `axes[0]` (the first *row*) drew only the first panel and left the
    # rest of Fig. 1 blank.
    for ax, (label, (row, span)) in zip(axes[:, 0], distributions.items()):
        row = np.asarray(row, dtype=float)
        ax.fill_between(np.arange(len(row)), row, color=PALETTE[0], alpha=0.75, lw=0)
        ax.axvspan(span[0], span[1] - 1, color=PALETTE[3], alpha=0.18,
                   label=f"needle [{span[0]}, {span[1]})")
        ax.set_xlim(0, max(len(row) - 1, 1))
        ax.set_ylabel("attn")
        ax.set_title(label, fontsize=8, loc="left")
        ax.legend(loc="upper left", fontsize=7)
    return finish(fig, "A retrieval head puts its mass on the needle token it is copying")


# --------------------------------------------------------------------------- Fig. 4 QA / CoT
def plot_task_qa(result: Mapping[str, Any]) -> plt.Figure:
    """Extractive QA F1 with retrieval heads vs random heads masked."""
    by_k = result["by_k"]
    ks = list(by_k)
    x = np.arange(len(ks))
    width = 0.35
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.bar(x - width / 2, [by_k[k]["retrieval_f1"] for k in ks], width,
           label="retrieval heads masked", color=PALETTE[3])
    ax.bar(x + width / 2, [by_k[k]["random_f1_mean"] for k in ks], width,
           yerr=[by_k[k].get("random_f1_std", 0.0) for k in ks], capsize=3,
           label="random heads masked", color=PALETTE[0])
    ax.axhline(result["baseline_f1"], ls=":", color="grey", lw=1, label="no masking")
    # Label the heads actually masked: a capped K would otherwise be shown at its
    # requested value (e.g. "K=64" when only 18 heads were removed).
    ax.set_xticks(x, [f"K={k}" if by_k[k].get("k_effective", k) == int(k)
                      else f"K={k}→{by_k[k]['k_effective']}" for k in ks])
    ax.set_ylabel("Extractive QA F1 (%)")
    ax.set_ylim(0, 103)
    # The README says the 8-sample measurement cannot separate the arms cleanly, so
    # the title must not assert the conclusion.
    ax.set_title("Extractive QA F1 with and without masking")
    ax.legend()
    return finish(fig, "Extractive QA: retrieval-masked vs random-masked arms")


def plot_task_cot(result: Mapping[str, Any]) -> plt.Figure:
    """CoT vs answer-only, with retrieval or random heads masked."""
    variants = list(result["results"])
    x = np.arange(len(variants))
    width = 0.26
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.bar(x - width, [result["results"][v]["baseline"] for v in variants], width,
           label="no masking", color=PALETTE[0])
    ax.bar(x, [result["results"][v]["retrieval_masked"] for v in variants], width,
           label="retrieval heads masked", color=PALETTE[3])
    ax.bar(x + width, [result["results"][v]["random_masked_mean"] for v in variants], width,
           yerr=[result["results"][v]["random_masked_std"] for v in variants], capsize=3,
           label="random heads masked", color=PALETTE[2])
    ax.set_xticks(x, [v.replace("_", " ") for v in variants])
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 103)
    # Symmetric with plot_task_qa: the README says the answer-only half also drops,
    # so the title must not assert the conclusion.
    ax.set_title("Chain-of-thought vs answer-only under masking")
    ax.legend()
    return finish(fig, "Chain-of-thought vs answer-only with retrieval heads masked")
