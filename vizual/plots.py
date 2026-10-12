"""
Figures over the tables of vizual.data. Every function draws into ax (or makes its own figure) and returns the axes;
colours and sizes come from vizual.style only.

Even scales: the grids of the runs are not round (lengths 1000, 2526, 4053, ...; depths 0, 11, 22, ...). The data
stay at their real positions and the ticks stand at round values (5k, 10k, ...; 0, 20, ...), see round_axis();
where a figure is made of blocks, the labels of the blocks are rounded or the samples are put into round bins.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from . import style
from .data import ANSWER_TYPES, NGRAMS, ROOT, round_bins, round_label, spearman, success_column, type_column

VALUE_LABELS = {"correct": "верные ответы, %", "facts_found": "доля названных фактов", "rouge1_recall": "ROUGE-1 recall", "rouge2_recall": "ROUGE-2 recall", "rouge3_recall": "ROUGE-3 recall",
                "rougeL_recall": "ROUGE-L recall", "success": "успех по ROUGE-1, %", "success2": "успех по ROUGE-2, %",
                "success3": "успех по ROUGE-3, %",
                "exact": "ответы с точной фразой, %", "n_tokens": "длина ответа, токены", "seconds": "время ответа, с"}
AXIS_LABELS = {"context_length": "длина контекста, токены", "depth_percent": "глубина иглы, %"}
TYPE_LABELS = {"exact": "точная фраза", "full": "успех без точной фразы", "partial": "неполный ответ", "miss": "нет совпадений"}

thousands = FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v else "0")


def new_ax(ax, width=6.4, height=4.0):
    return ax if ax is not None else plt.subplots(figsize=style.size(width, height))[1]


def numeric(df, value):
    """The column as numbers; a yes/no column becomes percent."""
    column = df[value]
    return column.astype(float) * 100 if column.dtype == bool else column.astype(float)


def round_axis(ax, which, lo, hi, step, fmt=None):
    """Ticks at the multiples of step that fall into [lo, hi]; the limits are left to the data."""
    ticks = np.arange(np.ceil(lo / step) * step, hi + step * 1e-9, step)
    axis = ax.xaxis if which == "x" else ax.yaxis
    axis.set_ticks(ticks)
    if fmt is not None:
        axis.set_major_formatter(fmt)


def titles(ax, title=None, subtitle=None, lift=0):
    """The title and, under it, the subtitle; lift: points left free above the axes, e.g. for a top axis."""
    t = style.THEME
    if subtitle:
        ax.annotate(subtitle, (0, 1), xycoords="axes fraction", xytext=(0, lift + 6), textcoords="offset points", ha="left", va="bottom",
                    fontsize=t.font - 1, color=t.ink_secondary)
    if title:
        ax.set_title(title, loc="left", pad=lift + 8 + (1.6 * t.font if subtitle else 0))


def save(fig, name, formats=("png",), folder="vizual/figures"):
    """Writes the figure into vizual/figures (from the repository root) and returns the paths."""
    folder = os.path.join(ROOT, folder)
    os.makedirs(folder, exist_ok=True)
    paths = [os.path.join(folder, f"{name}.{ext}") for ext in formats]
    for p in paths:
        getattr(fig, "figure", fig).savefig(p)
    return paths


# ---------------------------------------------------------------------------------------------------------------
# answers over context length and needle depth
# ---------------------------------------------------------------------------------------------------------------

def edges(centers):
    """Borders of the cells whose centres are the real grid values."""
    centers = np.asarray(centers, dtype=float)
    if len(centers) == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5])
    mid = (centers[:-1] + centers[1:]) / 2
    return np.concatenate([[2 * centers[0] - mid[0]], mid, [2 * centers[-1] - mid[-1]]])


def niah_heatmap(df, value="rouge1_recall", ax=None, labels="round", len_step=5000, depth_step=20, vmin=0, vmax=100,
                 colorbar=True, title=None, subtitle=None):
    """
    The needle-in-a-haystack picture: the mean of value over context length (x) and needle depth (y, 0 at the top).
        labels="round"  cells at the real grid values, ticks at the round ones (every len_step, depth_step)
        labels="cells"  a label under the cells themselves, the length a little rounded (2526 -> 2.5k)
        labels="bins"   the samples are put into round bins of len_step x depth_step first: fewer, larger blocks
    """
    t = style.THEME
    ax = new_ax(ax, 7.2, 3.4)
    data = df.assign(_v=numeric(df, value))
    if labels == "bins":
        data = data.assign(_x=round_bins(data["context_length"], len_step, "k"), _y=round_bins(data["depth_percent"], depth_step))
        grid = data.pivot_table(index="_y", columns="_x", values="_v", aggfunc="mean", observed=False)
    else:
        grid = data.pivot_table(index="depth_percent", columns="context_length", values="_v", aggfunc="mean")
    gap = dict(edgecolors=t.surface, linewidth=1.0 * t.scale)
    if labels == "round":
        mesh = ax.pcolormesh(edges(grid.columns), edges(grid.index), grid.to_numpy(), cmap=style.cmap(t.cmap_score), vmin=vmin, vmax=vmax, **gap)
        round_axis(ax, "x", grid.columns.min(), grid.columns.max(), len_step, thousands)
        round_axis(ax, "y", grid.index.min(), grid.index.max(), depth_step)
    else:
        nx, ny = grid.shape[1], grid.shape[0]
        mesh = ax.pcolormesh(np.arange(nx + 1), np.arange(ny + 1), grid.to_numpy(), cmap=style.cmap(t.cmap_score), vmin=vmin, vmax=vmax, **gap)
        every = 2 if nx > 12 else 1
        names_x = list(grid.columns) if labels == "bins" else [round_label(v) for v in grid.columns]
        names_y = list(grid.index) if labels == "bins" else [f"{v:g}" for v in grid.index]
        ax.set_xticks(np.arange(nx)[::every] + 0.5, names_x[::every])
        ax.set_yticks(np.arange(ny) + 0.5, names_y)
    low, high = sorted(ax.get_ylim())
    ax.set_ylim(high, low)   # depth 0 on top; set, not inverted: the panels of niah_grid share the axis
    ax.grid(False)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xlabel(AXIS_LABELS["context_length"])
    ax.set_ylabel(AXIS_LABELS["depth_percent"])
    if colorbar:
        bar = ax.figure.colorbar(mesh, ax=ax, pad=0.02, aspect=18)
        bar.set_label(VALUE_LABELS.get(value, value))
        bar.outline.set_visible(False)
    titles(ax, title, subtitle)
    return ax


def niah_grid(sweep, runs, value="rouge1_recall", columns=3, labels="round", title=None, **kwargs):
    """Small multiples of niah_heatmap. runs: {panel title: name in the column run}, drawn in this order, one colour scale."""
    names = list(runs)
    rows = int(np.ceil(len(names) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=style.size(4.4 * columns, 2.9 * rows + 0.6), sharex=True, sharey=True, squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        part = sweep[sweep["run"] == runs[name]]
        niah_heatmap(part, value, ax=ax, labels=labels, colorbar=False, **kwargs)
        ax.set_title(f"{name} · {numeric(part, value).mean():.0f}", loc="left", fontsize=style.THEME.font + 1)
    for i, ax in enumerate(axes.ravel()):
        ax.set_visible(i < len(names))
        if i % columns:
            ax.set_ylabel("")
        if i + columns < len(names):
            ax.set_xlabel("")
        else:
            ax.tick_params(labelbottom=True)   # the last panel of its column, also when the row below is shorter
    bar = fig.colorbar(axes[0, 0].collections[0], ax=axes, pad=0.015, aspect=14 * rows, shrink=0.9)
    bar.set_label(VALUE_LABELS.get(value, value))
    bar.outline.set_visible(False)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    return axes


def score_by(sweep, x="context_length", kind="top", value="rouge1_recall", step=None, bins=None, ax=None, title=None, subtitle=None):
    """
    The mean of value along context length or needle depth, a line per number of masked heads of one kind of mask;
    the run without a mask is the reference. The points stand at the real grid values under round ticks (step);
    with bins the samples are grouped into round bins of this width instead.
    """
    t = style.THEME
    ax = new_ax(ax, 7.2, 4.0)
    step = step or (5000 if x == "context_length" else 20)
    data = sweep[sweep["mask"].isin([kind, "none"])].assign(_v=numeric(sweep, value))
    counts = sorted(data.loc[data["mask"] == kind, "heads"].unique())
    colors = style.ordered(len(counts))
    if bins:
        data = data.assign(_x=round_bins(data[x], bins, "k" if x == "context_length" else ""))
    key = "_x" if bins else x
    series = [("none", 0, t.colors["none"])] + [(kind, k, c) for k, c in zip(counts, colors)]
    for mask, k, color in series:
        line = data[(data["mask"] == mask) & (data["heads"] == k)].groupby(key, observed=False)["_v"].mean()
        xs = np.arange(len(line)) if bins else line.index.to_numpy()
        ax.plot(xs, line.to_numpy(), color=color, marker="o", label="без маски" if mask == "none" else f"{k} голов",
                linestyle=(0, (1, 1.5)) if mask == "none" else "-", zorder=3 if mask == "none" else 2)
    if bins:
        ax.set_xticks(np.arange(len(line)), list(line.index))
    else:
        round_axis(ax, "x", data[x].min(), data[x].max(), step, thousands if x == "context_length" else None)
    ax.set_xlabel(AXIS_LABELS.get(x, x))
    ax.set_ylabel(VALUE_LABELS.get(value, value))
    if data["_v"].max() <= 100:
        ax.set_ylim(-3, 103)
        round_axis(ax, "y", 0, 100, 20)
    ax.legend(title=f"маска: {style.LABELS[kind]}", loc="center left", bbox_to_anchor=(1.01, 0.5), alignment="left")
    titles(ax, title, subtitle)
    return ax


# ---------------------------------------------------------------------------------------------------------------
# masking
# ---------------------------------------------------------------------------------------------------------------

def mask_curve(sweep, value="rouge1_recall", kinds=("top", "random", "random_groups"), total_heads=None, threshold=None,
               band=True, ax=None, title=None, subtitle=None):
    """
    The paper's masking figure: the mean of value against the number of masked heads, a line per kind of mask. Every
    line starts at the run without a mask. band: 95% interval of the mean over the samples. total_heads adds the
    share of all heads to the label of the axis; threshold draws a reference level (50 is the success border of ROUGE-1).
    """
    t = style.THEME
    ax = new_ax(ax, 6.8, 4.2)
    data = sweep.assign(_v=numeric(sweep, value))
    stats = data.groupby(["mask", "heads"])["_v"].agg(["mean", "sem"]).reset_index()
    base = stats[stats["mask"] == "none"]
    most = stats["heads"].max()
    for kind in kinds:
        line = pd.concat([base, stats[stats["mask"] == kind]]).sort_values("heads")
        if len(line) <= len(base):
            continue
        color = t.colors[kind]
        if band:
            ax.fill_between(line["heads"], line["mean"] - 1.96 * line["sem"].fillna(0), line["mean"] + 1.96 * line["sem"].fillna(0),
                            color=color, alpha=0.14, linewidth=0)
        ax.plot(line["heads"], line["mean"], color=color, marker="o", label=style.LABELS[kind], zorder=3)
    if threshold is not None:
        ax.axhline(threshold, color=t.axis, linewidth=1, zorder=1)
        ax.annotate(f"{threshold:g}", (0, threshold), xytext=(4, 3), textcoords="offset points", color=t.ink_muted, fontsize=t.font - 1)
    round_axis(ax, "x", 0, most, 20)
    ax.set_xlim(-most * 0.03, most * 1.03)
    if stats["mean"].max() <= 100:
        ax.set_ylim(-3, 103)
        round_axis(ax, "y", 0, 100, 20)
    ax.set_xlabel("замаскировано голов" + (f" (из {total_heads}; {most} — это {most / total_heads:.1%})" if total_heads else ""))
    ax.set_ylabel(VALUE_LABELS.get(value, value))
    ax.legend(loc="best")
    titles(ax, title, subtitle)
    return ax


def criteria_curve(sweep, kind="top", ax=None, title=None, subtitle=None):
    """
    The share of the answers that pass every criterion, against the number of masked heads of one kind: success by
    ROUGE-1, ROUGE-2, ROUGE-3 (recall above 50) and the exact phrase, from the mildest to the strictest.
    """
    t = style.THEME
    ax = new_ax(ax, 7.6, 4.4)
    data = sweep[sweep["mask"].isin([kind, "none"])]
    columns = [success_column(n) for n in NGRAMS] + ["exact"]
    names = [f"ROUGE-{n} recall > 50" for n in NGRAMS] + ["точная фраза"]
    share = data.groupby("heads")[columns].mean() * 100
    for column, name, color in zip(columns, names, style.ordered(len(columns))):
        ax.plot(share.index, share[column], color=color, marker="o", label=name)
    most = share.index.max()
    round_axis(ax, "x", 0, most, 20)
    ax.set_xlim(-most * 0.03, most * 1.03)
    ax.set_ylim(-3, 103)
    round_axis(ax, "y", 0, 100, 20)
    ax.set_xlabel(f"замаскировано голов ({style.LABELS[kind]})")
    ax.set_ylabel("доля ответов, прошедших критерий, %")
    ax.legend(title="критерий", loc="center left", bbox_to_anchor=(1.01, 0.5), alignment="left")
    titles(ax, title, subtitle)
    return ax


def stacked(df, x, column, order, labels, xlabel=None, legend=True, ax=None, title=None, subtitle=None):
    """
    Stacked shares of the values of a column for every value of x (the number of masked heads, the level, ...).
    order: the values from the best to the worst, the last one is drawn as the neutral rest; labels: {value: name}.
    """
    t = style.THEME
    ax = new_ax(ax, 6.8, 4.0)
    share = (df.groupby(x)[column].value_counts(normalize=True).unstack(fill_value=0) * 100).reindex(columns=order, fill_value=0)
    colors = dict(zip(order, list(reversed(style.ordered(len(order) - 1))) + [t.colors["muted"]]))
    xs, bottom = np.arange(len(share)), np.zeros(len(share))
    for name in order:
        ax.bar(xs, share[name], bottom=bottom, width=0.62, color=colors[name], edgecolor=t.surface, linewidth=1.5, label=labels[name])
        bottom += share[name].to_numpy()
    ax.set_xticks(xs, [str(k) for k in share.index])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 100)
    round_axis(ax, "y", 0, 100, 20)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel("доля ответов, %")
    if legend:
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), reverse=True)
    titles(ax, title, subtitle)
    return ax


def shares(sweep, column, order, labels, kind="top", **kwargs):
    """stacked() against the number of masked heads of one kind of mask; the run without a mask is the first bar."""
    return stacked(sweep[sweep["mask"].isin([kind, "none"])], "heads", column, order, labels,
                   xlabel=f"замаскировано голов ({style.LABELS[kind]})", **kwargs)


def answer_types(sweep, kind="top", n=1, **kwargs):
    """
    What the answers turn into as more heads of one kind are masked: stacked shares of exact / full / partial /
    miss by ROUGE-n (see data.answer_type). The partial answers are the paper's "incomplete retrieval".
    """
    return shares(sweep, type_column(n), ANSWER_TYPES, TYPE_LABELS, kind, **kwargs)


def fact_wordings(annotated, facts, levels, labels, kind="top", title=None):
    """
    A panel per fact of the reference: how the fact is worded in the answers (word for word, another form, a close
    substitute, absent) against the number of masked heads. annotated: paraphrase.annotate() of a sweep.
    """
    fig, axes = plt.subplots(1, len(facts), figsize=style.size(4.3 * len(facts) + 2.4, 4.2), sharey=True)
    for ax, fact in zip(axes, facts):
        shares(annotated, fact, levels, labels, kind, legend=ax is axes[-1], ax=ax, title=f"«{fact}»")
        ax.set_xlabel("замаскировано голов")
        if ax is not axes[0]:
            ax.set_ylabel("")
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    return axes


def compare_runs(sweep, runs, value="rouge1_recall", ax=None, title=None, subtitle=None):
    """Horizontal bars of the mean of value for chosen runs. runs: {label: name in the column run}, drawn top to bottom."""
    t = style.THEME
    ax = new_ax(ax, 6.8, 0.45 * len(runs) + 1.3)
    data = sweep.assign(_v=numeric(sweep, value))
    means = [data.loc[data["run"] == run, "_v"].mean() for run in runs.values()]
    kinds = [data.loc[data["run"] == run, "mask"].iloc[0] for run in runs.values()]
    ys = np.arange(len(runs))[::-1]
    ax.barh(ys, means, height=0.6, color=[t.colors[k] for k in kinds])
    for y, m in zip(ys, means):
        ax.annotate(f"{m:.1f}", (m, y), xytext=(5, 0), textcoords="offset points", va="center", color=t.ink_secondary, fontsize=t.font - 1)
    ax.set_yticks(ys, list(runs))
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    if max(means) <= 100:
        ax.set_xlim(0, 108)
        round_axis(ax, "x", 0, 100, 20)
    ax.set_xlabel(VALUE_LABELS.get(value, value))
    titles(ax, title, subtitle)
    return ax


# ---------------------------------------------------------------------------------------------------------------
# heads
# ---------------------------------------------------------------------------------------------------------------

SCORE_BINS = [("0", lambda s: s == 0), ("0 – 0.1", lambda s: (s > 0) & (s <= 0.1)), ("0.1 – 0.5", lambda s: (s > 0.1) & (s <= 0.5)),
              ("выше 0.5", lambda s: s > 0.5)]


def score_distribution(score, ax=None, title=None, subtitle=None):
    """The paper's figure 3: the share of the heads by their retrieval score; the heads above the threshold stand out."""
    t = style.THEME
    ax = new_ax(ax, 6.4, 4.2)
    score = np.ravel(score)
    counts = [int(test(score).sum()) for _, test in SCORE_BINS]
    share = [100 * c / len(score) for c in counts]
    colors = [t.grid, t.colors["muted"], style.RAMP[4], style.RAMP[8]]   # the same as in score_pie
    ax.bar(range(len(SCORE_BINS)), share, width=0.62, color=colors)
    for i, (s, c) in enumerate(zip(share, counts)):
        ax.annotate(f"{s:.1f}%\n{c} гол.", (i, s), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", color=t.ink)
    ax.set_xticks(range(len(SCORE_BINS)), [name for name, _ in SCORE_BINS])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, max(share) * 1.25)
    ax.set_xlabel("retrieval score")
    ax.set_ylabel("доля голов, %")
    titles(ax, title, subtitle)
    return ax


def score_pie(score, ax=None, title=None, subtitle=None):
    """The same shares as score_distribution, as parts of the whole: a ring with the number of heads in the middle."""
    t = style.THEME
    ax = new_ax(ax, 6.4, 4.2)
    score = np.ravel(score)
    counts = [int(test(score).sum()) for _, test in SCORE_BINS]
    colors = [t.grid, t.colors["muted"], style.RAMP[4], style.RAMP[8]]
    wedges, _ = ax.pie(counts, colors=colors, startangle=90, counterclock=False,
                       wedgeprops=dict(width=0.42, edgecolor=t.surface, linewidth=2))
    ax.text(0, 0, f"{len(score)}\nголов", ha="center", va="center", fontsize=t.font + 3, fontweight="semibold", linespacing=1.2)
    ax.legend(wedges, [f"{name}:  {c} гол. ({100 * c / len(score):.1f}%)" for (name, _), c in zip(SCORE_BINS, counts)],
              title="retrieval score", loc="center left", bbox_to_anchor=(1.0, 0.5), alignment="left")
    ax.set_aspect("equal")
    titles(ax, title, subtitle)
    return ax


def score_vs_frequency(ranking, top=100, threshold=0.1, ax=None, title=None, subtitle=None):
    """
    The paper's figure 4: for the best heads in the order of the score, the score and the activation frequency (the
    share of the samples in which the head copied at least one token). ranking: data.Heads.ranking().
    """
    t = style.THEME
    ax = new_ax(ax, 6.8, 4.0)
    part = ranking.iloc[:top]
    ax.plot(part.index, part["frequency"], color=t.colors["second"], label="частота активации")
    ax.plot(part.index, part["score"], color=t.colors["accent"], label="retrieval score")
    ax.axhline(threshold, color=t.axis, linewidth=1, zorder=1)
    ax.annotate(f"порог {threshold:g}", (top, threshold), xytext=(-2, 3), textcoords="offset points", ha="right", color=t.ink_muted, fontsize=t.font - 1)
    ax.set_xlim(1 - top * 0.02, top * 1.02)
    round_axis(ax, "x", 0, top, 10 if top <= 120 else 50)
    ax.set_ylim(0, 1.03)
    round_axis(ax, "y", 0, 1, 0.2)
    ax.set_xlabel("место головы по retrieval score")
    ax.set_ylabel("значение")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    titles(ax, title, subtitle)
    return ax


def head_heatmap(score, layers=None, vmax=None, mark=None, ax=None, title=None, subtitle=None, label="retrieval score"):
    """
    The paper's figure 5: the score of every head, layers along x, heads along y. mark: a list of (layer, head) to
    ring, e.g. the masked heads.
    """
    t = style.THEME
    ax = new_ax(ax, 8.4, 4.6)
    score = np.asarray(score)
    layers = list(layers) if layers is not None else list(range(score.shape[0]))
    mesh = ax.pcolormesh(edges(layers), np.arange(score.shape[1] + 1) - 0.5, score.T, cmap=style.cmap(t.cmap_heads), vmin=0,
                         vmax=vmax or max(float(score.max()), 1e-9), edgecolors=t.surface, linewidth=0.5 * t.scale)
    if mark:
        ax.scatter([l for l, _ in mark], [h for _, h in mark], s=60 * t.scale ** 2, facecolors="none", edgecolors=t.colors["second"], linewidths=1.3)
    round_axis(ax, "x", min(layers), max(layers), 5)
    round_axis(ax, "y", 0, score.shape[1] - 1, 5)
    ax.grid(False)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xlabel("слой")
    ax.set_ylabel("голова")
    bar = ax.figure.colorbar(mesh, ax=ax, pad=0.02, aspect=22)
    bar.set_label(label)
    bar.outline.set_visible(False)
    titles(ax, title, subtitle)
    return ax


def top_heads(ranking, k=20, threshold=0.1, ax=None, title=None, subtitle=None):
    """The k best heads as bars, the best on top; the label is layer-head."""
    t = style.THEME
    ax = new_ax(ax, 6.0, 0.3 * k + 1.2)
    part = ranking.iloc[:k]
    ys = np.arange(len(part))[::-1]
    ax.barh(ys, part["score"], height=0.62, color=[t.colors["accent"] if s > threshold else t.colors["muted"] for s in part["score"]])
    for y, s in zip(ys, part["score"]):
        ax.annotate(f"{s:.2f}", (s, y), xytext=(4, 0), textcoords="offset points", va="center", color=t.ink, fontsize=t.font - 1)
    ax.set_yticks(ys, part["head"])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlim(0, max(1.0, float(part["score"].max())) * 1.08)
    round_axis(ax, "x", 0, max(1.0, float(part["score"].max())), 0.2)
    ax.set_ylim(-0.7, len(part) - 0.3)
    ax.set_xlabel("retrieval score")
    ax.set_ylabel("слой-голова")
    titles(ax, title, subtitle)
    return ax


def layer_profile(score, layers=None, threshold=0.1, ax=None, title=None, subtitle=None):
    """How many heads above the threshold every layer has."""
    t = style.THEME
    ax = new_ax(ax, 7.2, 3.4)
    score = np.asarray(score)
    layers = list(layers) if layers is not None else list(range(score.shape[0]))
    counts = (score > threshold).sum(1)
    ax.bar(layers, counts, width=0.7, color=t.colors["accent"])
    round_axis(ax, "x", min(layers), max(layers), 5)
    ax.set_xlim(min(layers) - 0.8, max(layers) + 0.8)
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel("слой")
    ax.set_ylabel(f"голов со score выше {threshold:g}")
    titles(ax, title, subtitle)
    return ax


def correlation(scores, ax=None, title=None, subtitle=None):
    """
    The paper's figure 6 turned to our question: Spearman correlation of the head scores between conditions.
    scores: {name: per-head scores of the same heads}, e.g. by metric, by needle or by range of context length.
    """
    t = style.THEME
    names = list(scores)
    ax = new_ax(ax, 1.0 * len(names) + 2.6, 0.8 * len(names) + 1.6)
    matrix = np.array([[spearman(scores[a], scores[b]) for b in names] for a in names])
    low = min(0.0, float(matrix.min()))
    mesh = ax.pcolormesh(matrix, cmap=style.cmap(t.cmap_corr), vmin=low, vmax=1, edgecolors=t.surface, linewidth=2)
    for i in range(len(names)):
        for j in range(len(names)):
            dark = (matrix[i, j] - low) / (1 - low) > 0.55
            ax.text(j + 0.5, i + 0.5, f"{matrix[i, j]:.2f}", ha="center", va="center", color=t.surface if dark else t.ink, fontsize=t.font - 1)
    ax.set_xticks(np.arange(len(names)) + 0.5, names, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(len(names)) + 0.5, names)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.grid(False)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    bar = ax.figure.colorbar(mesh, ax=ax, pad=0.03, aspect=16)
    bar.set_label("корреляция Спирмена")
    bar.outline.set_visible(False)
    titles(ax, title, subtitle)
    return ax


# ---------------------------------------------------------------------------------------------------------------
# levels of the hard tasks
# ---------------------------------------------------------------------------------------------------------------

def run_styles(df):
    """
    {run: (colour, label)} — the colour is the one of the kind of the mask; of several runs of one kind the ones
    with fewer masked heads are lighter.
    """
    from matplotlib.colors import to_rgb
    t = style.THEME
    runs = df.groupby("run").agg(mask=("mask", "first"), heads=("heads", "first")).sort_values(["mask", "heads"])
    styles = {}
    for kind, part in runs.groupby("mask"):
        for i, (run, row) in enumerate(part.iterrows()):
            fade = 0.45 * (len(part) - 1 - i) / max(len(part) - 1, 1)
            color = tuple(c + (s - c) * fade for c, s in zip(to_rgb(t.colors[kind]), to_rgb(t.surface)))
            styles[run] = (color, style.LABELS[kind] if kind == "none" else f"{style.LABELS[kind]}, {row['heads']}")
    return styles


def level_lines(df, value="correct", ax=None, title=None, subtitle=None):
    """The mean of value at every level, a line per run; the levels stand in their order, the first is the reference."""
    ax = new_ax(ax, 7.2, 4.2)
    data = df.assign(_v=numeric(df, value))
    levels = sorted(data["level"].unique())
    styles = run_styles(data)
    for run in sorted(styles, key=lambda r: (data.loc[data["run"] == r, "mask"].iloc[0] != "none", r)):
        line = data[data["run"] == run].groupby("level")["_v"].mean().reindex(levels)
        ax.plot(range(len(levels)), line.to_numpy(), color=styles[run][0], marker="o", label=styles[run][1])
    ax.set_xticks(range(len(levels)), levels)
    ax.set_xlim(-0.25, len(levels) - 0.75)
    if data["_v"].max() <= 100:
        ax.set_ylim(-3, 103)
        round_axis(ax, "y", 0, 100, 20)
    ax.set_xlabel("уровень")
    ax.set_ylabel(VALUE_LABELS.get(value, value))
    ax.legend(title="маска", loc="center left", bbox_to_anchor=(1.01, 0.5), alignment="left")
    titles(ax, title, subtitle)
    return ax


def class_panels(df, order, labels, runs=None, x="level", xlabel="уровень", column="answer_class", title=None):
    """A panel per run: the stacked shares of the values of a column (the classes of the answers) at every level (or another x)."""
    runs = runs or sorted(df["run"].unique(), key=lambda r: (df.loc[df["run"] == r, "mask"].iloc[0] != "none", r))
    names = run_styles(df)
    fig, axes = plt.subplots(1, len(runs), figsize=style.size(3.6 * len(runs) + 1.2 + 0.085 * max(len(v) for v in labels.values()), 4.2), sharey=True, squeeze=False)
    for ax, run in zip(axes[0], runs):
        stacked(df[df["run"] == run], x, column, order, labels, xlabel=xlabel, legend=ax is axes[0, -1], ax=ax, title=names[run][1])
        if ax is not axes[0, 0]:
            ax.set_ylabel("")
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    return axes


def rank_profile(scores, order, top=60, reference=None, ylabel="retrieval score", legend=True, ax=None, title=None, subtitle=None):
    """
    The heads in the order of a fixed ranking (order: flat indices, the best first) and their score in every
    condition of scores, a line per condition: do the heads keep their values where the ranking was not made.
    reference: the scores the ranking was made from, drawn as the neutral line.
    """
    t = style.THEME
    ax = new_ax(ax, 7.6, 4.2)
    xs = np.arange(1, top + 1)
    if reference is not None:
        ax.plot(xs, np.ravel(reference)[order[:top]], color=t.colors["none"], linestyle=(0, (1, 1.5)), label="детекция", zorder=3)
    for (name, score), color in zip(scores.items(), style.ordered(len(scores))):
        ax.plot(xs, np.ravel(score)[order[:top]], color=color, label=name)
    ax.set_xlim(1 - top * 0.02, top * 1.02)
    round_axis(ax, "x", 0, top, 10)
    ax.set_ylim(0, None)
    ax.set_xlabel("место головы в рейтинге детекции")
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    titles(ax, title, subtitle)
    return ax


def group_bars(table, ylabel="средний retrieval score", xlabel="головы по месту в рейтинге детекции", ax=None, title=None, subtitle=None):
    """
    Grouped bars of a table: its rows are the groups along x (groups of heads), its columns the bars of a group
    (conditions). The first column is drawn neutral when it is "детекция".
    """
    t = style.THEME
    ax = new_ax(ax, 8.0, 4.2)
    columns = list(table.columns)
    reference = columns[0] == "детекция"
    colors = ([t.colors["muted"]] if reference else []) + style.ordered(len(columns) - reference)
    width = 0.8 / len(columns)
    for i, (column, color) in enumerate(zip(columns, colors)):
        ax.bar(np.arange(len(table)) + (i - (len(columns) - 1) / 2) * width, table[column], width=width * 0.9, color=color, label=column)
    ax.set_xticks(np.arange(len(table)), list(table.index))
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    titles(ax, title, subtitle)
    return ax


def head_table(table, label="retrieval score", vmax=None, ax=None, title=None, subtitle=None):
    """Blocks with numbers: the rows of the table are heads, the columns are conditions, the colour is the value."""
    t = style.THEME
    ax = new_ax(ax, 1.05 * table.shape[1] + 2.6, 0.3 * table.shape[0] + 1.4)
    values = table.to_numpy(dtype=float)
    vmax = vmax or float(np.nanmax(values))
    mesh = ax.pcolormesh(values, cmap=style.cmap(t.cmap_heads), vmin=0, vmax=vmax, edgecolors=t.surface, linewidth=1.5)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j + 0.5, i + 0.5, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=t.font - 2,
                    color=t.surface if values[i, j] / vmax > 0.55 else t.ink)
    ax.set_xticks(np.arange(table.shape[1]) + 0.5, list(table.columns))
    ax.set_yticks(np.arange(table.shape[0]) + 0.5, list(table.index))
    ax.invert_yaxis()
    ax.grid(False)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_ylabel("слой-голова")
    bar = ax.figure.colorbar(mesh, ax=ax, pad=0.03, aspect=24)
    bar.set_label(label)
    bar.outline.set_visible(False)
    titles(ax, title, subtitle)
    return ax


def score_scatter(x, y, threshold=0.1, mark=None, mark_label=None, top=None, xlabel="скор детекции на простой задаче", ylabel="скор на уровне",
                  ax=None, title=None, subtitle=None):
    """
    A point per head: its score in the detection (x) against its score in another condition (y). The heads the
    detection puts above the threshold stand out; on the diagonal the score is kept. The correlations over all
    heads and over the heads above the threshold are written in the corner. mark: the heads to stand out when x is
    not the detection score (a yes/no array over the heads) and mark_label their name; top: the upper limit of both axes.
    """
    t = style.THEME
    ax = new_ax(ax, 5.2, 5.0)
    x, y = np.ravel(x), np.ravel(y)
    found = x > threshold if mark is None else np.ravel(mark)
    top = top or max(1.0, float(x.max()), float(y.max()))
    ax.plot([0, top], [0, top], color=t.axis, linewidth=1, zorder=1)
    ax.scatter(x[~found], y[~found], s=14 * t.scale ** 2, color=t.colors["muted"], linewidths=0, alpha=0.7, label=f"остальные ({int((~found).sum())})", zorder=2)
    ax.scatter(x[found], y[found], s=34 * t.scale ** 2, color=style.RAMP[6], edgecolors=t.surface, linewidths=0.8,
               label=f"{mark_label or f'детекция выше {threshold:g}'} ({int(found.sum())})", zorder=3)
    r = lambda a, b: np.corrcoef(a, b)[0, 1]
    rank = lambda v: pd.Series(v).rank().to_numpy()
    note = (f"все головы: r = {r(x, y):.2f}, ρ = {r(rank(x), rank(y)):.2f}\n{mark_label or f'выше {threshold:g}'}: r = {r(x[found], y[found]):.2f}, "
            f"ρ = {r(rank(x[found]), rank(y[found])):.2f}")
    # the corner that the points leave free: under the diagonal for scores that are kept, above it for the marked heads
    where = dict(x=0.97, y=0.04, ha="right", va="bottom") if mark is None else dict(x=0.04, y=0.80, ha="left", va="top")
    ax.text(s=note, transform=ax.transAxes, fontsize=t.font - 1, linespacing=1.5, **where)
    ax.set_xlim(-0.03, top * 1.03)
    ax.set_ylim(-0.03, top * 1.03)
    round_axis(ax, "x", 0, top, 0.2 if top > 0.6 else 0.1)
    round_axis(ax, "y", 0, top, 0.2 if top > 0.6 else 0.1)
    ax.set_aspect("equal")
    ax.grid(axis="x", visible=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left")
    titles(ax, title, subtitle)
    return ax
