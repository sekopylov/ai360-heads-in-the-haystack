"""Compare mean head scores across metrics from the same detection run."""
import argparse
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def average_ranks(values):
    """One-based ranks; tied values get the average rank (Spearman)."""
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return ranks


def correlation(x, y):
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    detection = args.run / "detection"
    manifest = json.loads((detection / "run.json").read_text())
    if not manifest["complete"]:
        raise ValueError("Detection is incomplete")
    aggregation_path = detection / 'aggregation' / 'run.json'
    aggregation = json.loads(aggregation_path.read_text()) if aggregation_path.exists() else manifest
    if (aggregation_path.exists() and manifest.get('run_id')
            and aggregation.get('source_run_id') != manifest['run_id']):
        raise ValueError('Aggregation belongs to another detection run')
    files = aggregation["head_score_files"]
    if len(files) < 2:
        raise ValueError("At least two metrics are required")
    score_dir = aggregation_path.parent if aggregation_path.exists() else detection
    data = {name: json.loads((score_dir / filename).read_text())
            for name, filename in files.items()}
    names = list(data)
    heads = sorted(data[names[0]], key=lambda h: tuple(map(int, h.split("-"))))
    if not heads or any(set(scores) != set(heads) for scores in data.values()):
        raise ValueError("Metrics must have identical nonempty head sets")
    if any(not data[name][h] or len(data[name][h]) != len(data[names[0]][h])
           for name in names for h in heads):
        raise ValueError("Metrics must contain matched nonempty case histories")
    values = np.array([[np.mean(data[name][h]) for h in heads] for name in names])
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite score")
    ranks = np.array([average_ranks(row) for row in values])
    pearson = [[correlation(x, y) for y in values] for x in values]
    spearman = [[correlation(x, y) for y in ranks] for x in ranks]
    labels = [name.replace("needle_", "").replace("_v1", "").replace("_", " ")
              for name in names]
    output = args.run / "plots" / "metric-correlations"
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, matrix, title in zip(axes, [pearson, spearman], ["Pearson: values", "Spearman: ranks"]):
        numeric = np.array([[np.nan if v is None else v for v in row] for row in matrix])
        im = ax.imshow(numeric, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(names)), labels, rotation=25, ha="right")
        ax.set_yticks(range(len(names)), labels)
        ax.set_title(title)
        for i, j in itertools.product(range(len(names)), repeat=2):
            ax.text(j, i, "n/a" if matrix[i][j] is None else f"{matrix[i][j]:.3f}",
                    ha="center", va="center", color="white" if abs(numeric[i, j]) > .6 else "black")
    fig.colorbar(im, ax=axes, shrink=.8)
    fig.suptitle(f"{manifest['model_version']} | {len(heads)} heads | {manifest['attention_scope']}")
    fig.savefig(output / "correlation_matrices.png", dpi=180)
    plt.close(fig)
    pairs = list(itertools.combinations(range(len(names)), 2))
    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4.5),
                             squeeze=False, layout="constrained")
    for ax, (i, j) in zip(axes[0], pairs):
        ax.scatter(values[i], values[j], s=12, alpha=.35, edgecolors="none")
        ax.set_xlabel(labels[i])
        ax.set_ylabel(labels[j])
        p, s = pearson[i][j], spearman[i][j]
        ax.set_title(f"Pearson {p:.3f} | Spearman {s:.3f}" if p is not None and s is not None else "Undefined correlation")
        ax.grid(alpha=.2)
    fig.suptitle("Each point = one head; scores averaged over successful cases")
    fig.savefig(output / "pairwise_scatter.png", dpi=180)
    plt.close(fig)
    # Same head identity/colors across panels; label the union of top-5 heads
    # from each metric rather than attempting 1152 overlapping annotations.
    highlighted = sorted(set(int(index) for row in values
                             for index in np.argsort(-row, kind='stable')[:5]))
    fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 5.5), squeeze=False)
    colors = plt.get_cmap('tab20')
    for ax, (i, j) in zip(axes[0], pairs):
        ax.scatter(values[i], values[j], s=12, color='#94a3b8', alpha=.3, edgecolors='none')
        for number, index in enumerate(highlighted):
            color = colors(number % 20)
            ax.scatter(values[i, index], values[j, index], s=55, color=color,
                       edgecolors='black', linewidths=.4, zorder=3, label=heads[index])
            ax.annotate(heads[index], (values[i, index], values[j, index]),
                        xytext=(6, 8 if number % 2 == 0 else -14), textcoords='offset points',
                        fontsize=8, color=color, arrowprops={'arrowstyle': '-', 'color': color, 'lw': .5})
        ax.set_xlabel(labels[i])
        ax.set_ylabel(labels[j])
        ax.grid(alpha=.2)
        ax.margins(.16)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='lower center', ncol=min(len(highlighted), 8),
               title='Layer-head (zero-based); colored = union of top-5 per metric')
    fig.suptitle(f"{manifest['model_version']}: same labeled heads across metric pairs")
    fig.tight_layout(rect=(0, .15, 1, .94))
    fig.savefig(output / 'pairwise_heads.png', dpi=180)
    plt.close(fig)
    summary = {"metrics": names, "head_count": len(heads),
               "successful_cases": manifest["successful_cases"],
               "selected_cases": aggregation.get('selected_case_count', manifest['successful_cases']),
               "attention_scope": manifest["attention_scope"],
               "highlighted_heads": [heads[index] for index in highlighted],
               "pearson": pearson, "spearman": spearman,
               "zero_head_counts": {name: int(np.sum(values[i] == 0)) for i, name in enumerate(names)}}
    (output / "correlations.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Plots: {output}")


if __name__ == "__main__":
    main()
