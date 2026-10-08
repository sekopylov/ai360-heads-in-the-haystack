"""Plot retrieval-score distribution and paired masking accuracy offline."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from rouge_score import rouge_scorer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def plot_detection(run, out, detection, heads, scores, counts):
    """Detection-only plots, with histories aligned to successful cases."""
    title = detection.get("model_version", "Model")
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    rows = []
    for path in (run / "detection/results").glob("*_results.json"):
        row = read(path)
        if (row["model"] != detection["model"]
                or row["context_length"] not in detection["lengths"]
                or row["depth_percent"] not in detection["depths"]):
            continue
        rouge = scorer.score(row["expected_answer"], row["model_response"])["rouge1"]
        if abs(rouge.recall * 100 - row["score"]) > 1e-6:
            raise ValueError(f"Stored ROUGE differs from recalculated: {path}")
        rows.append(dict(case_id=row["case_id"], length=row["context_length"],
                         depth=row["depth_percent"], rouge_recall=row["score"],
                         rouge_f1=rouge.fmeasure * 100,
                         tokens=len(row.get("generated_token_ids", [])),
                         duration=row["test_duration_seconds"],
                         finish_reason=row.get("finish_reason"), source=str(path)))
    def case_order(row):
        suffix = row["case_id"].rsplit("-", 1)[-1]
        key = (0, int(suffix)) if suffix.isdigit() else (1, row["case_id"])
        return key, detection["lengths"].index(row["length"]), detection["depths"].index(row["depth"])
    rows.sort(key=case_order)
    if len(rows) != detection["completed_cases"]:
        raise ValueError("Detection result count differs from manifest")
    successful = [r for r in rows if r["rouge_recall"] > detection["success_threshold"]]
    if any(len(v) != len(successful) for v in heads.values()):
        raise ValueError("Head histories do not match successful detection cases")
    labels = [f'{r["case_id"]}\n{r["length"]:,} / {r["depth"]:g}%' for r in successful]
    ranked = sorted(scores, key=lambda h: (-scores[h], tuple(map(int, h.split("-")))))
    top = ranked[:15]
    fig, ax = plt.subplots(figsize=(10, 6))
    positions = list(range(len(top)))
    bars = ax.barh(positions, [scores[h] for h in top], color="#2563eb", alpha=.65,
                   label="Mean across successful cases")
    for i, label in enumerate(labels):
        ax.scatter([heads[h][i] for h in top], positions, s=30, label=label.replace("\n", " · "), zorder=3)
    ax.set_yticks(positions, top)
    ax.invert_yaxis()
    ax.set_xlim(0, max(1, max(scores.values())) * 1.05)
    ax.set_xlabel(f"Retrieval score ({detection.get('attention_scope', 'decode')}; not ROUGE)")
    ax.set_ylabel("Layer–head (zero-based)")
    ax.set_title(f"{title} · top 15 retrieval heads")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "top_heads.png", dpi=180)
    plt.close(fig)

    parsed = {tuple(map(int, h.split("-"))): s for h, s in scores.items()}
    layers, head_ids = sorted({l for l, h in parsed}), sorted({h for l, h in parsed})
    matrix = [[parsed.get((l, h), float("nan")) for h in head_ids] for l in layers]
    fig, ax = plt.subplots(figsize=(10, 9))
    heatmap = ax.imshow(matrix, aspect="auto", cmap="magma", vmin=0,
                        vmax=max(1, max(scores.values())), interpolation="nearest")
    ax.set_xticks(range(len(head_ids)), head_ids, fontsize=8)
    ax.set_yticks(range(len(layers)), layers, fontsize=8)
    ax.set_xlabel("Head (zero-based)")
    ax.set_ylabel("Layer (zero-based)")
    ax.set_title(f"{title} · mean retrieval score ({len(successful)} successful cases)")
    fig.colorbar(heatmap, ax=ax, label="Mean retrieval score")
    fig.tight_layout()
    fig.savefig(out / "head_heatmap.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = list(range(len(rows)))
    case_labels = [r["case_id"] for r in rows]
    axes[0].bar([p-.18 for p in x], [r["rouge_recall"] for r in rows], width=.36,
                color="#2563eb", label="Recall")
    axes[0].bar([p+.18 for p in x], [r["rouge_f1"] for r in rows], width=.36,
                color="#94a3b8", label="F1 (additional diagnostic)")
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("ROUGE-1 (%)")
    axes[0].legend(fontsize=8)
    for ax, key, label in [(axes[1], "tokens", "Generated tokens (thinking + answer)"),
                           (axes[2], "duration", "Generation time (seconds, incl. prefill)")]:
        bars = ax.bar(x, [r[key] for r in rows], color="#38bdf8")
        ax.bar_label(bars, fmt="%.0f", padding=3)
        ax.set_ylim(0, max(r[key] for r in rows) * 1.15 or 1)
        ax.set_ylabel(label)
    for ax in axes:
        ax.set_xticks(x, case_labels, rotation=25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{title} · detection only (no masking experiment)")
    fig.text(.5, .01, "Recall rewards quoting the expected answer; it does not establish semantic correctness.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .94))
    fig.savefig(out / "detection_summary.png", dpi=180)
    plt.close(fig)
    (out / "plot_data.json").write_text(json.dumps(dict(
        model=title, retrieval_metric=detection.get("retrieval_metric"),
        attention_scope=detection.get("attention_scope"), head_bin_counts=counts,
        mean_head_scores=scores, successful_case_order=successful, top_heads=top,
        cases=rows, zero_heads=sum(v == 0 for v in scores.values()),
        notes="Head means use successful cases only. F1 is diagnostic; the experiment uses recall.",
    ), indent=2), encoding="utf-8")
    print(f"Detection plots saved to {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("datasphere-results/full"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metric", help="select one metric from a multi-metric detection run")
    parser.add_argument("--threshold", type=float, default=50.0,
                        help="Successful answer means ROUGE-1 recall > threshold")
    args = parser.parse_args()
    out = args.output or args.run / "plots"
    if args.metric and not args.output:
        out = out / args.metric
    out.mkdir(parents=True, exist_ok=True)
    detection = read(args.run / "detection/run.json")
    if not detection["complete"]:
        raise ValueError("Detection is incomplete")
    score_file = "head_scores.json"
    if not args.metric and len(detection.get("retrieval_metrics", [])) > 1:
        raise ValueError("Multiple retrieval metrics: explicitly choose --metric NAME")
    if args.metric:
        files = detection.get("head_score_files", {})
        if args.metric not in files:
            raise ValueError(f"Metric not available in run: {args.metric}")
        score_file = files[args.metric]
        detection = dict(detection, retrieval_metric=args.metric)
    heads = read(args.run / "detection" / score_file)
    if not heads or any(not values for values in heads.values()):
        raise ValueError("No complete head-score history")
    scores = {head: mean(values) for head, values in heads.items()}
    # Disjoint intervals, covering all eligible full-attention heads.
    counts = [sum(0 <= s <= .1 for s in scores.values()),
              sum(.1 < s <= .5 for s in scores.values()),
              sum(.5 < s <= 1 for s in scores.values())]
    bin_labels = ["0–0.1", ">0.1–0.5", ">0.5–1"]
    colors = ["#94a3b8", "#38bdf8", "#2563eb"]
    if any(s > 1 for s in scores.values()):
        counts.append(sum(s > 1 for s in scores.values()))
        bin_labels.append(">1")
        colors.append("#dc2626")
    if sum(counts) != len(scores):
        raise ValueError("Head scores must be finite and nonnegative")
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(bin_labels,
                  [100 * n / len(scores) for n in counts],
                  color=colors)
    ax.bar_label(bars, labels=[f"{100*n/len(scores):.1f}% ({n} heads)" for n in counts], padding=5)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Share of full-attention heads (%)")
    ax.set_xlabel("Mean retrieval score across successful detection cases")
    ax.set_title(f"{detection.get('model_version', 'Model')} · {len(scores)} full-attention heads")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "head_distribution.png", dpi=180)
    plt.close(fig)

    if not list((args.run / "evaluation").rglob("run.json")):
        plot_detection(args.run, out, detection, heads, scores, counts)
        return

    groups = defaultdict(list)
    raw_results = []
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    prompts = {}
    for manifest in sorted((args.run / "evaluation").rglob("run.json")):
        config = read(manifest)
        if not config["complete"]:
            raise ValueError(f"Incomplete condition: {manifest}")
        for length in config["lengths"]:
            for depth in config["depths"]:
                # Filter by the current manifests, never include stale files.
                files = list((manifest.parent / "results").glob(
                    f"*_len_{length}_depth_{depth:g}_results.json"))
                if len(files) != 1:
                    raise ValueError(f"Expected one result: {manifest}, {length}, {depth}")
                row = read(files[0])
                key = (config["context_seed"], length, depth, row["case_id"])
                digest = row["prompt_sha256"]
                if key in prompts and prompts[key] != digest:
                    raise ValueError(f"Contexts differ between conditions: {key}")
                prompts[key] = digest
                condition = config["condition"]
                k = len(row["experiment"]["blocked_heads"])
                family = "baseline" if condition == "baseline" else (
                    "random" if condition.startswith("random") else "top")
                score = scorer.score(row["expected_answer"], row["model_response"])["rouge1"].recall * 100
                if abs(score - row["score"]) > 1e-6:
                    raise ValueError(f"Stored score differs from recalculated score: {files[0]}")
                groups[family, k].append(score)
                raw_results.append(dict(condition=family, k=k, score=score,
                                        context_seed=config["context_seed"],
                                        head_seed=row["experiment"]["seed"],
                                        depth=depth, length=length, source=str(files[0])))
    if ("baseline", 0) not in groups:
        raise ValueError("Baseline results missing")
    summary = []
    fig, ax = plt.subplots(figsize=(8, 5))
    for family, color, label in [("top", "#dc2626", "Top retrieval heads"),
                                 ("random", "#2563eb", "Random heads")]:
        ks = [0] + sorted(k for f, k in groups if f == family)
        ys = []
        for k in ks:
            values = groups["baseline", 0] if k == 0 else groups[family, k]
            accuracy = 100 * sum(v > args.threshold for v in values) / len(values)
            ys.append(accuracy)
            summary.append(dict(condition=family, k=k, n=len(values),
                                accuracy=accuracy, mean_rouge=mean(values),
                                std_rouge=stdev(values) if len(values) > 1 else 0,
                                min_rouge=min(values), max_rouge=max(values)))
        ax.plot(ks, ys, "o-", color=color, label=label, linewidth=2)
    ax.set_xticks(sorted({k for _, k in groups}))
    ax.set_ylim(-3, 105)
    ax.set_xlabel("Number of masked heads (legacy_uniform)")
    ax.set_ylabel(f"Accuracy: answers with ROUGE-1 recall > {args.threshold:g} (%)")
    ax.set_title(detection.get("model_version", "Model") + " · " + ", ".join(map(str, detection["lengths"])) + " tokens")
    ax.grid(alpha=.2)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "masking_accuracy.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for family, color, label in [("top", "#dc2626", "Top retrieval heads"),
                                 ("random", "#2563eb", "Random heads")]:
        points = [row for row in summary if row["condition"] == family]
        ks = [row["k"] for row in points]
        ax.fill_between(ks,
                        [max(0, row["mean_rouge"] - row["std_rouge"]) for row in points],
                        [min(100, row["mean_rouge"] + row["std_rouge"]) for row in points],
                        color=color, alpha=.15)
        for row in points:
            values = groups["baseline", 0] if row["k"] == 0 else groups[family, row["k"]]
            ax.scatter([row["k"]] * len(values), values, color=color,
                       alpha=.25, s=18, zorder=2)
        ax.plot(ks, [row["mean_rouge"] for row in points],
                "o-", color=color, label=label, linewidth=2)
    ax.axhline(50, color="#64748b", linestyle=":", linewidth=1.5,
               label="50% reference")
    ax.set_xticks(sorted({k for _, k in groups}))
    ax.set_ylim(-3, 105)
    ax.set_xlabel("Number of masked heads (legacy_uniform)")
    ax.set_ylabel("Mean ROUGE-1 recall (%)")
    ax.set_title(detection.get("model_version", "Model") + " · " + ", ".join(map(str, detection["lengths"])) + " tokens")
    ax.text(.02, .03, "Bands: mean ± 1 sample SD (clipped to 0–100%)\n"
            "Dots: individual context/depth tests (random includes repeats)",
            transform=ax.transAxes, fontsize=9)
    ax.grid(alpha=.2)
    ax.legend(loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "masking_mean_rouge.png", dpi=180)
    plt.close(fig)
    (out / "plot_data.json").write_text(json.dumps(
        {"threshold": args.threshold, "head_bin_counts": counts,
         "mean_head_scores": scores, "masking": summary,
         "spread": "Sample SD across individual context/depth/repeat tests, not a confidence interval",
         "raw_results": raw_results}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Plots saved to {out}")


if __name__ == "__main__":
    main()
