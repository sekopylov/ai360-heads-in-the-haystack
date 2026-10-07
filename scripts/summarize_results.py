#!/usr/bin/env python
"""Summarise a results tree into a compact markdown report.

Reads whatever the pipeline produced and prints the numbers that go into the
paper's claims, so a run can be reported without hand-copying JSON:

    .venv/bin/python scripts/summarize_results.py ds-results
    .venv/bin/python scripts/summarize_results.py ds-results --out docs/results.md

Only files that exist are reported; a partial run produces a partial report
rather than an error, which is what you want while a job is still going.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def model_keys(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir() if p.is_dir() and p.name != "figures")


def census_section(root: Path, keys: list[str]) -> list[str]:
    rows = ["| model | layers | scoreable layers | scoreable heads | head_dim | hybrid |",
            "|---|---|---|---|---|---|"]
    for key in keys:
        info = load(root / key / "model_info.json") or _info_from_summary(root, key)
        if not info:
            continue
        rows.append(
            f"| {info['name']} | {info['num_layers']} | {info['num_scoreable_layers']} "
            f"| {info['n_scoreable_heads']} | {info['head_dim']} | {info['is_hybrid']} |"
        )
    return rows


def _info_from_summary(root: Path, key: str) -> dict[str, Any] | None:
    for pairing in ("next_step", "same_step"):
        summary = load(root / key / f"summary_{pairing}.json")
        if summary and "model_info" in summary:
            return summary["model_info"]
    return None


def detection_section(root: Path, keys: list[str]) -> list[str]:
    out = ["| model | pairing | instances | recited | mean recall | top head | score | >0.1 | >0.5 |",
           "|---|---|---|---|---|---|---|---|---|"]
    for key in keys:
        for pairing in ("next_step", "same_step"):
            summary = load(root / key / f"summary_{pairing}.json")
            if not summary:
                continue
            sparsity = summary["sparsity"]
            top = summary["top_heads"][0] if summary["top_heads"] else {}
            n = sparsity["n_heads"]
            above = sparsity["thresholds"]
            out.append(
                f"| {summary['model']} | {pairing} | {summary['n_instances']} "
                f"| {summary.get('n_instances_recited', 'n/a')}/{summary['n_instances']} "
                f"| {fmt(summary['mean_needle_recall'], 3)} "
                f"| `{top.get('head', '?')}` | {fmt(top.get('score'))} "
                f"| {above['0.1']['n']}/{n} ({100 * above['0.1']['frac']:.1f}%) "
                f"| {above['0.5']['n']}/{n} ({100 * above['0.5']['frac']:.1f}%) |"
            )
    return out


def pairing_section(root: Path, keys: list[str]) -> list[str]:
    out = ["| model | top-10 overlap between pairings | next_step top heads | same_step top heads |",
           "|---|---|---|---|"]
    for key in keys:
        summary = load(root / key / "summary_next_step.json")
        if not summary or "pairing_comparison" not in summary:
            continue
        comparison = summary["pairing_comparison"]
        primary = comparison["primary"]["next_step"]
        secondary = comparison["secondary"]["same_step"]
        out.append(
            f"| {summary['model']} | {comparison['overlap']}/{comparison['top_k']} "
            f"| {', '.join(primary[:5])} | {', '.join(secondary[:5])} |"
        )
    return out


def masking_section(root: Path, keys: list[str]) -> list[str]:
    out = ["| model | K | % heads | retrieval f1 | retrieval exact | random f1 | random exact |",
           "|---|---|---|---|---|---|---|"]
    for key in keys:
        curve = load(root / key / "masking_curve.json")
        if not curve:
            continue
        n_heads = curve.get("n_scoreable_heads") or 1
        for i, k in enumerate(curve["k_values"]):
            exact = (curve.get("retrieval_exact_match") or [None] * len(curve["k_values"]))[i]
            rand_exact = (curve.get("random_exact_match_mean") or [None] * len(curve["k_values"]))[i]
            out.append(
                f"| {curve.get('model', key)} | {k} | {100 * k / n_heads:.1f}% "
                f"| {fmt(curve['retrieval'][i], 1)} | {fmt(exact, 1)} "
                f"| {fmt(curve['random_mean'][i], 1)} | {fmt(rand_exact, 1)} |"
            )
        out.append(f"| | baseline | | {fmt(curve.get('baseline'), 1)} "
                   f"| {fmt(curve.get('baseline_exact_match'), 1)} | | |")
    return out


def mixer_section(root: Path, keys: list[str]) -> list[str]:
    out = []
    for key in keys:
        ablation = load(root / key / "mixer_ablation.json")
        if not ablation:
            continue
        out.append(f"**{key}** — {ablation['n_full_layers']} full-attention layers, "
                   f"{ablation['n_linear_layers']} linear layers, "
                   f"baseline f1={fmt(ablation['baseline'], 1)}")
        out.append("")
        out.append("| K | full-attention masked | linear masked |")
        out.append("|---|---|---|")
        for i, k in enumerate(ablation["k_values"]):
            full = ablation["full_attention"][i] if i < len(ablation["full_attention"]) else None
            linear = (ablation["linear_attention"][i]
                      if i < len(ablation["linear_attention"]) else None)
            out.append(f"| {k} | {fmt(full, 1)} | {fmt(linear, 1)} |")
        out.append("")
    return out


def task_section(root: Path, keys: list[str]) -> list[str]:
    out = []
    for key in keys:
        qa = load(root / key / "task_qa.json")
        if qa:
            out.append(f"**{key} — extractive QA**: baseline F1 {fmt(qa['baseline_f1'], 1)} "
                       f"over {qa['n_samples']} samples")
            out.append("")
            out.append("| K | % heads | retrieval F1 (drop) | random F1 (drop) |")
            out.append("|---|---|---|---|")
            for k, row in qa["by_k"].items():
                out.append(f"| {k} | | {fmt(row['retrieval_f1'], 1)} "
                           f"({fmt(row['drop_retrieval'], 1)}) | {fmt(row['random_f1_mean'], 1)} "
                           f"({fmt(row['drop_random'], 1)}) |")
            out.append("")
        cot = load(root / key / "task_cot.json")
        if cot:
            out.append(f"**{key} — chain-of-thought**: K={cot['k']}, {cot['n_samples']} samples")
            out.append("")
            out.append("| variant | baseline | retrieval masked | random masked |")
            out.append("|---|---|---|---|")
            for variant, row in cot["results"].items():
                out.append(f"| {variant} | {fmt(row['baseline'], 1)} "
                           f"| {fmt(row['retrieval_masked'], 1)} "
                           f"| {fmt(row['random_masked_mean'], 1)} |")
            out.append("")
    return out


def correlation_section(root: Path) -> list[str]:
    correlation = load(root / "correlation.json")
    if not correlation:
        return []
    out = [f"Retrieval-score correlation (mode `{correlation['mode']}`):", ""]
    labels = correlation["labels"]
    out.append("| | " + " | ".join(labels) + " |")
    out.append("|---" * (len(labels) + 1) + "|")
    for i, label in enumerate(labels):
        cells = " | ".join(fmt(correlation["values"][i][j]) for j in range(len(labels)))
        out.append(f"| {label} | {cells} |")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", nargs="?", default="ds-results", help="results tree")
    parser.add_argument("--out", default=None, help="also write the report here")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"no such results directory: {root}")

    keys = model_keys(root)
    lines = [f"# Results — `{root}`", ""]
    for title, body in (
        ("Architecture census", census_section(root, keys)),
        ("Retrieval-head detection", detection_section(root, keys)),
        ("Pairing sensitivity", pairing_section(root, keys)),
        ("Masking curve", masking_section(root, keys)),
        ("Token-mixer ablation", mixer_section(root, keys)),
        ("Downstream tasks", task_section(root, keys)),
        ("Cross-model correlation", correlation_section(root)),
    ):
        if body:
            lines += [f"## {title}", ""] + body + [""]

    files = sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file())
    if files:
        lines += ["## Artifacts", ""] + [f"- `{p}`" for p in files] + [""]

    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"\n[summary] written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
