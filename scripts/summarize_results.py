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
import logging
import sys
from pathlib import Path
from collections.abc import Mapping
from typing import Any

# The script is run as `python scripts/summarize_results.py <root>` from the repo
# root, which puts `scripts/` on sys.path, not the repository: without this the
# `retrieval_heads` import (and therefore the stale-schema warning) never worked.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger("retrieval_heads.summarize")


def load(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    # Every artifact goes through here, so this is the one place that can warn that
    # correlation.json / overlap.json / summaries predate the current schema.
    try:
        from retrieval_heads.provenance import warn_if_stale
    except ImportError:  # pragma: no cover - the repo root is added below
        return payload
    try:
        warn_if_stale(payload, str(path), log=log)
    except TypeError:  # pragma: no cover - an older warn_if_stale signature
        log.warning("%s could not be checked against the current schema", path)
    return payload


def at(seq: Any, index: Any, default: Any = None) -> Any:
    """``seq[index]`` or ``default`` when it is missing or out of range.

    Works for lists by position and for mappings by key.  A partial run must
    produce a partial report, not an IndexError/KeyError.
    """
    if isinstance(seq, Mapping):
        return seq.get(index, default)
    if not isinstance(seq, (list, tuple)) or not isinstance(index, int) or index >= len(seq):
        return default
    return seq[index]


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
        if not info or "name" not in info:
            continue
        rows.append(
            f"| {info.get('name', key)} | {at(info, 'num_layers')} "
            f"| {at(info, 'num_scoreable_layers')} | {at(info, 'n_scoreable_heads')} "
            f"| {at(info, 'head_dim')} | {at(info, 'is_hybrid')} |"
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
            sparsity = at(summary, "sparsity", {}) or {}
            heads = at(summary, "top_heads", []) or []
            top = at(heads, 0, {}) or {}
            n = at(sparsity, "n_heads", 0)
            buckets = at(sparsity, "thresholds", {}) or {}
            lo = at(buckets, "0.1", {}) or {}
            hi = at(buckets, "0.5", {}) or {}
            instances = at(summary, "n_instances", "?")
            # The >0.1/>0.5 buckets are fixed by sparsity(); `score_threshold` is
            # the threshold the run itself used.
            out.append(
                f"| {at(summary, 'model', key)} | {pairing} | {instances} "
                f"| {at(summary, 'n_instances_recited', 'n/a')}/{instances} "
                f"| {fmt(at(summary, 'mean_needle_recall'), 3)} "
                f"| `{top.get('head', '?')}` | {fmt(top.get('score'))} "
                f"| {fmt(at(lo, 'n'))}/{n or 'n/a'} "
                f"({100 * at(lo, 'frac', 0.0):.1f}%) "
                f"| {fmt(at(hi, 'n'))}/{n or 'n/a'} "
                f"({100 * at(hi, 'frac', 0.0):.1f}%) |"
            )
            if at(summary, "sparsity_recited"):
                rec = at(summary, "sparsity_recited")
                rec_lo = at(at(rec, "thresholds", {}) or {}, "0.1", {}) or {}
                out.append(
                    f"| _{at(summary, 'model', key)} (recited only)_ | {pairing} | "
                    f"{at(rec, 'n_heads', 0)} | | | "
                    f"{at(at(summary, 'top_heads_recited', []) or [{}], 0, {}).get('head', '?')} | | "
                    f"{fmt(at(rec_lo, 'n'))} | |"
                )
    return out


def cross_model_section(root: Path, keys: list[str]) -> list[str]:
    """Read the cross-model artifacts, so their staleness warning is not skipped."""
    correlation = load(root / "correlation.json")
    overlap = load(root / "overlap.json")
    out: list[str] = []
    if correlation:
        if correlation.get("caveat"):
            out.append(f"> **caveat**: {correlation['caveat']}")
        labels = correlation.get("labels") or []
        out.append(f"- correlation: mode `{at(correlation, 'mode', '?')}`, models: "
                   f"{', '.join(labels) if labels else 'n/a'}")
    if overlap:
        out.append(f"- head overlap: Jaccard {fmt(at(overlap, 'jaccard'), 3)} at threshold "
                   f"{at(overlap, 'threshold')}, mode `{at(overlap, 'mode', '?')}`, "
                   f"shared {len(at(overlap, 'shared', []) or [])} head(s)")
    return out


def pairing_section(root: Path, keys: list[str]) -> list[str]:
    out = ["| model | top-10 overlap between pairings | next_step top heads | same_step top heads |",
           "|---|---|---|---|"]
    for key in keys:
        summary = load(root / key / "summary_next_step.json")
        if not summary or "pairing_comparison" not in summary:
            continue
        comparison = summary["pairing_comparison"]
        primary = at(at(comparison, "primary", {}), "next_step", [])
        secondary = at(at(comparison, "secondary", {}), "same_step", [])
        if not primary and not secondary:
            continue
        overlap = at(comparison, "overlap")
        top_k = at(comparison, "top_k")
        label = f"{overlap}/{top_k}" if overlap is not None else "n/a"
        out.append(
            f"| {at(summary, 'model', key)} | {label} "
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
        k_values = at(curve, "k_values", []) or []
        # Older artifacts have no k_effective; those runs never hit the cap, so
        # falling back to the requested K is exact for them.
        k_eff = curve.get("k_effective") or k_values
        for i, k in enumerate(k_values):
            eff = k_eff[i] if i < len(k_eff) else k
            label = str(k) if eff == k else f"{k} →{eff}"
            exact = at(curve.get("retrieval_exact_match"), i)
            rand_exact = at(curve.get("random_exact_match_mean"), i)
            std = at(curve.get("retrieval_std"), i)
            retrieval = fmt(at(curve.get("retrieval"), i), 1)
            if std:
                retrieval += f" ±{fmt(std, 1)}"
            out.append(
                f"| {curve.get('model', key)} | {label} | {100 * eff / n_heads:.1f}% "
                f"| {retrieval} | {fmt(exact, 1)} "
                f"| {fmt(at(curve.get('random_mean'), i), 1)} | {fmt(rand_exact, 1)} |"
            )
        out.append(f"| | baseline | | {fmt(curve.get('baseline'), 1)} "
                   f"| {fmt(curve.get('baseline_exact_match'), 1)} | | |")
        n_samples = curve.get("n_samples")
        if n_samples is not None:
            out.append(f"| | _{key}: {n_samples} samples per point; "
                       f"± is the spread across them_ | | | | | |")
        if curve.get("random_control_contaminated"):
            out.append(f"| | _{key}: random control was **contaminated** "
                       f"(no head below the threshold)_ | | | | | |")
    return out


def aligned_section(root: Path, keys: list[str]) -> list[str]:
    """Strict in-order matching (`credits_aligned`), which used to be write-only."""
    out = ["| model | pairing | strict-aligned top heads (mean score) |", "|---|---|---|"]
    for key in keys:
        for pairing in ("next_step", "same_step"):
            summary = load(root / key / f"summary_{pairing}.json")
            if not summary:
                continue
            rows = summary.get("aligned_top_heads") or []
            if not rows:
                continue
            heads = ", ".join(f"{r['head']} ({fmt(r['aligned_score'], 2)})" for r in rows[:5])
            out.append(f"| {summary['model']} | {pairing} | {heads} |")
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
        for i, k in enumerate(at(ablation, "k_values", []) or []):
            full = at(ablation.get("full_attention"), i)
            linear = at(ablation.get("linear_attention"), i)
            out.append(f"| {k} | {fmt(full, 1)} | {fmt(linear, 1)} |")
        out.append("")
    return out


def task_section(root: Path, keys: list[str]) -> list[str]:
    out = []
    for key in keys:
        qa = load(root / key / "task_qa.json")
        if qa:
            out.append(f"**{key} — extractive QA**: baseline F1 {fmt(at(qa, 'baseline_f1'), 1)} "
                       f"over {at(qa, 'n_samples', '?')} samples")
            out.append("")
            out.append("| K | % heads | retrieval F1 (drop) | random F1 (drop) |")
            out.append("|---|---|---|---|")
            n_heads = qa.get("n_scoreable_heads") or 0
            for k, row in (qa.get("by_k") or {}).items():
                eff = row.get("k_effective", int(k))
                pct = f"{100 * eff / n_heads:.1f}%" if n_heads else ""
                label = str(k) if eff == int(k) else f"{k} →{eff}"
                out.append(f"| {label} | {pct} | {fmt(at(row, 'retrieval_f1'), 1)} "
                           f"({fmt(at(row, 'drop_retrieval'), 1)}) | "
                           f"{fmt(at(row, 'random_f1_mean'), 1)} "
                           f"({fmt(at(row, 'drop_random'), 1)}) |")
            out.append("")
        cot = load(root / key / "task_cot.json")
        if cot:
            out.append(f"**{key} — chain-of-thought**: K={at(cot, 'k', '?')}, "
                       f"{at(cot, 'n_samples', '?')} samples")
            out.append("")
            out.append("| variant | baseline | retrieval masked | random masked |")
            out.append("|---|---|---|---|")
            for variant, row in (cot.get("results") or {}).items():
                out.append(f"| {variant} | {fmt(at(row, 'baseline'), 1)} "
                           f"| {fmt(at(row, 'retrieval_masked'), 1)} "
                           f"| {fmt(at(row, 'random_masked_mean'), 1)} |")
            out.append("")
    return out


def correlation_section(root: Path) -> list[str]:
    correlation = load(root / "correlation.json")
    if not correlation:
        return []
    out = [f"Retrieval-score correlation (mode `{at(correlation, 'mode', '?')}`):", ""]
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
        ("Cross-model agreement", cross_model_section(root, keys)),
        ("Strict-aligned matching", aligned_section(root, keys)),
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
