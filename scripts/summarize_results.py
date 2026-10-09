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
    warn_if_stale = _load_warn_if_stale()
    if warn_if_stale is None:
        log.warning("%s was NOT checked against the current schema: "
                    "retrieval_heads.provenance is unimportable here", path)
        return payload
    try:
        warn_if_stale(payload, str(path), log=log)
    except TypeError:  # pragma: no cover - an older warn_if_stale signature
        log.warning("%s could not be checked against the current schema", path)
    return payload


def _load_warn_if_stale():
    """`warn_if_stale` without importing the package (which pulls in torch).

    `import retrieval_heads.provenance` runs `retrieval_heads/__init__.py`, which
    imports `models` -> torch + transformers.  In an environment without torch that
    raised ImportError and the schema check was skipped *silently* -- exactly the
    check whose absence this script exists to report.  `provenance.py` itself only
    needs the standard library, so load it by path as a fallback.
    """
    try:
        from retrieval_heads.provenance import warn_if_stale

        return warn_if_stale
    except ImportError:
        pass
    try:
        import importlib.util

        path = REPO_ROOT / "retrieval_heads" / "provenance.py"
        spec = importlib.util.spec_from_file_location("_rh_provenance", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.warn_if_stale
    except Exception:  # noqa: BLE001 - never break the summarizer
        return None


def at(seq: Any, index: Any, default: Any = None) -> Any:
    """``seq[index]`` or ``default`` when it is missing or out of range.

    Works for lists by position and for mappings by key.  A partial run must
    produce a partial report, not an IndexError/KeyError.
    """
    if isinstance(seq, Mapping):
        return seq.get(index, default)
    if (not isinstance(seq, (list, tuple)) or not isinstance(index, int)
            or index < 0 or index >= len(seq)):
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
    out = ["| model | pairing | instances | recited | with copy | mean recall | top head | score | >0.1 | >0.5 |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    # Which positions criterion (2) searched is not a column: it is one value per
    # model/pairing, and it decides whether the score is the paper's `a in R^{|x|}`
    # or a lower bound that lets the question and the chat template compete.
    notes: list[str] = []
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
                f"| {at(summary, 'n_instances_with_copy', 'n/a')}/{instances} "
                f"| {fmt(at(summary, 'mean_needle_recall'), 3)} "
                f"| `{top.get('head', '?')}` | {fmt(top.get('score'))} "
                f"| {fmt(at(lo, 'n'))}/{n or 'n/a'} "
                f"({100 * at(lo, 'frac', 0.0):.1f}%) "
                f"| {fmt(at(hi, 'n'))}/{n or 'n/a'} "
                f"({100 * at(hi, 'frac', 0.0):.1f}%) |"
            )
            # `argmax_domain` is top-level in schema 7 and inside `config` before it.
            domain = at(summary, "argmax_domain") or at(at(summary, "config", {}) or {},
                                                        "argmax_domain")
            if domain:
                shift = at(summary, "argmax_domain_shift") or {}
                detail = ""
                if at(shift, "positions"):
                    detail = (f"; it moved on "
                              f"{100 * at(shift, 'share', 0.0):.1f}% of "
                              f"{at(shift, 'positions')} (layer, head, step) positions "
                              f"relative to the prompt domain")
                if domain == "prompt":
                    notes.append(
                        f"- `{key}`/{pairing}: criterion (2) searched the **prompt** "
                        f"domain, which includes the question and the chat template; "
                        f"the paper's `a in R^{{|x|}}` is the `haystack` domain "
                        f"(pass `--argmax-domain haystack`, the current default)"
                        f"{detail}"
                    )
                else:
                    notes.append(f"- `{key}`/{pairing}: criterion (2) searched the "
                                 f"**{domain}** domain{detail}")
                # The sink's position relative to `x` decides whether criterion (2)
                # is even reachable: with a chat template the sink precedes the
                # haystack, so it cannot win the argmax and the >0.1 share inflates.
                if at(summary, "sink_in_haystack") is False and domain == "haystack":
                    notes.append(
                        f"- `{key}`/{pairing}: the attention sink (sequence position 0) "
                        f"is **outside** the haystack span, so it cannot suppress "
                        f"criterion (2); the paper's template-free prompt puts it "
                        f"inside `x`"
                    )
            raw = at(summary, "sparsity_raw") or {}
            raw_lo = at(at(raw, "thresholds", {}) or {}, "0.1", {}) or {}
            if at(raw, "n_heads"):
                notes.append(
                    f"- `{key}`/{pairing}: the raw per-token denominator "
                    f"(`|k|` read literally, repeats counted) puts "
                    f"{fmt(at(raw_lo, 'n'))}/{at(raw, 'n_heads')} heads above 0.1, "
                    f"against {fmt(at(lo, 'n'))}/{n or 'n/a'} under `|unique(k)|`"
                )
            # Schema 8 scores every captured argmax domain in the same pass, so the
            # run says how much the "sparse" headline depends on the position set
            # instead of leaving it to a re-run.
            by_domain = at(summary, "sparsity_by_domain") or {}
            if len(by_domain) > 1:
                parts = []
                for name, sp in sorted(by_domain.items()):
                    b = at(at(sp, "thresholds", {}) or {}, "0.1", {}) or {}
                    parts.append(f"`{name}` {fmt(at(b, 'n'))}/{at(sp, 'n_heads')} "
                                 f"({100 * at(b, 'frac', 0.0):.1f}%)")
                notes.append(f"- `{key}`/{pairing}: the >0.1 share by argmax domain: "
                             + ", ".join(parts))
                overlap = at(summary, "domain_ranking_overlap") or {}
                for name, data in sorted((at(overlap, "by_domain", {}) or {}).items()):
                    notes.append(
                        f"- `{key}`/{pairing}: {at(data, 'overlap')}/"
                        f"{at(overlap, 'top_k')} of the `{at(overlap, 'primary_domain')}` "
                        f"top heads are also top under `{name}` (the masking arm ranks "
                        f"by the primary domain, so this is how much a domain change "
                        f"would change what it masks)"
                    )
            if at(summary, "sparsity_recited"):
                rec = at(summary, "sparsity_recited")
                rec_buckets = at(rec, "thresholds", {}) or {}
                rec_lo = at(rec_buckets, "0.1", {}) or {}
                rec_hi = at(rec_buckets, "0.5", {}) or {}
                rec_n = at(rec, "n_heads", 0)
                rec_top = at(at(summary, "top_heads_recited", []) or [{}], 0, {}) or {}
                out.append(
                    f"| _{at(summary, 'model', key)} (recited only)_ | {pairing} | "
                    f"{at(summary, 'n_instances_recited', 'n/a')} | (same) | (same) "
                    f"| {fmt(at(summary, 'mean_needle_recall'), 3)} "
                    f"| `{rec_top.get('head', '?')}` | {fmt(rec_top.get('score'))} "
                    f"| {fmt(at(rec_lo, 'n'))}/{rec_n or 'n/a'} "
                    f"({100 * at(rec_lo, 'frac', 0.0):.1f}%) "
                    f"| {fmt(at(rec_hi, 'n'))}/{rec_n or 'n/a'} "
                    f"({100 * at(rec_hi, 'frac', 0.0):.1f}%) |"
                )
    if notes:
        out += [""] + notes
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
        if at(overlap, "comparable") is False:
            # `shared=0` here means "the layouts cannot be compared", not "no shared
            # heads"; printing the count alone reads as the latter.
            out.append("- head overlap: not comparable (different layer/head layouts); "
                       "Jaccard suppressed")
        else:
            out.append(f"- head overlap: Jaccard {fmt(at(overlap, 'jaccard'), 3)} at "
                       f"threshold {at(overlap, 'threshold')}, mode "
                       f"`{at(overlap, 'mode', '?')}`, "
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
            # `control_exhausted` marks a point where the random arm drew the whole
            # sub-threshold pool, i.e. an unmatched comparison; it must not be read
            # like the others.
            if at(curve.get("control_exhausted", []) or [], i):
                label += " ⚠"
            out.append(
                f"| {curve.get('model', key)} | {label} | {100 * eff / n_heads:.1f}% "
                f"| {retrieval} | {fmt(exact, 1)} "
                f"| {fmt(at(curve.get('random_mean'), i), 1)} | {fmt(rand_exact, 1)} |"
            )
        if any(curve.get("control_exhausted") or []):
            out.append("| | _⚠ = the retrieval arm hit the control-pool size, so the "
                       "random arm drew the whole pool (unmatched)_ | | | | | |")
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
        out.append(f"**{key}** — {at(ablation, 'n_full_layers')} full-attention layers, "
                   f"{at(ablation, 'n_linear_layers')} linear layers, "
                   f"baseline f1={fmt(at(ablation, 'baseline'), 1)}")
        out.append("")
        out.append("| K | full-attention masked | linear masked |")
        out.append("|---|---|---|")
        full_std = at(ablation, "full_attention_std", []) or []
        linear_std = at(ablation, "linear_attention_std", []) or []
        for i, k in enumerate(at(ablation, "k_values", []) or []):
            full = at(ablation.get("full_attention"), i)
            linear = at(ablation.get("linear_attention"), i)
            # The spread matters here: at K=4 the two arms are 11.2 +/- 2.1 and
            # 14.6 +/- 18.8, i.e. not distinguishable.  The PDF had error bars; the
            # markdown table did not.
            out.append(f"| {k} | {fmt(full, 1)} ±{fmt(at(full_std, i), 1)} "
                       f"| {fmt(linear, 1)} ±{fmt(at(linear_std, i), 1)} |")
        out.append("")
        # The committed artifact predates `distinct_subsets`, so do not claim it is
        # there: say what a current run records and what an older one cannot.
        has_distinct = "distinct_subsets" in ablation
        out.append("_± is the spread across the sampled layer subsets (n_trials)"
                   + ("; `distinct_subsets` says how many were distinct._"
                      if has_distinct else
                      "; this artifact predates `distinct_subsets`, so the number of "
                      "distinct subsets is not recorded._"))
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
            has_spread = False
            for k, row in (qa.get("by_k") or {}).items():
                eff = row.get("k_effective", int(k))
                pct = f"{100 * eff / n_heads:.1f}%" if n_heads else ""
                label = str(k) if eff == int(k) else f"{k} →{eff}"
                retrieval = fmt(at(row, "retrieval_f1"), 1)
                std = at(row, "retrieval_f1_std")
                if std is not None:
                    # Per-sample spread of the retrieval arm (the random arm's ± is
                    # across trials, so the two are not the same statistic).
                    retrieval += f" ±{fmt(std, 1)}"
                    has_spread = True
                out.append(f"| {label} | {pct} | {retrieval} "
                           f"({fmt(at(row, 'drop_retrieval'), 1)}) | "
                           f"{fmt(at(row, 'random_f1_mean'), 1)} "
                           f"({fmt(at(row, 'drop_random'), 1)}) |")
            if has_spread:
                out.append("| | _± is the retrieval arm's spread across samples "
                           "(older artifacts have no per-sample scores)_ | | |")
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
    labels = at(correlation, "labels", []) or []
    values = at(correlation, "values", []) or []
    if not labels or not values:
        return out
    out.append("| | " + " | ".join(labels) + " |")
    out.append("|---" * (len(labels) + 1) + "|")
    for i, label in enumerate(labels):
        row = at(values, i, []) or []
        cells = " | ".join(fmt(at(row, j)) for j in range(len(labels)))
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
