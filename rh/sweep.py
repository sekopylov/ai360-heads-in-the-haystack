"""
One table over several runs of rh.run, e.g. a masking sweep. Reads the results only, no model.

python -m rh.sweep results/new/qwen3_mask_* --limit 256

A run that holds several kinds of samples is split by fields of its samples, e.g. the levels of hard_niah:

python -m rh.sweep "results/new/qwen3_hard/*/*" --by level
python -m rh.sweep "results/new/qwen3_hard/B/*" --by family,level
"""
import argparse
import glob
import json
import os

import numpy as np

from . import spool
from .metrics import answer_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="--out folders of rh.run; patterns are allowed")
    parser.add_argument("--limit", type=int, default=None, help="evaluate the answers cut at this token limit; default: whole answers")
    parser.add_argument("--by", type=lambda s: s.split(","), default=[], help="fields of the samples to split every run by, comma separated: level or family,level")
    args = parser.parse_args()

    rows = []
    for out in sorted(p for pattern in args.runs for p in (glob.glob(pattern) or [pattern])):
        if not os.path.exists(f"{out}/samples.jsonl"):
            continue
        with open(f"{out}/samples.jsonl", encoding="utf-8") as f:
            samples = [json.loads(l) for l in f if l.strip()]
        run = spool.load_json(f"{out}/spool/run.json") if os.path.exists(f"{out}/spool/run.json") else {}
        block = run.get("block_list") or []
        size = run.get("group_size", 1)
        whole = len({(l, h // size) for l, h in block})
        parts = {}
        for s in samples:
            parts.setdefault(tuple(str(s.get(field, "-")) for field in args.by), []).append(s)
        for key, part in parts.items():
            if args.limit is not None:
                at = [s["by_limit"][str(args.limit)] for s in part]
            else:
                # the stricter metrics (ROUGE-2, ROUGE-3, the exact phrase) are computed here for the runs made before they were stored
                at = [{**(s if "rouge3_recall" in s and "exact" in s else answer_metrics(s["reference"], s["response"])),
                       "rouge1_recall": s["rouge1_recall"], "success": s["success"], "truncated": not s.get("stopped", True)} for s in part]
            rows.append({"by": key, "run": os.path.basename(os.path.normpath(out)), "mask": run.get("mask", "?"), "heads": len(block), "groups": whole,
                         "cap": (run.get("args") or {}).get("max_new_tokens"), "longest": max(s["n_tokens"] for s in part),
                         "samples": len(at), "success": float(np.mean([a["success"] for a in at])),
                         "rouge": float(np.mean([a["rouge1_recall"] for a in at])), "truncated": float(np.mean([a["truncated"] for a in at])),
                         **{name: float(np.mean([a[field] for a in at])) if all(field in a for a in at) else float("nan")
                            for name, field in (("exact", "exact"), ("rouge2", "rouge2_recall"), ("rouge3", "rouge3_recall"))}})

    width = max([len(r["run"]) for r in rows] + [3])
    widths = [max([len(r["by"][i]) for r in rows] + [len(field)]) for i, field in enumerate(args.by)]
    split = "".join(f"{field:<{w}} " for field, w in zip(args.by, widths))
    print(f"{split}{'run':<{width}} {'mask':>13} {'heads':>6} {'groups':>7} {'cap':>5} {'longest':>8} {'samples':>8} {'success':>8} {'exact':>7} {'rouge':>7} {'rouge2':>7} {'rouge3':>7} {'truncated':>10}")
    previous = None
    for r in sorted(rows, key=lambda r: (r["by"], r["mask"] != "none", r["mask"], r["heads"]) if args.by else (r["mask"], r["heads"])):
        if args.by and previous not in (None, r["by"]):
            print()
        previous = r["by"]
        split = "".join(f"{value:<{w}} " for value, w in zip(r["by"], widths))
        print(f"{split}{r['run']:<{width}} {r['mask']:>13} {r['heads']:>6} {r['groups']:>7} {str(r['cap']):>5} {r['longest']:>8} {r['samples']:>8} {r['success']:>8.1%} {r['exact']:>7.1%} {r['rouge']:>7.1f} {r['rouge2']:>7.1f} {r['rouge3']:>7.1f} {r['truncated']:>10.1%}")
    caps = {r["cap"] for r in rows}
    if len(caps) > 1:
        print(f"WARNING: the runs have different caps of the answer length ({sorted(caps, key=str)}), their results are not comparable")
    for r in rows:
        if r["cap"] is not None and r["longest"] > r["cap"]:
            print(f"WARNING: {r['run']} has answers longer than its cap: the folder mixes samples of runs with different caps")
    if rows and not any(r["mask"] == "none" for r in rows):
        print("NOTE: there is no run without a mask among these: the effect of a mask is measured against it")
