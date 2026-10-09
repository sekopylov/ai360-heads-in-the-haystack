"""
One table over several runs of rh.run, e.g. a masking sweep. Reads the results only, no model.

python -m rh.sweep results/new/qwen3_mask_* --limit 256
"""
import argparse
import glob
import json
import os

import numpy as np

from . import spool

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="--out folders of rh.run; patterns are allowed")
    parser.add_argument("--limit", type=int, default=None, help="evaluate the answers cut at this token limit; default: whole answers")
    args = parser.parse_args()

    rows = []
    for out in sorted(p for pattern in args.runs for p in (glob.glob(pattern) or [pattern])):
        if not os.path.exists(f"{out}/samples.jsonl"):
            continue
        with open(f"{out}/samples.jsonl", encoding="utf-8") as f:
            samples = [json.loads(l) for l in f if l.strip()]
        if args.limit is not None:
            at = [s["by_limit"][str(args.limit)] for s in samples]
        else:
            at = [{"rouge1_recall": s["rouge1_recall"], "success": s["success"], "truncated": not s.get("stopped", True)} for s in samples]
        run = spool.load_json(f"{out}/spool/run.json") if os.path.exists(f"{out}/spool/run.json") else {}
        block = run.get("block_list") or []
        size = run.get("group_size", 1)
        whole = len({(l, h // size) for l, h in block})
        rows.append({"run": os.path.basename(os.path.normpath(out)), "mask": run.get("mask", "?"), "heads": len(block), "groups": whole,
                     "samples": len(at), "success": float(np.mean([a["success"] for a in at])),
                     "rouge": float(np.mean([a["rouge1_recall"] for a in at])), "truncated": float(np.mean([a["truncated"] for a in at]))})

    width = max([len(r["run"]) for r in rows] + [3])
    print(f"{'run':<{width}} {'mask':>13} {'heads':>6} {'groups':>7} {'samples':>8} {'success':>8} {'rouge':>7} {'truncated':>10}")
    for r in sorted(rows, key=lambda r: (r["mask"], r["heads"])):
        print(f"{r['run']:<{width}} {r['mask']:>13} {r['heads']:>6} {r['groups']:>7} {r['samples']:>8} {r['success']:>8.1%} {r['rouge']:>7.1f} {r['truncated']:>10.1%}")
