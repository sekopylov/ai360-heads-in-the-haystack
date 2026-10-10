"""
The levels of a hard_niah run side by side, for a fixed set of heads. Reads the results only, no model.

python -m rh.levels results/new/qwen3_hard/B/none --mask_file results/new/qwen3_detect/head_score_copy_count.json --top 20

For every family and level: the number of samples, the mean ROUGE-1 recall, and the head metrics of rh.metrics
averaged over the samples and over the --top best heads of --mask_file (without --mask_file: over all heads):
    copy        copy_count: copying of the answer from the needle
    needle      needle_sentence_mass: attention on the needle sentence
    compet      competing_mass: attention on all competing inserts together
    max         competing_max_mass: attention on the strongest competing insert
    share       needle / (needle + compet): 1 when the heads look at the needle only
    copy_c      copy_competing: copying from the competing inserts
    all         needle_sentence_mass over all heads, to compare with `needle`

Every sample is used, the successful and the failed ones: ROUGE does not tell a right answer from a confused one on
the levels with twins. The run must keep the attention (--save compact), so it is a run without a mask.
"""
import argparse
import json
import os

import numpy as np

from . import spool
from .run import ranked_heads

COLUMNS = (("copy", "copy_count"), ("needle", "needle_sentence_mass"), ("compet", "competing_mass"), ("max", "competing_max_mass"),
           ("copy_c", "copy_competing"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out", help="--out folder of rh.run with the task hard_niah")
    parser.add_argument("--mask_file", default=None, help="head_score json that ranks the heads")
    parser.add_argument("--top", type=int, default=20, help="number of the best heads of --mask_file")
    args = parser.parse_args()

    with open(f"{args.out}/samples.jsonl", encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]
    run_json = f"{args.out}/spool/run.json"
    run = spool.load_json(run_json) if os.path.exists(run_json) else {}
    groups = {}
    for s in samples:
        path = f"{args.out}/heads/{s['id']}.npz"
        if "level" not in s:
            raise SystemExit(f"{args.out} is not a run of hard_niah: its samples have no level")
        if os.path.exists(path):
            with np.load(path) as d:
                groups.setdefault((s["family"], s["level"]), []).append((s, {name: d[name] for _, name in COLUMNS if name in d.files}))
    if not groups:
        raise SystemExit("this run has no head metrics (it was made with --save none)")

    shape = next(iter(groups.values()))[0][1]["copy_count"].shape
    layers = run.get("attn_layers") or list(range(shape[0]))
    chosen = np.ones(shape, dtype=bool)
    if args.mask_file:
        chosen[:] = False
        for layer, head in ranked_heads(args.mask_file)[:args.top]:
            chosen[layers.index(layer), head] = True
    print(f"{int(chosen.sum())} heads" + (f": the best {args.top} of {args.mask_file}" if args.mask_file else ": all"))

    rows = []
    for (family, level), items in sorted(groups.items()):
        row = {"family": family, "level": level, "samples": len(items), "rouge": float(np.mean([s["rouge1_recall"] for s, _ in items]))}
        for short, name in COLUMNS:
            if all(name in heads for _, heads in items):
                values = np.stack([heads[name] for _, heads in items]).mean(0)
                row[short] = float(values[chosen].mean())
                if short == "needle":
                    row["all"] = float(values.mean())
        if "needle" in row and "compet" in row:
            row["share"] = row["needle"] / (row["needle"] + row["compet"]) if row["needle"] + row["compet"] else float("nan")
        rows.append(row)

    names = ["copy", "needle", "compet", "max", "share", "copy_c", "all"]
    print(f"{'family':<10} {'level':<16} {'samples':>7} {'rouge':>6} " + " ".join(f"{n:>7}" for n in names))
    for r in rows:
        print(f"{r['family']:<10} {r['level']:<16} {r['samples']:>7} {r['rouge']:>6.1f} " + " ".join(f"{r[n]:>7.3f}" if n in r else f"{'-':>7}" for n in names))
    spool.save_json(f"{args.out}/levels.json", {"mask_file": args.mask_file, "top": args.top if args.mask_file else None, "levels": rows})
