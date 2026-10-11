"""
Brings finished runs of rh.run to another grid of context lengths and needle depths, without the model. The new
folders look as if the runs had been started on that grid, so rh.sweep, rh.levels and the notebooks read them as
usual, and rh.run with that grid continues them: it generates only the samples that are missing.

A smaller grid out of a larger one: only the samples of the chosen lengths are kept
python -m rh.regrid results/new/qwen3_hard results/new/qwen3_hard_fast --lengths 1000,10158,19316,30000

A larger grid out of parts: the same runs computed on different lengths are joined
python -m rh.regrid results/new/qwen3_hard_fast results/new/qwen3_hard_rest results/new/qwen3_hard_full

The sources are <src>/<group>/<run>/ (e.g. qwen3_hard/B/top60); the same layout is written under the last folder.
The sources are not changed. A sample of a grid point is the same prompt on every grid: the haystack does not depend
on the lengths of the run, and the places of the inserts depend only on the name of the sample. Runs are joined only
when everything but the grid is the same in their settings: the model, the task, the levels, the masked heads.
"""
import argparse
import collections
import glob
import json
import os
import shutil

from . import metrics, spool

GRID = {"lengths", "depths", "s_len", "e_len", "context_intervals"}   # what regrid changes
PLACE = {"out", "spool", "buffer_gb", "chunk_mb", "with_metrics", "limit", "every", "model_path"}   # where and how, not what

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src", nargs="+", help="one or several folders with <group>/<run> subfolders of rh.run")
    parser.add_argument("dst", help="folder for the result, the same layout")
    parser.add_argument("--lengths", type=lambda s: [int(x) for x in s.split(",")], default=None, help="context lengths to keep, comma separated; default: all that the sources have")
    parser.add_argument("--depths", type=lambda s: [int(x) for x in s.split(",")], default=None, help="needle depths to keep; default: all")
    parser.add_argument("--groups", type=lambda s: s.split(","), default=None, help="only these <group> subfolders, e.g. A,B; default: all")
    parser.add_argument("--runs", type=lambda s: s.split(","), default=None, help="only these runs, e.g. none,top40,top60,random60; default: all")
    parser.add_argument("--model_path", default=None, help="write this model path into the settings of the result: rh.run continues "
                        "a folder only with the same --model_path, set it when the model now lies elsewhere")
    args = parser.parse_args()

    found = collections.defaultdict(list)   # (group, run) -> source folders that have it
    for src in args.src:
        for folder in sorted(glob.glob(f"{src}/*/*/")):
            group, name = os.path.normpath(folder).split(os.sep)[-2:]
            if (not args.groups or group in args.groups) and (not args.runs or name in args.runs) and os.path.exists(f"{folder}/samples.jsonl"):
                found[(group, name)].append(os.path.normpath(folder))

    for (group, name), folders in sorted(found.items()):
        out = f"{args.dst}/{group}/{name}"
        if os.path.exists(f"{out}/samples.jsonl"):
            print(f"{group}/{name}: {out} already has results, left as it is")
            continue
        runs = {f: spool.load_json(f"{f}/spool/run.json") for f in folders if os.path.exists(f"{f}/spool/run.json")}
        if len(runs) < len(folders):
            print(f"{group}/{name}: no spool/run.json in {[f for f in folders if f not in runs]}, skipped")
            continue
        first = runs[folders[0]]
        same = lambda r: ({k: v for k, v in r.get("args", {}).items() if k not in GRID | PLACE}, r.get("block_list"))
        other = [f for f in folders[1:] if same(runs[f]) != same(first)]
        if other:
            a, b = same(first)[0], same(runs[other[0]])[0]
            print(f"{group}/{name}: {other[0]} is another experiment than {folders[0]}, not joined: "
                  + ("; ".join(f"{k}: {a.get(k)!r} / {b.get(k)!r}" for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)) or "other masked heads"))
            continue

        kept, origin, there = [], {}, 0
        for folder in folders:
            with open(f"{folder}/samples.jsonl", encoding="utf-8") as f:
                rows = [json.loads(l) for l in f if l.strip()]
            there += len(rows)
            for r in rows:
                if r["id"] not in origin and (args.lengths is None or r["context_length"] in args.lengths) and (args.depths is None or r["depth_percent"] in args.depths):
                    origin[r["id"]] = folder
                    kept.append(r)
        if not kept:
            print(f"{group}/{name}: no samples of this grid, skipped")
            continue

        os.makedirs(f"{out}/spool", exist_ok=True)

        def write(tmp):
            with open(tmp, "w", encoding="utf-8") as f:
                for r in kept:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        spool.write_atomic(f"{out}/samples.jsonl", write)
        copied = collections.Counter()
        for sub, ext in (("heads", "npz"), ("outputs", "json")):
            for r in kept:
                path = f"{origin[r['id']]}/{sub}/{r['id']}.{ext}"
                if os.path.exists(path):
                    os.makedirs(f"{out}/{sub}", exist_ok=True)
                    shutil.copyfile(path, f"{out}/{sub}/{r['id']}.{ext}")
                    copied[sub] += 1
        # the settings as rh.run would have written them for this grid
        lengths = args.lengths or sorted({r["context_length"] for r in kept})
        depths = args.depths or sorted({d for run in runs.values() for d in (run.get("args", {}).get("depths") or [])}) or None
        new = dict(first.get("args", {}), lengths=lengths, depths=depths, out=out, spool=None)
        if args.model_path:
            new["model_path"] = args.model_path
        spool.save_json(f"{out}/spool/run.json", dict(first, args=new, regrid={"from": folders, "samples_there": there}))
        metrics.aggregate(f"{out}/spool", out)

        cells = collections.Counter((str(r.get("level", "-")), r["context_length"]) for r in kept)
        levels = collections.Counter(str(r.get("level", "-")) for r in kept)
        gaps = sorted({f"{level} len {length}" for level in levels for length in lengths if cells[(level, length)] < max(cells.values())})
        note = "" if not gaps else f"  NOT COMPLETE, fewer samples at: {', '.join(gaps[:8])}{' ...' if len(gaps) > 8 else ''}"
        print(f"{group}/{name}: {len(kept)} samples from {len(folders)} source(s) ({there} there), lengths {len(lengths)}, "
              f"heads {copied['heads']}, outputs {copied['outputs']} -> {out}{note}")
