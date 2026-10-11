"""
Makes run folders on a smaller grid out of finished runs on a larger one, without the model: keeps the samples of
the chosen context lengths and needle depths and everything that belongs to them. The new folders look as if the runs
had been started on the small grid, so rh.sweep, rh.levels and the notebooks read them as usual, and rh.run with the
small grid continues them: it generates only the samples that are missing.

python -m rh.regrid results/new/qwen3_hard results/new/qwen3_hard_fast --lengths 1000,10158,19316,30000

The source is <src>/<group>/<run>/ (e.g. qwen3_hard/B/top60); the same layout is written under <dst>. The source is
not changed. A sample of a grid point is the same prompt on both grids when the longest context is the same: the
haystack is read up to the longest context, and the places of the inserts depend only on the name of the sample.
"""
import argparse
import collections
import glob
import json
import os
import shutil

from . import metrics, spool

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="folder with <group>/<run> subfolders of rh.run")
    parser.add_argument("dst", help="folder for the reduced copies, the same layout")
    parser.add_argument("--lengths", type=lambda s: [int(x) for x in s.split(",")], required=True, help="context lengths to keep, comma separated")
    parser.add_argument("--depths", type=lambda s: [int(x) for x in s.split(",")], default=None, help="needle depths to keep; default: all")
    parser.add_argument("--groups", type=lambda s: s.split(","), default=None, help="only these <group> subfolders, e.g. A,B; default: all")
    parser.add_argument("--runs", type=lambda s: s.split(","), default=None, help="only these runs, e.g. none,top40,top60,random60; default: all")
    parser.add_argument("--model_path", default=None, help="write this model path into the settings of the copies: rh.run continues "
                        "a folder only with the same --model_path, set it when the model now lies elsewhere")
    args = parser.parse_args()

    for folder in sorted(glob.glob(f"{args.src}/*/*/")):
        group, name = os.path.normpath(folder).split(os.sep)[-2:]
        if (args.groups and group not in args.groups) or (args.runs and name not in args.runs) or not os.path.exists(f"{folder}/samples.jsonl"):
            continue
        out = f"{args.dst}/{group}/{name}"
        if os.path.exists(f"{out}/samples.jsonl"):
            print(f"{group}/{name}: {out} already has results, left as it is")
            continue
        run_json = f"{folder}/spool/run.json"
        if not os.path.exists(run_json):
            print(f"{group}/{name}: no spool/run.json, skipped")
            continue
        run = spool.load_json(run_json)
        old = run.get("args", {})
        if max(old.get("lengths") or [0]) != max(args.lengths):
            # with another longest context the haystack is read to another length: the prompts may differ
            print(f"{group}/{name}: its longest context is {max(old.get('lengths') or [0])}, the new one {max(args.lengths)}: skipped")
            continue
        with open(f"{folder}/samples.jsonl", encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        kept = [r for r in rows if r["context_length"] in args.lengths and (args.depths is None or r["depth_percent"] in args.depths)]

        os.makedirs(f"{out}/spool", exist_ok=True)

        def write(tmp):
            with open(tmp, "w", encoding="utf-8") as f:
                for r in kept:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        spool.write_atomic(f"{out}/samples.jsonl", write)
        copied = collections.Counter()
        for sub, ext in (("heads", "npz"), ("outputs", "json")):
            for r in kept:
                path = f"{folder}/{sub}/{r['id']}.{ext}"
                if os.path.exists(path):
                    os.makedirs(f"{out}/{sub}", exist_ok=True)
                    shutil.copyfile(path, f"{out}/{sub}/{r['id']}.{ext}")
                    copied[sub] += 1
        # the settings as rh.run would have written them for the small grid
        new = dict(old, lengths=args.lengths, out=out, spool=None)
        if args.depths is not None:
            new["depths"] = args.depths
        if args.model_path:
            new["model_path"] = args.model_path
        spool.save_json(f"{out}/spool/run.json", dict(run, args=new, regrid={"from": os.path.normpath(folder), "samples_there": len(rows)}))
        metrics.aggregate(f"{out}/spool", out)

        levels = collections.Counter(str(r.get("level", "-")) for r in kept)
        uneven = "" if len(set(levels.values())) <= 1 else "  NOT EVEN, the source run was not finished: " + ", ".join(f"{k} {v}" for k, v in sorted(levels.items()))
        print(f"{group}/{name}: {len(kept)} of {len(rows)} samples kept, heads {copied['heads']}, outputs {copied['outputs']} -> {out}{uneven}")
