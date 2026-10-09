"""
A whole masking sweep with one load of the model: the run without a mask and, for every number of heads, the best
heads, random heads and random groups of heads. Every configuration is an ordinary run of rh.run in its own subfolder
of --out (none, top20, random20, groups20, ...), so it can be continued, inspected and summarized in the same way.

python -m rh.mask_sweep --model_path Qwen/Qwen3-8B --needles data/needles_eval.jsonl \
    --mask_file results/new/qwen3_detect/head_score_copy_count.json --out results/new/qwen3_mask

Started again with the same arguments it skips what is done and continues what was interrupted. The state of every
configuration is in <out>/sweep_status.json; the table over the results is python -m rh.sweep <out>/*.
"""
import argparse
import json
import os
import sys
import time
import traceback

from . import run, spool

KINDS = {"top": "--mask_top", "random": "--mask_random", "groups": "--mask_random_groups"}


def configurations(kinds, counts, seed):
    """(subfolder, extra arguments of rh.run) for every configuration, the run without a mask first."""
    configs = [("none", [])] if "none" in kinds else []
    for kind in ("top", "random", "groups"):
        if kind in kinds:
            for k in counts:
                name = f"{kind}{k}" + (f"_s{seed}" if seed and kind != "top" else "")
                configs.append((name, [KINDS[kind], str(k)] + (["--seed", str(seed)] if kind != "top" else [])))
    return configs


def count_results(out):
    path = f"{out}/samples.jsonl"
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for l in f if l.strip())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False, description="other arguments are passed to every run of rh.run: the task, its grid, --max_new_tokens")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--mask_file', type=str, required=True, help='head_score json of the detection run')
    parser.add_argument('--out', type=str, required=True, help='folder of the sweep, one subfolder per configuration')
    parser.add_argument('--counts', type=lambda s: [int(x) for x in s.split(',')], default=[20, 40, 60, 80, 100, 120],
                        help='numbers of masked heads')
    parser.add_argument('--kinds', type=lambda s: s.split(','), default=["none", "top", "random", "groups"],
                        help='none, top, random (scattered random heads), groups (random whole groups of heads)')
    parser.add_argument('--seed', type=int, default=0, help='seed of the random heads; a non-zero seed gets its own subfolders')
    parser.add_argument('--dtype', type=str, default="auto")
    args, passed = parser.parse_known_args()
    unknown = set(args.kinds) - {"none", *KINDS}
    if unknown:
        parser.error(f"unknown kinds: {sorted(unknown)}")

    # every configuration is checked before the model is loaded: wrong arguments or an occupied folder stop the sweep here
    plan = []
    for name, extra in configurations(args.kinds, args.counts, args.seed):
        argv = (["--model_path", args.model_path, "--dtype", args.dtype, "--out", f"{args.out}/{name}", "--save", "none", "--with_metrics"]
                + (["--mask_file", args.mask_file] if name != "none" else []) + extra + passed)
        run_parser = run.build_parser(argv)
        run_args = run_parser.parse_args(argv)
        problem = run.check(run_args, run_parser)
        if problem:
            parser.error(f"{name}: {problem}")
        plan.append((name, run_args))
    print(f"{len(plan)} configurations: {', '.join(name for name, _ in plan)}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    status_path = f"{args.out}/sweep_status.json"
    status = spool.load_json(status_path) if os.path.exists(status_path) else {}
    for name, state in status.items():
        if state.get("status") == "running":
            print(f"{name}: the previous sweep stopped in the middle of it, it will be continued", flush=True)

    from . import model as rh_model
    enc, model, attn_layers = rh_model.load(args.model_path, args.dtype)

    interrupted = False
    for name, run_args in plan:
        start_time = time.time()
        state = {"status": "running", "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        status[name] = state
        spool.save_json(status_path, status)
        print(f"\n===== {name} =====", flush=True)
        try:
            counts = run.execute(run_args, enc, model, attn_layers)
            results = count_results(run_args.out)
            state.update(counts, results=results)
            if counts.get("metrics_exit_code"):
                state.update(status="failed", error=f"rh.metrics ended with code {counts['metrics_exit_code']}")
            elif not counts["completed"] or results < counts["selected"]:
                # the loop ended, but some samples have no result: the metrics lost them or stopped early
                state.update(status="incomplete", error=f"{results} results of {counts['selected']} samples")
            else:
                state["status"] = "done"
        except KeyboardInterrupt:
            state.update(status="interrupted", error="stopped by the user", results=count_results(run_args.out))
            interrupted = True
        except Exception as error:
            state.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
                         results=count_results(run_args.out))
            print(state["traceback"], flush=True)
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        state["seconds"] = round(time.time() - start_time, 1)
        spool.save_json(status_path, status)
        print(f"===== {name}: {state['status']}, {state.get('results', 0)} results, {state['seconds']:.0f}s"
              + (f" | {state['error']}" if state.get("error") else ""), flush=True)
        if interrupted:
            break

    print("\nsweep summary", flush=True)
    planned = [name for name, _ in plan]
    for name in planned:
        state = status.get(name, {"status": "not started"})
        print(f"  {name:<14} {state['status']:<12} {state.get('results', 0):>5} results  {state.get('error', '')}", flush=True)
    bad = [name for name in planned if status.get(name, {}).get("status") != "done"]
    if bad:
        print(f"NOT FINISHED: {', '.join(bad)}. Start the same command again to continue.", flush=True)
        sys.exit(1)
    print("all configurations are done", flush=True)
