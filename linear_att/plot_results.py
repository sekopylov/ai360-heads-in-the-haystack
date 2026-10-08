"""
Plots:
  1) number of heads by retrieval score bins  (<0.1, 0.1-0.4, >=0.4)  + layer x head heatmap
  2) NIAH accuracy vs number of masked heads (top-K retrieval vs K random) + NIAH heatmaps

  python plot_results.py --head_score head_score/mamba2-2.7b-hf.json \
                         --mask_results results/mask_mamba2-2.7b-hf_dt.json --out_dir figs
"""
import argparse
import json
import os

import matplotlib

from log_utils import add_logging_args, get_logger, setup_logging

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BINS = [("score < 0.1", None, 0.1), ("0.1 ≤ score < 0.4", 0.1, 0.4), ("score ≥ 0.4", 0.4, None)]


def load_score_matrix(path):
    d = json.load(open(path))
    keys = [tuple(int(t) for t in k.split("-")) for k in d]
    L = max(k[0] for k in keys) + 1
    H = max(k[1] for k in keys) + 1
    M = np.zeros((L, H))
    n_runs = 0
    for k, v in d.items():
        l, h = (int(t) for t in k.split("-"))
        M[l, h] = np.mean(v) if len(v) else 0.0
        n_runs = max(n_runs, len(v))
    return M, n_runs


def plot_counts(M, n_runs, title, out):
    s = M.ravel()
    counts = []
    for _, lo, hi in BINS:
        m = np.ones_like(s, dtype=bool)
        if lo is not None:
            m &= s >= lo
        if hi is not None:
            m &= s < hi
        counts.append(int(m.sum()))

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw=dict(width_ratios=[1, 1.1, 1.6]))
    ax = axes[0]
    colors = ["#9aa5b1", "#f0a202", "#d1495b"]
    bars = ax.bar([b[0] for b in BINS], counts, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("number of heads (log)")
    ax.set_title(f"{title}\n{s.size} heads, {n_runs} successful NIAH runs")
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c, f"{c}\n({100 * c / s.size:.2f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0.8, s.size * 3)
    ax.tick_params(axis="x", labelsize=9)

    ax = axes[1]
    srt = np.sort(s)[::-1]
    ax.plot(np.arange(1, len(srt) + 1), srt, color="#2e4057")
    ax.axhline(0.1, ls="--", c=colors[1], lw=1)
    ax.axhline(0.4, ls="--", c=colors[2], lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("head rank")
    ax.set_ylabel("retrieval score")
    ax.set_title("sorted retrieval scores")

    ax = axes[2]
    im = ax.imshow(M.T, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=max(M.max(), 1e-6))
    ax.set_xlabel("layer")
    ax.set_ylabel("head")
    ax.set_title("retrieval score per head")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    get_logger().info(f"saved {out}:  " + ", ".join(f"{b[0]}: {c}" for b, c in zip(BINS, counts)))
    top = np.argsort(-s)[:20]
    get_logger().info("top-20 heads (layer-head:score): " +
                      ", ".join(f"{i // M.shape[1]}-{i % M.shape[1]}:{s[i]:.3f}" for i in top))


def niah_matrix(runs):
    lens = sorted({r["len"] for r in runs})
    deps = sorted({r["depth"] for r in runs})
    A = np.full((len(deps), len(lens)), np.nan)
    acc = {}
    for r in runs:
        acc.setdefault((r["depth"], r["len"]), []).append(r["score"])
    for (d, l), v in acc.items():
        A[deps.index(d), lens.index(l)] = np.mean(v)
    return A, lens, deps


def plot_masking(res, title, out):
    top = {int(k): v["acc"] for k, v in res["top"].items()}
    ks = sorted(top)
    rnd_mean, rnd_std, rks = [], [], []
    for k in ks:
        if k == 0:
            rks.append(0), rnd_mean.append(top[0]), rnd_std.append(0)
            continue
        accs = [v["acc"] for v in res["random"].get(str(k), {}).values()]
        if accs:
            rks.append(k), rnd_mean.append(np.mean(accs)), rnd_std.append(np.std(accs))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(ks, [top[k] for k in ks], "o-", c="#d1495b", label="mask top-K retrieval heads")
    ax.errorbar(rks, rnd_mean, yerr=rnd_std, fmt="s--", c="#2e4057", capsize=3, label="mask K random heads")
    ax.set_xscale("symlog", linthresh=5)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("K (number of masked heads)")
    ax.set_ylabel("Needle-in-a-Haystack score (ROUGE-1 recall)")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    get_logger().info(f"saved {out}")
    for k, m, s in zip(rks, rnd_mean, rnd_std):
        get_logger().info(f"  K={k:4d}: top={top[k]:6.2f}   random={m:6.2f}±{s:.2f}")

    # NIAH heatmaps: no mask / top-Kmax / random-Kmax (paper-style)
    kmax = max(k for k in ks)
    panels = [("no masking", res["top"]["0"]["runs"])] if "0" in res["top"] else []
    panels.append((f"mask top-{kmax}", res["top"][str(kmax)]["runs"]))
    if res["random"].get(str(kmax)):
        panels.append((f"mask random-{kmax}", next(iter(res["random"][str(kmax)].values()))["runs"]))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4.2))
    axes = np.atleast_1d(axes)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("niah", ["#F0496E", "#EBB839", "#0CD79F"])
    for ax, (name, runs) in zip(axes, panels):
        A, lens, deps = niah_matrix(runs)
        im = ax.imshow(A, aspect="auto", cmap=cmap, vmin=0, vmax=100)
        ax.set_xticks(range(len(lens)))
        ax.set_xticklabels(lens, rotation=45)
        ax.set_yticks(range(len(deps)))
        ax.set_yticklabels(deps)
        ax.set_xlabel("context length (tokens)")
        ax.set_ylabel("needle depth (%)")
        ax.set_title(f"{name}: {np.nanmean(A):.1f}")
    fig.colorbar(im, ax=axes.tolist(), fraction=0.03)
    out2 = out.replace(".png", "_heatmaps.png")
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    get_logger().info(f"saved {out2}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head_score", default=None)
    ap.add_argument("--mask_results", nargs="*", default=[])
    ap.add_argument("--title", default=None)
    ap.add_argument("--out_dir", default="figs")
    add_logging_args(ap)
    args = ap.parse_args()
    setup_logging("plot_results", args)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.head_score:
        base = os.path.splitext(os.path.basename(args.head_score))[0]
        M, n_runs = load_score_matrix(args.head_score)
        plot_counts(M, n_runs, args.title or base, os.path.join(args.out_dir, f"{base}_counts.png"))
        soft = args.head_score.replace(".json", "_soft.json")
        if os.path.exists(soft) and soft != args.head_score:
            M, n_runs = load_score_matrix(soft)
            plot_counts(M, n_runs, (args.title or base) + " (soft score)",
                        os.path.join(args.out_dir, f"{base}_soft_counts.png"))
    for p in args.mask_results:
        res = json.load(open(p))
        base = os.path.splitext(os.path.basename(p))[0]
        plot_masking(res, args.title or base, os.path.join(args.out_dir, f"{base}.png"))


if __name__ == "__main__":
    main()