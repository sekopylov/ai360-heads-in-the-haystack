"""
Figures in the style of Wu et al. 2024 + a summary for the question
"are there retrieval heads among linear-attention (Gated DeltaNet) heads, or only among full-attention heads?"

  python plots.py --head_score head_score/Qwen3.5-0.8B.json --niah_dir results/niah --out figs

fig1_head_score_map.png      retrieval score of every head (layer x head), linear vs full layers
fig2_score_distribution.png  share of heads with score >= x (linear vs full) + max score per layer
fig3_niah_K<k>.png           NIAH heatmaps (paper style): baseline / top-K / random-K, per head type
fig4_k_sweep.png             mean NIAH score vs number of masked heads (top vs random, linear vs full)
summary.md                   numbers behind the figures
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

C_FULL, C_LIN = "#2a78d6", "#eb6834"  # categorical slots 1, 2
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NIAH_CMAP = LinearSegmentedColormap.from_list("niah", ["#F0496E", "#EBB839", "#0CD79F"])  # as in the paper's notebook
SEQ_CMAP = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#9cc2ee", "#2a78d6", "#0d3a73"])

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 11,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def load_scores(path):
    with open(path) as f:
        raw = json.loads(f.readline())
    mean = {k: float(np.mean(v)) if len(v) else 0.0 for k, v in raw.items()}
    meta_path = re.sub(r"(\.contrib)?\.json$", ".meta.json", path)
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else None
    return mean, meta


def kinds_from(meta, mean):
    if meta:
        return meta["kinds"], meta["layer_kinds"], meta["n_heads"]
    # fallback: Qwen3.5 pattern (every 4th layer is full attention)
    L = 1 + max(int(k.split("-")[0]) for k in mean)
    layer_kinds = ["full" if (l + 1) % 4 == 0 else "linear" for l in range(L)]
    n_heads = [1 + max(int(k.split("-")[1]) for k in mean if int(k.split("-")[0]) == l) for l in range(L)]
    return {k: layer_kinds[int(k.split("-")[0])] for k in mean}, layer_kinds, n_heads


# ------------------------------------------------------------------ fig 1
def fig_head_map(mean, layer_kinds, n_heads, out):
    L, Hmax = len(layer_kinds), max(n_heads)
    M = np.full((L, Hmax), np.nan)
    for k, v in mean.items():
        l, h = map(int, k.split("-"))
        M[l, h] = v
    fig, ax = plt.subplots(figsize=(8.5, 7))
    cmap = SEQ_CMAP.copy()
    cmap.set_bad("#ffffff")
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=max(0.1, np.nanmax(M)), aspect="auto")
    ax.set_xticks(range(Hmax))
    ax.set_yticks(range(L))
    ax.set_yticklabels([f"{l}  {'full' if t == 'full' else 'lin'}" for l, t in enumerate(layer_kinds)])
    for l, t in enumerate(layer_kinds):
        lab = ax.get_yticklabels()[l]
        lab.set_color(C_FULL if t == "full" else C_LIN)
        lab.set_fontweight("bold" if t == "full" else "normal")
        if t == "full":
            ax.add_patch(plt.Rectangle((-0.5, l - 0.5), n_heads[l], 1, fill=False, ec=C_FULL, lw=1.5))
    for l in range(L):
        for h in range(n_heads[l]):
            if M[l, h] >= 0.1:
                ax.text(h, l, f"{M[l, h]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if M[l, h] > 0.5 * np.nanmax(M) else INK)
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_title("Retrieval score per head (values >= 0.1 printed; framed rows = full attention)")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("mean retrieval score")
    fig.savefig(out)
    plt.close(fig)


# ------------------------------------------------------------------ fig 2
def fig_distribution(mean, kind_of, layer_kinds, out):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    xs = np.linspace(0, 1, 201)
    for kind, col in (("full", C_FULL), ("linear", C_LIN)):
        v = np.array([s for k, s in mean.items() if kind_of[k] == kind])
        if len(v) == 0:
            continue
        frac = np.array([(v >= x).mean() * 100 for x in xs])
        frac[frac == 0] = np.nan
        a.plot(xs, frac, color=col, lw=2, label=f"{kind} ({len(v)} heads, {(v >= 0.1).sum()} with score>=0.1)")
    a.axvline(0.1, color=INK2, lw=1, ls=":")
    a.set_yscale("log")
    a.set_ylim(0.2, 120)
    a.text(0.11, 0.3, "0.1 = paper's threshold", color=INK2, fontsize=8)
    a.set_xlabel("retrieval score x")
    a.set_ylabel("% of heads with score >= x")
    a.set_title("Retrieval heads are rare: distribution by head type")
    a.legend(frameon=False, fontsize=8.5)
    a.grid(axis="y", color=GRID, lw=0.6)

    L = len(layer_kinds)
    mx = [max([s for k, s in mean.items() if int(k.split("-")[0]) == l] or [0]) for l in range(L)]
    cols = [C_FULL if t == "full" else C_LIN for t in layer_kinds]
    b.bar(range(L), mx, color=cols, width=0.8, edgecolor="white", linewidth=1)
    b.set_xticks(range(L))
    b.set_xlabel("layer")
    b.set_ylabel("max retrieval score in layer")
    b.set_title("Strongest head per layer")
    b.grid(axis="y", color=GRID, lw=0.6)
    b.set_axisbelow(True)
    from matplotlib.patches import Patch
    b.legend(handles=[Patch(color=C_LIN, label="linear attention (Gated DeltaNet)"),
                      Patch(color=C_FULL, label="full attention")], frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ------------------------------------------------------------------ NIAH results
def load_runs(niah_dir):
    runs = {}
    for path in sorted(glob.glob(os.path.join(niah_dir, "*.jsonl"))):
        name = os.path.basename(path)[:-6]
        rows, meta = [], {}
        for line in open(path):
            d = json.loads(line)
            if "meta" in d:
                meta = d["meta"]
            else:
                rows.append(d)
        if rows:
            runs[name] = (rows, meta)
    return runs


def parse_run(name):
    if name == "baseline":
        return ("baseline", None, 0, None)
    m = re.match(r"(linear|full|all)_(top|random)(\d+)(?:_s(\d+))?$", name)
    if not m:
        return None
    return (m.group(1), m.group(2), int(m.group(3)), None if m.group(4) is None else int(m.group(4)))


def pivot(rows):
    Ls = sorted({r["context_length"] for r in rows})
    Ds = sorted({r["depth_percent"] for r in rows})
    P = np.full((len(Ds), len(Ls)), np.nan)
    acc = defaultdict(list)
    for r in rows:
        acc[(r["depth_percent"], r["context_length"])].append(r["score"])
    for (d, l), v in acc.items():
        P[Ds.index(d), Ls.index(l)] = np.mean(v)
    return P, Ls, Ds


def avg_runs(runs, names):
    rows = [r for n in names for r in runs[n][0]]
    return rows


def niah_heatmap(ax, rows, title, cbar=False):
    P, Ls, Ds = pivot(rows)
    im = ax.imshow(P, cmap=NIAH_CMAP, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(Ls)))
    ax.set_xticklabels([f"{l / 1000:.0f}k" if l >= 1000 else str(l) for l in Ls], rotation=45, fontsize=7)
    ax.set_yticks(range(len(Ds)))
    ax.set_yticklabels([f"{d:.0f}" for d in Ds], fontsize=7)
    ax.set_title(f"{title}\nmean score {np.nanmean(P):.1f}", fontsize=9.5)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def fig_niah(runs, K, out):
    parsed = {n: parse_run(n) for n in runs}
    panels = []
    if "baseline" in runs:
        panels.append(("No masking", ["baseline"]))
    for kind in ("linear", "full", "all"):
        top = [n for n, p in parsed.items() if p and p[0] == kind and p[1] == "top" and p[2] == K]
        rnd = [n for n, p in parsed.items() if p and p[0] == kind and p[1] == "random" and p[2] == K]
        label = {"linear": "linear-attn", "full": "full-attn", "all": "any"}[kind]
        if top:
            panels.append((f"Mask top-{K} {label} retrieval heads", top))
        if rnd:
            panels.append((f"Mask {K} random {label} heads (avg {len(rnd)} seeds)", rnd))
    if len(panels) <= 1:
        return False
    n = len(panels)
    cols = min(n, 3 if n > 4 else n)
    rows_ = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_, cols, figsize=(5.2 * cols, 4.3 * rows_), squeeze=False,
                             gridspec_kw={"hspace": 0.55, "wspace": 0.25})
    im = None
    for ax, (title, names) in zip(axes.flat, panels):
        im = niah_heatmap(ax, avg_runs(runs, names), title)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("depth percent")
    for ax in axes[-1, :]:
        ax.set_xlabel("context length (tokens)")
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label("score (ROUGE-1 recall)")
    fig.suptitle('Needle-in-a-Haystack with masked heads ("Dolores Park" needle)', fontweight="bold")
    fig.savefig(out)
    plt.close(fig)
    return True


def sweep_table(runs):
    """{(kind, mode): {K: [mean score per run/seed]}}"""
    tab = defaultdict(lambda: defaultdict(list))
    for name, (rows, _) in runs.items():
        p = parse_run(name)
        if not p or p[0] == "baseline":
            continue
        tab[(p[0], p[1])][p[2]].append(np.mean([r["score"] for r in rows]))
    return tab


def fig_sweep(runs, out):
    tab = sweep_table(runs)
    if not tab:
        return False
    base = np.mean([r["score"] for r in runs["baseline"][0]]) if "baseline" in runs else None
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    allK = set()
    for kind, col in (("linear", C_LIN), ("full", C_FULL), ("all", "#1baf7a")):
        for mode, ls, mk in (("top", "-", "o"), ("random", "--", "s")):
            d = tab.get((kind, mode))
            if not d:
                continue
            Ks = sorted(d)
            allK |= set(Ks)
            m = np.array([np.mean(d[k]) for k in Ks])
            ax.plot(Ks, m, ls=ls, marker=mk, ms=6, color=col, lw=2, mec="white", mew=1.2,
                    label=f"{kind}: {'top-K retrieval' if mode == 'top' else 'K random'}")
            if mode == "random" and any(len(d[k]) > 1 for k in Ks):
                lo = [min(d[k]) for k in Ks]
                hi = [max(d[k]) for k in Ks]
                ax.fill_between(Ks, lo, hi, color=col, alpha=0.12, lw=0)
            ax.annotate(f"{kind} {mode}", (Ks[-1], m[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK2)
    if base is not None:
        ax.axhline(base, color=INK2, lw=1, ls=":", label=f"no masking ({base:.1f})")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(allK))
    ax.set_xticklabels([str(k) for k in sorted(allK)])
    ax.set_ylim(0, 105)
    ax.set_xlabel("number of masked heads K")
    ax.set_ylabel("mean NIAH score")
    ax.set_title("Masking retrieval heads vs random heads (band = min..max over seeds)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=8, loc="lower left", handlelength=3.5)
    fig.savefig(out)
    plt.close(fig)
    return True


# ------------------------------------------------------------------ summary
def summary(mean, kind_of, runs, path, variant_note):
    lines = [f"# Retrieval heads in Qwen3.5: linear vs full attention\n", variant_note, ""]
    lines.append("| head type | #heads | max score | #score>=0.1 | %score>=0.1 | #score>=0.3 | mean score |")
    lines.append("|---|---|---|---|---|---|---|")
    for kind in ("linear", "full"):
        v = np.array([s for k, s in mean.items() if kind_of[k] == kind])
        if len(v):
            lines.append(f"| {kind} | {len(v)} | {v.max():.3f} | {(v >= .1).sum()} | {(v >= .1).mean() * 100:.1f}% | "
                         f"{(v >= .3).sum()} | {v.mean():.4f} |")
    lines += ["", "## Top-20 heads", "", "| rank | head (layer-head) | type | score |", "|---|---|---|---|"]
    for i, k in enumerate(sorted(mean, key=mean.get, reverse=True)[:20]):
        lines.append(f"| {i + 1} | {k} | {kind_of[k]} | {mean[k]:.3f} |")
    if runs:
        tab = sweep_table(runs)
        lines += ["", "## NIAH mean score after masking", ""]
        if "baseline" in runs:
            lines.append(f"no masking: **{np.mean([r['score'] for r in runs['baseline'][0]]):.1f}**\n")
        Ks = sorted({k for d in tab.values() for k in d})
        lines.append("| setting | " + " | ".join(f"K={k}" for k in Ks) + " |")
        lines.append("|---|" + "---|" * len(Ks))
        for (kind, mode), d in sorted(tab.items()):
            cells = [f"{np.mean(d[k]):.1f}" + (f" ±{np.std(d[k]):.1f}" if len(d[k]) > 1 else "") if k in d else "–"
                     for k in Ks]
            lines.append(f"| {kind} {mode} | " + " | ".join(cells) + " |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head_score", required=True)
    p.add_argument("--niah_dir", default="results/niah")
    p.add_argument("--out", default="figs")
    p.add_argument("--niah_k", default=None, help="comma list of K for NIAH heatmaps (default: all found)")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    mean, meta = load_scores(args.head_score)
    kind_of, layer_kinds, n_heads = kinds_from(meta, mean)
    for k in kind_of:
        mean.setdefault(k, 0.0)
    fig_head_map(mean, layer_kinds, n_heads, os.path.join(args.out, "fig1_head_score_map.png"))
    fig_distribution(mean, kind_of, layer_kinds, os.path.join(args.out, "fig2_score_distribution.png"))

    runs = load_runs(args.niah_dir) if os.path.isdir(args.niah_dir) else {}
    if runs:
        Ks = sorted({parse_run(n)[2] for n in runs if parse_run(n) and parse_run(n)[0] != "baseline"})
        if args.niah_k:
            Ks = [int(k) for k in args.niah_k.split(",")]
        for K in Ks:
            fig_niah(runs, K, os.path.join(args.out, f"fig3_niah_K{K}.png"))
        fig_sweep(runs, os.path.join(args.out, "fig4_k_sweep.png"))
    note = "Scores: " + ("|A|·||v|| (contrib)" if ".contrib." in args.head_score else "argmax of attention row (attn)")
    if meta:
        note += f"; detection used {meta['n_used']}/{meta['n_samples']} successful samples."
    summary(mean, kind_of, runs, os.path.join(args.out, "summary.md"), note)


if __name__ == "__main__":
    main()
