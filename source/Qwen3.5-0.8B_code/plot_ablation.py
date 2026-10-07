#!/usr/bin/env python
"""
Графики точности модели при блокировке голов (по results/<model>/ablation_samples.jsonl).

Три группы блокируемых голов:
  retrieval -- K голов с наибольшим retrieval rate (mode=top)
  обычные   -- K голов с наименьшим rate (mode=bottom)
  случайные -- K случайных голов вне top-K (mode=random, среднее и ± std по сидам)

Файлы в той же папке:
  ablation_accuracy.png   accuracy и ROUGE-1 recall от K (слева/справа)
  ablation_breakdown.png  accuracy по длине контекста и по глубине иглы для максимального K

  python plot_ablation.py --dir results/Qwen3.5-0.8B-Base
"""
import argparse
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SERIES = [  # mode, подпись, цвет (слоты 1, 3, 2 референсной палитры)
    ("top", "Retrieval heads (top-K)", "#2a78d6"),
    ("bottom", "Обычные головы (K с наименьшим rate)", "#1baf7a"),
    ("random", "Случайные головы", "#eb6834"),
]
INK, INK2, GRID, BASE = "#0b0b0b", "#52514e", "#e3e2dd", "#8a8983"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, length=0)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def load(d):
    with open(f"{d}/ablation_samples.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def per_seed(rows, mode, k, key):
    by = defaultdict(list)
    for r in rows:
        if r["mode"] == mode and r["k"] == k:
            by[r["seed"]].append(r[key] * (100 if key == "ok" else 1))
    return [float(np.mean(v)) for v in by.values()]


def make_plots(d):
    rows = load(d)
    base = [r for r in rows if r["mode"] == "baseline"]
    ks = sorted({r["k"] for r in rows if r["mode"] != "baseline"})
    modes = [s for s in SERIES if any(r["mode"] == s[0] for r in rows)]
    n_per = len(base)
    base_acc = np.mean([r["ok"] for r in base]) * 100 if base else float("nan")
    base_rouge = np.mean([r["rouge"] for r in base]) if base else float("nan")

    # --- 1. метрики от K ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, key, ylabel, b in ((axes[0], "ok", "Accuracy, % проб (ROUGE-1 recall > порога)", base_acc),
                               (axes[1], "rouge", "Средний ROUGE-1 recall", base_rouge)):
        style(ax)
        ax.axhline(b, color=BASE, linestyle="--", linewidth=1.5)
        ax.text(ks[-1], b, f" baseline {b:.0f}", color=INK2, va="bottom", ha="right", fontsize=9)
        for mode, label, color in modes:
            xs, ys, err = [0], [b], [0.0]
            for k in ks:
                v = per_seed(rows, mode, k, key)
                if v:
                    xs.append(k)
                    ys.append(np.mean(v))
                    err.append(np.std(v) if len(v) > 1 else 0.0)
            ax.errorbar(xs, ys, yerr=err, color=color, linewidth=2, marker="o", markersize=6,
                        markeredgecolor="white", markeredgewidth=1.5, capsize=3, label=label)
            ax.annotate(f"{ys[-1]:.0f}", (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=10, color=INK)
        ax.set_xticks([0] + ks)
        ax.set_xlabel("Число заблокированных голов K")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.set_xlim(-0.5, ks[-1] * 1.08)
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle(f"Блокировка голов Qwen3.5: retrieval vs обычные vs случайные  "
                 f"(проб на условие: {n_per}; random: среднее ± std по сидам)",
                 color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{d}/ablation_accuracy.png", dpi=160)
    plt.close(fig)

    # --- 2. разрезы по длине и глубине для максимального K ---
    kmax = ks[-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, field, xlabel in ((axes[0], "length", "Длина контекста, токены"),
                              (axes[1], "depth", "Глубина иглы, %")):
        style(ax)
        xs = sorted({r[field] for r in rows})

        def curve(sel):
            by = defaultdict(list)
            for r in sel:
                by[r[field]].append(r["ok"] * 100)
            return [np.mean(by[x]) if by[x] else np.nan for x in xs]

        ax.plot(xs, curve(base), color=BASE, linestyle="--", linewidth=1.5, label="baseline")
        for mode, label, color in modes:
            sel = [r for r in rows if r["mode"] == mode and r["k"] == kmax]
            ax.plot(xs, curve(sel), color=color, linewidth=2, marker="o", markersize=6,
                    markeredgecolor="white", markeredgewidth=1.5, label=f"{label}, K={kmax}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Accuracy, %")
        ax.set_ylim(-3, 103)
        ax.set_xticks(xs)
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle(f"Accuracy при K={kmax} по длине контекста и глубине иглы", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{d}/ablation_breakdown.png", dpi=160)
    plt.close(fig)
    print("Сохранено:", f"{d}/ablation_accuracy.png", f"{d}/ablation_breakdown.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="results/<model>")
    make_plots(ap.parse_args().dir)