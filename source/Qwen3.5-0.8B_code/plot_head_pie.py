#!/usr/bin/env python
"""
Пай-чарт: сколько голов с retrieval rate < 0.1, 0.1..0.4 и > 0.4.

rate головы = среднее по успешным пробам из head_score/<model>.json
(ключи "layer-head" -> список скоров).

  python plot_head_pie.py --scores head_score/Qwen3.5-0.8B-Base.json
"""
import argparse
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", default="plots/head_pie.png")
    ap.add_argument("--low", type=float, default=0.1)
    ap.add_argument("--high", type=float, default=0.4)
    a = ap.parse_args()

    with open(a.scores, encoding="utf-8") as f:
        counter = json.load(f)
    rates = {k: (float(np.mean(v)) if len(v) else 0.0) for k, v in counter.items()}
    n_samples = max((len(v) for v in counter.values()), default=0)

    groups = {
        f"rate < {a.low}": [k for k, r in rates.items() if r < a.low],
        f"{a.low} ≤ rate ≤ {a.high}": [k for k, r in rates.items() if a.low <= r <= a.high],
        f"rate > {a.high}": [k for k, r in rates.items() if r > a.high],
    }
    total = len(rates)
    print(f"Голов всего: {total}, успешных проб: {n_samples}")
    for name, heads in groups.items():
        heads_sorted = sorted(heads, key=lambda k: -rates[k])
        print(f"  {name}: {len(heads)}  {[f'{k}={rates[k]:.2f}' for k in heads_sorted[:15]]}")

    sizes = [len(h) for h in groups.values()]
    colors = ["#c9ced6", "#f2a93b", "#2f6fdb"]
    labels = list(groups.keys())

    def fmt(p):
        n = int(round(p * total / 100))
        return f"{n}\n({p:.0f}%)" if n > 0 else ""

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.pie(sizes, labels=None, colors=colors, autopct=fmt, startangle=90, counterclock=False,
           wedgeprops=dict(edgecolor="white", linewidth=1.5), textprops=dict(fontsize=11))
    ax.legend([f"{l}  —  {s}" for l, s in zip(labels, sizes)], loc="lower center",
              bbox_to_anchor=(0.5, -0.12), frameon=False)
    ax.set_title(f"Retrieval score голов ({total} голов full-attention слоёв)")
    ax.axis("equal")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(a.out, dpi=160)
    print("Сохранено:", a.out)


if __name__ == "__main__":
    main()