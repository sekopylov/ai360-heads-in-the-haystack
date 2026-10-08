#!/usr/bin/env python
"""
Этап 4: графики результатов маскирования.

  figures/<model>_masking_bar.png      средний ROUGE-1 recall и accuracy по условиям
  figures/<model>_masking_heatmaps.png NIAH-карты (длина контекста x глубина иглы) по условиям
"""
import argparse
import glob
import json
import os

import numpy as np

from rh_common import DEFAULT_MODEL, model_tag, setup_logger, stage

ORDER = ["baseline", "random", "bottom", "top"]
TITLES = {"baseline": "без маскирования", "top": "top-K (наибольший score)",
          "bottom": "bottom-K (наименьший score)", "random": "случайные K"}
COLORS = {"baseline": "#6c757d", "top": "#d1495b", "bottom": "#2e86ab", "random": "#f2a541"}


def load_records(res_dir):
    recs = {}
    for f in sorted(glob.glob(os.path.join(res_dir, "*.jsonl"))):
        cond = os.path.basename(f)[:-6]
        recs[cond] = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    return recs


def grid(records, lengths, depths):
    """Средний score в ячейке (глубина, длина)."""
    m = np.full((len(depths), len(lengths)), np.nan)
    li = {int(v): i for i, v in enumerate(lengths)}
    di = {round(float(v), 4): i for i, v in enumerate(depths)}
    acc = {}
    for r in records:
        key = (di[round(r["depth_percent"], 4)], li[r["context_length"]])
        acc.setdefault(key, []).append(r["score"])
    for (i, j), v in acc.items():
        m[i, j] = np.mean(v)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--out_dir", default="figures")
    a = p.parse_args()
    log = setup_logger("04_plots")
    tag = model_tag(a.model_path)
    res_dir = f"results/masking/{tag}"
    os.makedirs(a.out_dir, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    with stage(log, "Загрузка результатов"):
        with open(os.path.join(res_dir, "summary.json"), encoding="utf-8") as f:
            data = json.load(f)
        meta, summ = data["meta"], data["conditions"]
        recs = load_records(res_dir)
        # объединяем случайные сиды в одно условие для карты
        rnd = [r for c, rs in recs.items() if c.startswith("random_s") for r in rs]
        if rnd:
            recs["random"] = rnd
        conds = [c for c in ORDER if c in summ]
        log.info(f"условия: {conds}; K={meta['k']}")

    with stage(log, "Столбчатая диаграмма"):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for ax, key, std_key, ylabel in ((axes[0], "mean_score", "std_score", "ROUGE-1 recall, %"),
                                         (axes[1], "accuracy", "std_accuracy", "accuracy, %")):
            vals = [summ[c][key] for c in conds]
            errs = [summ[c].get(std_key, 0) for c in conds]
            bars = ax.bar(range(len(conds)), vals, yerr=errs, capsize=5,
                          color=[COLORS[c] for c in conds], edgecolor="white")
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}", ha="center", fontsize=10)
            ax.set_xticks(range(len(conds)))
            ax.set_xticklabels([TITLES[c].replace("K", str(meta["k"])) for c in conds],
                               rotation=15, fontsize=9)
            ax.set_ylim(0, 110)
            ax.set_ylabel(ylabel)
            ax.spines[["top", "right"]].set_visible(False)
        axes[1].set_title(f"accuracy = доля проб с recall > {meta['threshold']:.0f}%", fontsize=10)
        fig.suptitle(f"{tag}: Needle-in-a-Haystack при выкидывании {meta['k']} голов")
        fig.tight_layout()
        out = os.path.join(a.out_dir, f"{tag}_masking_bar.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        log.info(f"сохранено: {out}")

    with stage(log, "NIAH-тепловые карты"):
        lengths, depths = meta["lengths"], meta["depths"]
        cmap = LinearSegmentedColormap.from_list("niah", ["#F0496E", "#EBB839", "#0CD79F"])
        fig, axes = plt.subplots(1, len(conds), figsize=(4.2 * len(conds), 4.2), squeeze=False)
        for ax, c in zip(axes[0], conds):
            m = grid(recs.get(c, []), lengths, depths)
            im = ax.imshow(m, cmap=cmap, vmin=0, vmax=100, aspect="auto")
            ax.set_title(f"{TITLES[c].replace('K', str(meta['k']))}\nсредний {summ[c]['mean_score']:.1f}",
                         fontsize=10)
            ax.set_xticks(range(len(lengths)))
            ax.set_xticklabels([f"{l // 1000}k" if l >= 1000 else str(l) for l in lengths], fontsize=7)
            ax.set_yticks(range(len(depths)))
            ax.set_yticklabels([f"{d:.0f}" for d in depths], fontsize=7)
            ax.set_xlabel("длина контекста")
        axes[0][0].set_ylabel("глубина иглы, %")
        fig.colorbar(im, ax=axes[0].tolist(), fraction=0.02, label="ROUGE-1 recall")
        out = os.path.join(a.out_dir, f"{tag}_masking_heatmaps.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"сохранено: {out}")


if __name__ == "__main__":
    main()