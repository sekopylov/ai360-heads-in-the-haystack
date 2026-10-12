"""Three poster donuts: fractions of GLM heads in shared mean-score bins."""
import csv
import json

import numpy as np
from matplotlib.patches import Patch

from plot_poster_glm_head_scores import LABELS, METRICS, OUT, load_scores, plt

NAME = 'head_score_distribution_three_metrics'
BIN_LABELS = ('0–0,1', '>0,1–0,5', '>0,5–1')


def bin_counts(values):
    values = np.asarray(values)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1.000001):
        raise ValueError('Invalid mean head scores')
    counts = [int(np.count_nonzero(values <= .1)),
              int(np.count_nonzero((values > .1) & (values <= .5))),
              int(np.count_nonzero(values > .5))]
    if sum(counts) != values.size:
        raise ValueError('Score bins do not cover all heads')
    return counts


def main():
    matrices, _, provenance = load_scores()
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'figure.facecolor': 'none', 'axes.facecolor': 'none'})
    # Bin midpoints on the same magma scale as the head heatmaps.
    colors = plt.get_cmap('magma')([.05, .3, .75])
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.5), layout='constrained')
    rows = []
    for ax, metric, label, values in zip(axes, METRICS, LABELS, matrices):
        counts = bin_counts(values)
        total = values.size
        nonzero = [idx for idx, count in enumerate(counts) if count]
        ax.pie([counts[idx] for idx in nonzero],
               colors=[colors[idx] for idx in nonzero], startangle=90, counterclock=False,
               wedgeprops=dict(width=.42, edgecolor='white', linewidth=1.7),
               autopct=lambda percent: f'{percent:.1f}%'.replace('.', ',') if percent >= 3 else '',
               pctdistance=.79, textprops=dict(color='white', fontsize=13))
        ax.set_title(label.replace('\n', ' '), fontsize=17, weight='normal', pad=16)
        legend_labels = [f'{bin_label}: {count} ({100*count/total:.2f}%)'.replace('.', ',')
                         for bin_label, count in zip(BIN_LABELS, counts)]
        ax.legend([Patch(facecolor=color) for color in colors], legend_labels,
                  loc='upper center', bbox_to_anchor=(.5, -.03), frameon=False, fontsize=12)
        rows.extend(dict(metric=metric, score_range=bin_label, heads=count,
                         percent=100*count/total, total_heads=total)
                    for bin_label, count in zip(BIN_LABELS, counts))
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(OUT / f'{NAME}.{ext}', dpi=220, bbox_inches='tight', transparent=True)
    plt.close(fig)
    with (OUT / 'head_score_distribution.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    provenance.update(figure=NAME, chart_type='donut', x_axis=None, y_axis=None,
                      head_axis_labels_displayed=False,
                      panel_titles=[label.replace('\n', ' ') for label in LABELS],
                      bins=['[0, 0.1]', '(0.1, 0.5]', '(0.5, 1]'],
                      bin_colors=[list(color) for color in colors],
                      note='Fractions of head counts, not fractions of summed score, attention, or accuracy.')
    (OUT / 'head_score_distribution_provenance.json').write_text(
        json.dumps(provenance, indent=2), encoding='utf-8')
    lines = ['# GLM: доли голов по диапазонам score', '',
             f'![Три распределения]({NAME}.png)', '',
             f'[SVG]({NAME}.svg) · [PDF для печати]({NAME}.pdf) · '
             '[CSV](head_score_distribution.csv) · [Источники](head_score_distribution_provenance.json)', '',
             'Фон PNG/SVG/PDF прозрачен, включая пустые центры кольцевых диаграмм.', '',
             'Три диаграммы — Retrieval Score, Attention mass, QRScore. Каждая включает все '
             '1280 Q-голов GLM; диапазоны среднего score одинаковы: [0; 0,1], (0,1; 0,5], (0,5; 1]. '
             'Это доли **количества голов**, не доли суммарного внимания и не качество ответов. '
             'Проценты всех диапазонов, включая маленькие секторы, подписаны под диаграммами.', '',
             'Использованы те же проверенные рейтинги по 27 совпадающим survey-prompts, что '
             'на [картах голов](HEAD_SCORES.md) и в выборе масок. Палитра magma взята из карт '
             'голов, цвета секторов соответствуют серединам диапазонов. Retrieval Score — '
             'краткая подпись multiset-версии. Границы: ровно 0,1 входит в первый диапазон, '
             'ровно 0,5 — во второй. Значения score не округляются до распределения.', '',
             'Воспроизведение из updated: `.venv/bin/python plot_poster_glm_score_pies.py`.', '']
    (OUT / 'HEAD_SCORE_DISTRIBUTION.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
