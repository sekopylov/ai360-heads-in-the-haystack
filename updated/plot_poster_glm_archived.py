"""Re-export the two archived poster layouts with transparent backgrounds."""
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from plot_poster_glm_top40 import OUT, ROOT, RANKINGS, load_local, plt, rescore
from plot_qrscore_experiments import DEPTHS, LENGTHS, load_archive


def main():
    conditions = ('top40', 'top80', 'top100')
    datasets = {}
    for slug, label, metric in RANKINGS:
        datasets[label], _ = load_local(slug, metric, conditions)
    qr_path = ROOT / 'colab-results/glm4-6v-flash-no-thinking-paul-graham-qrscore-top-mask-results.zip'
    qr, _, _, _ = load_archive(qr_path, 40)
    datasets['QRScore'] = rescore([r for r in qr if r['condition'] in conditions])
    matrices = {k: [] for k in (40, 80, 100)}
    prompts = {}
    expected_grid = {(l, d) for l in LENGTHS for d in DEPTHS}
    for label, rows in datasets.items():
        for k in matrices:
            group = [r for r in rows if r['condition'] == f'top{k}']
            lookup = {(r['context_length'], r['depth_percent']): r['rouge1_recall'] for r in group}
            assert len(group) == len(lookup) == 110 and set(lookup) == expected_grid
            for row in group:
                key = row['context_length'], row['depth_percent']
                assert prompts.setdefault(key, row['prompt_sha256']) == row['prompt_sha256']
            values = np.array([[lookup[l, d] for l in LENGTHS] for d in DEPTHS])
            matrices[k].append((label, float(values.mean()), values))
    # These archived layouts predate the muted palette and bottom control.
    cmap = LinearSegmentedColormap.from_list('archived_poster',
        [(0, '#d90019'), (.25, '#ff2525'), (.5, '#ffe52c'), (1, '#20ed39')])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'figure.facecolor': 'none', 'axes.facecolor': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})

    def panel(ax, label, score, values, k=None):
        im = ax.imshow(values, cmap=cmap, vmin=0, vmax=100, aspect='auto', interpolation='nearest')
        prefix = f'top-{k} · ' if k is not None else ''
        ax.set_title(f'{prefix}{label}\nСредний recall: {score:.1f}%', fontsize=15, pad=10)
        ax.set_xlabel('Длина контекста, токенов', fontsize=12)
        ax.set_xticks(range(len(LENGTHS)), [f'{l:,}'.replace(',', ' ') for l in LENGTHS],
                      fontsize=8, rotation=45, ha='right')
        ax.set_yticks(range(len(DEPTHS)), DEPTHS, fontsize=10)
        return im

    def save(fig, name):
        fig.supxlabel('Paul Graham / San Francisco · no thinking · 110 одинаковых prompts · baseline: 100% recall\n'
                      'Красный = низкое качество, зелёный = высокое · legacy_uniform masking при decode.', fontsize=10)
        for ext in ('png', 'svg', 'pdf'):
            fig.savefig(OUT / f'{name}.{ext}', dpi=220, bbox_inches='tight', transparent=True)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6.7), sharey=True, layout='constrained')
    for ax, (label, score, values) in zip(axes, matrices[100]):
        im = panel(ax, label, score, values)
    axes[0].set_ylabel('Глубина иголки, %', fontsize=12)
    cb = fig.colorbar(im, ax=axes, ticks=(0, 25, 50, 75, 100), shrink=.9, pad=.015)
    cb.set_label('ROUGE-1 recall, %', fontsize=12)
    fig.suptitle('GLM-4.6V-Flash · маскирование top-100 голов', fontsize=21, weight='bold')
    save(fig, 'top100_three_rankings')

    fig, axes = plt.subplots(3, 3, figsize=(16, 16), sharex=True, sharey=True, layout='constrained')
    for row_idx, k in enumerate((40, 80, 100)):
        for ax, (label, score, values) in zip(axes[row_idx], matrices[k]):
            im = panel(ax, label, score, values, k=k)
        axes[row_idx, 0].set_ylabel('Глубина иголки, %', fontsize=12)
    cb = fig.colorbar(im, ax=axes, ticks=(0, 25, 50, 75, 100), shrink=.8, pad=.012)
    cb.set_label('ROUGE-1 recall, %', fontsize=12)
    fig.suptitle('GLM-4.6V-Flash · top-40 → top-80 → top-100', fontsize=22, weight='bold')
    save(fig, 'masking_progression_40_80_100')
    print('Re-exported top100 and archived 40/80/100 progression with transparency.')


if __name__ == '__main__':
    main()
