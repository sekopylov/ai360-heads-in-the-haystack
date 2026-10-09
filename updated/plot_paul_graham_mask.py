"""Rescore downloaded San Francisco masking results and build a PNG/SVG gallery."""
from pathlib import Path
import csv
import json
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from rouge_score.rouge_scorer import RougeScorer
from rouge_score.tokenizers import DefaultTokenizer

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'datasphere-results/qwen3-8b-no-thinking-paul-graham-attention-mass-mask-san-francisco'
OUT = ROOT / 'report-assets/qwen3-8b-no-thinking-paul-graham-mask'
HEADS = (0, 20, 40, 60, 80, 100, 120)
METRICS = ('rouge1_recall', 'rouge2_recall', 'rouge3_recall', 'rouge123_mean_recall', 'exact_match')
LABELS = ('ROUGE-1 recall', 'ROUGE-2 recall', 'ROUGE-3 recall', 'Среднее ROUGE-1/2/3 recall', 'Точная фраза иголки')
COLORS = ('#168577', '#3574b4', '#8a67b0', '#d7a33e', '#df8050', '#cc5265', '#663d68')
BG, INK = '#fafaf8', '#263449'
CMAP = LinearSegmentedColormap.from_list('needle_score', ['#f7e4df', '#e6a07c', '#f0e8cf', '#91c6b5', '#16796e'])


def rescore(rows):
    scorer = RougeScorer(['rouge1', 'rouge2', 'rouge3'], use_stemmer=True)
    tokenizer = DefaultTokenizer(use_stemmer=False)
    for row in rows:
        scores = scorer.score(row['expected_answer'], row['model_response'])
        for n in (1, 2, 3):
            score = scores[f'rouge{n}']
            for component in ('recall', 'precision', 'fmeasure'):
                row[f'rouge{n}_{component}'] = getattr(score, component) * 100
        for component in ('recall', 'precision', 'fmeasure'):
            row[f'rouge123_mean_{component}'] = mean(row[f'rouge{n}_{component}'] for n in (1, 2, 3))
        target = tokenizer.tokenize(row['expected_answer'])
        response = tokenizer.tokenize(row['model_response'])
        row['exact_match'] = 100.0 if target and any(response[i:i + len(target)] == target for i in range(len(response) - len(target) + 1)) else 0.0
    return rows


def load():
    rows = []
    for path in sorted(SOURCE.glob('evaluation/context-42/evaluation/*/results/*_results.json')):
        row = json.loads(path.read_text())
        condition = path.parent.parent.name
        row.update(condition=condition, blocked_heads=0 if condition == 'baseline' else int(condition.removeprefix('top')), source=str(path.relative_to(ROOT)))
        rows.append(row)
    lengths = sorted({r['context_length'] for r in rows})
    depths = sorted({r['depth_percent'] for r in rows})
    keys = {(r['blocked_heads'], r['context_length'], r['depth_percent']) for r in rows}
    assert len(rows) == len(keys) == 1400
    assert len(lengths) == 20 and len(depths) == 10
    assert keys == {(h, l, d) for h in HEADS for l in lengths for d in depths}
    # All conditions must evaluate identical prepared prompts at each grid position.
    for length in lengths:
        for depth in depths:
            assert len({r['prompt_sha256'] for r in rows if r['context_length'] == length and r['depth_percent'] == depth}) == 1
    return rescore(rows), lengths, depths


def save(fig, name):
    fig.set_layout_engine('constrained', rect=(0, .04, 1, .96))
    fig.text(.02, .012, 'Qwen3-8B · no thinking · Paul Graham / San Francisco · старый mass-рейтинг · legacy_uniform · выход ≤256 токенов', fontsize=8, color='#687587')
    for ext in ('png', 'svg'):
        fig.savefig(OUT / 'plots' / f'{name}.{ext}', dpi=170, bbox_inches='tight')
    plt.close(fig)


def main():
    rows, lengths, depths = load()
    (OUT / 'plots').mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'figure.facecolor': BG, 'axes.facecolor': BG, 'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    (OUT / 'scores.json').write_text(json.dumps({'source': str(SOURCE), 'scope': 'saved model_response; no prompt; no rescoring of rankings', 'cases': rows}, ensure_ascii=False, indent=2))
    fields = ['condition', 'blocked_heads', 'context_length', 'depth_percent', 'score', *[f'rouge{n}_{c}' for n in (1, 2, 3) for c in ('recall', 'precision', 'fmeasure')], *[f'rouge123_mean_{c}' for c in ('recall', 'precision', 'fmeasure')], 'exact_match', 'finish_reason', 'test_duration_seconds', 'source']
    with (OUT / 'cases.csv').open('w') as f:
        writer = csv.DictWriter(f, fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
    summary = []
    gallery = []
    for heads in HEADS:
        group = [r for r in rows if r['blocked_heads'] == heads]
        summary.append({'blocked_heads': heads, 'cases': len(group), **{m: mean(r[m] for r in group) for m in METRICS}})
        fig, axes = plt.subplots(2, 3, figsize=(17, 12), layout='constrained')
        fig.suptitle(f'Извлечение иголки · отключено {heads} голов', fontsize=22, weight='bold')
        for ax, metric, label in zip(axes.flat, METRICS, LABELS):
            lookup = {(r['context_length'], r['depth_percent']): r[metric] for r in group}
            matrix = np.array([[lookup[l, d] for d in depths] for l in lengths])
            im = ax.imshow(matrix, vmin=0, vmax=100, cmap=CMAP, aspect='auto')
            ax.set(title=label, xlabel='Глубина иголки, %', ylabel='Длина контекста, токенов')
            ax.set_xticks(range(len(depths)), [f'{d:g}' for d in depths])
            ax.set_yticks(range(len(lengths)), [f'{l:,}'.replace(',', ' ') for l in lengths])
            for i in range(len(lengths)):
                for j in range(len(depths)):
                    ax.text(j, i, f'{matrix[i,j]:.0f}', ha='center', va='center', fontsize=7, color='white' if matrix[i,j] >= 80 else INK)
            fig.colorbar(im, ax=ax, shrink=.75, label='Оценка, %')
        axes.flat[-1].axis('off')
        axes.flat[-1].text(.05, .8, '\n\n'.join(f'{label}: {mean(r[m] for r in group):.2f}%' for m, label in zip(METRICS, LABELS)), fontsize=13)
        name = f'heads_{heads:03d}_heatmaps'; save(fig, name); gallery.append((f'{heads} отключённых голов — все метрики', name))
    for metric, label in zip(METRICS, LABELS):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), layout='constrained')
        fig.suptitle(label, fontsize=20, weight='bold')
        for heads, color in zip(HEADS, COLORS):
            group = [r for r in rows if r['blocked_heads'] == heads]
            for ax, field, grid in ((axes[0], 'context_length', lengths), (axes[1], 'depth_percent', depths)):
                ax.plot(grid, [mean(r[metric] for r in group if r[field] == x) for x in grid], color=color, marker='o', markersize=3, linewidth=2, label=f'{heads} голов')
        axes[0].set(xlabel='Длина контекста, токенов', ylabel='Средняя оценка, %', title='Усреднение по 10 глубинам')
        axes[1].set(xlabel='Глубина иголки, %', title='Усреднение по 20 длинам')
        axes[2].bar([str(h) for h in HEADS], [mean(r[metric] for r in rows if r['blocked_heads'] == h) for h in HEADS], color=COLORS)
        axes[2].set(xlabel='Отключено голов', title='Вся сетка: 200 примеров на условие')
        for ax in axes:
            ax.set_ylim(0, 105); ax.grid(axis='y', alpha=.18); ax.set_axisbelow(True)
        axes[1].legend(ncol=2, fontsize=9)
        name = f'comparison_{metric}'; save(fig, name); gallery.append((label + ' — сравнение условий', name))
    with (OUT / 'summary.csv').open('w') as f:
        writer = csv.DictWriter(f, list(summary[0])); writer.writeheader(); writer.writerows(summary)
    lines = ['# Paul Graham: masking без thinking', '', 'Исходные JSON скачаны из задания bt1k7k78e6t20kj241br. Проверены все 1400 уникальных результатов и совпадение prompt_sha256 между условиями.', '', 'Сетка: 20 длин × 10 глубин × 7 условий. Рейтинг голов: старый thinking survey attention-mass, НЕ новый no-thinking survey. Маска legacy_uniform применялась при decode.', '', 'ROUGE-1/2/3 — overlap словесных n-грамм со stemming, шкала 0–100. Основной композит — арифметическое среднее трёх recall; это предложенная сводная метрика, не стандартный ROUGE. Precision и F1 тоже сохранены в CSV/JSON: recall не штрафует лишний текст. Exact match — наличие полной эталонной фразы подряд после нормализации регистра и пунктуации, без stemming; это не семантическая проверка.', '', 'Оценка идёт по сохранённому model_response. Исходные score и результаты не изменены. Контрольные слова/перефразирование могут влиять на оценки; графики не доказывают функциональную специализацию голов.', '', '| Отключено | ROUGE-1 | ROUGE-2 | ROUGE-3 | Среднее 1/2/3 | Exact match |', '|---:|---:|---:|---:|---:|---:|']
    for row in summary:
        lines.append('| ' + str(row['blocked_heads']) + ' | ' + ' | '.join(f'{row[m]:.2f}' for m in METRICS) + ' |')
    lines += ['', 'Данные: [cases.csv](cases.csv), [summary.csv](summary.csv), [scores.json](scores.json). Все картинки доступны в PNG и SVG.', '']
    for title, name in gallery:
        lines += [f'## {title}', '', f'![{title}](plots/{name}.png)', '', f'[SVG](plots/{name}.svg)', '']
    (OUT / 'README.md').write_text('\n'.join(lines))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
