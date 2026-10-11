"""Build paired Ministral/GLM masking galleries without modifying source results."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np
from plot_paul_graham_mask import plt, rescore, METRICS, LABELS, CMAP, BG, INK
from retrieval_heads.experiment.scoring import rank_heads

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'report-assets/ministral3-8b-paul-graham-mask'
KS = (5, 10, 15, 20, 40, 60, 80, 100)
CONDITIONS = ('baseline', *(f'top{k}' for k in KS), *(f'bottom{k}' for k in KS))
LENGTHS = (1000, 4053, 7105, 10158, 13211, 16263, 19316, 22368, 25421, 28474, 30000)
DEPTHS = (0, 11, 22, 33, 44, 56, 67, 78, 89, 100)
JOBS = {'attention-mass': 'bt1a99kjkl896agbasll', 'multiset': 'bt1r9dq8r7o220va20k5'}
MODEL_LABEL = 'Ministral 3 8B'
MODEL_SLUG = 'ministral3-8b'
DOWNLOAD_PREFIX = 'ministral3'
SURVEY = ROOT / 'datasphere-results/ministral3-context-only-download/ministral3-8b-context-only-survey'
SURVEY_GALLERY = '../ministral3-8b-context-only-survey/README.md'


def load(ranking):
    source = ROOT / f'datasphere-results/{DOWNLOAD_PREFIX}-{ranking}-mask-download' / (
        f'{MODEL_SLUG}-no-thinking-paul-graham-{ranking}-mask-san-francisco')
    manifests = list(source.glob('evaluation/context-42/evaluation/*/run.json'))
    if len(manifests) != len(CONDITIONS):
        raise ValueError(f'Expected {len(CONDITIONS)} condition manifests: {source}')
    for path in manifests:
        manifest = json.loads(path.read_text())
        if (not manifest['complete'] or manifest['completed_cases'] != len(LENGTHS) * len(DEPTHS)
                or manifest['max_new_tokens'] != 256
                or tuple(manifest['lengths']) != LENGTHS or tuple(manifest['depths']) != DEPTHS):
            raise ValueError(f'Incomplete or unexpected condition manifest: {path}')
    rows, keys = [], set()
    metric = 'needle_attention_mass_v1' if ranking == 'attention-mass' else 'needle_token_multiset_v1'
    score_file = SURVEY / 'detection/aggregation' / f'head_scores_{metric}.json'
    ranked = [head for head, _ in rank_heads(json.loads(score_file.read_text()))]
    for path in sorted(source.glob('evaluation/context-42/evaluation/*/results/*_results.json')):
        row = json.loads(path.read_text())
        condition = path.parent.parent.name
        k = 0 if condition == 'baseline' else int(condition.removeprefix('top').removeprefix('bottom'))
        key = (condition, row['context_length'], row['depth_percent'])
        if key in keys:
            raise ValueError(f'Duplicate result: {key}')
        keys.add(key)
        experiment = row['experiment']
        if experiment['condition'] != condition or len(experiment['blocked_heads']) != k:
            raise ValueError(f'Condition/head-count mismatch: {path}')
        expected_heads = set() if not k else set(ranked[-k:] if condition.startswith('bottom') else ranked[:k])
        if set(map(tuple, experiment['blocked_heads'])) != expected_heads:
            raise ValueError(f'Masked heads differ from frozen survey ranking: {path}')
        if experiment['mask_mode'] != 'legacy_uniform' or experiment['decode_attention'] != 'sdpa_flash':
            raise ValueError(f'Unexpected masking protocol: {path}')
        row.update(condition=condition, masked_heads=k, ranking=ranking,
                   generated_tokens=len(row['generated_token_ids']), source=str(path.relative_to(ROOT)))
        rows.append(row)
    expected = {(c, l, d) for c in CONDITIONS for l in LENGTHS for d in DEPTHS}
    if keys != expected:
        raise ValueError(f'Incomplete/unexpected grid: {len(keys)} of {len(expected)}; missing={len(expected-keys)}')
    prompts = {}
    for row in rows:
        key = (row['context_length'], row['depth_percent'])
        if prompts.setdefault(key, row['prompt_sha256']) != row['prompt_sha256']:
            raise ValueError(f'Unpaired prompts: {key}')
    rows = rescore(rows)
    for row in rows:
        if abs(row['rouge1_recall'] - row['score']) > 1e-6:
            raise ValueError(f"Stored ROUGE differs from recalculation: {row['source']}")
        if any(not np.isfinite(row[m]) for m in METRICS):
            raise ValueError(f"Nonfinite score: {row['source']}")
    return rows, prompts


def save(fig, folder, name, subtitle):
    (folder / 'plots').mkdir(parents=True, exist_ok=True)
    fig.set_layout_engine('constrained', rect=(0, .055, 1, .945))
    fig.text(.01, .008, f'{MODEL_LABEL} · no thinking · Paul Graham / San Francisco · {subtitle} · '
             'legacy_uniform decode · BF16 · ≤256 output tokens', fontsize=8, color='#687587')
    for ext in ('png', 'svg'):
        fig.savefig(folder / 'plots' / f'{name}.{ext}', dpi=150, bbox_inches='tight')
    plt.close(fig)


def heatmap(ax, matrix, title, delta=False):
    im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r' if delta else CMAP,
                   vmin=-100 if delta else 0, vmax=100)
    ax.set(title=title, xlabel='Глубина иголки, %', ylabel='Длина контекста, токенов')
    ax.set_xticks(range(len(DEPTHS)), [str(d) for d in DEPTHS])
    ax.set_yticks(range(len(LENGTHS)), [f'{l:,}'.replace(',', ' ') for l in LENGTHS])
    for i, j in np.ndindex(matrix.shape):
        value = matrix[i, j]
        color = 'white' if (abs(value) >= 65 if delta else value >= 80) else INK
        ax.text(j, i, f'{value:+.0f}' if delta else f'{value:.0f}',
                ha='center', va='center', fontsize=7, color=color)
    return im


def matrix(rows, metric):
    lookup = {(r['context_length'], r['depth_percent']): r[metric] for r in rows}
    return np.array([[lookup[l, d] for d in DEPTHS] for l in LENGTHS])


def write_csv(path, rows, fields=None):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fields or list(rows[0]), extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def build(ranking, rows):
    folder = OUT / ranking
    folder.mkdir(parents=True, exist_ok=True)
    groups = {c: [r for r in rows if r['condition'] == c] for c in CONDITIONS}
    baseline = matrix(groups['baseline'], 'rouge123_mean_recall')
    summary, gallery = [], []
    for condition, group in groups.items():
        summary.append(dict(condition=condition, masked_heads=group[0]['masked_heads'], cases=len(group),
            **{m: mean(r[m] for r in group) for m in METRICS},
            length_stops=sum(r['finish_reason'] in {'length', 'max_new_tokens'} for r in group),
            mean_generated_tokens=mean(r['generated_tokens'] for r in group),
            mean_duration_seconds=mean(r['test_duration_seconds'] for r in group)))
        fig, axes = plt.subplots(2, 3, figsize=(17, 10), layout='constrained')
        fig.suptitle(f'{ranking} · {condition} · 110 случаев', fontsize=20)
        for ax, metric, label in zip(axes.flat, METRICS, LABELS):
            im = heatmap(ax, matrix(group, metric), label)
            fig.colorbar(im, ax=ax, shrink=.7, label='%')
        axes.flat[-1].axis('off')
        axes.flat[-1].text(.03, .85, '\n\n'.join(
            f'{label}: {summary[-1][metric]:.2f}%' for metric, label in zip(METRICS, LABELS)) +
            f"\n\nВыход по лимиту: {summary[-1]['length_stops']}/110", va='top', fontsize=12)
        name = f'{condition}_heatmaps'
        save(fig, folder, name, ranking)
        gallery.append((condition + ': ROUGE-1/2/3, среднее и exact match', name))
    for metric, label in zip(METRICS, LABELS):
        fig, axes = plt.subplots(2, 3, figsize=(17, 10), layout='constrained')
        fig.suptitle(f'{ranking} · {label}', fontsize=20)
        colors = plt.get_cmap('viridis')(np.linspace(.12, .9, len(KS)))
        for row_index, selection in enumerate(('top', 'bottom')):
            for k, color in zip(KS, colors):
                group = groups[f'{selection}{k}']
                for ax, field, grid in ((axes[row_index, 0], 'context_length', LENGTHS),
                                        (axes[row_index, 1], 'depth_percent', DEPTHS)):
                    ax.plot(grid, [mean(r[metric] for r in group if r[field] == x) for x in grid],
                            marker='o', markersize=3, color=color, label=f'k={k}')
            base_mean = mean(r[metric] for r in groups['baseline'])
            for ax in axes[row_index, :2]:
                ax.axhline(base_mean, color='black', linestyle='--', label='baseline')
            axes[row_index, 0].set(title=f'{selection}: среднее по глубинам', xlabel='Длина контекста')
            axes[row_index, 1].set(title=f'{selection}: среднее по длинам', xlabel='Глубина, %')
            axes[row_index, 1].legend(ncol=3, fontsize=8)
            axes[row_index, 2].bar([str(k) for k in (0, *KS)],
                [base_mean, *(mean(r[metric] for r in groups[f'{selection}{k}']) for k in KS)],
                color=['#888888', *colors])
            axes[row_index, 2].set(title=f'{selection}: вся сетка', xlabel='Маскировано Q-голов')
        for ax in axes.flat:
            ax.set_ylim(0, 105)
            ax.set_ylabel('Оценка, %')
            ax.grid(axis='y', alpha=.2)
        name = f'comparison_{metric}'
        save(fig, folder, name, ranking)
        gallery.append((label + ': длины, глубины и k; top и bottom', name))
    for k in KS:
        fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout='constrained')
        fig.suptitle(f'{ranking} · k={k} · изменение среднего ROUGE-1/2/3 против baseline')
        for ax, selection in zip(axes, ('top', 'bottom')):
            im = heatmap(ax, matrix(groups[f'{selection}{k}'], 'rouge123_mean_recall') - baseline,
                         selection, delta=True)
            fig.colorbar(im, ax=ax, label='Изменение, п.п.')
        name = f'delta_baseline_k{k:03d}'
        save(fig, folder, name, ranking)
        gallery.append((f'k={k}: парное изменение относительно baseline', name))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout='constrained')
    for selection, color in [('top', '#3574b4'), ('bottom', '#168577')]:
        selected = [summary[0], *(next(s for s in summary if s['condition'] == f'{selection}{k}') for k in KS)]
        for ax, field, label in zip(axes,
                ('length_stops', 'mean_generated_tokens', 'mean_duration_seconds'),
                ('Число ответов по лимиту /110', 'Средняя длина ответа, токенов', 'Средняя длительность, с')):
            ax.plot((0, *KS), [s[field] for s in selected], marker='o', color=color, label=selection)
            ax.set(xlabel='Маскировано Q-голов', title=label)
            ax.grid(alpha=.2)
            ax.legend()
    save(fig, folder, 'generation_diagnostics', ranking)
    gallery.append(('Длина выхода, обрывы и длительность', 'generation_diagnostics'))
    write_csv(folder / 'summary.csv', summary)
    write_csv(folder / 'cases.csv', rows, ['ranking', 'condition', 'masked_heads', 'context_length',
        'depth_percent', 'score', *[f'rouge{n}_{c}' for n in (1, 2, 3) for c in ('recall', 'precision', 'fmeasure')],
        'rouge123_mean_recall', 'rouge123_mean_precision', 'rouge123_mean_fmeasure', 'exact_match',
        'generated_tokens', 'finish_reason', 'test_duration_seconds', 'prompt_sha256', 'source'])
    (folder / 'scores.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    lines = [f'# {MODEL_LABEL}: Paul Graham — {ranking}', '', f'Job `{JOBS[ranking]}`; 1870/1870 результатов.',
        '', '11 длин ×10 глубин ×17 условий: baseline, top/bottom k=5,10,15,20,40,60,80,100.',
        'Prompt-хэши совпадают между всеми условиями. Маска legacy_uniform при decode; K/V и prefill не менялись.',
        '', 'Метрики пересчитаны по сохранённому model_response тем же кодом, что для Qwen. Исходные JSON не изменены.',
        'ROUGE recall со stemming; среднее 1/2/3 — сводная нестандартная метрика. Exact match — присутствие полной',
        'нормализованной эталонной фразы подряд, не семантическое совпадение. Recall не штрафует лишний текст.',
        'Все результаты, включая обрывы по лимиту, включены. Сравнение top/bottom не доказывает специализацию голов.',
        '', '| Условие | ROUGE-1 | ROUGE-2 | ROUGE-3 | Среднее | Exact | По лимиту |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for s in summary:
        lines.append('| ' + s['condition'] + ' | ' + ' | '.join(f'{s[m]:.2f}' for m in METRICS) +
                     f" | {s['length_stops']} |")
    lines += ['', '[CSV случаев](cases.csv) · [Сводка](summary.csv) · [JSON оценок](scores.json)', '']
    for title, name in gallery:
        lines += [f'## {title}', '', f'![{title}](plots/{name}.png)', '', f'[SVG](plots/{name}.svg)', '']
    (folder / 'README.md').write_text('\n'.join(lines))
    print(ranking, 'figures=', len(gallery), 'finish_reasons=', dict(Counter(r['finish_reason'] for r in rows)))
    return summary


def compare_rankings(runs):
    quality = []
    for ranking, rows in runs.items():
        for condition in CONDITIONS:
            group = [r for r in rows if r['condition'] == condition]
            quality.append(dict(ranking=ranking, condition=condition,
                **{f'rouge123_mean_{c}': mean(r[f'rouge123_mean_{c}'] for r in group)
                   for c in ('recall', 'precision', 'fmeasure')},
                length_stops=sum(r['finish_reason'] in {'length', 'max_new_tokens'} for r in group)))
    write_csv(OUT / 'quality_summary.csv', quality)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout='constrained')
    for axrow, selection in zip(axes, ('top', 'bottom')):
        for ax, component in zip(axrow, ('recall', 'precision', 'fmeasure')):
            metric = f'rouge123_mean_{component}'
            for ranking, color in [('attention-mass', '#3574b4'), ('multiset', '#168577')]:
                rows = runs[ranking]
                conditions = ['baseline', *(f'{selection}{k}' for k in KS)]
                ax.plot((0, *KS), [mean(r[metric] for r in rows if r['condition'] == c) for c in conditions],
                        marker='o', color=color, label=ranking)
            ax.set(title=f'{selection} · {component}', xlabel='Маскировано Q-голов',
                   ylabel='Среднее ROUGE-1/2/3, %', ylim=(0, 105))
            ax.grid(alpha=.2)
            ax.legend()
    save(fig, OUT, 'ranking_comparison', 'два рейтинга · recall / precision / F1')


def main():
    global OUT, JOBS, MODEL_LABEL, MODEL_SLUG, DOWNLOAD_PREFIX, SURVEY, SURVEY_GALLERY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('ministral', 'glm'), default='ministral')
    args = parser.parse_args()
    if args.model == 'glm':
        MODEL_LABEL = 'GLM-4.6V-Flash'
        MODEL_SLUG = DOWNLOAD_PREFIX = 'glm4-6v-flash'
        OUT = ROOT / 'report-assets/glm4-6v-flash-paul-graham-mask'
        JOBS = {'attention-mass': 'bt1cskrr10furlk74bdu', 'multiset': 'bt1af8nq5baf9a9t6b6f'}
        SURVEY = ROOT / 'datasphere-results/glm4-6v-flash-survey-download/glm4-6v-flash-no-thinking-context-only-survey'
        SURVEY_GALLERY = '../glm4-6v-flash-no-thinking-survey/README.md'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'figure.facecolor': BG,
                         'axes.facecolor': BG, 'svg.fonttype': 'none'})
    runs = {ranking: load(ranking) for ranking in JOBS}
    if runs['attention-mass'][1] != runs['multiset'][1]:
        raise ValueError('Prompts differ between rankings')
    baselines = [{(r['context_length'], r['depth_percent']): r['generated_token_ids']
                  for r in rows if r['condition'] == 'baseline'} for rows, _ in runs.values()]
    differences = sum(baselines[0][key] != baselines[1][key] for key in baselines[0])
    for ranking, (rows, _) in runs.items():
        build(ranking, rows)
    compare_rankings({ranking: rows for ranking, (rows, _) in runs.items()})
    if args.model == 'glm':
        cleaned_summary = []
        for ranking, (rows, _) in runs.items():
            cleaned = rescore([dict(r, model_response=r['model_response']
                .replace('<|begin_of_box|>', ' ').replace('<|end_of_box|>', ' ')) for r in rows])
            for condition in CONDITIONS:
                group = [r for r in cleaned if r['condition'] == condition]
                cleaned_summary.append(dict(ranking=ranking, condition=condition,
                    **{f'rouge123_mean_{c}': mean(r[f'rouge123_mean_{c}'] for r in group)
                       for c in ('recall', 'precision', 'fmeasure')},
                    exact_match=mean(r['exact_match'] for r in group)))
        write_csv(OUT / 'box_normalized_quality_summary.csv', cleaned_summary)
    (OUT / 'README.md').write_text('\n'.join([
        f'# {MODEL_LABEL}: Paul Graham masking', '',
        'Оба задания SUCCESS. Проверено 3740 уникальных результатов, по1870 на рейтинг.',
        'Одинаковые 110 prompt-хэшей для всей сетки и обоих рейтингов.',
        f'Различия baseline generated_token_ids между заданиями: {differences}/110.', '',
        '[Attention mass — полная галерея](attention-mass/README.md)', '',
        '[Multiset — полная галерея](multiset/README.md)', '',
        'Всего 63 фигуры, каждая в PNG и SVG. Исходные результаты не изменялись.', '',
        '![Сравнение рейтингов](plots/ranking_comparison.png)', '',
        '[SVG](plots/ranking_comparison.svg)', '',
        'Общая фигура: recall, precision и F1. [Сводная CSV](quality_summary.csv).', '',
        f'Detection-карты слоёв и голов: [survey]({SURVEY_GALLERY}).', '',
        'Не сравнивать напрямую с Qwen как контролируемую абляцию: отличаются модель, prompt, сетка и ranking.',
    ] + (['', '[Подробный анализ и оговорки GLM-разметки](ANALYSIS.md)', '',
          '[Диагностические оценки с box-маркерами, заменёнными пробелами](box_normalized_quality_summary.csv)',
          'Эти оценки не заменяют исходные score и основные графики.'] if args.model == 'glm' else [])))
    print('baseline_token_differences=', differences)
    print('gallery=', OUT / 'README.md')


if __name__ == '__main__':
    main()
