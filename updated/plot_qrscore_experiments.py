"""Offline QRScore survey pies and paired masking galleries from saved ZIPs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean
from zipfile import ZipFile

import numpy as np
from plot_paul_graham_mask import plt, rescore, METRICS, LABELS, CMAP, BG, INK
from retrieval_heads.experiment.scoring import rank_heads

ROOT = Path(__file__).resolve().parent
MODELS = {
    'qwen3-8b': ('Qwen3-8B', 'qwen3-8b-no-thinking-paul-graham-qrscore-mask', 36),
    'glm4-6v-flash': ('GLM-4.6V-Flash', 'glm4-6v-flash-no-thinking-paul-graham-qrscore-top-mask', 40),
    'ministral3-8b': ('Ministral 3 8B', 'ministral3-8b-no-thinking-paul-graham-qrscore-top-mask', 34),
}
KS = (5, 10, 15, 20, 40, 60, 80, 100)
LENGTHS = (1000, 4053, 7105, 10158, 13211, 16263, 19316, 22368, 25421, 28474, 30000)
DEPTHS = (0, 11, 22, 33, 44, 56, 67, 78, 89, 100)


def load_archive(path, layers):
    """Validate complete grid, paired prompts and the exact frozen head ranking."""
    with ZipFile(path) as z:
        if z.testzip() is not None:
            raise ValueError(f'Corrupt ZIP: {path}')
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate ZIP members')
        prefix = names[0].split('/')[0]
        def read(relative):
            return json.loads(z.read(f'{prefix}/{relative}'))
        aggregation = read('ranking/run.json')
        histories = read('ranking/head_scores_query_retrieval_score_v1.json')
        scores = read('ranking/mean_scores.json')
        expected_heads = {f'{l}-{h}' for l in range(layers) for h in range(32)}
        if (not aggregation['complete'] or not aggregation['filters']['all_cases']
                or aggregation['selected_case_count'] != 27
                or aggregation['retrieval_metrics'] != ['query_retrieval_score_v1']
                or set(histories) != expected_heads or set(scores) != expected_heads):
            raise ValueError('Unexpected survey scope or head geometry')
        for h, values in histories.items():
            if (len(values) != 27 or not all(np.isfinite(v) and 0 <= v <= 1.000001 for v in values)
                    or not np.isclose(mean(values), scores[h], rtol=0, atol=1e-12)):
                raise ValueError(f'Invalid frozen score history: {h}')
        ranking = [h for h, _ in rank_heads(histories)]
        conditions = ('baseline', *(f'top{k}' for k in KS))
        if any('/evaluation/bottom' in n for n in names):
            conditions += tuple(f'bottom{k}' for k in KS)
        rows, prompts, keys = [], {}, set()
        for condition in conditions:
            folder = f'evaluation/context-42/evaluation/{condition}'
            run = read(f'{folder}/run.json')
            if (not run['complete'] or run['completed_cases'] != 110
                    or tuple(run['lengths']) != LENGTHS or tuple(run['depths']) != DEPTHS
                    or run['max_new_tokens'] != 256):
                raise ValueError(f'Incomplete/unexpected manifest: {condition}')
            for n in sorted(names):
                if not n.startswith(f'{prefix}/{folder}/results/') or not n.endswith('_results.json'):
                    continue
                row = json.loads(z.read(n))
                k = 0 if condition == 'baseline' else int(condition.replace('top', '').replace('bottom', ''))
                expected = set() if not k else set(ranking[-k:] if condition.startswith('bottom') else ranking[:k])
                exp = row['experiment']
                if (set(map(tuple, exp['blocked_heads'])) != expected or exp['condition'] != condition
                        or exp['mask_mode'] != 'legacy_uniform' or exp['decode_attention'] != 'sdpa_flash'):
                    raise ValueError(f'Masking/ranking mismatch: {n}')
                key = condition, row['context_length'], row['depth_percent']
                if key in keys:
                    raise ValueError(f'Duplicate case: {key}')
                keys.add(key)
                grid_key = key[1:]
                if prompts.setdefault(grid_key, row['prompt_sha256']) != row['prompt_sha256']:
                    raise ValueError(f'Unpaired prompts: {grid_key}')
                row.update(condition=condition, masked_heads=k,
                           generated_tokens=len(row['generated_token_ids']), source=n)
                rows.append(row)
        expected_grid = {(c, l, d) for c in conditions for l in LENGTHS for d in DEPTHS}
        if keys != expected_grid:
            raise ValueError('Incomplete/unexpected result grid')
    rescore(rows)
    if any(not np.isclose(r['score'], r['rouge1_recall'], atol=1e-6) for r in rows):
        raise ValueError('Stored ROUGE differs from recalculation')
    return rows, scores, conditions, aggregation


def write_csv(path, rows, fields=None):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fields or list(rows[0]), extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def save(fig, folder, name, model, note):
    fig.set_layout_engine('constrained', rect=(0, .04, 1, .95))
    fig.text(.01, .008, f'{model} · no thinking · QRScore ranking · {note}', fontsize=8, color='#687587')
    folder.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'svg'):
        fig.savefig(folder / f'{name}.{ext}', dpi=150, bbox_inches='tight')
    plt.close(fig)


def matrix(rows, metric):
    lookup = {(r['context_length'], r['depth_percent']): r[metric] for r in rows}
    return np.array([[lookup[l, d] for d in DEPTHS] for l in LENGTHS])


def heatmap(ax, values, title, delta=False):
    im = ax.imshow(values, aspect='auto', cmap='RdBu_r' if delta else CMAP,
                   vmin=-100 if delta else 0, vmax=100)
    ax.set(title=title, xlabel='Глубина иголки, %', ylabel='Длина контекста, токенов')
    ax.set_xticks(range(len(DEPTHS)), DEPTHS, fontsize=8)
    ax.set_yticks(range(len(LENGTHS)), [f'{l:,}'.replace(',', ' ') for l in LENGTHS], fontsize=8)
    for i, j in np.ndindex(values.shape):
        v = values[i, j]
        ax.text(j, i, f'{v:+.0f}' if delta else f'{v:.0f}', ha='center', va='center', fontsize=7,
                color='white' if (abs(v) >= 65 if delta else v >= 80) else INK)
    return im


def build(slug, out):
    model, run_name, layers = MODELS[slug]
    source = ROOT / 'colab-results' / f'{run_name}-results.zip'
    rows, scores, conditions, aggregation = load_archive(source, layers)
    folder = out / slug
    plots = folder / 'plots'
    plots.mkdir(parents=True, exist_ok=True)
    gallery = []
    counts = [sum(0 <= v <= .1 for v in scores.values()),
              sum(.1 < v <= .5 for v in scores.values()),
              sum(.5 < v <= 1.000001 for v in scores.values())]
    bins = ['0–0,1', '>0,1–0,5', '>0,5–1']
    colors = ['#94a3b8', '#38bdf8', '#2563eb']
    fig, ax = plt.subplots(figsize=(9, 6), layout='constrained')
    nonzero = [i for i, n in enumerate(counts) if n]
    ax.pie([counts[i] for i in nonzero], colors=[colors[i] for i in nonzero], startangle=90,
           wedgeprops=dict(width=.42, edgecolor=BG),
           autopct=lambda p: f'{p:.1f}%' if p >= 2 else '', pctdistance=.79)
    ax.text(0, 0, f'{len(scores)}\nQ-голов', ha='center', va='center', fontsize=19)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in colors],
              [f'QRScore {b}: {n} ({100*n/len(scores):.2f}%)' for b, n in zip(bins, counts)],
              loc='lower center', bbox_to_anchor=(.5, -.07), frameon=False)
    ax.set_title(f'{model} · Survey: доли голов по среднему QRScore', fontsize=15)
    save(fig, plots, 'survey_head_distribution', model, '27/27 survey cases, включая неудачные ответы')
    gallery.append(('Survey — доли голов (не доли суммарного attention)', 'survey_head_distribution'))
    write_csv(folder / 'survey_distribution.csv', [dict(qrscore_range=b, heads=n, percent=100*n/len(scores))
                                                 for b, n in zip(bins, counts)])
    groups = {c: [r for r in rows if r['condition'] == c] for c in conditions}
    summaries = [dict(condition=c, masked_heads=g[0]['masked_heads'], cases=len(g),
                      **{m: mean(r[m] for r in g) for m in METRICS}) for c, g in groups.items()]
    for condition, group in groups.items():
        fig, axes = plt.subplots(2, 3, figsize=(17, 10), layout='constrained')
        fig.suptitle(f'{model} · {condition} · Paul Graham · 110 случаев', fontsize=19)
        for ax, metric, label in zip(axes.flat, METRICS, LABELS):
            im = heatmap(ax, matrix(group, metric), label)
            fig.colorbar(im, ax=ax, shrink=.7, label='%')
        axes.flat[-1].axis('off')
        axes.flat[-1].text(.03, .85, '\n\n'.join(f'{label}: {mean(r[m] for r in group):.2f}%'
                                                 for m, label in zip(METRICS, LABELS)), va='top', fontsize=12)
        name = f'{condition}_heatmaps'
        save(fig, plots, name, model, 'legacy_uniform decode · ≤256 output tokens')
        gallery.append((f'{condition} — ROUGE-1/2/3, среднее и точная фраза', name))
    selections = ['top'] + (['bottom'] if 'bottom5' in groups else [])
    for selection in selections:
        for k in KS:
            fig, ax = plt.subplots(figsize=(8, 7), layout='constrained')
            im = heatmap(ax, matrix(groups[f'{selection}{k}'], 'rouge123_mean_recall') -
                         matrix(groups['baseline'], 'rouge123_mean_recall'),
                         f'{model} · {selection}{k} − baseline\nСреднее ROUGE-1/2/3 recall', delta=True)
            fig.colorbar(im, ax=ax, label='Изменение, п.п.')
            name = f'{selection}{k}_delta_baseline'
            save(fig, plots, name, model, 'парное сравнение одинаковых prompts')
            gallery.append((f'{selection}{k} — изменение относительно baseline', name))
    for metric, label in zip(METRICS, LABELS):
        fig, axes = plt.subplots(len(selections), 3, figsize=(17, 5*len(selections)), squeeze=False, layout='constrained')
        fig.suptitle(f'{model} · {label}', fontsize=18)
        palette = plt.get_cmap('viridis')(np.linspace(.08, .9, len(KS)))
        for axes_row, selection in zip(axes, selections):
            for k, color in zip((0, *KS), ('#111111', *palette)):
                group = groups['baseline' if not k else f'{selection}{k}']
                for ax, field, grid in ((axes_row[0], 'context_length', LENGTHS),
                                        (axes_row[1], 'depth_percent', DEPTHS)):
                    ax.plot(grid, [mean(r[metric] for r in group if r[field] == x) for x in grid],
                            color=color, marker='o', markersize=3, linestyle='--' if not k else '-',
                            label='baseline' if not k else f'k={k}')
            axes_row[0].set(title=f'{selection}: по длине', xlabel='Длина контекста, токенов')
            axes_row[1].set(title=f'{selection}: по глубине', xlabel='Глубина иголки, %')
            axes_row[1].legend(ncol=3, fontsize=8)
            axes_row[2].bar([str(k) for k in (0, *KS)],
                           [mean(r[metric] for r in groups['baseline' if not k else f'{selection}{k}'])
                            for k in (0, *KS)], color=['#888888', *palette])
            axes_row[2].set(title=f'{selection}: вся сетка', xlabel='Маскировано Q-голов')
        for ax in axes.flat:
            ax.set_ylim(0, 105)
            ax.set_ylabel('Оценка, %')
            ax.grid(axis='y', alpha=.2)
        name = f'comparison_{metric}'
        save(fig, plots, name, model, 'baseline — настоящая кривая по длинам/глубинам')
        gallery.append((f'{label} — длины, глубины и число голов', name))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout='constrained')
    for selection in selections:
        groups_k = [groups['baseline'], *(groups[f'{selection}{k}'] for k in KS)]
        values = [[sum(r['finish_reason'] in {'length', 'max_new_tokens'} for r in g) for g in groups_k],
                  [mean(r['generated_tokens'] for r in g) for g in groups_k],
                  [mean(r['test_duration_seconds'] for r in g) for g in groups_k]]
        for ax, values_k, label in zip(axes, values, ['Ответы по лимиту /110', 'Средняя длина ответа', 'Средняя длительность, с']):
            ax.plot((0, *KS), values_k, marker='o', label=selection)
            ax.set(title=label, xlabel='Маскировано Q-голов')
            ax.grid(alpha=.2)
            ax.legend()
    fig.suptitle(f'{model} · Диагностика генерации')
    save(fig, plots, 'generation_diagnostics', model, 'длительность включает prefill; это не причинный benchmark')
    gallery.append(('Диагностика генерации', 'generation_diagnostics'))
    write_csv(folder / 'summary.csv', summaries)
    write_csv(folder / 'cases.csv', rows, ['condition', 'masked_heads', 'context_length', 'depth_percent',
              *METRICS, 'generated_tokens', 'finish_reason', 'test_duration_seconds', 'prompt_sha256', 'source'])
    (folder / 'provenance.json').write_text(json.dumps(dict(source=str(source),
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(), source_run_id=aggregation['source_run_id'],
        survey_cases=27, heads=len(scores), masking_cases=len(rows), paired_prompts=True), indent=2))
    lines = [f'# {model} — QRScore / no thinking', '',
             f'{len(rows)} случаев masking; survey: все 27 случаев, {len(scores)} Q-голов.', '',
             'Survey: одна круговая диаграмма количества голов по диапазонам среднего QRScore.',
             'Masking: legacy_uniform только при decode; prefill и K/V не изменены.',
             'ROUGE пересчитан по сохранённым ответам со stemming. Среднее ROUGE-1/2/3 — нестандартный композит.',
             '«Точная фраза» — присутствие нормализованного эталона подряд, не семантическое совпадение.',
             'Все ответы включены; исходные ZIP не изменены. Prompt-хэши совпадают между условиями.', '',
             '[Сводка](summary.csv) · [Случаи](cases.csv) · [Доли голов](survey_distribution.csv) · [Источники](provenance.json)', '']
    for title, name in gallery:
        lines += [f'## {title}', '', f'![{title}](plots/{name}.png)', '', f'[SVG](plots/{name}.svg)', '']
    (folder / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'{slug}: {len(rows)} verified cases; {len(gallery)} figures; survey bins={counts}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'report-assets/qrscore-colab')
    args = parser.parse_args()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'figure.facecolor': BG,
                         'axes.facecolor': BG, 'text.color': INK, 'axes.labelcolor': INK,
                         'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    for slug in MODELS:
        build(slug, args.output)
    (args.output / 'README.md').write_text('# QRScore: survey и отключение голов\n\n' + '\n\n'.join(
        f'[{model}](./{slug}/README.md)' for slug, (model, _, _) in MODELS.items()) +
        '\n\nQwen: top и bottom. GLM и Ministral: только top. Все модели — no thinking.\n', encoding='utf-8')


if __name__ == '__main__':
    main()
