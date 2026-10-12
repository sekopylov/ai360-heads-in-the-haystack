"""Muted poster heatmaps: GLM rankings across head counts, context length on X."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from plot_paul_graham_mask import plt, rescore
from plot_qrscore_experiments import load_archive, LENGTHS, DEPTHS
from retrieval_heads.experiment.scoring import rank_heads

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'report-assets/poster/glm-top40-three-metrics'
SURVEY = ROOT / 'datasphere-results/glm4-6v-flash-survey-download/glm4-6v-flash-no-thinking-context-only-survey'
RANKINGS = [('multiset', 'Token multiset', 'needle_token_multiset_v1'),
            ('attention-mass', 'Attention mass', 'needle_attention_mass_v1')]
KS = (20, 40, 60, 80)
PROGRESSION_KS = (20, 40, 80)
CONDITIONS = ('baseline', *(f'top{k}' for k in KS))
CONTROL_LABEL = 'Token multiset (bottom control)'
CONTROL_CONDITIONS = tuple(f'bottom{k}' for k in PROGRESSION_KS)
POSTER_CMAP = LinearSegmentedColormap.from_list(
    'poster_red_yellow_green', [(0, '#b3323b'), (.25, '#cf5145'),
                               (.5, '#edcd6a'), (.75, '#83ad68'), (1, '#177343')])


def load_local(slug, metric, conditions=CONDITIONS):
    source = ROOT / f'datasphere-results/glm4-6v-flash-{slug}-mask-download' / (
        f'glm4-6v-flash-no-thinking-paul-graham-{slug}-mask-san-francisco')
    history = json.loads((SURVEY / 'detection/aggregation' / f'head_scores_{metric}.json').read_text())
    ranked = [h for h, _ in rank_heads(history)]
    rows = []
    for condition in conditions:
        k = 0 if condition == 'baseline' else int(condition.removeprefix('top').removeprefix('bottom'))
        blocked = set() if not k else set(ranked[-k:] if condition.startswith('bottom') else ranked[:k])
        folder = source / f'evaluation/context-42/evaluation/{condition}'
        run = json.loads((folder / 'run.json').read_text())
        assert run['complete'] and run['completed_cases'] == 110
        assert tuple(run['lengths']) == LENGTHS and tuple(run['depths']) == DEPTHS
        assert run['max_new_tokens'] == 256
        for path in sorted((folder / 'results').glob('*_results.json')):
            row = json.loads(path.read_text())
            e = row['experiment']
            assert e['mask_mode'] == 'legacy_uniform' and e['decode_attention'] == 'sdpa_flash'
            assert set(map(tuple, e['blocked_heads'])) == blocked
            row.update(condition=condition, source=str(path.relative_to(ROOT)))
            rows.append(row)
    assert len(rows) == len(conditions) * 110
    rescore(rows)
    assert all(abs(r['score'] - r['rouge1_recall']) < 1e-6 for r in rows)
    return rows, source


def main():
    data = {}
    provenance = {}
    for slug, label, metric in RANKINGS:
        data[label], source = load_local(slug, metric)
        provenance[label] = {'source': str(source.relative_to(ROOT)), 'ranking_metric': metric}
    qr_path = ROOT / 'colab-results/glm4-6v-flash-no-thinking-paul-graham-qrscore-top-mask-results.zip'
    qr, _, _, _ = load_archive(qr_path, 40)
    data['QRScore'] = [r for r in qr if r['condition'] in CONDITIONS]
    rescore(data['QRScore'])
    provenance['QRScore'] = {'source': str(qr_path.relative_to(ROOT)),
                             'sha256': hashlib.sha256(qr_path.read_bytes()).hexdigest(),
                             'ranking_metric': 'query_retrieval_score_v1'}
    top_labels = tuple(data)
    data[CONTROL_LABEL], control_source = load_local(
        'multiset', 'needle_token_multiset_v1', CONTROL_CONDITIONS)
    provenance[CONTROL_LABEL] = {
        'source': str(control_source.relative_to(ROOT)),
        'ranking_metric': 'needle_token_multiset_v1',
        'selection': 'bottom', 'randomized': False,
        'conditions': CONTROL_CONDITIONS,
    }
    expected = {(l, d) for l in LENGTHS for d in DEPTHS}
    hashes = {}
    output_rows = []
    summary = []
    matrices = {k: [] for k in KS}
    control_matrices = {}
    for label, rows in data.items():
        conditions = CONTROL_CONDITIONS if label == CONTROL_LABEL else CONDITIONS
        for condition in conditions:
            group = [r for r in rows if r['condition'] == condition]
            keys = [(r['context_length'], r['depth_percent']) for r in group]
            assert len(keys) == len(set(keys)) == 110 and set(keys) == expected
            for r in group:
                key = r['context_length'], r['depth_percent']
                assert hashes.setdefault(key, r['prompt_sha256']) == r['prompt_sha256']
                output_rows.append(dict(ranking=label, condition=condition,
                    context_length=key[0], depth_percent=key[1], rouge1_recall=r['rouge1_recall'],
                    prompt_sha256=r['prompt_sha256'], source=r['source']))
            score = mean(r['rouge1_recall'] for r in group)
            summary.append(dict(ranking=label, condition=condition, cases=110, mean_rouge1_recall=score))
            if condition == 'baseline':
                assert all(r['rouge1_recall'] == 100 for r in group)
            else:
                lookup = {(r['context_length'], r['depth_percent']): r['rouge1_recall'] for r in group}
                k = int(condition.removeprefix('top').removeprefix('bottom'))
                # Rows = depth, columns = context length: literal axis transpose.
                values = np.array([[lookup[l, d] for l in LENGTHS] for d in DEPTHS])
                if label == CONTROL_LABEL:
                    control_matrices[k] = ('Token multiset', score, values)
                else:
                    matrices[k].append((label, score, values))
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'figure.facecolor': 'none', 'axes.facecolor': 'none', 'axes.spines.top': False,
                         'axes.spines.right': False})
    def panel(ax, label, score, values, compact=False, show_xlabel=True):
        im = ax.imshow(values, cmap=POSTER_CMAP, vmin=0, vmax=100, aspect='auto', interpolation='nearest')
        display_label = 'Retrieval Score' if label == 'Token multiset' else label
        title = f'Средний recall: {score:.1f}%'
        ax.set_title(title if compact else f'{display_label}\n{title}', fontsize=13 if compact else 15, pad=10)
        ax.set_xlabel('Длина контекста, токенов' if show_xlabel else '', fontsize=12)
        ax.set_xticks(range(len(LENGTHS)), [f'{l:,}'.replace(',', ' ') for l in LENGTHS],
                      fontsize=8, rotation=45, ha='right')
        ax.set_yticks(range(len(DEPTHS)), DEPTHS, fontsize=10)
        return im

    def save(fig, name):
        for ext in ('png', 'svg', 'pdf'):
            fig.savefig(OUT / f'{name}.{ext}', dpi=220, bbox_inches='tight', transparent=True)
        plt.close(fig)

    for k in KS:
        fig, axes = plt.subplots(1, 3, figsize=(16, 6.7), sharey=True, layout='constrained')
        for ax, (label, score, values) in zip(axes, matrices[k]):
            im = panel(ax, label, score, values)
        axes[0].set_ylabel('Глубина иголки, %', fontsize=12)
        cb = fig.colorbar(im, ax=axes, ticks=(0,25,50,75,100), shrink=.9, pad=.015)
        cb.set_label('ROUGE-1 recall, %', fontsize=12)
        fig.suptitle(f'GLM-4.6V-Flash · маскирование top-{k} голов', fontsize=21, weight='bold')
        save(fig, f'top{k}_three_rankings')

    fig = plt.figure(figsize=(17, 19), layout='constrained')
    # Reserve a separate title band for extra space above the column headers.
    fig.get_layout_engine().set(rect=(0, 0, 1, .945))
    grid = fig.add_gridspec(4, 4, width_ratios=(.55, 1, 1, 1))
    axes = np.empty((4, 3), dtype=object)
    row_labels = ('Bottom\n(RS)', 'Retrieval\nScore', 'Attention\nmass', 'QRScore')
    # A dedicated label column avoids repeating method names above every map.
    # Bottom control comes first; mask sizes still increase left to right.
    for row_idx, row_label in enumerate(row_labels):
        label_ax = fig.add_subplot(grid[row_idx, 0])
        label_ax.set_axis_off()
        label_ax.text(.68, .5, row_label, ha='right', va='center', fontsize=16, weight='normal')
        for col_idx, k in enumerate(PROGRESSION_KS):
            shared = axes[0, 0] if row_idx or col_idx else None
            ax = fig.add_subplot(grid[row_idx, col_idx + 1], sharex=shared, sharey=shared)
            axes[row_idx, col_idx] = ax
            label, score, values = control_matrices[k] if row_idx == 0 else matrices[k][row_idx - 1]
            im = panel(ax, label, score, values, compact=True, show_xlabel=row_idx == 3)
            if row_idx == 0:
                ax.set_title(f'{k} голов\nСредний recall: {score:.1f}%', fontsize=15, pad=10)
            if col_idx == 0:
                ax.set_ylabel('Глубина иголки, %', fontsize=12, labelpad=10)
            ax.tick_params(labelbottom=row_idx == 3, labelleft=col_idx == 0)
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), ticks=(0,25,50,75,100), shrink=.8, pad=.05)
    cb.set_label('ROUGE-1 recall, %', fontsize=12)
    fig.suptitle('GLM-4.6V-Flash · маскирование голов', fontsize=22, weight='bold', y=.995)
    save(fig, 'masking_progression_20_40_80')
    for name, rows in [('cases.csv', output_rows), ('summary.csv', summary)]:
        with (OUT / name).open('w', newline='') as f:
            writer = csv.DictWriter(f, list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (OUT / 'provenance.json').write_text(json.dumps(dict(sources=provenance,
        measured_top40_cases=330,
        measured_masking_cases=330*len(KS) + 110*len(CONTROL_CONDITIONS), topks=KS,
        measured_bottom_control_cases=110*len(CONTROL_CONDITIONS),
        progression_topks=PROGRESSION_KS,
        imputed_cases=0, matched_prompt_hashes=110,
        baseline_cases_checked=330, lengths=LENGTHS, depths=DEPTHS,
        note='Different ranking and masking runs, identical evaluated prompts. All cells are measured.'), indent=2))
    lines = ['# GLM: три рейтинга retrieval heads для плаката', '',
        'Ось X — длина контекста, ось Y — глубина needle. Общая умеренно насыщенная шкала: '
        '0% кирпично-красный → 50% песочный → 100% тёмно-зелёный.', '',
        'Фон актуальных PNG/SVG/PDF прозрачен для размещения на плакате.', '',
        'В исходной [статье, раздел 2](https://arxiv.org/html/2404.15574v1#S2) метрика называется '
        'Retrieval Score (RS). На графиках используется краткая подпись «Retrieval Score»: '
        'данные получены нашей multiset-версией с ограничением повторных hit по частотам токенов '
        'needle, а не неизменённой legacy-реализацией авторов. Bottom (RS) — нижние головы '
        'того же multiset-рейтинга. В CSV сохранено прежнее имя Token multiset.', '',
        '## Где multiset начинает заметно снижать извлечение', '',
        'ROUGE-1 recall: top10 — 98.2%, top15 — 90.3%, top20 — 75.8%, '
        'top40 — 84.0%, top60 — 64.8%, top80 — 45.8%. '
        'Зависимость немонотонная: top40 лучше top20. После top40 снижение продолжается '
        'при top60/80. Для наглядной промежуточной карты подойдёт top80.', '',
        '| top-k | Token multiset | Attention mass | QRScore |', '|---|---:|---:|---:|']
    summary_lookup = {(r['condition'],r['ranking']):r['mean_rouge1_recall'] for r in summary}
    lines += [f'| {k} | ' + ' | '.join(f'{summary_lookup[f"top{k}",label]:.1f}%' for label in top_labels) + ' |' for k in KS]
    lines += ['', '[Cases CSV](cases.csv) · [Средние](summary.csv) · [Источники](provenance.json)', '',
        '## Прогрессия top20 → top40 → top80', '',
        'Столбцы слева направо — 20, 40, 80 отключённых голов. Первая строка — bottom-k по '
        'Token multiset; затем top-k по Token multiset, Attention mass и QRScore. Названия '
        'строк вынесены слева, подпись длины контекста и числовые метки X — только в последней строке.', '',
        'Bottom — головы с наименьшим score, не случайная выборка. Все 330 контрольных случаев '
        'взяты из исходного полного эксперимента; prompt-хэши и фактические наборы масок проверены. '
        'Средний ROUGE-1 recall: bottom20/40/80 — 100%. При этом bottom40/80 достигают лимита '
        'генерации в 56/93 случаях из 110: высокий recall не означает отсутствие повторов или '
        'сохранность качества всего ответа.', '',
        '![Прогрессия маскирования](masking_progression_20_40_80.png)', '',
        '[SVG](masking_progression_20_40_80.svg) · [PDF](masking_progression_20_40_80.pdf)', '',
        '## Оценки всех голов по трём метрикам', '',
        '[Отдельная фигура: слой по X, номер головы по Y](HEAD_SCORES.md) · '
        '[SVG](head_scores_three_metrics.svg) · [PDF](head_scores_three_metrics.pdf)', '',
        '[Три круговые диаграммы распределения голов](HEAD_SCORE_DISTRIBUTION.md) · '
        '[SVG](head_score_distribution_three_metrics.svg) · [PDF](head_score_distribution_three_metrics.pdf)', '',
        '## Схема подсчёта Attention mass', '',
        '[Учебная схема с матрицей внимания](ATTENTION_MASS_SCHEME.md) · '
        '[SVG](attention_mass_scheme.svg) · [PDF](attention_mass_scheme.pdf)', '']
    for k in KS:
        name = f'top{k}_three_rankings'
        lines += [f'## Top-{k}', '', f'![Top-{k}]({name}.png)', '',
                  f'[SVG]({name}.svg) · [PDF для печати]({name}.pdf)', '']
    lines += ['Все клетки измерены; нулевых замен нет. Baseline = 100% recall в каждой позиции.',
        'Prompt-хэши совпадают между тремя рейтингами и baseline. Шкала одинакова у всех карт.',
        'Masking: legacy_uniform, только decode, BF16/A100, no-thinking, seed42, выход ≤256 токенов.',
        'Token multiset — модификация исходного hit-score, QRScore — дополнительный query-focused подход. '
        'Это собственное сравнение рейтингов в протоколе исходной статьи.', '',
        'Воспроизведение из updated: `.venv/bin/python plot_poster_glm_top40.py`.', '']
    (OUT / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
