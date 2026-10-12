"""Offline paired GLM question/needle mass analysis from the completed survey ZIP."""
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from plot_metric_correlations import average_ranks, correlation
from plot_three_metric_correlations import load as load_historical

ROOT = Path(__file__).resolve().parent
NAME = 'glm4-6v-flash-no-thinking-question-mass-survey'
ARCHIVE = ROOT / 'colab-results' / f'{NAME}-results.zip'
OUT = ROOT / 'report-assets/qrscore-colab/glm-question-mass'
METRICS = ('needle_attention_mass_v1', 'question_attention_mass_v1', 'query_retrieval_score_v1')
LABELS = ('Needle mass', 'Question mass', 'Needle + question', 'QRScore')
COLORS = ('#207c9d', '#e28439', '#298567', '#8559a6')


def read_data():
    with ZipFile(ARCHIVE) as z:
        assert z.testzip() is None
        run = json.loads(z.read(f'{NAME}/detection/run.json'))
        assert run['complete'] and run['completed_cases'] == 27
        rows = [json.loads(z.read(n)) for n in sorted(z.namelist())
                if '/detection/results/' in n and n.endswith('_results.json')]
    expected = {(f'detect-{c}', l, d) for c in (1, 2, 3)
                for l in (8000, 16000, 30000) for d in (15, 45, 75)}
    key = lambda r: (r['case_id'], r['context_length'], r['depth_percent'])
    assert len(rows) == 27 and {key(r) for r in rows} == expected
    heads = [f'{l}-{h}' for l in range(40) for h in range(32)]
    for r in rows:
        e = r['experiment']
        assert e['run_id'] == run['run_id']
        assert e['question_attention_span'] == e['qrscore_query_span']
        g, q = e['qrscore_gold_span'], e['question_attention_span']
        assert 0 <= g[0] < g[1] <= q[0] < q[1]
        counts = e['retrieval_qualifying_steps']
        assert counts[METRICS[0]] == counts[METRICS[1]] > 0
        assert all(set(e['retrieval_scores'][m]) == set(heads) for m in METRICS)
    data = np.array([[[r['experiment']['retrieval_scores'][m][h] for h in heads]
                      for m in METRICS] for r in rows])
    assert np.isfinite(data).all() and np.all((data >= 0) & (data <= 1.000001))
    assert np.all(data[:, 0] + data[:, 1] <= 1.002)  # BF16 attention reduction tolerance
    mean = data.mean(axis=0)
    values = np.array([mean[0], mean[1], mean[0] + mean[1], mean[2]])
    old_heads, old_values, provenance = load_historical('glm4-6v-flash')
    assert heads == old_heads
    hashes = {(c['case_id'], c['length'], c['depth']): c['prompt_sha256'] for c in provenance['cases']}
    assert all(hashes[key(r)] == r['prompt_sha256'] for r in rows)
    return heads, rows, data, values, old_values, run, provenance


def stats(values):
    return {'pearson': correlation(values[0], values[1]),
            'spearman': correlation(average_ranks(values[0]), average_ranks(values[1]))}


def ranks(x):
    return len(x) + 1 - average_ranks(x)  # rank 1 is best; ties averaged


def top(x, k):
    return set(np.argsort(-x, kind='stable')[:k])


def save(fig, name):
    fig.supxlabel('GLM-4.6V-Flash · no thinking · 27 paired cases · 1280 Q-heads · '
                  'decode mass: identical needle-token-qualified steps; QRScore: prefill', fontsize=9)
    for ext in ('png', 'svg'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=170, bbox_inches='tight')
    plt.close(fig)


def write_csv(name, fields, rows):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    heads, rows, data, v, old, run, provenance = read_data()
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'figure.facecolor': '#fafaf8', 'svg.fonttype': 'none'})
    rank = np.array([ranks(x) for x in v])
    share = np.divide(v[1], v[2], out=np.zeros_like(v[1]), where=v[2] > 0)
    improvement = abs(rank[0] - rank[3]) - abs(rank[2] - rank[3])
    summary = {label: stats([v[i], v[3]]) for i, label in enumerate(LABELS[:3])}
    overlaps = [{ 'k': k, 'needle': len(top(v[0], k) & top(v[3], k)),
                 'sum': len(top(v[2], k) & top(v[3], k)),
                 'question': len(top(v[1], k) & top(v[3], k))}
                for k in (5, 10, 15, 20, 40, 60, 80, 100, 200)]
    summary.update(top_k_overlaps=overlaps,
        mean_absolute_rank_error_needle=float(np.mean(abs(rank[0] - rank[3]))),
        mean_absolute_rank_error_sum=float(np.mean(abs(rank[2] - rank[3]))),
        heads_rank_error_improved=int(np.sum(improvement > 0)),
        heads_rank_error_worsened=int(np.sum(improvement < 0)))

    fig, axes = plt.subplots(1, 3, figsize=(19, 6), layout='constrained')
    lower = min(v[v > 0]) * .65
    for i, ax in enumerate(axes):
        points = ax.scatter(v[i], v[3], c=share, vmin=0, vmax=1, cmap='plasma',
                            s=15, alpha=.7, linewidths=0)
        ax.plot([lower, 1], [lower, 1], '--', color='#888', lw=1)
        ax.set(xscale='log', yscale='log', xlim=(lower, 1.1), ylim=(lower, 1.1),
               xlabel=LABELS[i], ylabel='QRScore',
               title=f'{LABELS[i]} vs QRScore\nPearson {summary[LABELS[i]]["pearson"]:.3f} · '
                     f'Spearman {summary[LABELS[i]]["spearman"]:.3f}')
    fig.colorbar(points, ax=axes, label='Question / (needle + question)', shrink=.8)
    fig.suptitle('Does adding question attention bring decode scores closer to QRScore?', fontsize=17)
    save(fig, 'score_scatter')

    fig, axes = plt.subplots(1, 3, figsize=(19, 6), layout='constrained')
    for i, ax in zip((0, 2), axes[:2]):
        ax.scatter(rank[i], rank[3], c=share, cmap='plasma', vmin=0, vmax=1,
                   s=12, alpha=.6, linewidths=0)
        ax.plot([1, 1280], [1, 1280], '--', color='#888')
        ax.set(xlabel=f'{LABELS[i]} rank (1 = best)', ylabel='QRScore rank (1 = best)',
               title=f'Mean absolute rank error: {np.mean(abs(rank[i]-rank[3])):.1f}')
    for key, label, color in [('needle', 'Needle', COLORS[0]), ('sum', 'Needle + question', COLORS[2]),
                               ('question', 'Question', COLORS[1])]:
        axes[2].plot([r['k'] for r in overlaps], [100*r[key]/r['k'] for r in overlaps],
                     'o-', label=label, color=color)
    axes[2].set(xlabel='top-k', ylabel='Overlap with QRScore top-k, %', ylim=(-2, 102))
    axes[2].legend()
    fig.suptitle('Ranking agreement: needle vs needle + question', fontsize=17)
    save(fig, 'ranking_agreement')

    fig, axes = plt.subplots(1, 4, figsize=(23, 11), layout='constrained')
    shared_norm = LogNorm(vmin=min(v[:3][v[:3] > 0]), vmax=max(v[:3].flat))
    grids = [v[0], v[1], share, improvement]
    titles = ['Needle mass', 'Question mass', 'Question / (needle + question)',
              'Rank error reduction toward QRScore\npositive = sum is closer']
    for i, ax in enumerate(axes):
        kwargs = {'norm': shared_norm, 'cmap': 'viridis'} if i < 2 else (
            {'vmin': 0, 'vmax': 1, 'cmap': 'plasma'} if i == 2 else
            {'vmin': -max(abs(improvement)), 'vmax': max(abs(improvement)), 'cmap': 'RdBu_r'})
        im = ax.imshow(grids[i].reshape(40, 32), aspect='auto', **kwargs)
        ax.set(title=titles[i], xlabel='Q-head', ylabel='Layer')
        ax.set_xticks(range(0, 32, 4)); ax.set_yticks(range(0, 40, 2))
        fig.colorbar(im, ax=ax, shrink=.75)
    fig.suptitle('Where does question attention change the picture?', fontsize=17)
    save(fig, 'head_heatmaps')

    old_mass_top, old_qr_top = top(old[1], 20), top(old[2], 20)
    groups = [sorted(old_qr_top - old_mass_top), sorted(old_mass_top - old_qr_top),
              sorted(old_qr_top & old_mass_top)]
    group_labels = ['Previous QRScore-only top-20', 'Previous needle-only top-20', 'Previous shared top-20']
    fig, axes = plt.subplots(3, 1, figsize=(18, 13), layout='constrained')
    for ax, indices, label in zip(axes, groups, group_labels):
        x = np.arange(len(indices))
        for i in (0, 1, 3):
            ax.bar(x + ((0, 1, 3).index(i)-1)*.25, v[i, indices], width=.25,
                   color=COLORS[i], label=LABELS[i])
        ax.set_xticks(x, [heads[j] for j in indices])
        ax.set(title=f'{label} · {len(indices)} heads', ylabel='Mean score', xlabel='Layer-head')
        ax.legend(ncol=3)
    fig.suptitle('Heads in the previous top-20 disagreement', fontsize=17)
    save(fig, 'discordant_top20')

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), layout='constrained')
    for i in (0, 1, 2):
        axes[0].plot(range(40), v[i].reshape(40,32).mean(axis=1), color=COLORS[i], label=LABELS[i])
    axes[0].set(xlabel='Layer', ylabel='Mean score over 32 Q-heads', title='Attention destination by layer')
    axes[0].legend()
    case_stats = []
    for row, scores in zip(rows, data):
        s = scores[0]+scores[1]
        a, b = stats([scores[0],scores[2]]), stats([s,scores[2]])
        case_stats.append({'case_id': row['case_id'], 'length': row['context_length'],
                           'depth': row['depth_percent'], 'qualifying_steps': row['experiment']['retrieval_qualifying_steps'][METRICS[0]],
                           'needle_spearman': a['spearman'], 'sum_spearman': b['spearman'],
                           'delta_spearman': b['spearman']-a['spearman']})
    for c, color in zip(('detect-1','detect-2','detect-3'), COLORS):
        selected = [r for r in case_stats if r['case_id']==c]
        axes[1].scatter([r['needle_spearman'] for r in selected],
                        [r['sum_spearman'] for r in selected], label=c, color=color, s=45)
    lo = min(min(r['needle_spearman'],r['sum_spearman']) for r in case_stats)-.015
    axes[1].plot([lo,1],[lo,1],'--',color='#888')
    axes[1].set(xlim=(lo,1),ylim=(lo,1),xlabel='Spearman: needle vs QRScore',
                ylabel='Spearman: sum vs QRScore', title='Each dot = one case · above diagonal = improvement')
    axes[1].legend()
    save(fig, 'layers_and_cases')

    head_rows = [{'head':h,'layer':i//32,'q_head':i%32, 'needle_mass':v[0,i],
                 'question_mass':v[1,i],'sum_mass':v[2,i],'qrscore':v[3,i],
                 'question_share':share[i], 'needle_rank':rank[0,i], 'sum_rank':rank[2,i],
                 'qrscore_rank':rank[3,i], 'rank_error_reduction':improvement[i]} for i,h in enumerate(heads)]
    write_csv('head_scores.csv', list(head_rows[0]), head_rows)
    write_csv('case_correlations.csv', list(case_stats[0]), case_stats)
    write_csv('top_k_overlap.csv', list(overlaps[0]), overlaps)
    summary['previous_top20_groups'] = {label:{'heads':[heads[i] for i in indices],
        'mean_question_mass':float(v[1,indices].mean()), 'mean_needle_mass':float(v[0,indices].mean()),
        'mean_question_share':float(share[indices].mean())} for label,indices in zip(group_labels,groups)}
    summary['cases_spearman_improved'] = sum(r['delta_spearman'] > 0 for r in case_stats)
    (OUT / 'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    (OUT / 'provenance.json').write_text(json.dumps({'archive':str(ARCHIVE.relative_to(ROOT)),
        'sha256':hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(), 'run':run,
        'historical':provenance, 'matched_prompt_hashes':27,
        'note':'Paired measurements from one new run; historical rankings only define diagnostic groups.'},indent=2))
    lines = ['# GLM: needle + question mass и QRScore', '',
        '1280 Q-голов, 27 случаев. Все три исходные метрики измерены в одном новом прогоне.',
        'Needle и question mass используют одинаковые decode-шаги: generated token ID встречается в span expected_answer.',
        'Needle здесь означает span expected_answer; QRScore использует полную вставленную needle.',
        'Question span и needle не пересекаются. Их сумма — decode-внимание к объединению этих позиций.',
        'Score усредняются сначала по qualifying steps случая, затем поровну по 27 случаям.',
        'Доля вопроса — mean(question) / (mean(needle) + mean(question)), без нормализации на длину spans.', '',
        '## Совпадение с QRScore', '',
        '| Decode score | Pearson | Spearman |', '|---|---:|---:|']
    for label in LABELS[:3]:
        lines.append(f'| {label} | {summary[label]["pearson"]:.4f} | {summary[label]["spearman"]:.4f} |')
    lines += ['', f'Spearman улучшается после добавления question mass в {summary["cases_spearman_improved"]}/27 отдельных случаях.',
              '', '| top-k | Needle ∩ QR | (Needle + question) ∩ QR |', '|---|---:|---:|']
    lines += [f'| {r["k"]} | {r["needle"]} | {r["sum"]} |' for r in overlaps]
    lines += ['', '## Что получилось', '',
        'Простая добавка question mass не объясняет расхождение рейтингов: '
        'Spearman падает с 0.9629 до 0.8869, средняя абсолютная ошибка ранга '
        'растёт с 77.8 до 129.7. В каждом из 27 случаев Spearman также снижается.',
        'Top-20 сохраняет пересечение 7/20; небольшое улучшение есть для top-100 '
        '(69 → 71) и top-200 (146 → 151).',
        'У 13 голов прежнего QRScore-only top-20 средняя question mass = 0.01338, '
        'needle mass = 0.60908, средняя доля вопроса = 2.16%. '
        'У 13 mass-only голов соответствующие значения 0.00598, 0.80298 и 0.77%. '
        'Эти группы различаются, но question mass мала относительно needle mass '
        'и не меняет пересечение top-20.',
        'Наши данные не поддерживают объяснение «mass недооценивает QRScore-heads '
        'главным образом из-за неучтённого внимания к вопросу» в таком виде. '
        'Они не исключают переноса информации через вопрос: небольшое внимание '
        'тоже может иметь причинный вклад, а информация может передаваться через '
        'другие позиции и residual stream. Для этого нужен отдельный причинный тест.', '']
    lines += ['', 'Корреляция не доказывает перенос информации об ответе через вопрос или причинную важность головы.',
              'Сумма — проверяемый диагностический score; masking по этому рейтингу ещё не выполнялся.',
              '27 случаев — три вопроса на девяти позициях/длинах; головы и случаи не считаются независимыми репликациями.',
              'Прежние prompts совпали в 27/27 случаях; диагностические top-20 группы взяты из прежних рейтингов.', '',
              '[Scores CSV](head_scores.csv) · [По случаям](case_correlations.csv) · '
              '[Пересечения top-k](top_k_overlap.csv) · [Сводка](summary.json) · [Источники](provenance.json)', '']
    for name,title in [('score_scatter','Исходные score: needle, question, сумма против QRScore'),
                       ('ranking_agreement','Ранги и пересечения top-k'),('head_heatmaps','Все слои и головы'),
                       ('discordant_top20','Головы из несовпадающих прежних top-20'),
                       ('layers_and_cases','Распределение по слоям и устойчивость по случаям')]:
        lines += [f'## {title}', '', f'![{title}]({name}.png)', '', f'[SVG]({name}.svg)', '']
    lines += ['Воспроизведение из `updated/`: `.venv/bin/python plot_glm_question_mass.py`.', '']
    (OUT/'README.md').write_text('\n'.join(lines))
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
