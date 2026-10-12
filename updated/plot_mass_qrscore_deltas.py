"""Paired top-k answer-quality deltas: attention-mass masking minus QRScore masking."""
import csv
import json
from pathlib import Path
from statistics import mean

import numpy as np
from plot_paul_graham_mask import plt, rescore, METRICS, LABELS
from plot_qrscore_experiments import ROOT, KS, LENGTHS, DEPTHS, MODELS, heatmap, save

SOURCES = {
    'glm4-6v-flash': 'glm4-6v-flash-attention-mass-mask-download/glm4-6v-flash-no-thinking-paul-graham-attention-mass-mask-san-francisco',
    'ministral3-8b': 'ministral3-attention-mass-mask-download/ministral3-8b-no-thinking-paul-graham-attention-mass-mask-san-francisco',
}
OUT = ROOT / 'report-assets/qrscore-colab/attention-mass-minus-qrscore'


def load_pairs(slug):
    source = ROOT / 'datasphere-results' / SOURCES[slug]
    qr_path = ROOT / 'report-assets/qrscore-colab' / slug / 'cases.csv'
    with qr_path.open() as f:
        qr = {(r['condition'], int(r['context_length']), float(r['depth_percent'])): r
              for r in csv.DictReader(f) if r['condition'].startswith('top')}
    mass_rows = []
    for k in KS:
        folder = source / f'evaluation/context-42/evaluation/top{k}'
        run = json.loads((folder / 'run.json').read_text())
        if (not run['complete'] or run['completed_cases'] != 110 or run['max_new_tokens'] != 256
                or tuple(run['lengths']) != LENGTHS or tuple(run['depths']) != DEPTHS):
            raise ValueError(f'Unexpected/incomplete attention-mass manifest: {folder}')
        for path in sorted((folder / 'results').glob('*_results.json')):
            r = json.loads(path.read_text())
            exp = r['experiment']
            if (exp['condition'] != f'top{k}' or len(exp['blocked_heads']) != k
                    or exp['mask_mode'] != 'legacy_uniform' or exp['decode_attention'] != 'sdpa_flash'):
                raise ValueError(f'Unexpected masking settings: {path}')
            r.update(condition=f'top{k}', source=str(path))
            mass_rows.append(r)
    rescore(mass_rows)
    pairs, seen = [], set()
    for r in mass_rows:
        key = r['condition'], r['context_length'], r['depth_percent']
        if key in seen or key not in qr:
            raise ValueError(f'Duplicate/unmatched mass case: {key}')
        seen.add(key)
        q = qr[key]
        if r['prompt_sha256'] != q['prompt_sha256']:
            raise ValueError(f'Unpaired prompts: {key}')
        if not np.isclose(r['score'], r['rouge1_recall'], atol=1e-6):
            raise ValueError('Stored/recomputed mass ROUGE differs')
        pair = dict(condition=key[0], context_length=key[1], depth_percent=key[2],
                    prompt_sha256=r['prompt_sha256'], mass_source=r['source'], qr_source=q['source'])
        for m in METRICS:
            pair[f'mass_{m}'] = r[m]
            pair[f'qr_{m}'] = float(q[m])
            pair[m] = r[m] - float(q[m])
        pairs.append(pair)
    expected = {(f'top{k}', l, d) for k in KS for l in LENGTHS for d in DEPTHS}
    if seen != expected:
        raise ValueError('Incomplete comparison grid')
    return pairs, source, qr_path


def delta_matrix(group, metric):
    lookup = {(r['context_length'], r['depth_percent']): r[metric] for r in group}
    return np.array([[lookup[l, d] for d in DEPTHS] for l in LENGTHS])


def build(slug):
    model = MODELS[slug][0]
    pairs, source, qr_path = load_pairs(slug)
    folder = OUT / slug
    plots = folder / 'plots'
    plots.mkdir(parents=True, exist_ok=True)
    gallery, summary = [], []
    for k in KS:
        group = [r for r in pairs if r['condition'] == f'top{k}']
        fig, axes = plt.subplots(2, 3, figsize=(17, 10), layout='constrained')
        fig.suptitle(f'{model} · top{k} · Δ = attention mass − QRScore', fontsize=19)
        for ax, metric, label in zip(axes.flat, METRICS, LABELS):
            im = heatmap(ax, delta_matrix(group, metric), label, delta=True)
            fig.colorbar(im, ax=ax, shrink=.7, label='Δ качества, п.п.')
        axes.flat[-1].axis('off')
        axes.flat[-1].text(.03, .9, 'QRScore — точка отсчёта.\n\n'
            'Синий (Δ < 0): качество ниже у mass.\nКрасный (Δ > 0): качество ниже у QRScore.\n\n' +
            '\n\n'.join(f'{label}: {mean(r[m] for r in group):+.2f} п.п.'
                        for m, label in zip(METRICS, LABELS)), va='top', fontsize=11)
        name = f'top{k}_delta_heatmaps'
        save(fig, plots, name, model, 'парные prompts · 110 случаев · Δ mass − QRScore, п.п.')
        gallery.append((f'top{k}: все метрики качества', name))
        summary.append(dict(condition=f'top{k}', cases=len(group), **{m: mean(r[m] for r in group) for m in METRICS}))
    for metric, label in zip(METRICS, LABELS):
        fig, axes = plt.subplots(2, 4, figsize=(23, 12), layout='constrained')
        fig.suptitle(f'{model} · {label} · Δ = attention mass − QRScore', fontsize=20)
        for ax, k in zip(axes.flat, KS):
            im = heatmap(ax, delta_matrix([r for r in pairs if r['condition'] == f'top{k}'], metric),
                         f'top{k}', delta=True)
        fig.colorbar(im, ax=axes, shrink=.7, label='Δ качества, п.п.')
        name = f'overview_{metric}'
        save(fig, plots, name, model, 'все top-k · одинаковая шкала −100…+100 п.п.')
        gallery.append((f'{label}: все top-k на одной странице', name))
    for filename, rows in [('cases.csv', pairs), ('summary.csv', summary)]:
        with (folder / filename).open('w', newline='') as f:
            writer = csv.DictWriter(f, list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (folder / 'provenance.json').write_text(json.dumps(dict(mass_source=str(source),
        qr_scores=str(qr_path), formula='mass_quality - qr_quality', matched_prompt_hashes=True,
        paired_cases=len(pairs), topks=KS, lengths=LENGTHS, depths=DEPTHS,
        note='Different masking runs and survey rankings; not a same-run causal isolation.'), indent=2))
    lines = [f'# {model}: attention mass − QRScore', '',
        'QRScore — точка отсчёта. Δ = качество ответа после masking по mass минус качество после masking по QRScore.',
        'Единицы — процентные пункты ROUGE/точной фразы, не разница head-score.',
        'Синий: mass оставил меньше качества; красный: QRScore оставил меньше качества.',
        '880 пар: 8 top-k ×11 длин ×10 глубин. Prompt-хэши совпадают; все ответы включены.',
        'Оба прогона no thinking, legacy_uniform decode, выход ≤256 токенов. Исходные результаты не изменены.',
        'Источники survey/прогоны разные: дельта не изолирует исключительно эффект формулы метрики.',
        'Среднее ROUGE-1/2/3 — нестандартный композит; точная фраза — наличие нормализованного эталона подряд.', '',
        '[CSV пар](cases.csv) · [Средние дельты](summary.csv) · [Источники](provenance.json)', '']
    for title, name in gallery:
        lines += [f'## {title}', '', f'![{title}](plots/{name}.png)', '', f'[SVG](plots/{name}.svg)', '']
    (folder / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    print(slug, len(pairs), 'verified pairs;', len(gallery), 'figures', flush=True)


def main():
    for slug in SOURCES:
        build(slug)
    (OUT / 'README.md').write_text('# Δ качества: attention mass − QRScore\n\n' + '\n\n'.join(
        f'[{MODELS[slug][0]}]({slug}/README.md)' for slug in SOURCES) +
        '\n\nQwen исключён по запросу. QRScore — точка отсчёта; шкала в процентных пунктах.\n', encoding='utf-8')


if __name__ == '__main__':
    main()
