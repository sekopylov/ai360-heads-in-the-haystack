"""Compare multiset/mass/QRScore on matched survey prompts and head identities."""
import csv
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from plot_metric_correlations import average_ranks, correlation
from plot_qrscore_experiments import MODELS, ROOT
from plot_paul_graham_mask import plt, BG

SOURCES = {
    'qwen3-8b': 'qwen3-8b-no-thinking-survey',
    'glm4-6v-flash': 'glm4-6v-flash-survey-download/glm4-6v-flash-no-thinking-context-only-survey',
    'ministral3-8b': 'ministral3-context-only-download/ministral3-8b-context-only-survey',
}
METRICS = ['needle_token_multiset_v1', 'needle_attention_mass_v1', 'query_retrieval_score_v1']
LABELS = ['Multiset', 'Attention mass', 'QRScore']
OUT = ROOT / 'report-assets/qrscore-colab/metric-correlations'


def case_key(case):
    return case['case_id'], case['length'], case['depth']


def load(slug):
    _, run_name, layers = MODELS[slug]
    source = ROOT / 'datasphere-results' / SOURCES[slug]
    detection = json.loads((source / 'detection/run.json').read_text())
    aggregation = json.loads((source / 'detection/aggregation/run.json').read_text())
    if not detection['complete'] or not aggregation['complete'] or aggregation['source_run_id'] != detection['run_id']:
        raise ValueError('Incomplete/mismatched historical survey')
    old_cases = aggregation['selected_cases']
    old_keys = [case_key(c) for c in old_cases]
    if len(set(old_keys)) != len(old_keys):
        raise ValueError('Duplicate historical cases')
    data = [json.loads((source / 'detection/aggregation' / aggregation['head_score_files'][m]).read_text())
            for m in METRICS[:2]]
    with ZipFile(ROOT / 'colab-results' / f'{run_name}-results.zip') as z:
        prefix = z.namelist()[0].split('/')[0]
        qr_run = json.loads(z.read(f'{prefix}/ranking/run.json'))
        qr = json.loads(z.read(f'{prefix}/ranking/head_scores_query_retrieval_score_v1.json'))
    if not qr_run['complete'] or not qr_run['filters']['all_cases']:
        raise ValueError('Incomplete/unexpected QRScore source')
    qr_cases = qr_run['selected_cases']
    qr_index = {case_key(c): i for i, c in enumerate(qr_cases)}
    if len(qr_index) != len(qr_cases) or not set(old_keys) <= set(qr_index):
        raise ValueError('Survey cases cannot be aligned')
    for case in old_cases:
        if case['prompt_sha256'] != qr_cases[qr_index[case_key(case)]]['prompt_sha256']:
            raise ValueError(f'Unmatched prompt: {case_key(case)}')
    heads = sorted(qr, key=lambda h: tuple(map(int, h.split('-'))))
    expected = {f'{l}-{h}' for l in range(layers) for h in range(32)}
    if set(heads) != expected or any(set(d) != expected for d in data):
        raise ValueError('Head identities/geometry differ')
    if any(len(v) != len(old_cases) for d in data for v in d.values()) or any(len(v) != len(qr_cases) for v in qr.values()):
        raise ValueError('Score histories do not match case manifests')
    data.append({h: [qr[h][qr_index[k]] for k in old_keys] for h in heads})
    values = np.array([[np.mean(d[h]) for h in heads] for d in data])
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite score')
    provenance = dict(historical_source=str(source), historical_run_id=detection['run_id'],
        qr_source_run_id=qr_run['source_run_id'], matched_cases=len(old_cases),
        qr_available_cases=len(qr_cases), matched_prompt_hashes=True,
        cases=[{k: c[k] for k in ('case_id', 'length', 'depth', 'prompt_sha256')} for c in old_cases],
        scope={'multiset': 'all_decode_tokens', 'mass': 'all_decode_tokens', 'qr': 'question-to-needle prefill'},
        note='Different runs, matched prompts; historical successful-case selection retained for all three metrics.')
    return heads, values, provenance


def draw(ax, matrix, title):
    numeric = np.array([[np.nan if v is None else v for v in row] for row in matrix])
    im = ax.imshow(numeric, vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set_xticks(range(3), LABELS, rotation=20, ha='right')
    ax.set_yticks(range(3), LABELS)
    ax.set_title(title)
    for i, j in np.ndindex((3, 3)):
        v = matrix[i][j]
        ax.text(j, i, 'n/a' if v is None else f'{v:.3f}', ha='center', va='center',
                fontsize=13, color='white' if v is not None and abs(v) > .6 else 'black')
    return im


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': BG, 'svg.fonttype': 'none'})
    combined, combined_axes = plt.subplots(3, 2, figsize=(12, 16), layout='constrained')
    results = {}
    for model_index, (slug, (label, _, _)) in enumerate(MODELS.items()):
        heads, values, provenance = load(slug)
        ranks = [average_ranks(v) for v in values]
        pearson = [[correlation(x, y) for y in values] for x in values]
        spearman = [[correlation(x, y) for y in ranks] for x in ranks]
        folder = OUT / slug
        folder.mkdir(exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), layout='constrained')
        for ax, common_ax, matrix, title in zip(axes, combined_axes[model_index], [pearson, spearman],
                                              ['Pearson · значения', 'Spearman · ранги']):
            im = draw(ax, matrix, title)
            combined_im = draw(common_ax, matrix, f'{label} · {title}\n{len(heads)} голов · {provenance["matched_cases"]} случаев')
        fig.colorbar(im, ax=axes, shrink=.75, label='Корреляция')
        fig.suptitle(f'{label} · no thinking\n{len(heads)} Q-голов · {provenance["matched_cases"]} одинаковых случаев')
        fig.supxlabel('Multiset/mass: decode; QRScore: prefill. Разные прогоны, совпадающие prompt-хэши.', fontsize=9)
        for ext in ('png', 'svg'):
            fig.savefig(folder / f'correlation_matrices.{ext}', dpi=170, bbox_inches='tight')
        plt.close(fig)
        with (folder / 'head_scores.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['head', *METRICS])
            writer.writerows([h, *values[:, i]] for i, h in enumerate(heads))
        for name, matrix in [('pearson', pearson), ('spearman', spearman)]:
            with (folder / f'{name}.csv').open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['metric', *LABELS])
                writer.writerows([label_m, *row] for label_m, row in zip(LABELS, matrix))
        result = dict(model=label, metrics=METRICS, head_count=len(heads), pearson=pearson,
                      spearman=spearman, provenance=provenance)
        (folder / 'correlations.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
        results[slug] = result
        print(label, 'cases', provenance['matched_cases'], 'Pearson', pearson, 'Spearman', spearman, flush=True)
    combined.colorbar(combined_im, ax=combined_axes, shrink=.6, label='Корреляция')
    combined.suptitle('Корреляции оценок Q-голов: multiset / attention mass / QRScore', fontsize=17)
    combined.supxlabel('Каждая корреляция — по головам внутри одной модели, не между моделями.\n'
                       'Оценки усреднены по совпадающим случаям; разные стадии attention и разные прогоны.', fontsize=10)
    for ext in ('png', 'svg'):
        combined.savefig(OUT / f'all_models_correlation_matrices.{ext}', dpi=170, bbox_inches='tight')
    plt.close(combined)
    lines = ['# Корреляции: multiset / mass / QRScore', '',
        'Pearson — по средним значениям; Spearman — по рангам, с усреднением рангов при ties.',
        'Наблюдение — одна и та же Q-голова внутри модели. Головы разных моделей не сопоставляются.',
        'Multiset и mass взяты из старых no-thinking survey (decode), QRScore — из новых survey (prefill).',
        'Это корреляции разных прогонов, не контролируемое сравнение метрик на одной генерации.',
        'Совпадение prompt-хэшей проверено для каждого случая. Выборка старой агрегации сохранена:',
        'Qwen и GLM — 27 случаев; Ministral — 23 успешных, QRScore также усреднён только по этим 23.', '',
        '![Все модели](all_models_correlation_matrices.png)', '', '[SVG](all_models_correlation_matrices.svg)', '']
    for slug, result in results.items():
        lines += [f'## {result["model"]}', '', f'![{result["model"]}]({slug}/correlation_matrices.png)', '',
                  f'[SVG]({slug}/correlation_matrices.svg) · [Pearson CSV]({slug}/pearson.csv) · '
                  f'[Spearman CSV]({slug}/spearman.csv) · [Оценки голов]({slug}/head_scores.csv) · '
                  f'[Источники]({slug}/correlations.json)', '']
    (OUT / 'README.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
