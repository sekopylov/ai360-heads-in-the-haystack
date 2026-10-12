"""Poster companion: three frozen GLM head-score maps, layers on X, heads on Y."""
import csv
import hashlib
import json
from statistics import mean
from zipfile import ZipFile

import numpy as np

from plot_poster_glm_top40 import OUT, ROOT, SURVEY, plt

LAYERS = 40
HEADS = 32
METRICS = ('needle_token_multiset_v1', 'needle_attention_mass_v1', 'query_retrieval_score_v1')
LABELS = ('Retrieval\nScore', 'Attention\nmass', 'QRScore')
NAME = 'head_scores_three_metrics'


def case_key(case):
    return case['case_id'], case['length'], case['depth']


def load_scores():
    aggregation_dir = SURVEY / 'detection/aggregation'
    aggregation = json.loads((aggregation_dir / 'run.json').read_text())
    detection = json.loads((SURVEY / 'detection/run.json').read_text())
    if (not aggregation['complete'] or not detection['complete']
            or aggregation['source_run_id'] != detection['run_id']):
        raise ValueError('Incomplete or mismatched historical survey')
    cases = aggregation['selected_cases']
    if len(cases) != aggregation['selected_case_count'] or len(cases) != 27:
        raise ValueError('Unexpected historical survey case count')
    old_keys = [case_key(c) for c in cases]
    if len(set(old_keys)) != len(old_keys):
        raise ValueError('Duplicate survey cases')
    histories = []
    sources = []
    for metric in METRICS[:2]:
        path = aggregation_dir / aggregation['head_score_files'][metric]
        histories.append(json.loads(path.read_text()))
        sources.append(dict(metric=metric, path=str(path.relative_to(ROOT)),
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

    archive = ROOT / 'colab-results/glm4-6v-flash-no-thinking-paul-graham-qrscore-top-mask-results.zip'
    with ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or z.testzip() is not None:
            raise ValueError('Corrupt or duplicate-member QRScore archive')
        prefix = names[0].split('/')[0]
        member = f'{prefix}/ranking/head_scores_query_retrieval_score_v1.json'
        qr = json.loads(z.read(member))
        frozen_means = json.loads(z.read(f'{prefix}/ranking/mean_scores.json'))
        qr_run = json.loads(z.read(f'{prefix}/ranking/run.json'))
    if (not qr_run['complete'] or not qr_run['filters']['all_cases']
            or qr_run['retrieval_metrics'] != [METRICS[2]]):
        raise ValueError('Incomplete or unexpected QRScore survey')
    qr_cases = qr_run['selected_cases']
    qr_indices = {case_key(c): i for i, c in enumerate(qr_cases)}
    if (len(qr_indices) != len(qr_cases) or len(qr_cases) != 27
            or qr_run['selected_case_count'] != len(qr_cases)
            or set(qr_indices) != set(old_keys)):
        raise ValueError('Survey case selections differ')
    for case in cases:
        if case['prompt_sha256'] != qr_cases[qr_indices[case_key(case)]]['prompt_sha256']:
            raise ValueError(f'Unmatched survey prompt: {case_key(case)}')

    expected = {f'{layer}-{head}' for layer in range(LAYERS) for head in range(HEADS)}
    if set(qr) != expected or set(frozen_means) != expected:
        raise ValueError('Unexpected QRScore head geometry')
    for head, values in qr.items():
        if len(values) != len(qr_cases) or not np.isclose(mean(values), frozen_means[head], rtol=0, atol=1e-12):
            raise ValueError(f'Frozen QRScore mean mismatch: {head}')
    histories.append({h: [qr[h][qr_indices[key]] for key in old_keys] for h in qr})
    sources.append(dict(metric=METRICS[2], path=str(archive.relative_to(ROOT)), member=member,
                        sha256=hashlib.sha256(archive.read_bytes()).hexdigest()))

    matrices = []
    rows = []
    for metric, history in zip(METRICS, histories):
        if set(history) != expected:
            raise ValueError(f'Unexpected head geometry: {metric}')
        for head, values in history.items():
            if (len(values) != len(cases)
                    or not all(np.isfinite(v) and 0 <= v <= 1.000001 for v in values)):
                raise ValueError(f'Invalid score history: {metric}/{head}')
        scores = {h: mean(values) for h, values in history.items()}
        # Stable numeric head identity: rows = head ID, columns = layer ID.
        matrix = np.array([[scores[f'{layer}-{head}'] for layer in range(LAYERS)] for head in range(HEADS)])
        matrices.append(matrix)
        for layer in range(LAYERS):
            for head in range(HEADS):
                rows.append(dict(metric=metric, layer=layer, head=head,
                                 mean_score=scores[f'{layer}-{head}'], cases=len(cases)))
    provenance = dict(model='GLM-4.6V-Flash', sources=sources, layers=LAYERS, heads_per_layer=HEADS,
                      heads_per_metric=len(expected), matched_prompt_count=len(cases),
                      selected_cases=[{k: c[k] for k in ('case_id', 'length', 'depth', 'prompt_sha256')} for c in cases],
                      row_order=METRICS, x_axis='layer (zero-based)', y_axis='Q-head (zero-based)',
                      color_scale=dict(type='linear', vmin=0, vmax=1, cmap='magma', colorbar_displayed=False),
                      head_axis_labels_displayed=True, title_displayed=False,
                      normalization='None; arithmetic means of saved head scores, not ROUGE.',
                      imputed_cases=0)
    return matrices, rows, provenance


def main():
    matrices, rows, provenance = load_scores()
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'figure.facecolor': 'none', 'axes.facecolor': 'none', 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig = plt.figure(figsize=(17, 14.4), layout='constrained')
    fig.get_layout_engine().set(hspace=.06)
    # Same label-column / total chart-width ratio as the masking companion.
    grid = fig.add_gridspec(3, 2, width_ratios=(.55, 3))
    axes = []
    for idx, (label, values) in enumerate(zip(LABELS, matrices)):
        label_ax = fig.add_subplot(grid[idx, 0])
        label_ax.set_axis_off()
        label_ax.text(.68, .5, label, ha='right', va='center', fontsize=16, weight='normal')
        shared = axes[0] if axes else None
        ax = fig.add_subplot(grid[idx, 1], sharex=shared, sharey=shared)
        axes.append(ax)
        # Restore the magma palette of the original survey/head_heatmap plots.
        ax.imshow(values, cmap='magma', vmin=0, vmax=1, aspect='auto', interpolation='nearest')
        ax.set_ylabel('Номер головы', fontsize=12, labelpad=10)
        ax.set_yticks(range(HEADS), range(HEADS), fontsize=8)
        ax.set_xticks(range(LAYERS), range(LAYERS), fontsize=9)
        ax.tick_params(labelbottom=True)
        ax.set_xlabel('Слой', fontsize=12)
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(OUT / f'{NAME}.{ext}', dpi=220, bbox_inches='tight', transparent=True)
    plt.close(fig)

    with (OUT / 'head_scores.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / 'head_scores_provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    lines = [
        '# GLM: оценки всех голов по трём метрикам', '',
        f'![Оценки голов]({NAME}.png)', '',
        f'[SVG]({NAME}.svg) · [PDF для печати]({NAME}.pdf) · [CSV](head_scores.csv) · '
        '[Источники и проверка](head_scores_provenance.json)', '',
        'Фон PNG/SVG/PDF прозрачен для размещения на плакате.', '',
        'Три карты расположены сверху вниз в том же порядке, что строки рейтингов на фигуре '
        'маскирования: Retrieval Score, Attention mass, QRScore. Bottom-контроль не имеет отдельной '
        'метрики и здесь не повторяется. Названия слева — обычным начертанием, отступы '
        'согласованы с фигурой маскирования; номера слоёв и подпись X показаны под каждой картой. Палитра '
        '**magma** — как в исходных survey/head_heatmap: чёрный → фиолетовый → оранжевый → '
        'светло-жёлтый. Общий заголовок и цветовая шкала скрыты; номера голов 0–31 '
        'и подпись оси голов показаны на каждой карте.', '',
        'По X — слои 0–39, по Y — Q-головы 0–31; одна клетка — одна из 1280 голов. '
        'Все три карты используют общую **линейную шкалу 0–1**, без нормировки на максимум '
        'или ранг. Цвет показывает среднюю исходную оценку головы, **не ROUGE и не качество ответа**. '
        'Малые ненулевые значения могут выглядеть как нулевые из-за линейной шкалы; точные оценки '
        'доступны в CSV.', '',
        'Использованы ровно те замороженные рейтинги, по которым выбирались маски на соседней фигуре: '
        '`needle_token_multiset_v1`, `needle_attention_mass_v1`, `query_retrieval_score_v1`. '
        'Retrieval Score — краткая подпись нашей multiset-версии, не неизменённого legacy-кода авторов. '
        'Attention mass измерен на decode по прежнему span эталонного ответа, а не в эксперименте '
        'full-needle; QRScore — вопрос → needle на prefill.', '',
        'Средние рассчитаны по 27 одинаковым survey-prompts; совпадение case ID, длины, глубины '
        'и prompt_sha256 проверено, порядок историй выровнен. Проверены все 1280 идентификаторов '
        'голов для каждой метрики, диапазоны и длины историй, а также совпадение QRScore '
        'с сохранёнными средними. Нет пропусков или подставленных нулей.', '',
        'Воспроизведение из updated: `.venv/bin/python plot_poster_glm_head_scores.py`.', '',
    ]
    (OUT / 'HEAD_SCORES.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(output=str(OUT / f'{NAME}.svg'), heads_per_metric=LAYERS*HEADS,
                          matched_prompts=provenance['matched_prompt_count'],
                          ranges={metric: [float(values.min()), float(values.max())]
                                  for metric, values in zip(METRICS, matrices)}), indent=2))


if __name__ == '__main__':
    main()
