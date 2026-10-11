"""Build independently rescored tables from downloaded top5–40 masking JSON."""
import csv
import json
from pathlib import Path
from statistics import mean

from plot_paul_graham_mask import rescore, METRICS, LABELS

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'datasphere-results/qwen3-8b-no-thinking-paul-graham-new-mass-top5-40-mask-san-francisco'
OUT = ROOT / 'report-assets/qwen3-8b-no-thinking-paul-graham-new-mass-top5-40'
HEADS = (0, 5, 10, 15, 20, 25, 30, 35, 40)


def table(headers, values):
    return ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---:' for _ in headers) + ' |',
            *['| ' + ' | '.join(str(v) if isinstance(v, str) else f'{v:.2f}' for v in row) + ' |' for row in values]]


def main():
    rows = []
    for path in sorted(SOURCE.glob('evaluation/context-42/evaluation/*/results/*_results.json')):
        row = json.loads(path.read_text()); condition = path.parent.parent.name
        row.update(condition=condition, blocked_heads=0 if condition == 'baseline' else int(condition.removeprefix('top')), source=str(path.relative_to(ROOT)))
        rows.append(row)
    lengths = sorted({r['context_length'] for r in rows}); depths = sorted({r['depth_percent'] for r in rows})
    keys = {(r['blocked_heads'], r['context_length'], r['depth_percent']) for r in rows}
    assert len(rows) == len(keys) == 1800
    assert len(lengths) == 20 and len(depths) == 10
    assert keys == {(h, l, d) for h in HEADS for l in lengths for d in depths}
    for l in lengths:
        for d in depths:
            assert len({r['prompt_sha256'] for r in rows if r['context_length'] == l and r['depth_percent'] == d}) == 1
    rescore(rows)
    assert max(abs(r['score'] - r['rouge1_recall']) for r in rows) < 1e-6
    summary = []
    for h in HEADS:
        group = [r for r in rows if r['blocked_heads'] == h]
        summary.append({'blocked_heads': h, 'cases': len(group), **{m: mean(r[m] for r in group) for m in METRICS},
                        'generation_minutes': sum(r['test_duration_seconds'] for r in group) / 60,
                        'max_new_tokens_count': sum(r.get('finish_reason') == 'max_new_tokens' for r in group)})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'scores.json').write_text(json.dumps({'source': str(SOURCE), 'cases': rows}, ensure_ascii=False, indent=2))
    fields = ['condition', 'blocked_heads', 'context_length', 'depth_percent', 'score', *[f'rouge{n}_{c}' for n in (1, 2, 3) for c in ('recall', 'precision', 'fmeasure')], *[f'rouge123_mean_{c}' for c in ('recall', 'precision', 'fmeasure')], 'exact_match', 'finish_reason', 'test_duration_seconds', 'source']
    for filename, data, names in [('cases.csv', rows, fields), ('summary.csv', summary, list(summary[0]))]:
        with (OUT / filename).open('w') as f:
            writer = csv.DictWriter(f, names, extrasaction='ignore'); writer.writeheader(); writer.writerows(data)
    lines = ['# Новый no-thinking mass-рейтинг: top5–40', '',
             'Job bt1fsj89jp523daa3n8g: SUCCESS, завершён 10 октября 2026 в 03:03:41 МСК; от создания 2 ч 23 мин 32 с.', '',
             '1800 исходных JSON скачаны из DataSphere. Полная сетка: 20 длин × 10 глубин × 9 условий, одинаковые prompt SHA256 между условиями. Рейтинг — новый no-thinking survey, модель — Qwen3-8B без thinking, A100, legacy_uniform только при decode, выход 256, seed 42.', '',
             'ROUGE-1/2/3: recall со stemming ×100. Композит — арифметическое среднее трёх recall, не стандартная метрика ROUGE. Exact phrase — наличие полной нормализованной эталонной фразы подряд, без stemming. Precision и F1 также сохранены. Пересчёт не меняет исходные оценки, рейтинги или ответы; семантическое перефразирование и отрицание метрики не проверяют.', '',
             '## Итог по 200 примерам на условие', '']
    lines += table(['Голов', 'ROUGE-1', 'ROUGE-2', 'ROUGE-3', 'Среднее 1/2/3', 'Exact phrase, %'], [[str(s['blocked_heads']), *[s[m] for m in METRICS]] for s in summary])
    lines += ['', '## Время генераций и остановка по лимиту', '']
    lines += table(['Голов', 'Генерации, мин', 'Лимит выхода, случаев'], [[str(s['blocked_heads']), s['generation_minutes'], str(s['max_new_tokens_count'])] for s in summary])
    for field, grid, title in [('context_length', lengths, 'По длине — среднее по 10 глубинам'), ('depth_percent', depths, 'По глубине — среднее по 20 длинам')]:
        lines += ['', '## ' + title, '']
        for metric, label in zip(METRICS, LABELS):
            lines += ['### ' + label, '']
            lines += table(['Длина' if field == 'context_length' else 'Глубина, %', *[str(h) for h in HEADS]], [[f'{x:g}', *[mean(r[metric] for r in rows if r['blocked_heads'] == h and r[field] == x) for h in HEADS]] for x in grid])
            lines += ['']
    lines += ['## Полные таблицы «длина × глубина» для каждого числа голов', '']
    for h in HEADS:
        lines += [f'### {h} голов', '']
        for metric, label in zip(METRICS, LABELS):
            lookup = {(r['context_length'], r['depth_percent']): r[metric] for r in rows if r['blocked_heads'] == h}
            lines += [f'#### {label}', '']
            lines += table(['Длина / глубина, %', *[f'{d:g}' for d in depths]], [[str(l), *[lookup[l, d] for d in depths]] for l in lengths])
            lines += ['']
    lines += ['Данные: [cases.csv](cases.csv), [summary.csv](summary.csv), [scores.json](scores.json).']
    (OUT / 'README.md').write_text('\n'.join(lines))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
