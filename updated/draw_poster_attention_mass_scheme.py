"""Vector teaching diagram for the implemented needle attention-mass score."""
import json

import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from plot_poster_glm_top40 import OUT, plt

NAME = 'attention_mass_scheme'
INK = '#242638'
MUTED = '#697080'
ACCENT = '#db7438'
GREEN = '#28724b'

# Columns: 3 context keys, 4 target-span keys, 2 question keys, 2 generated keys.
# NaN denotes a future key that does not yet exist (causally unavailable).
ATTENTION = np.array([
    [.04, .03, .03, .10, .05, .35, .15, .15, .10, np.nan, np.nan],
    [.08, .06, .06, .05, .05, .10, .10, .20, .15, .15, np.nan],
    [.03, .02, .05, .10, .05, .20, .40, .05, .04, .04, .02],
])
NEEDLE_SPAN = slice(3, 7)
NEEDLE_TOKEN_IDS = (10, 11, 12, 13)
OUTPUT_TOKEN_IDS = (12, 99, 13)


def example_score():
    if not np.allclose(np.nansum(ATTENTION, axis=1), 1, rtol=0, atol=1e-12):
        raise ValueError('Illustrative attention rows must sum to one')
    masses = ATTENTION[:, NEEDLE_SPAN].sum(axis=1)
    eligible = np.array([token in NEEDLE_TOKEN_IDS for token in OUTPUT_TOKEN_IDS])
    return masses, eligible, float(masses[eligible].mean())


def main():
    masses, eligible, score = example_score()
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                         'pdf.fonttype': 42, 'figure.facecolor': 'none', 'axes.facecolor': 'none'})
    fig = plt.figure(figsize=(18, 8.4))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, 1580), ylim=(730, 0))
    ax.set_axis_off()

    def text(x, y, value, size=13, color=INK, ha='left', weight='normal'):
        return ax.text(x, y, value, fontsize=size, color=color, ha=ha, va='center', weight=weight)

    def card(x, y, width, height, color, fill_alpha=.06):
        ax.add_patch(FancyBboxPatch((x, y), width, height,
            boxstyle='round,pad=0,rounding_size=10', facecolor=color, alpha=fill_alpha, linewidth=0))
        ax.add_patch(FancyBboxPatch((x, y), width, height,
            boxstyle='round,pad=0,rounding_size=10', facecolor='none', edgecolor=color,
            linewidth=1.3, alpha=.65))

    def arrow(start, end, color=ACCENT):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=14,
                                    linewidth=1.6, color=color))

    text(28, 33, 'Attention mass', size=24, weight='bold')
    text(28, 72, 'Одна голова h · attention probabilities после softmax', size=15, color=MUTED)
    text(300, 115, 'Ключи: позиции в prompt и уже сгенерированной истории', size=13, color=MUTED)

    x0, y0, cell_w, cell_h = 300, 205, 62, 72
    columns = (r'$c_1$', r'$c_2$', r'$c_3$', r'$n_1$', r'$n_2$', r'$n_3$', r'$n_4$',
               r'$q_1$', r'$q_2$', r'$g_1$', r'$g_2$')
    groups = [(0, 3, 'Контекст', MUTED), (3, 4, 'Needle: span N', ACCENT),
              (7, 2, 'Вопрос', MUTED), (9, 2, 'История', MUTED)]
    for start, width, label, color in groups:
        left, right = x0 + start*cell_w, x0 + (start+width)*cell_w
        ax.plot([left+3, right-3], [155, 155], color=color, linewidth=2)
        text((left+right)/2, 139, label, size=13, color=color, ha='center')
    for idx, label in enumerate(columns):
        text(x0+(idx+.5)*cell_w, 184, label, size=15, ha='center')

    text(28, 167, 'Шаг → выходной токен', size=13, color=MUTED)
    labels = (r'$t=1 \;\to\; n_3$', r'$t=2 \;\to\; u$', r'$t=3 \;\to\; n_4$')
    cmap = plt.get_cmap('magma')
    for row in range(3):
        center = y0+(row+.5)*cell_h
        text(28, center-10, labels[row], size=19, color=INK if eligible[row] else MUTED)
        text(28, center+18, 'токен из needle' if eligible[row] else 'u отсутствует в needle',
             size=12, color=GREEN if eligible[row] else MUTED)
        for col, probability in enumerate(ATTENTION[row]):
            x, y = x0+col*cell_w, y0+row*cell_h
            if np.isnan(probability):
                ax.add_patch(Rectangle((x+1, y+1), cell_w-2, cell_h-2,
                    facecolor=MUTED, alpha=.08, linewidth=0))
                text(x+cell_w/2, y+cell_h/2, '—', size=15, color=MUTED, ha='center')
            else:
                ax.add_patch(Rectangle((x+1, y+1), cell_w-2, cell_h-2,
                    facecolor=cmap(probability), edgecolor='none'))
                text(x+cell_w/2, y+cell_h/2, f'{probability:.2f}'.replace('.', ','),
                     size=14, color='white', ha='center')
        arrow((x0+11*cell_w+10, center), (1040, center), color=ACCENT if eligible[row] else MUTED)
        card(1052, center-29, 475, 58, GREEN if eligible[row] else MUTED)
        label = f'm{row+1} = {masses[row]:.2f}'.replace('.', ',')
        text(1074, center, label, size=20, color=GREEN if eligible[row] else MUTED)
        text(1240, center, 'включаем в среднее' if eligible[row] else 'не учитываем',
             size=14, color=GREEN if eligible[row] else MUTED)

    ax.add_patch(Rectangle((x0+3*cell_w-3, y0-4), 4*cell_w+6, 3*cell_h+8,
                           facecolor='none', edgecolor=ACCENT, linewidth=2.5))
    text(x0+5*cell_w, 450, 'Суммируем ВСЕ вероятности внутри N, не только максимум',
         size=12, color=ACCENT, ha='center')
    text(x0+10*cell_w, 476, '— будущая позиция', size=11, color=MUTED, ha='center')

    card(300, 510, 682, 152, ACCENT)
    text(321, 536, '1. Масса на needle для выбранной строки', size=15, color=ACCENT)
    for idx, row in enumerate((0, 2)):
        terms = ' + '.join(f'{value:.2f}'.replace('.', ',') for value in ATTENTION[row, NEEDLE_SPAN])
        value = f'{masses[row]:.2f}'.replace('.', ',')
        text(321, 582+idx*40, f'm{row+1} = {terms} = {value}', size=17)

    arrow((995, 586), (1040, 586), color=GREEN)
    card(1052, 510, 475, 152, GREEN)
    text(1074, 536, '2. Среднее по подходящим шагам', size=15, color=GREEN)
    text(1074, 586, 'score = (0,65 + 0,75) / 2', size=20)
    text(1074, 627, f'= {score:.2f}'.replace('.', ','), size=25, color=GREEN, weight='bold')

    text(300, 700, r'$S_h = \frac{1}{|T|}\sum_{t\in T}\sum_{j\in N} A^{(h)}_{t,j}$', size=21)
    text(795, 700, r'$T = \{t:\;w_t\in\mathrm{tokens}(N)\}$', size=17)
    text(28, 700, 'Учебный пример', size=12, color=MUTED)

    for ext in ('svg', 'pdf', 'png'):
        fig.savefig(OUT/f'{NAME}.{ext}', dpi=220, bbox_inches='tight', transparent=True)
    plt.close(fig)

    example = dict(kind='illustrative_not_measured', metric='needle_attention_mass_v1',
                   attention=np.where(np.isnan(ATTENTION), 0, ATTENTION).tolist(),
                   unavailable_keys=np.isnan(ATTENTION).tolist(), needle_span=[3, 7],
                   needle_token_ids=NEEDLE_TOKEN_IDS, output_token_ids=OUTPUT_TOKEN_IDS,
                   needle_masses=masses.tolist(), eligible_steps=eligible.tolist(), score=score,
                   axes=dict(rows='decode step predicting output w_t', columns='key positions'),
                   note='Sum across needle keys; mean only across output tokens present in target span.')
    (OUT/f'{NAME}.json').write_text(json.dumps(example, indent=2), encoding='utf-8')
    lines = ['# Схема подсчёта Attention mass', '',
             f'![Attention mass]({NAME}.png)', '',
             f'[SVG]({NAME}.svg) · [PDF]({NAME}.pdf) · [Числа примера]({NAME}.json)', '',
             'Это учебный пример, не измеренные результаты GLM. Фон прозрачный, SVG содержит '
             'векторные клетки, рамки, стрелки и редактируемый текст.', '',
             'Строки — attention на шаге decode, который предсказывает соответствующий выходной '
             'токен w_t; это не утверждение, что сам w_t уже является query этой строки. '
             'Столбцы — позиции ключей: контекст, четыре токена целевого span N, вопрос, история '
             'генерации. Каждая строка вероятностей суммируется в 1. Будущие позиции недоступны '
             'и обозначены тире. g1/g2 — уже сгенерированные токены, не новые части needle.', '',
             'Для каждого шага m_t = сумма A[t,j] по j внутри N. В score включаются только шаги, '
             'где ID предсказанного токена встречается в целевом span: n3 и n4 включаются, '
             'u исключается. Поэтому score = (0.65 + 0.75) / 2 = 0.70, а не среднее трёх строк. '
             'Остальные ключи остаются в знаменателе softmax; нормировки только внутри needle нет. '
             'Максимум внимания и точное копирование одного токена не требуются.', '',
             'Схема соответствует NeedleAttentionMassCollector в retrieval_heads/experiment/scoring.py. '
             'В исходном сравнении N — needle_span эталонного ответа; это не обязательно весь '
             'вставленный case.needle. В full-needle эксперименте меняется целевой диапазон, '
             'но token gate остаётся прежним. QRScore отличается выбором строк (токены вопроса '
             'на prefill), Retrieval Score — подсчётом совпавших argmax-hit.', '',
             'Рейтинг головы в survey затем берётся как среднее её score по выбранным запросам. '
             'Этот дополнительный уровень усреднения не показан в учебной матрице.', '',
             'Воспроизведение из updated: `.venv/bin/python draw_poster_attention_mass_scheme.py`.', '']
    (OUT/'ATTENTION_MASS_SCHEME.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(output=str(OUT/f'{NAME}.svg'), masses=masses.tolist(),
                          eligible_steps=eligible.tolist(), score=score), indent=2))


if __name__ == '__main__':
    main()
