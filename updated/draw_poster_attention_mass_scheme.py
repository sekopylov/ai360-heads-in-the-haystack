"""Compact purple diagram comparing attention mass and multiset retrieval score."""
import json
import numpy as np
from matplotlib.patches import Rectangle
from plot_poster_glm_top40 import OUT, plt

NAME = 'attention_mass_scheme'
INK, MUTED, ACCENT = '#303039', '#77717d', '#713d8f'
TOKENS = ('…', '…', 'Dolores', 'Park', '…', '…', '…', '?')
PROMPT_TOKEN_IDS = (31, 32, 10, 11, 33, 34, 35, 36)
OUTPUT_TOKEN_ID = 11
NEEDLE_SPAN = slice(2, 4)
CURRENT_ATTENTION = np.array([.05, .05, .35, .40, .03, .02, .06, .04])


def example_attention():
    matrix = np.full((8, 8), np.nan)
    for row in range(7):
        matrix[row, :row+1] = 1 / (row+1)
    matrix[-1] = CURRENT_ATTENTION
    assert np.allclose(np.nansum(matrix, axis=1), 1, rtol=0, atol=1e-12)
    assert OUTPUT_TOKEN_ID in PROMPT_TOKEN_IDS[NEEDLE_SPAN]
    maximum_position = int(CURRENT_ATTENTION.argmax())
    assert NEEDLE_SPAN.start <= maximum_position < NEEDLE_SPAN.stop
    assert PROMPT_TOKEN_IDS[maximum_position] == OUTPUT_TOKEN_ID
    return matrix, float(CURRENT_ATTENTION[NEEDLE_SPAN].sum())


def main():
    matrix, mass = example_attention()
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                         'pdf.fonttype': 42, 'figure.facecolor': 'none', 'axes.facecolor': 'none'})
    fig = plt.figure(figsize=(14, 9.2))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, 985), ylim=(647, 0), aspect='equal')
    ax.set_axis_off()

    def text(x, y, value, size=13, color=INK, ha='left'):
        ax.text(x, y, value, fontsize=size, color=color, ha=ha, va='center')

    x0, y0, cell = 24, 128, 58
    width = 8*cell
    bottom = y0+width
    active_y = bottom-cell/2
    left, right = x0+2*cell, x0+4*cell
    text(x0+width/2, 31, 'Attention одной головы', size=20, ha='center')
    text(x0+width/2, 61, 'Позиции токенов в контексте', size=12, color=MUTED, ha='center')
    text((left+right)/2, 85, 'needle', size=14, color=ACCENT, ha='center')
    ax.plot([left+3, right-3], [99, 99], color=ACCENT, linewidth=2)
    for col, token in enumerate(TOKENS):
        text(x0+(col+.5)*cell, 114, token, size=10, ha='center',
             color=ACCENT if 2 <= col < 4 else MUTED)

    cmap = plt.get_cmap('magma')
    for row in range(8):
        for col in range(8):
            selected, in_needle = row == 7, 2 <= col < 4
            available = np.isfinite(matrix[row, col])
            if selected:
                fill = cmap(matrix[row, col])
            elif available and in_needle:
                fill = '#ebe3f2'
            elif available:
                fill = '#eeedf0'
            else:
                fill = 'none'
            x, y = x0+col*cell, y0+row*cell
            ax.add_patch(Rectangle((x+1, y+1), cell-2, cell-2,
                                  facecolor=fill, edgecolor='#d9d5dd', linewidth=.6))
            if selected:
                text(x+cell/2, y+cell/2, f'{matrix[row,col]:.2f}'.replace('.', ','),
                     size=14, color='white', ha='center')
    # The key interval and its probabilities are highlighted without arrows.
    ax.add_patch(Rectangle((left-2, y0-2), right-left+4, width+4,
                           facecolor='none', edgecolor=ACCENT, linewidth=1.8))
    ax.add_patch(Rectangle((left-3, bottom-cell-3), right-left+6, cell+6,
                           facecolor='none', edgecolor=ACCENT, linewidth=2.5))

    # The explanations are separate from the token-row annotation.
    rx = 540
    text(rx, 56, 'Attention mass', size=21, color=ACCENT)
    text(rx, 103, 'Сумма вероятностей внимания\nна позициях needle', size=14)
    text(rx, 158, r'$m_t^{(h)} = \sum_{j\in N} A^{(h)}_{t,j}$', size=21)
    text(rx, 219, r'$S_{\mathrm{mass}}^{(h)} = \frac{1}{|T_N|}\sum_{t\in T_N} m_t^{(h)}$', size=21)
    text(rx, 275, 'N — диапазон needle.\nT_N — шаги, где выходной токен есть в needle.', size=11, color=MUTED)

    ax.plot([rx, 950], [314, 314], color='#d9d5dd', linewidth=1)
    text(rx, 345, 'Retrieval Score · multiset', size=20, color=ACCENT)
    text(rx, 385, 'Argmax — на «Park» внутри needle:\n+1 hit, если квота токена не исчерпана.', size=13)
    text(rx, 452, r'$S_{\mathrm{RS}}^{(h)} = \frac{\sum_v\min(H_h(v),\,C_N(v))}{|N|}$', size=20)
    text(rx, 507, 'H_h(v) — совпавшие argmax-hit.\nC_N(v) — число вхождений токена в needle.', size=11, color=MUTED)

    # This label aligns directly with the selected row, without a connector.
    text(rx, active_y-14, 'На этом шаге сгенерирован', size=14)
    text(rx, active_y+16, 'токен «Park»', size=20, color=ACCENT)
    text(x0+width/2, 624, 'Выделены вероятности, входящие в сумму.', size=12, color=MUTED, ha='center')

    for ext in ('svg', 'pdf', 'png'):
        fig.savefig(OUT/f'{NAME}.{ext}', dpi=220, bbox_inches='tight', transparent=True)
    plt.close(fig)
    example = dict(kind='illustrative_not_measured',
                   metrics=['needle_attention_mass_v1', 'needle_token_multiset_v1'],
                   output_token='Park', output_token_id=OUTPUT_TOKEN_ID,
                   prompt_token_ids=PROMPT_TOKEN_IDS, needle_span=[2, 4],
                   current_attention=CURRENT_ATTENTION.tolist(), step_needle_mass=mass,
                   argmax_position=int(CURRENT_ATTENTION.argmax()),
                   multiset_hit_increment=1, multiset_score_increment=1/2,
                   note='First Park hit: the token quota is available. +1 applies to the hit counter, not final score.',
                   palette='magma in active row; purple and gray elsewhere', square_matrix=True)
    (OUT/f'{NAME}.json').write_text(json.dumps(example, indent=2), encoding='utf-8')
    lines = ['# Матрица внимания: Attention mass и Retrieval Score (multiset)', '',
             f'![Схема]({NAME}.png)', '',
             f'[SVG]({NAME}.svg) · [PDF]({NAME}.pdf) · [Пример]({NAME}.json)', '',
             'Учебный пример, не измеренные результаты GLM. Матрица квадратная, фон прозрачен, '
             'всё в SVG векторное. Текущая строка — в палитре magma из прежних карт голов; '
             'прочие элементы — серые и фиолетовые. Стрелок и широкого пустого поля слева нет.', '',
             'На текущем шаге сгенерирован токен Park. Его ID есть в целевом span Dolores Park. '
             'Attention mass на этом шаге — сумма выделенных вероятностей на всех позициях '
             'needle. Общий score головы — среднее масс по шагам T_N, для которых выходной ID '
             'встречается в needle. N — набор позиций целевого span.', '',
             'У multiset учитывается другое событие: argmax внимания расположен внутри N, '
             'а ID ключевого токена в этой позиции совпадает с выходным ID. В примере '
             'максимум находится на Park, поэтому при ещё доступной квоте прибавляется '
             'один hit. Это не +1 к итоговому score: прирост score равен 1/|N|.', '',
             'Формула multiset: сумма по ID токенов v значений min(H_h(v), C_N(v)), делённая '
             'на число токенов в N. H_h(v) считает все совпавшие argmax-hit для ID v; C_N(v) '
             'задаёт квоту — частоту этого ID в целевом span. Поэтому повторные hit сверх '
             'квоты не увеличивают score. Токеном в формулах является ID, не буквальное слово.', '',
             'Строки матрицы — query-позиции, столбцы — key-позиции. Последняя query-строка '
             'используется при предсказании Park: выходной токен ещё не изображён как query/key. '
             'Верхний треугольник недоступен из-за causal mask. Другие строки схематичные, '
             'с равномерным вниманием по доступным ключам. Строки вероятностей суммируются в единицу.', '',
             'В прежнем сравнении N — needle_span эталонного ответа, не обязательно весь '
             'case.needle. Подписи слов в примере условные: реальная токенизация может '
             'разбивать слова на несколько ID. Усреднение по survey-запросам здесь не показано.', '',
             'Воспроизведение из updated: .venv/bin/python draw_poster_attention_mass_scheme.py', '']
    (OUT/'ATTENTION_MASS_SCHEME.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(output=str(OUT/f'{NAME}.svg'), step_needle_mass=mass,
                          multiset_hit_increment=1), indent=2))


if __name__ == '__main__':
    main()
