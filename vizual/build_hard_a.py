"""
Builds vizual/hard_A.ipynb (axis A of the hard tasks: the question is reworded) from the cells below and runs it.

python vizual/build_hard_a.py            # build and run
python vizual/build_hard_a.py --no-run   # build only
"""
import os
import sys

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# Сложные задачи, ось A: перефразированный вопрос

**Вопрос ноутбука:** сохраняют ли головы, найденные на простой задаче, свою роль, когда вопрос перестаёт дословно повторять иглу.

Игла и стог те же на всех уровнях оси; меняется только вопрос. Вставок нет.

| Часть | Что в ней | Прогоны |
|---|---|---|
| 1 | уровни оси и как проверяется ответ | — |
| 2 | **метрики ответов**: без маски и под масками; те же расчёты без семейства `oslo`; формулировки ответов (перефраз, замены, формы слов); формулировки без `oslo` | все четыре |
| 3 | **метрики голов**: те же графики, что на исходной задаче, по уровням; сохранение рейтинга; контроли | только без маски |

Прогоны: без маски (внимание сохранено, есть метрики голов) и три маски по рейтингу детекции на простой задаче — 40 и 60 лучших голов, 60 случайных. В каждом 3 семейства игл × 4 уровня × 11 длин × 6 глубин = 792 примера.
""")

md("## 0. Настройки")
code("""
import os, sys
sys.path.insert(0, os.path.abspath(".."))   # корень репозитория: ноутбук лежит в vizual/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vizual import data, hard, plots, style

RUNS = "results/new/qwen3_hard_fast/A/*"      # все прогоны оси
NONE = "results/new/qwen3_hard_fast/A/none"   # прогон без маски: в нём метрики голов
DETECT = "results/new/qwen3_detect"           # детекция на простой задаче: рейтинг, по которому строились маски

METRIC = "copy_recall"    # скор головы: copy_recall — как в статье, copy_count — как в коде авторов
THRESHOLD = 0.1           # порог retrieval-головы из статьи
SAVE = False              # сохранять рисунки в vizual/figures/

style.use(scale=1.0, dpi=110)
pd.set_option("display.max_colwidth", 220)

def show(ax, name):
    if SAVE:
        plots.save(np.ravel(ax)[0].figure, name)
    plt.show()

def panels(n, width=5.0, height=4.2, extra=0.0, **kwargs):
    # n панелей в ряд, по одной на уровень
    return plt.subplots(1, n, figsize=style.size(width * n + extra, height), **kwargs)[1]
""")
code("""
sweep = hard.load(RUNS)                       # ответы читаются из outputs/, метрики ответа считаются по ним
levels = sorted(sweep["level"].unique())
families = sorted(sweep["family"].unique())
none = sweep[sweep["run"] == "none"]
print(f"прогонов: {sweep['run'].nunique()}, примеров в каждом: {len(none)}; семейства: {families}; уровни: {levels}")
""")

md("""
## 1. Уровни и как мы их проверяем

### 1.1. Что меняется на каждом уровне

| Уровень | Что сделано с вопросом | Что проверяет | Что нужно модели сверх поиска |
|---|---|---|---|
| **A0** | дословный: повторяет слова иглы | точка отсчёта для всех осей | ничего |
| **A1** | перефразирован, ключ (название города и места) сохранён | чистый перефраз: слова вопроса не совпадают со словами иглы | сопоставить синонимы (`visitor` → `guests`, `asked` → `requested`) |
| **A2** | дословный, но ключ заменён описанием («the Portuguese capital») | поиск без названного ключа | знать, что город — столица страны |
| **A3** | и перефраз, и замена ключа | оба усложнения вместе | и то и другое; эффекты не разделяются |

Чистый эффект перефраза — это A1 против A0; чистый эффект замены ключа — A2 против A0. A3 показывает только их сумму.

Игла, эталон и стог (эссе Пола Грэма) на всех четырёх уровнях одни и те же. Ключ иглы — реальный город, а ответ выдуман, поэтому по памяти ответить нельзя: верный ответ возможен только из текста.
""")
code("""
# иглы, эталоны и факты, по которым проверяется ответ
first = none.drop_duplicates("family").set_index("family")
needles = {name: data.read_jsonl(data.path(f"data/needles_hard/{name}/axis_A.jsonl"))[0]["needle"] for name in families}
display(pd.DataFrame({"игла": needles, "эталон (ответ внутри иглы)": first["reference"],
                      "факты ответа (допустимые написания)": first["answer_key"].map(lambda key: "; ".join(" / ".join(fact) for fact in key))}))

# вопросы по уровням
display(none.drop_duplicates(["family", "level"]).pivot(index="level", columns="family", values="question"))
""")

md("""
### 1.2. Как проверяется ответ

Ответ считается верным не по совпадению строки с эталоном, а по **фактам**: в ответе должны быть названы оба ключевых факта иглы (таблица выше). Правило — `fact_class` из `rh/metrics.py`.

1. Ответ и факты приводятся к одному виду: нижний регистр, без пунктуации, Unicode NFKC.
2. Факт засчитан, если любое из его допустимых написаний стоит в ответе целым словом или фразой («guest book» или «guestbook»).
3. По числу найденных фактов ответ получает класс:

| Класс | Условие | Что это значит на оси A |
|---|---|---|
| **верно** | названы оба факта | модель нашла иглу и передала ответ, в любых словах |
| **частично** | назван один факт из двух | ответ найден, но оборван: «ring the brass bell» без гостевой книги |
| **иное** | ни одного факта | ответ не из иглы: отказ, рассуждение о тексте или ответ по памяти (на `oslo` — «Oslo Cathedral») |

Остальные классы правила (смешано, путаница, шум) требуют фактов вставок. На оси A вставок нет, поэтому эти классы здесь невозможны.

**Чего проверка не ловит.** Факт засчитывается только в написании из списка: «ring the bell» вместо «ring the brass bell» попадёт в «частично». И наоборот, названные факты не гарантируют, что остальная часть ответа верна.

### 1.3. Как проверяются головы

- Метрики голов считаются только в прогоне без маски: под маской внимание не сохранялось.
- В скор головы на уровне входят только примеры с **верным** ответом (по фактам). В статье условием был ROUGE-1 > 50; здесь он заменён, потому что на перефразе ROUGE занижен при верном ответе (раздел 2.1).
- Рейтинг голов **не переопределяется**: он взят из детекции на простой задаче. На каждом уровне смотрим, что с теми же головами.
- Уровни сравниваются на одних и тех же местах иглы (семейство, длина, глубина) — тех, где ответ верен на всех четырёх уровнях. Иначе разница между уровнями смешалась бы с разницей в наборах примеров.
""")

# ------------------------------------------------------------------------------------------------------- answers
md("""
# 2. Метрики ответов

| Метрика | Годится? | Почему |
|---|---|---|
| класс ответа по фактам | да, основная | не зависит от формулировки ответа |
| доля названных фактов | да | та же проверка без порога «оба факта» |
| ROUGE-1/2/3 recall | только как вспомогательная | эталон на всех уровнях один, близнецов нет, так что запрет из README (ответ с близнеца проходит порог) здесь не действует; но на A1 и A3 модель пересказывает ответ словами вопроса, и recall падает при верном ответе |
| точная фраза | нет | по той же причине: верный ответ «prepared with smoked lemon peel» вместо «cooked with…» не проходит |

## 2.1. Без маски

Насколько сама задача труднее без вмешательства в модель.
""")
code("""
print("верные ответы без маски, %")
display(hard.by_level(none, by=("family", "level")))
show(plots.stacked(none, "level", "answer_class", hard.CLASSES, hard.CLASS_LABELS, xlabel="уровень", title="Без маски: классы ответов по уровням"), "A_none_classes")
""")

md("""
### Почему ROUGE и точная фраза здесь вспомогательные

Только верные ответы без маски. Класс по фактам говорит «верно», а строковые метрики падают там, где модель пересказала ответ словами вопроса.
""")
code("""
right = none[none["correct"]]
check = right.groupby(["family", "level"]).agg(верных=("id", "size"), точная_фраза=("exact", "mean"), rouge1=("rouge1_recall", "mean"),
                                               rouge2=("rouge2_recall", "mean"), rouge3=("rouge3_recall", "mean"))
check["точная_фраза"] *= 100
display(check.round(1))

# верные ответы без точной фразы и неверные: самые короткие на семейство и уровень
reworded = right[~right["exact"]].sort_values("n_tokens").groupby(["family", "level"]).head(1)
wrong = none[~none["correct"]].sort_values("n_tokens").groupby(["family", "level"]).head(2)
display(pd.concat([reworded, wrong])[["family", "level", "answer_class", "rouge1_recall", "reference", "response"]].round(0))
""")

md("""
## 2.2. Под масками

Головы отключаются по рейтингу детекции на простой задаче. Если роль голов сохраняется, маска лучших голов должна ломать ответы на перефразированном вопросе так же, как на дословном, а маска случайных — не ломать.
""")
code("""
print("верные ответы, %")
display(hard.by_level(sweep))
show(plots.level_lines(sweep, "correct", title="Верные ответы по уровням"), "A_correct_by_level")
show(plots.class_panels(sweep, hard.CLASSES, hard.CLASS_LABELS, title="Классы ответов по уровням"), "A_classes_by_run")

# те же верные ответы в разрезе длины контекста и глубины иглы: панель на прогон, линия на уровень
show(plots.level_gradation(sweep, "context_length", title="Верные ответы по длине контекста"), "A_correct_by_length")
show(plots.level_gradation(sweep, "depth_percent", title="Верные ответы по глубине иглы"), "A_correct_by_depth")
""")

md("""
### Классы ответов по длине контекста и глубине иглы

Те же классы, что выше, в разрезе сетки прогона. Столбец панелей — прогон, строка — уровень; верхняя строка — все уровни вместе. Столбики стоят на настоящих значениях сетки; сетка неравномерна на краю (длины 28.5k и 30k, глубины 89 и 100 стоят ближе друг к другу), поэтому последние два столбика уже остальных. В одном столбике по уровню 18 примеров на длину и 33 на глубину (три семейства).
""")
code("""
show(plots.class_gradation(sweep, "context_length", hard.CLASSES, hard.CLASS_LABELS, title="Классы ответов по длине контекста"), "A_classes_by_length")
show(plots.class_gradation(sweep, "depth_percent", hard.CLASSES, hard.CLASS_LABELS, title="Классы ответов по глубине иглы"), "A_classes_by_depth")
""")
code("""
# то же числом: доля каждого класса, %, все уровни вместе
for column, name in (("context_length", "длина контекста"), ("depth_percent", "глубина иглы")):
    print(name)
    table = (pd.crosstab([sweep["run"], sweep[column]], sweep["answer_class"], normalize="index") * 100).reindex(
        columns=[c for c in hard.CLASSES if c in set(sweep["answer_class"])]).rename(columns=hard.CLASS_LABELS).round(1)
    display(table.unstack("run").swaplevel(axis=1).sort_index(axis=1, level=0, sort_remaining=False))
""")

md("""
### На тех же примерах

Уровень A3 труднее уже без маски, поэтому доля верных ответов под маской смешивает два эффекта. Здесь берутся только примеры, на которые модель без маски ответила верно, и считается, какая их доля осталась верной под маской. Строка `n` — число таких примеров.
""")
code("""
kept = hard.paired(sweep)
display(kept)

base = kept.loc[kept.index != "n", levels[0]]
print(f"разница с {levels[0]}, процентные пункты: плюс — уровень устойчивее к маске, минус — уязвимее")
display(kept.loc[kept.index != "n"].sub(base, axis=0).round(1))
""")

md("""
### По семействам

Шаблоны игл разной сложности: сравнивать уровни нужно внутри семейства и смотреть, совпадает ли направление эффекта.
""")
code("""
display(hard.by_level(sweep, by=("run", "family", "level")))

axes = panels(len(families), 5.2, extra=2.6, sharey=True)
for ax, family in zip(axes, families):
    plots.level_lines(sweep[sweep["family"] == family], "correct", ax=ax, title=family)
    if ax is not axes[-1]:
        ax.get_legend().remove()
    if ax is not axes[0]:
        ax.set_ylabel("")
show(axes, "A_correct_by_family")
""")

md("""
### Вспомогательные метрики и длина ответа

ROUGE здесь не критерий правильности, но показывает, насколько ответ остаётся списанным из иглы.
""")
code("""
axes = panels(3, 6.0)
for ax, (value, name) in zip(axes, {"rouge1_recall": "ROUGE-1 recall", "rouge2_recall": "ROUGE-2 recall", "n_tokens": "Длина ответа"}.items()):
    plots.level_lines(sweep, value, ax=ax, title=name)
    if ax is not axes[-1]:
        ax.get_legend().remove()
show(axes, "A_aux_by_level")

display(sweep.groupby(["run", "level"]).agg(верно=("correct", "mean"), фактов=("facts_found", "mean"), rouge1=("rouge1_recall", "mean"),
                                           rouge2=("rouge2_recall", "mean"), rouge3=("rouge3_recall", "mean"), токенов=("n_tokens", "mean")).round(2))
""")

md("""
### Где ломается: длина и глубина

Доля верных ответов по длине контекста и глубине иглы, панель на уровень. В ячейке три примера (по одному на семейство).
""")
code("""
for run in [r for r in ("none", "top40", "top60", "random60") if r in set(sweep["run"])]:
    part = sweep[sweep["run"] == run]
    name = plots.run_styles(sweep)[run][1]
    show(plots.niah_grid(part.assign(run=part["level"]), {level: level for level in levels}, value="correct", columns=4,
                         title=f"Верные ответы: {name}"), f"A_niah_{run}")
""")

md("""
### Примеры ответов под маской

По два самых коротких ответа каждого класса на каждом уровне для выбранного прогона.
""")
code("""
RUN = "top60"
part = sweep[sweep["run"] == RUN].sort_values("n_tokens").groupby(["level", "answer_class"]).head(2)
part.assign(класс=part["answer_class"].map(hard.CLASS_LABELS)).sort_values(["level", "answer_class"])[["family", "level", "класс", "reference", "response"]]
""")

md("""
## 2.3. Те же расчёты без семейства `oslo`

Семейство `oslo` ведёт себя иначе, чем два других: уже без маски на A3 верных ответов 62% (модель отвечает по памяти про Oslo Cathedral), и почти вся потеря под случайной маской приходится на него. Чтобы отделить эффект оси от свойства одной иглы, все расчёты разделов 2.1–2.2 повторены на двух оставшихся семействах: 2 семейства × 4 уровня × 11 длин × 6 глубин = 528 примеров на прогон.
""")
code("""
EXCLUDE = ["oslo"]                                        # семейства, которые убираем
rest = sweep[~sweep["family"].isin(EXCLUDE)]
rest_none = rest[rest["run"] == "none"]
print(f"семейства: {sorted(rest['family'].unique())}; примеров в прогоне: {len(rest_none)}")

print("без маски: верные ответы, %")
display(hard.by_level(rest_none, by=("family", "level")))
show(plots.stacked(rest_none, "level", "answer_class", hard.CLASSES, hard.CLASS_LABELS, xlabel="уровень",
                   title="Без маски и без oslo: классы ответов по уровням"), "A_rest_none_classes")
""")
code("""
print("верные ответы, %: без oslo")
display(hard.by_level(rest))
print("для сравнения — все три семейства")
display(hard.by_level(sweep))

axes = panels(2, 5.6, extra=2.6, sharey=True)
plots.level_lines(sweep, "correct", ax=axes[0], title="Все семейства")
plots.level_lines(rest, "correct", ax=axes[1], title="Без oslo")
axes[0].get_legend().remove()
axes[1].set_ylabel("")
show(axes, "A_rest_correct_by_level")

show(plots.class_panels(rest, hard.CLASSES, hard.CLASS_LABELS, title="Классы ответов по уровням, без oslo"), "A_rest_classes_by_run")

# те же верные ответы в разрезе длины контекста и глубины иглы: панель на прогон, линия на уровень
show(plots.level_gradation(rest, "context_length", title="Верные ответы по длине контекста, без oslo"), "A_rest_correct_by_length")
show(plots.level_gradation(rest, "depth_percent", title="Верные ответы по глубине иглы, без oslo"), "A_rest_correct_by_depth")

show(plots.class_gradation(rest, "context_length", hard.CLASSES, hard.CLASS_LABELS, title="Классы ответов по длине контекста, без oslo"), "A_rest_classes_by_length")
show(plots.class_gradation(rest, "depth_percent", hard.CLASSES, hard.CLASS_LABELS, title="Классы ответов по глубине иглы, без oslo"), "A_rest_classes_by_depth")
""")

md("""
### На тех же примерах, без `oslo`

Доля примеров, верных без маски, которые остались верными под маской.
""")
code("""
kept_rest = hard.paired(rest)
display(kept_rest)

base = kept_rest.loc[kept_rest.index != "n", levels[0]]
print(f"разница с {levels[0]}, процентные пункты: плюс — уровень устойчивее к маске, минус — уязвимее")
display(kept_rest.loc[kept_rest.index != "n"].sub(base, axis=0).round(1))

print("та же разница по всем трём семействам")
display(kept.loc[kept.index != "n"].sub(kept.loc[kept.index != "n", levels[0]], axis=0).round(1))
""")

md("### Вспомогательные метрики и длина ответа, без `oslo`")
code("""
axes = panels(3, 6.0)
for ax, (value, name) in zip(axes, {"rouge1_recall": "ROUGE-1 recall", "rouge2_recall": "ROUGE-2 recall", "n_tokens": "Длина ответа"}.items()):
    plots.level_lines(rest, value, ax=ax, title=name)
    if ax is not axes[-1]:
        ax.get_legend().remove()
show(axes, "A_rest_aux_by_level")

display(rest.groupby(["run", "level"]).agg(верно=("correct", "mean"), фактов=("facts_found", "mean"), rouge1=("rouge1_recall", "mean"),
                                          rouge2=("rouge2_recall", "mean"), rouge3=("rouge3_recall", "mean"), токенов=("n_tokens", "mean")).round(2))
""")

md("""
### Где ломается: длина и глубина, без `oslo`

В ячейке два примера (по одному на семейство).
""")
code("""
for run in [r for r in ("none", "top40", "top60", "random60") if r in set(rest["run"])]:
    part = rest[rest["run"] == run]
    name = plots.run_styles(rest)[run][1]
    show(plots.niah_grid(part.assign(run=part["level"]), {level: level for level in levels}, value="correct", columns=4,
                         title=f"Верные ответы без oslo: {name}"), f"A_rest_niah_{run}")
""")

md("""
## 2.4. Формулировки ответов: перефраз, замены, формы слов

Класс по фактам говорит, названы ли ключевые факты. Здесь тот же ответ разбирается подробнее: эталон разбит на части, и для каждой части смотрим, как она записана — дословно, другой формой тех же слов, близкой заменой или никак. Это тот же разбор, что в ноутбуке по задаче статьи, с теми же уровнями и структурами.

| Уровень формулировки | Что это | Пример |
|---|---|---|
| дословно | формулировка эталона | `cooked with` |
| другая форма слова | те же слова в другой форме, с другим артиклем или в другом порядке | `cooking it with`; `a brass bell`; `black clay bowl` |
| синоним, смысл сохранён | другие слова с тем же смыслом | `prepared using`; `when they arrive` вместо `at the entrance` |
| потеряна деталь | часть названа, но без существенной детали | `ring a bell` без `brass`; `smoked lemon` без `peel`; `red roof` без `slate` |
| нет | части в ответе нет | — |

Часть считается сохранённой на первых трёх уровнях. Списки формулировок лежат в `vizual/paraphrase.py` (`FACTS`) и собраны вручную по ответам этих прогонов: это перечень того, что модель действительно писала, а не словарь синонимов. Отнесение формулировки к уровню — моё решение, оно видно в таблице ниже.

| Структура ответа | Условие |
|---|---|
| точная фраза | эталон стоит в ответе дословно и подряд |
| все части дословно, другой порядок или вставка | все части дословно, но фраза не совпадает |
| все части, есть форма слова или синоним | все части сохранены, хотя бы одна не дословно — **перефразированный полный ответ** |
| не все части, остальные дословно | часть пропала, оставшиеся дословно — обрезанный ответ |
| не все части, есть форма, синоним или потеря детали | часть пропала или потеряла деталь |
| ни одной части | ответ не из иглы |
""")
code("""
from vizual import paraphrase

worded = paraphrase.annotate(sweep)
worded_none = worded[worded["run"] == "none"]
fact_names = {family: list(paraphrase.FACTS[worded.loc[worded["family"] == family, "reference"].iloc[0]]) for family in families}

# что именно написано вместо формулировки эталона: по всем прогонам
pd.set_option("display.max_rows", 200)
paraphrase.wordings(worded, top=5).set_index(["эталон", "факт", "уровень"])
""")

md("""
### Без маски: как модель формулирует ответ на каждом уровне

Перефраз вопроса (A1, A3) меняет формулировку ответа и без вмешательства в модель.
""")
code("""
ax = plt.subplots(figsize=style.size(11.5, 4.2))[1]
show(plots.stacked(worded_none, "level", "structure", paraphrase.STRUCTURES, paraphrase.STRUCTURE_LABELS, xlabel="уровень", ax=ax,
                   title="Без маски: структура ответа по уровням"), "A_structure_none")

print("без маски: доля ответов каждой структуры, %")
display((pd.crosstab([worded_none["family"], worded_none["level"]], worded_none["structure"], normalize="index") * 100).reindex(
    columns=[k for k in paraphrase.STRUCTURES if k in set(worded_none["structure"])]).rename(columns=paraphrase.STRUCTURE_LABELS).round(1))
""")
code("""
# как записана каждая часть эталона, без маски: панель на часть, ряд на семейство
for family in families:
    part = worded_none[worded_none["family"] == family]
    names = fact_names[family]
    axes = panels(len(names), 4.2, extra=2.4, sharey=True)
    for ax, fact in zip(axes, names):
        plots.stacked(part, "level", fact, paraphrase.LEVELS, paraphrase.LEVEL_LABELS, xlabel="уровень", legend=ax is axes[-1], ax=ax, title=f"«{fact}»")
        if ax is not axes[0]:
            ax.set_ylabel("")
    axes[0].figure.suptitle(f"{family}: формулировка частей ответа без маски", x=0.01, ha="left", fontweight="bold")
    show(axes, f"A_fact_wordings_{family}")
""")

md("""
### Как строковые метрики и класс по фактам оценивают каждую структуру

По всем прогонам оси. `верно` — доля ответов класса «верно» по фактам; `частей` — сколько частей эталона сохранено (дословно, формой слова или синонимом; у `oslo` частей три, у остальных четыре).

Разбор по частям строже класса по фактам: класс требует двух ключевых фактов, а здесь проверяется весь эталон. Поэтому ответ без `at the entrance` или `with a green pen` верен по фактам, но относится к «не все части».
""")
code("""
by_structure = worded.groupby("structure").agg(ответов=("id", "size"), частей=("kept_facts", "mean"), верно=("correct", "mean"), точная_фраза=("exact", "mean"),
                                               rouge1=("rouge1_recall", "mean"), rouge2=("rouge2_recall", "mean"), rouge3=("rouge3_recall", "mean"),
                                               успех1=("success", "mean"), успех2=("success2", "mean"), успех3=("success3", "mean")).reindex(paraphrase.STRUCTURES)
by_structure[["верно", "точная_фраза", "успех1", "успех2", "успех3"]] *= 100
display(by_structure.round(1).rename(index=paraphrase.STRUCTURE_LABELS))

print("структура против класса по фактам, число ответов")
display(pd.crosstab(worded["structure"], worded["answer_class"]).reindex(paraphrase.STRUCTURES).rename(index=paraphrase.STRUCTURE_LABELS, columns=hard.CLASS_LABELS))
""")

md("### Под масками")
code("""
show(plots.class_panels(worded, paraphrase.STRUCTURES, paraphrase.STRUCTURE_LABELS, column="structure", title="Структура ответа по уровням"), "A_structure_by_run")

full = worded["structure"].isin(["exact", "reordered", "reworded"])
print("ответы со всеми частями эталона (дословно, формой слова или синонимом), %")
display(hard.by_level(worded.assign(full=full), "full"))
print("из них перефразированные (не точная фраза), %")
display(hard.by_level(worded[full].assign(reworded=lambda d: d["structure"] != "exact"), "reworded"))
""")
code("""
# примеры: по два самых коротких ответа каждой структуры
examples = worded[~worded["exact"]].sort_values("n_tokens").groupby("structure").head(2)
examples.assign(структура=examples["structure"].map(paraphrase.STRUCTURE_LABELS)).set_index("структура").loc[
    [paraphrase.STRUCTURE_LABELS[k] for k in paraphrase.STRUCTURES if k in set(examples["structure"])],
    ["run", "family", "level", "answer_class", "rouge1_recall", "response"]].round(0)
""")

md("""
## 2.5. Формулировки ответов без семейства `oslo`

Тот же разбор, что в разделе 2.4, на семействах `cairo` и `lisbon`. У `oslo` на A3 треть ответов без маски — не из иглы, и это смешивается с изменением формулировки; здесь остаются только семейства, у которых без маски все ответы верны.
""")
code("""
worded_rest = worded[~worded["family"].isin(EXCLUDE)]
worded_rest_none = worded_rest[worded_rest["run"] == "none"]
print(f"семейства: {sorted(worded_rest['family'].unique())}; примеров в прогоне: {len(worded_rest_none)}")

ax = plt.subplots(figsize=style.size(11.5, 4.2))[1]
show(plots.stacked(worded_rest_none, "level", "structure", paraphrase.STRUCTURES, paraphrase.STRUCTURE_LABELS, xlabel="уровень", ax=ax,
                   title="Без маски и без oslo: структура ответа по уровням"), "A_rest_structure_none")

print("без маски: доля ответов каждой структуры, %")
display((pd.crosstab([worded_rest_none["family"], worded_rest_none["level"]], worded_rest_none["structure"], normalize="index") * 100).reindex(
    columns=[k for k in paraphrase.STRUCTURES if k in set(worded_rest_none["structure"])]).rename(columns=paraphrase.STRUCTURE_LABELS).round(1))
""")

md("### Как метрики оценивают каждую структуру, без `oslo`")
code("""
by_structure_rest = worded_rest.groupby("structure").agg(ответов=("id", "size"), частей=("kept_facts", "mean"), верно=("correct", "mean"), точная_фраза=("exact", "mean"),
                                                         rouge1=("rouge1_recall", "mean"), rouge2=("rouge2_recall", "mean"), rouge3=("rouge3_recall", "mean"),
                                                         успех1=("success", "mean"), успех2=("success2", "mean"), успех3=("success3", "mean")).reindex(paraphrase.STRUCTURES)
by_structure_rest[["верно", "точная_фраза", "успех1", "успех2", "успех3"]] *= 100
display(by_structure_rest.round(1).rename(index=paraphrase.STRUCTURE_LABELS))

print("структура против класса по фактам, число ответов")
display(pd.crosstab(worded_rest["structure"], worded_rest["answer_class"]).reindex(paraphrase.STRUCTURES).rename(index=paraphrase.STRUCTURE_LABELS, columns=hard.CLASS_LABELS))
""")

md("### Под масками, без `oslo`")
code("""
show(plots.class_panels(worded_rest, paraphrase.STRUCTURES, paraphrase.STRUCTURE_LABELS, column="structure", title="Структура ответа по уровням, без oslo"), "A_rest_structure_by_run")

full_rest = worded_rest["structure"].isin(["exact", "reordered", "reworded"])
print("ответы со всеми частями эталона (дословно, формой слова или синонимом), %: без oslo")
display(hard.by_level(worded_rest.assign(full=full_rest), "full"))
print("то же по всем трём семействам")
display(hard.by_level(worded.assign(full=full), "full"))
print("из полных ответов перефразированные (не точная фраза), %: без oslo")
display(hard.by_level(worded_rest[full_rest].assign(reworded=lambda d: d["structure"] != "exact"), "reworded"))
""")

# --------------------------------------------------------------------------------------------------------- heads
md("""
# 3. Метрики голов

Только прогон без маски. Скоры считаются по верно отвеченным примерам.

| Метрика | Годится? | Почему |
|---|---|---|
| `copy_recall`, `copy_count` — скор копирования из статьи | да, с контролем | игла не менялась, ответ по-прежнему списывается из неё; но скор зависит от того, насколько дословно модель отвечает (раздел 3.4) |
| частота активации | да | доля примеров, где голова скопировала хотя бы один токен иглы |
| `needle_mass` — масса внимания на ответе в игле | да | не требует дословного ответа: меряет, куда голова смотрит, а не что копирует |
| `needle_sentence_mass` — масса на всей игле | да | то же для всего предложения иглы, включая ключ («Lisbon Tram Museum»), который на A2 и A3 в вопросе не назван |
| `needle_attention_mass` | да | масса на ответе на тех шагах, где модель выдаёт токен из ответа |
| `competing_mass`, `competing_max_mass`, `copy_competing`, `supporting_mass` | нет | это внимание на вставки, а на оси A вставок нет: значения тождественно равны нулю |
| `copy_sentence` | дублирует `copy_count` | копирование из всей иглы; отличается только токенами вне ответа |
""")
code("""
heads = data.Heads(NONE, samples=none, ok="correct")      # в скор входят верные ответы
detect = data.Heads(DETECT)                               # детекция на простой задаче (условие — успех по ROUGE-1, как в статье)
reference = detect.score(METRIC)                          # скор детекции, [слой, голова]
order = np.argsort(-reference.ravel(), kind="stable")     # головы от лучшей к худшей по детекции
total_heads = reference.size

common = hard.common_cells(heads.samples)                 # места иглы, отвеченные верно на всех уровнях
print(f"примеров с метриками голов: {len(heads.samples)}; на местах, верных на всех уровнях: {common.sum()} ({common.sum() // len(levels)} на уровень)")

scores = hard.level_scores(heads, METRIC, where=common)                                              # {уровень: [слой, голова]}
rankings = {level: heads.ranking(METRIC, hard.level_mask(heads, level, common)) for level in levels}   # рейтинг голов самого уровня
""")

md("""
## 3.1. Те же графики, что на исходной задаче

Каждый график из ноутбука по задаче статьи, построенный отдельно для каждого уровня. Здесь рейтинг считается заново на самом уровне — чтобы увидеть, как выглядела бы детекция, проведённая на перефразированном вопросе.

### Распределение голов по скору (рис. 3 статьи)
""")
code("""
axes = panels(len(levels), 5.0, sharey=True)
for ax, level in zip(axes, levels):
    plots.score_distribution(scores[level], ax=ax, title=level)
    if ax is not axes[0]:
        ax.set_ylabel("")
show(axes, "A_score_distribution")

axes = panels(len(levels), 5.6, 3.6)
for ax, level in zip(axes, levels):
    plots.score_pie(scores[level], ax=ax, title=level)
show(axes, "A_score_pie")

share = lambda s: {"ровно 0": (s == 0).mean() * 100, "0 – 0.1": ((s > 0) & (s <= 0.1)).mean() * 100, "0.1 – 0.5": ((s > 0.1) & (s <= 0.5)).mean() * 100,
                   "выше 0.5": (s > 0.5).mean() * 100, f"голов выше {THRESHOLD}": float((s > THRESHOLD).sum())}
print("доля голов, %")
display(pd.DataFrame({"детекция": share(reference), **{level: share(scores[level]) for level in levels}}).T.round(1))
""")

md("""
### Скор и частота активации (рис. 4 статьи)

Головы упорядочены по скору самого уровня. Частота активации — доля примеров уровня, в которых голова скопировала хотя бы один токен иглы.
""")
code("""
axes = panels(len(levels), 5.4, extra=2.2, sharey=True)
for ax, level in zip(axes, levels):
    plots.score_vs_frequency(rankings[level], top=100, threshold=THRESHOLD, ax=ax, title=level)
    ax.set_xlabel("место головы по скору уровня")
    if ax is not axes[-1]:
        ax.get_legend().remove()
    if ax is not axes[0]:
        ax.set_ylabel("")
show(axes, "A_score_vs_frequency")

print("голов, активных во всех примерах уровня:", {level: int((rankings[level]["frequency"] == 1).sum()) for level in levels})
""")

md("""
### Скор каждой головы (рис. 5 статьи)

Одна шкала цвета на всех панелях.
""")
code("""
vmax = max(float(s.max()) for s in scores.values())
fig, axes = plt.subplots(2, 2, figsize=style.size(17, 9.2))
for ax, level in zip(axes.ravel(), levels):
    plots.head_heatmap(scores[level], heads.layers, vmax=vmax, ax=ax, title=level)
show(axes, "A_head_heatmap")
""")

md("""
### 20 лучших голов уровня

Рейтинг самого уровня. Сравнение с рейтингом детекции — в разделе 3.2.
""")
code("""
axes = panels(len(levels), 4.6, 7.2)
for ax, level in zip(axes, levels):
    plots.top_heads(rankings[level], k=20, threshold=THRESHOLD, ax=ax, title=level)
    if ax is not axes[0]:
        ax.set_ylabel("")
show(axes, "A_top_heads")
""")

md("""
## 3.2. Сохраняется ли рейтинг детекции

Порядок голов на каждом уровне против порядка детекции на простой задаче: корреляция Спирмена по всем головам и число общих голов среди 20, 40 и 60 лучших. Строка A0 — это перенос на другие иглы без усложнения; остальные уровни сравниваются с ней.
""")
code("""
display(hard.ranking_table(scores, reference))

print("то же против A0 (те же иглы, меняется только вопрос)")
display(hard.ranking_table({level: scores[level] for level in levels[1:]}, scores[levels[0]]))
""")

md("""
### Корреляция скоров: исходные головы и наша задача

Точка — голова: по горизонтали её скор в детекции на простой задаче, по вертикали — на уровне оси A. На диагонали скор сохранён. Красные точки — головы, которые детекция относит к retrieval-головам (скор выше порога).

В углу и в таблице ниже: r — корреляция Пирсона (совпадают ли сами значения), ρ — Спирмена (совпадает ли порядок). По всем головам корреляцию во многом задают сотни голов со скором около нуля, поэтому отдельно посчитано по исходным retrieval-головам: по тем, что выше порога, и по 20, 40 и 60 лучшим.
""")
code("""
axes = panels(len(levels), 5.0, 5.2, sharey=True)
for ax, level in zip(axes, levels):
    plots.score_scatter(reference, scores[level], threshold=THRESHOLD, ax=ax, title=level)
    if ax is not axes[0]:
        ax.set_ylabel("")
        ax.get_legend().remove()
show(axes, "A_score_scatter")

display(hard.correlation_table(scores, reference, order, threshold=THRESHOLD).unstack("условие").swaplevel(axis=1).sort_index(axis=1, level=0, sort_remaining=False))
""")
code("""
# между всеми условиями сразу: корреляция Спирмена по головам детекции выше порога
found = reference.ravel() > THRESHOLD
show(plots.correlation({"детекция": reference.ravel()[found], **{level: scores[level].ravel()[found] for level in levels}},
                       title=f"Порядок {int(found.sum())} исходных retrieval-голов"), "A_score_correlation")
""")

md("""
## 3.3. Значения голов детекции по уровням

Головы стоят в порядке детекции и не переупорядочиваются. Если роль сохраняется, линии уровней идут вместе.
""")
code("""
show(plots.rank_profile(scores, order, top=60, reference=reference, title="Скор голов детекции на каждом уровне"), "A_rank_profile")

print("средний скор голов, которые отключаются масками")
display(hard.top_means({"детекция": reference, **scores}, order))

top = 30
table = pd.DataFrame({"детекция": reference.ravel()[order[:top]], **{level: scores[level].ravel()[order[:top]] for level in levels}},
                     index=[heads.name(i) for i in order[:top]])
show(plots.head_table(table, title=f"{top} лучших голов детекции по уровням"), "A_head_table")
""")

md("""
## 3.4. Контроль: дословные ответы

Скор копирования засчитывает шаг, только если голова смотрит на токен иглы и модель выдаёт именно его. Если модель пересказывает ответ, скопированных токенов меньше, и скор падает без всякой смены голов. Поэтому тот же расчёт повторён только на ответах с точной фразой: там число скопированных токенов одинаково на всех уровнях.
""")
code("""
exact = heads.samples["exact"].to_numpy()
print("ответов с точной фразой:", {level: int((exact & heads.samples["correct"] & (heads.samples["level"] == level)).sum()) for level in levels})
exact_scores = hard.level_scores(heads, METRIC, where=exact)

axes = panels(2, 6.6, extra=1.6, sharey=True)
plots.rank_profile(scores, order, top=60, legend=False, ax=axes[0], title="Все верные ответы")
plots.rank_profile(exact_scores, order, top=60, ax=axes[1], title="Только дословные ответы")
axes[1].set_ylabel("")
show(axes, "A_exact_control")

print("средний скор лучших голов детекции: все верные ответы")
display(hard.top_means(scores, order))
print("только дословные ответы")
display(hard.top_means(exact_scores, order))
""")

md("""
## 3.5. Куда головы смотрят: масса внимания

Масса не зависит от формулировки ответа. Головы стоят в порядке детекции (по копированию).
""")
code("""
mass_metrics = {"needle_mass": "Масса на ответе в игле", "needle_sentence_mass": "Масса на всей игле", "needle_attention_mass": "Масса на ответе на шагах копирования"}
masses = {metric: hard.level_scores(heads, metric, where=common) for metric in mass_metrics}

axes = panels(3, 6.0, extra=1.4, sharey=True)
for ax, (metric, name) in zip(axes, mass_metrics.items()):
    plots.rank_profile(masses[metric], order, top=60, ylabel="масса внимания", legend=ax is axes[-1], ax=ax, title=name)
    if ax is not axes[0]:
        ax.set_ylabel("")
show(axes, "A_mass_profile")

for metric, name in mass_metrics.items():
    print(f"{name} ({metric}): среднее у лучших голов детекции")
    display(hard.top_means(masses[metric], order))
""")

md("""
## 3.6. По семействам

Совпадает ли направление эффекта в каждом семействе: средний скор 20 лучших голов детекции по уровням.
""")
code("""
family_top = pd.DataFrame({family: hard.top_means(hard.level_scores(heads, METRIC, where=common & (heads.samples["family"] == family).to_numpy()), order).loc["20 лучших"]
                           for family in families}).T
display(family_top)
show(plots.group_bars(family_top, xlabel="семейство игл", ylabel="средний скор 20 лучших голов", title="20 лучших голов детекции: скор по семействам"), "A_family_top20")
""")

md("""
## 3.7. Появляются ли новые головы

Головы, которые на уровне входят в 20 лучших, но в детекции стояли за пределами первых 60.
""")
code("""
place = {int(i): rank + 1 for rank, i in enumerate(order)}
rows = []
for level in levels:
    for i in np.argsort(-scores[level].ravel(), kind="stable")[:20]:
        if place[int(i)] > 60:
            rows.append({"уровень": level, "голова": heads.name(i), "место в детекции": place[int(i)], "скор детекции": round(float(reference.ravel()[i]), 3),
                         **{l: round(float(scores[l].ravel()[i]), 3) for l in levels}})
pd.DataFrame(rows) if rows else print("таких голов нет")
""")

md("""
# 4. Выводы

Написаны по результатам этого ноутбука на 2026-10-12; числа взяты из таблиц выше. При пересчёте на других прогонах текст нужно сверить заново — он не обновляется сам.

**На чём стоят выводы.** Одна модель (Qwen3-8B), три семейства игл, 792 примера на прогон, по одному набору случайных голов. В точке «уровень × длина × глубина» три примера, поэтому надёжны только агрегаты: по уровню (198 примеров), по глубине (132–165), по длине (72–90). Различия в несколько процентных пунктов между отдельными уровнями или точками сетки — в пределах шума.

## 4.1. Главное

1. **Головы, найденные на простой задаче, сохраняют роль на перефразированном вопросе.** Рейтинг относительно детекции: Спирмен 0.83–0.85, общих голов 17–18 из 20 и 35 из 40. Относительно A0 — 0.95–0.96 и 38–39 из 40. Новых голов нет.
2. **Перефраз вопроса сам по себе поиск не ломает.** Без маски A1 и A2 дают 100% верных ответов. A3 даёт 87%, и вся потеря — одна игла (`oslo`, 62%): модель отвечает по памяти про Oslo Cathedral вместо текста.
3. **Отключение лучших голов ломает ответы на всех уровнях:** 40 голов оставляют 69–86% верных, 60 голов — 20–40%. Случайные 60 голов оставляют 76–100%.
4. **Падение скора копирования на A1 и A3 — артефакт формулировки ответа, а не смена голов.** У 20 лучших голов 0.68 → 0.61, но на дословных ответах скор одинаков на всех уровнях: 0.68 / 0.68 / 0.67 / 0.67.
5. **Строковые метрики на этой оси вводят в заблуждение в обе стороны.** Точная фраза бракует 35–38% верных ответов на A1 и A3; ROUGE-1 под маской 60 голов засчитывает 87% ответов на A0 при 30% верных по фактам.

## 4.2. Зависимости в ответах

### От маски

| Маска | верно | частично | иное |
|---|---|---|---|
| без маски | 96.8 | 0 | 3.2 |
| 60 случайных | 90.7 | 0.3 | 9.1 |
| 40 лучших | 75.6 | 19.4 | 4.9 |
| 60 лучших | 30.1 | 38.8 | 31.2 |

- **Маска лучших голов сначала обрезает ответ, потом теряет его.** При 40 головах неверные ответы — почти только «частично» (назван один факт из двух); при 60 к ним добавляется «иное».
- **Дословность исчезает раньше правильности.** При 40 головах верных по фактам 71% (A0), а точных фраз 2.5%; ответов со всеми частями эталона — 37%.
- **Случайная маска на этой оси не безобидна,** но весь её вред приходится на `oslo` (73.5% верных против 98.5–100% у остальных). Под случайной маской ответы резко удлиняются: 148–169 токенов на A1 и A3 против 35–49 без маски.

### От уровня задачи

| Маска | A0 | A1 | A2 | A3 |
|---|---|---|---|---|
| без маски | 100 | 100 | 100 | 87.4 |
| 60 случайных | 100 | 90.9 | 95.5 | 76.3 |
| 40 лучших | 70.7 | 85.9 | 76.8 | 69.2 |
| 60 лучших | 29.8 | 39.9 | 20.2 | 30.3 |

- **Перефраз вопроса (A1) под маской устойчивее дословного вопроса:** +15 п.п. при 40 головах, +10 при 60. Без `oslo` разница больше: +19 и +20 п.п.
- **Замена ключа описанием (A2) устойчивости не добавляет:** как A0 при 40 головах, ниже A0 при 60.
- **Направление эффекта не совпадает между иглами.** При 40 головах `cairo` даёт 53% на A0 и 91% на A1, `lisbon` — 96% и 96%, `oslo` — 64% и 71%. Вывод «перефраз защищает от маски» держится на двух иглах из трёх.

### От глубины иглы

Верные ответы, %, все уровни вместе:

| Маска | 0 | 22 | 44 | 67 | 89 | 100 |
|---|---|---|---|---|---|---|
| без маски | 100 | 96.2 | 96.2 | 94.7 | 93.9 | 100 |
| 60 случайных | 94.7 | 87.1 | 88.6 | 87.1 | 87.9 | 98.5 |
| 40 лучших | 93.2 | 65.9 | 66.7 | 68.9 | 75.0 | 84.1 |
| 60 лучших | 47.0 | 22.7 | 28.8 | 22.7 | 23.5 | 35.6 |

- **Края контекста устойчивее середины при любой маске.** Игла в самом начале (глубина 0) почти не страдает от 40 голов: 93% против 66–75% в середине.
- **Край в начале сильнее края в конце:** при 40 головах 93% против 84%, при 60 — 47% против 36%.
- **Тип ошибки тоже зависит от глубины.** При 40 головах на глубине 0 «частичных» ответов 5%, в середине — 23–27%. При 60 головах «иное» составляет 16–23% на краях и 32–40% в середине.
- **Без маски зависимость от глубины есть только на A3** и создана теми же ответами `oslo` по памяти: на краях их нет, в середине они есть.
- **Преимущество глубины 0 несут не все иглы:** под 60 головами у `lisbon` 80% на глубине 0, у `cairo` 48%, у `oslo` 14% (у `oslo` лучше всего конец контекста, 36%).

### От длины контекста

| Маска | короткие (до 7k) | длинные (от 22k) | тренд по 11 длинам (Спирмен) |
|---|---|---|---|
| без маски | 99.1 | 95.1 | −0.85 |
| 60 случайных | 96.3 | 86.8 | −0.91 |
| 40 лучших | 83.8 | 70.8 | −0.84 |
| 60 лучших | 38.9 | 22.6 | −0.79 |

- **С ростом контекста верных ответов меньше при любой маске,** и чем сильнее маска, тем больше абсолютная потеря: −4, −10, −13, −16 п.п.
- **При 60 головах самый короткий контекст стоит особняком:** 54% на 1000 токенов и 19–36% на всех остальных длинах.
- **С длиной растёт доля «иное», а не «частично».** При 60 головах — от 6% на 1000 токенов до 35–46% на длинах от 22k: на длинном контексте модель чаще теряет ответ целиком.
- **Падение без маски и под случайной маской — снова `oslo` на A3.**

### От иглы

| Маска | `cairo` | `lisbon` | `oslo` |
|---|---|---|---|
| без маски | 100 | 100 | 90.5 |
| 60 случайных | 98.5 | 100 | 73.5 |
| 40 лучших | 76.9 | 89.8 | 60.2 |
| 60 лучших | 22.3 | 47.7 | 20.1 |

- **`lisbon` — самая устойчивая игла на этой оси** при любой маске; модель отвечает на неё дословно на всех уровнях.
- **`cairo` пересказывает ответ на перефразированный вопрос:** на A1 83% полных ответов записаны формой слова или синонимом (`prepared using` вместо `cooked with`). По всем прогонам такие перефразированные полные ответы верны по фактам в 98.8% случаев.
- **`oslo` — источник почти всех аномалий оси:** ответы по памяти на A3, вред случайной маски, резкая зависимость от длины (83% на 1000 токенов и не больше 21% дальше при 60 головах).

## 4.3. Головы

- **Значения и порядок исходных голов сохраняются.** По 62 головам, которые детекция ставит выше порога 0.1: Пирсон 0.94–0.96, Спирмен 0.88–0.91 на всех уровнях. Между самими уровнями порядок почти не меняется: 0.98–1.00.
- **Замена ключа описанием на порядок голов не влияет** (A0 и A2 совпадают), перефраз влияет слабо.
- **Распределение скора то же:** голов выше порога 79–85 на уровнях оси (62 в детекции — разница игл, она есть уже на A0); голов, активных во всех примерах, 29–31.
- **Внимание к игле при перефразе слабеет, хотя копирование сохраняется.** У 20 лучших голов масса на ответе в игле падает с 0.33 (A0) до 0.24 (A3), на всей игле — с 0.40 до 0.29. Замена ключа (A2) и перефраз (A1) дают по −0.04…−0.07, вместе — сумму.
- **Вывод по головам не зависит от иглы:** у `lisbon` скор 20 лучших голов 0.68–0.69 на всех уровнях; у `cairo` и `oslo` он ниже на A1 и A3 ровно там, где ответ пересказан.

## 4.4. В сравнении с другими осями

Верные ответы по фактам, %, среднее по уровням оси (A0 — общая точка отсчёта):

| Ось | без маски | 60 случайных | 40 лучших | 60 лучших |
|---|---|---|---|---|
| A0: дословный вопрос, без вставок | 100 | 100 | 70.7 | 29.8 |
| A: перефраз вопроса (A1–A3) | 95.8 | 87.6 | 77.3 | 30.1 |
| B: близнецы (B1–B4) | 99.8 | 98.8 | 60.8 | 54.6 |
| N: шум (N1–N4) | 100 | 100 | 66.1 | 28.7 |

Головы, 20 лучших по детекции на простой задаче:

| Ось | Спирмен рейтинга с A0 | общих голов в топ-20 с A0 | скор копирования | масса на игле | избирательность к вставкам |
|---|---|---|---|---|---|
| A0 | — | — | 0.68 | 0.40 | — |
| A | 0.95–0.96 | 17–19 | 0.61–0.67 (на дословных ответах 0.67–0.68) | 0.29–0.37 | — |
| B | 0.95 | 19 | 0.61–0.65 | 0.33–0.37 | 0.77–0.84 |
| N | 0.96–0.97 | 19 | 0.68–0.69 | 0.40–0.41 | 0.97–0.99 |

- **Набор голов один и тот же на всех трёх осях.** Ни на одном уровне в двадцатку не вошла голова из-за пределов первых 60 мест детекции.
- **Усложнения различаются тем, что они делают с вниманием этих голов.** Шум не меняет ничего; близнецы забирают часть внимания (до 0.09 массы на самого сильного близнеца) и снижают скор копирования; перефраз вопроса снижает массу на игле сильнее всего, не меняя копирования.
- **Без маски ни одно усложнение не ломает модель,** кроме A3 на одной игле. Запас прочности большой: эффект усложнения виден в ответах только тогда, когда часть голов отключена.
- **Маска 40 лучших голов делает усложнения различимыми:** близнецы (B1–B3) −12…−15 п.п. к A0, шум −3…−7 п.п., перефраз вопроса (A1) +15 п.п., замена ключа (A2) +6 п.п.
- **При 60 головах шум не отличается от A0, а близнецы дают выигрыш:** при трёх близнецах рядом и при записи с тем же ключом верных ответов больше, чем без вставок, у двух игл из трёх (`oslo` и `cairo`), у третьей — только на B4. Одновременно под обеими масками с близнецами растёт доля дословных ответов — у всех трёх игл при 40 головах (подробно — в выводах оси B).

## 4.5. Ограничения и что стоит проверить

- **Три иглы ведут себя по-разному,** и порядок их устойчивости меняется от оси к оси и от маски к маске. Любой вывод «уровень X труднее уровня Y под маской» нужно подтверждать на большем числе игл.
- **Один набор случайных голов** (seed 0). Вред случайной маски на оси A может быть свойством этого набора.
- **Метрики голов посчитаны только по верным ответам без маски.** Что делают головы в примерах, где модель ошиблась, и что делают оставшиеся головы под маской, эти данные не показывают: под маской внимание не сохранялось.
- **Класс по фактам требует точного написания.** «Ring the bell» без `brass` — это «иное», а не «частично». Доля «иное» под маской 60 голов поэтому включает ответы из иглы с потерянной деталью (раздел 2.4).
- **Маска — равномерное внимание головы на шагах генерации,** а не удаление её выхода; при прогоне контекста головы работают как обычно.
""")

if __name__ == "__main__":
    nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
    if "--no-run" not in sys.argv:
        from nbclient import NotebookClient
        NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": HERE}}).execute()
    nbf.write(nb, os.path.join(HERE, "hard_A.ipynb"))
    print("written", os.path.join(HERE, "hard_A.ipynb"))
