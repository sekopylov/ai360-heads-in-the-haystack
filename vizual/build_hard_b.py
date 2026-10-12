"""
Builds vizual/hard_B.ipynb (axis B of the hard tasks: twins of the needle in the context) from the cells below and runs it.

python vizual/build_hard_b.py            # build and run
python vizual/build_hard_b.py --no-run   # build only
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
# Сложные задачи, ось B: близнецы иглы

**Вопрос ноутбука:** сохраняют ли головы, найденные на простой задаче, свою роль, когда в тексте рядом с иглой стоят записи того же вида с другими фактами — и не начинают ли они смотреть на эти записи.

Игла и вопрос те же, что на уровне A0; в стог добавлены близнецы.

| Часть | Что в ней | Прогоны |
|---|---|---|
| 1 | уровни оси и как проверяется ответ | — |
| 2 | **метрики ответов**: без маски и под масками; те же расчёты без семейства `oslo`; формулировки ответов; формулировки без `oslo` | все четыре |
| 3 | **метрики голов**: те же графики, что на исходной задаче, по уровням; сохранение рейтинга; контроли; игла против близнецов | только без маски |

Прогоны: без маски (внимание сохранено, есть метрики голов) и три маски по рейтингу детекции на простой задаче — 40 и 60 лучших голов, 60 случайных. В каждом 3 семейства игл × 4 уровня × 11 длин × 6 глубин = 792 примера. Точка отсчёта — уровень A0 из прогонов оси A: та же игла, тот же вопрос, та же сетка и те же маски, но без вставок.
""")

md("## 0. Настройки")
code("""
import os, sys
sys.path.insert(0, os.path.abspath(".."))   # корень репозитория: ноутбук лежит в vizual/

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vizual import data, hard, paraphrase, plots, style

RUNS = "results/new/qwen3_hard_fast/B/*"          # все прогоны оси
NONE = "results/new/qwen3_hard_fast/B/none"       # прогон без маски: в нём метрики голов
REFERENCE = "results/new/qwen3_hard_fast/A/*"     # прогоны оси A: из них берётся уровень A0, точка отсчёта
BASE = "A0"
DETECT = "results/new/qwen3_detect"               # детекция на простой задаче: рейтинг, по которому строились маски

METRIC = "copy_recall"    # скор головы: copy_recall — как в статье, copy_count — как в коде авторов
THRESHOLD = 0.1           # порог retrieval-головы из статьи
EXCLUDE = ["oslo"]        # семейства, которые убираются в разделах «без oslo»
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
# ответы читаются из outputs/, метрики ответа считаются по ним; уровень A0 добавлен из прогонов оси A
base = hard.load(REFERENCE)
sweep = pd.concat([base[base["level"] == BASE], hard.load(RUNS)], ignore_index=True)
sweep["twin_words"] = hard.twin_words(sweep)
levels = sorted(sweep["level"].unique())            # A0, B1, B2, B3, B4
twin_levels = [level for level in levels if level != BASE]
families = sorted(sweep["family"].unique())
none = sweep[sweep["run"] == "none"]
print(f"прогонов: {sweep['run'].nunique()}; семейства: {families}; уровни: {levels}")
print("примеров в прогоне по уровням:")
display(sweep.groupby(["run", "level"]).size().unstack())

# маски оси B те же, что у оси A: иначе A0 не был бы точкой отсчёта
for run in sorted(set(sweep["run"]) - {"none"}):
    same = data.run_info(REFERENCE.replace("*", run)).get("block_list") == data.run_info(RUNS.replace("*", run)).get("block_list")
    print(f"{run}: маска совпадает с осью A — {same}")
""")

md("""
## 1. Уровни и как мы их проверяем

### 1.1. Что меняется на каждом уровне

Близнец — предложение того же шаблона, что игла, но про другой город и с другими фактами: «Every visitor to the **Porto** Tram Museum is asked to pull the iron lever…». На вопрос про Лиссабон оно не отвечает, но выглядит как ответ.

| Уровень | Что добавлено в стог | Где стоит | Что проверяет |
|---|---|---|---|
| **A0** | ничего | — | точка отсчёта: та же игла и вопрос без вставок |
| **B1** | один близнец | далеко: на противоположной глубине, `(глубина иглы + 50) mod 100` | мешает ли сам факт существования похожей записи |
| **B2** | три близнеца | на случайных границах предложений по всему контексту | число конкурентов |
| **B3** | три близнеца | рядом: в пределах 200 токенов от иглы | близость конкурентов к игле |
| **B4** | одна запись с **тем же ключом**, но о другом (сотрудники музея вместо посетителей; хлеб вместо супа; самый новый дом вместо самого старого) | рядом | различает ли модель записи про одно и то же место |

B1 → B2 меняет число близнецов, B2 → B3 — только их расстояние до иглы. B4 — отдельный случай: ключ совпадает с вопросом, отличается предмет.

Длина вставок вычитается из стога, поэтому промпт каждого уровня той же длины, что на A0.
""")
code("""
# иглы, эталоны и факты, по которым проверяется ответ
first = none.drop_duplicates("family").set_index("family")
needles = {name: data.read_jsonl(data.path(f"data/needles_hard/{name}/axis_B.jsonl"))[0]["needle"] for name in families}
display(pd.DataFrame({"игла": needles, "вопрос": first["question"], "эталон (ответ внутри иглы)": first["reference"],
                      "факты ответа (допустимые написания)": first["answer_key"].map(lambda key: "; ".join(" / ".join(fact) for fact in key))}))

# вставки: текст, факты и уровни, на которых они стоят
rows = []
for name in families:
    with open(data.path(f"data/needles_hard/{name}/inserts_B.json"), encoding="utf-8") as f:
        spec = json.load(f)
    for insert, item in spec["inserts"].items():
        used = [f"{level} ({cfg['placement']})" for level, cfg in spec["levels"].items() if insert in cfg["use"]]
        rows.append({"семейство": name, "вставка": insert, "уровни": ", ".join(used), "текст": item["text"],
                     "факты вставки": "; ".join(" / ".join(fact) for fact in item["facts"])})
display(pd.DataFrame(rows).set_index(["семейство", "вставка"]))

# фактическое расстояние вставок от иглы, токены
offsets = pd.DataFrame([{"level": row["level"], "offset": item["offset"]} for row in none.to_dict("records") for item in row["inserts"].values()])
print("расстояние вставки от иглы, токены")
display(offsets.groupby("level")["offset"].describe()[["count", "min", "50%", "max"]].round(0).astype(int))
""")

md("""
### 1.2. Как проверяется ответ

Ответ проверяется по **фактам**, правило — `fact_class` из `rh/metrics.py`. На оси B у каждого примера два набора фактов: факты иглы (`answer_key`) и факты близнецов (`confusion_facts`).

1. Ответ и факты приводятся к одному виду: нижний регистр, без пунктуации, Unicode NFKC.
2. Факт засчитан, если любое из его допустимых написаний стоит в ответе целым словом или фразой.
3. Класс ответа:

| Класс | Условие | Что это значит на оси B |
|---|---|---|
| **верно** | оба факта иглы, ни одного факта близнецов | модель нашла нужную запись и не перепутала её с близнецом |
| **смешано** | оба факта иглы и хотя бы один факт близнеца | модель перечислила несколько записей, включая нужную |
| **частично** | один факт иглы из двух, фактов близнецов нет | ответ найден, но оборван |
| **путаница** | не оба факта иглы и есть факт близнеца | ответ списан с близнеца — именно та ошибка, ради которой сделана ось |
| **иное** | ни одного факта | отказ, рассуждение или ответ с потерей деталей («ring the bell») |

Класс «шум» на оси B невозможен: вставок с ролью шума нет.

**ROUGE здесь не годится как критерий.** Близнец повторяет шаблон иглы, поэтому ответ, списанный с него, набирает высокий ROUGE-1 recall и может пройти порог 50 (по README — 54–85 для дословно списанного близнеца). Что получается на данных — в разделе 2.1.

**Чего проверка не ловит.** Факт засчитывается только в написании из данных. «Pull the lever» вместо «pull the iron lever» путаницей не считается и попадает в «иное». Поэтому рядом с классами считается более мягкий признак — **слова близнецов в ответе**: последнее слово каждого факта близнеца (`lever`, `clock`, `raven`), если его нет в самом эталоне и вопросе. Это не класс, а контроль того, что путаница не занижена.

### 1.3. Как проверяются головы

- Метрики голов считаются только в прогоне без маски: под маской внимание не сохранялось.
- В скор головы на уровне входят только примеры с **верным** ответом (по фактам).
- Рейтинг голов **не переопределяется**: он взят из детекции на простой задаче.
- Уровни сравниваются на одних и тех же местах иглы (семейство, длина, глубина) — тех, где ответ верен на всех уровнях, включая A0.
- Новое на этой оси: у каждой вставки размечен свой отрезок, поэтому для каждой головы известна масса внимания на близнецах и число токенов, скопированных из них (раздел 3.6).
""")

# ------------------------------------------------------------------------------------------------------- answers
md("""
# 2. Метрики ответов

| Метрика | Годится? | Почему |
|---|---|---|
| класс ответа по фактам | да, основная | отличает ответ из иглы от ответа, списанного с близнеца |
| доля названных фактов иглы | да | та же проверка без порога «оба факта» |
| слова близнецов в ответе | да, как контроль | мягкий признак путаницы, не зависящий от точного написания факта |
| ROUGE-1/2/3 recall | **нет** | близнец повторяет шаблон иглы: списанный с него ответ может пройти порог; и, как на оси A, порог ROUGE-1 проходят ответы с одним фактом из двух |
| точная фраза | только как вспомогательная | ответ из близнеца её не проходит, но и верный пересказ тоже |

## 2.1. Без маски

Насколько сама задача труднее без вмешательства в модель.
""")
code("""
print("верные ответы без маски, %")
display(hard.by_level(none, by=("family", "level")))
show(plots.stacked(none, "level", "answer_class", hard.CLASSES, hard.CLASS_LABELS, xlabel="уровень", title="Без маски: классы ответов по уровням"), "B_none_classes")

wrong = none[~none["correct"]]
print(f"неверных ответов без маски: {len(wrong)} из {len(none)}")
display(wrong[["family", "level", "answer_class", "rouge1_recall", "reference", "response"]].round(0))
""")

md("""
### Почему ROUGE здесь не критерий

Все прогоны оси, уровни с близнецами. По каждому классу ответа — средний ROUGE и доля ответов, которые прошли бы критерий статьи (ROUGE-1 recall > 50) и более строгие. Если бы ROUGE-1 был критерием, «успехом» считались бы и неверные ответы из строк ниже первой.
""")
code("""
twins = sweep[sweep["level"] != BASE]
check = twins.groupby("answer_class").agg(ответов=("id", "size"), rouge1=("rouge1_recall", "mean"), rouge2=("rouge2_recall", "mean"), rouge3=("rouge3_recall", "mean"),
                                          успех_rouge1=("success", "mean"), успех_rouge2=("success2", "mean"), успех_rouge3=("success3", "mean"),
                                          точная_фраза=("exact", "mean")).reindex([c for c in hard.CLASSES if c in set(twins["answer_class"])])
check[["успех_rouge1", "успех_rouge2", "успех_rouge3", "точная_фраза"]] *= 100
display(check.round(1).rename(index=hard.CLASS_LABELS))

confused = twins[twins["answer_class"] == "confusion"].sort_values("n_tokens")
print("ответы класса «путаница»: все, от коротких к длинным")
display(confused[["run", "family", "level", "rouge1_recall", "reference", "response"]].round(0))
""")

md("""
## 2.2. Под масками

Головы отключаются по рейтингу детекции на простой задаче. Два вопроса: ломает ли маска лучших голов ответы при близнецах так же, как без них (A0), и во что превращаются ответы — в обрезанные или в списанные с близнеца.
""")
code("""
print("верные ответы, %")
display(hard.by_level(sweep))
show(plots.level_lines(sweep, "correct", title="Верные ответы по уровням"), "B_correct_by_level")
show(plots.class_panels(sweep, hard.CLASSES, hard.CLASS_LABELS, title="Классы ответов по уровням"), "B_classes_by_run")

# те же верные ответы в разрезе длины контекста и глубины иглы: панель на прогон, линия на уровень
show(plots.level_gradation(sweep, "context_length", title="Верные ответы по длине контекста"), "B_correct_by_length")
show(plots.level_gradation(sweep, "depth_percent", title="Верные ответы по глубине иглы"), "B_correct_by_depth")
""")

md("""
### Путаница с близнецами

Доля ответов, в которых есть факты близнецов: классы «путаница» и «смешано» вместе, и мягкий признак — слова близнецов в ответе.
""")
code("""
with_twins = sweep[sweep["level"] != BASE].assign(twin_facts=lambda d: d["answer_class"].isin(["confusion", "mixed"]))

axes = panels(2, 7.4, extra=2.4)
plots.level_bars(with_twins, "twin_facts", ylabel="доля ответов, %", ax=axes[0], title="Факты близнецов в ответе (путаница + смешано)")
plots.level_bars(with_twins, "twin_words", ylabel="доля ответов, %", ax=axes[1], title="Слова близнецов в ответе")
axes[0].get_legend().remove()
top = max(ax.get_ylim()[1] for ax in axes)
for ax in axes:
    ax.set_ylim(0, top)
show(axes, "B_twin_confusion")

print("ответы с фактами близнецов (путаница + смешано), %")
display(hard.by_level(with_twins, "twin_facts"))
print("ответы со словами близнецов, %")
display(hard.by_level(with_twins, "twin_words"))

print("число ответов каждого класса")
display(pd.crosstab([sweep["run"], sweep["level"]], sweep["answer_class"]).reindex(columns=[c for c in hard.CLASSES if c in set(sweep["answer_class"])]).rename(columns=hard.CLASS_LABELS))
""")

md("""
#### Во что превращаются неверные ответы

Только неверные ответы на уровнях с близнецами: какая их доля обрезана («частично»), потеряла факты («иное») или содержит факты близнеца («путаница», «смешано»). Под столбиком — число неверных ответов; там, где их единицы, доли шумные.
""")
code("""
failed = with_twins[~with_twins["correct"]]
kinds = [c for c in hard.CLASSES if c != "correct"]
axes = plots.class_panels(failed, kinds, hard.CLASS_LABELS, title="Неверные ответы на уровнях с близнецами: чем они неверны")
counts = failed.groupby(["run", "level"]).size()
order_runs = sorted(failed["run"].unique(), key=lambda r: (failed.loc[failed["run"] == r, "mask"].iloc[0] != "none", r))
for ax, run in zip(axes[0], order_runs):
    present = sorted(failed.loc[failed["run"] == run, "level"].unique())
    ax.set_xticklabels([f"{level}\\nn = {counts[(run, level)]}" for level in present])
show(axes, "B_failed_kinds")

# сама путаница: сколько таких ответов и на что они похожи больше — на иглу или на близнеца
print("число ответов с фактами близнецов")
display(pd.crosstab([with_twins["run"], with_twins["level"]], with_twins["answer_class"])[[c for c in ("confusion", "mixed") if c in set(with_twins["answer_class"])]]
        .rename(columns=hard.CLASS_LABELS).unstack("level", fill_value=0))
""")

md("""
### Классы ответов по длине контекста и глубине иглы

Те же классы, что выше, в разрезе сетки прогона. Столбец панелей — прогон, строка — уровень; верхняя строка — все уровни вместе. Столбики стоят на настоящих значениях сетки; сетка неравномерна на краю (длины 28.5k и 30k, глубины 89 и 100 стоят ближе друг к другу), поэтому последние два столбика уже остальных. В одном столбике по уровню 18 примеров на длину и 33 на глубину (три семейства).
""")
code("""
show(plots.class_gradation(sweep, "context_length", hard.CLASSES, hard.CLASS_LABELS, title="Классы ответов по длине контекста"), "B_classes_by_length")
show(plots.class_gradation(sweep, "depth_percent", hard.CLASSES, hard.CLASS_LABELS, title="Классы ответов по глубине иглы"), "B_classes_by_depth")
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

Только примеры, на которые модель без маски ответила верно: какая их доля осталась верной под маской. Строка `n` — число таких примеров.
""")
code("""
kept = hard.paired(sweep)
display(kept)

base_kept = kept.loc[kept.index != "n", BASE]
print(f"разница с {BASE}, процентные пункты: плюс — уровень устойчивее к маске, минус — уязвимее")
display(kept.loc[kept.index != "n"].sub(base_kept, axis=0).round(1))
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
show(axes, "B_correct_by_family")
""")

md("""
### Длина ответа и вспомогательные метрики

ROUGE приведён только как описание того, насколько ответ похож на эталон; критерием он здесь не служит.
""")
code("""
axes = panels(3, 6.0)
for ax, (value, name) in zip(axes, {"facts_found": "Доля названных фактов иглы", "exact": "Точная фраза", "n_tokens": "Длина ответа"}.items()):
    plots.level_lines(sweep, value, ax=ax, title=name)
    if ax is not axes[-1]:
        ax.get_legend().remove()
show(axes, "B_aux_by_level")

display(sweep.groupby(["run", "level"]).agg(верно=("correct", "mean"), фактов=("facts_found", "mean"), слова_близнецов=("twin_words", "mean"),
                                           точная_фраза=("exact", "mean"), rouge1=("rouge1_recall", "mean"), токенов=("n_tokens", "mean")).round(2))
""")

md("""
### Где ломается: длина и глубина

Доля верных ответов по длине контекста и глубине иглы, панель на уровень. В ячейке три примера (по одному на семейство).
""")
code("""
for run in [r for r in ("none", "top40", "top60", "random60") if r in set(sweep["run"])]:
    part = sweep[sweep["run"] == run]
    name = plots.run_styles(sweep)[run][1]
    show(plots.niah_grid(part.assign(run=part["level"]), {level: level for level in levels}, value="correct", columns=len(levels),
                         title=f"Верные ответы: {name}"), f"B_niah_{run}")
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

На оси A семейство `oslo` вело себя иначе, чем два других, и расчёты повторялись без него. Здесь то же сделано для сопоставимости: на оси B `oslo` тоже выделяется, но в другую сторону — под маской 60 лучших голов оно почти не страдает (раздел «По семействам» выше).
""")
code("""
rest = sweep[~sweep["family"].isin(EXCLUDE)]
rest_none = rest[rest["run"] == "none"]
print(f"семейства: {sorted(rest['family'].unique())}; примеров без маски: {len(rest_none)}")

print("без маски: верные ответы, %")
display(hard.by_level(rest_none, by=("family", "level")))

print("верные ответы, %: без oslo")
display(hard.by_level(rest))
print("для сравнения — все три семейства")
display(hard.by_level(sweep))

axes = panels(2, 5.6, extra=2.6, sharey=True)
plots.level_lines(sweep, "correct", ax=axes[0], title="Все семейства")
plots.level_lines(rest, "correct", ax=axes[1], title="Без oslo")
axes[0].get_legend().remove()
axes[1].set_ylabel("")
show(axes, "B_rest_correct_by_level")

show(plots.class_panels(rest, hard.CLASSES, hard.CLASS_LABELS, title="Классы ответов по уровням, без oslo"), "B_rest_classes_by_run")

# те же верные ответы в разрезе длины контекста и глубины иглы: панель на прогон, линия на уровень
show(plots.level_gradation(rest, "context_length", title="Верные ответы по длине контекста, без oslo"), "B_rest_correct_by_length")
show(plots.level_gradation(rest, "depth_percent", title="Верные ответы по глубине иглы, без oslo"), "B_rest_correct_by_depth")

show(plots.class_gradation(rest, "context_length", hard.CLASSES, hard.CLASS_LABELS, title="Классы ответов по длине контекста, без oslo"), "B_rest_classes_by_length")
show(plots.class_gradation(rest, "depth_percent", hard.CLASSES, hard.CLASS_LABELS, title="Классы ответов по глубине иглы, без oslo"), "B_rest_classes_by_depth")
""")

md("""
### На тех же примерах, без `oslo`

Доля примеров, верных без маски, которые остались верными под маской.
""")
code("""
kept_rest = hard.paired(rest)
display(kept_rest)

print(f"разница с {BASE}, процентные пункты: без oslo")
display(kept_rest.loc[kept_rest.index != "n"].sub(kept_rest.loc[kept_rest.index != "n", BASE], axis=0).round(1))
print("та же разница по всем трём семействам")
display(kept.loc[kept.index != "n"].sub(base_kept, axis=0).round(1))
""")

md("""
### Где ломается: длина и глубина, без `oslo`

В ячейке два примера (по одному на семейство).
""")
code("""
for run in [r for r in ("none", "top40", "top60", "random60") if r in set(rest["run"])]:
    part = rest[rest["run"] == run]
    name = plots.run_styles(rest)[run][1]
    show(plots.niah_grid(part.assign(run=part["level"]), {level: level for level in levels}, value="correct", columns=len(levels),
                         title=f"Верные ответы без oslo: {name}"), f"B_rest_niah_{run}")
""")

md("""
## 2.4. Формулировки ответов: перефраз, замены, формы слов

Класс по фактам говорит, названы ли ключевые факты. Здесь тот же ответ разбирается подробнее: эталон разбит на части, и для каждой части смотрим, как она записана. Это тот же разбор, что в ноутбуках по задаче статьи и по оси A, с теми же списками формулировок: игла и эталон те же.

| Уровень формулировки | Что это | Пример |
|---|---|---|
| дословно | формулировка эталона | `cooked with` |
| другая форма слова | те же слова в другой форме, с другим артиклем или в другом порядке | `cooking it with`; `a brass bell`; `black clay bowl` |
| синоним, смысл сохранён | другие слова с тем же смыслом | `prepared using`; `when they arrive` вместо `at the entrance` |
| потеряна деталь | часть названа, но без существенной детали | `ring a bell` без `brass`; `smoked lemon` без `peel`; `red roof` без `slate` |
| нет | части в ответе нет | — |

Часть считается сохранённой на первых трёх уровнях. Списки формулировок лежат в `vizual/paraphrase.py` (`FACTS`); они собраны вручную по ответам прогонов оси A, и отнесение формулировки к уровню — моё решение.

| Структура ответа | Условие |
|---|---|
| точная фраза | эталон стоит в ответе дословно и подряд |
| все части дословно, другой порядок или вставка | все части дословно, но фраза не совпадает |
| все части, есть форма слова или синоним | все части сохранены, хотя бы одна не дословно — перефразированный полный ответ |
| не все части, остальные дословно | часть пропала, оставшиеся дословно — обрезанный ответ |
| не все части, есть форма, синоним или потеря детали | часть пропала или потеряла деталь |
| ни одной части | ответ не из иглы — в том числе списанный с близнеца |

Этот разбор смотрит только на части **иглы**. Списан ли ответ с близнеца, он не различает — это делает класс по фактам (раздел 2.2).
""")
code("""
worded = paraphrase.annotate(sweep)
worded_none = worded[worded["run"] == "none"]
fact_names = {family: list(paraphrase.FACTS[worded.loc[worded["family"] == family, "reference"].iloc[0]]) for family in families}

# что именно написано вместо формулировки эталона: по всем прогонам
pd.set_option("display.max_rows", 200)
paraphrase.wordings(worded, top=5).set_index(["эталон", "факт", "уровень"])
""")

md("### Без маски: как модель формулирует ответ на каждом уровне")
code("""
ax = plt.subplots(figsize=style.size(11.5, 4.2))[1]
show(plots.stacked(worded_none, "level", "structure", paraphrase.STRUCTURES, paraphrase.STRUCTURE_LABELS, xlabel="уровень", ax=ax,
                   title="Без маски: структура ответа по уровням"), "B_structure_none")

print("без маски: доля ответов каждой структуры, %")
display((pd.crosstab([worded_none["family"], worded_none["level"]], worded_none["structure"], normalize="index") * 100).reindex(
    columns=[k for k in paraphrase.STRUCTURES if k in set(worded_none["structure"])]).rename(columns=paraphrase.STRUCTURE_LABELS).round(1))
""")
code("""
# как записана каждая часть эталона под маской 60 лучших голов: панель на часть, ряд на семейство
WORDING_RUN = "top60"
for family in families:
    part = worded[(worded["run"] == WORDING_RUN) & (worded["family"] == family)]
    names = fact_names[family]
    axes = panels(len(names), 4.6, extra=2.4, sharey=True)
    for ax, fact in zip(axes, names):
        plots.stacked(part, "level", fact, paraphrase.LEVELS, paraphrase.LEVEL_LABELS, xlabel="уровень", legend=ax is axes[-1], ax=ax, title=f"«{fact}»")
        if ax is not axes[0]:
            ax.set_ylabel("")
    axes[0].figure.suptitle(f"{family}: формулировка частей ответа, {plots.run_styles(sweep)[WORDING_RUN][1]}", x=0.01, ha="left", fontweight="bold")
    show(axes, f"B_fact_wordings_{family}")
""")

md("""
### Как класс по фактам оценивает каждую структуру

По всем прогонам оси. `верно` — доля ответов класса «верно» по фактам; `частей` — сколько частей эталона сохранено (у `oslo` частей три, у остальных четыре).
""")
code("""
by_structure = worded.groupby("structure").agg(ответов=("id", "size"), частей=("kept_facts", "mean"), верно=("correct", "mean"), слова_близнецов=("twin_words", "mean"),
                                               точная_фраза=("exact", "mean"), rouge1=("rouge1_recall", "mean")).reindex(paraphrase.STRUCTURES)
by_structure[["верно", "слова_близнецов", "точная_фраза"]] *= 100
display(by_structure.round(1).rename(index=paraphrase.STRUCTURE_LABELS))

print("структура против класса по фактам, число ответов")
display(pd.crosstab(worded["structure"], worded["answer_class"]).reindex(paraphrase.STRUCTURES).reindex(
    columns=[c for c in hard.CLASSES if c in set(worded["answer_class"])]).rename(index=paraphrase.STRUCTURE_LABELS, columns=hard.CLASS_LABELS))
""")

md("### Под масками")
code("""
show(plots.class_panels(worded, paraphrase.STRUCTURES, paraphrase.STRUCTURE_LABELS, column="structure", title="Структура ответа по уровням"), "B_structure_by_run")

full = worded["structure"].isin(["exact", "reordered", "reworded"])
print("ответы со всеми частями эталона (дословно, формой слова или синонимом), %")
display(hard.by_level(worded.assign(full=full), "full"))
print("из них перефразированные (не точная фраза), %")
display(hard.by_level(worded[full].assign(reworded=lambda d: d["structure"] != "exact"), "reworded"))
""")
code("""
# примеры: по два самых коротких ответа каждой структуры на уровнях с близнецами
examples = worded[(~worded["exact"]) & (worded["level"] != BASE)].sort_values("n_tokens").groupby("structure").head(2)
examples.assign(структура=examples["structure"].map(paraphrase.STRUCTURE_LABELS)).set_index("структура").loc[
    [paraphrase.STRUCTURE_LABELS[k] for k in paraphrase.STRUCTURES if k in set(examples["structure"])],
    ["run", "family", "level", "answer_class", "response"]]
""")

md("""
## 2.5. Формулировки ответов без семейства `oslo`

Тот же разбор, что в разделе 2.4, на семействах `cairo` и `lisbon`.
""")
code("""
worded_rest = worded[~worded["family"].isin(EXCLUDE)]
worded_rest_none = worded_rest[worded_rest["run"] == "none"]
print(f"семейства: {sorted(worded_rest['family'].unique())}; примеров без маски: {len(worded_rest_none)}")

ax = plt.subplots(figsize=style.size(11.5, 4.2))[1]
show(plots.stacked(worded_rest_none, "level", "structure", paraphrase.STRUCTURES, paraphrase.STRUCTURE_LABELS, xlabel="уровень", ax=ax,
                   title="Без маски и без oslo: структура ответа по уровням"), "B_rest_structure_none")

show(plots.class_panels(worded_rest, paraphrase.STRUCTURES, paraphrase.STRUCTURE_LABELS, column="structure", title="Структура ответа по уровням, без oslo"), "B_rest_structure_by_run")

full_rest = worded_rest["structure"].isin(["exact", "reordered", "reworded"])
print("ответы со всеми частями эталона (дословно, формой слова или синонимом), %: без oslo")
display(hard.by_level(worded_rest.assign(full=full_rest), "full"))
print("то же по всем трём семействам")
display(hard.by_level(worded.assign(full=full), "full"))

print("структура против класса по фактам, число ответов: без oslo")
display(pd.crosstab(worded_rest["structure"], worded_rest["answer_class"]).reindex(paraphrase.STRUCTURES).reindex(
    columns=[c for c in hard.CLASSES if c in set(worded_rest["answer_class"])]).rename(index=paraphrase.STRUCTURE_LABELS, columns=hard.CLASS_LABELS))
""")

# --------------------------------------------------------------------------------------------------------- heads
md("""
# 3. Метрики голов

Только прогон без маски. Скоры считаются по верно отвеченным примерам.

| Метрика | Годится? | Почему |
|---|---|---|
| `copy_recall`, `copy_count` — скор копирования из статьи | да | игла и вопрос не менялись, ответ списывается из иглы; близнецы в этот скор не входят — засчитывается только токен внутри ответа иглы |
| частота активации | да | доля примеров, где голова скопировала хотя бы один токен иглы |
| `needle_mass`, `needle_attention_mass` — масса внимания на ответе в игле | да | куда голова смотрит, независимо от формулировки ответа |
| `needle_sentence_mass` — масса на всей игле | да, основная для сравнения с близнецами | игла и близнец — целые предложения одного шаблона, поэтому сравнивать нужно предложение с предложением |
| `competing_max_mass` — масса на самом сильном близнеце | да | один близнец против одной иглы: сравнимо между уровнями с одним и с тремя близнецами |
| `competing_mass` — масса на всех близнецах вместе | да, с оговоркой | на B2 и B3 это сумма по трём вставкам, на B1 и B4 — по одной; между ними напрямую не сравнивается |
| `copy_sentence` и `copy_competing` — копирование из иглы и из близнецов | да | оба делятся на длину ответа иглы, поэтому сравнимы друг с другом и с `copy_count` |
| `supporting_mass` | нет | это внимание на вставки с частью ответа, на оси B таких нет: значения равны нулю |

На уровне A0 вставок нет, поэтому метрики близнецов там равны нулю по построению.
""")
code("""
heads = data.Heads(NONE, samples=none, ok="correct")      # в скор входят верные ответы; примеры A0 берутся из прогона оси A
detect = data.Heads(DETECT)                               # детекция на простой задаче (условие — успех по ROUGE-1, как в статье)
reference = detect.score(METRIC)                          # скор детекции, [слой, голова]
order = np.argsort(-reference.ravel(), kind="stable")     # головы от лучшей к худшей по детекции
place = {int(i): rank + 1 for rank, i in enumerate(order)}
total_heads = reference.size

common = hard.common_cells(heads.samples)                 # места иглы, отвеченные верно на всех уровнях
print(f"примеров с метриками голов: {len(heads.samples)}; на местах, верных на всех уровнях: {common.sum()} ({common.sum() // len(levels)} на уровень)")

scores = hard.level_scores(heads, METRIC, where=common)                                              # {уровень: [слой, голова]}
rankings = {level: heads.ranking(METRIC, hard.level_mask(heads, level, common)) for level in levels}   # рейтинг голов самого уровня
""")

md("""
## 3.1. Те же графики, что на исходной задаче

Каждый график из ноутбука по задаче статьи, построенный отдельно для каждого уровня. Здесь рейтинг считается заново на самом уровне — чтобы увидеть, как выглядела бы детекция, проведённая на тексте с близнецами.

### Распределение голов по скору (рис. 3 статьи)
""")
code("""
axes = panels(len(levels), 4.6, sharey=True)
for ax, level in zip(axes, levels):
    plots.score_distribution(scores[level], ax=ax, title=level)
    if ax is not axes[0]:
        ax.set_ylabel("")
show(axes, "B_score_distribution")

axes = panels(len(levels), 5.6, 3.6)
for ax, level in zip(axes, levels):
    plots.score_pie(scores[level], ax=ax, title=level)
show(axes, "B_score_pie")

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
axes = panels(len(levels), 4.8, extra=2.2, sharey=True)
for ax, level in zip(axes, levels):
    plots.score_vs_frequency(rankings[level], top=100, threshold=THRESHOLD, ax=ax, title=level)
    ax.set_xlabel("место головы по скору уровня")
    if ax is not axes[-1]:
        ax.get_legend().remove()
    if ax is not axes[0]:
        ax.set_ylabel("")
show(axes, "B_score_vs_frequency")

print("голов, активных во всех примерах уровня:", {level: int((rankings[level]["frequency"] == 1).sum()) for level in levels})
""")

md("""
### Скор каждой головы (рис. 5 статьи)

Одна шкала цвета на всех панелях.
""")
code("""
vmax = max(float(s.max()) for s in scores.values())
rows = -(-len(levels) // 2)
fig, axes = plt.subplots(rows, 2, figsize=style.size(17, 4.6 * rows))
for ax, level in zip(axes.ravel(), levels):
    plots.head_heatmap(scores[level], heads.layers, vmax=vmax, ax=ax, title=level)
for ax in axes.ravel()[len(levels):]:
    ax.set_visible(False)
show(axes, "B_head_heatmap")
""")

md("""
### 20 лучших голов уровня

Рейтинг самого уровня. Сравнение с рейтингом детекции — в разделе 3.2.
""")
code("""
axes = panels(len(levels), 4.2, 7.2)
for ax, level in zip(axes, levels):
    plots.top_heads(rankings[level], k=20, threshold=THRESHOLD, ax=ax, title=level)
    if ax is not axes[0]:
        ax.set_ylabel("")
show(axes, "B_top_heads")
""")

md("""
## 3.2. Сохраняется ли рейтинг детекции

Порядок голов на каждом уровне против порядка детекции на простой задаче: корреляция Спирмена по всем головам и число общих голов среди 20, 40 и 60 лучших. Строка A0 — это перенос на другие иглы без усложнения; остальные уровни сравниваются с ней.
""")
code("""
display(hard.ranking_table(scores, reference))

print(f"то же против {BASE} (та же игла и вопрос, добавлены только близнецы)")
display(hard.ranking_table({level: scores[level] for level in twin_levels}, scores[BASE]))
""")

md("""
### Корреляция скоров: исходные головы и наша задача

Точка — голова: по горизонтали её скор в детекции на простой задаче, по вертикали — на уровне. На диагонали скор сохранён. Выделены головы, которые детекция относит к retrieval-головам (скор выше порога).

r — корреляция Пирсона (совпадают ли сами значения), ρ — Спирмена (совпадает ли порядок). По всем головам корреляцию во многом задают сотни голов со скором около нуля, поэтому отдельно посчитано по исходным retrieval-головам.
""")
code("""
axes = panels(len(levels), 4.6, 4.9, sharey=True)
for ax, level in zip(axes, levels):
    plots.score_scatter(reference, scores[level], threshold=THRESHOLD, ax=ax, title=level)
    if ax is not axes[0]:
        ax.set_ylabel("")
        ax.get_legend().remove()
show(axes, "B_score_scatter")

display(hard.correlation_table(scores, reference, order, threshold=THRESHOLD).unstack("условие").swaplevel(axis=1).sort_index(axis=1, level=0, sort_remaining=False))
""")
code("""
# между всеми условиями сразу: корреляция Спирмена по головам детекции выше порога
found = reference.ravel() > THRESHOLD
show(plots.correlation({"детекция": reference.ravel()[found], **{level: scores[level].ravel()[found] for level in levels}},
                       title=f"Порядок {int(found.sum())} исходных retrieval-голов"), "B_score_correlation")
""")

md("""
## 3.3. Значения голов детекции по уровням

Головы стоят в порядке детекции и не переупорядочиваются. Если роль сохраняется, линии уровней идут вместе.
""")
code("""
show(plots.rank_profile(scores, order, top=60, reference=reference, title="Скор голов детекции на каждом уровне"), "B_rank_profile")

print("средний скор голов, которые отключаются масками")
display(hard.top_means({"детекция": reference, **scores}, order))

top = 30
table = pd.DataFrame({"детекция": reference.ravel()[order[:top]], **{level: scores[level].ravel()[order[:top]] for level in levels}},
                     index=[heads.name(i) for i in order[:top]])
show(plots.head_table(table, title=f"{top} лучших голов детекции по уровням"), "B_head_table")
""")

md("""
## 3.4. Контроль: дословные ответы

Скор копирования зависит от того, насколько дословно модель отвечает. Тот же расчёт повторён только на ответах с точной фразой: там число скопированных токенов одинаково на всех уровнях.
""")
code("""
exact = heads.samples["exact"].to_numpy()
print("ответов с точной фразой:", {level: int((exact & heads.samples["correct"] & (heads.samples["level"] == level)).sum()) for level in levels})
exact_scores = hard.level_scores(heads, METRIC, where=exact)

axes = panels(2, 6.6, extra=1.6, sharey=True)
plots.rank_profile(scores, order, top=60, legend=False, ax=axes[0], title="Все верные ответы")
plots.rank_profile(exact_scores, order, top=60, ax=axes[1], title="Только дословные ответы")
axes[1].set_ylabel("")
show(axes, "B_exact_control")

print("средний скор лучших голов детекции: все верные ответы")
display(hard.top_means(scores, order))
print("только дословные ответы")
display(hard.top_means(exact_scores, order))
""")

md("""
## 3.5. Куда головы смотрят: масса внимания на игле

Головы стоят в порядке детекции (по копированию).
""")
code("""
mass_metrics = {"needle_mass": "Масса на ответе в игле", "needle_sentence_mass": "Масса на всей игле", "needle_attention_mass": "Масса на ответе на шагах копирования"}
masses = {metric: hard.level_scores(heads, metric, where=common) for metric in mass_metrics}

axes = panels(3, 6.0, extra=1.4, sharey=True)
for ax, (metric, name) in zip(axes, mass_metrics.items()):
    plots.rank_profile(masses[metric], order, top=60, ylabel="масса внимания", legend=ax is axes[-1], ax=ax, title=name)
    if ax is not axes[0]:
        ax.set_ylabel("")
show(axes, "B_mass_profile")

for metric, name in mass_metrics.items():
    print(f"{name} ({metric}): среднее у лучших голов детекции")
    display(hard.top_means(masses[metric], order))
""")

md("""
## 3.6. Игла против близнецов

Главный раздел оси: смотрят ли retrieval-головы на близнецов и копируют ли из них. Все значения — на тех же верно отвеченных примерах, то есть модель в итоге ответила правильно.

### Масса внимания: игла и самый сильный близнец

`needle_sentence_mass` — доля внимания головы на предложении иглы; `competing_max_mass` — на том близнеце, который на данном шаге получает больше всего. Это сравнение «одно предложение против одного предложения», оно одинаково осмысленно при одном близнеце и при трёх.
""")
code("""
needle = masses["needle_sentence_mass"]
rival = hard.level_scores(heads, "competing_max_mass", where=common)
rivals_all = hard.level_scores(heads, "competing_mass", where=common)

top_mass = max(float(needle[level].ravel()[order[:60]].max()) for level in levels)
axes = panels(len(twin_levels), 4.8, extra=1.6, sharey=True)
for ax, level in zip(axes, twin_levels):
    plots.rank_profile({"игла": needle[level], "самый сильный близнец": rival[level]}, order, top=60, ylabel="масса внимания",
                       legend=ax is axes[-1], ax=ax, title=level)
    ax.set_ylim(0, top_mass * 1.05)
    if ax is not axes[0]:
        ax.set_ylabel("")
show(axes, "B_needle_vs_twin")

print("масса на всей игле: среднее у лучших голов детекции")
display(hard.top_means(needle, order))
print("масса на самом сильном близнеце")
display(hard.top_means({level: rival[level] for level in twin_levels}, order))
print("масса на всех близнецах вместе (на B2 и B3 их три)")
display(hard.top_means({level: rivals_all[level] for level in twin_levels}, order))
""")

md("""
### Избирательность

Какая доля внимания, отданного игле или её самому сильному близнецу, приходится на иглу: `игла / (игла + близнец)`. 1 — голова смотрит только на иглу, 0.5 — поровну, 0 — только на близнеца.
""")
code("""
select = {level: hard.selectivity(needle[level], rival[level]) for level in twin_levels}
ax = plots.rank_profile(select, order, top=60, ylabel="избирательность: доля иглы", title="Избирательность голов детекции")
ax.set_ylim(0, 1.0)
ax.axhline(0.5, color=style.THEME.axis, linewidth=1, zorder=1)
show(ax, "B_selectivity")

print("средняя избирательность лучших голов детекции")
display(hard.top_means(select, order))

rest_heads = order[60:]
looking = {level: (needle[level].ravel()[rest_heads] + rival[level].ravel()[rest_heads]) > 0.05 for level in twin_levels}
print("для сравнения — остальные головы, которые смотрят на иглу или близнеца (сумма массы выше 0.05): число и средняя избирательность")
display(pd.DataFrame({level: {"голов": int(looking[level].sum()), "избирательность": float(select[level].ravel()[rest_heads][looking[level]].mean())} for level in twin_levels}).round(3))
""")

md("""
### Игла и близнец по головам

Точка — голова: по горизонтали масса на игле, по вертикали — на самом сильном близнеце. Выделены 60 лучших голов детекции. Чем ниже точка под диагональю, тем избирательнее голова.
""")
code("""
best = np.zeros(total_heads, dtype=bool)
best[order[:60]] = True
limit = max(float(needle[level].max()) for level in twin_levels) * 1.02
axes = panels(len(twin_levels), 4.8, 5.0, sharey=True)
for ax, level in zip(axes, twin_levels):
    plots.score_scatter(needle[level], rival[level], mark=best, mark_label="60 лучших детекции", top=limit, xlabel="масса на игле",
                        ylabel="масса на самом сильном близнеце", ax=ax, title=level)
    if ax is not axes[0]:
        ax.set_ylabel("")
        ax.get_legend().remove()
show(axes, "B_needle_twin_scatter")
""")
code("""
# головы, которые больше всего смотрят на близнецов, на уровне с тремя близнецами рядом
LOOK = "B3"
most = np.argsort(-rival[LOOK].ravel(), kind="stable")[:20]
pd.DataFrame({"голова": [heads.name(i) for i in most], "место в детекции": [place[int(i)] for i in most],
              "скор копирования на уровне": scores[LOOK].ravel()[most], "масса на игле": needle[LOOK].ravel()[most],
              "масса на самом сильном близнеце": rival[LOOK].ravel()[most], "избирательность": select[LOOK].ravel()[most]}).round(3)
""")

md("""
### Копирование: из иглы и из близнецов

`copy_sentence` — число шагов, на которых голова смотрит на токен иглы и модель выдаёт именно его; `copy_competing` — то же для токенов близнецов. Оба делятся на длину ответа иглы. Близнец повторяет шаблон иглы, поэтому часть его токенов совпадает с токенами верного ответа (`is asked to`, `and`, `a`): ненулевой `copy_competing` на верном ответе означает, что голова взяла общий токен из близнеца, а не из иглы.
""")
code("""
copy_needle = hard.level_scores(heads, "copy_sentence", where=common)
copy_twin = hard.level_scores(heads, "copy_competing", where=common)

top_copy = max(float(copy_needle[level].ravel()[order[:60]].max()) for level in twin_levels)
axes = panels(len(twin_levels), 4.8, extra=1.6, sharey=True)
for ax, level in zip(axes, twin_levels):
    plots.rank_profile({"из иглы": copy_needle[level], "из близнецов": copy_twin[level]}, order, top=60, ylabel="скор копирования",
                       legend=ax is axes[-1], ax=ax, title=level)
    ax.set_ylim(0, top_copy * 1.05)
    if ax is not axes[0]:
        ax.set_ylabel("")
show(axes, "B_copy_needle_vs_twin")

print("копирование из иглы (copy_sentence): среднее у лучших голов детекции")
display(hard.top_means(copy_needle, order))
print("копирование из близнецов (copy_competing)")
display(hard.top_means({level: copy_twin[level] for level in twin_levels}, order))
print("доля копирования из близнецов в общем копировании, %")
display((hard.top_means({level: copy_twin[level] for level in twin_levels}, order)
         / (hard.top_means({level: copy_twin[level] for level in twin_levels}, order) + hard.top_means({level: copy_needle[level] for level in twin_levels}, order)) * 100).round(1))
""")

md("""
## 3.7. По семействам

Совпадает ли направление эффекта в каждом семействе: у 20 лучших голов детекции — скор копирования и избирательность по уровням.
""")
code("""
def family_mask(family):
    return common & (heads.samples["family"] == family).to_numpy()

family_top = pd.DataFrame({family: hard.top_means(hard.level_scores(heads, METRIC, where=family_mask(family)), order).loc["20 лучших"] for family in families}).T
print("средний скор копирования 20 лучших голов детекции")
display(family_top)
show(plots.group_bars(family_top, xlabel="семейство игл", ylabel="средний скор 20 лучших голов", title="20 лучших голов детекции: скор по семействам"), "B_family_top20")

rows = {}
for family in families:
    n = hard.level_scores(heads, "needle_sentence_mass", where=family_mask(family), levels=twin_levels)
    r = hard.level_scores(heads, "competing_max_mass", where=family_mask(family), levels=twin_levels)
    rows[family] = hard.top_means({level: hard.selectivity(n[level], r[level]) for level in twin_levels}, order).loc["20 лучших"]
family_select = pd.DataFrame(rows).T
print("средняя избирательность 20 лучших голов детекции")
display(family_select)
show(plots.group_bars(family_select, xlabel="семейство игл", ylabel="избирательность 20 лучших голов", title="20 лучших голов детекции: избирательность по семействам"), "B_family_selectivity")
""")

md("""
## 3.8. Появляются ли новые головы

Головы, которые на уровне входят в 20 лучших по скору копирования, но в детекции стояли за пределами первых 60.
""")
code("""
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

1. **Головы, найденные на простой задаче, сохраняют роль при близнецах.** Рейтинг относительно детекции: Спирмен 0.83–0.85, общих голов 17 из 20 и 35 из 40. Относительно A0 — 0.95 и 19 из 20. Новых голов нет.
2. **Без маски близнецы почти не мешают:** 99.5–100% верных ответов на всех уровнях, включая три близнеца рядом с иглой и запись с тем же ключом.
3. **Модель не путает иглу с близнецом даже под маской.** Ответов класса «путаница» 8 из 3163 на уровнях с близнецами; слова близнецов встречаются не чаще чем в 3.5% ответов. Маска обрезает ответ, а не подменяет его.
4. **Retrieval-головы на близнецов смотрят, но слабо:** на игле 0.33–0.37 массы внимания, на самом сильном близнеце 0.07–0.09. Избирательность 0.77–0.84 и падает с числом и близостью близнецов.
5. **Эффект близнецов под маской разный при 40 и при 60 головах** и сильно зависит от иглы — см. ниже.

## 4.2. Зависимости в ответах

### От маски

Все уровни ноутбука, включая A0:

| Маска | верно | частично | иное | путаница и смешано |
|---|---|---|---|---|
| без маски | 99.8 | 0 | 0 | 0.2 |
| 60 случайных | 99.0 | 0 | 0 | 1.0 |
| 40 лучших | 62.7 | 36.6 | 0.6 | 0.1 |
| 60 лучших | 49.6 | 35.2 | 14.9 | 0.3 |

- **Неверные ответы под маской лучших голов — это обрезка.** При 40 головах почти все они «частичные»; при 60 добавляется «иное». Фактов близнецов в них практически нет.
- **Случайная маска безобидна.** Единственное заметное отклонение — B4 (запись с тем же ключом): 96.5% верных и 3.5% ответов с фактами вставки, это 7 ответов.
- **При близнецах под маской остаётся больше дословных ответов, чем без них:** точных фраз 19–38% на B1–B4 против 2.5% на A0 при 40 головах. Причину эти данные не показывают.

### От уровня задачи

| Маска | A0 | B1 | B2 | B3 | B4 |
|---|---|---|---|---|---|
| без маски | 100 | 100 | 99.5 | 99.5 | 100 |
| 60 случайных | 100 | 99.0 | 99.5 | 100 | 96.5 |
| 40 лучших | 70.7 | 58.6 | 55.6 | 57.6 | 71.2 |
| 60 лучших | 29.8 | 41.9 | 48.5 | 58.1 | 69.7 |

- **При 40 головах близнецы вредят:** −12…−15 п.п. к A0 на B1–B3, независимо от их числа и расстояния до иглы. Запись с тем же ключом (B4) не вредит.
- **При 60 головах уровни с близнецами выше A0,** и тем выше, чем ближе и «похожее» вставка. Этот результат создаёт в основном `oslo` (85–98% верных). Без него B1 и B2 не лучше A0 (−9 и −1 п.п.), а B3 и B4 остаются выше (+12 и +27 п.п.).
- **Число и близость близнецов влияют на головы** (раздел 4.3), **но не на ответы без маски.**

### От глубины иглы

Верные ответы, %, все уровни вместе:

| Маска | 0 | 22 | 44 | 67 | 89 | 100 |
|---|---|---|---|---|---|---|
| без маски | 100 | 98.8 | 100 | 100 | 100 | 100 |
| 60 случайных | 100 | 98.2 | 99.4 | 98.2 | 99.4 | 98.8 |
| 40 лучших | 78.2 | 52.7 | 57.6 | 53.9 | 67.3 | 66.7 |
| 60 лучших | 72.7 | 43.0 | 46.7 | 44.8 | 43.6 | 46.7 |

- **Игла в самом начале контекста устойчивее всего:** 78% против 53–58% в середине при 40 головах, 73% против 43–47% при 60.
- **Конец контекста выигрывает меньше, чем на оси A:** при 40 головах 67% на глубинах 89 и 100, при 60 головах преимущества нет вовсе.
- **На глубине 0 меньше и обрезанных, и потерянных ответов:** «частично» 22% против 41–46% в середине при 40 головах; «иное» 4% против 16–18% при 60.
- **Преимущество глубины 0 под 60 головами несёт `cairo`** (95% против 22–38% на остальных глубинах); у `oslo` зависимости от глубины нет (71–87% везде).
- **Без маски и под случайной маской зависимости от глубины нет.**

### От длины контекста

| Маска | короткие (до 7k) | длинные (от 22k) | тренд по 11 длинам (Спирмен) |
|---|---|---|---|
| без маски | 100 | 99.5 | −0.40 |
| 60 случайных | 98.5 | 98.6 | −0.07 |
| 40 лучших | 69.6 | 58.6 | −0.80 |
| 60 лучших | 51.8 | 45.5 | −0.59 |

- **Длина контекста влияет только под маской лучших голов:** −11 п.п. при 40 головах, −6 при 60.
- **Без маски близнецы не делают длинный контекст труднее:** единственная точка ниже 100% — 97.8% на 28.5k.
- **Доля «иное» при 60 головах растёт с длиной** (6% на 1000 токенов, 11–23% на длинах от 10k), как и на оси A.

### От иглы

| Маска | `cairo` | `lisbon` | `oslo` |
|---|---|---|---|
| без маски | 99.7 | 100 | 99.7 |
| 60 случайных | 99.1 | 99.1 | 98.8 |
| 40 лучших | 49.7 | 86.7 | 51.8 |
| 60 лучших | 39.7 | 29.4 | 79.7 |

- **Порядок игл переворачивается между масками.** При 40 головах устойчивее всех `lisbon` (87%), при 60 — `oslo` (80%), а `lisbon` становится худшей (29%).
- **`oslo` под 60 головами при близнецах почти не страдает** (85–98% на B1–B4 против 32% на A0). Это самый сильный эффект в ноутбуке и он целиком принадлежит одной игле.
- **Без маски все три иглы ведут себя одинаково.**

## 4.3. Головы

- **Скор копирования снижается с числом и близостью близнецов:** у 20 лучших голов 0.68 (A0) → 0.65 (один далеко) → 0.62 (три случайно) → 0.61 (три рядом); запись с тем же ключом — 0.64. На дословных ответах числа те же, то есть это не артефакт формулировки.
- **Масса на игле снижается так же:** на всей игле 0.41 → 0.37 → 0.35 → 0.33 (B4 — 0.36).
- **Избирательность к игле высокая, но не полная:** 0.84 при одном близнеце далеко, 0.79 при трёх случайных, 0.77 при трёх рядом, 0.80 при записи с тем же ключом. У остальных голов, которые вообще смотрят на иглу или близнеца, она ниже: 0.66–0.75.
- **Из близнецов головы почти не копируют:** 7–10% от общего копирования; это общие с иглой токены шаблона.
- **Порядок голов устойчивее значений.** По 62 исходным головам Спирмен 0.91–0.92 на всех уровнях; Пирсон по 20 лучшим падает с 0.94 (A0) до 0.80 (B3) — значения лучших голов меняются неравномерно.
- **Больше всего на близнецов смотрят не лучшие головы детекции:** среди 20 голов с наибольшей массой на близнеце (уровень B3) только четыре из первых двадцати мест детекции.
- **По иглам направление одно:** избирательность падает от B1 к B3 во всех трёх семействах (`lisbon` 0.87 → 0.83, `cairo` 0.85 → 0.76, `oslo` 0.80 → 0.73).

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
- **При 60 головах картина зашумлена иглами:** шум не отличается от A0, а «выигрыш» близнецов создаёт одно семейство.

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
        NotebookClient(nb, timeout=1200, kernel_name="python3", resources={"metadata": {"path": HERE}}).execute()
    nbf.write(nb, os.path.join(HERE, "hard_B.ipynb"))
    print("written", os.path.join(HERE, "hard_B.ipynb"))
