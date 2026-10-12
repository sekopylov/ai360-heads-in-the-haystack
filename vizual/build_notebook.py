"""
Builds vizual/paper_task.ipynb from the cells below and runs it, so the notebook is stored with its figures.

python vizual/build_notebook.py            # build and run
python vizual/build_notebook.py --no-run   # build only

The notebook can be edited by hand as usual; this script is for remaking it from scratch.
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
# Retrieval-головы Qwen3-8B: задача из статьи

Метрики и графики по прогонам на задаче «игла в стоге сена» из статьи [Retrieval Head Mechanistically Explains Long-Context Factuality](https://arxiv.org/html/2404.15574v1).

Ноутбук читает только готовые результаты `rh.run` / `rh.metrics` (`samples.jsonl`, `heads/*.npz`, `spool/run.json`), модель не нужна.

| Раздел | Прогон | Что показывает | Рисунок статьи |
|---|---|---|---|
| 1 | детекция | качество без маски; распределение голов по скору; скор и частота активации; карта голов | 3, 4, 5 |
| 2 | маскирование | качество от числа отключённых голов по ROUGE-1, ROUGE-2, ROUGE-3 и по критериям на них; карты «длина × глубина»; типы ответов; формулировки ответов (перефразы, замены, формы слов) | 1, 7, 9 |

**Ровные шкалы.** Сетка прогонов не круглая: длины 1000, 2526, 4053, …, глубины 0, 11, 22, … Данные стоят на своих настоящих местах, а деления осей — на круглых значениях (5k, 10k, …; 0, 20, …).

Код графиков — в `vizual/plots.py`, чтение данных — в `vizual/data.py`, цвета и размеры — в `vizual/style.py`.
""")

md("## 0. Настройки")
code("""
import os, sys
sys.path.insert(0, os.path.abspath(".."))   # корень репозитория: ноутбук лежит в vizual/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vizual import data, paraphrase, plots, style

DETECT = "results/new/qwen3_detect"        # детекция: внимание сохранено, есть скоры голов
MASK = "results/new/qwen3_mask/*"          # перебор масок по рейтингу copy_count

METRIC = "copy_recall"    # скор головы: copy_recall — как в статье (доля скопированных токенов иглы), copy_count — как в коде авторов
THRESHOLD = 0.1           # порог retrieval-головы из статьи
LIMIT = None              # None или 256 — целые ответы (предел прогонов — 256 токенов); 50, 64, 128 — обрезка, для ROUGE-2/3 нужен пересчёт rh.metrics --rescore
SAVE = False              # сохранять рисунки в vizual/figures/

# Внешний вид: всё меняется здесь. Примеры:
#   style.use(scale=1.3, dpi=160)                        крупнее, для слайдов
#   style.use(colors={"top": "#4a3aa7"})                  другой цвет серии
#   style.use(cmap_score="RdYlGn", cmap_heads="viridis")  другая гамма карт (любое имя matplotlib; по умолчанию красная)
#   style.use(font=13)                                     крупнее подписи
style.use(scale=1.0, dpi=110)

def show(ax, name):
    if SAVE:
        plots.save(np.ravel(ax)[0].figure, name)
    plt.show()
""")

md("""
## 1. Детекция голов

3 иглы × 20 длин × 10 глубин = 600 примеров, без маски. Скор головы считается по успешным примерам (ROUGE-1 recall > 50), как в статье.
""")
code("""
heads = data.Heads(DETECT)
detect = heads.samples
score = heads.score(METRIC)          # [слой, голова]
ranking = heads.ranking(METRIC)
total_heads = score.size

print(f"примеров: {len(detect)}, успешных: {detect['success'].sum()} ({detect['success'].mean():.1%}), средний ROUGE-1 recall: {detect['rouge1_recall'].mean():.1f}")
print(f"ROUGE-2 recall: {detect['rouge2_recall'].mean():.1f}, ROUGE-3 recall: {detect['rouge3_recall'].mean():.1f}; "
      f"успех по ROUGE-2: {detect['success2'].mean():.1%}, по ROUGE-3: {detect['success3'].mean():.1%}, точная фраза: {detect['exact'].mean():.1%}")
print(f"голов: {total_heads} ({len(heads.layers)} слоёв × {heads.n_heads}); метрики голов: {[m for m in heads.metrics if '@' not in m]}")
""")

md("""
### 1.1. Качество без маски

Средний ROUGE-1 recall по длине контекста и глубине иглы.
""")
code("""
show(plots.niah_heatmap(detect, title="Детекция, без маски"), "detect_niah")
""")
code("""
# по иглам: успех и ROUGE
detect.groupby("needle_idx").agg(примеров=("id", "size"), rouge1=("rouge1_recall", "mean"), rouge2=("rouge2_recall", "mean"), rouge3=("rouge3_recall", "mean"),
                                 успех1=("success", "mean"), успех2=("success2", "mean"), успех3=("success3", "mean"), точная_фраза=("exact", "mean"),
                                 токенов=("n_tokens", "mean")).round(3)
""")

md("""
### 1.2. Распределение голов по скору (рис. 3 статьи)

В статье 3–6% голов имеют скор выше 0.1, 45–73% — ровно 0.
""")
code("""
fig, axes = plt.subplots(1, 2, figsize=style.size(14, 4.4), gridspec_kw={"width_ratios": [1, 1.2]})
plots.score_distribution(score, ax=axes[0], title="Распределение голов по retrieval score")
plots.score_pie(score, ax=axes[1], title="Доли голов")
show(axes, "score_distribution")
above = int((score > THRESHOLD).sum())
print(f"выше порога {THRESHOLD}: {above} из {total_heads} голов ({above / total_heads:.1%}); ровно 0: {(score == 0).mean():.1%}")
""")

md("""
### 1.3. Скор и частота активации (рис. 4 статьи)

Частота активации — доля примеров, в которых голова скопировала хотя бы один токен иглы. Головы упорядочены по скору.
""")
code("""
show(plots.score_vs_frequency(ranking, top=100, threshold=THRESHOLD, title="Скор и частота активации, 100 лучших голов"), "score_vs_frequency")
always = int((ranking["frequency"] == 1).sum())
print(f"активны во всех успешных примерах: {always} голов; выше порога {THRESHOLD}: {(ranking['score'] > THRESHOLD).sum()}")
""")

md("### 1.4. Где находятся головы (рис. 5 статьи)")
code("""
show(plots.head_heatmap(score, heads.layers, title="Retrieval score каждой головы"), "head_heatmap")
show(plots.layer_profile(score, heads.layers, THRESHOLD, title="Retrieval-головы по слоям"), "layer_profile")
show(plots.top_heads(ranking, k=20, threshold=THRESHOLD, title="20 лучших голов"), "top_heads")
""")
code("""
ranking.head(20).round(3)
""")

md("""
## 2. Маскирование

Другая игла (про Сан-Франциско), 20 длин × 10 глубин = 200 примеров на прогон. Головы отключаются по рейтингу детекции (`copy_count`): лучшие, случайные россыпью, случайные целыми группами.

**Метрики ответа.** ROUGE-n recall — доля n-грамм эталона, которые есть в ответе (со стеммингом, в процентах):

| Метрика | Что считается | В эталоне этой задачи |
|---|---|---|
| ROUGE-1 recall | отдельные слова, порядок не важен — метрика статьи | 12 слов |
| ROUGE-2 recall | пары соседних слов (двуграммы) | 11 двуграмм |
| ROUGE-3 recall | тройки соседних слов (триграммы) | 10 триграмм |

Эталон: `eat a sandwich and sit in Dolores Park on a sunny day`. Двуграммы и триграммы требуют, чтобы слова стояли рядом и в том же порядке, поэтому случайные совпадения служебных слов (`a`, `and`, `in`, `on`) их не набирают.

**Критерии на этих метриках** (ответ либо проходит, либо нет):

| Критерий | Условие | На этой задаче |
|---|---|---|
| успех по ROUGE-1 | ROUGE-1 recall > 50 — критерий статьи | не меньше 7 слов из 12 |
| успех по ROUGE-2 | ROUGE-2 recall > 50 | не меньше 6 двуграмм из 11 |
| успех по ROUGE-3 | ROUGE-3 recall > 50 | не меньше 6 триграмм из 10 |
| точная фраза | эталон стоит в ответе дословно и подряд | все 12 слов подряд |

В таблице ниже: `rouge1`–`rouge3` — средний recall, `success`, `success2`, `success3`, `exact` — доля ответов, прошедших критерий, в процентах.
""")
code("""
sweep = data.load_sweep(MASK, limit=LIMIT)
table = data.summary(sweep)
table
""")

md("""
### 2.1. Качество от числа отключённых голов (рис. 7 статьи)

Полоса вокруг линии — 95% интервал среднего по 200 примерам.
""")
code("""
show(plots.mask_curve(sweep, "rouge1_recall", total_heads=total_heads, threshold=50, title="Качество от числа отключённых голов"), "mask_curve_rouge")
""")

md("""
#### Слова, двуграммы, триграммы

Средний recall по трём метрикам. Горизонтальная линия — порог критерия (50).
""")
code("""
fig, axes = plt.subplots(1, 3, figsize=style.size(17, 4.4), sharey=True)
for ax, n in zip(axes, (1, 2, 3)):
    plots.mask_curve(sweep, f"rouge{n}_recall", threshold=50, ax=ax, title=f"ROUGE-{n} recall")
show(axes, "mask_curve_ngrams")
""")

md("""
#### Критерии

Доля ответов, прошедших критерий: успех по ROUGE-1, ROUGE-2, ROUGE-3. Затем все критерии на одном рисунке для маски лучших голов — от самого мягкого к самому строгому.
""")
code("""
fig, axes = plt.subplots(1, 3, figsize=style.size(17, 4.4), sharey=True)
for ax, n in zip(axes, (1, 2, 3)):
    plots.mask_curve(sweep, data.success_column(n), ax=ax, title=f"Успех по ROUGE-{n} (recall > 50)")
show(axes, "mask_curve_criteria")

show(plots.criteria_curve(sweep, "top", title="Критерии успеха: маска лучших голов"), "criteria_top")
show(plots.criteria_curve(sweep, "random", title="Критерии успеха: маска случайных голов"), "criteria_random")

fig, axes = plt.subplots(1, 2, figsize=style.size(12, 4.4))
plots.mask_curve(sweep, "exact", ax=axes[0], title="Точная фраза")
plots.mask_curve(sweep, "n_tokens", ax=axes[1], title="Длина ответа")
show(axes, "mask_curve_other")
""")
code("""
# числом: средний recall и доля ответов, прошедших критерий (%), по видам маски
names = {"rouge1_recall": "ROUGE-1", "rouge2_recall": "ROUGE-2", "rouge3_recall": "ROUGE-3",
         "success": "ROUGE-1 > 50", "success2": "ROUGE-2 > 50", "success3": "ROUGE-3 > 50", "exact": "точная фраза"}
numbers = sweep[sweep["mask"] != "none"].assign(маска=lambda d: d["mask"].map(style.LABELS)).groupby(["маска", "heads"])[list(names)].mean()
numbers[list(names)[3:]] *= 100
numbers.round(1).rename(columns=names)
""")

md("""
### 2.2. Карты «длина × глубина» (рис. 1 статьи)

Число в заголовке панели — среднее значение метрики по прогону. Сначала ROUGE-1, затем те же карты по двуграммам и триграммам.
""")
code("""
counts = sorted(sweep.loc[sweep["mask"] == "top", "heads"].unique())
show(plots.niah_grid(sweep, {"без маски": "none"} | {f"лучшие {k}": f"top{k}" for k in counts}, columns=4,
                     title="Маска лучших голов, ROUGE-1 recall"), "niah_top")
for n in (2, 3):
    show(plots.niah_grid(sweep, {"без маски": "none"} | {f"лучшие {k}": f"top{k}" for k in counts}, value=f"rouge{n}_recall", columns=4,
                         title=f"Маска лучших голов, ROUGE-{n} recall"), f"niah_top_rouge{n}")
""")

md("""
#### Случайные головы

Те же карты для масок случайных голов (россыпью) по трём метрикам, затем сравнение трёх видов маски при наибольшем числе голов.
""")
code("""
for n in (1, 2, 3):
    show(plots.niah_grid(sweep, {"без маски": "none"} | {f"случайные {k}": f"random{k}" for k in counts}, value=f"rouge{n}_recall", columns=4,
                         title=f"Маска случайных голов, ROUGE-{n} recall"), f"niah_random_rouge{n}")
most = counts[-1]
for n in (1, 2, 3):
    show(plots.niah_grid(sweep, {f"лучшие {most}": f"top{most}", f"случайные {most}": f"random{most}", f"случайные группы {most}": f"groups{most}"},
                         value=f"rouge{n}_recall", title=f"{most} голов: лучшие и случайные, ROUGE-{n} recall"), f"niah_top_vs_random_rouge{n}")
""")

md("""
### 2.3. Зависимость от длины и глубины

Точки стоят на настоящих значениях сетки, в каждой 10–20 примеров.
""")
code("""
for n in (1, 2, 3):
    show(plots.score_by(sweep, "context_length", kind="top", value=f"rouge{n}_recall", title=f"По длине контекста, ROUGE-{n}"), f"by_length_rouge{n}")
for n in (1, 2, 3):
    show(plots.score_by(sweep, "depth_percent", kind="top", value=f"rouge{n}_recall", title=f"По глубине иглы, ROUGE-{n}"), f"by_depth_rouge{n}")
""")

md("""
### 2.4. Что происходит с ответами (по мотивам рис. 9 статьи)

Каждый ответ относится ровно к одному типу. Условия проверяются сверху вниз, ответ получает первый подошедший тип. Типы строятся по одной из трёх метрик — ROUGE-1, ROUGE-2 или ROUGE-3; правило одно и то же, меняется только единица счёта (слова, двуграммы, триграммы).

| Тип | Критерий (n = 1, 2, 3) | Что это значит |
|---|---|---|
| точная фраза | эталон стоит в ответе дословно и подряд; регистр и пунктуация не учитываются | модель списала ответ из иглы целиком |
| успех без точной фразы | дословной фразы нет, но ROUGE-n recall > 50 | больше половины n-грамм эталона есть, но фраза оборвана, переставлена или пересказана |
| неполный ответ | ROUGE-n recall от 0 до 50 | найдена только часть ответа — «incomplete retrieval» статьи |
| нет совпадений | ROUGE-n recall = 0 | ни одной n-граммы эталона |

По словам (n = 1) в «неполный ответ» попадает и ответ не по игле, в котором случайно совпали служебные слова эталона, поэтому тип «нет совпадений» там почти всегда пуст. По двуграммам и триграммам такие ответы уходят в «нет совпадений».

Ниже — критерии с числом ответов и примером из одного прогона, для каждой из трёх метрик.
""")
code("""
RUN = "top80"   # прогон, из которого берутся примеры
part = sweep[sweep["run"] == RUN]
print("эталон:", part["reference"].iloc[0])

def type_table(n):
    column, rows = data.type_column(n), []
    for kind in data.ANSWER_TYPES:
        found = part[part[column] == kind]
        row = found.loc[found["n_tokens"].idxmin()] if len(found) else None   # самый короткий ответ этого типа
        rows.append({"метрика": f"ROUGE-{n}", "тип": plots.TYPE_LABELS[kind], "критерий": data.type_criteria(n)[kind], f"ответов в {RUN}": len(found),
                     "recall примера": round(row[f"rouge{n}_recall"], 1) if row is not None else "—", "пример ответа": row["response"] if row is not None else "—"})
    return pd.DataFrame(rows)

pd.set_option("display.max_colwidth", 400)
pd.concat([type_table(n) for n in (1, 2, 3)]).set_index(["метрика", "тип"])
""")
code("""
for n in (1, 2, 3):
    show(plots.answer_types(sweep, "top", n=n, title=f"Типы ответов по ROUGE-{n}: маска лучших голов"), f"answer_types_top_rouge{n}")
for n in (1, 2, 3):
    show(plots.answer_types(sweep, "random", n=n, title=f"Типы ответов по ROUGE-{n}: маска случайных голов"), f"answer_types_random_rouge{n}")
""")

md("""
### 2.5. Перефразы, замены и формы слов

ROUGE и точная фраза сравнивают строки. Здесь ответ разбирается по смыслу: эталон разбит на четыре факта, и для каждого факта смотрим, как он записан в ответе.

| Уровень | Что это | Пример для «eat a sandwich» |
|---|---|---|
| дословно | формулировка эталона | `eat a sandwich` |
| другая форма слова | те же слова в другой форме | `eating a sandwich` |
| синоним, смысл сохранён | другие слова с тем же смыслом | `have a sandwich` |
| потеряна деталь | факт назван без существенной детали | `enjoy a meal` без сэндвича; `the park` без названия |
| нет | факта в ответе нет | — |

Факты эталона: `eat a sandwich`, `sit`, `Dolores Park`, `sunny day`. Списки форм и замен лежат в `vizual/paraphrase.py` (`FACTS`). Они собраны вручную по ответам этих прогонов: это перечень того, что модель действительно писала, а не словарь синонимов. Для другой иглы нужен свой список.

По формулировкам фактов ответ целиком получает одну из шести структур:

| Структура | Условие |
|---|---|
| точная фраза | эталон стоит в ответе дословно и подряд |
| все части дословно, другой порядок или вставка | все четыре факта дословно, но фраза не совпадает: части переставлены или между ними вставлены слова |
| все части, есть форма слова или синоним | все четыре факта сохранены, хотя бы один не дословно |
| не все части, остальные дословно | часть фактов пропала, оставшиеся дословно — обрезанный ответ |
| не все части, есть форма, синоним или потеря детали | часть фактов пропала или потеряла деталь |
| ни одной части | ответ не по игле |
""")
code("""
worded = paraphrase.annotate(sweep)
facts = list(paraphrase.FACTS[worded["reference"].iloc[0]])
changed = worded[~worded["exact"]]
print(f"ответов: {len(worded)}, без точной фразы: {len(changed)}")
print(f"  с формой слова хотя бы в одном факте: {(changed[facts] == 'form').any(axis=1).sum()}")
print(f"  с синонимом хотя бы в одном факте: {(changed[facts] == 'paraphrase').any(axis=1).sum()}")
print(f"  с потерей детали хотя бы в одном факте: {(changed[facts] == 'reduced').any(axis=1).sum()}")

# что именно написано вместо формулировки эталона
paraphrase.wordings(worded)
""")

md("""
#### Как строковые метрики оценивают каждую структуру

По всем прогонам маскирования. `фактов` — сколько из четырёх фактов сохранено (дословно, формой слова или синонимом); столбцы «успех» — доля ответов, прошедших критерий.
""")
code("""
by_structure = worded.groupby("structure").agg(ответов=("id", "size"), фактов=("kept_facts", "mean"), rouge1=("rouge1_recall", "mean"),
                                               rouge2=("rouge2_recall", "mean"), rouge3=("rouge3_recall", "mean"), успех1=("success", "mean"),
                                               успех2=("success2", "mean"), успех3=("success3", "mean")).reindex(paraphrase.STRUCTURES)
by_structure[["успех1", "успех2", "успех3"]] *= 100
by_structure.round(1).rename(index=paraphrase.STRUCTURE_LABELS)
""")
code("""
for kind in ("top", "random"):
    ax = plt.subplots(figsize=style.size(11.5, 4.2))[1]   # шире обычного: у легенды длинные подписи
    show(plots.shares(worded, "structure", paraphrase.STRUCTURES, paraphrase.STRUCTURE_LABELS, kind, ax=ax,
                      title=f"Структура ответа: маска «{style.LABELS[kind]}»"), f"structure_{kind}")
show(plots.fact_wordings(worded, facts, paraphrase.LEVELS, paraphrase.LEVEL_LABELS, "top", title="Как записан каждый факт: маска лучших голов"), "fact_wordings_top")
""")
code("""
# примеры: по два самых коротких ответа каждой структуры
pd.set_option("display.max_colwidth", 300)
examples = changed.sort_values("n_tokens").groupby("structure").head(2)
examples.assign(структура=examples["structure"].map(paraphrase.STRUCTURE_LABELS)).set_index("структура").loc[
    [paraphrase.STRUCTURE_LABELS[k] for k in paraphrase.STRUCTURES if k in set(examples["structure"])],
    ["run", "rouge1_recall", "rouge2_recall", "rouge3_recall", "response"]].round(0)
""")

if __name__ == "__main__":
    nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
    if "--no-run" not in sys.argv:
        from nbclient import NotebookClient
        NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": HERE}}).execute()
    nbf.write(nb, os.path.join(HERE, "paper_task.ipynb"))
    print("written", os.path.join(HERE, "paper_task.ipynb"))
