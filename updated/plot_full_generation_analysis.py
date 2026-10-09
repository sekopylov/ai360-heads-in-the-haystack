"""Publication-style full-generation ROUGE dashboard for both masking runs."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

ROOT = Path(__file__).parent / "report-assets/qwen3-8b-three-corpora-mask"
OUT = ROOT / "full-generation-rouge/plots"
RANKS = ("multiset", "attention-mass")
CONDITIONS = ("baseline", "top4", "bottom4", "top16", "bottom16", "top64", "bottom64")
LABELS = ("Baseline", "Top 4", "Bottom 4", "Top 16", "Bottom 16", "Top 64", "Bottom 64")
CORPORA = ("paul-graham", "eugene-onegin", "hero-of-our-time")
CORPUS_LABELS = ("Paul Graham", "Евгений Онегин", "Герой нашего времени")
LENGTHS, DEPTHS = (8000, 16000, 32000, 48000), (15, 45, 75)
BLUE, ORANGE, GREEN, RED, GOLD = "#2563a6", "#d56737", "#278975", "#c94855", "#e1ad42"
BG, INK, GRID = "#fafaf8", "#263449", "#e2e7ec"
SCORE_CMAP = LinearSegmentedColormap.from_list("score", ["#f9e9e1", "#eab585", "#f1ead6", "#9ac7b8", "#237c72"])
FAIL_CMAP = LinearSegmentedColormap.from_list("fail", ["#f5f1ed", "#f3d4bb", "#e59d87", "#bc4355"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "figure.facecolor": BG, "axes.facecolor": BG,
                     "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK,
                     "axes.edgecolor": GRID, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.spines.left": False,
                     "axes.spines.bottom": False, "axes.titleweight": "bold",
                     "axes.titlepad": 14, "legend.frameon": False,
                     "svg.fonttype": "none", "savefig.facecolor": BG})


def key(row):
    return row["rank"], row["condition"], row.get("case_id", row.get("corpus")), row.get("context_length", row.get("length")), row.get("depth_percent", row.get("depth"))


def load():
    old = json.loads((ROOT / "analysis.json").read_text())["cases"]
    rescored = json.loads((ROOT / "full-generation-rouge/scores.json").read_text())["cases"]
    lookup = {(r["rank"], r["condition"], r["source"]): r for r in rescored}
    rows = []
    for r in old:
        s = lookup[(r["rank"], r["condition"], r["source"])]
        rows.append({**r, "final_answer_score": r["score"], "score": s["full_generation_recall"],
                     "precision": s["full_generation_precision"], "f1": s["full_generation_f1"]})
    assert len(rows) == 504 and len(lookup) == 504
    return rows


def group(rows, rank, condition):
    return [r for r in rows if r["rank"] == rank and r["condition"] == condition]


def frame(title, subtitle, *, width=14, height=7, nrows=1, ncols=2):
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), squeeze=False)
    fig.subplots_adjust(left=.08, right=.96, bottom=.19, top=.71, wspace=.28, hspace=.7)
    fig.text(.08, .945, title, fontsize=21, weight="bold", va="top")
    fig.text(.08, .882, subtitle, fontsize=11, color="#637086", va="top")
    return fig, axes


def save(fig, name, note):
    fig.text(.08, .045, note, fontsize=9, color="#637086", va="bottom")
    for extension in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{extension}", dpi=190)
    plt.close(fig)


def legend(fig, ax, ncol=3):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.073, .835), ncol=ncol, fontsize=10)


def heat(ax, matrix, *, vmax=100, cmap=SCORE_CMAP, decimals=0):
    im = ax.imshow(matrix, vmin=0, vmax=vmax, cmap=cmap, aspect="auto")
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            v = matrix[i][j]
            ax.text(j, i, f"{v:.{decimals}f}", ha="center", va="center",
                    fontsize=10, color="white" if v > vmax*.78 else INK)
    ax.set_xticks(np.arange(len(matrix[0]))-.5, minor=True)
    ax.set_yticks(np.arange(len(matrix))-.5, minor=True)
    ax.grid(which="minor", color=BG, linewidth=3)
    ax.tick_params(which="both", length=0)
    return im


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load()
    (OUT / "plot_data.json").write_text(json.dumps({"score_scope": "thinking_and_answer_no_prompt", "cases": rows}, ensure_ascii=False, indent=2))
    x = np.arange(7)
    fig, axes = frame("Извлечение с учётом мыслей", "ROUGE-1 recall по think + ответу · 36 одинаковых prompts на условие", ncols=1, height=7)
    ax = axes[0, 0]
    for j, rank in enumerate(RANKS):
        vals = [mean(r["score"] for r in group(rows, rank, c)) for c in CONDITIONS]
        bars = ax.bar(x + (j-.5)*.34, vals, .31, label=rank, color=(BLUE, ORANGE)[j], zorder=3)
        ax.bar_label(bars, fmt="%.1f", padding=6, fontsize=11, color=(BLUE, ORANGE)[j], weight="bold")
    ax.set(xticks=x, xticklabels=LABELS, ylim=(0, 112), ylabel="ROUGE-1 recall, %")
    ax.grid(axis="y", color=GRID, zorder=0); legend(fig, ax, 2)
    save(fig, "01_scores", "Полный сгенерированный текст. Входной контекст не включён; исходные answer-only оценки сохранены отдельно.")

    fig, axes = frame("Как заканчивается генерация", "Причины остановки не меняются при пересчёте ROUGE · число случаев из 36", height=7)
    for ax, rank in zip(axes[0], RANKS):
        bottom = np.zeros(7)
        for stage, label, color in [("eos", "Достигнут EOS", GREEN), ("answer_capped", "Обрыв ответа", GOLD),
                                    ("unfinished_think", "Обрыв think", RED)]:
            vals = np.array([sum(r["stage"] == stage for r in group(rows, rank, c)) for c in CONDITIONS])
            ax.bar(x, vals, .64, bottom=bottom, color=color, label=label, zorder=3)
            for i, n in enumerate(vals):
                if n: ax.text(i, bottom[i]+n/2, str(n), ha="center", va="center", fontsize=10,
                              color=INK if color==GOLD else "white", weight="bold")
            bottom += vals
        ax.set(title=rank, xticks=x, xticklabels=LABELS, ylim=(0, 37), ylabel="Случаи")
        ax.tick_params(axis="x", labelrotation=30); ax.grid(axis="y", color=GRID, zorder=0)
    legend(fig, axes[0, 0]); save(fig, "02_termination", "109 незавершённых think и 7 обрывов ответа. Лимит 2048 охватывает мысли и ответ вместе.")

    for field, name, title, subtitle, cmap, vmax in [
        ("score", "03_length_depth_scores", "Качество по длине и позиции needle", "Средний ROUGE-1 recall по think + ответу · каждая ячейка: 3 корпуса", SCORE_CMAP, 100),
        ("empty_answer", "04_length_depth_failures", "Где модель не выдаёт финальный ответ", "Пустой финальный ответ ≠ нулевой ROUGE по полному тексту · число случаев из 3", FAIL_CMAP, 3)]:
        fig, axes = frame(title, subtitle, nrows=2, ncols=7, width=19, height=10)
        fig.subplots_adjust(left=.075, right=.92, bottom=.14, top=.76, wspace=.23, hspace=.72)
        for a, rank in enumerate(RANKS):
            fig.text(.075, .805 if a==0 else .43, rank, fontsize=14, weight="bold", color=(BLUE, ORANGE)[a])
            for b, cond in enumerate(CONDITIONS):
                g = group(rows, rank, cond)
                matrix = [[(mean(r[field] for r in g if r["length"]==length and r["depth"]==depth) if field=="score"
                            else sum(r[field] for r in g if r["length"]==length and r["depth"]==depth)) for depth in DEPTHS] for length in LENGTHS]
                ax=axes[a,b]; im=heat(ax, matrix, vmax=vmax, cmap=cmap)
                ax.set(title=LABELS[b], xticks=range(3), xticklabels=["15%", "45%", "75%"],
                       yticks=range(4), yticklabels=["8k", "16k", "32k", "48k"] if b==0 else [""]*4)
                ax.tick_params(labelsize=10)
        bar=fig.add_axes([.94,.18,.012,.53]); fig.colorbar(im,cax=bar, ticks=[0,25,50,75,100] if field=="score" else [0,1,2,3])
        save(fig,name,"Один context seed 42. Шкала одинакова для обоих рейтингов; входные prompts попарно совпадают.")

    fig, axes = frame("Результат на каждом корпусе", "ROUGE-1 recall по think + ответу · 12 prompts на корпус и условие", nrows=2, ncols=1, height=9)
    fig.subplots_adjust(left=.23,right=.94,bottom=.15,top=.78,hspace=.65)
    for ax,rank in zip(axes[:,0], RANKS):
        matrix=[[mean(r["score"] for r in group(rows,rank,c) if r["corpus"]==corp) for c in CONDITIONS] for corp in CORPORA]
        heat(ax,matrix,decimals=1); ax.set(title=rank,xticks=x,xticklabels=LABELS,yticks=range(3),yticklabels=CORPUS_LABELS)
    save(fig,"05_corpora","По одной выдуманной needle на корпус. Это фиксированная небольшая выборка, не оценка обобщения на все тексты.")

    fig, axes = frame("Бюджет генерации", "Все 36 наблюдений показаны точками · линия внутри коробки: медиана", height=7)
    for ax,rank,color in zip(axes[0],RANKS,(BLUE,ORANGE)):
        data=[[r["tokens"] for r in group(rows,rank,c)] for c in CONDITIONS]
        ax.boxplot(data,positions=x,widths=.48,patch_artist=True,showfliers=False,
                   boxprops={"facecolor":color,"alpha":.18,"edgecolor":color},medianprops={"color":color,"linewidth":2},
                   whiskerprops={"color":color},capprops={"color":color})
        rng=np.random.default_rng(42)
        for i,v in enumerate(data): ax.scatter(i+rng.uniform(-.17,.17,len(v)),v,s=15,color=color,alpha=.55,zorder=3)
        ax.axhline(2048,color=RED,ls="--",lw=1.4,label="Лимит 2048")
        ax.set(title=rank,xticks=x,xticklabels=LABELS,ylim=(0,2170),ylabel="Выходные токены: think + ответ")
        ax.tick_params(axis="x",labelrotation=30);ax.grid(axis="y",color=GRID)
    legend(fig,axes[0,0]);save(fig,"06_tokens","Лимит достигнут в 116 генерациях; пересчёт текстовой оценки не устраняет зацикливание.")

    fig, axes = frame("Повторы и длина ответа", "Покрытие токенов повторными точными 32-граммами · каждый маркер: одна генерация", height=7)
    for ax,rank in zip(axes[0],RANKS):
        for stage,label,color in [("eos","EOS",GREEN),("answer_capped","Обрыв ответа",GOLD),("unfinished_think","Обрыв think",RED)]:
            g=[r for r in rows if r["rank"]==rank and r["stage"]==stage]
            ax.scatter([r["tokens"] for r in g],[100*r["repeat_fraction"] for r in g],s=26,alpha=.5,color=color,label=label,edgecolors="none")
        ax.set(title=rank,xlabel="Выходные токены",ylabel="Повторное покрытие, %",xlim=(0,2200),ylim=(-3,103));ax.grid(color=GRID,alpha=.7)
    legend(fig,axes[0,0]);save(fig,"07_repetition","Доля повторов — диагностика текста, не ROUGE и не доказательство нейронного механизма обрыва.")

    fig, axes = frame("Что модель успела найти внутри think", "Только незавершённые рассуждения · характерные слова needle в окне 60 слов", height=7)
    for ax,rank in zip(axes[0],RANKS):
        bottom=np.zeros(7)
        for level,label,color in [("all_details","Все детали",GREEN),("partial","Часть деталей",GOLD),("no_anchor","Нет характерного якоря",RED)]:
            vals=np.array([sum(r["stage"]=="unfinished_think" and r["evidence"]["level"]==level for r in group(rows,rank,c)) for c in CONDITIONS])
            ax.bar(x,vals,.64,bottom=bottom,color=color,label=label,zorder=3)
            for i,n in enumerate(vals):
                if n:ax.text(i,bottom[i]+n/2,str(n),ha="center",va="center",color=INK if color==GOLD else "white",weight="bold",fontsize=10)
            bottom+=vals
        ax.set(title=rank,xticks=x,xticklabels=LABELS,ylim=(0,37),ylabel="Случаи");ax.tick_params(axis="x",labelrotation=30);ax.grid(axis="y",color=GRID,zorder=0)
    legend(fig,axes[0,0]);save(fig,"08_evidence_in_unfinished_think","Лексическое присутствие не равно правильному утверждению: цитаты, гипотезы и отрицание тоже содержат слова ответа.")

    fig, axes = frame("Какие слои затрагивает вмешательство", "Число голов с uniform attention · индексы слоёв начинаются с нуля", height=7)
    for ax,rank in zip(axes[0],RANKS):
        for cond,label,color,ls in [("top4","Top 4",GREEN,":"),("top16","Top 16",GOLD,"--"),("top64","Top 64",BLUE,"-"),("bottom64","Bottom 64",ORANGE,"-")]:
            example=group(rows,rank,cond)[0];original=json.loads(Path(example["source"]).read_text())
            counts=Counter(l for l,h in original["experiment"]["blocked_heads"])
            ax.plot(range(36),[counts[i] for i in range(36)],color=color,ls=ls,lw=2,label=label)
        ax.set(title=rank,xlabel="Слой",ylabel="Головы с uniform attention",ylim=(-.5,33),xlim=(0,35));ax.grid(axis="y",color=GRID)
    legend(fig,axes[0,0],4);save(fig,"09_head_layers","Наборы bottom64 совпадают лишь по 6 головам; у multiset все 64 находятся в последних трёх слоях.")

    fig, axes = frame("Что изменилось после включения think в оценку", "Средняя оценка на тех же prompts · серый: только ответ, цветной: think + ответ", height=8)
    for ax,rank,color in zip(axes[0],RANKS,(BLUE,ORANGE)):
        before=[mean(r["final_answer_score"] for r in group(rows,rank,c)) for c in CONDITIONS]
        after=[mean(r["score"] for r in group(rows,rank,c)) for c in CONDITIONS]
        for i,(a,b) in enumerate(zip(before,after)):
            ax.plot([a,b],[i,i],color=color,lw=3,alpha=.5)
            ax.text(103,i,f"+{b-a:.1f}",va="center",fontsize=10,color=color)
        ax.scatter(before,x,s=55,color="#a1aaba",label="Только ответ",zorder=3)
        ax.scatter(after,x,s=65,color=color,label="Think + ответ",zorder=4)
        ax.set(title=rank,yticks=x,yticklabels=LABELS,xlim=(-3,117),xticks=[0,25,50,75,100],xlabel="ROUGE-1 recall, %")
        ax.invert_yaxis();ax.grid(axis="x",color=GRID)
    save(fig,"10_score_change","Прирост score — следствие расширения оцениваемого текста, не улучшение генерации или устранение ошибок.")

    titles=[("01_scores","Сводные оценки"),("02_termination","Завершение генерации"),("03_length_depth_scores","Оценки по длине и глубине"),
            ("04_length_depth_failures","Пустые финальные ответы"),("05_corpora","Отдельные корпуса"),("06_tokens","Токены"),
            ("07_repetition","Повторы"),("08_evidence_in_unfinished_think","Детали внутри think"),("09_head_layers","Распределение голов"),("10_score_change","Изменение оценки")]
    lines=["# Полная галерея: ROUGE по think + ответу","","Все графики оформлены заново. В графиках качества используется пересчитанный full-generation ROUGE-1 recall со stemming. Диагностики остановки, токенов, повторов и выбранных голов используют исходные наблюдения: пересчёт ROUGE не меняет их.","",
           "504 файла двух заданий; baseline продублирован, не является независимым повтором. Входной контекст исключён. Ненулевой score не означает правильный ответ: общие слова, гипотезы и отрицания засчитываются.",""]
    for name,title in titles:
        lines += [f"## {title}","",f"![{title}]({name}.png)","",f"[PNG]({name}.png) · [SVG]({name}.svg)",""]
    lines += ["## Воспроизведение","","Из `updated/`: `.venv/bin/python plot_full_generation_analysis.py`.","",
              "[Данные всех графиков](plot_data.json) · [Таблица пересчитанных оценок](../README.md). Исходная галерея answer-only не изменена."]
    (OUT/"README.md").write_text("\n".join(lines),encoding="utf-8")
    print(f"Saved 10 redesigned PNG/SVG charts and gallery: {OUT}")


if __name__=="__main__":
    main()
