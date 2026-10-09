"""Offline paired analysis of the two completed three-corpus Qwen3 mask runs.

No model loading or GPU required. Lexical evidence is NOT semantic correctness.
Usage: .venv/bin/python analyze_mask_runs.py
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rouge_score import rouge_scorer

RANKS = ("multiset", "attention-mass")
CONDITIONS = ("baseline", "top4", "bottom4", "top16", "bottom16", "top64", "bottom64")
CORPORA = ("paul-graham", "eugene-onegin", "hero-of-our-time")
LENGTHS = (8000, 16000, 32000, 48000)
DEPTHS = (15, 45, 75)
COLORS = ("#2563eb", "#e85d04")


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def evidence(text, corpus):
    """All distinctive facts within a 60-word window; no entailment claim."""
    tokens = words(text)
    patterns = {
        "paul-graham": ("velvet pelican 47", "silver", "thimble"),
        "eugene-onegin": ("green", "glass", "compass", "nearest", "violin"),
        "hero-of-our-time": ("porcelain", "beetle", "blue", "dots"),
    }[corpus]
    def matches(window):
        joined = " " + " ".join(window) + " "
        return all(" " + p + " " in joined for p in patterns) and (
            corpus != "hero-of-our-time" or "nine" in window or "9" in window)
    for start in range(len(tokens)):
        window = tokens[start:start + 60]
        if matches(window):
            return {"level": "all_details", "word_offset": start,
                    "snippet": " ".join(window)}
    anchors = {"paul-graham": ("velvet", "pelican", "thimble"),
               "eugene-onegin": ("compass",),
               "hero-of-our-time": ("beetle",)}[corpus]
    hit = next((i for i, w in enumerate(tokens) if w in anchors), None)
    return {"level": "partial" if hit is not None else "no_anchor",
            "word_offset": hit,
            "snippet": " ".join(tokens[max(0, hit - 15):hit + 45]) if hit is not None else ""}


def repetition(ids, raw):
    """Exact 32-token n-gram repetition; measures later repeated coverage."""
    width = 32
    positions = defaultdict(list)
    for i in range(max(0, len(ids) - width + 1)):
        positions[tuple(ids[i:i + width])].append(i)
    covered = set()
    for offsets in positions.values():
        if len(offsets) >= 2:
            for offset in offsets[1:]:
                covered.update(range(offset, offset + width))
    dominant = max(positions.values(), key=len, default=[])
    ws = words(raw)
    wc = Counter(tuple(ws[i:i + 12]) for i in range(max(0, len(ws) - 11)))
    phrase, count = wc.most_common(1)[0] if wc else ((), 0)
    fraction = len(covered) / max(1, len(ids))
    return {"repeat_fraction": fraction, "max_ngram_count": len(dominant),
            "dominant_first_token": dominant[0] if dominant else None,
            "dominant_second_token": dominant[1] if len(dominant) > 1 else None,
            "strong_repeat": len(dominant) >= 6 and fraction >= .15,
            "repeated_phrase": " ".join(phrase), "phrase_count": count}


def load_rows(root):
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    rows = []
    originals = {}
    for rank in RANKS:
        base = root / f"qwen3-8b-three-corpora-48k-no-yarn-{rank}-mask/evaluation/context-42/evaluation"
        for condition in CONDITIONS:
            paths = sorted((base / condition / "results").glob("*_results.json"))
            if len(paths) != 36:
                raise ValueError(f"Expected 36 cases: {base / condition}, got {len(paths)}")
            for path in paths:
                r = json.loads(path.read_text())
                raw, answer = r["raw_model_response"], r["model_response"]
                corpus = next(c for c in CORPORA if r["case_id"].startswith(c))
                key = (rank, condition, corpus, r["context_length"], int(r["depth_percent"]))
                if key in originals:
                    raise ValueError(f"Duplicate case {key}")
                originals[key] = r
                if abs(scorer.score(r["expected_answer"], answer)["rouge1"].recall * 100 - r["score"]) > 1e-6:
                    raise ValueError(f"Score mismatch {path}")
                closed = "</think>" in raw
                thinking = raw.split("</think>", 1)[0].replace("<think>", "")
                capped = r["finish_reason"] == "max_new_tokens"
                row = dict(rank=rank, condition=condition, corpus=corpus,
                           length=r["context_length"], depth=int(r["depth_percent"]),
                           score=r["score"], tokens=len(r["generated_token_ids"]),
                           duration=r["test_duration_seconds"], finish_reason=r["finish_reason"],
                           capped=capped, closed_think=closed, empty_answer=not answer.strip(),
                           stage="unfinished_think" if not closed and not answer.strip() else
                                 "answer_capped" if capped else "eos",
                           evidence=evidence(thinking, corpus), answer_evidence=evidence(answer, corpus),
                           raw_rouge_recall=scorer.score(r["expected_answer"], raw)["rouge1"].recall * 100,
                           prompt_sha256=r["prompt_sha256"], source=str(path.resolve()),
                           **repetition(r["generated_token_ids"], raw))
                if capped and row["tokens"] != 2048:
                    raise ValueError(f"Unexpected cap length {path}")
                rows.append(row)
    by_key = {(r["rank"], r["condition"], r["corpus"], r["length"], r["depth"]): r for r in rows}
    for r in rows:
        baseline = by_key[(r["rank"], "baseline", r["corpus"], r["length"], r["depth"])]
        other = by_key[(RANKS[1] if r["rank"] == RANKS[0] else RANKS[0], "baseline", r["corpus"], r["length"], r["depth"])]
        if r["prompt_sha256"] != baseline["prompt_sha256"] or r["prompt_sha256"] != other["prompt_sha256"]:
            raise ValueError("Unpaired prompt hashes")
        r["baseline_score"] = baseline["score"]
        r["baseline_tokens"] = baseline["tokens"]
        r["delta_score"] = r["score"] - baseline["score"]
    for corpus in CORPORA:
        for length in LENGTHS:
            for depth in DEPTHS:
                a = originals[(RANKS[0], "baseline", corpus, length, depth)]
                b = originals[(RANKS[1], "baseline", corpus, length, depth)]
                if a["generated_token_ids"] != b["generated_token_ids"]:
                    raise ValueError("Baseline token sequences differ between rankings")
    return rows, originals


def save(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=170, bbox_inches="tight")
    fig.savefig(out / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def plot(rows, originals, out):
    def group(rank, cond):
        return [r for r in rows if r["rank"] == rank and r["condition"] == cond]
    x = np.arange(len(CONDITIONS))
    fig, ax = plt.subplots(figsize=(11, 5))
    for j, rank in enumerate(RANKS):
        bars = ax.bar(x + (j - .5) * .36, [mean(r["score"] for r in group(rank, c)) for c in CONDITIONS],
                      .36, color=COLORS[j], label=rank)
        ax.bar_label(bars, fmt="%.1f", fontsize=8)
    ax.set(xticks=x, xticklabels=CONDITIONS, ylim=(0, 108), ylabel="Mean final-answer ROUGE-1 recall (%)",
           title="Qwen3-8B · 36 paired prompts per condition · A100 / legacy_uniform")
    ax.legend(); ax.grid(axis="y", alpha=.15)
    save(fig, out, "01_scores")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, rank in zip(axes, RANKS):
        bottom = np.zeros(7)
        for stage, color, label in [("eos", "#22a06b", "EOS"), ("answer_capped", "#e9a23b", "Answer started, capped"),
                                    ("unfinished_think", "#dc2626", "Think unfinished, capped")]:
            vals = np.array([sum(r["stage"] == stage for r in group(rank, c)) for c in CONDITIONS])
            ax.bar(x, vals, bottom=bottom, color=color, label=label)
            for i, n in enumerate(vals):
                if n: ax.text(i, bottom[i] + n / 2, str(n), ha="center", va="center", fontsize=9)
            bottom += vals
        ax.set(xticks=x, xticklabels=CONDITIONS, ylim=(0, 38), title=rank, ylabel="Cases / 36")
        ax.tick_params(axis="x", rotation=30)
    axes[1].legend(loc="upper center", bbox_to_anchor=(.5, 1.22), ncol=3, fontsize=8)
    save(fig, out, "02_termination")

    for field, name, cmap, vmax in [("score", "03_length_depth_scores", "viridis", 100),
                                     ("empty_answer", "04_length_depth_failures", "Reds", 3)]:
        fig, axes = plt.subplots(2, 7, figsize=(19, 7), constrained_layout=True)
        for j, rank in enumerate(RANKS):
            for i, cond in enumerate(CONDITIONS):
                matrix = [[(mean(r[field] for r in group(rank, cond) if r["length"] == length and r["depth"] == depth)
                             if field == "score" else sum(r[field] for r in group(rank, cond) if r["length"] == length and r["depth"] == depth))
                            for depth in DEPTHS] for length in LENGTHS]
                ax = axes[j, i]; im = ax.imshow(matrix, vmin=0, vmax=vmax, cmap=cmap, aspect="auto")
                for a in range(4):
                    for b in range(3):
                        ax.text(b, a, f"{matrix[a][b]:.0f}", ha="center", va="center", fontsize=9,
                                color="white" if matrix[a][b] > vmax * .55 else "black")
                ax.set(xticks=range(3), xticklabels=["15%", "45%", "75%"],
                       yticks=range(4), yticklabels=["8k", "16k", "32k", "48k"], title=cond)
                if i == 0: ax.set_ylabel(rank)
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.6,
                     label="Mean final-answer recall (%)" if field == "score" else "Empty answers / 3 corpora")
        save(fig, out, name)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
    for ax, rank in zip(axes, RANKS):
        matrix = [[mean(r["score"] for r in group(rank, c) if r["corpus"] == corpus) for c in CONDITIONS] for corpus in CORPORA]
        im = ax.imshow(matrix, vmin=0, vmax=100, cmap="viridis", aspect="auto")
        for a in range(3):
            for b in range(7): ax.text(b, a, f"{matrix[a][b]:.1f}", ha="center", va="center", fontsize=8, color="white" if matrix[a][b] < 70 else "black")
        ax.set(xticks=x, xticklabels=CONDITIONS, yticks=range(3), yticklabels=CORPORA, title=rank)
    fig.colorbar(im, ax=list(axes), shrink=.7, label="Recall (%)")
    save(fig, out, "05_corpora")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, rank in zip(axes, RANKS):
        data = [[r["tokens"] for r in group(rank, c)] for c in CONDITIONS]
        ax.boxplot(data, tick_labels=CONDITIONS, showfliers=False)
        for i, values in enumerate(data):
            ax.scatter(np.linspace(i + .8, i + 1.2, len(values)), values, s=12, alpha=.5)
        ax.axhline(2048, color="#dc2626", linestyle="--", label="2048-token hard cap")
        ax.set(title=rank, ylabel="Generated tokens (think + answer)", ylim=(0, 2150))
        ax.tick_params(axis="x", rotation=30)
    axes[1].legend(loc="upper center", bbox_to_anchor=(.5, 1.20), ncol=2, fontsize=8)
    save(fig, out, "06_tokens")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, rank in zip(axes, RANKS):
        for stage, color in [("eos", "#22a06b"), ("answer_capped", "#e9a23b"), ("unfinished_think", "#dc2626")]:
            selected = [r for r in rows if r["rank"] == rank and r["stage"] == stage]
            ax.scatter([r["tokens"] for r in selected], [100*r["repeat_fraction"] for r in selected],
                       s=22, alpha=.45, label=stage, color=color)
        ax.set(title=rank, xlabel="Generated tokens", ylabel="Repeated 32-token coverage (%)", ylim=(-3, 103))
    axes[1].legend(loc="upper center", bbox_to_anchor=(.5, 1.20), ncol=3, fontsize=8)
    save(fig, out, "07_repetition")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, rank in zip(axes, RANKS):
        bottom = np.zeros(7)
        for level, label, color in [("all_details", "All distinctive details (lexical)", "#22a06b"),
                                    ("partial", "Object/password fragment only", "#e9a23b"),
                                    ("no_anchor", "No distinctive anchor", "#dc2626")]:
            vals = np.array([sum(r["stage"] == "unfinished_think" and r["evidence"]["level"] == level for r in group(rank, c)) for c in CONDITIONS])
            ax.bar(x, vals, bottom=bottom, color=color, label=label)
            for i, n in enumerate(vals):
                if n: ax.text(i, bottom[i] + n/2, str(n), ha="center", va="center", fontsize=8)
            bottom += vals
        ax.set(title=rank, xticks=x, xticklabels=CONDITIONS, ylabel="Unfinished think cases", ylim=(0, 38))
        ax.tick_params(axis="x", rotation=30)
    axes[1].legend(loc="upper center", bbox_to_anchor=(.5, 1.23), ncol=1, fontsize=8)
    save(fig, out, "08_evidence_in_unfinished_think")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for j, (ax, rank) in enumerate(zip(axes, RANKS)):
        for cond, marker in [("top4", "o"), ("top16", "s"), ("top64", "^"), ("bottom64", "x")]:
            example = next(r for k, r in originals.items() if k[:2] == (rank, cond))
            counts = Counter(l for l, h in example["experiment"]["blocked_heads"])
            ax.plot(range(36), [counts[i] for i in range(36)], marker=marker, markersize=3, label=cond)
        ax.set(title=rank, xlabel="Layer (zero-based)", ylabel="Heads made uniform per layer")
        ax.legend(fontsize=8)
    save(fig, out, "09_head_layers")


def write_cases(rows, originals, out):
    lines = ["# Все обрывы по лимиту", "", "116 карточек; каждый случай проверяется по исходному JSON. Обрыв — достижение 2048 токенов, а не ошибка GPU.", "",
             "`all_details` означает совместное присутствие характерных слов в окне 60 слов, не семантическую проверку: возможны отрицание, цитата или гипотеза. Повторы — точные 32-токенные n-граммы. Полный текст сохранён в исходном JSON.", ""]
    labels = {"all_details": "все характерные детали (лексически)", "partial": "часть деталей", "no_anchor": "нет характерного объекта/фрагмента пароля"}
    for n, r in enumerate([r for r in rows if r["capped"]], 1):
        raw = originals[(r["rank"], r["condition"], r["corpus"], r["length"], r["depth"])]
        baseline = originals[(r["rank"], "baseline", r["corpus"], r["length"], r["depth"])]
        if r['stage'] == 'answer_capped':
            diagnosis = "Think завершён, но ответ повторяется и EOS не достигнут до лимита. Это не обрыв перехода из think; проблема продолжается в финальном ответе."
        elif r['evidence']['level'] == 'all_details':
            diagnosis = "Характерные детали needle уже появились внутри think, затем рассуждение повторяется без перехода к финальному ответу. Полная потеря доступности факта не объясняет этот случай; утверждение может сопровождаться сомнением или отрицанием."
        elif r['evidence']['level'] == 'partial':
            diagnosis = "В think есть фрагмент нужного объекта/пароля, но полного набора деталей лексический тест не обнаружил. Вместе с незавершённым think наблюдается потеря/искажение деталей или бесконечная перепроверка."
        else:
            diagnosis = "В think не найден характерный объект/фрагмент пароля. Текст уходит в общие формулировки, другой объект или поиск по нерелевантным цитатам; это не чистый случай «нашёл факт, но не закончил». Отсутствие слов не доказывает отсутствие внутреннего представления факта."
        special = {
            ('attention-mass', 'top16', 'hero-of-our-time', 32000, 75): "Ручная проверка: есть полный смысловой вариант «porcelain beetle with nine blue spots», затем «No, that's not in the text». Строгий словарный тест dots его считает partial: это отрицание/перепроверка почти верно восстановленного факта, а не отсутствие деталей.",
            ('attention-mass', 'bottom64', 'hero-of-our-time', 48000, 15): "Ручная проверка: цикл повторяет правильный полный объект и отвергает его: «No, that's not it». Доступ к формулировке есть, принятия/завершения нет.",
            ('attention-mass', 'bottom64', 'paul-graham', 16000, 45): "Ручная проверка: в начале правильный velvet-pelican-47, в хвосте он мутирует в velvet-pelvic-47. Это пример деградации детали во время цикла.",
            ('attention-mass', 'bottom64', 'paul-graham', 48000, 15): "Ручная проверка: полный пароль есть в think, но финальный повторяющийся ответ оставляет только место хранения, без самого пароля.",
            ('attention-mass', 'top64', 'eugene-onegin', 32000, 15): "Ручная проверка: финальный ответ утверждает snow globe вместо compass. ROUGE recall 36.36 не означает правильный ответ.",
            ('multiset', 'top64', 'paul-graham', 32000, 45): "Ручная проверка: ответ выдумывает пароль emergency, бумагу и safe place. ROUGE recall 45.45 обеспечен общими словами, не извлечением needle.",
            ('multiset', 'top4', 'eugene-onegin', 48000, 75): "Повторные цитаты есть (48.2% покрытия), но одна 32-грамма встречается лишь 5 раз. Поэтому строгий strong_repeat=False; это не отсутствие повторов. По тексту продолжается безуспешный поиск среди стихотворных цитат.",
        }.get((r['rank'], r['condition'], r['corpus'], r['length'], r['depth']))
        lines += [f"## {n:03d}. {r['rank']} / {r['condition']} / {r['corpus']} / {r['length']} / {r['depth']}%", "",
                  f"[Исходный JSON]({r['source']})", "",
                  f"- Причина остановки: `{r['finish_reason']}`, {r['tokens']} токенов. Стадия: `{r['stage']}`.",
                  f"- Финальная оценка: {r['score']:.2f}; baseline: {r['baseline_score']:.2f}, {r['baseline_tokens']} токенов, `{baseline['finish_reason']}`. SHA256 prompt совпадает.",
                  f"- Детали в think: {labels[r['evidence']['level']]}; детали в ответе: {labels[r['answer_evidence']['level']] }.",
                  f"- Повторы: {r['repeat_fraction']:.1%} токенов покрыты повторными 32-граммами; максимум {r['max_ngram_count']} вхождений одной 32-граммы. Признак сильных повторов: {r['strong_repeat']}.",
                  f"- Доминирующая 32-грамма: первое вхождение с токена {r['dominant_first_token']}, второе с {r['dominant_second_token']} (не точная граница начала цикла).", "",
                  f"Разбор: {diagnosis}" + (f" {special}" if special else ""), "",
                  "Начало генерации:", "", "```text", raw['raw_model_response'][:1100], "```", ""]
        if r['evidence']['snippet']:
            lines += ["Окно с деталями needle (нормализованные слова):", "", "```text", r['evidence']['snippet'], "```", ""]
        lines += [f"Наиболее частая 12-словная фраза ({r['phrase_count']} вхождений):", "", "> " + r['repeated_phrase'], "",
                  "Конец генерации:", "", "```text", raw['raw_model_response'][-750:], "```", "",
                  "Ответ baseline:", "", "```text", baseline['model_response'] or "[нет финального ответа]", "```", ""]
    (out / "TRUNCATION_CASES.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path(__file__).parent / "datasphere-results")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "report-assets/qwen3-8b-three-corpora-mask")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, originals = load_rows(args.runs)
    summaries = []
    for rank in RANKS:
        for condition in CONDITIONS:
            selected = [r for r in rows if r["rank"] == rank and r["condition"] == condition]
            summaries.append(dict(rank=rank, condition=condition, mean_score=mean(r["score"] for r in selected),
                                  cases=len(selected), capped=sum(r["capped"] for r in selected),
                                  unfinished=sum(r["stage"] == "unfinished_think" for r in selected),
                                  strong_repeat=sum(r["strong_repeat"] for r in selected),
                                  mean_tokens=mean(r["tokens"] for r in selected),
                                  duration_seconds=sum(r["duration"] for r in selected),
                                  evidence_unfinished=dict(Counter(r["evidence"]["level"] for r in selected if r["stage"] == "unfinished_think"))))
    payload = dict(notes="Lexical evidence != semantic correctness; hard cap includes thinking and answer. Paired hashes and baseline tokens checked; stored scores recalculated.",
                   summary=summaries, cases=rows)
    (args.output / "analysis.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    flat = [{**{k: v for k, v in r.items() if k not in ('evidence', 'answer_evidence')},
             'think_evidence': r['evidence']['level'], 'answer_evidence': r['answer_evidence']['level']} for r in rows]
    with (args.output / "cases.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    write_cases(rows, originals, args.output)
    plot(rows, originals, args.output)
    print(json.dumps(summaries, indent=2))
    print(f"Saved analysis, 116 case cards and 9 plots to {args.output}")


if __name__ == "__main__":
    main()
