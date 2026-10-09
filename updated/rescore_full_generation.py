"""Save separate ROUGE diagnostics over thinking + answer; never include prompt."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from statistics import mean

from rouge_score import rouge_scorer

RANKS = ("multiset", "attention-mass")
CONDITIONS = ("baseline", "top4", "bottom4", "top16", "bottom16", "top64", "bottom64")


def clean_generation(raw):
    # Drop Qwen control markers, retaining every word of thinking and answer.
    return re.sub(r"<\|[^<>]*\|>|</?think>", " ", raw).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path(__file__).parent / "datasphere-results")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "report-assets/qwen3-8b-three-corpora-mask/full-generation-rouge")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    rows, summaries = [], []
    for rank in RANKS:
        root = args.runs / f"qwen3-8b-three-corpora-48k-no-yarn-{rank}-mask/evaluation/context-42/evaluation"
        for condition in CONDITIONS:
            paths = sorted((root / condition / "results").glob("*_results.json"))
            if len(paths) != 36:
                raise ValueError(f"Expected 36 cases in {root / condition}, got {len(paths)}")
            group = []
            target = args.output / rank / condition
            target.mkdir(parents=True, exist_ok=True)
            for path in paths:
                original_bytes = path.read_bytes()
                original = json.loads(original_bytes)
                response = clean_generation(original["raw_model_response"])
                rouge = scorer.score(original["expected_answer"], response)["rouge1"]
                original_recall = scorer.score(original["expected_answer"], original["model_response"])["rouge1"].recall * 100
                if abs(original_recall - original["score"]) > 1e-6:
                    raise ValueError(f"Original score verification failed: {path}")
                row = dict(rank=rank, condition=condition, case_id=original["case_id"],
                           context_length=original["context_length"], depth_percent=original["depth_percent"],
                           final_answer_score=original["score"], full_generation_recall=100*rouge.recall,
                           full_generation_precision=100*rouge.precision, full_generation_f1=100*rouge.fmeasure,
                           finish_reason=original["finish_reason"], tokens=len(original["generated_token_ids"]),
                           prompt_sha256=original["prompt_sha256"], source=str(path.resolve()))
                if row["full_generation_recall"] + 1e-6 < original_recall:
                    raise ValueError(f"Full text recall lower than answer-only recall: {path}")
                group.append(row); rows.append(row)
                result = dict(row, metric="rouge1", use_stemmer=True,
                              evaluation_scope="full_generated_text_thinking_and_answer_no_prompt",
                              expected_answer=original["expected_answer"], evaluated_text=response)
                (target / path.name).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                if path.read_bytes() != original_bytes:
                    raise ValueError(f"Source unexpectedly changed: {path}")
            summary = dict(rank=rank, condition=condition, cases=len(group),
                           final_answer_mean=mean(r["final_answer_score"] for r in group),
                           full_generation_mean=mean(r["full_generation_recall"] for r in group),
                           full_generation_mean_precision=mean(r["full_generation_precision"] for r in group),
                           full_generation_mean_f1=mean(r["full_generation_f1"] for r in group),
                           zeros_before=sum(r["final_answer_score"] == 0 for r in group),
                           zeros_after=sum(r["full_generation_recall"] == 0 for r in group))
            summaries.append(summary)
    payload = dict(metric="rouge1", use_stemmer=True, score_scale="percent",
                   evaluation_scope="full_generated_text_thinking_and_answer_no_prompt",
                   notes="Diagnostic recall: quotes, speculation, negation and repeated correct words can score highly. Prompt is excluded to prevent needle leakage. Baseline is duplicated across the two jobs, not an independent replicate.",
                   summary=summaries, cases=rows)
    (args.output / "scores.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for filename, records in [("cases.csv", rows), ("summary.csv", summaries)]:
        with (args.output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    lines = ["# ROUGE по полной генерации: think + ответ", "",
             "Все 504 случая пересчитаны отдельно. Исходные JSON и оценки не изменены. Использован тот же ROUGE-1 recall со stemming, шкала 0–100. Из raw_model_response удалены только служебные маркеры Qwen; содержимое think сохранено целиком.", "",
             "**Prompt/исходный контекст не включён:** там уже находится needle, и оценка по нему тривиально завысила бы результат. «Полный текст» здесь означает только сгенерированные моделью мысли и ответ.", "",
             "| Рейтинг | Условие | Только ответ | Think + ответ | Нулей до → после |",
             "|---|---|---:|---:|---:|"]
    for s in summaries:
        lines.append(f"| {s['rank']} | {s['condition']} | {s['final_answer_mean']:.2f} | {s['full_generation_mean']:.2f} | {s['zeros_before']} → {s['zeros_after']} |")
    lines += ["", "![Сравнение оценок](comparison.png)", "",
              "Это дополнительная диагностика доступности слов ответа, а не замена основной метрики. Правильная цитата с последующим отрицанием тоже засчитывается; высокий recall не требует законченного ответа и почти не штрафует бесконечные повторы. Precision и F1 дополнительно сохранены, чтобы видеть объём постороннего текста.", "",
              "[Все оценки JSON](scores.json) · [Все случаи CSV](cases.csv) · [Сводка CSV](summary.csv). Индивидуальные пересчёты находятся в подкаталогах каждого рейтинга/условия.", "",
              "[Полная галерея: 10 переоформленных графиков](plots/README.md). Воспроизведение галереи: `.venv/bin/python plot_full_generation_analysis.py`.", "",
              "Воспроизведение из updated/: `.venv/bin/python rescore_full_generation.py`."]
    (args.output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    x = np.arange(len(CONDITIONS))
    for ax, rank in zip(axes, RANKS):
        selected = [s for s in summaries if s["rank"] == rank]
        for offset, field, label, color in [(-.18, "final_answer_mean", "Final answer only", "#94a3b8"),
                                           (.18, "full_generation_mean", "Thinking + answer", "#2563eb")]:
            bars = ax.bar(x + offset, [s[field] for s in selected], .36, label=label, color=color)
            ax.bar_label(bars, fmt="%.1f", fontsize=8)
        ax.set(title=rank, xticks=x, xticklabels=CONDITIONS, ylim=(0, 110), ylabel="Mean ROUGE-1 recall (%)")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=.15)
    axes[1].legend(loc="upper center", bbox_to_anchor=(.5, 1.18), ncol=2)
    for extension in ("png", "svg"):
        fig.savefig(args.output / f"comparison.{extension}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summaries, indent=2))
    print(f"Saved {len(rows)} rescored cases to {args.output}; zeros remaining: {sum(r['full_generation_recall']==0 for r in rows)}")


if __name__ == "__main__":
    main()
