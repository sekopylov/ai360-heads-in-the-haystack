"""Downstream influence of retrieval heads (paper Sec. 5).

Two families of task, both evaluated with and without head masking:

* **Extractive QA** -- the answer must be pulled out of the provided context.
  Masking retrieval heads should hurt badly.
* **Reasoning (chain-of-thought)** -- with CoT the model has to refer back to the
  question and to its own earlier steps, so retrieval heads matter; with
  answer-only prompting the model leans on parametric knowledge in the FFN
  layers, so masking matters much less.  That contrast is the point of Sec. 5.3.

Datasets are loaded from JSONL so the real MMLU / GSM8K / MuSiQue files can be
dropped in.  Small built-in samples keep the whole pipeline runnable offline.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from retrieval_heads.attention import HeadMasker
from retrieval_heads.generation import greedy_ids
from retrieval_heads.masking import control_pool, draw_control_subsets, matched_k
from retrieval_heads.models import ModelInfo
from retrieval_heads.scoring import RetrievalScores
from retrieval_heads.utils import HeadRef, eos_ids, get_logger, squad_f1

log = get_logger("downstream")


# --------------------------------------------------------------------------- samples
@dataclass
class QASample:
    context: str
    question: str
    answer: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningSample:
    question: str
    answer: str
    meta: dict[str, Any] = field(default_factory=dict)


def load_qa_jsonl(path: str | Path) -> list[QASample]:
    """Read ``{"context", "question", "answer"}`` records."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out.append(QASample(row["context"], row["question"], row["answer"], row.get("meta", {})))
    return out


def load_reasoning_jsonl(path: str | Path) -> list[ReasoningSample]:
    """Read ``{"question", "answer"}`` records (GSM8K / MMLU / MuSiQue)."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out.append(ReasoningSample(row["question"], row["answer"], row.get("meta", {})))
    return out


#: Offline fallback so the pipeline is testable without downloading anything.
#: Each item is deliberately a *synthetic, dated* fact so the answer cannot come
#: from parametric knowledge -- the same control the paper applies.
def builtin_qa_samples() -> list[QASample]:
    return [
        QASample(
            context=(
                "The Kestrel Institute published its quarterly logistics review on 14 March 2031. "
                "The review states that the northern depot processed 7241 crates, while the "
                "southern depot processed 3180 crates. Staff turnover at the northern depot was "
                "reported as 11 percent, and the southern depot reported 4 percent. The institute "
                "recommended consolidating both depots into a single facility by 2033."
            ),
            question="How many crates did the northern depot process?",
            answer="7241",
        ),
        QASample(
            context=(
                "Dr. Helena Marsh accepted the Trelawney Prize for computational botany in 2029. "
                "Her citation mentions the discovery of a drought-resistant lupin variant, which "
                "she named Lupinus ferrugineus after the rust-coloured markings on its leaves. "
                "The prize committee noted that her field trials ran for six consecutive seasons "
                "across three continents."
            ),
            question="What did Dr. Helena Marsh name after the rust-coloured markings on its leaves?",
            answer="Lupinus ferrugineus",
        ),
        QASample(
            context=(
                "The municipal council of Brackwater approved the harbour dredging project in a "
                "vote of 9 to 4 on 2 February 2030. The project budget was set at 4.6 million "
                "crowns, with completion expected within 27 months. Two councillors abstained, "
                "citing unresolved questions about sediment disposal."
            ),
            question="What was the approved budget for the harbour dredging project?",
            answer="4.6 million crowns",
        ),
        QASample(
            context=(
                "The Ostrander cable car line reopened on 8 September 2032 after eleven months of "
                "closure. The operator reported that the replacement haul rope was manufactured in "
                "the town of Fellwick by the Delaney works. Fares were held at 3 crowns for the "
                "first year, and the line carried 41000 passengers in its first month."
            ),
            question="In which town was the replacement haul rope manufactured?",
            answer="Fellwick",
        ),
        QASample(
            context=(
                "A survey of the Marrowbone wetland conducted in spring 2031 counted 214 breeding "
                "pairs of black terns, up from 168 in the previous survey. The report attributed "
                "the increase to the removal of an invasive reed, Phragmites australis, from the "
                "northern marsh. Water quality readings remained within the acceptable band."
            ),
            question="Which invasive reed was removed from the northern marsh?",
            answer="Phragmites australis",
        ),
        QASample(
            context=(
                "The Tallow Bridge restoration was completed in October 2033 at a final cost of "
                "9.2 million marks. The contractor, Redmayne and Sons, used 340 tonnes of "
                "reclaimed granite from a demolished warehouse in the port district. The bridge "
                "reopened to pedestrians two weeks ahead of schedule."
            ),
            question="How much reclaimed granite did the contractor use?",
            answer="340 tonnes",
        ),
        QASample(
            context=(
                "In 2034 the Palewell Observatory published a catalogue of 1270 variable stars. "
                "The catalogue was compiled by the astronomer Ines Falk over fourteen years using "
                "a 60-centimetre reflector. Falk noted that the faintest objects in the catalogue "
                "required exposures of more than nine hours."
            ),
            question="How many variable stars were in the Palewell catalogue?",
            answer="1270",
        ),
        QASample(
            context=(
                "The Verity line of cargo airships entered service in 2030. The largest vessel, "
                "the Verity Four, has a payload of 62 tonnes and a cruising speed of 118 "
                "kilometres per hour. Its first commercial route connected the inland depot at "
                "Carnsdale with the coastal terminal at Otterhaven."
            ),
            question="What is the cruising speed of the Verity Four?",
            answer="118 kilometres per hour",
        ),
    ]


def builtin_reasoning_samples() -> list[ReasoningSample]:
    """Small arithmetic and logic items that a sub-1B model can actually solve.

    Calibration matters here.  The paper evaluates GSM8K / MMLU / MuSiQue on a 7B
    model; dropping literal GSM8K items onto Qwen3.5-0.8B puts the *baseline* at
    the floor (measured: 12.5% answer-only, 0% with CoT at 128 new tokens), and a
    floor baseline cannot show whether masking heads hurts.  These items keep the
    same structure -- multi-step, needing the question text carried along -- while
    staying within reach of a 0.8B model, so the ablation has signal.  The real
    benchmarks load through ``load_reasoning_jsonl`` when available.
    """
    return [
        ReasoningSample(question="What is 13 plus 26?", answer="39"),
        ReasoningSample(question="What is 7 times 8?", answer="56"),
        ReasoningSample(question="A box holds 6 apples. How many apples are in 4 boxes?", answer="24"),
        ReasoningSample(question="Ana had 15 stickers and gave away 6. How many are left?", answer="9"),
        ReasoningSample(question="A train travels 5 km per minute. How far does it go in 12 minutes?", answer="60"),
        ReasoningSample(question="What is 100 minus 37?", answer="63"),
        ReasoningSample(question="There are 9 rows of 7 chairs. How many chairs in total?", answer="63"),
        ReasoningSample(question="A pen costs 4 crowns. How much do 9 pens cost?", answer="36"),
    ]


# --------------------------------------------------------------------------- metrics
#: A decimal literal that is not glued to another digit or dot, so "3.10.12" is left
#: alone while "0.50" and "63.0" are canonicalised.
_NUMBER_RE = re.compile(r"(?<![\d.])-?\d+(?:\.\d+)?(?![\d.])")


def _canonical_number(match: re.Match[str]) -> str:
    """Rewrite a numeric literal to one canonical spelling (``63.0`` -> ``63``).

    The metrics used to compare numbers **as strings**: ``accuracy("63.0", "63")``
    was 0 because ``"63.0" != "63"``, and ``word_f1`` scored ``"0.50"`` against
    ``"0.5"`` as a mismatch.  That only bites once real datasets are wired in (the
    built-ins happen to use one spelling), but it is a wrong metric, not a missing
    feature.  Canonicalising here fixes both metrics at once, because both tokenise
    the normalised string.
    """
    raw = match.group(0)
    try:
        value = float(raw)
    except ValueError:  # pragma: no cover - the regex only emits numeric literals
        return raw
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(value)


def _numbers_equal(a: str, b: str) -> bool:
    """Numeric equality of two literal strings, tolerating float representation."""
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    except ValueError:  # pragma: no cover - the regex only emits numeric literals
        return a == b


def _normalise(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[,\$]", "", text)
    text = _NUMBER_RE.sub(_canonical_number, text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .\n\t")


def word_f1(prediction: str, target: str) -> float:
    """SQuAD-style word-level F1, used for extractive QA."""
    return squad_f1(_normalise(prediction).split(), _normalise(target).split())


def final_answer(text: str) -> str:
    """Pull the answer out of a CoT completion (``#### x`` or ``Answer: x``)."""
    for pattern in (r"####\s*(.+)", r"[Aa]nswer\s*[:=]\s*(.+)"):
        found = re.findall(pattern, text)
        if found:
            return found[-1].strip().split("\n")[0]
    return text.strip().split("\n")[-1]


def accuracy(prediction: str, target: str) -> float:
    """Normalised exact match, tolerant of trailing punctuation and units.

    Containment is checked on **word boundaries**, not as a raw substring: a raw
    ``t in p`` marked ``"163"`` correct for the target ``"63"`` and ``"19"``
    correct for ``"9"``, which inflates accuracy on the short numeric answers the
    built-in reasoning set uses.

    Numbers are compared *numerically*, not as strings (``63.0`` == ``63``,
    ``0.50`` == ``0.5``): the old ``pn[0] == tn[0]`` was a spelling test.
    """
    p, t = _normalise(prediction), _normalise(target)
    if p == t:
        return 1.0
    # Numeric answers: compare the first number found.  Heuristic on purpose --
    # "4.6 million crowns" counts as 4.6 -- and it only affects `cot`; extractive
    # QA uses word F1.
    pn, tn = re.findall(r"-?\d+(?:\.\d+)?", p), re.findall(r"-?\d+(?:\.\d+)?", t)
    if pn and tn and _numbers_equal(pn[0], tn[0]):
        return 1.0
    if not t:
        return 0.0
    # Boundaries include '.' and '-': without them a decimal tail matched, so
    # accuracy("1.63", "63") and accuracy("6.5", "6") were both credited.
    return 1.0 if re.search(rf"(?<![\w.\-]){re.escape(t)}(?![\w.\-])", p) else 0.0


# --------------------------------------------------------------------------- generation
@torch.no_grad()
def _generate_text(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int,
    attn_impl: str = "sdpa",
    prefill_chunk: int | None = None,
) -> str:
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    # Shared loop with scoring/masking: chunked prefill (findings section 18) and
    # one EOS policy live in retrieval_heads.generation.
    generated = greedy_ids(
        model, ids, max_new_tokens=max_new_tokens, eos=eos_ids(model, tokenizer),
        tokenizer=tokenizer, attn_impl=attn_impl, prefill_chunk=prefill_chunk,
    )
    return tokenizer.decode(generated, skip_special_tokens=True)


def _chat(tokenizer: Any, user: str, *, enable_thinking: bool | None,
          chat_template: bool = True, system_prompt: str | None = None) -> str:
    """`--no-chat-template` actually disables the template here, too."""
    if not chat_template:
        return user
    from retrieval_heads.haystack import render_chat

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user})
    return render_chat(tokenizer, messages, enable_thinking=enable_thinking)


# --------------------------------------------------------------------------- evaluators
def extractive_qa_scores(
    model: Any,
    tokenizer: Any,
    info: ModelInfo,
    samples: Sequence[QASample],
    *,
    masked_heads: Sequence[HeadRef] = (),
    max_new_tokens: int = 24,
    enable_thinking: bool | None = False,
    prefill_chunk: int | None = None,
    chat_template: bool = True,
    system_prompt: str | None = None,
) -> list[float]:
    """Per-sample word-level F1 (percent) on extractive QA.

    The per-sample values are what :func:`qa_ablation` records as the retrieval arm's
    spread: a single mean could not say whether a drop came from every item or from
    one, and the paper's claim is about the aggregate.
    """
    masker = HeadMasker(model, info, masked_heads) if masked_heads else None
    scores: list[float] = []
    try:
        for sample in samples:
            prompt = _chat(
                tokenizer,
                f"{sample.context}\n\nQuestion: {sample.question}\nAnswer with the shortest exact span from the document.",
                enable_thinking=enable_thinking, chat_template=chat_template,
                system_prompt=system_prompt,
            )
            text = _generate_text(model, tokenizer, prompt, max_new_tokens=max_new_tokens,
                                  prefill_chunk=prefill_chunk)
            scores.append(100.0 * word_f1(text, sample.answer))
    finally:
        if masker is not None:
            masker.remove()
    return scores


def evaluate_extractive_qa(
    model: Any,
    tokenizer: Any,
    info: ModelInfo,
    samples: Sequence[QASample],
    *,
    masked_heads: Sequence[HeadRef] = (),
    max_new_tokens: int = 24,
    enable_thinking: bool | None = False,
    prefill_chunk: int | None = None,
    chat_template: bool = True,
    system_prompt: str | None = None,
) -> float:
    """Mean word-level F1 on extractive QA."""
    scores = extractive_qa_scores(
        model, tokenizer, info, samples, masked_heads=masked_heads,
        max_new_tokens=max_new_tokens, enable_thinking=enable_thinking,
        prefill_chunk=prefill_chunk, chat_template=chat_template,
        system_prompt=system_prompt,
    )
    return float(np.mean(scores)) if scores else 0.0


def evaluate_reasoning(
    model: Any,
    tokenizer: Any,
    info: ModelInfo,
    samples: Sequence[ReasoningSample],
    *,
    cot: bool = True,
    masked_heads: Sequence[HeadRef] = (),
    max_new_tokens: int = 256,
    enable_thinking: bool | None = False,
    prefill_chunk: int | None = None,
    chat_template: bool = True,
    system_prompt: str | None = None,
) -> float:
    """Accuracy on reasoning tasks, with or without chain-of-thought."""
    instruction = (
        "Solve the problem. Think step by step, then write the final answer on its own line "
        "prefixed with '#### '."
        if cot else
        "Solve the problem. Reply with the final answer only, prefixed with '#### '."
    )
    masker = HeadMasker(model, info, masked_heads) if masked_heads else None
    hits = []
    try:
        for sample in samples:
            prompt = _chat(tokenizer, f"{sample.question}\n\n{instruction}",
                           enable_thinking=enable_thinking, chat_template=chat_template,
                           system_prompt=system_prompt)
            text = _generate_text(model, tokenizer, prompt, max_new_tokens=max_new_tokens,
                                  prefill_chunk=prefill_chunk)
            hits.append(accuracy(final_answer(text), sample.answer))
    finally:
        if masker is not None:
            masker.remove()
    return 100.0 * float(np.mean(hits)) if hits else 0.0


# --------------------------------------------------------------------------- ablations
def qa_ablation(
    model: Any,
    tokenizer: Any,
    info: ModelInfo,
    scores: RetrievalScores,
    samples: Sequence[QASample],
    *,
    k_values: Sequence[int] = (8,),
    n_random_trials: int = 3,
    seed: int = 0,
    max_new_tokens: int = 24,
    prefill_chunk: int | None = 4096,
    enable_thinking: bool | None = False,
    chat_template: bool = True,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Extractive QA: baseline, retrieval heads masked, random heads masked.

    The retrieval arm is the top K heads *by score*, and the random arm is drawn
    from heads at or below the threshold, so both remove exactly the same number
    of heads and the control contains no retrieval heads.
    """
    baseline_scores = extractive_qa_scores(model, tokenizer, info, samples,
                                           max_new_tokens=max_new_tokens,
                                           prefill_chunk=prefill_chunk,
                                           enable_thinking=enable_thinking,
                                           chat_template=chat_template,
                                           system_prompt=system_prompt)
    baseline = float(np.mean(baseline_scores)) if baseline_scores else 0.0
    log.info("QA baseline F1=%.1f", baseline)
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {"task": "extractive_qa", "baseline_f1": baseline,
                           # Per-sample, so a drop can be attributed to every item or
                           # to one outlier instead of only being a mean.
                           "baseline_f1s": baseline_scores,
                           "baseline_f1_std": float(np.std(baseline_scores)),
                           "n_samples": len(samples), "k_values": list(k_values),
                           "n_scoreable_heads": info.n_scoreable_heads,
                           "model": info.name, "max_new_tokens": max_new_tokens,
                           "prefill_chunk": prefill_chunk,
                           "enable_thinking": enable_thinking,
                           "chat_template": chat_template,
                           "system_prompt": system_prompt,
                           # How the heads were chosen: the names are in `by_k`, but
                           # without these a reader cannot tell under which conditions
                           # (and argmax domain) they were selected.
                           "threshold": scores.threshold,
                           "pairing": getattr(scores, "pairing", None),
                           "argmax_domain": (getattr(scores, "meta", None) or {}).get(
                               "argmax_domain"),
                           "by_k": {}}
    retrieval_ranked = scores.ranked_heads()
    pool, contaminated = control_pool(scores)
    if contaminated:
        log.warning("every scoreable head is above the %.2f threshold; the QA random "
                    "arm is drawn from all heads and is contaminated", scores.threshold)
    out["n_non_retrieval_heads"] = len(pool)
    out["random_control_contaminated"] = contaminated
    for k in k_values:
        k_eff = matched_k(k, len(pool))
        if k_eff <= 0:
            continue
        if k_eff < k:
            log.warning("QA k=%d exceeds the %d non-retrieval heads available; using k=%d",
                        k, len(pool), k_eff)
        top = retrieval_ranked[:k_eff]
        retrieval_scores = extractive_qa_scores(model, tokenizer, info, samples,
                                                masked_heads=top,
                                                max_new_tokens=max_new_tokens,
                                                prefill_chunk=prefill_chunk,
                                                enable_thinking=enable_thinking,
                                                chat_template=chat_template,
                                                system_prompt=system_prompt)
        f1_retrieval = float(np.mean(retrieval_scores)) if retrieval_scores else 0.0
        trials, overlaps, picks = [], [], []
        # Same control-drawing rule as `masking_curve`: no repeated subset while the
        # pool allows it, so a small pool does not silently shrink the trial count.
        for pick in draw_control_subsets(pool, k_eff, n_random_trials, rng):
            picks.append([str(h) for h in pick])
            # How many of the "random" heads are actually retrieval heads: the
            # audit trail for the control, stored instead of trusted.
            overlaps.append(sum(1 for h in pick if h in set(top)))
            trials.append(evaluate_extractive_qa(model, tokenizer, info, samples,
                                                 masked_heads=pick, max_new_tokens=max_new_tokens,
                                                 prefill_chunk=prefill_chunk,
                                                 enable_thinking=enable_thinking,
                                                 chat_template=chat_template,
                                                 system_prompt=system_prompt))
        out["by_k"][str(k)] = {
            "k_effective": k_eff,
            "retrieval_f1": f1_retrieval,
            # The retrieval arm's per-sample spread: with one mean, a K that hurts
            # half the items and helps the other half reads the same as a K that
            # does nothing.
            "retrieval_f1s": retrieval_scores,
            "retrieval_f1_std": float(np.std(retrieval_scores)),
            "random_f1_mean": float(np.mean(trials)),
            "random_f1_std": float(np.std(trials)),
            "random_retrieval_overlap": overlaps,
            "drop_retrieval": baseline - f1_retrieval,
            "drop_random": baseline - float(np.mean(trials)),
            "masked_heads": [str(h) for h in top],
            "random_picks": picks,
        }
        log.info("QA k=%d: retrieval F1=%.1f (drop %.1f) | random F1=%.1f (drop %.1f)",
                 k, f1_retrieval, baseline - f1_retrieval, np.mean(trials),
                 baseline - float(np.mean(trials)))
    return out


def cot_ablation(
    model: Any,
    tokenizer: Any,
    info: ModelInfo,
    scores: RetrievalScores,
    samples: Sequence[ReasoningSample],
    *,
    k: int = 8,
    n_random_trials: int = 2,
    seed: int = 0,
    max_new_tokens: int = 256,
    prefill_chunk: int | None = 4096,
    enable_thinking: bool | None = False,
    chat_template: bool = True,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Reasoning accuracy with/without CoT, and with/without retrieval heads.

    Same matched-arm rule as :func:`qa_ablation`: top-K by score against K heads
    drawn from the non-retrieval pool.
    """
    retrieval_ranked = scores.ranked_heads()
    pool, contaminated = control_pool(scores)
    if contaminated:
        log.warning("every scoreable head is above the %.2f threshold; the CoT random "
                    "arm is drawn from all heads and is contaminated", scores.threshold)
    k_eff = matched_k(k, len(pool))
    if k_eff < k:
        log.warning("CoT k=%d exceeds the %d non-retrieval heads available; using k=%d",
                    k, len(pool), k_eff)
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {"task": "cot_reasoning", "n_samples": len(samples), "k": k,
                           "k_effective": k_eff, "n_scoreable_heads": info.n_scoreable_heads,
                           "n_non_retrieval_heads": len(pool),
                           "random_control_contaminated": contaminated,
                           "model": info.name, "max_new_tokens": max_new_tokens,
                           "prefill_chunk": prefill_chunk,
                           "enable_thinking": enable_thinking,
                           "chat_template": chat_template,
                           "system_prompt": system_prompt,
                           "threshold": scores.threshold,
                           "pairing": getattr(scores, "pairing", None),
                           "argmax_domain": (getattr(scores, "meta", None) or {}).get(
                               "argmax_domain"),
                           "results": {}}

    for cot in (False, True):
        variant = "cot" if cot else "answer_only"
        baseline = evaluate_reasoning(model, tokenizer, info, samples, cot=cot,
                                      max_new_tokens=max_new_tokens,
                                      prefill_chunk=prefill_chunk,
                                      enable_thinking=enable_thinking,
                                      chat_template=chat_template,
                                      system_prompt=system_prompt)
        masked_retrieval = evaluate_reasoning(model, tokenizer, info, samples, cot=cot,
                                              masked_heads=retrieval_ranked[:k_eff],
                                              max_new_tokens=max_new_tokens,
                                              prefill_chunk=prefill_chunk,
                                              enable_thinking=enable_thinking,
                                              chat_template=chat_template,
                                              system_prompt=system_prompt)
        trials, overlaps, picks = [], [], []
        for pick in draw_control_subsets(pool, k_eff, n_random_trials, rng):
            picks.append([str(h) for h in pick])
            overlaps.append(sum(1 for h in pick if h in set(retrieval_ranked[:k_eff])))
            trials.append(evaluate_reasoning(model, tokenizer, info, samples, cot=cot,
                                             masked_heads=pick, max_new_tokens=max_new_tokens,
                                             prefill_chunk=prefill_chunk,
                                             enable_thinking=enable_thinking,
                                             chat_template=chat_template,
                                             system_prompt=system_prompt))
        out["results"][variant] = {
            "baseline": baseline,
            "retrieval_masked": masked_retrieval,
            "random_masked_mean": float(np.mean(trials)),
            "random_masked_std": float(np.std(trials)),
            "random_retrieval_overlap": overlaps,
            "drop_retrieval": baseline - masked_retrieval,
            "drop_random": baseline - float(np.mean(trials)),
            # The intervention sets: this artifact used to record neither.
            "masked_heads": [str(h) for h in retrieval_ranked[:k_eff]],
            "random_picks": picks,
        }
        log.info("CoT=%s: baseline=%.1f retrieval-masked=%.1f random-masked=%.1f",
                 cot, baseline, masked_retrieval, np.mean(trials))
    return out
