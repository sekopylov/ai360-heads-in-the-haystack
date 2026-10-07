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
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from retrieval_heads.attention import HeadMasker, set_attn_implementation
from retrieval_heads.models import ModelInfo, model_device
from retrieval_heads.scoring import RetrievalScores
from retrieval_heads.utils import HeadRef, get_logger, save_json

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
    return [
        ReasoningSample(
            question="A workshop builds 14 lanterns per day. How many lanterns does it build in 3 weeks?",
            answer="294",
        ),
        ReasoningSample(
            question="Mira has 48 shells and gives away a quarter of them. How many shells does she have left?",
            answer="36",
        ),
        ReasoningSample(
            question="A train travels 60 km in 45 minutes. How far does it travel in 2 hours at the same speed?",
            answer="160",
        ),
        ReasoningSample(
            question="A crate holds 24 bottles. A lorry carries 35 crates and makes 4 trips. How many bottles does it deliver?",
            answer="3360",
        ),
        ReasoningSample(
            question="Elena buys 7 notebooks at 3 crowns each and pays with a 50-crown note. How much change does she receive?",
            answer="29",
        ),
        ReasoningSample(
            question="A tank holds 180 litres and is filled at 12 litres per minute. How many minutes does it take to fill three quarters of the tank?",
            answer="11.25",
        ),
        ReasoningSample(
            question="A school has 9 classes with 28 pupils each. If 45 pupils are absent, how many are present?",
            answer="207",
        ),
        ReasoningSample(
            question="A printer produces 18 pages per minute. How many pages does it produce in 2 hours and 30 minutes?",
            answer="2700",
        ),
    ]


# --------------------------------------------------------------------------- metrics
def _normalise(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[,\$]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .\n\t")


def word_f1(prediction: str, target: str) -> float:
    """SQuAD-style word-level F1, used for extractive QA."""
    from collections import Counter

    pred = _normalise(prediction).split()
    gold = _normalise(target).split()
    if not pred or not gold:
        return 0.0
    overlap = sum((Counter(pred) & Counter(gold)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def final_answer(text: str) -> str:
    """Pull the answer out of a CoT completion (``#### x`` or ``Answer: x``)."""
    for pattern in (r"####\s*(.+)", r"[Aa]nswer\s*[:=]\s*(.+)"):
        found = re.findall(pattern, text)
        if found:
            return found[-1].strip().split("\n")[0]
    return text.strip().split("\n")[-1]


def accuracy(prediction: str, target: str) -> float:
    """Normalised exact match, tolerant of trailing punctuation and units."""
    p, t = _normalise(prediction), _normalise(target)
    if p == t:
        return 1.0
    # numeric answers: compare the first number found
    pn, tn = re.findall(r"-?\d+(?:\.\d+)?", p), re.findall(r"-?\d+(?:\.\d+)?", t)
    if pn and tn and pn[0] == tn[0]:
        return 1.0
    return 1.0 if t and t in p else 0.0


# --------------------------------------------------------------------------- generation
@torch.no_grad()
def _generate_text(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int,
    attn_impl: str = "sdpa",
) -> str:
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    ids = ids.to(model_device(model))
    restore = set_attn_implementation(model, attn_impl)
    eos = set()
    for source in (getattr(model, "config", None), getattr(model, "generation_config", None)):
        value = getattr(source, "eos_token_id", None) if source is not None else None
        if isinstance(value, int):
            eos.add(value)
        elif isinstance(value, (list, tuple, set)):
            eos.update(int(v) for v in value)
    if getattr(tokenizer, "eos_token_id", None) is not None:
        eos.add(int(tokenizer.eos_token_id))
    try:
        out = model(input_ids=ids, use_cache=True)
        cache = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        generated: list[int] = []
        for _ in range(max_new_tokens):
            token = int(nxt[0, 0])
            if token in eos:
                break
            generated.append(token)
            out = model(input_ids=nxt, past_key_values=cache, use_cache=True)
            nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
    finally:
        if restore is not None:
            set_attn_implementation(model, restore)
    return tokenizer.decode(generated, skip_special_tokens=True)


def _chat(tokenizer: Any, user: str, *, enable_thinking: bool | None) -> str:
    messages = [{"role": "user", "content": user}]
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


# --------------------------------------------------------------------------- evaluators
@dataclass
class TaskResult:
    task: str
    variant: str
    metric: str
    value: float
    n: int
    masked_heads: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task, "variant": self.variant, "metric": self.metric,
            "value": self.value, "n": self.n, "masked_heads": self.masked_heads,
        }


def evaluate_extractive_qa(
    model: Any,
    tokenizer: Any,
    info: ModelInfo,
    samples: Sequence[QASample],
    *,
    masked_heads: Sequence[HeadRef] = (),
    max_new_tokens: int = 24,
    enable_thinking: bool | None = False,
) -> float:
    """Mean word-level F1 on extractive QA."""
    masker = HeadMasker(model, info, masked_heads) if masked_heads else None
    scores = []
    try:
        for sample in samples:
            prompt = _chat(
                tokenizer,
                f"{sample.context}\n\nQuestion: {sample.question}\nAnswer with the shortest exact span from the document.",
                enable_thinking=enable_thinking,
            )
            text = _generate_text(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
            scores.append(100.0 * word_f1(text, sample.answer))
    finally:
        if masker is not None:
            masker.remove()
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
            prompt = _chat(tokenizer, f"{sample.question}\n\n{instruction}", enable_thinking=enable_thinking)
            text = _generate_text(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
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
) -> dict[str, Any]:
    """Extractive QA: baseline, retrieval heads masked, random heads masked."""
    baseline = evaluate_extractive_qa(model, tokenizer, info, samples, max_new_tokens=max_new_tokens)
    log.info("QA baseline F1=%.1f", baseline)
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {"task": "extractive_qa", "baseline_f1": baseline,
                           "n_samples": len(samples), "k_values": list(k_values), "by_k": {}}
    retrieval_ranked = scores.heads_above()
    pool = info.scoreable_heads
    for k in k_values:
        top = retrieval_ranked[:k]
        f1_retrieval = evaluate_extractive_qa(model, tokenizer, info, samples,
                                              masked_heads=top, max_new_tokens=max_new_tokens)
        trials = []
        for _ in range(n_random_trials):
            pick = [pool[i] for i in rng.permutation(len(pool))[:k]]
            trials.append(evaluate_extractive_qa(model, tokenizer, info, samples,
                                                 masked_heads=pick, max_new_tokens=max_new_tokens))
        out["by_k"][str(k)] = {
            "retrieval_f1": f1_retrieval,
            "random_f1_mean": float(np.mean(trials)),
            "random_f1_std": float(np.std(trials)),
            "drop_retrieval": baseline - f1_retrieval,
            "drop_random": baseline - float(np.mean(trials)),
            "masked_heads": [str(h) for h in top],
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
) -> dict[str, Any]:
    """Reasoning accuracy with/without CoT, and with/without retrieval heads."""
    retrieval_ranked = scores.heads_above()
    pool = info.scoreable_heads
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {"task": "cot_reasoning", "n_samples": len(samples), "k": k, "results": {}}

    for cot in (False, True):
        variant = "cot" if cot else "answer_only"
        baseline = evaluate_reasoning(model, tokenizer, info, samples, cot=cot,
                                      max_new_tokens=max_new_tokens)
        masked_retrieval = evaluate_reasoning(model, tokenizer, info, samples, cot=cot,
                                              masked_heads=retrieval_ranked[:k],
                                              max_new_tokens=max_new_tokens)
        trials = []
        for _ in range(n_random_trials):
            pick = [pool[i] for i in rng.permutation(len(pool))[:k]]
            trials.append(evaluate_reasoning(model, tokenizer, info, samples, cot=cot,
                                             masked_heads=pick, max_new_tokens=max_new_tokens))
        out["results"][variant] = {
            "baseline": baseline,
            "retrieval_masked": masked_retrieval,
            "random_masked_mean": float(np.mean(trials)),
            "random_masked_std": float(np.std(trials)),
            "drop_retrieval": baseline - masked_retrieval,
            "drop_random": baseline - float(np.mean(trials)),
        }
        log.info("CoT=%s: baseline=%.1f retrieval-masked=%.1f random-masked=%.1f",
                 cot, baseline, masked_retrieval, np.mean(trials))
    return out
