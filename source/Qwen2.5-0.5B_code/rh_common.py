"""
Общие утилиты для пайплайна retrieval heads (Wu et al., 2024) на Qwen2.5-0.5B-Instruct.

Почему Qwen2.5-0.5B-Instruct:
  * семейство Qwen2 есть среди реализованных в репозитории (source/modeling_qwen2.py);
  * ~0.5B параметров -- самая маленькая из доступных моделей этого семейства;
  * все 24 слоя -- обычный (full) self-attention, без линейного/гибридного внимания,
    поэтому каждую из 24 x 14 = 336 голов можно и оценить, и замаскировать.

Модуль используется всеми скриптами пайплайна: загрузка модели, построение
стога с иглой, жадная генерация, хуки для весов внимания и для маскирования голов.
"""
import contextlib
import glob
import inspect
import json
import logging
import os
import random
import sys
import time

import numpy as np

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# Игла из оригинального NIAH-теста (используется для оценки качества при маскировании)
SF_NEEDLE = {
    "needle": "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n",
    "question": "What is the best thing to do in San Francisco?",
    "real_needle": "eat a sandwich and sit in Dolores Park on a sunny day",
}


# --------------------------------------------------------------------------- #
# логирование
# --------------------------------------------------------------------------- #
def setup_logger(name, log_dir="logs"):
    """Логгер в консоль и в logs/<name>.log (файл дописывается)."""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(name)-10s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


@contextlib.contextmanager
def stage(logger, title):
    """Логирует начало/конец этапа и его длительность."""
    logger.info(f"===== [START] {title} =====")
    t0 = time.time()
    try:
        yield
    except Exception:
        logger.exception(f"===== [FAIL]  {title} ({time.time() - t0:.1f}s) =====")
        raise
    logger.info(f"===== [DONE]  {title} ({time.time() - t0:.1f}s) =====")


def model_tag(model_path):
    return model_path.rstrip("/").split("/")[-1]


# --------------------------------------------------------------------------- #
# модель
# --------------------------------------------------------------------------- #
def get_decoder_layers(model):
    for path in ("model.layers", "model.language_model.layers", "language_model.model.layers"):
        obj = model
        try:
            for p in path.split("."):
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("Не нашёл список decoder-слоёв; выведи print(model) и поправь get_decoder_layers")


def load_model(model_path, dtype, device, logger):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Загружаю токенизатор и конфиг: {model_path}")
    tok = AutoTokenizer.from_pretrained(model_path)
    cfg = AutoConfig.from_pretrained(model_path)
    n_layers = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_heads
    layer_types = getattr(cfg, "layer_types", None)
    if layer_types is not None and any(t != "full_attention" for t in layer_types):
        logger.warning(f"Не все слои full_attention: {sorted(set(layer_types))}. "
                       "Скоры для sliding/linear слоёв будут некорректны.")
    if getattr(cfg, "use_sliding_window", False):
        logger.warning("В конфиге включено sliding window -- длинный контекст может обрезаться.")
    logger.info(f"Архитектура: {cfg.model_type}, слоёв={n_layers}, голов/слой={n_heads}, "
                f"KV-голов={getattr(cfg, 'num_key_value_heads', n_heads)}, head_dim={head_dim}, "
                f"всего голов={n_layers * n_heads}, max_pos={getattr(cfg, 'max_position_embeddings', '?')}")

    torch_dtype = getattr(torch, dtype)
    if device == "cpu" and torch_dtype != torch.float32:
        logger.warning("CPU: переключаю dtype на float32")
        torch_dtype = torch.float32
    logger.info(f"Загружаю веса (dtype={torch_dtype}, device={device}, attn=sdpa)")
    kw = dict(attn_implementation="sdpa")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch_dtype, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype, **kw)
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Модель загружена: {n_params / 1e6:.0f}M параметров")

    info = dict(n_layers=n_layers, n_heads=n_heads, head_dim=head_dim, model_type=cfg.model_type)
    return model, tok, info


def set_attn_impl(model, name):
    """Переключение sdpa <-> eager (eager нужен, чтобы получить веса внимания)."""
    try:
        model.set_attn_implementation(name)
        return
    except Exception:
        pass
    for m in model.modules():
        c = getattr(m, "config", None)
        if c is not None and hasattr(c, "_attn_implementation"):
            c._attn_implementation = name


def _logits_kw(model):
    """В разных версиях transformers аргумент называется по-разному."""
    params = inspect.signature(model.forward).parameters
    if "logits_to_keep" in params:
        return {"logits_to_keep": 1}
    if "num_logits_to_keep" in params:
        return {"num_logits_to_keep": 1}
    return {}


def eos_ids_of(model, tok):
    eos = set()
    g = getattr(model.generation_config, "eos_token_id", None)
    for e in ([g] if isinstance(g, int) else (g or [])):
        eos.add(int(e))
    if tok.eos_token_id is not None:
        eos.add(int(tok.eos_token_id))
    return eos


class AttnGrabber:
    """forward-хуки на self_attn: сохраняют attn_weights [1, H, q, kv] (только в eager)."""

    def __init__(self, model):
        self.enabled = False
        self.weights = {}
        self.handles = []
        for li, layer in enumerate(get_decoder_layers(model)):
            self.handles.append(layer.self_attn.register_forward_hook(self._hook(li)))

    def _hook(self, li):
        def hook(module, args, output):
            if self.enabled and isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                self.weights[li] = output[1].detach()
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


class HeadMasker:
    """
    Абляция голов: forward-pre-хук на o_proj обнуляет выход выбранных голов
    (вход o_proj -- конкатенация выходов голов [.., H * head_dim]).
    Это эквивалентно удалению вклада головы в residual stream.
    """

    def __init__(self, model, n_heads, head_dim):
        self.n_heads, self.head_dim = n_heads, head_dim
        self.by_layer = {}
        self.handles = []
        for li, layer in enumerate(get_decoder_layers(model)):
            self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(self._hook(li)))

    def set_heads(self, heads):
        self.by_layer = {}
        for l, h in heads:
            self.by_layer.setdefault(int(l), []).append(int(h))

    def _hook(self, li):
        def hook(module, args):
            hs = self.by_layer.get(li)
            if not hs:
                return None
            x = args[0]
            v = x.reshape(*x.shape[:-1], self.n_heads, self.head_dim).clone()
            v[..., hs, :] = 0
            return (v.reshape(x.shape),) + tuple(args[1:])
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


def greedy_generate(model, tok, input_ids, max_new, eos_ids, grabber=None, on_step=None):
    """
    Prefill через sdpa (без матрицы n x n и без логитов для всех позиций),
    decode по одному токену. Если передан grabber -- decode идёт в eager и
    после каждого шага вызывается on_step(weights, token_tensor).
    """
    import torch

    lk = _logits_kw(model)
    with torch.no_grad():
        set_attn_impl(model, "sdpa")
        if grabber is not None:
            grabber.enabled = False
        out = model(input_ids=input_ids[:, :-1], use_cache=True, **lk)
        past = out.past_key_values
        del out

        if grabber is not None:
            set_attn_impl(model, "eager")
            grabber.enabled = True
        inp = input_ids[:, -1:]
        generated = []
        try:
            for _ in range(max_new):
                if grabber is not None:
                    grabber.weights.clear()
                o = model(input_ids=inp, past_key_values=past, use_cache=True, **lk)
                past = o.past_key_values
                nxt = o.logits[0, -1].argmax()
                tid = int(nxt.item())
                if tid in eos_ids:
                    break
                generated.append(tid)
                if grabber is not None:
                    if not grabber.weights:
                        raise RuntimeError("Хуки не получили attn_weights -- eager не включился "
                                           "(нужен transformers>=4.48)")
                    on_step(grabber.weights, nxt)
                piece = tok.decode([tid])
                if "\n" in piece and tok.decode(generated).strip():
                    break
                inp = nxt.view(1, 1)
        finally:
            if grabber is not None:
                grabber.enabled = False
                grabber.weights.clear()
                set_attn_impl(model, "sdpa")
    return generated


# --------------------------------------------------------------------------- #
# данные: стог + игла
# --------------------------------------------------------------------------- #
_FILLER_WORDS = ("river city market window garden letter morning paper engine village story "
                 "mountain question teacher office winter season problem company number history "
                 "silver kitchen forest bridge doctor student evening machine harbor library").split()


def _synthetic_text(n_words, seed=0):
    rng = random.Random(seed)
    sents = []
    while n_words > 0:
        k = rng.randint(8, 18)
        w = [rng.choice(_FILLER_WORDS) for _ in range(k)]
        sents.append(" ".join(w).capitalize() + ".")
        n_words -= k
    return " ".join(sents)


def read_haystack_ids(tok, hay_dir, need_tokens, logger):
    files = sorted(glob.glob(os.path.join(hay_dir, "*.txt"))) if hay_dir else []
    if not files:
        logger.warning(f"В '{hay_dir}' нет *.txt -- использую синтетический стог "
                       "(только для отладки, результаты будут нерепрезентативны)")
        text = _synthetic_text(need_tokens * 2)
    else:
        text = ""
        while len(text.split()) < need_tokens:
            for f in files:
                with open(f, "r", encoding="utf-8", errors="replace") as fh:
                    text += fh.read() + "\n"
    ids = tok(text, add_special_tokens=False)["input_ids"]
    logger.info(f"Стог '{hay_dir or 'synthetic'}': {len(files)} файлов, {len(ids)} токенов")
    return ids


def load_detect_needles(haystack_dir, logger):
    """haystack_for_detect/needles.jsonl + part1..partN (как в оригинальном репо)."""
    path = os.path.join(haystack_dir, "needles.jsonl")
    if not os.path.exists(path):
        logger.warning(f"Нет {path} -- использую одну иглу про San Francisco")
        pg = "PaulGrahamEssays" if os.path.isdir("PaulGrahamEssays") else None
        return [dict(SF_NEEDLE, hay_dir=pg)]
    # как в оригинале: i-я игла -> part{i+1}; если игл больше, чем папок, папки идут по кругу
    parts = sorted(d for d in glob.glob(os.path.join(haystack_dir, "part*")) if os.path.isdir(d))
    if not parts:
        logger.warning(f"В {haystack_dir} нет папок part*")
    needles = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(l for l in f if l.strip()):
            nd = json.loads(line)
            own = os.path.join(haystack_dir, f"part{i + 1}")
            nd["hay_dir"] = own if os.path.isdir(own) else (parts[i % len(parts)] if parts else None)
            needles.append(nd)
    logger.info(f"Загружено игл: {len(needles)} из {path}; стоги: {[n['hay_dir'] for n in needles]}")
    return needles


def period_ids_of(tok):
    """Токены, заканчивающиеся точкой -- сюда вставляем иглу (граница предложения)."""
    return {i for t, i in tok.get_vocab().items() if t.rstrip("ĊĠ ▁").endswith(".")}


def build_context(tok, hay_ids, needle_ids, length, depth, buffer, period_ids):
    budget = max(int(length) - buffer - len(needle_ids), 0)
    ctx = hay_ids[:budget]
    if depth >= 100:
        new = ctx + needle_ids
    else:
        ins = int(len(ctx) * depth / 100)
        while ins > 0 and ctx[ins - 1] not in period_ids:
            ins -= 1
        new = ctx[:ins] + needle_ids + ctx[ins:]
    return tok.decode(new)


def build_prompt_text(tok, context, question, use_chat):
    q = f"Based on the content of the book, Question: {question}\nAnswer:"
    if use_chat and getattr(tok, "chat_template", None):
        msgs = [{"role": "user", "content": f"<book>{context}</book>\n{q}"}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return context + q


def find_needle_span(tok, text, real_needle):
    """Токены иглы через offset_mapping. Возвращает (ids, start, end) или (ids|None, -1, -1)."""
    cs = text.find(real_needle)
    if cs < 0:
        return None, -1, -1
    ce = cs + len(real_needle)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    hit = [i for i, (a, b) in enumerate(offs) if b > cs and a < ce]
    if not hit:
        return ids, -1, -1
    return ids, hit[0], hit[-1] + 1


_scorer = None


def rouge_recall(reference, response):
    global _scorer
    if _scorer is None:
        from rouge_score import rouge_scorer
        _scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    return _scorer.score(reference, response)["rouge1"].recall * 100


# --------------------------------------------------------------------------- #
# скоры голов
# --------------------------------------------------------------------------- #
def load_head_scores(path):
    """{'l-h': [scores]} -> {(l, h): mean}"""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        l, h = (int(x) for x in k.split("-"))
        out[(l, h)] = float(np.mean(v)) if len(v) else 0.0
    return out