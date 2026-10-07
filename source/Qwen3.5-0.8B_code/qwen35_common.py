"""
Общие утилиты для Retrieval Head экспериментов на Qwen3.5 (гибрид Gated DeltaNet + Gated Attention).

Почему не нужны патченные modeling_*.py из faiss_attn/:
  * В свежем transformers механизм внимания подключается через AttentionInterface.
    Мы регистрируем свою функцию "retrieval_sdpa":
      - prefill (q_len > 1)  -> обычный SDPA (быстро, без матрицы внимания L x L);
      - decode  (q_len == 1) -> явное вычисление softmax(QK^T), из которого берём
        top-1 позицию для каждой головы (для retrieval score) и, при необходимости,
        маскируем головы (обнуляем query -> равномерное внимание, как в оригинале).
  * Матрицы внимания целиком не хранятся (output_attentions не нужен) -> экономия памяти.

Важно про архитектуру Qwen3.5-0.8B:
  24 слоя, из них только 6 — полноценное внимание (слои 3, 7, 11, 15, 19, 23),
  по 8 query-голов (2 KV-головы, GQA). Остальные 18 слоёв — линейное внимание
  (Gated DeltaNet), у них нет матрицы внимания, поэтому retrieval heads ищутся
  только среди 6 x 8 = 48 голов.
"""
import glob

import torch
from transformers import AttentionInterface, AutoConfig, AutoTokenizer
from transformers.integrations.sdpa_attention import sdpa_attention_forward

ATTN_NAME = "retrieval_sdpa"


class _AttnState:
    def __init__(self):
        self.record = False  # сохранять top-1 индекс ключа на шагах декодирования
        self.block = {}      # {layer_idx: [head_idx, ...]} — головы для маскирования (только decode)
        self.top1 = {}       # {layer_idx: LongTensor[num_heads]} — результат последнего шага

    def reset(self):
        self.record = False
        self.block = {}
        self.top1 = {}


STATE = _AttnState()


def _repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def retrieval_attention_forward(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    """query: [B, Hq, q, D], key/value: [B, Hkv, k, D]. Возвращает ([B, q, Hq, D], None)."""
    layer = getattr(module, "layer_idx", None)
    if query.shape[2] == 1 and (STATE.record or STATE.block):
        heads = STATE.block.get(layer)
        if heads:
            query = query.clone()
            query[:, heads] = 0  # нулевой query -> равномерное внимание по всем ключам
        n_rep = query.shape[1] // key.shape[1]
        k = _repeat_kv(key, n_rep)
        v = _repeat_kv(value, n_rep)
        if scaling is None:
            scaling = query.shape[-1] ** -0.5
        scores = torch.matmul(query, k.transpose(-1, -2)).float() * scaling
        if attention_mask is not None:
            m = attention_mask[..., : k.shape[-2]]
            if m.dtype == torch.bool:
                scores = scores.masked_fill(~m, float("-inf"))
            else:
                scores = scores + m.float()
        probs = scores.softmax(dim=-1)
        if STATE.record and layer is not None:
            STATE.top1[layer] = probs[0, :, -1].argmax(dim=-1).cpu()
        out = torch.matmul(probs.to(v.dtype), v)
        return out.transpose(1, 2).contiguous(), None
    return sdpa_attention_forward(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs)


AttentionInterface.register(ATTN_NAME, retrieval_attention_forward)
try:  # маски в формате SDPA для нашей реализации (в новых версиях transformers)
    from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
    AttentionMaskInterface.register(ATTN_NAME, sdpa_mask)
except Exception:
    pass


def text_config(config):
    return config.get_text_config() if hasattr(config, "get_text_config") else config


def load_model_and_tokenizer(model_path):
    """Загружает Qwen3.5 (мультимодальный чекпоинт или текстовый) с нашей функцией внимания."""
    from transformers import AutoModelForCausalLM
    try:
        from transformers import AutoModelForImageTextToText as AutoMM
    except ImportError:
        AutoMM = None

    enc = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    kw = dict(dtype=torch.bfloat16, device_map="auto", attn_implementation=ATTN_NAME)

    # Официальный чекпоинт — Qwen3_5ForConditionalGeneration (с vision-энкодером, он просто не используется).
    loaders = []
    if "ConditionalGeneration" in str(getattr(config, "architectures", "")) and AutoMM is not None:
        loaders.append(AutoMM)
    loaders.append(AutoModelForCausalLM)
    last_err = None
    for loader in loaders:
        try:
            model = loader.from_pretrained(model_path, **kw).eval()
            break
        except (ValueError, KeyError) as e:
            last_err = e
    else:
        raise RuntimeError(f"Не удалось загрузить модель {model_path}: {last_err}")

    tcfg = text_config(model.config)
    layer_types = getattr(tcfg, "layer_types", None) or ["full_attention"] * tcfg.num_hidden_layers
    full_layers = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    head_num = tcfg.num_attention_heads
    print(f"loaded {type(model).__name__}; layers={tcfg.num_hidden_layers}, "
          f"full-attention layers={full_layers}, heads per layer={head_num}, kv heads={tcfg.num_key_value_heads}")
    return model, enc, full_layers, head_num


def input_device(model):
    return model.get_input_embeddings().weight.device


def stop_token_ids(enc):
    ids = set()
    for t in ["<|im_end|>", "<|endoftext|>"]:
        i = enc.convert_tokens_to_ids(t)
        if isinstance(i, int) and i != enc.unk_token_id:
            ids.add(i)
    if enc.eos_token_id is not None:
        ids.add(enc.eos_token_id)
    return ids


def period_token_ids(enc):
    """Все токены, которые декодируются в '.' (с пробелами/переносами вокруг)."""
    ids = set()
    for tok, i in enc.get_vocab().items():
        try:
            s = enc.convert_tokens_to_string([tok])
        except Exception:
            continue
        if s.strip() == ".":
            ids.add(i)
    return ids


def build_input_ids(enc, context, question, use_chat):
    if use_chat:
        messages = [{"role": "user", "content": f"<book>{context}</book>\nBased on the content of the book, Question: {question}\nAnswer:"}]
        try:
            text = enc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = enc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = enc(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    else:
        text = context + f"Based on the content of the book, Question: {question}\nAnswer:"
        ids = enc(text, return_tensors="pt")["input_ids"]
    return ids


def read_haystack(haystack_dir, enc, max_tokens):
    files = sorted(glob.glob(f"{haystack_dir}/*.txt"))
    if not files:
        raise FileNotFoundError(f"В {haystack_dir} нет .txt файлов — запустите setup.sh")
    context = ""
    while len(enc.encode(context)) < max_tokens:
        for f in files:
            with open(f, "r") as fh:
                context += fh.read()
    return context


def insert_needle(enc, context, needle, depth_percent, context_length, period_ids, buffer=200):
    tokens_needle = enc.encode(needle)
    tokens_context = enc.encode(context)
    context_length -= buffer
    if len(tokens_context) + len(tokens_needle) > context_length:
        tokens_context = tokens_context[: context_length - len(tokens_needle)]
    if depth_percent == 100:
        new_tokens = tokens_context + tokens_needle
    else:
        insertion_point = int(len(tokens_context) * (depth_percent / 100))
        while insertion_point > 0 and tokens_context[insertion_point - 1] not in period_ids:
            insertion_point -= 1
        print("insertion at %d" % insertion_point)
        new_tokens = tokens_context[:insertion_point] + tokens_needle + tokens_context[insertion_point:]
    return enc.decode(new_tokens)


def find_needle_idx(enc, prompt_ids, needle):
    needle_ids = enc(needle, add_special_tokens=False)["input_ids"]
    needle_set = set(needle_ids)
    span_len = len(needle_ids)
    ids = prompt_ids.tolist()
    for i in range(len(ids)):
        overlap = len(set(ids[i: i + span_len]) & needle_set) / len(needle_set)
        if overlap > 0.9:
            return i, i + span_len
    return -1, -1


@torch.no_grad()
def prefill_and_decode(model, enc, input_ids, decode_len=50, on_step=None):
    """Prefill через SDPA, затем жадное декодирование по одному токену.
    on_step(token_id) вызывается после каждого шага (STATE.top1 уже заполнен)."""
    dev = input_device(model)
    input_ids = input_ids.to(dev)
    stops = stop_token_ids(enc)
    record, block = STATE.record, STATE.block
    STATE.record, STATE.block = False, {}  # prefill без записи/маскирования, как в оригинале
    out = model(input_ids=input_ids[:, :-1], use_cache=True, return_dict=True, logits_to_keep=1)
    STATE.record, STATE.block = record, block
    past = out.past_key_values
    inp = input_ids[:, -1:]
    output = []
    for _ in range(decode_len):
        out = model(input_ids=inp, past_key_values=past, use_cache=True, return_dict=True)
        past = out.past_key_values
        tok = out.logits[0, -1].argmax()
        output.append(tok.item())
        if on_step is not None:
            on_step(tok)
        if tok.item() in stops:
            break
        piece = enc.decode([tok.item()])
        if "\n" in piece and enc.decode(output, skip_special_tokens=True).strip():
            break
        inp = tok.view(1, 1)
    return enc.decode(output, skip_special_tokens=True).strip()