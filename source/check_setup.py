#!/usr/bin/env python
"""Этап 0: проверка окружения, данных и конфига модели перед долгими прогонами."""
import argparse
import importlib
import os
import sys

from rh_common import DEFAULT_MODEL, setup_logger, stage


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--haystack_dir", default="./haystack_for_detect")
    p.add_argument("--eval_haystack_dir", default="./PaulGrahamEssays")
    a = p.parse_args()
    log = setup_logger("00_setup")

    with stage(log, "Проверка пакетов"):
        missing = []
        for pkg in ("torch", "transformers", "numpy", "matplotlib", "rouge_score"):
            try:
                m = importlib.import_module(pkg)
                log.info(f"  {pkg:<13} {getattr(m, '__version__', 'ok')}")
            except ImportError:
                missing.append(pkg)
                log.error(f"  {pkg:<13} НЕ УСТАНОВЛЕН")
        if missing:
            log.error(f"Установи: pip install -r requirements.txt  (не хватает: {missing})")
            sys.exit(1)
        import torch
        import transformers
        from packaging import version
        if version.parse(transformers.__version__) < version.parse("4.48"):
            log.error("Нужен transformers>=4.48 (eager-attention должен возвращать веса)")
            sys.exit(1)
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                pr = torch.cuda.get_device_properties(i)
                log.info(f"  GPU {i}: {pr.name}, {pr.total_memory / 2**30:.1f} GB")
        else:
            log.warning("  GPU не найден -- на CPU прогон будет очень долгим, уменьши длины/сетку")

    with stage(log, "Проверка данных"):
        nd = os.path.join(a.haystack_dir, "needles.jsonl")
        if os.path.exists(nd):
            n = sum(1 for l in open(nd, encoding="utf-8") if l.strip())
            parts = [os.path.join(a.haystack_dir, f"part{i + 1}") for i in range(n)]
            log.info(f"  детекция: {nd} -- {n} игл, part-папки: "
                     f"{sum(os.path.isdir(x) for x in parts)}/{n}")
        else:
            log.warning(f"  {nd} не найден -- детекция пойдёт на одной игле (San Francisco)")
        if os.path.isdir(a.eval_haystack_dir):
            n = len([f for f in os.listdir(a.eval_haystack_dir) if f.endswith(".txt")])
            log.info(f"  оценка маскирования: {a.eval_haystack_dir} -- {n} txt-файлов")
        else:
            log.warning(f"  {a.eval_haystack_dir} не найден -- для оценки будет синтетический стог")

    with stage(log, "Проверка конфига модели"):
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(a.model_path)
        lt = getattr(cfg, "layer_types", None)
        log.info(f"  {a.model_path}: {cfg.model_type}, {cfg.num_hidden_layers} слоёв x "
                 f"{cfg.num_attention_heads} голов, layer_types="
                 f"{sorted(set(lt)) if lt else 'все full_attention'}")


if __name__ == "__main__":
    main()