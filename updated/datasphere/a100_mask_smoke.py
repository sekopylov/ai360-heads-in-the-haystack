"""Isolated A100 benchmark; does not change the production masking backend."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
import sys
import time
from types import MethodType

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AttentionInterface, AttentionMaskInterface
from transformers.masking_utils import sdpa_mask

from retrieval_heads.attention import AttentionRequest
from retrieval_heads.attention.backend import observable_eager_attention
from retrieval_heads.attention.controller import AttentionController
from retrieval_heads.experiment.data import ContextBuilder, load_validation_cases
from retrieval_heads.experiment.runner import ExperimentRunner
from retrieval_heads.experiment.scoring import rank_heads
from retrieval_heads.experiment.storage import read_json, result_payload, write_json
from retrieval_heads.models import create_model

# Transformers treats any backend name containing 'flash' as its external
# flash-attn package, even when registered. These are native PyTorch SDPA.
FLASH_PREFILL = "a100_smoke_sdpa_prefill"
FLASH_DECODE = "a100_smoke_sdpa_decode"


def flash_prefill(module, query, key, value, attention_mask, scaling=None,
                  dropout=0.0, is_causal=None, **kwargs):
    if attention_mask is not None:
        raise ValueError("Smoke requires an unpadded prompt without an explicit mask")
    causal = getattr(module, "is_causal", True) if is_causal is None else is_causal
    # No math fallback, no physical KV repetition: native Flash GQA.
    with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
        output = F.scaled_dot_product_attention(
            query, key, value, dropout_p=dropout, scale=scaling,
            is_causal=bool(query.shape[2] > 1 and causal),
            enable_gqa=query.shape[1] != key.shape[1],
        )
    return output.transpose(1, 2).contiguous(), None


def flash_decode(module, query, key, value, attention_mask, scaling=None,
                 dropout=0.0, retrieval_attention_controller=None, **kwargs):
    controller = retrieval_attention_controller
    if query.shape[2] != 1 or attention_mask is not None or dropout:
        raise ValueError("Flash smoke supports only unpadded single-token inference")
    if controller is None or controller.request.capture != "none" or controller.request.needle_span is not None:
        raise ValueError("Flash smoke is masking-only, without attention capture")
    if controller.request.mask_mode != "legacy_uniform":
        raise ValueError("Flash smoke only implements legacy_uniform")
    blocked = controller.blocked_heads(int(module.layer_idx))
    if blocked:
        # Q=0 makes every QK logit zero before softmax. K/V stay shared GQA.
        query = query.clone()
        query[:, blocked, :, :] = 0
    with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
        output = F.scaled_dot_product_attention(
            query, key, value, scale=scaling, dropout_p=0.0,
            is_causal=False, enable_gqa=query.shape[1] != key.shape[1],
        )
    return output.transpose(1, 2).contiguous(), None


@contextmanager
def flash_observable_decode(self):
    previous = self.model.config._attn_implementation
    self.model.set_attn_implementation(FLASH_DECODE)
    try:
        yield
    finally:
        self.model.set_attn_implementation(previous)


def check_kernel():
    from types import SimpleNamespace
    torch.manual_seed(42)
    query = torch.randn(1, 32, 1, 128, device="cuda", dtype=torch.float16)
    key = torch.randn(1, 8, 4096, 128, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    controller = AttentionController((0,))
    controller.start(AttentionRequest(blocked_heads=frozenset({(0, 1), (0, 7)})))
    controller.begin_step()
    args = (SimpleNamespace(layer_idx=0, num_key_value_groups=4, training=False),
            query, key, value, None)
    eager, _ = observable_eager_attention(*args, retrieval_attention_controller=controller)
    flash, _ = flash_decode(*args, retrieval_attention_controller=controller)
    torch.testing.assert_close(flash, eager, rtol=0.01, atol=0.002)
    error = float((flash - eager).abs().max())
    controller.end_step(0)
    controller.finish()
    return {"max_absolute_error": error, "rtol": 0.01, "atol": 0.002}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--head-scores", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError("This smoke requires an Ampere-or-newer CUDA GPU")
    AttentionInterface.register(FLASH_PREFILL, flash_prefill)
    AttentionInterface.register(FLASH_DECODE, flash_decode)
    for name in (FLASH_PREFILL, FLASH_DECODE):
        AttentionMaskInterface.register(name, sdpa_mask)
    report = {"complete": False, "gpu": torch.cuda.get_device_name(),
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "kernel_check": check_kernel(), "results": []}
    print(f"[runtime] {report}", flush=True)
    started = time.perf_counter()
    model = create_model("qwen3_8b_no_yarn_64k", model_id=args.model,
                         device_map="cuda:0", dtype="float16",
                         prefill_attention="sdpa_memory_efficient",
                         attention_scope="all_decode_tokens")
    report["model_load_seconds"] = time.perf_counter() - started
    ranked = [head for head, _ in rank_heads(read_json(args.head_scores))
              if head in set(model.eligible_heads)]
    if len(ranked) < 64:
        raise ValueError("Need at least 64 eligible ranked heads")
    request = AttentionRequest(blocked_heads=frozenset(ranked[:64]))
    cases = {case.case_id: case for case in load_validation_cases(args.validation_root)}
    builder = ContextBuilder(model.tokenizer, max_context_length=48000,
                             period_tokens=model.period_tokens, context_seed=42,
                             random_start=True)
    runner = ExperimentRunner(model, builder, None, max_new_tokens=512)
    # Warm up both kernels on a short input; excluded from reported comparisons.
    short = runner.prepare(cases["eugene-onegin-glass-compass"], context_length=1000, depth_percent=45)
    original_decode = model._observable_decode
    for backend in ("eager", "flash"):
        model.prefill_attention = "sdpa_memory_efficient" if backend == "eager" else FLASH_PREFILL
        model.model.set_attn_implementation(model.prefill_attention if backend == "flash" else
                                           "retrieval_heads_sdpa_memory_efficient")
        model._observable_decode = original_decode if backend == "eager" else MethodType(flash_observable_decode, model)
        model.generate(short.prompt, max_new_tokens=8, attention=request)
    jobs = [("eugene-onegin-glass-compass", length, backend, 512)
            for length in (32000, 48000) for backend in ("eager", "flash")]
    # Real planned corpora, full output budget, to check quality and output length.
    jobs += [(case_id, 48000, "flash", 2048) for case_id in
             ("paul-graham-velvet-pelican", "hero-of-our-time-porcelain-beetle")]
    for index, (case_id, length, backend, budget) in enumerate(jobs, 1):
        runner.max_new_tokens = budget
        prepared = runner.prepare(cases[case_id], context_length=length, depth_percent=45)
        model.prefill_attention = "sdpa_memory_efficient" if backend == "eager" else FLASH_PREFILL
        model.model.set_attn_implementation("retrieval_heads_sdpa_memory_efficient" if backend == "eager" else FLASH_PREFILL)
        model._observable_decode = original_decode if backend == "eager" else MethodType(flash_observable_decode, model)
        # CUDA events separate prefill/decode without synchronizing every token.
        events = []
        def before_forward(module, inputs):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            events.append((begin, end))
        def after_forward(module, inputs, output):
            events[-1][1].record()
        before = model.model.register_forward_pre_hook(before_forward)
        after = model.model.register_forward_hook(after_forward)
        print(f"[smoke] {index}/{len(jobs)} {backend} {case_id} length={length} budget={budget}", flush=True)
        try:
            torch.cuda.synchronize()
            result = runner.run(prepared, attention=request)
            torch.cuda.synchronize()
        finally:
            before.remove()
            after.remove()
        payload = result_payload(result, args.model, experiment={
            "adapter": "qwen3_8b_no_yarn_64k", "backend": backend,
            "mask_mode": "legacy_uniform", "blocked_heads": sorted(request.blocked_heads),
            "context_seed": 42, "max_new_tokens": budget,
        })
        payload.update(generated_tokens=len(result.generation.token_ids),
                       peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                       prefill_gpu_seconds=events[0][0].elapsed_time(events[0][1]) / 1000,
                       decode_gpu_seconds=sum(begin.elapsed_time(end) for begin, end in events[1:]) / 1000)
        payload["tokens_per_second"] = payload["generated_tokens"] / payload["test_duration_seconds"]
        write_json(args.output_root / f"{index}_{backend}_{case_id}_{length}.json", payload)
        report["results"].append(payload)
        write_json(args.output_root / "summary.json", report)
        print(f"[result] seconds={result.duration_seconds:.3f} tokens={payload['generated_tokens']} "
              f"tok/s={payload['tokens_per_second']:.2f} prefill={payload['prefill_gpu_seconds']:.3f} "
              f"decode={payload['decode_gpu_seconds']:.3f} score={result.score:.2f}", flush=True)
    report["complete"] = True
    write_json(args.output_root / "summary.json", report)


if __name__ == "__main__":
    main()
