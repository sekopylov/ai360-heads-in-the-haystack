"""Frozen pre-refactor decode and Qwen3 policies, for numerical parity checks."""
from __future__ import annotations

import torch

from retrieval_heads.attention.backend import CONTROLLER_ARGUMENT
from retrieval_heads.attention.controller import validate_blocked_heads
from retrieval_heads.attention.types import AttentionObserver, AttentionRequest
from retrieval_heads.models.base import GenerationResult, Prompt


def historical_encode_prompt(self, context: str, question: str) -> Prompt:
    messages = [
        {
            "role": "user",
            "content": (
                f"<book>{context}</book>\n"
                f"Based on the content of the book, Question: {question}\n"
                "Answer:"
            ),
        }
    ]
    text = self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **self._chat_template_kwargs(),
    )
    encoded = self.tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    return Prompt(
        input_ids=input_ids,
        token_ids=input_ids[0].tolist(),
    )


def historical_filter(self, prompt: Prompt):
    if self.attention_scope == "all_decode_tokens":
        return lambda token_id: True
    prompt_tail = self.tokenizer.decode(prompt.token_ids[-32:], skip_special_tokens=False)
    thinking = prompt_tail.rfind("<think>") > prompt_tail.rfind("</think>")
    opening = self.tokenizer.encode("<think>", add_special_tokens=False)
    closing = self.tokenizer.encode("</think>", add_special_tokens=False)
    recent: list[int] = []
    width = max(len(opening), len(closing), 1)

    def publish(token_id: int) -> bool:
        nonlocal thinking
        recent.append(token_id)
        del recent[:-width]
        if closing and recent[-len(closing):] == closing:
            thinking = False
            return False  # The closing marker itself is not an answer token.
        if opening and recent[-len(opening):] == opening:
            thinking = True
            return False
        return not thinking

    return publish


def historical_result(self, prompt: Prompt, generated: list[int], reason: str) -> GenerationResult:
    raw = self.tokenizer.decode(generated, skip_special_tokens=False)
    prompt_tail = self.tokenizer.decode(prompt.token_ids[-32:], skip_special_tokens=False)
    thinking = prompt_tail.rfind("<think>") > prompt_tail.rfind("</think>") or "<think>" in raw
    closing_ids = self.tokenizer.encode("</think>", add_special_tokens=False)
    closing_end = None
    for index in range(len(generated) - len(closing_ids) + 1):
        if closing_ids and generated[index:index + len(closing_ids)] == closing_ids:
            closing_end = index + len(closing_ids)
            break
    if closing_end is not None:
        answer_ids = generated[closing_end:]
    elif thinking:
        # An unfinished thought is not a final answer, even if it quotes the needle.
        answer_ids = []
    else:
        answer_ids = generated
    return GenerationResult(
        token_ids=generated,
        text=self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip(),
        raw_text=raw,
        finish_reason=reason,
        final_answer_started=not thinking or closing_end is not None,
    )


@torch.inference_mode()
def historical_generate(
    self,
    prompt: Prompt,
    *,
    max_new_tokens: int,
    attention: AttentionRequest,
    observer: AttentionObserver | None = None,
) -> GenerationResult:
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if self.decode_attention == "sdpa_flash":
        from retrieval_heads.attention.flash import validate_flash_request
        validate_flash_request(attention, observer)
    if attention.mask_mode not in {"zero_output", "legacy_uniform"}:
        raise ValueError(f"Unknown mask mode: {attention.mask_mode!r}")
    validate_blocked_heads(attention.blocked_heads, self.eligible_heads)
    input_ids = prompt.input_ids.to(self.input_device)
    if input_ids.shape[1] < 2:
        raise ValueError("Prompt must contain at least two tokens")
    if input_ids.shape[1] + max_new_tokens > self._language_config().max_position_embeddings:
        raise ValueError("Prompt and generated tokens exceed the model context limit")

    if input_ids.is_cuda:
        torch.cuda.reset_peak_memory_stats(input_ids.device)
    # Unobserved prefill; its kernel is selected independently of decode.
    prefill = self.model(
        input_ids=input_ids[:, :-1],
        use_cache=True,
        logits_to_keep=1,
    )
    cache = prefill.past_key_values
    if input_ids.is_cuda:
        print(f"[memory] prefill={self.prefill_attention} "
              f"peak_allocated_gib={torch.cuda.max_memory_allocated(input_ids.device) / 1024**3:.3f}",
              flush=True)
    current_token = input_ids[:, -1:]
    generated: list[int] = []
    finish_reason = "max_new_tokens"
    publish_attention = self._attention_filter(prompt)

    self.attention.start(attention, observer)
    try:
        with self._observable_decode():
            for _ in range(max_new_tokens):
                self.attention.begin_step()
                outputs = self.model(
                    input_ids=current_token,
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                    **{CONTROLLER_ARGUMENT: self.attention},
                )
                cache = outputs.past_key_values
                current_token = outputs.logits[:, -1, :].argmax(
                    dim=-1,
                    keepdim=True,
                )
                token_id = int(current_token.item())
                generated.append(token_id)
                self.attention.end_step(token_id, publish=publish_attention(token_id))

                reason = self._stop_reason(token_id)
                if reason is not None:
                    finish_reason = reason
                    break
    finally:
        self.attention.finish()

    if input_ids.is_cuda:
        print(f"[memory] generation peak_allocated_gib="
              f"{torch.cuda.max_memory_allocated(input_ids.device) / 1024**3:.3f}", flush=True)
    return self._generation_result(prompt, generated, finish_reason)
