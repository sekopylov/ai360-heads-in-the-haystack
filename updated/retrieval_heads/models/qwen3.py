"""Dense Qwen3 with observed attention and shared thinking answer extraction."""
from transformers import Qwen3ForCausalLM

from .base import GenerationResult, Prompt
from .qwen_common import QwenAdapter


class Qwen3Adapter(QwenAdapter):
    model_class = Qwen3ForCausalLM
    attention_scope = "answer_only"

    def __init__(self, model_id: str = "Qwen/Qwen3-4B-Thinking-2507", **kwargs):
        super().__init__(model_id, **kwargs)

    def _discover_attention_layers(self) -> tuple[int, ...]:
        if self.model.config.model_type != "qwen3":
            raise ValueError("The qwen3 adapter requires a dense Qwen3 checkpoint")
        return tuple(index for index, layer in enumerate(self.model.model.layers)
                     if getattr(layer.self_attn, "sliding_window", None) is None)

    def _stop_reason(self, token_id: int) -> str | None:
        # Thinking includes newlines; it must be allowed to finish naturally.
        return "eos" if token_id in self._eos_token_ids else None

    def _attention_filter(self, prompt: Prompt):
        if self.attention_scope == "all_decode_tokens":
            return super()._attention_filter(prompt)
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

    def _generation_result(self, prompt: Prompt, generated: list[int], reason: str) -> GenerationResult:
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
        )
