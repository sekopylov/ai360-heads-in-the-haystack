"""Model output conventions, independent of loading and cached generation."""
from dataclasses import dataclass

from .base import Prompt


@dataclass(frozen=True)
class AnswerContent:
    text: str
    final_answer_started: bool = True


class PlainOutputPolicy:
    def __init__(self, tokenizer, attention_scope: str):
        self.tokenizer = tokenizer
        self.attention_scope = attention_scope

    def attention_filter(self, prompt: Prompt):
        return lambda token_id: True

    def answer(self, prompt: Prompt, generated: list[int]) -> AnswerContent:
        return AnswerContent(
            self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        )


class QwenThinkingOutputPolicy(PlainOutputPolicy):
    """Preserve Qwen3's existing marker detection and final-answer extraction."""

    def attention_filter(self, prompt: Prompt):
        if self.attention_scope == "all_decode_tokens":
            return super().attention_filter(prompt)
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
                return False
            if opening and recent[-len(opening):] == opening:
                thinking = True
                return False
            return not thinking

        return publish

    def answer(self, prompt: Prompt, generated: list[int]) -> AnswerContent:
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
            answer_ids = []
        else:
            answer_ids = generated
        return AnswerContent(
            self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip(),
            final_answer_started=not thinking or closing_end is not None,
        )
