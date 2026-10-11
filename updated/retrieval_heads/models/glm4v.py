"""Text-only GLM-4.6V-Flash experiments with native no-thinking prompts."""
import re
import torch
from transformers import Glm4vForConditionalGeneration

from .base import Prompt
from .output import AnswerContent, PlainOutputPolicy
from .transformers_common import TransformersAdapter
from ..experiment.prompting import context_only_messages


class Glm4vOutputPolicy(PlainOutputPolicy):
    """GLM Thinking emits <think>…</think><answer>…</answer>.

    These markers are NOT special tokens in the official tokenizer, so
    skip_special_tokens alone cannot produce an answer suitable for scoring.
    Untagged output (including Base checkpoints) remains supported.
    """

    def attention_filter(self, prompt: Prompt):
        if self.attention_scope == "all_decode_tokens":
            return super().attention_filter(prompt)
        markers = {marker: self.tokenizer.encode(marker, add_special_tokens=False)
                   for marker in ("<think>", "</think>", "<answer>", "</answer>")}
        recent = []
        width = max(1, *(len(ids) for ids in markers.values()))
        thinking = False

        def publish(token_id):
            nonlocal thinking
            recent.append(token_id)
            del recent[:-width]
            for marker, ids in markers.items():
                if ids and recent[-len(ids):] == ids:
                    if marker == "<think>":
                        thinking = True
                    elif marker in ("</think>", "<answer>"):
                        thinking = False
                    return False
            return not thinking

        return publish

    def answer(self, prompt: Prompt, generated: list[int]) -> AnswerContent:
        raw = self.tokenizer.decode(generated, skip_special_tokens=True)
        if "<answer>" in raw:
            text = raw.split("<answer>", 1)[1].split("</answer>", 1)[0]
            return AnswerContent(text.strip(), final_answer_started=True)
        if "<think>" in raw:
            if "</think>" not in raw:
                return AnswerContent("", final_answer_started=False)
            raw = raw.split("</think>", 1)[1]
        return AnswerContent(raw.split("</answer>", 1)[0].strip())


class Glm46VFlashNoThinkingAdapter(TransformersAdapter):
    """Native multimodal wrapper, text-only input and nested text Q heads.

    The official 4.6V-Flash template supports enable_thinking=False: it adds
    /nothink to the user turn and an empty think block to the assistant prefix.
    Keep the checkpoint's partial M-RoPE, GQA and context limit intact.
    """

    model_class = Glm4vForConditionalGeneration
    supports_flash_decode = True
    output_policy_class = Glm4vOutputPolicy
    supported_attention_scopes = frozenset({"all_decode_tokens", "answer_only"})

    def __init__(self, model_id: str = "zai-org/GLM-4.6V-Flash", **kwargs):
        super().__init__(model_id, **kwargs)
        for config in (self._language_config(), self.model.generation_config):
            ids = getattr(config, "eos_token_id", None)
            if ids is not None:
                self._eos_token_ids.update(
                    int(token) for token in (ids if isinstance(ids, (list, tuple, set)) else [ids])
                )

    def _discover_attention_layers(self) -> tuple[int, ...]:
        if self.model.config.model_type != "glm4v":
            raise ValueError("The glm4_6v_flash_no_thinking adapter requires a native glm4v checkpoint")
        if self._language_config().model_type != "glm4v_text":
            raise ValueError("The GLM-4.6V-Flash checkpoint requires a glm4v_text decoder")
        return tuple(range(len(self._language_model.layers)))

    def _language_config(self):
        return self.model.config.text_config

    @property
    def _language_model(self):
        return self.model.model.language_model

    @property
    def input_device(self) -> torch.device:
        return self._language_model.embed_tokens.weight.device

    def encode_prompt(self, context: str, question: str) -> Prompt:
        encoded = self.tokenizer.apply_chat_template(
            context_only_messages(context, question),
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=False,
        )
        input_ids = encoded["input_ids"]
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        # Fail loudly for old/incompatible templates that silently ignore the
        # kwarg (e.g. GLM-4.1V). Do not silently run a thinking experiment.
        tail = self.tokenizer.decode(input_ids[0, -32:].tolist(), skip_special_tokens=False)
        if re.search(r"<think>\s*</think>\s*$", tail) is None:
            raise ValueError(
                "GLM no-thinking requires the GLM-4.6V-Flash chat template "
                "with enable_thinking=False support"
            )
        return Prompt(input_ids=input_ids, token_ids=input_ids[0].tolist())
