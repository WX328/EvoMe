import json
from typing import Any

from json_repair import repair_json
from openai import OpenAI

from app.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    EVOME_MAX_TOKENS_CHAT,
    EVOME_MAX_TOKENS_CONTEXT,
    EVOME_MAX_TOKENS_EVALUATION,
    EVOME_MAX_TOKENS_GATE,
    EVOME_MAX_TOKENS_JSON_DEFAULT,
    EVOME_MAX_TOKENS_REFLECTION,
    EVOME_MAX_TOKENS_POST_TURN,
    EVOME_MAX_TOKENS_RELATION,
    EVOME_MAX_TOKENS_REVISION,
)
from app.telemetry import GLOBAL_TOKEN_USAGE


class DeepSeekClient:
    def __init__(self) -> None:
        if not DEEPSEEK_API_KEY:
            raise RuntimeError("缺少 DEEPSEEK_API_KEY。请检查 .env 文件。")
        self.client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    @staticmethod
    def _system_text(messages: list[dict[str, str]]) -> str:
        return "\n".join(
            str(item.get("content", ""))
            for item in messages
            if str(item.get("role", "")) == "system"
        ).lower()

    @classmethod
    def _infer_purpose(cls, messages: list[dict[str, str]], *, fallback: str) -> str:
        text = cls._system_text(messages)
        markers = (
            ("learning_eligibility_gate", "learning_eligibility"),
            ("personal memory gate", "personal_memory_gate"),
            ("evome 的 experience gate", "experience_gate"),
            ("evoMe 的 evaluator".lower(), "evaluation"),
            ("context_builder", "context_builder"),
            ("module: reflection", "reflection"),
            ("module: post_turn_assessor", "post_turn_assessor"),
            ("hypothesis_relation_judge", "relation_judge"),
            ("proposed hypothesis revision", "revision_analysis"),
            ("candidate revision", "candidate_validation"),
            ("policy normalization", "policy_normalization"),
            ("ancestor", "ancestor_revalidation"),
        )
        for marker, purpose in markers:
            if marker in text:
                return purpose
        return fallback

    @staticmethod
    def _thinking_enabled_for(purpose: str) -> bool:
        return purpose in {
            "fast_chat",
            "revision_analysis",
            "candidate_validation",
            "policy_normalization",
            "ancestor_revalidation",
        }

    @staticmethod
    def _max_tokens_for(purpose: str) -> int:
        if purpose == "fast_chat":
            return EVOME_MAX_TOKENS_CHAT
        if purpose == "evaluation":
            return EVOME_MAX_TOKENS_EVALUATION
        if purpose in {"learning_eligibility", "personal_memory_gate", "experience_gate"}:
            return EVOME_MAX_TOKENS_GATE
        if purpose == "context_builder":
            return EVOME_MAX_TOKENS_CONTEXT
        if purpose == "reflection":
            return EVOME_MAX_TOKENS_REFLECTION
        if purpose == "post_turn_assessor":
            return EVOME_MAX_TOKENS_POST_TURN
        if purpose == "relation_judge":
            return EVOME_MAX_TOKENS_RELATION
        if purpose in {
            "revision_analysis",
            "candidate_validation",
            "policy_normalization",
            "ancestor_revalidation",
        }:
            return EVOME_MAX_TOKENS_REVISION
        return EVOME_MAX_TOKENS_JSON_DEFAULT

    @staticmethod
    def _record_usage(*, purpose: str, response: Any) -> None:
        usage = getattr(response, "usage", None)
        model = str(getattr(response, "model", DEEPSEEK_MODEL) or DEEPSEEK_MODEL)
        choices = getattr(response, "choices", None) or []
        finish_reason = (
            str(getattr(choices[0], "finish_reason", "") or "") if choices else ""
        )
        GLOBAL_TOKEN_USAGE.record_usage(
            purpose=purpose,
            model=model,
            usage=usage,
            finish_reason=finish_reason,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.4,
        *,
        purpose: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        resolved_purpose = purpose or self._infer_purpose(messages, fallback="fast_chat")
        response = self.client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens or self._max_tokens_for(resolved_purpose),
            stream=False,
        )
        self._record_usage(purpose=resolved_purpose, response=response)
        return response.choices[0].message.content or ""

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.4,
        *,
        purpose: str = "fast_chat",
        max_tokens: int | None = None,
    ):
        response = self.client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens or self._max_tokens_for(purpose),
            stream=True,
            stream_options={"include_usage": True},
        )
        final_usage = None
        final_model = DEEPSEEK_MODEL
        finish_reason = ""
        for chunk in response:
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                final_usage = usage
            final_model = str(getattr(chunk, "model", final_model) or final_model)
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            chunk_finish_reason = getattr(choice, "finish_reason", None)
            if chunk_finish_reason:
                finish_reason = str(chunk_finish_reason)
            content = choice.delta.content or ""
            if content:
                yield content
        GLOBAL_TOKEN_USAGE.record_usage(
            purpose=purpose,
            model=final_model,
            usage=final_usage,
            finish_reason=finish_reason,
        )

    def json_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        *,
        purpose: str | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        resolved_purpose = purpose or self._infer_purpose(messages, fallback="json_internal")
        response = self.client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens or self._max_tokens_for(resolved_purpose),
            response_format={"type": "json_object"},
            extra_body={
                "thinking": {
                    "type": (
                        "enabled"
                        if self._thinking_enabled_for(resolved_purpose)
                        else "disabled"
                    )
                }
            },
            stream=False,
        )
        self._record_usage(purpose=resolved_purpose, response=response)
        choice = response.choices[0]
        finish_reason = str(getattr(choice, "finish_reason", "") or "")
        if finish_reason == "length":
            print(f"\n⚠️ LLM JSON 输出触发长度上限：purpose={resolved_purpose}")
        content = (choice.message.content or "{}").strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            print("\n⚠️ LLM 返回了非标准 JSON")
            print(f"JSON Error: {exc}")
            print(f"Raw content:\n{content}\n")

        try:
            repaired = repair_json(content, return_objects=True)
            if isinstance(repaired, dict):
                return repaired
        except Exception as exc:
            print(f"JSON repair failed: {exc}")

        return {}
