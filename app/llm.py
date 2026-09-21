import json

from json_repair import repair_json
from openai import OpenAI

from app.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
)


class DeepSeekClient:

    def __init__(self) -> None:

        if not DEEPSEEK_API_KEY:
            raise RuntimeError(
                "缺少 DEEPSEEK_API_KEY。"
                "请检查 .env 文件。"
            )

        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )


    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.4,
    ) -> str:

        response = self.client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            temperature=temperature,
            stream=False,
        )

        return (
            response.choices[0].message.content
            or ""
        )


    def stream_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.4,
    ):
        response = self.client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            temperature=temperature,
            stream=True,
        )

        for chunk in response:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content or ""
            if content:
                yield content


    def json_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
    ) -> dict:

        response = self.client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            temperature=temperature,
            response_format={
                "type": "json_object"
            },
            stream=False,
        )

        content = (
            response.choices[0].message.content
            or "{}"
        ).strip()

        # -----------------------------
        # 1. 优先尝试严格 JSON
        # -----------------------------
        try:

            return json.loads(content)

        except json.JSONDecodeError as e:

            print(
                "\n⚠️ LLM 返回了非标准 JSON"
            )

            print(
                f"JSON Error: {e}"
            )

            print(
                f"Raw content:\n{content}\n"
            )

        # -----------------------------
        # 2. 尝试自动修复
        # -----------------------------
        try:

            repaired = repair_json(
                content,
                return_objects=True
            )

            if isinstance(
                repaired,
                dict
            ):
                return repaired

        except Exception as e:

            print(
                f"JSON repair failed: {e}"
            )

        # -----------------------------
        # 3. 最终降级
        # -----------------------------
        return {}
