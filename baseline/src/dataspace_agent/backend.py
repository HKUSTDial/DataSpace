from __future__ import annotations

import os
from typing import Any, Protocol

from .config import ModelConfig
from .types import ModelTurn, ToolCall


class ModelBackend(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        timeout_seconds: float | None = None,
    ) -> ModelTurn:
        ...


class OpenAICompatibleBackend:
    """Chat Completions backend for OpenRouter and compatible endpoints."""

    def __init__(self, config: ModelConfig):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required for API runs. "
                "Install the baseline package before running."
            ) from exc

        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"environment variable {config.api_key_env} is not set"
            )
        self.config = config
        client_args: dict[str, Any] = {"api_key": api_key}
        if config.base_url:
            client_args["base_url"] = config.base_url
        if config.default_headers:
            client_args["default_headers"] = config.default_headers
        self.client = OpenAI(**client_args)

    def cancel(self) -> None:
        """Close the transport so an interrupted batch does not await I/O."""

        self.client.close()

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        timeout_seconds: float | None = None,
    ) -> ModelTurn:
        request: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": self.config.max_output_tokens,
        }
        if self.config.temperature is not None:
            request["temperature"] = self.config.temperature
        if self.config.extra_body:
            request["extra_body"] = self.config.extra_body
        if timeout_seconds is not None:
            request["timeout"] = max(1.0, float(timeout_seconds))

        response = self.client.chat.completions.create(**request)
        choice = response.choices[0]
        message = choice.message

        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": message.content or "",
        }
        message_payload = message.model_dump(exclude_none=True)
        for field in ("reasoning", "reasoning_details"):
            if field in message_payload:
                assistant_message[field] = message_payload[field]
        tool_calls: list[ToolCall] = []
        serialized_calls: list[dict[str, Any]] = []
        for call in message.tool_calls or []:
            serialized_calls.append(
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments or "{}",
                    },
                }
            )
            tool_calls.append(
                ToolCall(
                    call_id=call.id,
                    name=call.function.name,
                    arguments_json=call.function.arguments or "{}",
                )
            )
        if serialized_calls:
            assistant_message["tool_calls"] = serialized_calls

        usage = (
            response.usage.model_dump(exclude_none=True)
            if response.usage is not None
            else {}
        )
        return ModelTurn(
            assistant_message=assistant_message,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=choice.finish_reason,
            response_id=getattr(response, "id", None),
            resolved_model=getattr(response, "model", None),
        )
