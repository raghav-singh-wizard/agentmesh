"""Thin async wrapper around Anthropic's Messages API with retries.

Exposes a single `call()` method that supports tool-use (native Claude tool
schema). The orchestrator uses this for both planning and critique.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from anthropic import AsyncAnthropic
from anthropic.types import Message
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from agentmesh.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class LLMResponse:
    """Normalised result from an LLM call."""

    text: str                             # concatenated text blocks
    tool_uses: list[dict[str, Any]]       # list of {id, name, input}
    stop_reason: str | None
    tokens_in: int
    tokens_out: int
    raw: Message


class AnthropicLLM:
    """Async client for Claude with tool-use support and automatic retry."""

    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        max_retries: int = 3,
    ) -> None:
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set.")
        self._client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries

    async def call(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Invoke the model once. Retries transient errors."""

        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": system,
            "messages": messages,
        }
        if tools:
            params["tools"] = tools

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        ):
            with attempt:
                message = await self._client.messages.create(**params)
                return self._normalise(message)

        raise RuntimeError("unreachable: retry loop exited without response")

    @staticmethod
    def _normalise(message: Message) -> LLMResponse:
        text_parts: list[str] = []
        tool_uses: list[dict[str, Any]] = []

        for block in message.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "tool_use":
                tool_uses.append(
                    {"id": block.id, "name": block.name, "input": dict(block.input or {})}
                )

        usage = message.usage
        return LLMResponse(
            text="\n".join(text_parts).strip(),
            tool_uses=tool_uses,
            stop_reason=message.stop_reason,
            tokens_in=getattr(usage, "input_tokens", 0),
            tokens_out=getattr(usage, "output_tokens", 0),
            raw=message,
        )

    async def aclose(self) -> None:
        await self._client.close()
