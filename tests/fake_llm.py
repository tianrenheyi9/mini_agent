from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from app.agent.llm import LLMResponse


class FakeLLM:
    """Scripted LLM used by tests and local development."""

    def __init__(self, responses: Iterable[str | Exception]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict[str, Any]]] = []

    async def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        temperature: float = 0.0,
    ) -> LLMResponse:
        self.calls.append([dict(message) for message in messages])
        if not self._responses:
            raise AssertionError("FakeLLM has no scripted response left")

        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item

        return LLMResponse(
            content=item,
            model="fake",
            prompt_tokens=None,
            completion_tokens=None,
            finish_reason="stop",
            latency_ms=0,
        )
