from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)

RETRY_DELAYS_SECONDS = (0.5, 1.5)
RETRYABLE_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})


async def _sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


class LLMError(Exception):
    """Raised when an LLM call cannot be completed."""


class LLMConfigError(LLMError):
    pass


class LLMRequestError(LLMError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        attempts: int = 1,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.attempts = attempts


class LLMResponseError(LLMError):
    pass


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    finish_reason: str | None
    latency_ms: int


class LLMClient:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._transport = transport

    async def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        temperature: float = 0.0,
    ) -> LLMResponse:
        if not self._settings.llm_configured:
            raise LLMConfigError(
                "LLM is not configured: set LLM_BASE_URL, LLM_API_KEY and LLM_MODEL"
            )

        url = f"{self._settings.llm_base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self._settings.llm_model,
            "messages": list(messages),
            "temperature": temperature,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self._settings.llm_api_key}",
            "Content-Type": "application/json",
        }
        timeout = httpx.Timeout(
            self._settings.llm_timeout_seconds,
            connect=min(10.0, self._settings.llm_timeout_seconds),
        )
        client_kwargs: dict[str, Any] = {
            "timeout": timeout,
            "transport": self._transport,
        }
        if self._settings.llm_proxy:
            client_kwargs["proxy"] = self._settings.llm_proxy

        attempts = len(RETRY_DELAYS_SECONDS) + 1
        started = time.perf_counter()

        async with httpx.AsyncClient(**client_kwargs) as client:
            for attempt in range(attempts):
                try:
                    response = await client.post(url, json=payload, headers=headers)
                except httpx.TransportError as exc:
                    if attempt < attempts - 1:
                        delay = RETRY_DELAYS_SECONDS[attempt]
                        logger.warning(
                            "LLM transport error %s; retrying in %.1fs (attempt %d/%d)",
                            type(exc).__name__,
                            delay,
                            attempt + 1,
                            attempts,
                        )
                        await _sleep(delay)
                        continue
                    raise LLMRequestError(
                        f"LLM request failed after {attempts} attempts: {type(exc).__name__}",
                        attempts=attempts,
                    ) from exc

                if response.status_code in RETRYABLE_STATUS_CODES:
                    if attempt < attempts - 1:
                        delay = RETRY_DELAYS_SECONDS[attempt]
                        logger.warning(
                            "LLM returned HTTP %d; retrying in %.1fs (attempt %d/%d)",
                            response.status_code,
                            delay,
                            attempt + 1,
                            attempts,
                        )
                        await _sleep(delay)
                        continue
                    raise LLMRequestError(
                        f"LLM returned HTTP {response.status_code} after {attempts} attempts: "
                        f"{_body_snippet(response)}",
                        status_code=response.status_code,
                        attempts=attempts,
                    )

                if response.status_code >= 400:
                    raise LLMRequestError(
                        f"LLM returned HTTP {response.status_code}: {_body_snippet(response)}",
                        status_code=response.status_code,
                        attempts=attempt + 1,
                    )

                return self._parse_response(response, started)

        raise LLMRequestError("LLM request failed", attempts=attempts)

    def _parse_response(self, response: httpx.Response, started: float) -> LLMResponse:
        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise LLMResponseError("LLM returned a non-JSON body") from exc

        if not isinstance(data, dict):
            raise LLMResponseError("LLM response body is not a JSON object")

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMResponseError("LLM response has no choices")

        choice = choices[0]
        if not isinstance(choice, dict):
            raise LLMResponseError("LLM response choice is malformed")

        message = choice.get("message")
        if not isinstance(message, dict):
            raise LLMResponseError("LLM response has no message")

        content = message.get("content")
        if isinstance(content, list):
            content = "".join(
                part["text"]
                for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
        if not isinstance(content, str):
            raise LLMResponseError("LLM response content is not text")

        usage = data.get("usage")
        if not isinstance(usage, dict):
            usage = {}

        return LLMResponse(
            content=content,
            model=str(data.get("model") or self._settings.llm_model),
            prompt_tokens=_optional_int(usage.get("prompt_tokens")),
            completion_tokens=_optional_int(usage.get("completion_tokens")),
            finish_reason=(
                choice.get("finish_reason")
                if isinstance(choice.get("finish_reason"), str)
                else None
            ),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _body_snippet(response: httpx.Response, limit: int = 200) -> str:
    try:
        text = " ".join(response.text.split())
    except httpx.ResponseNotRead:
        return "<unreadable body>"
    if not text:
        return "<empty body>"
    return text[:limit] + ("..." if len(text) > limit else "")
