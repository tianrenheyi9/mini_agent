"""
基于 OpenAI 官方 SDK 的 LLM 客户端。

这个客户端只做四件事：
1. 检查配置是否完整；
2. 调用 `AsyncOpenAI.chat.completions.create()`；
3. 把 SDK 的异常转换成项目统一的异常类型；
4. 把 SDK 的响应转换成项目自己的 `LLMResponse`。

超时和重试都交给官方 SDK：
- `timeout` 控制单次请求超时；
- `max_retries=2` 让 SDK 自动重试连接失败、超时、408/409/429/5xx。
这样就不需要自己维护 httpx 客户端和重试循环。
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from openai import APIConnectionError, APIError, APIStatusError, AsyncOpenAI
from openai.types.chat import ChatCompletion

from app.config import Settings

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """所有 LLM 相关异常的基类，方便调用方统一捕获。"""


class LLMConfigError(LLMError):
    """配置缺失或配置不完整，例如没有填 API Key。"""


class LLMRequestError(LLMError):
    """
    请求最终失败。

    `status_code` 是服务端返回的 HTTP 状态码；网络错误时为 None。
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMResponseError(LLMError):
    """请求成功，但响应内容不符合预期（例如不是 JSON、缺少 choices）。"""


@dataclass(frozen=True)
class LLMResponse:
    """
    一次成功调用的结构化结果。

    不直接返回裸字符串，是因为 Agent 后续还需要记录 token 用量、
    结束原因和耗时，这些信息都放在这里。
    """

    content: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    finish_reason: str | None
    latency_ms: int


class LLMClient:
    """
    LLM 调用客户端，对外只暴露一个 `chat()` 方法。

    `client` 参数用于测试注入：
    - 生产环境不传，客户端会创建真实的 `AsyncOpenAI`；
    - 测试时传入一个假的 SDK 客户端，就能不联网地模拟成功和异常。
    """

    def __init__(
        self,
        settings: Settings,
        *,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self._settings = settings
        self._client = client

    def _create_client(self) -> AsyncOpenAI:
        """用当前配置创建真实的 SDK 客户端。"""
        return AsyncOpenAI(
            api_key=self._settings.llm_api_key,
            base_url=self._settings.llm_base_url,
            timeout=self._settings.llm_timeout_seconds,
            # 让 SDK 自动重试临时性错误，最多重试 2 次。
            max_retries=2,
        )

    async def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """
        发送一次对话请求并返回解析后的结果。

        `messages` 是 OpenAI 格式的消息列表，例如：
            [{"role": "system", "content": "..."},
             {"role": "user", "content": "..."}]
        """

        # 第 1 步：检查必填配置。缺少配置时直接报错，不发无意义的网络请求。
        if not self._settings.llm_configured:
            raise LLMConfigError(
                "LLM is not configured: set LLM_BASE_URL, LLM_API_KEY and LLM_MODEL"
            )

        client = self._client or self._create_client()
        started = time.perf_counter()

        # 第 2 步：调用 OpenAI SDK。
        # base_url 指向任意兼容服务，SDK 会自动拼出 /chat/completions。
        try:
            completion = await client.chat.completions.create(
                model=self._settings.llm_model,
                messages=list(messages),
                temperature=temperature,
                stream=False,
            )
        except APIConnectionError as exc:
            # 网络层错误，包括连接失败和超时（APITimeoutError 是它的子类）。
            logger.warning("LLM connection error: %s", type(exc).__name__)
            raise LLMRequestError(
                f"LLM request failed: {type(exc).__name__}"
            ) from exc
        except APIStatusError as exc:
            # 服务端返回了非 2xx 状态码。
            logger.warning("LLM returned HTTP %d", exc.status_code)
            raise LLMRequestError(
                f"LLM returned HTTP {exc.status_code}: {_status_error_message(exc)}",
                status_code=exc.status_code,
            ) from exc
        except APIError as exc:
            # 其余 SDK 错误，典型场景是响应结构校验失败。
            logger.warning("LLM response error: %s", type(exc).__name__)
            raise LLMResponseError(
                f"LLM response error: {type(exc).__name__}"
            ) from exc
        except json.JSONDecodeError as exc:
            # SDK 在 200 响应的 body 不是合法 JSON 时会直接抛出底层解析错误。
            logger.warning("LLM returned a non-JSON body")
            raise LLMResponseError("LLM returned a non-JSON body") from exc

        # 第 3 步：请求成功，把 SDK 对象转换成项目自己的 LLMResponse。
        return self._parse_completion(completion, started)

    def _parse_completion(
        self,
        completion: ChatCompletion,
        started: float,
    ) -> LLMResponse:
        """把 SDK 的 ChatCompletion 对象转换成 LLMResponse。"""

        # SDK 在响应没有声明 application/json 时，可能把原始文本直接返回。
        # 这里做一次类型检查，保证上层拿到的永远是结构化结果或明确异常。
        if not isinstance(completion, ChatCompletion):
            raise LLMResponseError("LLM response is not a chat completion")

        if not completion.choices:
            raise LLMResponseError("LLM response has no choices")

        choice = completion.choices[0]
        content = choice.message.content

        # SDK 的类型声明是 str | None；None 通常表示模型只返回了工具调用或空响应。
        if content is None:
            raise LLMResponseError("LLM response content is empty")
        if not isinstance(content, str):
            raise LLMResponseError("LLM response content is not text")

        usage = completion.usage
        return LLMResponse(
            content=content,
            model=completion.model or self._settings.llm_model,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            finish_reason=choice.finish_reason,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )


def _status_error_message(exc: APIStatusError, limit: int = 200) -> str:
    """
    截取 SDK 异常信息的一小段放进错误消息。

    限制长度是为了避免日志被超长 HTML 或堆栈刷屏；
    同时也避免把完整响应（可能含敏感信息）写进日志。
    """
    text = " ".join(str(exc).split())
    if not text:
        return "<empty error>"
    return text[:limit] + ("..." if len(text) > limit else "")
