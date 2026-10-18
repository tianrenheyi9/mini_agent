"""
OpenAI 兼容的 LLM 客户端。

为什么不用官方 SDK？
- 不同厂商的 SDK 接口不统一，切换成本高；
- 直接发 HTTP 请求更容易看清协议细节，也方便面试时讲解。

这个客户端负责：
1. 组装 `POST /chat/completions` 请求；
2. 设置超时与可选代理；
3. 遇到临时故障（网络错误、超时、5xx、429）时自动重试；
4. 把不同失败转换成明确的异常类型；
5. 从响应中提取文本、token 用量、结束原因和耗时。
"""

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

# 每次重试前等待的秒数。元组长度是 2，表示最多重试 2 次，总共尝试 3 次。
# 使用递增等待（退避）可以避免服务端刚出故障就被连续请求打满。
RETRY_DELAYS_SECONDS = (0.5, 1.5)

# 这些 HTTP 状态码通常表示“临时故障”，值得重试。
# 401/403 这类是配置或权限问题，重试没有意义。
RETRYABLE_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})


async def _sleep(seconds: float) -> None:
    """
    单独包一层 asyncio.sleep。

    这样测试时可以把它替换成“不真正等待”的假函数，
    从而快速验证重试次数，而不必让测试真的等 2 秒。
    """
    await asyncio.sleep(seconds)


class LLMError(Exception):
    """所有 LLM 相关异常的基类，方便调用方统一捕获。"""


class LLMConfigError(LLMError):
    """配置缺失或配置不完整，例如没有填 API Key。"""


class LLMRequestError(LLMError):
    """
    请求最终失败（可能已经重试过）。

    status_code: HTTP 状态码；网络错误时为 None。
    attempts:    一共尝试了几次，便于日志和排查。
    """

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
    """HTTP 请求成功，但响应内容不符合预期（例如不是 JSON、缺少 choices）。"""


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
    LLM 调用客户端。

    `transport` 参数是测试用的“入口”：
    - 生产环境传 None，客户端会正常走网络；
    - 测试时传入 httpx.MockTransport，就能在不联网的情况下模拟任意响应。
    这种写法叫“依赖注入”，可以让核心逻辑更容易测试。
    """

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

        # 第 2 步：组装请求地址、请求体和请求头。
        # rstrip("/") 是为了兼容用户配置成 "https://xxx/v1/" 的情况。
        url = f"{self._settings.llm_base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self._settings.llm_model,
            "messages": list(messages),
            "temperature": temperature,
            "stream": False,  # MVP 阶段使用非流式响应，逻辑更简单
        }
        headers = {
            "Authorization": f"Bearer {self._settings.llm_api_key}",
            "Content-Type": "application/json",
        }

        # 第 3 步：设置超时。总超时控制整个请求，连接超时单独限制在 10 秒内。
        timeout = httpx.Timeout(
            self._settings.llm_timeout_seconds,
            connect=min(10.0, self._settings.llm_timeout_seconds),
        )

        # 如果配置了代理（访问境外 API 时常用），就传给 httpx。
        client_kwargs: dict[str, Any] = {
            "timeout": timeout,
            "transport": self._transport,
        }
        if self._settings.llm_proxy:
            client_kwargs["proxy"] = self._settings.llm_proxy

        attempts = len(RETRY_DELAYS_SECONDS) + 1
        started = time.perf_counter()

        # 在同一个 AsyncClient 内完成所有重试，复用连接更高效。
        async with httpx.AsyncClient(**client_kwargs) as client:
            for attempt in range(attempts):
                # 第 4 步：发请求。
                # httpx.TransportError 是所有网络层异常的基类，
                # 包括连接失败、读写超时、连接被重置等。
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
                    # 最后一次也失败：转换成统一的业务异常，并保留异常链。
                    raise LLMRequestError(
                        f"LLM request failed after {attempts} attempts: {type(exc).__name__}",
                        attempts=attempts,
                    ) from exc

                # 第 5 步：服务端返回了响应，判断状态码。
                # 5xx / 429 等临时错误：还有重试次数就继续。
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

                # 4xx（如 401 鉴权失败、400 参数错误）重试也不会成功，立即失败。
                if response.status_code >= 400:
                    raise LLMRequestError(
                        f"LLM returned HTTP {response.status_code}: {_body_snippet(response)}",
                        status_code=response.status_code,
                        attempts=attempt + 1,
                    )

                # 第 6 步：2xx 成功响应，解析 JSON 并返回结构化结果。
                return self._parse_response(response, started)

        # 正常情况下上面的循环一定会 return 或 raise；
        # 这行只是为了让类型检查和阅读者明确知道“所有路径都有结果”。
        raise LLMRequestError("LLM request failed", attempts=attempts)

    def _parse_response(self, response: httpx.Response, started: float) -> LLMResponse:
        """把 OpenAI 格式的响应 JSON 转换成 LLMResponse。"""

        # 响应必须是合法 JSON。
        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise LLMResponseError("LLM returned a non-JSON body") from exc

        if not isinstance(data, dict):
            raise LLMResponseError("LLM response body is not a JSON object")

        # OpenAI 兼容格式的核心结构：
        # {"choices": [{"message": {"content": "..."}, "finish_reason": "stop"}], ...}
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

        # 大多数厂商返回字符串；少数多模态接口会返回分段数组。
        # 这里把文本段拼起来，保证上层永远拿到字符串。
        if isinstance(content, list):
            content = "".join(
                part["text"]
                for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
        if not isinstance(content, str):
            raise LLMResponseError("LLM response content is not text")

        # usage 和 finish_reason 不是所有厂商都返回，缺失时用 None。
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
    """只有值确实是整数时才返回，否则返回 None（避免把字符串当成 token 数）。"""
    return value if isinstance(value, int) else None


def _body_snippet(response: httpx.Response, limit: int = 200) -> str:
    """
    截取响应体的一小段放进错误信息。

    限制长度是为了避免日志被超长 HTML 或堆栈刷屏；
    同时也避免把完整响应（可能含敏感信息）写进日志。
    """
    try:
        text = " ".join(response.text.split())
    except httpx.ResponseNotRead:
        return "<unreadable body>"
    if not text:
        return "<empty body>"
    return text[:limit] + ("..." if len(text) > limit else "")
