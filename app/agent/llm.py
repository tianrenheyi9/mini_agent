"""
基于 OpenAI 官方 SDK 的 LLM 客户端。

我们使用 `AsyncOpenAI` 负责：
- 组装并发送 `POST /chat/completions` 请求；
- 把响应解析成带类型的 Python 对象。

客户端自己负责：
- 配置检查与超时/代理；
- 显式重试（0.5s、1.5s）；
- 把 SDK 的异常转换成项目统一的异常类型；
- 提取文本、token 用量、结束原因和耗时。

为什么 `max_retries=0`？
- 官方 SDK 自带重试，但它的等待时间带随机抖动，不方便在测试里精确断言；
- 关闭 SDK 重试、保留项目自己的重试循环后，重试策略清晰可控，
  也便于后续把每次重试记录进 Trace。
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import httpx
from openai import APIConnectionError, APIError, APIStatusError, AsyncOpenAI
from openai.types.chat import ChatCompletion

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
    import asyncio

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

        # 第 2 步：准备底层 HTTP 客户端。
        # openai SDK 默认自己创建 httpx 客户端；这里显式传入，
        # 是为了支持测试注入 MockTransport 以及可选的代理配置。
        timeout = httpx.Timeout(
            self._settings.llm_timeout_seconds,
            connect=min(10.0, self._settings.llm_timeout_seconds),
        )
        http_client_kwargs: dict[str, Any] = {
            "timeout": timeout,
            "transport": self._transport,
        }
        if self._settings.llm_proxy:
            http_client_kwargs["proxy"] = self._settings.llm_proxy

        attempts = len(RETRY_DELAYS_SECONDS) + 1
        started = time.perf_counter()

        async with httpx.AsyncClient(**http_client_kwargs) as http_client:
            # base_url 指向任意 OpenAI 兼容服务；SDK 会自动拼出 /chat/completions。
            client = AsyncOpenAI(
                api_key=self._settings.llm_api_key,
                base_url=self._settings.llm_base_url,
                max_retries=0,
                http_client=http_client,
            )

            for attempt in range(attempts):
                # 第 3 步：调用 SDK。SDK 内部会用 httpx 发请求并做响应校验。
                try:
                    completion = await client.chat.completions.create(
                        model=self._settings.llm_model,
                        messages=list(messages),
                        temperature=temperature,
                        stream=False,
                    )
                except APIConnectionError as exc:
                    # 网络层错误，包括连接失败和超时（APITimeoutError 是它的子类）。
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
                except APIStatusError as exc:
                    # 服务端返回了非 2xx 状态码。
                    status = exc.status_code
                    if status in RETRYABLE_STATUS_CODES and attempt < attempts - 1:
                        delay = RETRY_DELAYS_SECONDS[attempt]
                        logger.warning(
                            "LLM returned HTTP %d; retrying in %.1fs (attempt %d/%d)",
                            status,
                            delay,
                            attempt + 1,
                            attempts,
                        )
                        await _sleep(delay)
                        continue
                    raise LLMRequestError(
                        f"LLM returned HTTP {status}: {_status_error_message(exc)}",
                        status_code=status,
                        attempts=(
                            attempts if status in RETRYABLE_STATUS_CODES else attempt + 1
                        ),
                    ) from exc
                except APIError as exc:
                    # 其余 SDK 错误，典型场景是响应结构校验失败（例如缺少 choices）。
                    raise LLMResponseError(
                        f"LLM response error: {type(exc).__name__}"
                    ) from exc
                except json.JSONDecodeError as exc:
                    # SDK 在 200 响应的 body 不是合法 JSON 时会直接抛出底层解析错误。
                    raise LLMResponseError("LLM returned a non-JSON body") from exc

                # 第 4 步：请求成功，把 SDK 对象转换成项目自己的 LLMResponse。
                return self._parse_completion(completion, started)

        # 正常情况下上面的循环一定会 return 或 raise；
        # 这行只是为了让类型检查和阅读者明确知道“所有路径都有结果”。
        raise LLMRequestError("LLM request failed", attempts=attempts)

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
