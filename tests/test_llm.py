"""
LLM 客户端单元测试。

所有测试都使用 httpx.MockTransport：
- 不联网、不需要 API Key；
- 可以精确模拟 200、401、503、超时等场景；
- 结果稳定，适合放进 CI。
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import httpx
import pytest

from app.agent import llm as llm_module
from app.agent.llm import (
    LLMClient,
    LLMConfigError,
    LLMRequestError,
    LLMResponseError,
)
from app.config import Settings, get_settings

BASE_URL = "https://example.test/v1"


def make_settings(**overrides: Any) -> Settings:
    """
    构造测试用的配置。

    Settings 是 frozen dataclass，不能直接改字段，
    所以用 dataclasses.replace 生成一个新对象，只覆盖需要变化的字段。
    """
    values: dict[str, Any] = {
        "llm_base_url": BASE_URL,
        "llm_api_key": "test-key",
        "llm_model": "test-model",
        "llm_proxy": "",
        "llm_timeout_seconds": 5.0,
    }
    values.update(overrides)
    return replace(get_settings(), **values)


def success_body(content: str = "hello") -> dict[str, Any]:
    """构造一个最小的 OpenAI 兼容成功响应。"""
    return {
        "model": "test-model",
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
    }


@pytest.fixture
def recorded_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """
    把重试时的等待替换成“只记录、不真的睡”。

    否则测试重试逻辑会真的等 0.5 + 1.5 秒，拖慢整个测试套件。
    """
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(llm_module, "_sleep", fake_sleep)
    return sleeps


async def test_success_request_and_response() -> None:
    """成功路径：请求格式正确，响应字段被完整解析。"""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        # MockTransport 会把请求交给我们，可以在这里检查 URL、请求头和请求体。
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=success_body("你好"))

    client = LLMClient(make_settings(), transport=httpx.MockTransport(handler))
    result = await client.chat([{"role": "user", "content": "hi"}])

    assert captured["url"] == f"{BASE_URL}/chat/completions"
    assert captured["auth"] == "Bearer test-key"
    assert captured["body"]["model"] == "test-model"
    assert captured["body"]["messages"] == [{"role": "user", "content": "hi"}]
    assert captured["body"]["temperature"] == 0.0
    assert result.content == "你好"
    assert result.model == "test-model"
    assert result.prompt_tokens == 11
    assert result.completion_tokens == 7
    assert result.finish_reason == "stop"
    assert result.latency_ms >= 0


async def test_retries_on_server_error_then_succeeds(
    recorded_sleeps: list[float],
) -> None:
    """临时 500 错误：等待 0.5 秒后重试，第二次成功。"""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(500, text="boom")
        return httpx.Response(200, json=success_body("ok"))

    client = LLMClient(make_settings(), transport=httpx.MockTransport(handler))
    result = await client.chat([{"role": "user", "content": "hi"}])

    assert calls == 2
    assert result.content == "ok"
    assert recorded_sleeps == [0.5]


async def test_retries_on_timeout_then_succeeds() -> None:
    """网络超时：属于可重试错误，第二次请求成功后正常返回。"""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            # 抛出 httpx 的超时异常，模拟网络层失败。
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(200, json=success_body("recovered"))

    client = LLMClient(make_settings(), transport=httpx.MockTransport(handler))
    result = await client.chat([{"role": "user", "content": "hi"}])

    assert calls == 2
    assert result.content == "recovered"


async def test_non_retryable_client_error_fails_immediately() -> None:
    """401 鉴权失败：重试没有意义，只请求一次就报错。"""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, text="invalid api key")

    client = LLMClient(make_settings(), transport=httpx.MockTransport(handler))

    with pytest.raises(LLMRequestError) as exc_info:
        await client.chat([{"role": "user", "content": "hi"}])

    assert calls == 1
    assert exc_info.value.status_code == 401
    assert exc_info.value.attempts == 1
    assert "invalid api key" in str(exc_info.value)


async def test_retries_exhausted_on_server_error(
    recorded_sleeps: list[float],
) -> None:
    """持续 503：总共尝试 3 次，等待序列是 0.5、1.5 秒，最后抛出异常。"""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, text="unavailable")

    client = LLMClient(make_settings(), transport=httpx.MockTransport(handler))

    with pytest.raises(LLMRequestError) as exc_info:
        await client.chat([{"role": "user", "content": "hi"}])

    assert calls == 3
    assert recorded_sleeps == [0.5, 1.5]
    assert exc_info.value.status_code == 503
    assert exc_info.value.attempts == 3


async def test_missing_config_raises_without_request() -> None:
    """没有 API Key 时直接报配置错误，不应该发起任何网络请求。"""
    client = LLMClient(make_settings(llm_api_key=""))

    with pytest.raises(LLMConfigError):
        await client.chat([{"role": "user", "content": "hi"}])


async def test_non_json_body_raises_response_error() -> None:
    """HTTP 200 但响应不是 JSON：说明服务端契约不符，抛出 ResponseError。"""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    client = LLMClient(make_settings(), transport=httpx.MockTransport(handler))

    with pytest.raises(LLMResponseError):
        await client.chat([{"role": "user", "content": "hi"}])


async def test_content_parts_are_joined() -> None:
    """有些兼容接口把 content 返回成分段数组，客户端应拼接成字符串。"""
    body = success_body()
    body["choices"][0]["message"]["content"] = [
        {"type": "text", "text": "he"},
        {"type": "text", "text": "llo"},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    client = LLMClient(make_settings(), transport=httpx.MockTransport(handler))
    result = await client.chat([{"role": "user", "content": "hi"}])

    assert result.content == "hello"


async def test_empty_choices_raises_response_error() -> None:
    """choices 为空说明响应结构不完整，应该明确报错而不是返回空字符串。"""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    client = LLMClient(make_settings(), transport=httpx.MockTransport(handler))

    with pytest.raises(LLMResponseError):
        await client.chat([{"role": "user", "content": "hi"}])
