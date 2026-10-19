"""
LLM 客户端单元测试。

这些测试不联网，也不需要 API Key：
- 用 `FakeOpenAI` 替代真实的 `AsyncOpenAI`；
- `FakeOpenAI` 的 `create()` 按脚本返回 `ChatCompletion` 或抛出 SDK 异常。

SDK 自身负责的超时与重试不在单测里重复验证；
这里重点验证“成功路径解析”和“异常到项目异常的映射”。
"""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
from openai import APIConnectionError, APIError, APIStatusError
from openai.types.chat import ChatCompletion

from app.agent import llm as llm_module
from app.agent.llm import (
    LLMClient,
    LLMConfigError,
    LLMRequestError,
    LLMResponseError,
)
from app.config import Settings, get_settings

BASE_URL = "https://example.test/v1"


class FakeCompletions:
    """模拟 SDK 中的 `client.chat.completions`。"""

    def __init__(self, outcomes: list[Any]) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        # 记录调用参数，测试可以断言 Agent 传了什么给模型。
        self.calls.append(kwargs)
        if not self._outcomes:
            raise AssertionError("FakeOpenAI has no scripted outcome left")

        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeChat:
    def __init__(self, outcomes: list[Any]) -> None:
        self.completions = FakeCompletions(outcomes)


class FakeOpenAI:
    """只实现 LLMClient 用到的那部分 SDK 接口。"""

    def __init__(self, outcomes: list[Any]) -> None:
        self.chat = FakeChat(outcomes)


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
        "llm_timeout_seconds": 5.0,
    }
    values.update(overrides)
    return replace(get_settings(), **values)


def success_body(content: str = "hello") -> dict[str, Any]:
    """构造一个最小的 OpenAI 兼容成功响应（字段满足 SDK 的校验要求）。"""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


def success_completion(content: str = "hello") -> ChatCompletion:
    """把字典转换成真实 SDK 类型，确保解析逻辑面对的对象和线上一致。"""
    return ChatCompletion.model_validate(success_body(content))


def status_error(status_code: int, message: str = "boom") -> APIStatusError:
    """构造一个带最小 response 的 SDK 状态码异常。"""
    response = SimpleNamespace(
        status_code=status_code,
        headers={},
        request=None,
        text=message,
        reason_phrase="",
    )
    return APIStatusError(message, response=response, body=None)


async def test_client_is_created_with_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """不注入 client 时，应按配置创建 AsyncOpenAI，并把超时和重试次数传进去。"""
    captured: dict[str, Any] = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            self.chat = FakeChat([success_completion("ok")])

    # 替换模块里的 AsyncOpenAI，避免真的创建网络客户端。
    monkeypatch.setattr(llm_module, "AsyncOpenAI", FakeAsyncOpenAI)

    result = await LLMClient(make_settings()).chat(
        [{"role": "user", "content": "hi"}]
    )

    assert captured["api_key"] == "test-key"
    assert captured["base_url"] == BASE_URL
    assert captured["timeout"] == 5.0
    assert captured["max_retries"] == 2
    assert result.content == "ok"


async def test_success_request_and_response() -> None:
    """成功路径：传给 SDK 的参数正确，响应字段被完整解析。"""
    fake = FakeOpenAI([success_completion("你好")])
    client = LLMClient(make_settings(), client=fake)

    result = await client.chat([{"role": "user", "content": "hi"}])

    call = fake.chat.completions.calls[0]
    assert call["model"] == "test-model"
    assert call["messages"] == [{"role": "user", "content": "hi"}]
    assert call["temperature"] == 0.0
    assert call["stream"] is False

    assert result.content == "你好"
    assert result.model == "test-model"
    assert result.prompt_tokens == 11
    assert result.completion_tokens == 7
    assert result.finish_reason == "stop"
    assert result.latency_ms >= 0


async def test_connection_error_is_mapped() -> None:
    """网络错误应转换成 LLMRequestError，并且没有 HTTP 状态码。"""
    fake = FakeOpenAI([APIConnectionError(request=None)])
    client = LLMClient(make_settings(), client=fake)

    with pytest.raises(LLMRequestError) as exc_info:
        await client.chat([{"role": "user", "content": "hi"}])

    assert exc_info.value.status_code is None


async def test_status_error_is_mapped() -> None:
    """服务端 401 应转换成带 status_code 的 LLMRequestError。"""
    fake = FakeOpenAI([status_error(401, "invalid api key")])
    client = LLMClient(make_settings(), client=fake)

    with pytest.raises(LLMRequestError) as exc_info:
        await client.chat([{"role": "user", "content": "hi"}])

    assert exc_info.value.status_code == 401
    assert "invalid api key" in str(exc_info.value)


async def test_generic_api_error_is_mapped() -> None:
    """其他 SDK 错误（如响应结构校验失败）应转换成 LLMResponseError。"""
    fake = FakeOpenAI([APIError("bad response", None, body=None)])
    client = LLMClient(make_settings(), client=fake)

    with pytest.raises(LLMResponseError):
        await client.chat([{"role": "user", "content": "hi"}])


async def test_missing_config_raises_without_request() -> None:
    """没有 API Key 时直接报配置错误，不应该调用 SDK。"""
    fake = FakeOpenAI([])
    client = LLMClient(make_settings(llm_api_key=""), client=fake)

    with pytest.raises(LLMConfigError):
        await client.chat([{"role": "user", "content": "hi"}])

    assert fake.chat.completions.calls == []


async def test_non_json_body_raises_response_error() -> None:
    """SDK 抛出的 JSON 解析错误应转换成 LLMResponseError。"""
    fake = FakeOpenAI([json.JSONDecodeError("Expecting value", "not json", 0)])
    client = LLMClient(make_settings(), client=fake)

    with pytest.raises(LLMResponseError):
        await client.chat([{"role": "user", "content": "hi"}])


async def test_none_content_raises_response_error() -> None:
    """模型没有返回文本时，应该明确报错而不是返回空字符串。"""
    completion = success_completion()
    completion.choices[0].message.content = None
    fake = FakeOpenAI([completion])
    client = LLMClient(make_settings(), client=fake)

    with pytest.raises(LLMResponseError):
        await client.chat([{"role": "user", "content": "hi"}])


async def test_empty_choices_raises_response_error() -> None:
    """choices 为空说明响应结构不完整，应该明确报错。"""
    body = success_body()
    body["choices"] = []
    completion = ChatCompletion.model_validate(body)
    fake = FakeOpenAI([completion])
    client = LLMClient(make_settings(), client=fake)

    with pytest.raises(LLMResponseError):
        await client.chat([{"role": "user", "content": "hi"}])


async def test_missing_usage_yields_none_tokens() -> None:
    """部分兼容服务不返回 usage，token 字段应降级为 None。"""
    body = success_body()
    body.pop("usage")
    completion = ChatCompletion.model_validate(body)
    fake = FakeOpenAI([completion])
    client = LLMClient(make_settings(), client=fake)

    result = await client.chat([{"role": "user", "content": "hi"}])

    assert result.prompt_tokens is None
    assert result.completion_tokens is None
