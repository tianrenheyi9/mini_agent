"""ToolRegistry 的单元测试。"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.agent.prompt import render_tool_catalog
from app.tools import build_default_registry
from app.tools.base import (
    ToolContext,
    ToolExecutionError,
    ToolSpec,
)
from app.tools.registry import ToolRegistry


async def echo_handler(
    input_data: dict[str, Any],
    context: ToolContext,
) -> dict[str, Any]:
    return {"value": input_data["value"], "session_id": context.session_id}


ECHO = ToolSpec(
    name="echo",
    description="原样返回 value",
    parameters={
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    },
    handler=echo_handler,
)


def context() -> ToolContext:
    return ToolContext(session_id="s_test")


def test_register_and_list_tools() -> None:
    """注册后可以按名称查找，并保持注册顺序。"""
    registry = ToolRegistry()
    registry.register(ECHO)

    assert registry.names() == ["echo"]
    assert registry.get("echo") is ECHO
    assert registry.get("missing") is None


def test_duplicate_registration_is_rejected() -> None:
    """重复注册同名工具应该在启动阶段直接报错。"""
    registry = ToolRegistry()
    registry.register(ECHO)

    with pytest.raises(ValueError, match="已经注册过"):
        registry.register(ECHO)


def test_schema_must_be_object() -> None:
    """工具参数 Schema 顶层必须是 object。"""
    bad = ToolSpec(
        name="bad",
        description="bad",
        parameters={"type": "string"},
        handler=echo_handler,
    )
    registry = ToolRegistry()

    with pytest.raises(ValueError, match="type=object"):
        registry.register(bad)


async def test_successful_execution() -> None:
    """成功执行时返回 ok=True、data 和耗时。"""
    registry = ToolRegistry()
    registry.register(ECHO)

    result = await registry.execute("echo", {"value": "hello"}, context())

    assert result.ok is True
    assert result.data == {"value": "hello", "session_id": "s_test"}
    assert result.error is None
    assert result.duration_ms >= 0


async def test_unknown_tool_returns_error() -> None:
    """未知工具不抛异常，而是返回带可用列表的错误。"""
    registry = ToolRegistry()
    registry.register(ECHO)

    result = await registry.execute("weather", {}, context())

    assert result.ok is False
    assert "未知工具" in result.error
    assert "echo" in result.error


async def test_schema_error_prevents_handler_call() -> None:
    """参数不合法时 handler 不应被执行。"""
    registry = ToolRegistry()
    registry.register(ECHO)

    result = await registry.execute("echo", {"value": 123}, context())

    assert result.ok is False
    assert "参数校验失败" in result.error
    assert "input.value" in result.error


async def test_non_object_input_is_rejected() -> None:
    """input 必须是 JSON 对象。"""
    registry = ToolRegistry()
    registry.register(ECHO)

    result = await registry.execute("echo", "hello", context())  # type: ignore[arg-type]

    assert result.ok is False
    assert "必须是 JSON 对象" in result.error


async def test_business_error_is_returned_as_message() -> None:
    """ToolExecutionError 的信息会原样回填给模型。"""

    async def failing_handler(
        input_data: dict[str, Any],
        context: ToolContext,
    ) -> None:
        raise ToolExecutionError("待办不存在")

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="fail",
            description="总是失败",
            parameters={"type": "object", "properties": {}},
            handler=failing_handler,
        )
    )

    result = await registry.execute("fail", {}, context())

    assert result.ok is False
    assert result.error == "待办不存在"


async def test_unexpected_exception_is_wrapped() -> None:
    """未预期异常不向调用方抛出，只返回简短错误类型。"""

    async def crashing_handler(
        input_data: dict[str, Any],
        context: ToolContext,
    ) -> None:
        raise RuntimeError("boom")

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="crash",
            description="会崩溃",
            parameters={"type": "object", "properties": {}},
            handler=crashing_handler,
        )
    )

    result = await registry.execute("crash", {}, context())

    assert result.ok is False
    assert result.error == "工具执行失败：RuntimeError"


async def test_timeout_is_reported() -> None:
    """工具执行超过 timeout 时返回超时错误。"""

    async def slow_handler(
        input_data: dict[str, Any],
        context: ToolContext,
    ) -> None:
        await asyncio.sleep(0.05)

    registry = ToolRegistry(timeout_seconds=0.01)
    registry.register(
        ToolSpec(
            name="slow",
            description="很慢",
            parameters={"type": "object", "properties": {}},
            handler=slow_handler,
        )
    )

    result = await registry.execute("slow", {}, context())

    assert result.ok is False
    assert "超时" in result.error


def test_default_registry_exposes_all_tools() -> None:
    """默认注册表包含题目要求的四个工具，并能渲染成 Prompt 目录。"""
    registry = build_default_registry()

    assert registry.names() == ["todo", "calculator", "search", "weather"]

    catalog = render_tool_catalog(registry.tools())
    for name in registry.names():
        assert f"### {name}" in catalog

