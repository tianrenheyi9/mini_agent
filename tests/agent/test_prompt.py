"""System Prompt 构建与纠错消息的单元测试。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from app.agent.prompt import (
    build_repair_message,
    build_system_prompt,
    render_todos,
)


@dataclass(frozen=True)
class ToolStub:
    """满足 ToolLike 协议的最小测试对象。"""

    name: str
    description: str
    parameters: dict[str, Any]


WEATHER = ToolStub(
    name="weather",
    description="查询指定城市的天气",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)
TODO = ToolStub(
    name="todo",
    description="管理当前会话的待办",
    parameters={
        "type": "object",
        "properties": {"action": {"type": "string"}},
        "required": ["action"],
    },
)

FIXED_NOW = datetime(
    2026,
    10,
    19,
    9,
    30,
    0,
    tzinfo=timezone(timedelta(hours=8)),
)


def test_prompt_contains_protocol_and_tools() -> None:
    """Prompt 必须包含输出协议与完整的工具目录。"""
    prompt = build_system_prompt([WEATHER, TODO], now=FIXED_NOW)

    assert "action" in prompt
    assert "final" in prompt
    assert "weather" in prompt
    assert "查询指定城市的天气" in prompt
    assert '"city"' in prompt
    assert "todo" in prompt


def test_prompt_contains_summary_todos_and_time() -> None:
    """摘要、待办与当前时间都要注入 Prompt。"""
    prompt = build_system_prompt(
        [WEATHER],
        summary="用户在北京，偏好简短回答。",
        todos=[
            {"id": "t_1", "content": "给妈妈买生日礼物", "done": False},
            {"id": "t_2", "content": "提交周报", "done": True},
        ],
        now=FIXED_NOW,
    )

    assert "用户在北京，偏好简短回答。" in prompt
    assert "- [ ] t_1: 给妈妈买生日礼物" in prompt
    assert "- [x] t_2: 提交周报" in prompt
    assert "2026-10-19 09:30:00 +0800" in prompt


def test_prompt_uses_placeholders_when_empty() -> None:
    """没有摘要和待办时给出明确的占位文本。"""
    prompt = build_system_prompt([], now=FIXED_NOW)

    assert "（暂无历史摘要）" in prompt
    assert "（暂无待办）" in prompt
    assert "（当前没有可用工具" in prompt


def test_render_todos_empty() -> None:
    """空待办列表返回占位文本。"""
    assert render_todos([]) == "（暂无待办）"


def test_repair_message_contains_error_and_raw_output() -> None:
    """纠错消息要告诉模型具体错在哪里，并给出正确格式。"""
    message = build_repair_message(
        '{"action": "weather", "final": "冲突"}',
        "action 和 final 不能同时出现",
    )

    assert "action 和 final 不能同时出现" in message
    assert '{"action": "weather", "final": "冲突"}' in message
    assert '"thought"' in message
    assert '"final"' in message


def test_repair_message_truncates_long_output() -> None:
    """原始输出过长时要截断，避免再次撑大上下文。"""
    message = build_repair_message("x" * 2000, "格式错误", max_raw_chars=20)

    assert "x" * 20 in message
    assert "x" * 21 not in message
    assert "格式错误" in message
