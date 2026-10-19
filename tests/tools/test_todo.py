"""todo 工具的单元测试。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.tools.base import ToolContext
from app.tools.registry import ToolRegistry
from app.tools.todo import build_todo_tool

FIXED_NOW = datetime(
    2026,
    10,
    19,
    9,
    0,
    0,
    tzinfo=timezone(timedelta(hours=8)),
)


def make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(build_todo_tool())
    return registry


def make_context() -> ToolContext:
    return ToolContext(session_id="s_test", todos=[], now=FIXED_NOW)


async def test_add_and_list_todos() -> None:
    """添加两条待办后，list 能看到它们，id 递增。"""
    registry = make_registry()
    context = make_context()

    first = await registry.execute(
        "todo", {"action": "add", "content": "给妈妈买生日礼物"}, context
    )
    second = await registry.execute(
        "todo", {"action": "add", "content": "提交周报"}, context
    )

    assert first.ok is True
    assert first.data["todo"]["id"] == "t_1"
    assert first.data["todo"]["done"] is False
    assert first.data["todo"]["created_at"] == "2026-10-19T09:00:00+08:00"
    assert second.data["todo"]["id"] == "t_2"

    listed = await registry.execute("todo", {"action": "list"}, context)
    assert listed.ok is True
    assert listed.data["count"] == 2
    assert [item["content"] for item in listed.data["todos"]] == [
        "给妈妈买生日礼物",
        "提交周报",
    ]


async def test_complete_todo() -> None:
    """complete 按 id 把待办标记为完成。"""
    registry = make_registry()
    context = make_context()
    await registry.execute(
        "todo", {"action": "add", "content": "写周报"}, context
    )

    result = await registry.execute(
        "todo", {"action": "complete", "id": "t_1"}, context
    )

    assert result.ok is True
    assert result.data["todo"]["done"] is True
    assert context.todos[0]["done"] is True


async def test_delete_todo() -> None:
    """delete 按 id 删除待办。"""
    registry = make_registry()
    context = make_context()
    await registry.execute(
        "todo", {"action": "add", "content": "写周报"}, context
    )

    result = await registry.execute(
        "todo", {"action": "delete", "id": "t_1"}, context
    )

    assert result.ok is True
    assert result.data["deleted"]["content"] == "写周报"
    assert context.todos == []


async def test_clear_todos() -> None:
    """clear 清空全部待办并返回数量。"""
    registry = make_registry()
    context = make_context()
    await registry.execute(
        "todo", {"action": "add", "content": "A"}, context
    )
    await registry.execute(
        "todo", {"action": "add", "content": "B"}, context
    )

    result = await registry.execute("todo", {"action": "clear"}, context)

    assert result.ok is True
    assert result.data["cleared"] == 2
    assert context.todos == []


async def test_add_requires_content() -> None:
    """add 缺少内容时返回业务错误。"""
    registry = make_registry()

    result = await registry.execute(
        "todo", {"action": "add", "content": "   "}, make_context()
    )

    assert result.ok is False
    assert "content 不能为空" in result.error


async def test_complete_requires_existing_id() -> None:
    """complete 不存在的 id 时返回业务错误。"""
    registry = make_registry()

    result = await registry.execute(
        "todo", {"action": "complete", "id": "t_99"}, make_context()
    )

    assert result.ok is False
    assert "待办不存在" in result.error


async def test_unknown_action_is_rejected_by_schema() -> None:
    """enum 校验会在 handler 执行前拦截未知 action。"""
    registry = make_registry()

    result = await registry.execute(
        "todo", {"action": "rename"}, make_context()
    )

    assert result.ok is False
    assert "参数校验失败" in result.error


async def test_context_list_is_mutated_in_place() -> None:
    """工具直接修改上下文中的待办列表，便于 Session 统一持久化。"""
    registry = make_registry()
    context = make_context()

    await registry.execute(
        "todo", {"action": "add", "content": "买牛奶"}, context
    )

    assert len(context.todos) == 1
    assert context.todos[0]["content"] == "买牛奶"
