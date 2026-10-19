"""
todo 工具：管理当前会话的待办事项。

关键设计：
- 待办状态保存在 Session 中，而不是让模型“记住”；
- Runtime 每轮会把待办快照写进 System Prompt；
- 工具只修改 `context.todos`，持久化交给 SessionManager。

支持的动作：
- add：添加待办（需要 content）
- list：查看全部待办
- complete：标记完成（需要 id）
- delete：删除（需要 id）
- clear：清空全部
"""

from __future__ import annotations

from typing import Any

from app.tools.base import ToolContext, ToolExecutionError, ToolSpec


def build_todo_tool() -> ToolSpec:
    """创建 todo 工具。"""
    return ToolSpec(
        name="todo",
        description=(
            "管理当前会话的待办事项。"
            "支持 add（添加）、list（查看）、complete（完成）、"
            "delete（删除）、clear（清空）。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "list", "complete", "delete", "clear"],
                    "description": "要执行的操作",
                },
                "content": {
                    "type": "string",
                    "description": "待办内容；action=add 时必填",
                },
                "id": {
                    "type": "string",
                    "description": "待办 id，例如 t_1；action=complete/delete 时必填",
                },
            },
            "required": ["action"],
        },
        handler=_todo_handler,
    )


async def _todo_handler(
    input_data: dict[str, Any],
    context: ToolContext,
) -> dict[str, Any]:
    action = input_data["action"]
    todos = context.todos

    if action == "add":
        content = str(input_data.get("content", "")).strip()
        if not content:
            raise ToolExecutionError("action=add 时 content 不能为空")

        todo = {
            "id": _next_todo_id(todos),
            "content": content,
            "done": False,
            "created_at": context.current_time.isoformat(timespec="seconds"),
        }
        todos.append(todo)
        return {"todo": todo, "count": len(todos)}

    if action == "list":
        return {"todos": list(todos), "count": len(todos)}

    if action == "complete":
        todo = _require_todo(todos, input_data.get("id"))
        todo["done"] = True
        return {"todo": todo}

    if action == "delete":
        index = _find_todo_index(todos, input_data.get("id"))
        if index is None:
            raise ToolExecutionError(f"待办不存在：{input_data.get('id')!r}")
        deleted = todos.pop(index)
        return {"deleted": deleted, "count": len(todos)}

    if action == "clear":
        count = len(todos)
        todos.clear()
        return {"cleared": count}

    # 正常情况下 Schema 的 enum 已经拦截了未知 action，
    # 这里再兜底一次，避免绕过 Registry 直接调用 handler。
    raise ToolExecutionError(f"未知的 todo action：{action!r}")


def _next_todo_id(todos: list[dict[str, Any]]) -> str:
    """
    生成 t_1、t_2 这样的递增 id。

    取现有最大编号 + 1，而不是使用 uuid：
    测试更稳定，Prompt 里的 id 也更短、更容易被模型准确引用。
    """
    used_numbers: list[int] = []
    for todo in todos:
        todo_id = todo.get("id")
        if (
            isinstance(todo_id, str)
            and todo_id.startswith("t_")
            and todo_id[2:].isdigit()
        ):
            used_numbers.append(int(todo_id[2:]))
    return f"t_{max(used_numbers, default=0) + 1}"


def _find_todo_index(
    todos: list[dict[str, Any]],
    todo_id: Any,
) -> int | None:
    """按 id 查找待办下标；找不到返回 None。"""
    if not isinstance(todo_id, str) or not todo_id:
        return None
    for index, todo in enumerate(todos):
        if todo.get("id") == todo_id:
            return index
    return None


def _require_todo(
    todos: list[dict[str, Any]],
    todo_id: Any,
) -> dict[str, Any]:
    """按 id 查找待办；找不到时抛出业务错误。"""
    index = _find_todo_index(todos, todo_id)
    if index is None:
        raise ToolExecutionError(f"待办不存在：{todo_id!r}")
    return todos[index]
