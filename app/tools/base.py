"""
工具基础层。

这里定义所有工具共用的“合同”：

- ToolSpec：一个工具的完整描述（名称、用途、参数 Schema、执行函数）；
- ToolContext：工具执行时由 Runtime 注入的上下文（会话 id、待办列表、当前时间）；
- ToolResult：统一的执行结果，成功和失败都用同一种结构；
- validate_parameters：一个精简版 JSON Schema 校验器。

为什么参数要由 Runtime 校验，而不是让模型自己保证？
- 模型可能漏字段、传错类型、编造参数；
- 在真正执行工具前拦截错误，可以把清晰的错误信息回填给模型，让它自我纠正。
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# 工具执行函数统一是异步的，参数是“已校验的输入 + 运行上下文”。
ToolHandler = Callable[[dict[str, Any], "ToolContext"], Awaitable[Any]]


@dataclass(frozen=True)
class ToolSpec:
    """一个工具的静态描述，会被注册到 ToolRegistry。"""

    name: str  # 唯一名称，模型在 action 字段里使用它
    description: str  # 告诉模型“什么时候该用这个工具”
    parameters: dict[str, Any]  # JSON Schema，描述 input 的结构
    handler: ToolHandler  # 真正执行工具的函数


@dataclass
class ToolContext:
    """
    工具执行时由 Runtime 注入的上下文。

    模型只能看到并填写 input，看不到 session_id，
    这样可以避免模型编造或操作其他会话的数据。

    todos 直接指向 Session 中的待办列表：
    工具修改它，就等于修改了会话状态；持久化由 SessionManager 统一负责。
    """

    session_id: str
    todos: list[dict[str, Any]] = field(default_factory=list)
    now: datetime | None = None  # 测试时可传入固定时间，保证结果可断言

    @property
    def current_time(self) -> datetime:
        """当前时间；未注入时使用带本地时区的当前时间。"""
        return self.now or datetime.now().astimezone()


@dataclass(frozen=True)
class ToolResult:
    """工具执行的统一结果，无论成功还是失败都返回它。"""

    tool: str
    ok: bool
    data: Any = None
    error: str | None = None
    duration_ms: int = 0


class ToolExecutionError(Exception):
    """
    工具自身检测到的“业务错误”。

    例如：待办 id 不存在、表达式非法、查询为空。
    这类错误是预期内的，Registry 会把 message 原样回填给模型，
    不打印堆栈，也不当成程序崩溃。
    """


# ---------------------------------------------------------------------------
# 精简版 JSON Schema 校验
# ---------------------------------------------------------------------------
# 只支持工具参数真正用得到的子集：
# type / required / properties / items / enum / minimum / maximum。
# 不追求完整实现 JSON Schema 规范，避免过度设计。

_TYPE_CHECKERS: dict[str, Callable[[Any], bool]] = {
    "object": lambda value: isinstance(value, dict),
    "array": lambda value: isinstance(value, list),
    "string": lambda value: isinstance(value, str),
    # 注意：Python 中 bool 是 int 的子类，必须显式排除。
    "integer": lambda value: isinstance(value, int)
    and not isinstance(value, bool),
    "number": lambda value: isinstance(value, (int, float))
    and not isinstance(value, bool),
    "boolean": lambda value: isinstance(value, bool),
    "null": lambda value: value is None,
}


def validate_parameters(
    schema: Mapping[str, Any],
    value: Any,
    path: str = "input",
) -> list[str]:
    """
    按 Schema 校验参数，返回所有错误信息。

    返回列表而不是抛异常，是为了一次性告诉模型所有问题，
    避免它改一个字段、又被下一个错误拦住。
    """
    errors: list[str] = []
    expected_type = schema.get("type")

    # 1. 类型检查。
    if expected_type is not None:
        if not isinstance(expected_type, str) or expected_type not in _TYPE_CHECKERS:
            return [f"{path}: Schema type 配置无效：{expected_type!r}"]
        if not _TYPE_CHECKERS[expected_type](value):
            return [
                f"{path}: 期望 {expected_type}，实际是 {_describe_type(value)}"
            ]

    # 2. 枚举检查。
    if "enum" in schema:
        allowed = schema["enum"]
        if isinstance(allowed, list) and value not in allowed:
            options = "、".join(repr(item) for item in allowed)
            errors.append(f"{path}: 必须是以下值之一：{options}")

    # 3. 数值范围检查。
    if expected_type in {"integer", "number"}:
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, (int, float)) and value < minimum:
            errors.append(f"{path}: 不能小于 {minimum}")
        if isinstance(maximum, (int, float)) and value > maximum:
            errors.append(f"{path}: 不能大于 {maximum}")

    # 4. 对象：检查必填字段和子字段。
    if expected_type == "object":
        required = schema.get("required", [])
        if isinstance(required, list):
            for key in required:
                if key not in value:
                    errors.append(f"{path}.{key}: 必填字段缺失")

        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            for key, child_schema in properties.items():
                if key in value and isinstance(child_schema, Mapping):
                    errors.extend(
                        validate_parameters(
                            child_schema,
                            value[key],
                            f"{path}.{key}",
                        )
                    )

    # 5. 数组：逐个元素校验。
    if expected_type == "array":
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                errors.extend(
                    validate_parameters(item_schema, item, f"{path}[{index}]")
                )

    return errors


def _describe_type(value: Any) -> str:
    """把 Python 类型转换成更容易读的中文说明。"""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def normalize_whitespace(text: str) -> str:
    """把连续空白压成一个空格，便于错误信息和日志展示。"""
    return " ".join(text.split())


def split_search_terms(query: str) -> list[str]:
    """
    把查询拆成简单词项。

    mock 搜索不需要真正的分词器：
    按空白和中英文标点切分，足够支持演示与测试。
    """
    return [
        term
        for term in re.split(r"[\s,，。？?！!、;；:：()（）]+", query.strip().lower())
        if term
    ]
