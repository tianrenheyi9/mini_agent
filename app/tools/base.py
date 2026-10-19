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

# 工具执行函数的类型别名。
# 参数 1：已经通过 Schema 校验的 input；
# 参数 2：Runtime 注入的 ToolContext；
# 返回值 Awaitable[Any] 表示它必须是一个 async 函数（或返回可 await 对象）。
ToolHandler = Callable[[dict[str, Any], "ToolContext"], Awaitable[Any]]


@dataclass(frozen=True)
class ToolSpec:
    """
    一个工具的静态描述，会被注册到 ToolRegistry。

    frozen=True 表示注册后不可修改，避免运行途中工具定义被意外改掉。
    """

    name: str  # 唯一名称，模型在输出协议的 action 字段里使用它
    description: str  # 告诉模型“这个工具能做什么、什么时候该用”
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

    # field(default_factory=list) 会为每个 ToolContext 创建独立的空列表。
    # 如果直接写 todos: list = []，所有实例会共享同一个列表，这是 Python 常见坑。
    todos: list[dict[str, Any]] = field(default_factory=list)

    now: datetime | None = None  # 测试时可传入固定时间，保证结果可断言

    @property
    def current_time(self) -> datetime:
        """当前时间；未注入时使用带本地时区的当前时间。"""
        return self.now or datetime.now().astimezone()


@dataclass(frozen=True)
class ToolResult:
    """
    工具执行的统一结果，无论成功还是失败都返回它。

    这样 Agent 循环不需要为每个工具写 try/except：
    只要检查 ok 字段即可。
    """

    tool: str  # 工具名，便于 Trace 记录
    ok: bool  # True 表示成功，False 表示失败
    data: Any = None  # 成功时的结构化返回值
    error: str | None = None  # 失败时给模型看的错误信息
    duration_ms: int = 0  # 执行耗时，便于性能排查


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
#
# 字典的 key 是 Schema 里的 type，value 是一个判断函数。
# 这样校验时只要 _TYPE_CHECKERS[expected_type](value) 即可。
_TYPE_CHECKERS: dict[str, Callable[[Any], bool]] = {
    "object": lambda value: isinstance(value, dict),
    "array": lambda value: isinstance(value, list),
    "string": lambda value: isinstance(value, str),
    # 注意：Python 中 bool 是 int 的子类，
    # 所以校验 integer 时必须显式排除 True/False。
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
            # Schema 本身写错了，属于开发期问题，直接返回。
            return [f"{path}: Schema type 配置无效：{expected_type!r}"]
        if not _TYPE_CHECKERS[expected_type](value):
            # 类型都不对，后面的 required/properties 检查没有意义，直接返回。
            return [
                f"{path}: 期望 {expected_type}，实际是 {_describe_type(value)}"
            ]

    # 2. 枚举检查。
    # 例如 todo 的 action 只能是 add/list/complete/delete/clear。
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
            # required 只检查“键是否存在”，不检查值是否为空。
            # 值是否为空由具体工具根据业务规则判断。
            for key in required:
                if key not in value:
                    errors.append(f"{path}.{key}: 必填字段缺失")

        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            for key, child_schema in properties.items():
                # 只校验实际传了的字段；没传且非必填的字段直接跳过。
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
    """
    把 Python 类型转换成更容易读的中文说明。

    直接打印 type(value) 会得到 <class 'int'> 这种不友好的文本；
    这里统一成 string/integer/array 等 Schema 术语。
    """
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
