"""
LLM 输出协议解析器。

模型每轮应该只输出一个 JSON 对象，二选一：

1. 调用工具：
   {"thought": "...", "action": "weather", "input": {"city": "北京"}}

2. 最终回答：
   {"thought": "...", "final": "北京今天多云，18~26℃。"}

但真实模型经常不守规矩，例如：
- 把 JSON 包在 ```json 代码块里；
- 在 JSON 前后加一段解释；
- 输出多个对象，其中只有一个是真正的决策；
- 漏字段、字段类型不对。

所以解析器分两步：
1. 从文本中“找 JSON”（围栏优先，其次扫描平衡大括号）；
2. 校验 JSON 是否符合协议。
任何失败都返回 ParseError，而不是抛异常，让 Agent 循环可以决定是否纠错重试。
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, TypeAlias

# 匹配 ```json ... ``` 或 ``` ... ```，DOTALL 让 . 可以跨行匹配。
_FENCED_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

# 用来区分“json.loads 失败”和“真的解析出了 None”的哨兵值。
_INVALID_JSON = object()


@dataclass(frozen=True)
class ToolCall:
    """模型决定调用工具。"""

    thought: str
    action: str
    input: dict[str, Any]


@dataclass(frozen=True)
class FinalAnswer:
    """模型决定直接回复用户。"""

    thought: str
    final: str


@dataclass(frozen=True)
class ParseError:
    """模型输出无法解析成合法协议。"""

    reason: str


# 解析结果只可能是这三种之一，调用方用 isinstance 区分。
ParseResult: TypeAlias = ToolCall | FinalAnswer | ParseError


def parse(raw: str) -> ParseResult:
    """
    解析模型的一次原始输出。

    解析顺序：
    1. 先尝试所有 Markdown 围栏中的内容（通常模型把 JSON 放在这里）；
    2. 再扫描整段文本中的平衡大括号对象；
    3. 返回第一个“既合法 JSON、又符合协议”的对象。

    如果找到了合法 JSON 但协议不对（例如 action/final 同时存在），
    会记住第一条协议错误，在所有候选都尝试完后返回它，
    这样纠错提示能告诉模型具体错在哪里。
    """
    if not isinstance(raw, str) or not raw.strip():
        return ParseError("模型输出为空")

    text = raw.strip()
    first_protocol_error: ParseError | None = None

    for candidate in _iter_json_candidates(text):
        data = _load_json(candidate)
        if data is _INVALID_JSON:
            continue

        result = _validate_protocol(data)
        if isinstance(result, ParseError):
            # 记录错误但继续找，后面可能还有真正的决策对象。
            if first_protocol_error is None:
                first_protocol_error = result
            continue

        return result

    if first_protocol_error is not None:
        return first_protocol_error
    return ParseError("没有找到合法的 JSON 对象")


def _iter_json_candidates(text: str) -> Iterator[str]:
    """按“围栏优先、全文扫描兜底”的顺序产出可能的 JSON 片段。"""
    # 围栏里的内容优先级最高，避免被正文里举例的 JSON 干扰。
    for block in _FENCED_BLOCK_RE.findall(text):
        block = block.strip()
        if block:
            yield block

    # 整段文本本身也可能就是一个 JSON（包括数组或普通值）。
    # 这样像 [1, 2, 3] 这样的输入也能得到“必须是 JSON 对象”的明确错误。
    yield text

    # 没有围栏或围栏里不是目标对象时，扫描整段文本。
    yield from _balanced_objects(text)


def _balanced_objects(text: str) -> Iterator[str]:
    """
    扫描文本中所有“括号平衡”的 {...} 片段。

    这里不能简单用正则，因为 JSON 里可能嵌套对象，
    而且字符串内部也可能出现花括号和转义引号。
    """
    for start, char in enumerate(text):
        if char != "{":
            continue
        candidate = _read_balanced_object(text, start)
        if candidate is not None:
            yield candidate


def _read_balanced_object(text: str, start: int) -> str | None:
    """
    从 start 位置的 '{' 开始，找到与它匹配的 '}'。

    需要维护两个状态：
    - depth：当前嵌套层级；
    - in_string / escaped：是否在字符串里、是否处于转义状态。
    """
    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    # 没找到闭合括号，说明这段不是完整对象。
    return None


def _load_json(candidate: str) -> Any:
    """安全地解析 JSON；失败时返回哨兵值，不抛异常。"""
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return _INVALID_JSON


def _validate_protocol(data: Any) -> ParseResult:
    """校验 JSON 结构是否符合 Agent 协议。"""

    if not isinstance(data, dict):
        return ParseError("输出必须是 JSON 对象，不能是数组或普通值")

    has_action = "action" in data
    has_final = "final" in data

    if has_action and has_final:
        return ParseError("action 和 final 不能同时出现，必须二选一")
    if not has_action and not has_final:
        return ParseError("缺少 action 或 final，必须二选一")

    thought = data.get("thought", "")
    if thought is None:
        thought = ""
    if not isinstance(thought, str):
        return ParseError("thought 必须是字符串")

    if has_final:
        final = data.get("final")
        if not isinstance(final, str) or not final.strip():
            return ParseError("final 必须是非空字符串")
        return FinalAnswer(thought=thought.strip(), final=final.strip())

    action = data.get("action")
    if not isinstance(action, str) or not action.strip():
        return ParseError("action 必须是非空字符串")

    tool_input = data.get("input", {})
    if tool_input is None:
        tool_input = {}
    if not isinstance(tool_input, dict):
        return ParseError("input 必须是 JSON 对象")

    return ToolCall(
        thought=thought.strip(),
        action=action.strip(),
        input=tool_input,
    )
