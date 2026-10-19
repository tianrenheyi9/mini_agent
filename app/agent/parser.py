"""
LLM 输出协议解析器。

模型每轮应该只输出一个 JSON 对象，二选一：

1. 调用工具：
   {"thought": "...", "action": "weather", "input": {"city": "北京"}}

2. 最终回答：
   {"thought": "...", "final": "北京今天多云，18~26℃。"}

但真实模型经常不守规矩，例如：
- 把 JSON 包在 ```json 代码块里（围栏）；
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

# 匹配 ```json ... ``` 或 ``` ... ```。
# - (?:json)?   表示语言标记可有可无；
# - \s*         吃掉语言标记后面的换行/空格；
# - (.*?)       非贪婪地取围栏里的内容；
# - re.DOTALL   让 . 也能匹配换行，JSON 通常跨多行；
# - re.IGNORECASE 让 ```JSON 也能匹配。
_FENCED_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

# 用来区分“json.loads 失败”和“真的解析出了 None”的哨兵值。
# 因为 json.loads("null") 本身会正常返回 None，不能拿 None 当失败标记。
_INVALID_JSON = object()


# 下面三个 dataclass 都加了 frozen=True：
# 对象创建后字段不能修改，避免解析结果在向下传递时被意外改掉。
@dataclass(frozen=True)
class ToolCall:
    """模型决定调用工具。"""

    thought: str  # 模型给出的思考过程；协议里允许省略，省略时是空字符串
    action: str  # 要调用的工具名，例如 weather
    input: dict[str, Any]  # 工具参数，已校验为 JSON 对象


@dataclass(frozen=True)
class FinalAnswer:
    """模型决定直接回复用户。"""

    thought: str  # 思考过程，主要用于 UI 展示和排障
    final: str  # 直接展示给用户的自然语言回答


@dataclass(frozen=True)
class ParseError:
    """模型输出无法解析成合法协议。"""

    reason: str  # 人类可读的错误原因，后续会用于生成纠错提示


# 解析结果只可能是这三种之一。
# TypeAlias 只是给类型起个名字，运行时不产生新类型；
# 调用方通过 isinstance(result, ToolCall) 等方式区分。
ParseResult: TypeAlias = ToolCall | FinalAnswer | ParseError


def parse(raw: str) -> ParseResult:
    """
    解析模型的一次原始输出。

    解析顺序：
    1. 先尝试所有 Markdown 围栏中的内容（通常模型把 JSON 放在这里）；
    2. 再尝试整段文本本身；
    3. 最后扫描全文中的平衡大括号对象；
    4. 返回第一个“既合法 JSON、又符合协议”的对象。

    如果找到了合法 JSON 但协议不对（例如 action/final 同时存在），
    会记住第一条协议错误，在所有候选都尝试完后返回它，
    这样纠错提示能告诉模型具体错在哪里。
    """
    # 空字符串或纯空白没有任何解析价值，直接返回错误。
    if not isinstance(raw, str) or not raw.strip():
        return ParseError("模型输出为空")

    # 去掉首尾空白，后续扫描时位置计算和候选内容都更干净。
    text = raw.strip()

    # 记录“第一条协议错误”，但不要立刻返回：
    # 后面可能还有一个真正正确的对象（例如正文里先举例、再给真正的决策）。
    first_protocol_error: ParseError | None = None

    # _iter_json_candidates 是生成器，会按优先级逐个产出候选片段。
    for candidate in _iter_json_candidates(text):
        # 第一步：把候选文本转成 Python 对象。
        # 失败时返回哨兵值 _INVALID_JSON，而不是抛异常。
        data = _load_json(candidate)
        if data is _INVALID_JSON:
            continue

        # 第二步：检查结构是否符合 action/final 协议。
        result = _validate_protocol(data)
        if isinstance(result, ParseError):
            # 这是合法 JSON，但字段不对。
            # 先记下来，然后继续找后面的候选。
            if first_protocol_error is None:
                first_protocol_error = result
            continue

        # 能走到这里，result 一定是 ToolCall 或 FinalAnswer。
        # 找到第一个有效决策就立即返回，不再扫描剩余文本。
        return result

    # 所有候选都试完了：
    # 优先返回记住的协议错误，因为它比“没找到 JSON”更有诊断价值。
    if first_protocol_error is not None:
        return first_protocol_error
    return ParseError("没有找到合法的 JSON 对象")


def _iter_json_candidates(text: str) -> Iterator[str]:
    """
    按“围栏优先、全文扫描兜底”的顺序产出可能的 JSON 片段。

    这个函数用了 yield，所以调用它不会立刻执行，
    而是每被 for 取一次，才继续往下执行到下一个 yield。
    """
    # 1. 围栏里的内容优先级最高。
    # findall 返回的是每个围栏内部的内容，不包括 ``` 和 json 标记。
    for block in _FENCED_BLOCK_RE.findall(text):
        block = block.strip()
        if block:
            yield block

    # 2. 整段文本本身也可能就是一个 JSON（包括数组或普通值）。
    # 这样像 [1, 2, 3] 这样的输入也能得到“必须是 JSON 对象”的明确错误。
    yield text

    # 3. 最后扫全文里的平衡大括号对象，处理“JSON 前后夹着解释文字”的情况。
    yield from _balanced_objects(text)


def _balanced_objects(text: str) -> Iterator[str]:
    """
    扫描文本中所有“括号平衡”的 {...} 片段。

    这里不能简单用正则，因为：
    - JSON 里可能嵌套对象，例如 {"a": {"b": 1}}；
    - 字符串内部也可能出现花括号，例如 {"query": "怎么用 { }"}。
    """
    # enumerate 同时拿到下标和字符；每个 '{' 都可能是某个对象的起点。
    for start, char in enumerate(text):
        if char != "{":
            continue

        candidate = _read_balanced_object(text, start)
        if candidate is not None:
            yield candidate


def _read_balanced_object(text: str, start: int) -> str | None:
    """
    从 start 位置的 '{' 开始，找到与它匹配的 '}'，返回完整子串。

    这是一个小型状态机，需要维护三个状态：
    - depth：当前嵌套层级，遇到 { 加 1，遇到 } 减 1；
    - in_string：当前是否在 JSON 字符串内部；
    - escaped：当前字符是否被反斜杠转义。
    """
    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]

        # 在字符串内部时，括号不参与嵌套计数。
        # 只有反斜杠和引号有特殊含义：
        # - \" 表示转义引号，不会结束字符串；
        # - \\ 表示转义反斜杠，后面的字符重新按普通规则处理。
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        # 在字符串外部时，只关心引号和花括号。
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            # depth 回到 0，说明找到了与 start 配对的右括号。
            if depth == 0:
                return text[start : index + 1]

    # 遍历到结尾都没回到 0，说明这段不是完整对象。
    return None


def _load_json(candidate: str) -> Any:
    """安全地解析 JSON；失败时返回哨兵值，不抛异常。"""
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # 候选可能只是正文里一段不完整的示例，跳过即可。
        return _INVALID_JSON


def _validate_protocol(data: Any) -> ParseResult:
    """校验 JSON 结构是否符合 Agent 协议。"""

    # json.loads 可能解析出数组、字符串、数字等，协议只接受对象。
    if not isinstance(data, dict):
        return ParseError("输出必须是 JSON 对象，不能是数组或普通值")

    # 用 “键是否存在” 来判断，而不是看值是否为 None。
    # 这样 {"action": null} 会被识别为“出现了 action”，后面再报类型错误，
    # 而不是误判成“没有 action”。
    has_action = "action" in data
    has_final = "final" in data

    if has_action and has_final:
        return ParseError("action 和 final 不能同时出现，必须二选一")
    if not has_action and not has_final:
        return ParseError("缺少 action 或 final，必须二选一")

    # thought 是可选字段：没写或写成 null 都当作空字符串。
    thought = data.get("thought", "")
    if thought is None:
        thought = ""
    if not isinstance(thought, str):
        return ParseError("thought 必须是字符串")

    # 分支一：最终回答。
    if has_final:
        final = data.get("final")
        if not isinstance(final, str) or not final.strip():
            return ParseError("final 必须是非空字符串")
        return FinalAnswer(thought=thought.strip(), final=final.strip())

    # 分支二：调用工具。
    action = data.get("action")
    if not isinstance(action, str) or not action.strip():
        return ParseError("action 必须是非空字符串")

    # input 是可选字段：没写或写成 null 都当作空对象。
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
