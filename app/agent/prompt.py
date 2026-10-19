"""
System Prompt 与协议文本的构建。

System Prompt 每轮都要发送给模型，承担四个职责：
1. 告诉模型它是一个可以使用工具的 Agent；
2. 约定严格的 JSON 输出协议；
3. 列出工具目录（名称、描述、参数 Schema），让模型自主决策；
4. 注入会话摘要、待办快照和当前时间。

工具目录来自未来的 ToolRegistry。为了避免 prompt 模块依赖工具实现，
这里只要求传入对象具有 `name / description / parameters` 三个属性。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Protocol


class ToolLike(Protocol):
    """
    工具目录渲染所需的最小接口。

    Protocol 是 Python 的“结构化类型”：
    只要一个对象拥有下面三个属性，就自动满足这个协议，不需要显式继承。
    因此后续 ToolRegistry 里的 ToolSpec 可以直接传进来，prompt 模块无需 import 它。
    """

    name: str
    description: str
    parameters: Mapping[str, Any]


def render_tool_catalog(tools: Sequence[ToolLike]) -> str:
    """
    把工具列表渲染成 System Prompt 中的“可用工具”段落。

    每个工具包含：
    - 名称：模型调用时写在 action 字段里的值；
    - 描述：告诉模型什么时候该用它；
    - 参数 Schema：告诉模型 input 应该是什么结构。
    """
    if not tools:
        return "（当前没有可用工具，请直接用 final 回复用户。）"

    blocks: list[str] = []
    for tool in tools:
        # ensure_ascii=False 让中文描述保持可读，而不是变成 \uXXXX；
        # indent=2 让 Schema 有换行和缩进，模型更容易准确理解结构。
        schema = json.dumps(
            tool.parameters,
            ensure_ascii=False,
            indent=2,
        )
        blocks.append(
            f"### {tool.name}\n"
            f"用途：{tool.description}\n"
            f"参数 Schema：\n"
            f"```json\n{schema}\n```"
        )

    # 用空行分隔多个工具，Prompt 里的层级更清楚。
    return "\n\n".join(blocks)


def render_todos(todos: Sequence[Mapping[str, Any]]) -> str:
    """
    把会话中的待办渲染成快照。

    `[ ]` 表示未完成，`[x]` 表示已完成。
    id 必须显示出来，模型后续才能精确地完成任务或删除任务。
    """
    if not todos:
        return "（暂无待办）"

    lines: list[str] = []
    for todo in todos:
        # get 的第二个参数是默认值：
        # 即使某条待办数据缺字段，也不会抛 KeyError。
        todo_id = str(todo.get("id", "?"))
        content = str(todo.get("content", "")).strip()
        mark = "x" if todo.get("done") else " "
        lines.append(f"- [{mark}] {todo_id}: {content}")
    return "\n".join(lines)


def build_system_prompt(
    tools: Sequence[ToolLike],
    *,
    summary: str = "",
    todos: Sequence[Mapping[str, Any]] = (),
    now: datetime | None = None,
) -> str:
    """
    组装完整的 System Prompt。

    参数：
    - tools：当前注册的工具；
    - summary：会话压缩后的长期摘要，没有就传空字符串；
    - todos：当前会话的待办列表；
    - now：当前时间，测试时可以传入固定值，保证输出可断言。
    """
    # now 为空时取当前带时区的时间；
    # %z 会输出形如 +0800 的时区偏移，避免模型误判“今天”。
    current_time = (now or datetime.now().astimezone()).strftime(
        "%Y-%m-%d %H:%M:%S %z"
    )

    # 空摘要用占位文本，避免 Prompt 里出现一个空段落让模型困惑。
    summary_text = summary.strip() or "（暂无历史摘要）"

    catalog = render_tool_catalog(tools)
    todo_text = render_todos(todos)

    # 注意：下面是 f-string，JSON 示例里的花括号必须写成 {{ 和 }}，
    # 渲染后才会变成单个 { 和 }。
    return f"""你是一个可以调用工具的智能助手。
你需要根据用户目标、会话摘要和工具执行结果，决定下一步是调用工具还是直接回复。

## 输出协议（必须严格遵守）
每次回复必须且只能输出一个 JSON 对象，不要输出 JSON 之外的任何文字，也不要用 Markdown 代码块包裹。

调用工具时输出：
{{"thought": "一句话说明你为什么调用这个工具", "action": "工具名", "input": {{"参数名": "参数值"}}}}

最终回答时输出：
{{"thought": "一句话说明你的判断", "final": "直接展示给用户的完整回答"}}

规则：
1. `action` 和 `final` 必须二选一，不能同时出现。
2. `thought` 可选，但建议填写，用来记录当前判断。
3. `input` 必须是 JSON 对象，字段名和类型必须符合工具 Schema。
4. 只能使用下面列出的工具，不要编造工具名或参数。
5. 工具返回错误时，先根据错误信息修正参数；无法修正时向用户说明。
6. 拿到足够信息后立即输出 final，不要为了调用工具而调用工具。
7. final 应该是完整、可直接展示给用户的自然语言回答。

## 可用工具
{catalog}

## 会话摘要
{summary_text}

## 当前待办
{todo_text}

## 当前时间
{current_time}
"""


def build_repair_message(
    raw_output: str,
    error: str,
    *,
    max_raw_chars: int = 1200,
) -> str:
    """
    构造“协议纠错”消息。

    当解析失败时，Agent 循环会把这条消息追加到上下文中，
    让模型看到自己哪里错了，并要求它重新输出。
    原始输出会截断，避免把超长内容再次塞进上下文。
    """
    snippet = raw_output.strip()

    # 只保留前 max_raw_chars 个字符。
    # 加省略号是为了让模型知道内容被截断了，而不是它原本就这么短。
    if len(snippet) > max_raw_chars:
        snippet = snippet[:max_raw_chars] + "…"

    return f"""你的上一条输出不符合约定协议。
解析错误：{error}
你刚才的输出（已截断）：
{snippet}

请重新输出，并且只输出一个合法的 JSON 对象：
- 调用工具：{{"thought": "...", "action": "工具名", "input": {{...}}}}
- 最终回答：{{"thought": "...", "final": "..."}}"""
