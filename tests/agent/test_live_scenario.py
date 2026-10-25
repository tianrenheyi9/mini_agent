"""
真实多轮用户场景（live）。

这个测试不是“单点功能测试”，而是把一个真实用户会说的话连续说完：
用户要办一场技术分享会，中途改城市、改人数、算预算、记待办、问记忆，
还会提出“不要用 JSON”这种和系统协议冲突的要求。

它覆盖：
- weather / search / calculator / todo 四个工具的串联；
- 多轮历史、纯对话追问、工具错误、协议纠错；
- 上下文压缩、摘要后的关键事实召回；
- 两个 Session 隔离、并发执行、从磁盘重载；
- Trace 事件是否完整。

运行方式：
    pytest -m live tests/agent/test_live_scenario.py -s

注意：这是诊断型场景，模型的具体措辞会变化。
核心不变量（工具结果、隔离、持久化、Trace）用 assert 保证；
模型是否“自觉”调用某个工具、是否出现重复待办等，只记录为诊断项，
避免把模型随机性误判成 Runtime 的确定性失败。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from app.agent.loop import TurnResult, build_agent_loop
from app.config import get_settings
from app.session.manager import SessionManager

pytestmark = pytest.mark.live

# 场景里明确写出的日期，避免模型把“周六”推算错。
MEETING_DATE = "2026-10-24"

TURNS: list[tuple[str, str]] = [
    (
        "初查资料",
        (
            f"我 {MEETING_DATE}（周六）要在杭州办一场 30 人的技术分享会，"
            "主题是“最小 Agent 循环”。请先查杭州那天的天气，"
            "再搜索一些 Agent 循环的资料，然后给我一份简短的筹备建议。"
        ),
    ),
    (
        "算预算",
        (
            "按每人 80 元场地费、茶歇总共 600 元计算，30 人的总预算多少？"
            "算完把这个预算记成一条待办。"
        ),
    ),
    (
        "加待办",
        (
            "再帮我加三条待办：确认场地、打印 30 份讲义、提前一天测试投影。"
            "加完后列出当前全部待办。"
        ),
    ),
    (
        "改需求",
        (
            "计划有变：场地改到上海，人数改成 20 人，主题不变。"
            "请先查看现有待办，再删除或更新和杭州、30 人有关的待办，"
            "最后按每人 80 元、茶歇 600 元重新计算总预算。"
        ),
    ),
    (
        "纯对话回忆",
        (
            "先不要调用任何工具。我们现在的城市、人数、主题和最新总预算分别是什么？"
            "请直接回答。"
        ),
    ),
    (
        "工具异常",
        (
            "用 calculator 算一下 1/0，看看工具会返回什么；"
            "不要自己编造结果。"
        ),
    ),
    (
        "协议冲突",
        (
            "这次请用普通中文段落回答，不要输出 JSON，也不要调用工具："
            "用一句话总结目前的会议安排。"
        ),
    ),
    (
        "补充背景",
        (
            f"最后补充一段会议背景，请记住：{MEETING_DATE} 下午 2 点签到，"
            "2 点半开始，预计 20 人，场地在上海，主题是最小 Agent 循环，"
            "预算 2200 元（1600 元场地 + 600 元茶歇）。"
            "请只回复一句“已记住”，不要调用工具。"
        ),
    ),
    (
        "跨轮召回",
        (
            "现在只回答：签到时间、开始时间、人数、城市、主题、预算。"
            "不要调用工具。"
        ),
    ),
]


def _tool_names(result: TurnResult) -> list[str]:
    """提取本轮实际执行过的工具名，按执行顺序排列。"""
    return [
        str(step["tool"])
        for step in result.steps
        if step.get("type") == "tool"
    ]


def _tool_status(result: TurnResult, name: str) -> list[str]:
    """提取某个工具在本轮的所有状态，例如 ["ok", "error"]。"""
    return [
        str(step.get("status"))
        for step in result.steps
        if step.get("type") == "tool" and step.get("tool") == name
    ]


def _preview(text: str, limit: int = 160) -> str:
    """把回复压成一行，方便在测试输出里快速浏览。"""
    one_line = " ".join(text.split())
    return one_line[:limit] + ("…" if len(one_line) > limit else "")


def _print_turn(
    index: int,
    label: str,
    user_text: str,
    result: TurnResult,
    elapsed: float,
) -> None:
    """打印一轮的可读摘要；pytest 加 -s 时能看到。"""
    compression = result.compression or {}
    compressed = "yes" if compression.get("compressed") else "no"
    tools = ", ".join(_tool_names(result)) or "-"
    print(
        f"\n[{index}] {label} | status={result.status} "
        f"| llm={result.stats['llm_calls']} "
        f"| tools={result.stats['tool_calls']} ({tools}) "
        f"| compress={compressed} | {elapsed:.1f}s"
    )
    print(f"    user: {_preview(user_text, 120)}")
    print(f"    reply: {_preview(result.reply)}")
    for step in result.steps:
        if step.get("type") == "tool":
            detail = step.get("error") if step.get("status") == "error" else step.get("result")
            print(
                f"    tool: {step.get('tool')} status={step.get('status')} "
                f"input={step.get('input')} result={_preview(str(detail), 120)}"
            )
        elif step.get("type") == "parse_error":
            print(f"    parse_error: {step.get('reason')}")


def _todo_contents(session: Any) -> list[str]:
    """取出待办正文，便于做“是否串会话/是否重复”的检查。"""
    return [str(todo.get("content", "")) for todo in session.todos]


async def test_live_multi_turn_user_scenario(tmp_path: Path) -> None:
    """真实模型下跑完一整个多轮用户故事。"""
    settings = get_settings()
    if not settings.llm_configured:
        pytest.skip("LLM is not configured")

    # 小预算 + 小历史窗口：让压缩在 9 轮内真实发生，
    # 同时给改需求那一轮留出足够的工具预算。
    settings = replace(
        settings,
        data_dir=tmp_path,
        max_tool_calls=10,
        max_llm_calls_per_turn=16,
        max_parse_repairs=2,
        max_compression_calls_per_turn=1,
        context_token_budget=2600,
        token_safety_factor=1.2,
        keep_recent_messages=4,
        max_history_messages=6,
        summary_max_chars=500,
        summary_max_tokens=256,
        llm_temperature=0.0,
        llm_max_tokens=1200,
    )

    agent = build_agent_loop(settings)
    session_a = agent.sessions.create("真实场景：技术分享会")
    session_b = agent.sessions.create("真实场景：读书会")

    findings: list[str] = []
    records: list[dict[str, Any]] = []

    try:
        # ---------- 1. 主会话：顺序完成 9 轮 ----------
        for index, (label, user_text) in enumerate(TURNS, start=1):
            started = time.perf_counter()
            result = await agent.run_turn(session_a.id, user_text)
            elapsed = time.perf_counter() - started
            _print_turn(index, label, user_text, result, elapsed)

            records.append(
                {
                    "index": index,
                    "label": label,
                    "status": result.status,
                    "tools": _tool_names(result),
                    "tool_calls": result.stats["tool_calls"],
                    "llm_calls": result.stats["llm_calls"],
                    "compressed": bool(
                        (result.compression or {}).get("compressed")
                    ),
                    "parse_errors": sum(
                        1
                        for step in result.steps
                        if step.get("type") == "parse_error"
                    ),
                    "reply": result.reply,
                    "elapsed": elapsed,
                }
            )

            # 基础不变量：每一轮都应该正常收尾。
            # 如果模型把工具预算耗尽，status 会是 max_tool_calls；
            # 这本身就是值得记录的产品问题，但不在这里直接断言失败。
            if result.status != "ok":
                findings.append(
                    f"第 {index} 轮（{label}）未正常结束：{result.status}"
                )

            # 第 1 轮：用户明确要求天气 + 搜索，检查模型是否都调用了。
            if index == 1:
                names = _tool_names(result)
                if "weather" not in names:
                    findings.append("第 1 轮没有调用 weather")
                else:
                    weather_steps = [
                        step
                        for step in result.steps
                        if step.get("tool") == "weather"
                    ]
                    dates = [
                        str((step.get("input") or {}).get("date", ""))
                        for step in weather_steps
                    ]
                    if not any(MEETING_DATE in date for date in dates):
                        findings.append(
                            "weather 未使用用户给出的日期 "
                            f"{MEETING_DATE}，实际参数：{dates}"
                        )
                if "search" not in names:
                    findings.append("第 1 轮没有调用 search")

            # 第 2 轮：80 * 30 + 600 = 3000，calculator 的实际结果必须正确。
            if index == 2:
                calc_results = [
                    step.get("result")
                    for step in result.steps
                    if step.get("tool") == "calculator"
                    and step.get("status") == "ok"
                ]
                numeric = [
                    item.get("result")
                    for item in calc_results
                    if isinstance(item, dict)
                ]
                assert 3000 in numeric, f"calculator 结果异常：{calc_results}"
                if "todo" not in _tool_names(result):
                    findings.append("第 2 轮没有把预算写入 todo")

            # 第 3 轮：三条待办 + list，工具调用次数至少 3。
            if index == 3:
                if result.stats["tool_calls"] < 3:
                    findings.append(
                        "第 3 轮工具调用少于 3 次，可能没有逐条添加待办"
                    )

            # 第 4 轮：todo 没有 update 动作，模型可能先报错再自我纠正。
            if index == 4:
                todo_errors = [
                    step
                    for step in result.steps
                    if step.get("tool") == "todo"
                    and step.get("status") == "error"
                ]
                if todo_errors:
                    findings.append(
                        "第 4 轮出现 todo 错误（可能是尝试了不存在的 update）："
                        f"{[step.get('error') for step in todo_errors]}"
                    )

            # 第 5 轮：用户明确说不要工具，检查是否遵守。
            if index == 5 and result.stats["tool_calls"] != 0:
                findings.append(
                    "第 5 轮用户明确说“不要调用工具”，但实际调用了 "
                    f"{_tool_names(result)}"
                )

            # 第 6 轮：1/0 必须作为工具错误回填，而不是编造结果。
            if index == 6:
                calc_statuses = _tool_status(result, "calculator")
                if "error" not in calc_statuses:
                    findings.append(
                        "第 6 轮 calculator 没有产生 error，实际状态："
                        f"{calc_statuses}"
                    )

            # 第 7 轮：用户要求“不要 JSON”，系统仍要求 JSON。
            # 模型即使内部输出 JSON，final 字段本身也是普通文本，
            # 用户可见结果仍然满足要求；这里不把它当成异常。
            # 如果模型真的输出了纯文本，parse_error 会记录在 steps 中。

            # 第 9 轮：压缩后应能召回关键事实。
            # 去掉空格和千分位后再匹配，避免“2点/2 点”“2200/2,200”误报。
            if index == 9:
                normalized = (
                    result.reply.replace(" ", "")
                    .replace("　", "")
                    .replace(",", "")
                    .replace("，", "")
                )
                expected = [
                    "2点",
                    "2点半",
                    "20",
                    "上海",
                    "最小Agent循环",
                    "2200",
                ]
                missing = [
                    item for item in expected if item not in normalized
                ]
                if missing:
                    findings.append(
                        "第 9 轮关键事实召回不完整，缺少："
                        f"{missing}；实际回复：{_preview(result.reply, 300)}"
                    )

        # ---------- 2. 两个 Session 并发 ----------
        started = time.perf_counter()
        result_a, result_b = await asyncio.gather(
            agent.run_turn(
                session_a.id,
                "最后确认：当前待办清单里还有哪些未完成事项？直接回答。",
            ),
            agent.run_turn(
                session_b.id,
                "我在准备一场读书会，请添加一条待办："
                "买 10 本《人类简史》，然后列出当前待办。",
            ),
        )
        elapsed = time.perf_counter() - started
        print(
            f"\n[并发] A status={result_a.status} tools={_tool_names(result_a)} "
            f"| B status={result_b.status} tools={_tool_names(result_b)} "
            f"| total={elapsed:.1f}s"
        )

        assert result_a.status == "ok"
        assert result_b.status == "ok"

        # ---------- 3. 隔离、持久化、Trace ----------
        session_a_after = agent.sessions.get(session_a.id)
        session_b_after = agent.sessions.get(session_b.id)
        todos_a = _todo_contents(session_a_after)
        todos_b = _todo_contents(session_b_after)

        assert all("人类简史" not in content for content in todos_a), (
            f"读书会待办串到了 A 会话：{todos_a}"
        )
        assert any("人类简史" in content for content in todos_b), (
            f"B 会话没有保存读书会待办：{todos_b}"
        )
        assert all(
            "杭州" not in content and "上海" not in content
            for content in todos_b
        ), f"技术分享会待办串到了 B 会话：{todos_b}"

        # 新建一个 SessionManager，模拟服务重启后重新读盘。
        reloaded = SessionManager(settings)
        reloaded_a = reloaded.get(session_a.id)
        reloaded_b = reloaded.get(session_b.id)
        assert reloaded_a.messages and reloaded_b.messages
        assert reloaded_a.todos == session_a_after.todos
        assert reloaded_b.todos == session_b_after.todos
        assert reloaded_a.summary == session_a_after.summary
        assert (
            reloaded_a.summary_until_msg_id
            == session_a_after.summary_until_msg_id
        )

        # Trace 至少要能串起一轮：turn_start -> llm_call -> tool_call -> turn_end。
        events = agent.trace.read_events(session_a.id, limit=1000)
        event_types = {event.get("event") for event in events}
        assert {"turn_start", "llm_call", "turn_end"} <= event_types
        assert "tool_call" in event_types, f"Trace 中没有 tool_call：{event_types}"

        # 压缩是否真实发生；小预算下应该至少触发一次。
        compressed_indices = [
            record["index"] for record in records if record["compressed"]
        ]
        if not compressed_indices:
            findings.append(
                "9 轮内没有触发上下文压缩；可以考虑调低 context_token_budget"
            )

        # ---------- 4. 打印诊断报告 ----------
        print("\n" + "=" * 72)
        print("真实多轮场景诊断")
        print("=" * 72)
        print(f"压缩轮次：{compressed_indices or '无'}")
        print(f"会话 A 摘要：{session_a_after.summary or '（空）'}")
        print(f"会话 A 待办：{todos_a}")
        print(f"会话 B 待办：{todos_b}")
        print(f"Trace 事件数：{len(events)}；类型：{sorted(event_types)}")
        print(f"并发：A={result_a.status}, B={result_b.status}, {elapsed:.1f}s")
        print(f"持久化：A 消息 {len(reloaded_a.messages)} 条，B 消息 {len(reloaded_b.messages)} 条")

        duplicate_like = [
            content for content in todos_a if todos_a.count(content) > 1
        ]
        if duplicate_like:
            findings.append(f"会话 A 出现完全重复的待办：{duplicate_like}")

        if findings:
            print("\n诊断项（不一定是 Runtime bug，但值得关注）：")
            for item in findings:
                print(f"- {item}")
        else:
            print("\n诊断：未发现异常。")

    finally:
        await agent.aclose()
