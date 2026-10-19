"""
真实 LLM 的 Agent 联调测试。

与 `tests/llm/test_live_llm.py` 不同：这里不是直接调用 `chat()`，
而是走完整的 `AgentLoop.run_turn()`，验证：
- 模型能按自定义协议发起工具调用；
- 工具结果能回填给模型；
- 最终回答能返回；
- todo 状态能落盘。

默认 `pytest` 会跳过（-m "not live"）；手动运行：
    pytest -m live -q
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from app.agent.loop import build_agent_loop
from app.config import get_settings

pytestmark = pytest.mark.live


async def test_live_agent_uses_calculator_and_todo(tmp_path) -> None:
    """真实模型通过 Loop 调用 calculator 和 todo，并返回最终回答。"""
    settings = get_settings()
    if not settings.llm_configured:
        pytest.skip("LLM is not configured")

    # 使用临时目录，避免污染项目 data/；预算放宽一些，给模型多步操作空间。
    settings = replace(
        settings,
        data_dir=tmp_path,
        max_tool_calls=6,
        max_llm_calls_per_turn=10,
        max_parse_repairs=2,
        context_token_budget=20_000,
        max_history_messages=40,
    )

    agent = build_agent_loop(settings)
    session = agent.sessions.create("live-agent-test")
    try:
        result = await agent.run_turn(
            session.id,
            (
                "请先用 calculator 计算 2 的 10 次方，"
                "再用 todo 添加一条待办，content 为“2 的 10 次方是 1024”，"
                "最后告诉我结果。"
            ),
        )

        assert result.status == "ok"
        assert "1024" in result.reply

        tool_names = [
            step["tool"]
            for step in result.steps
            if step["type"] == "tool" and step["status"] == "ok"
        ]
        assert "calculator" in tool_names
        assert "todo" in tool_names
        assert result.stats["tool_calls"] >= 2

        reloaded = agent.sessions.get(session.id)
        assert any(
            "1024" in todo.get("content", "")
            for todo in reloaded.todos
        )
    finally:
        await agent.aclose()
