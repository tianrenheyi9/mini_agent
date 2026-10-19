"""
真实 LLM 冒烟测试（需要 .env 中配置好 API Key）。

默认的 `pytest` 会通过 pytest.ini 中的 `-m "not live"` 跳过本文件，
避免没有 Key 的环境或 CI 失败。

手动运行：
    pytest -m live -q
"""

from __future__ import annotations

import pytest

from app.agent.llm import LLMClient
from app.config import get_settings

# 给整个文件打上 live 标记，便于按标记筛选。
pytestmark = pytest.mark.live


async def test_live_chat() -> None:
    """用真实模型发一条最简单的中文指令，确认网络、鉴权和响应解析都正常。"""
    settings = get_settings()
    if not settings.llm_configured:
        pytest.skip("LLM is not configured")

    client = LLMClient(settings)
    result = await client.chat(
        [{"role": "user", "content": "只回复两个字：你好"}],
        temperature=0.0,
    )

    assert result.content.strip()
    assert result.finish_reason is not None
