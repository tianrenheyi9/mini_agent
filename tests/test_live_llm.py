from __future__ import annotations

import pytest

from app.agent.llm import LLMClient
from app.config import get_settings

pytestmark = pytest.mark.live


async def test_live_chat() -> None:
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
