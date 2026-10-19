"""
测试用的假 LLM。

真实 LLM 的回复不可预测，而且需要网络和 API Key。
在测试 Agent 循环时，我们希望“模型这一轮返回什么”完全可控，
所以写一个 FakeLLM：按预先给定的脚本依次返回结果。

它和 LLMClient 拥有相同的 `chat()` 方法签名，
因此 AgentLoop 不需要知道自己拿到的是真模型还是假模型。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from app.agent.llm import LLMResponse


class FakeLLM:
    """按脚本返回预设内容或抛出预设异常的测试替身。"""

    def __init__(self, responses: Iterable[str | Exception]) -> None:
        # 把脚本转成列表，方便用 pop(0) 依次取出。
        self._responses = list(responses)
        # 记录每次调用收到的 messages，测试中可以断言“上下文里有什么”。
        self.calls: list[list[dict[str, Any]]] = []

    async def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool | None = None,
    ) -> LLMResponse:
        # 深拷贝一份消息，避免后续代码修改列表影响测试记录。
        self.calls.append([dict(message) for message in messages])

        if not self._responses:
            # 脚本用完了还继续调用，通常说明循环次数或测试预期写错了。
            raise AssertionError("FakeLLM has no scripted response left")

        item = self._responses.pop(0)
        if isinstance(item, Exception):
            # 允许脚本注入异常，用来测试重试、降级等异常路径。
            raise item

        return LLMResponse(
            content=item,
            model="fake",
            prompt_tokens=None,
            completion_tokens=None,
            finish_reason="stop",
            latency_ms=0,
        )
