"""
search 工具（Mock）。

笔试允许 mock 搜索。为了保证测试确定性，这里用固定语料 +
简单的关键词打分，不访问网络。

真实接入时只需要替换 `_search_documents()` 的实现，
ToolSpec、参数 Schema 和上层 Agent 都不需要改。
"""

from __future__ import annotations

from typing import Any

from app.tools.base import (
    ToolContext,
    ToolExecutionError,
    ToolSpec,
    split_search_terms,
)

# 固定语料：id、标题、正文、关键词、URL。
_DOCUMENTS: list[dict[str, Any]] = [
    {
        "id": "doc-agent-loop",
        "title": "如何实现最小 Agent 循环",
        "content": (
            "Agent 循环通常包含：接收用户输入、调用 LLM、解析工具调用、"
            "执行工具、把结果回填给模型，直到模型输出最终回答。"
            "需要限制最大轮次，并为每次工具调用记录 trace。"
        ),
        "keywords": ["agent", "循环", "工具调用", "llm", "runtime", "最大轮次"],
        "url": "mock://docs/agent-loop",
    },
    {
        "id": "doc-openai-chat",
        "title": "OpenAI Chat Completions 接口",
        "content": (
            "Chat Completions 使用 messages 数组描述对话，"
            "响应中的 choices[0].message.content 是模型输出，"
            "usage 字段包含 token 用量。"
        ),
        "keywords": ["openai", "chat", "completions", "messages", "api"],
        "url": "mock://docs/openai-chat",
    },
    {
        "id": "doc-fastapi",
        "title": "FastAPI 快速入门",
        "content": (
            "FastAPI 基于类型注解自动生成接口文档。"
            "用 APIRouter 组织路由，用 TestClient 写接口测试。"
        ),
        "keywords": ["fastapi", "python", "路由", "api", "测试"],
        "url": "mock://docs/fastapi",
    },
    {
        "id": "doc-json",
        "title": "JSON Schema 入门",
        "content": (
            "JSON Schema 用 type、properties、required、enum 等关键字"
            "描述 JSON 数据的结构，常用于工具参数校验。"
        ),
        "keywords": ["json", "schema", "校验", "参数", "工具"],
        "url": "mock://docs/json-schema",
    },
    {
        "id": "doc-context",
        "title": "上下文压缩与 Memory",
        "content": (
            "当对话历史超过预算时，可以把较早的消息总结成摘要，"
            "保留最近若干条原文，并把待办等结构化状态放在 System Prompt 中。"
        ),
        "keywords": ["上下文", "压缩", "memory", "摘要", "token"],
        "url": "mock://docs/context",
    },
    {
        "id": "doc-python-async",
        "title": "Python asyncio 基础",
        "content": (
            "async def 定义协程，await 等待异步结果。"
            "asyncio.wait_for 可以为协程设置超时。"
        ),
        "keywords": ["python", "asyncio", "async", "await", "超时"],
        "url": "mock://docs/python-async",
    },
]


def build_search_tool() -> ToolSpec:
    """创建 search 工具。"""
    return ToolSpec(
        name="search",
        description=(
            "在本地 mock 文档库中检索资料。"
            "当用户询问技术概念、需要参考资料时使用。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词",
                },
                "limit": {
                    "type": "integer",
                    "description": "最多返回几条结果，默认 3",
                    "minimum": 1,
                    "maximum": 5,
                },
            },
            "required": ["query"],
        },
        handler=_search_handler,
    )


async def _search_handler(
    input_data: dict[str, Any],
    context: ToolContext,
) -> dict[str, Any]:
    query = input_data["query"].strip()
    if not query:
        raise ToolExecutionError("搜索关键词不能为空")

    limit = input_data.get("limit", 3)
    results = search_documents(query)[:limit]
    return {
        "query": query,
        "results": results,
        "count": len(results),
    }


def search_documents(query: str) -> list[dict[str, Any]]:
    """
    对固定语料打分并排序。

    打分规则（不追求真实搜索质量，只保证确定性和可解释性）：
    - 完整查询出现在标题：+5
    - 完整查询出现在关键词：+4
    - 完整查询出现在正文：+3
    - 拆出的词项出现在标题/关键词：+2，出现在正文：+1
    """
    normalized = query.strip().lower()
    if not normalized:
        return []

    terms = split_search_terms(normalized)
    scored: list[tuple[int, int, dict[str, Any]]] = []

    for index, document in enumerate(_DOCUMENTS):
        title = document["title"].lower()
        content = document["content"].lower()
        keywords = " ".join(document["keywords"]).lower()

        score = 0
        if normalized in title:
            score += 5
        if normalized in keywords:
            score += 4
        if normalized in content:
            score += 3

        for term in terms:
            if term in title:
                score += 2
            elif term in keywords:
                score += 2
            elif term in content:
                score += 1

        if score > 0:
            scored.append((score, index, document))

    # 分数高的在前；同分时保持语料原顺序（Python 的 sorted 是稳定排序）。
    scored.sort(key=lambda item: -item[0])

    return [
        {
            "id": document["id"],
            "title": document["title"],
            "snippet": _snippet(document["content"]),
            "url": document["url"],
        }
        for _, _, document in scored
    ]


def _snippet(content: str, limit: int = 120) -> str:
    """截取正文开头作为摘要。"""
    if len(content) <= limit:
        return content
    return content[:limit] + "…"
