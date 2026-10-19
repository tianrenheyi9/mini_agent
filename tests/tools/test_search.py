"""search 工具（mock）的单元测试。"""

from __future__ import annotations

from app.tools.base import ToolContext
from app.tools.registry import ToolRegistry
from app.tools.search import build_search_tool, search_documents


def test_finds_agent_loop_document() -> None:
    """查询 Agent 循环时应优先命中对应文档。"""
    results = search_documents("Agent 循环")

    assert results
    assert results[0]["id"] == "doc-agent-loop"


def test_finds_fastapi_document() -> None:
    """英文关键词同样可以命中。"""
    results = search_documents("FastAPI")

    assert results
    assert results[0]["id"] == "doc-fastapi"


def test_finds_json_schema_document() -> None:
    """多个关键词组合也能命中正确文档。"""
    results = search_documents("JSON Schema")

    assert results
    assert results[0]["id"] == "doc-json"


def test_no_results() -> None:
    """没有匹配时返回空列表，而不是报错。"""
    assert search_documents("量子力学") == []


def test_results_are_deterministic() -> None:
    """同样的查询必须返回同样的顺序。"""
    assert search_documents("上下文 压缩") == search_documents("上下文 压缩")


async def test_limit_is_respected() -> None:
    """limit 参数限制返回数量。"""
    registry = ToolRegistry()
    registry.register(build_search_tool())
    context = ToolContext(session_id="s_test")

    result = await registry.execute(
        "search", {"query": "python", "limit": 1}, context
    )

    assert result.ok is True
    assert result.data["count"] == 1
    assert len(result.data["results"]) == 1


async def test_limit_out_of_range_is_rejected() -> None:
    """limit 超出 1~5 时由 Schema 拦截。"""
    registry = ToolRegistry()
    registry.register(build_search_tool())

    result = await registry.execute(
        "search",
        {"query": "python", "limit": 10},
        ToolContext(session_id="s_test"),
    )

    assert result.ok is False
    assert "不能大于 5" in result.error


async def test_empty_query_is_rejected() -> None:
    """空查询返回业务错误。"""
    registry = ToolRegistry()
    registry.register(build_search_tool())

    result = await registry.execute(
        "search", {"query": "   "}, ToolContext(session_id="s_test")
    )

    assert result.ok is False
    assert "搜索关键词不能为空" in result.error
