"""weather 工具（mock）的单元测试。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.tools.base import ToolContext
from app.tools.registry import ToolRegistry
from app.tools.weather import build_weather_tool

FIXED_NOW = datetime(
    2026,
    10,
    19,
    9,
    0,
    0,
    tzinfo=timezone(timedelta(hours=8)),
)


def context() -> ToolContext:
    return ToolContext(session_id="s_test", now=FIXED_NOW)


async def test_known_city_returns_weather() -> None:
    """已知城市返回完整天气字段。"""
    registry = ToolRegistry()
    registry.register(build_weather_tool())

    result = await registry.execute(
        "weather",
        {"city": "北京", "date": "2026-10-19"},
        context(),
    )

    assert result.ok is True
    data = result.data
    assert data["city"] == "北京"
    assert data["date"] == "2026-10-19"
    assert data["condition"]
    assert data["low_celsius"] < data["high_celsius"]
    assert "℃" in data["temperature"]
    assert data["advice"]


async def test_same_city_and_date_are_stable() -> None:
    """同城同日多次调用结果必须一致。"""
    registry = ToolRegistry()
    registry.register(build_weather_tool())

    first = await registry.execute(
        "weather", {"city": "上海", "date": "2026-10-19"}, context()
    )
    second = await registry.execute(
        "weather", {"city": "上海", "date": "2026-10-19"}, context()
    )

    assert first.data == second.data


async def test_default_date_uses_context_time() -> None:
    """不传 date 时使用上下文提供的当前日期。"""
    registry = ToolRegistry()
    registry.register(build_weather_tool())

    result = await registry.execute("weather", {"city": "广州"}, context())

    assert result.ok is True
    assert result.data["date"] == "2026-10-19"


async def test_unknown_city_uses_default_profile() -> None:
    """未知城市也能返回稳定结果，不会报错。"""
    registry = ToolRegistry()
    registry.register(build_weather_tool())

    result = await registry.execute(
        "weather", {"city": "拉萨", "date": "2026-10-19"}, context()
    )

    assert result.ok is True
    assert result.data["city"] == "拉萨"


async def test_invalid_date_is_rejected() -> None:
    """日期格式错误时返回业务错误。"""
    registry = ToolRegistry()
    registry.register(build_weather_tool())

    result = await registry.execute(
        "weather", {"city": "北京", "date": "10/19/2026"}, context()
    )

    assert result.ok is False
    assert "YYYY-MM-DD" in result.error


async def test_empty_city_is_rejected() -> None:
    """空城市名返回业务错误。"""
    registry = ToolRegistry()
    registry.register(build_weather_tool())

    result = await registry.execute("weather", {"city": "   "}, context())

    assert result.ok is False
    assert "城市不能为空" in result.error
