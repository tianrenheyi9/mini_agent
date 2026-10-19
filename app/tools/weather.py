"""
weather 工具（Mock）。

笔试允许 mock 外部数据。这里用“城市 + 日期”的哈希生成稳定天气：
- 同一个城市、同一天，多次调用结果完全一致；
- 不同城市/日期会得到不同结果；
- 不访问网络，测试不需要 API Key。

真实接入时替换 `_lookup_weather()` 即可。
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from typing import Any

from app.tools.base import ToolContext, ToolExecutionError, ToolSpec

# 城市 -> (基础低温, 温差, 可选天气)
_CITY_PROFILES: dict[str, tuple[int, int, tuple[str, ...]]] = {
    "北京": (18, 8, ("晴", "多云", "阴", "小雨")),
    "上海": (22, 7, ("多云", "阴", "小雨", "晴")),
    "广州": (26, 6, ("多云", "雷阵雨", "晴", "小雨")),
    "深圳": (27, 5, ("多云", "晴", "阵雨", "阴")),
    "杭州": (21, 8, ("晴", "多云", "小雨", "阴")),
    "成都": (19, 7, ("阴", "小雨", "多云", "晴")),
}
_DEFAULT_PROFILE = (20, 8, ("晴", "多云", "阴", "小雨"))


def build_weather_tool() -> ToolSpec:
    """创建 weather 工具。"""
    return ToolSpec(
        name="weather",
        description=(
            "查询指定城市某天的天气（mock 数据，结果稳定）。"
            "当用户询问天气、气温、是否下雨时使用。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名，例如 北京",
                },
                "date": {
                    "type": "string",
                    "description": "日期，格式 YYYY-MM-DD；不传则使用今天",
                },
            },
            "required": ["city"],
        },
        handler=_weather_handler,
    )


async def _weather_handler(
    input_data: dict[str, Any],
    context: ToolContext,
) -> dict[str, Any]:
    city = input_data["city"].strip()
    if not city:
        raise ToolExecutionError("城市不能为空")

    date_text = input_data.get("date")
    if date_text is None or not str(date_text).strip():
        target_date = context.current_time.date()
    else:
        target_date = _parse_date(str(date_text).strip())

    weather = _lookup_weather(city, target_date)
    return {
        "city": city,
        "date": target_date.isoformat(),
        **weather,
    }


def _parse_date(text: str) -> date:
    """校验 YYYY-MM-DD 格式。"""
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ToolExecutionError("date 必须是 YYYY-MM-DD 格式") from exc


def _lookup_weather(city: str, target_date: date) -> dict[str, Any]:
    """
    根据城市和日期生成稳定天气。

    使用 hashlib.md5 而不是 Python 内置 hash()：
    内置 hash() 对字符串带随机盐，进程重启后结果会变。
    """
    base_low, span, conditions = _CITY_PROFILES.get(city, _DEFAULT_PROFILE)

    digest = hashlib.md5(
        f"{city}|{target_date.isoformat()}".encode("utf-8")
    ).digest()

    condition = conditions[digest[0] % len(conditions)]
    offset = digest[1] % 5 - 2  # -2 ~ +2
    low = base_low + offset
    high = low + span

    return {
        "condition": condition,
        "low_celsius": low,
        "high_celsius": high,
        "temperature": f"{low}~{high}℃",
        "advice": _advice(condition, high),
    }


def _advice(condition: str, high: int) -> str:
    """给一句简单的出行建议，让 mock 结果更像真实工具。"""
    if "雨" in condition:
        return "出门记得带伞。"
    if high >= 30:
        return "天气偏热，注意防晒补水。"
    if high <= 10:
        return "天气偏冷，注意保暖。"
    return "天气适宜出行。"
