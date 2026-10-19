"""工具实现与注册表。"""

from app.tools.calculator import build_calculator_tool
from app.tools.registry import ToolRegistry
from app.tools.search import build_search_tool
from app.tools.todo import build_todo_tool
from app.tools.weather import build_weather_tool


def build_default_registry(*, timeout_seconds: float = 10.0) -> ToolRegistry:
    """
    创建包含全部默认工具的注册表。

    注册顺序就是 Prompt 中工具目录的顺序：
    todo（题目必需）放在最前，随后是 calculator、search、weather。
    """
    registry = ToolRegistry(timeout_seconds=timeout_seconds)
    for build_tool in (
        build_todo_tool,
        build_calculator_tool,
        build_search_tool,
        build_weather_tool,
    ):
        registry.register(build_tool())
    return registry


__all__ = [
    "ToolRegistry",
    "build_default_registry",
    "build_calculator_tool",
    "build_search_tool",
    "build_todo_tool",
    "build_weather_tool",
]
