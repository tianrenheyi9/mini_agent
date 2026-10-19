"""
工具注册表。

职责：
1. 保存 name -> ToolSpec 的映射；
2. 拒绝重复注册，尽早暴露配置错误；
3. 统一执行工具：参数校验、超时控制、异常转换、耗时统计。

Agent 循环只依赖 `execute()`，不需要知道具体有哪些工具，
也不需要为每个工具写 if/else。
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from app.tools.base import (
    ToolContext,
    ToolExecutionError,
    ToolResult,
    ToolSpec,
    validate_parameters,
)

logger = logging.getLogger(__name__)


class ToolRegistry:
    """工具注册与执行中心。"""

    def __init__(self, *, timeout_seconds: float = 10.0) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._timeout_seconds = timeout_seconds

    def register(self, spec: ToolSpec) -> None:
        """注册一个工具；名称重复时直接报错。"""
        if not spec.name or not spec.name.strip():
            raise ValueError("工具名称不能为空")
        if not spec.description.strip():
            raise ValueError(f"工具 {spec.name!r} 缺少描述")
        if not callable(spec.handler):
            raise ValueError(f"工具 {spec.name!r} 的 handler 不可调用")
        if spec.parameters.get("type") != "object":
            raise ValueError(
                f"工具 {spec.name!r} 的参数 Schema 必须是 type=object"
            )
        if spec.name in self._tools:
            raise ValueError(f"工具 {spec.name!r} 已经注册过")

        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        """按名称查找工具；不存在时返回 None。"""
        return self._tools.get(name)

    def names(self) -> list[str]:
        """返回所有工具名，顺序与注册顺序一致。"""
        return list(self._tools)

    def tools(self) -> list[ToolSpec]:
        """返回所有 ToolSpec，供 Prompt 渲染工具目录。"""
        return list(self._tools.values())

    async def execute(
        self,
        name: str,
        raw_input: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """
        执行工具，并把所有失败都转换成 ToolResult。

        这样 Agent 循环永远不需要 try/except：
        工具报错只是一条 observation，模型可以据此自我纠正。
        """
        started = time.perf_counter()

        spec = self._tools.get(name)
        if spec is None:
            available = "、".join(self._tools) or "（无）"
            return self._failure(
                name,
                f"未知工具 {name!r}。可用工具：{available}",
                started,
            )

        if not isinstance(raw_input, dict):
            return self._failure(name, "工具参数必须是 JSON 对象", started)

        # 在执行前校验参数，避免工具内部出现 AttributeError/KeyError。
        errors = validate_parameters(spec.parameters, raw_input)
        if errors:
            return self._failure(
                name,
                "参数校验失败：" + "；".join(errors),
                started,
            )

        try:
            data = await asyncio.wait_for(
                spec.handler(raw_input, context),
                timeout=self._timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning("Tool %s timed out after %.1fs", name, self._timeout_seconds)
            return self._failure(
                name,
                f"工具执行超时（>{self._timeout_seconds:g} 秒）",
                started,
            )
        except ToolExecutionError as exc:
            # 预期内的业务错误，不打印堆栈。
            logger.info("Tool %s returned an error: %s", name, exc)
            return self._failure(name, str(exc), started)
        except Exception as exc:  # noqa: BLE001 - 边界层必须兜底
            # 未预期的程序错误：完整堆栈进日志，给模型的信息保持简短。
            logger.exception("Tool %s crashed", name)
            return self._failure(
                name,
                f"工具执行失败：{type(exc).__name__}",
                started,
            )

        return ToolResult(
            tool=name,
            ok=True,
            data=data,
            duration_ms=_elapsed_ms(started),
        )

    @staticmethod
    def _failure(name: str, message: str, started: float) -> ToolResult:
        return ToolResult(
            tool=name,
            ok=False,
            error=message,
            duration_ms=_elapsed_ms(started),
        )


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
