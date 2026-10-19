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
        # dict 在 Python 3.7+ 保持插入顺序，
        # 所以 Prompt 里的工具目录顺序 = 注册顺序，测试可以稳定断言。
        self._tools: dict[str, ToolSpec] = {}
        self._timeout_seconds = timeout_seconds

    def register(self, spec: ToolSpec) -> None:
        """
        注册一个工具。

        这些检查都在应用启动阶段执行，
        配置写错时能立刻失败，而不是等模型调用时才暴露。
        """
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

        # 第 1 步：查找工具。未知工具不是程序错误，
        # 而是模型决策错误，所以把可用工具列表回填给它。
        spec = self._tools.get(name)
        if spec is None:
            available = "、".join(self._tools) or "（无）"
            return self._failure(
                name,
                f"未知工具 {name!r}。可用工具：{available}",
                started,
            )

        # 第 2 步：确认 input 是 JSON 对象。
        # 虽然解析器已经保证 input 是 dict，但注册表也是执行边界，需要再兜底一次。
        if not isinstance(raw_input, dict):
            return self._failure(name, "工具参数必须是 JSON 对象", started)

        # 第 3 步：按工具自己的 Schema 校验参数。
        # 这一步会在 handler 执行前拦住缺字段、类型错误、枚举越界等问题。
        errors = validate_parameters(spec.parameters, raw_input)
        if errors:
            return self._failure(
                name,
                "参数校验失败：" + "；".join(errors),
                started,
            )

        # 第 4 步：执行工具，并设置超时。
        # wait_for 会在超时后取消协程，避免一个卡住的工具拖死整个请求。
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
            # 预期内的业务错误，例如“待办不存在”“除以 0”。
            # 这类错误只记录 info，不打印堆栈。
            logger.info("Tool %s returned an error: %s", name, exc)
            return self._failure(name, str(exc), started)
        except Exception as exc:  # noqa: BLE001 - 边界层必须兜底
            # 未预期的程序错误：完整堆栈进日志，给模型的信息保持简短，
            # 避免把文件路径、堆栈等内部细节泄漏到对话里。
            logger.exception("Tool %s crashed", name)
            return self._failure(
                name,
                f"工具执行失败：{type(exc).__name__}",
                started,
            )

        # 第 5 步：成功返回。
        # 注意：asyncio.CancelledError 继承自 BaseException，不会被 except Exception 捕获，
        # 因此客户端断开时的取消操作会正常向上传播。
        return ToolResult(
            tool=name,
            ok=True,
            data=data,
            duration_ms=_elapsed_ms(started),
        )

    @staticmethod
    def _failure(name: str, message: str, started: float) -> ToolResult:
        """构造一个失败结果，避免每个错误分支重复写字段。"""
        return ToolResult(
            tool=name,
            ok=False,
            error=message,
            duration_ms=_elapsed_ms(started),
        )


def _elapsed_ms(started: float) -> int:
    """把 perf_counter 的秒数差转换成整数毫秒。"""
    return int((time.perf_counter() - started) * 1000)
