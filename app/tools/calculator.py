"""
calculator 工具：安全地计算算术表达式。

为什么不直接用 eval()？
- eval 会执行任意 Python 代码，例如 __import__("os").system(...)；
- 这里用 ast 解析表达式，只允许数字、四则运算、整除、取模和乘方；
- 变量、函数调用、属性访问、下标、lambda 等一律拒绝。
"""

from __future__ import annotations

import ast
import math
from typing import Any

from app.tools.base import ToolContext, ToolExecutionError, ToolSpec

# 限制表达式长度和指数大小，避免 2**1000000 这类输入耗尽 CPU/内存。
MAX_EXPRESSION_LENGTH = 200
MAX_EXPONENT = 100
MAX_INTEGER_BITS = 10_000


def build_calculator_tool() -> ToolSpec:
    """创建 calculator 工具。"""
    return ToolSpec(
        name="calculator",
        description=(
            "计算数学表达式，支持 + - * / // % ** 和括号。"
            "当用户需要精确计算时使用。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "要计算的表达式，例如 (12 + 8) * 3 / 2",
                }
            },
            "required": ["expression"],
        },
        handler=_calculator_handler,
    )


async def _calculator_handler(
    input_data: dict[str, Any],
    context: ToolContext,
) -> dict[str, Any]:
    expression = input_data["expression"]
    result = evaluate_expression(expression)
    return {
        "expression": expression.strip(),
        "result": result,
    }


def evaluate_expression(expression: str) -> int | float:
    """解析并计算表达式，非法输入统一抛出 ToolExecutionError。"""
    if not isinstance(expression, str) or not expression.strip():
        raise ToolExecutionError("表达式不能为空")

    expression = expression.strip()
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ToolExecutionError(
            f"表达式过长（最多 {MAX_EXPRESSION_LENGTH} 个字符）"
        )

    try:
        # mode="eval" 只允许单个表达式，不允许语句（如 import、赋值）。
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError) as exc:
        raise ToolExecutionError("表达式语法错误") from exc

    try:
        result = _eval_node(tree.body)
    except ZeroDivisionError as exc:
        raise ToolExecutionError("不能除以 0") from exc
    except OverflowError as exc:
        raise ToolExecutionError("计算结果过大") from exc
    except TypeError as exc:
        raise ToolExecutionError("表达式包含不支持的类型") from exc

    if isinstance(result, complex):
        raise ToolExecutionError("不支持复数结果")
    if isinstance(result, float) and not math.isfinite(result):
        raise ToolExecutionError("计算结果不是有限数值")
    if isinstance(result, int) and result.bit_length() > MAX_INTEGER_BITS:
        raise ToolExecutionError("计算结果过大")
    if isinstance(result, bool):
        raise ToolExecutionError("不支持布尔值")

    return result


def _eval_node(node: ast.AST) -> int | float:
    """递归计算 AST 节点；只放行白名单中的节点类型。"""

    # 数字常量。注意 bool 是 int 的子类，要单独排除。
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolExecutionError("只支持数字常量")
        return node.value

    # 正负号，例如 -5、+3。
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ToolExecutionError("不支持的一元运算符")

    # 二元运算，例如 1 + 2、2 ** 10。
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)

        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.Pow):
            if abs(right) > MAX_EXPONENT:
                raise ToolExecutionError(
                    f"指数绝对值不能超过 {MAX_EXPONENT}"
                )
            return left**right

        raise ToolExecutionError("不支持的二元运算符")

    # 其他节点（变量名、函数调用、属性访问等）全部拒绝。
    raise ToolExecutionError(
        f"不支持的表达式类型：{type(node).__name__}"
    )
