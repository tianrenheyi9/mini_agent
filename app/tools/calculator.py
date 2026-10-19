"""
calculator 工具：安全地计算算术表达式。

为什么不直接用 eval()？
- eval 会执行任意 Python 代码，例如 __import__("os").system(...)；
- 这里用 ast（抽象语法树）解析表达式，只允许数字和少量运算符；
- 变量、函数调用、属性访问、下标、lambda 等一律拒绝。

整个流程：
表达式字符串 -> ast.parse -> 递归检查/计算 AST -> 数字结果
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
                    # description 会原样进入 Prompt 的工具目录，
                    # 写清楚示例能降低模型传错格式的概率。
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
    """工具入口：取出表达式，调用安全计算函数，返回结构化结果。"""
    expression = input_data["expression"]
    result = evaluate_expression(expression)
    return {
        "expression": expression.strip(),
        "result": result,
    }


def evaluate_expression(expression: str) -> int | float:
    """解析并计算表达式，非法输入统一抛出 ToolExecutionError。"""
    # 空字符串没有任何可计算内容。
    if not isinstance(expression, str) or not expression.strip():
        raise ToolExecutionError("表达式不能为空")

    expression = expression.strip()

    # 长度限制是第一道防线，防止超长表达式拖慢解析。
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ToolExecutionError(
            f"表达式过长（最多 {MAX_EXPRESSION_LENGTH} 个字符）"
        )

    try:
        # mode="eval" 只允许单个表达式，不允许语句（如 import、赋值）。
        # 例如 "1 + 2" 可以，但 "import os" 会直接语法错误。
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError) as exc:
        raise ToolExecutionError("表达式语法错误") from exc

    try:
        result = _eval_node(tree.body)
    except ZeroDivisionError as exc:
        # 1 / 0、1 // 0、1 % 0 都会走到这里。
        raise ToolExecutionError("不能除以 0") from exc
    except OverflowError as exc:
        # 浮点数太大时 Python 可能抛 OverflowError。
        raise ToolExecutionError("计算结果过大") from exc
    except TypeError as exc:
        # 例如对不兼容类型做运算；正常白名单下很少出现，兜底处理。
        raise ToolExecutionError("表达式包含不支持的类型") from exc

    # 以下是对计算结果的二次检查。
    if isinstance(result, complex):
        raise ToolExecutionError("不支持复数结果")
    if isinstance(result, float) and not math.isfinite(result):
        # 例如 1e308 * 10 会得到 inf。
        raise ToolExecutionError("计算结果不是有限数值")
    if isinstance(result, int) and result.bit_length() > MAX_INTEGER_BITS:
        raise ToolExecutionError("计算结果过大")
    if isinstance(result, bool):
        # bool 是 int 子类，理论上不会走到这里，保留防御。
        raise ToolExecutionError("不支持布尔值")

    return result


def _eval_node(node: ast.AST) -> int | float:
    """
    递归计算 AST 节点。

    AST 是一棵树：例如 1 + 2 * 3 会被解析成
    BinOp(1, Add, BinOp(2, Mult, 3))。
    我们只放行白名单中的节点类型，其余全部报错。
    """

    # 数字常量。注意 bool 是 int 的子类，要单独排除。
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolExecutionError("只支持数字常量")
        return node.value

    # 正负号，例如 -5、+3。
    # 先递归计算操作数，再套用符号。
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ToolExecutionError("不支持的一元运算符")

    # 二元运算，例如 1 + 2、2 ** 10。
    # 先递归计算左右子树，再按运算符计算。
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
            # 乘方最容易产生超大数，单独限制指数范围。
            if abs(right) > MAX_EXPONENT:
                raise ToolExecutionError(
                    f"指数绝对值不能超过 {MAX_EXPONENT}"
                )
            return left**right

        raise ToolExecutionError("不支持的二元运算符")

    # 其他节点（变量名 Name、函数调用 Call、属性访问 Attribute、
    # 下标 Subscript、lambda 等）全部拒绝。
    # 这是安全白名单的核心：不认识的节点一律不执行。
    raise ToolExecutionError(
        f"不支持的表达式类型：{type(node).__name__}"
    )
