"""calculator 工具的单元测试。"""

from __future__ import annotations

import pytest

from app.tools.base import ToolExecutionError
from app.tools.calculator import evaluate_expression


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("1 + 2", 3),
        ("(12 + 8) * 3 / 2", 30.0),
        ("2 ** 10", 1024),
        ("7 // 2", 3),
        ("7 % 2", 1),
        ("-5 + 3", -2),
        ("2 ** -1", 0.5),
        ("3.5 * 2", 7.0),
    ],
)
def test_supported_arithmetic(expression: str, expected: int | float) -> None:
    """白名单内的算术运算结果正确。"""
    assert evaluate_expression(expression) == expected


def test_division_by_zero() -> None:
    """除零要变成可读的业务错误。"""
    with pytest.raises(ToolExecutionError, match="不能除以 0"):
        evaluate_expression("1 / 0")


def test_empty_expression() -> None:
    """空表达式直接拒绝。"""
    with pytest.raises(ToolExecutionError, match="不能为空"):
        evaluate_expression("   ")


def test_expression_too_long() -> None:
    """超长表达式拒绝执行，防止滥用。"""
    with pytest.raises(ToolExecutionError, match="表达式过长"):
        evaluate_expression("1 + " * 60 + "1")


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('whoami')",
        "open('secret.txt').read()",
        "x + 1",
        "abs(-1)",
        "[1, 2, 3]",
        "'hello'",
        "True",
    ],
)
def test_rejects_unsafe_or_unsupported_expressions(expression: str) -> None:
    """变量、函数调用、属性访问、列表、字符串、布尔值全部拒绝。"""
    with pytest.raises(ToolExecutionError):
        evaluate_expression(expression)


def test_exponent_limit() -> None:
    """指数过大时拒绝，避免 2**1000000 卡死进程。"""
    with pytest.raises(ToolExecutionError, match="指数绝对值"):
        evaluate_expression("2 ** 101")


def test_overflow_to_infinity() -> None:
    """浮点溢出成 inf 时视为无效结果。"""
    with pytest.raises(ToolExecutionError, match="有限数值"):
        evaluate_expression("1e308 * 10")


def test_complex_result_is_rejected() -> None:
    """负数开方会产生复数，不在支持范围内。"""
    with pytest.raises(ToolExecutionError, match="复数"):
        evaluate_expression("(-1) ** 0.5")
