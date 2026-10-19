"""参数 Schema 校验的单元测试。"""

from __future__ import annotations

from app.tools.base import validate_parameters

SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "days": {"type": "integer", "minimum": 1, "maximum": 7},
        "tags": {"type": "array", "items": {"type": "string"}},
        "mode": {"type": "string", "enum": ["fast", "slow"]},
    },
    "required": ["city"],
}


def test_valid_input_has_no_errors() -> None:
    """完全合法的输入不应产生错误。"""
    errors = validate_parameters(
        SCHEMA,
        {"city": "北京", "days": 3, "tags": ["天气"], "mode": "fast"},
    )
    assert errors == []


def test_missing_required_field() -> None:
    """缺少必填字段时要指出具体路径。"""
    errors = validate_parameters(SCHEMA, {"days": 3})
    assert errors == ["input.city: 必填字段缺失"]


def test_wrong_type() -> None:
    """类型不匹配时要说明期望值和实际值。"""
    errors = validate_parameters(SCHEMA, {"city": 123})
    assert errors == ["input.city: 期望 string，实际是 integer"]


def test_bool_is_not_integer() -> None:
    """Python 的 bool 是 int 子类，但 Schema 里必须区分。"""
    errors = validate_parameters(SCHEMA, {"city": "北京", "days": True})
    assert errors == ["input.days: 期望 integer，实际是 boolean"]


def test_number_range() -> None:
    """minimum / maximum 要生效。"""
    errors = validate_parameters(SCHEMA, {"city": "北京", "days": 0})
    assert errors == ["input.days: 不能小于 1"]

    errors = validate_parameters(SCHEMA, {"city": "北京", "days": 9})
    assert errors == ["input.days: 不能大于 7"]


def test_enum() -> None:
    """不在 enum 中的值要被拒绝。"""
    errors = validate_parameters(SCHEMA, {"city": "北京", "mode": "turbo"})
    assert errors == ["input.mode: 必须是以下值之一：'fast'、'slow'"]


def test_nested_array_items() -> None:
    """数组元素类型也要逐个校验。"""
    errors = validate_parameters(SCHEMA, {"city": "北京", "tags": ["a", 1]})
    assert errors == ["input.tags[1]: 期望 string，实际是 integer"]


def test_multiple_errors_are_collected() -> None:
    """一次返回所有错误，方便模型一次改完。"""
    errors = validate_parameters(
        SCHEMA,
        {"days": "three", "tags": [1], "mode": "turbo"},
    )
    assert "input.city: 必填字段缺失" in errors
    assert "input.days: 期望 integer，实际是 string" in errors
    assert "input.tags[0]: 期望 string，实际是 integer" in errors
    assert "input.mode: 必须是以下值之一：'fast'、'slow'" in errors
