"""输出协议解析器的单元测试。"""

from __future__ import annotations

from app.agent.parser import FinalAnswer, ParseError, ToolCall, parse


def test_parses_plain_tool_call() -> None:
    """最标准的工具调用：纯 JSON，无围栏、无前后缀。"""
    result = parse(
        '{"thought": "先查天气", "action": "weather", "input": {"city": "北京"}}'
    )

    assert isinstance(result, ToolCall)
    assert result.thought == "先查天气"
    assert result.action == "weather"
    assert result.input == {"city": "北京"}


def test_parses_plain_final_answer() -> None:
    """最标准的最终回答。"""
    result = parse('{"thought": "信息够了", "final": "北京今天多云。"}')

    assert isinstance(result, FinalAnswer)
    assert result.thought == "信息够了"
    assert result.final == "北京今天多云。"


def test_parses_fenced_json() -> None:
    """模型把 JSON 包在 ```json 围栏里时也能解析。"""
    raw = """```json
{"action": "todo", "input": {"action": "list"}}
```"""
    result = parse(raw)

    assert isinstance(result, ToolCall)
    assert result.action == "todo"
    assert result.input == {"action": "list"}


def test_parses_json_with_surrounding_prose() -> None:
    """JSON 前后有解释文字时，扫描平衡括号仍能找到对象。"""
    raw = '我想先查一下天气。\n{"action": "weather", "input": {"city": "上海"}}\n请稍等。'
    result = parse(raw)

    assert isinstance(result, ToolCall)
    assert result.input == {"city": "上海"}


def test_handles_nested_objects_and_braces_in_strings() -> None:
    """嵌套对象、字符串内部的花括号都不能打断括号匹配。"""
    raw = (
        '{"action": "search", "input": {'
        '"query": "JSON 里怎么用 { 和 }？", '
        '"options": {"limit": 3}'
        "}}"
    )
    result = parse(raw)

    assert isinstance(result, ToolCall)
    assert result.input["query"] == "JSON 里怎么用 { 和 }？"
    assert result.input["options"] == {"limit": 3}


def test_handles_escaped_quotes() -> None:
    """字符串里的转义引号不能提前结束字符串。"""
    raw = '{"final": "他说：\\"你好\\""}'
    result = parse(raw)

    assert isinstance(result, FinalAnswer)
    assert result.final == '他说："你好"'


def test_prefers_fenced_block_over_prose_example() -> None:
    """正文里举例的 JSON 不能覆盖真正放在围栏里的决策。"""
    raw = """格式示例是 {"action": "wrong", "input": {}}。
真正要执行的是：
```json
{"action": "weather", "input": {"city": "上海"}}
```"""
    result = parse(raw)

    assert isinstance(result, ToolCall)
    assert result.action == "weather"
    assert result.input == {"city": "上海"}


def test_skips_invalid_json_candidate() -> None:
    """前面的 JSON 片段解析失败时，应继续寻找后面的合法对象。"""
    raw = '{not json} 然后才是 {"action": "todo", "input": {"action": "list"}}'
    result = parse(raw)

    assert isinstance(result, ToolCall)
    assert result.action == "todo"


def test_skips_protocol_invalid_candidate() -> None:
    """前面的对象是合法 JSON 但不符合协议时，继续寻找后面的对象。"""
    raw = '{"foo": 1} 然后 {"final": "好的"}'
    result = parse(raw)

    assert isinstance(result, FinalAnswer)
    assert result.final == "好的"


def test_reports_both_action_and_final() -> None:
    """action 和 final 同时出现属于协议错误。"""
    result = parse(
        '{"action": "weather", "input": {}, "final": "同时给了两个"}'
    )

    assert isinstance(result, ParseError)
    assert "不能同时出现" in result.reason


def test_reports_missing_action_and_final() -> None:
    """两个字段都没有时属于协议错误。"""
    result = parse('{"thought": "只有思考"}')

    assert isinstance(result, ParseError)
    assert "缺少 action 或 final" in result.reason


def test_reports_invalid_action_type() -> None:
    """action 必须是字符串。"""
    result = parse('{"action": 123, "input": {}}')

    assert isinstance(result, ParseError)
    assert "action 必须是非空字符串" in result.reason


def test_reports_invalid_input_type() -> None:
    """input 必须是对象。"""
    result = parse('{"action": "weather", "input": "北京"}')

    assert isinstance(result, ParseError)
    assert "input 必须是 JSON 对象" in result.reason


def test_reports_empty_final() -> None:
    """final 不能是空字符串。"""
    result = parse('{"final": "   "}')

    assert isinstance(result, ParseError)
    assert "final 必须是非空字符串" in result.reason


def test_reports_missing_json() -> None:
    """完全没有 JSON 时给出明确错误。"""
    result = parse("我不知道该怎么回答。")

    assert isinstance(result, ParseError)
    assert "没有找到合法的 JSON 对象" in result.reason


def test_reports_empty_output() -> None:
    """空输出单独报错，便于上层提示模型重试。"""
    result = parse("   ")

    assert isinstance(result, ParseError)
    assert "输出为空" in result.reason


def test_reports_non_object_json() -> None:
    """JSON 数组或普通值不符合协议。"""
    result = parse("[1, 2, 3]")

    assert isinstance(result, ParseError)
    assert "必须是 JSON 对象" in result.reason


def test_defaults_thought_and_input() -> None:
    """thought 和 input 可以省略，分别降级为空字符串和空对象。"""
    result = parse('{"action": "todo"}')

    assert isinstance(result, ToolCall)
    assert result.thought == ""
    assert result.input == {}


def test_preserves_chinese_content() -> None:
    """中文和 emoji 应原样保留。"""
    result = parse('{"final": "已完成 ✅ 本周周报"}')

    assert isinstance(result, FinalAnswer)
    assert result.final == "已完成 ✅ 本周周报"
