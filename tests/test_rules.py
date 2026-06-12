"""Tests for declarative :class:`FieldRule` checks."""

from langchain_tool_args_validation_middleware import FieldRule


def _numbers_rule(**kwargs) -> FieldRule:
    return FieldRule(
        path="numbers.*",
        check=lambda v: isinstance(v, int) and 0 < v < 100,
        error=lambda v: f"value {v!r} is out of range",
        **kwargs,
    )


def test_per_element_flags_only_offenders():
    rule = _numbers_rule()
    errors = rule("my_tool", {"numbers": [50, -1, 100]})
    assert len(errors) == 2
    assert "numbers[1]" in errors[0] and "-1" in errors[0]
    assert "numbers[2]" in errors[1] and "100" in errors[1]


def test_all_valid_returns_no_errors():
    rule = _numbers_rule()
    assert rule("my_tool", {"numbers": [1, 50, 99]}) == []


def test_tool_targeting_is_a_noop_for_other_tools():
    rule = _numbers_rule(tools=["my_tool"])
    assert rule("other_tool", {"numbers": [-1]}) == []
    assert rule("my_tool", {"numbers": [-1]}) != []


def test_tools_none_applies_everywhere():
    rule = _numbers_rule()
    assert rule("any_tool", {"numbers": [-1]}) != []


def test_list_level_rule_receives_whole_list():
    rule = FieldRule(
        path="numbers",
        check=lambda v: isinstance(v, list) and 1 <= len(v) <= 10,
        error="numbers must contain between 1 and 10 items",
    )
    assert rule("my_tool", {"numbers": []}) != []
    assert rule("my_tool", {"numbers": [1, 2]}) == []


def test_nested_path():
    rule = FieldRule(
        path="config.thresholds.*",
        check=lambda v: v > 0,
        error="threshold must be positive",
    )
    errors = rule("t", {"config": {"thresholds": [10, -1, 5]}})
    assert len(errors) == 1
    assert "config.thresholds[1]" in errors[0]


def test_static_error_string():
    rule = FieldRule(path="x", check=lambda v: False, error="bad x")
    (msg,) = rule("t", {"x": 1})
    assert msg == "Tool 't' argument 'x': bad x"


def test_missing_path_skipped_by_default():
    rule = _numbers_rule()
    assert rule("my_tool", {}) == []


def test_missing_path_errors_when_requested():
    rule = FieldRule(
        path="numbers",
        check=lambda v: True,
        error="unused",
        when_missing="error",
    )
    (msg,) = rule("my_tool", {})
    assert "missing required argument 'numbers'" in msg


def test_star_on_dict_values():
    rule = FieldRule(path="scores.*", check=lambda v: v >= 0, error="negative")
    errors = rule("t", {"scores": {"a": 1, "b": -5}})
    assert len(errors) == 1
    assert "scores.b" in errors[0]


def test_star_on_scalar_yields_nothing():
    rule = FieldRule(path="numbers.*", check=lambda v: False, error="x")
    assert rule("t", {"numbers": 5}) == []


def test_drops_into_extra_validators():
    # A FieldRule must satisfy the ExtraValidator contract: (name, args) -> list[str].
    from langchain_tool_args_validation_middleware import ToolArgsValidationMiddleware

    rule = _numbers_rule(tools=["my_tool"])
    mw = ToolArgsValidationMiddleware(extra_validators=[rule])
    assert mw._run_extra_validators("my_tool", {"numbers": [-1]}) != []
    assert mw._run_extra_validators("my_tool", {"numbers": [50]}) == []
