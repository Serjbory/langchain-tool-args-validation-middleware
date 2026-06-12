"""Tests for schema resolution and the two validation backends."""

from pydantic import BaseModel

from langchain_tool_args_validation_middleware._validation import (
    format_issues,
    resolve_validators,
)
from tests.conftest import make_tool


class _Args(BaseModel):
    a: int
    note: str | None = None


JSON_SCHEMA = {
    "type": "object",
    "properties": {"b": {"type": "integer"}},
    "required": ["b"],
    "additionalProperties": False,
}


def test_resolves_pydantic_and_jsonschema_tools():
    validators = resolve_validators(
        [make_tool("p", _Args), make_tool("j", JSON_SCHEMA)],
        json_schema_validator_class=None,
    )
    assert set(validators) == {"p", "j"}


def test_unknown_schema_shape_is_skipped():
    validators = resolve_validators(
        [make_tool("weird", object())], json_schema_validator_class=None
    )
    assert validators == {}


def test_pydantic_valid_and_invalid():
    v = resolve_validators([make_tool("p", _Args)], json_schema_validator_class=None)[
        "p"
    ]
    assert v.validate({"a": 1}) == []
    issues = v.validate({})  # missing required 'a'
    assert any("a" in str(i.path) for i in issues)


def test_jsonschema_valid_and_invalid():
    v = resolve_validators(
        [make_tool("j", JSON_SCHEMA)], json_schema_validator_class=None
    )["j"]
    assert v.validate({"b": 5}) == []
    assert v.validate({}) != []  # missing required 'b'
    assert v.validate({"b": 5, "extra": 1}) != []  # additionalProperties: False


def test_format_issues_is_readable():
    v = resolve_validators([make_tool("p", _Args)], json_schema_validator_class=None)[
        "p"
    ]
    text = format_issues("p", v.validate({}))
    assert "Tool 'p'" in text
    assert "omit it entirely" in text
