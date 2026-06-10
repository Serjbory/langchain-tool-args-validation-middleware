"""Tests for strip_empty — especially the placeholder-string footgun."""

from langchain_tool_arg_validation import strip_empty
from langchain_tool_arg_validation._strip import DEFAULT_PLACEHOLDER_STRINGS


def test_removes_none_and_empty_containers():
    assert strip_empty({"a": 1, "b": None, "c": {}, "d": []}) == {"a": 1}


def test_recurses_into_nested_dicts_and_lists():
    out = strip_empty({"x": {"y": None, "z": 2}, "items": [{"k": None, "v": 1}]})
    assert out == {"x": {"z": 2}, "items": [{"v": 1}]}


def test_keeps_meaningful_strings_by_default():
    # "NA" is Namibia's ISO code — must never be silently dropped by default.
    assert strip_empty({"country": "NA", "note": "none"}) == {
        "country": "NA",
        "note": "none",
    }


def test_placeholder_stripping_is_opt_in():
    out = strip_empty(
        {"country": "NA", "note": "null"},
        placeholder_strings=DEFAULT_PLACEHOLDER_STRINGS,
    )
    # "null" is in the default set; "NA"/"na" deliberately are NOT.
    assert out == {"country": "NA"}


def test_does_not_mutate_input():
    src = {"a": 1, "b": None}
    strip_empty(src)
    assert src == {"a": 1, "b": None}


def test_keeps_falsy_but_meaningful_scalars():
    # 0 and False are not "empty" and must survive.
    assert strip_empty({"count": 0, "flag": False, "name": ""}) == {
        "count": 0,
        "flag": False,
        "name": "",
    }
