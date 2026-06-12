"""End-to-end middleware tests — batch handling, strip write-back, retries."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel

from langchain_tool_args_validation_middleware import (
    ToolArgsValidationError,
    ToolArgsValidationMiddleware,
    detect_langchain_internal_ids,
)
from tests.conftest import FakeRequest, FakeResponse, ai_with_calls, call, make_tool


class AArgs(BaseModel):
    a: int
    note: str | None = None
    tags: list[str] = []


class BArgs(BaseModel):
    b: int


TOOLS = [make_tool("toolA", AArgs), make_tool("toolB", BArgs)]


def _request(messages=None):
    return FakeRequest(messages or [HumanMessage(content="hi")], TOOLS)


class ScriptedHandler:
    """Returns a pre-scripted sequence of responses, recording each request."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        return self._responses.pop(0)

    async def acall(self, request):
        return self(request)


# --------------------------------------------------------------------------- #
# Batch handling
# --------------------------------------------------------------------------- #


def test_partial_batch_failure_responds_to_every_call():
    """One invalid + one valid call → both get a ToolMessage on retry."""
    bad = FakeResponse(
        [ai_with_calls(call("toolA", "1"), call("toolB", "2", b=5))]  # toolA missing a
    )
    good = FakeResponse(
        [ai_with_calls(call("toolA", "1", a=1), call("toolB", "2", b=5))]
    )
    handler = ScriptedHandler(bad, good)
    mw = ToolArgsValidationMiddleware(tools=TOOLS, max_retries=2)

    out = mw.wrap_model_call(_request(), handler)

    assert out is good
    assert len(handler.requests) == 2  # initial + 1 retry
    retry_msgs = handler.requests[1].messages
    tool_msgs = [m for m in retry_msgs if isinstance(m, ToolMessage)]
    # Every tool_call in the failed batch must have exactly one response.
    assert {m.tool_call_id for m in tool_msgs} == {"1", "2"}
    sibling = next(m for m in tool_msgs if m.tool_call_id == "2")
    assert "not executed" in sibling.content

    # Provider requirement: the failed AIMessage must precede its ToolMessages.
    ai_idx = next(
        i for i, m in enumerate(retry_msgs) if isinstance(m, AIMessage) and m.tool_calls
    )
    first_tool_idx = next(
        i for i, m in enumerate(retry_msgs) if isinstance(m, ToolMessage)
    )
    assert ai_idx < first_tool_idx


def test_all_valid_returns_immediately_without_retry():
    good = FakeResponse([ai_with_calls(call("toolA", "1", a=1))])
    handler = ScriptedHandler(good)
    mw = ToolArgsValidationMiddleware(tools=TOOLS)

    out = mw.wrap_model_call(_request(), handler)

    assert out is good
    assert len(handler.requests) == 1


def test_no_tool_calls_passes_through():
    resp = FakeResponse([ai_with_calls()])  # plain answer, no calls
    handler = ScriptedHandler(resp)
    mw = ToolArgsValidationMiddleware(tools=TOOLS)
    assert mw.wrap_model_call(_request(), handler) is resp


def test_unknown_tool_passes_through():
    resp = FakeResponse([ai_with_calls(call("mystery", "9", whatever=1))])
    handler = ScriptedHandler(resp)
    mw = ToolArgsValidationMiddleware(tools=TOOLS)
    assert mw.wrap_model_call(_request(), handler) is resp


# --------------------------------------------------------------------------- #
# strip_empty_values write-back
# --------------------------------------------------------------------------- #


def test_strip_writes_back_cleaned_args():
    resp = FakeResponse([ai_with_calls(call("toolA", "1", a=1, note=None, tags=[]))])
    handler = ScriptedHandler(resp)
    mw = ToolArgsValidationMiddleware(tools=TOOLS, strip_empty_values=True)

    out = mw.wrap_model_call(_request(), handler)

    assert len(handler.requests) == 1  # cleaned args validate fine → no retry
    assert out.result[0].tool_calls[0]["args"] == {"a": 1}  # cleaned in place


def test_strip_disabled_keeps_original_args():
    resp = FakeResponse([ai_with_calls(call("toolA", "1", a=1, note=None, tags=[]))])
    handler = ScriptedHandler(resp)
    mw = ToolArgsValidationMiddleware(tools=TOOLS, strip_empty_values=False)

    mw.wrap_model_call(_request(), handler)

    assert resp.result[0].tool_calls[0]["args"] == {"a": 1, "note": None, "tags": []}


# --------------------------------------------------------------------------- #
# Retry exhaustion / on_failure
# --------------------------------------------------------------------------- #


def test_fail_open_passes_through_after_exhaustion():
    bad = FakeResponse([ai_with_calls(call("toolA", "1"))])  # always invalid
    handler = ScriptedHandler(bad, bad, bad)
    mw = ToolArgsValidationMiddleware(tools=TOOLS, max_retries=2, on_failure="pass")

    out = mw.wrap_model_call(_request(), handler)

    assert out is bad
    assert len(handler.requests) == 3  # initial + 2 retries


def test_on_failure_raise():
    bad = FakeResponse([ai_with_calls(call("toolA", "1"))])
    handler = ScriptedHandler(bad, bad)
    mw = ToolArgsValidationMiddleware(tools=TOOLS, max_retries=1, on_failure="raise")

    with pytest.raises(ToolArgsValidationError):
        mw.wrap_model_call(_request(), handler)


def test_last_retry_response_is_validated():
    """A retry that *succeeds on the final attempt* must be accepted, not treated
    as exhausted. Regression test for the loop skipping the last response."""
    bad = FakeResponse([ai_with_calls(call("toolA", "1"))])
    good = FakeResponse([ai_with_calls(call("toolA", "1", a=1))])
    handler = ScriptedHandler(bad, good)  # second (final) attempt is valid
    mw = ToolArgsValidationMiddleware(tools=TOOLS, max_retries=1, on_failure="raise")

    out = mw.wrap_model_call(_request(), handler)

    assert out is good  # not raised, even though it took the last allowed retry
    assert len(handler.requests) == 2


def test_max_retries_zero_validates_once_without_retrying():
    """max_retries=0 still validates the single response; valid output passes."""
    good = FakeResponse([ai_with_calls(call("toolA", "1", a=1))])
    handler = ScriptedHandler(good)
    mw = ToolArgsValidationMiddleware(tools=TOOLS, max_retries=0, on_failure="raise")

    out = mw.wrap_model_call(_request(), handler)

    assert out is good
    assert len(handler.requests) == 1  # no retry attempted


def test_max_retries_zero_does_not_retry_invalid():
    bad = FakeResponse([ai_with_calls(call("toolA", "1"))])
    handler = ScriptedHandler(bad)
    mw = ToolArgsValidationMiddleware(tools=TOOLS, max_retries=0, on_failure="pass")

    out = mw.wrap_model_call(_request(), handler)

    assert out is bad
    assert len(handler.requests) == 1


def test_errors_accumulate_across_retries():
    """Each failed attempt's AIMessage + error ToolMessages stay in the convo."""
    bad1 = FakeResponse([ai_with_calls(call("toolA", "1"))])
    bad2 = FakeResponse([ai_with_calls(call("toolA", "1", a="oops"))])
    good = FakeResponse([ai_with_calls(call("toolA", "1", a=1))])
    handler = ScriptedHandler(bad1, bad2, good)
    mw = ToolArgsValidationMiddleware(tools=TOOLS, max_retries=2)

    out = mw.wrap_model_call(_request(), handler)

    assert out is good
    # Third request should carry both prior failed turns (2 AI + 2 ToolMessages).
    final_msgs = handler.requests[2].messages
    assert sum(isinstance(m, AIMessage) for m in final_msgs) == 2
    assert sum(isinstance(m, ToolMessage) for m in final_msgs) == 2


# --------------------------------------------------------------------------- #
# Lazy schema resolution + JSON Schema (MCP) path
# --------------------------------------------------------------------------- #


def test_lazy_resolution_from_request_tools():
    """With tools=None, schemas are resolved from request.tools and cached."""
    bad = FakeResponse([ai_with_calls(call("toolA", "1"))])
    good = FakeResponse([ai_with_calls(call("toolA", "1", a=1))])
    handler = ScriptedHandler(bad, good)
    mw = ToolArgsValidationMiddleware()  # no explicit tools

    out = mw.wrap_model_call(_request(), handler)

    assert out is good
    assert len(handler.requests) == 2
    # Cache populated under the request's tool-name set.
    assert frozenset({"toolA", "toolB"}) in mw._cache


JSON_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}


def test_jsonschema_mcp_tool_end_to_end():
    mcp_tools = [make_tool("weather", JSON_SCHEMA)]
    request = FakeRequest([HumanMessage(content="hi")], mcp_tools)
    bad = FakeResponse([ai_with_calls(call("weather", "1"))])  # missing 'city'
    good = FakeResponse([ai_with_calls(call("weather", "1", city="Berlin"))])
    handler = ScriptedHandler(bad, good)
    mw = ToolArgsValidationMiddleware(tools=mcp_tools)

    out = mw.wrap_model_call(request, handler)

    assert out is good
    err = handler.requests[1].messages[-1]
    assert "weather" in err.content


# --------------------------------------------------------------------------- #
# Extra validators + async parity
# --------------------------------------------------------------------------- #


def test_extra_validator_flags_langchain_ids():
    lc_id = "lc_12345678-1234-1234-1234-123456789abc"
    bad = FakeResponse([ai_with_calls(call("toolA", "1", a=1, note=lc_id))])
    good = FakeResponse([ai_with_calls(call("toolA", "1", a=1))])
    handler = ScriptedHandler(bad, good)
    mw = ToolArgsValidationMiddleware(
        tools=TOOLS, extra_validators=[detect_langchain_internal_ids]
    )

    out = mw.wrap_model_call(_request(), handler)

    assert out is good
    err = handler.requests[1].messages[-1]
    assert "internal ID" in err.content


async def test_async_parity():
    bad = FakeResponse([ai_with_calls(call("toolB", "2"))])
    good = FakeResponse([ai_with_calls(call("toolB", "2", b=7))])
    handler = ScriptedHandler(bad, good)
    mw = ToolArgsValidationMiddleware(tools=TOOLS)

    out = await mw.awrap_model_call(_request(), handler.acall)

    assert out is good
    assert len(handler.requests) == 2
