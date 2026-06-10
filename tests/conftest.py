"""Lightweight fakes for the ModelRequest/ModelResponse contract.

The middleware only touches ``request.messages``, ``request.tools``,
``request.override(messages=...)`` and ``response.result`` — so we fake those
rather than spinning up a full agent graph. Real ``AIMessage`` / ``ToolMessage``
are used because the middleware does ``isinstance`` checks on them.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage


class FakeRequest:
    def __init__(self, messages: list[AnyMessage], tools: list[Any]) -> None:
        self.messages = messages
        self.tools = tools

    def override(self, *, messages: list[AnyMessage]) -> FakeRequest:
        return FakeRequest(messages, self.tools)


class FakeResponse:
    def __init__(self, result: list[AnyMessage]) -> None:
        self.result = result


def make_tool(name: str, args_schema: Any) -> SimpleNamespace:
    """A minimal stand-in for BaseTool (resolve only reads name + args_schema)."""
    return SimpleNamespace(name=name, args_schema=args_schema)


def ai_with_calls(*calls: dict[str, Any]) -> AIMessage:
    return AIMessage(content="", tool_calls=list(calls))


def call(name: str, call_id: str, **args: Any) -> dict[str, Any]:
    return {"name": name, "id": call_id, "args": dict(args), "type": "tool_call"}
